import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import transformers
from typing import Optional, Union, List, Dict, Any
import signal
from subprocess import call
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTAttention
from models.int_llama_layer import QuantLlamaDecoderLayer
from torch.nn import CrossEntropyLoss

from quantize.int_linear import QuantLinear
from quantize.utils import smooth_and_quant_temporary, use_parameters, clear_temp_variable, set_quant_state



logger = logging.getLogger(__name__)

class out():
    def __init__(self, loss,
        logits,
        past_key_values,
        hidden_states,
        attentions):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions



def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """

    with torch.no_grad():
        outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return out(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

    # return CausalLMOutputWithPast(
    #     loss=loss,
    #     logits=logits,
    #     past_key_values=outputs.past_key_values,
    #     hidden_states=outputs.hidden_states,
    #     attentions=outputs.attentions,
    # )


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # With demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
   
    return encodings, option_lens
 


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]
        
        return batch
        

class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)


def model_quantization(model, model_name, w_train_bits, a_train_bits):

    model_nick_name = model_name.split("/")[-1]
    is_llama = 'llama' in model_name.lower()

    act_scales = torch.load(f'/home/Qitao/project/ZO_quant_new/act_scales/{model_nick_name}.pt')
    act_shifts = torch.load(f'/home/Qitao/project/ZO_quant_new/act_shifts/{model_nick_name}.pt')

    quant_args = {"weight_quant_params": {'n_bits': w_train_bits, 'per_channel_axes': [0], 'symmetric': False,
                                          'dynamic_method': 'per_channel', 'group_size': False, 'lwc': False,
                                          'disable_zero_point': False},
                  "act_quant_params": {'n_bits': a_train_bits, 'per_channel_axes': [], 'symmetric': False,
                                       'dynamic_method': 'per_token'},
                  "p_quant_params": {'n_bits': 16, 'metric': 'fix0to1'}}


    if not is_llama:
        layer_name_prefix = "model.decoder.layers"
        layers = model.model.decoder.layers
        Qlayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
    else:
        layer_name_prefix = "model.layers"
        layers = model.model.layers
        Qlayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1"
        }
    # qk = torch.load('/home/Qitao/project/ZO_quant_new/qk_llama1-7b.pth')

    alpha = 0.85
    qlinears = []
    for i in range(len(layers)):
        layer = layers[i]
        qlayer = Qlayer(config=model.config, ori_layer=layer, quant_args=quant_args, idx=i, zo_eps=1e-3)
        qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(
            torch.ones(layer.self_attn.q_proj.out_features, device=layer.self_attn.q_proj.weight.device,
                       dtype=torch.float16)))
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                qlinears.append(module)
                for key in pairs.keys():
                    if key in name:
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                        act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                         dtype=torch.float16).clamp(
                            min=1e-5)
                        # scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                        if is_llama:
                            min1 = 0.5 if key == 'q_proj' else 0.65
                            max1 = 0.7 if key == 'q_proj' else 0.8
                            # llama1-13b
                            # min1 = 0.5 if key == 'q_proj' else 0.7
                            # max1 = 0.7 if key == 'q_proj' else 0.8
                            # 0.5是284的ppl
                            refer = act / weight
                            # 对于llama2, min=0.6, max=0.7时,没有omniquant的时候是31
                            alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                            scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                        else:
                            if key == 'q_proj':
                                # max1 = 0.75
                                # min1 = 0.75 if i != 0 else 0.7
                                max1 = 0.7
                                min1 = 0.6
                                # llama是0.5到0.7
                                refer = act / weight
                                alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                            else:
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                        if not is_llama:
                            shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                               dtype=torch.float16)
                        else:
                            shift = torch.zeros_like(scale, device=weight.device, dtype=torch.float16)

                        if key == 'o_proj' or key == 'out_proj':
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(
                                torch.zeros(scale.shape, device=layer.self_attn.q_proj.weight.device,
                                            dtype=torch.float16)))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(
                                torch.ones(scale.shape, device=layer.self_attn.q_proj.weight.device,
                                           dtype=torch.float16)))
                        else:
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

        qlayer.init_smoothing()
        layers[i] = qlayer

    # resume = f'/home/Qitao/project/ZO_quant_new/omni_parameters/Llama-2-7b-w4a4.pth'
    # if resume:
    #     omni_parameters = torch.load(resume)
    #     for i in range(len(layers)):
    #         layers[i].load_state_dict(omni_parameters[i], strict=False)


    return model, qlinears
