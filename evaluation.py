from tasks import get_task
from run import Framework
# from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from metrics import calculate_metric
from tqdm import tqdm
from trainer import Quantizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, quantize_opt
import random
import os
from trainer import smooth_and_quant
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task = get_task('SST2')

framework = Framework(None, task, eval=True)

# model_path = '/home/qitao/MeZO/llm-awq/model/opt-1.3b-awq'
# framework.model = AutoAWQForCausalLM.from_quantized(model_path, torch_dtype=torch.float16)
# framework.model.device = torch.device('cuda')
# act_scales = torch.load("/home/qitao/ZO_quant_new/act_scales/opt-1.3b.pt")
# act_scales = torch.load("/home/qt97442/workdir/ZO_quant_new/act_scales/opt-1.3b.pt")

# act_scales = torch.load("/home/qitao/MeZO/ZO_quant_new/act_scales/opt-2.7b.pt")
# act_shifts = torch.load('/home/qitao/MeZO/ZO_quant_new/act_shifts/opt-2.7b.pt')

act_scales = torch.load("/home/qitao/ZO_quant_new/act_scales/opt-2.7b.pt")
act_shifts = torch.load("/home/qitao/ZO_quant_new/act_shifts/opt-2.7b.pt")


model_fp16 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-2.7b", torch_dtype=torch.float16, device_map="auto"
)
# model_fp16 = AutoModelForCausalLM.from_pretrained(
#     "/home/qitao/OmniQuant-main/save_dir", torch_dtype=torch.float16, device_map="auto"
# )

state = torch.load('/home/qitao/ZO_quant_new/best_model_SST2_W16A4_2P7B.pth')
# state = torch.load('/home/qitao/ZO_quant_new/best_model_RTE_W16A8_2P7B.pth')
# state = torch.load('/home/qitao/MeZO/ZO_quant_new/best_model_SST2_W16A16_FO_2P7B.pth')
model_fp16.load_state_dict(state, strict=False)


# act_shifts = torch.load('/home/qitao/MeZO/ZO_quant_new/act_scales/opt-2.7b.pt')
quant_args = {"weight_quant_params" : {'n_bits': 16, 'per_channel_axes': [0], 'symmetric': False, 'dynamic_method': 'per_channel', 'group_size': 128, 'lwc': False, 'disable_zero_point': False},
              "act_quant_params" : {'n_bits': 16, 'per_channel_axes': [], 'symmetric': False, 'dynamic_method': 'per_token'},
              "p_quant_params" : {'n_bits': 16, 'metric': 'fix0to1'}}
pairs = {
    "q_proj": "qkv",
    "out_proj": "out",
    "fc1": "fc1"
}
layer_name_prefix = "model.decoder.layers"
layers = model_fp16.model.decoder.layers


alpha = 0.75
for i in range(len(layers)):
    layer = layers[i]
    qlayer = QuantOPTDecoderLayer(config=None, ori_layer=layer, quant_args=quant_args)
    for name, module in qlayer.named_modules():
        if isinstance(module, QuantLinear):
            module.name = name
            for key in pairs.keys():
                if key in name:
                    act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device='cuda', dtype=torch.float16).clamp(
                        min=1e-5)
                    weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                    scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                    shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device='cuda', dtype=torch.float16)
                    qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                    qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

    layers[i] = qlayer
smooth_and_quant(model_fp16)
# import code
# code.interact(local=locals())
# smooth_lm(model_fp16, act_scales, alpha=0.75)
# model_fp16 = quantize_opt(model_fp16, w_bits=16, a_bits=4).to(torch.device('cuda'))
framework.model = model_fp16

# model_path = 'facebook/opt-2.7b'
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
#
# state = torch.load('/home/qitao/ZO_quant/best_model.pth')
# model.load_state_dict(state)
# model = model.to(torch.device('cuda'))
# exclude_list = ['model.decoder.embed_tokens.weight', 'model.decoder.embed_positions.weight'] + [name for name, param in model.named_parameters() if 'weight' not in name]
# quantizer = Quantizer(model, 4, 128, exclude_list)
# for name, param in model.named_parameters():
#     param.data = quantizer.pseudo_int_quantize_weight(param, name)
# framework.model = model

framework.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-2.7b', use_fast=False)

predictions = []

eval_dataset = task.valid_samples
length = len(eval_dataset)

with tqdm(total=len(eval_dataset), desc="Processing Samples", unit="sample") as pbar:
    for eval_sample in eval_dataset:
        predictions.append(
            framework.one_step_pred([], eval_sample, verbose=False)
        )
        pbar.update(1)


metric_name = getattr(task, "metric_name", "accuracy")
metrics = {metric_name: calculate_metric(predictions, metric_name)}

print(metrics)
