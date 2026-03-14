# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import code
import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split,
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
# from transformers.integrations.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
# from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    # ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    # default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers

from collections import defaultdict

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback
from tasks import get_task
from smoothquant.smooth import smooth_lm

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from utils import Prediction, encode_prompt
from metrics import calculate_metric
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from scipy.special import loggamma
from models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTAttention
from models.int_llama_layer import QuantLlamaDecoderLayer
from torch.nn import CrossEntropyLoss

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import dispatch_model
from math import inf
from datautils import get_loaders
import time
from quantize.utils import smooth_and_quant_inplace

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

import subprocess
from fake_quant import quantize_model
from quantize.int_linear import QuantLinear
from quantize.utils import smooth_and_quant_temporary, use_parameters, clear_temp_variable, smooth_ln
import re
from eval_wiki import evaluate
from quantize.utils import use_temp_parameters,set_quant_state

from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel.data_parallel import DataParallel

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

import logging
logging.getLogger("lm_eval").setLevel(logging.ERROR)


def get_nvidia_smi_output():
    try:
        # 运行 nvidia-smi 命令并获取输出
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 检查是否有错误输出
        if result.returncode == 0:
            return result.stdout  # 返回命令的标准输出
        else:
            return f"Error: {result.stderr}"  # 返回错误信息
    except FileNotFoundError:
        return "nvidia-smi command not found. Ensure NVIDIA drivers are installed."


def scale_to_range(lst, min_val=0.7, max_val=0.9):
    x_min, x_max = min(lst), max(lst)
    return [min_val + (x - x_min) * (max_val - min_val) / (x_max - x_min) for x in lst]


def smooth_and_quant(model):
    layers = model.model.decoder.layers if type(model) != DataParallel else model.module.model.decoder.layers
    for layer in layers:
        smooth_and_quant_temporary(layer, isllama=False)


class New_Linear(nn.Linear):
    def __init__(self, module):
        super().__init__(module.in_features, module.out_features)
        self.weight = module.weight
        self.bias = module.bias

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def replace_linear_layers(model, quant_args):
    for name, module in model.named_children():  # 遍历子模块
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantLinear(module, weight_quant_params=quant_args['weight_quant_params'],
                                             act_quant_params=quant_args['act_quant_params'], ))
        else:
            replace_linear_layers(module, quant_args)


class Quantizer:
    def __init__(self, model, bits, groups, exclude_list):
        self.scales = {}
        self.zero_points = {}
        self.names = [each[0] for each in list(model.named_parameters())]
        self.bits = bits
        self.qmin = - 2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1
        self.exclude_list = exclude_list
        self.g = groups
        self.symmetric = False

    def perturbation_quant(self, z, zo_eps, name):

        if name not in self.exclude_list and z.dim() > 1:
            ori_shape = z.shape
            scales = self.scales[name]
            z = z.reshape(scales.shape[0], -1)
            z_int = self.stochastic_rounding(z / scales)
            z = z_int * scales

            return z.reshape(ori_shape)

        else:
            return z

    @staticmethod
    def stochastic_rounding(tensor: torch.Tensor) -> torch.Tensor:
        floor_tensor = torch.floor(tensor)
        prob = tensor - floor_tensor
        rand_vals = torch.rand_like(tensor)
        return torch.where(rand_vals < prob, floor_tensor + 1, floor_tensor)

    def pseudo_int_quantize_weight(self, param, name):

        if name not in self.exclude_list and param.dim() > 1:
            org_w_shape = param.shape
            if self.g > 0:
                assert org_w_shape[-1] % self.g == 0
                w = param.reshape(-1, self.g)

            else:
                w = param
            assert w.dim() == 2
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2 ** self.bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

            assert torch.isnan(scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            self.scales[name] = scales
            self.zero_points[name] = zeros

            # w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales

            w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)

            w = (w - zeros) * scales

            assert torch.isnan(w).sum() == 0

            w = w.reshape(org_w_shape)

        else:
            w = param

        return w

    # def quantize(self, model):
    #     perturbs = {}
    #     for name, param in model.named_parameters():
    #         if name not in self.exclude_list and param.dim() > 1:
    #             param.data, perturb = self.pseudo_quantize_tensor(param.data, self.bits, q_group_size=self.g, inplace=False)
    #             perturbs[name] = perturb
    #
    #     return perturbs


class OurTrainer(Trainer):
    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.args.train_batch_size,
        )

    def draw(self, base_scale, omni_scale, i):

        base_scale = base_scale.cpu().float()
        omni_scale = omni_scale.cpu().float()

        plt.figure(figsize=(14, 5))
        plt.plot(base_scale.numpy(), label="base_scale", alpha=0.8)
        plt.plot(omni_scale.numpy(), label="omni_scale", alpha=0.8)

        plt.title("Value Comparison Across 4096 Dimensions")
        plt.xlabel("Dimension Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('{}.jpg'.format(i))

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                # max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                # num_train_epochs = math.ceil(args.num_train_epochs)
                # num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs

                max_steps = 0
                num_train_epochs = 0
                num_train_samples = 0
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        self.sharded_ddp = None
        self.fsdp = None

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                # and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )

        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.do_grad_scaling = False
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        self.task = get_task(self.args.task_name)
        self.best_objective = 0
        # # What parameters to optimize
        self.w_eval_bits = 16
        self.a_eval_bits = 16
        self.w_train_bits = self.args.w_train_bits
        self.a_train_bits = self.args.a_train_bits

        self.loss_fct = nn.CrossEntropyLoss()

        self.act_scales, _ = self.update_scales(model, self.tokenizer, train_dataloader)

        self.act_scales = torch.load(f'/home/Qitao/project/ZO_quant_new/act_scales/{self.args.model_nick_name}.pt')
        self.act_shifts = torch.load(f'/home/Qitao/project/ZO_quant_new/act_shifts/{self.args.model_nick_name}.pt')

        quant_args = {"weight_quant_params": {'n_bits': self.w_train_bits, 'per_channel_axes': [0], 'symmetric': False,
                                              'dynamic_method': 'per_channel', 'group_size': 128, 'lwc': False,
                                              'disable_zero_point': False},
                      "act_quant_params": {'n_bits': self.a_train_bits, 'per_channel_axes': [], 'symmetric': False,
                                           'dynamic_method': 'per_token'},
                      "p_quant_params": {'n_bits': 16, 'metric': 'fix0to1'}}

        self.isllama = self.args.isllama
        if not self.args.isllama:
            self.layer_name_prefix = "model.decoder.layers"
            layers = model.model.decoder.layers
            Qlayer = QuantOPTDecoderLayer
            self.pairs = {
                "q_proj": "qkv",
                "out_proj": "out",
                "fc1": "fc1"
            }
        else:
            self.layer_name_prefix = "model.layers"
            layers = model.model.layers
            Qlayer = QuantLlamaDecoderLayer
            self.pairs = {
                "q_proj": "qkv",
                "o_proj": "out",
                "up_proj": "fc1"
            }

        # args.alpha = 0.5
        self.scale_range = []
        self.diff_list = []

        self.named_alpha_to_optim = []
        self.weight4alpha = []
        self.act4alpha = []

        self.named_parameters_to_optim = []
        self.named_smoothing_to_optim = []


        # qk = torch.load('/home/Qitao/project/ZO_quant_new/qk_llama1-7b.pth')

        # args.alpha = 0.85
        # # opt6.7B
        # args.alpha = 0.85
        # args.alpha = 0.5

        self.qlinears = []
        for i in range(len(layers)):
            layer = layers[i]
            qlayer = Qlayer(config=model.config, ori_layer=layer, quant_args=quant_args, idx=i, zo_eps=self.args.zo_eps)
            qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(
                torch.ones(layer.self_attn.q_proj.out_features, device=layer.self_attn.q_proj.weight.device, dtype=torch.float16)))
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    self.qlinears.append(module)
                    for key in self.pairs.keys():
                        if key in name:
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            act = self.act_scales[f"{self.layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                        dtype=torch.float16).clamp(
                                min=1e-5)
                            # scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                            if args.isllama:
                                min1 = 0.75
                                max1 = 0.75
                                # llama1-7b
                                # min1 = 0.5 if key == 'q_proj' else 0.65
                                # max1 = 0.7 if key == 'q_proj' else 0.8
                                # llama1-13b
                                # min1 = 0.5 if key == 'q_proj' else 0.7
                                # max1 = 0.7 if key == 'q_proj' else 0.8
                                # 0.5是284的ppl
                                refer = act / weight
                                # 对于llama2, min=0.6, max=0.7时,没有omniquant的时候是31
                                alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                            else:
                                # # OPT-13B
                                # max1 = 1.05
                                # min1 = 0.75

                                # # OPT-2.7B W3A16
                                # max1 = 0.2
                                # min1 = 0.2

                                # # OPT-6.7B OPT-2.7B W4A4
                                max1 = 0.7
                                min1 = 0.55
                                # llama是0.5到0.7
                                refer = act / weight
                                alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                                # else:
                                #     scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                            if not self.args.isllama:
                                shift = self.act_shifts[f"{self.layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                                   dtype=torch.float16)
                            else:
                                shift = torch.zeros_like(scale, device=weight.device, dtype=torch.float16)

                            if key == 'o_proj' or key == 'out_proj':
                                qlayer.register_parameter(f"{self.pairs[key]}_smooth_shift", torch.nn.Parameter(torch.zeros(scale.shape, device=layer.self_attn.q_proj.weight.device, dtype=torch.float16)))
                                qlayer.register_parameter(f"{self.pairs[key]}_smooth_scale", torch.nn.Parameter(torch.ones(scale.shape, device=layer.self_attn.q_proj.weight.device, dtype=torch.float16)))
                            else:
                                qlayer.register_parameter(f"{self.pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                                qlayer.register_parameter(f"{self.pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

            qlayer.init_smoothing()
            layers[i] = qlayer

        # # state = torch.load('/home/Qitao/project/ZO_quant_new/Llama-2-7b-hf-W4A4.pth')
        # # print(model.load_state_dict(state, strict=False))
        #
        # # state = torch.load('/home/Qitao/project/ZO_quant_new/llama-13b-W4A4.pth')
        # # print(model.load_state_dict(state, strict=False))
        #
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        #     if ('q_proj.weight' in name or 'v_proj.weight' in name) and 'quantizer' not in name:
        #         # if 'q_proj' in name:
        #             self.named_parameters_to_optim.append((name, param))
        #         # else:
        #         #     self.named_parameters_to_optim.insert(len(self.named_parameters_to_optim) - 1, (name, param))

        for name, param in model.named_parameters():
            param.requires_grad = False
            # if 'bias' not in name:
            if 'bias' not in name and 'norm' not in name:
                if 'scale' not in name and 'shift' not in name and 'bound' not in name:
                        if 'mlp.up_proj' in name:
                            self.named_parameters_to_optim.insert(len(self.named_parameters_to_optim) - 1, (name, param))
                        else:
                            self.named_parameters_to_optim.append((name, param))
                else:
                    self.named_smoothing_to_optim.append((name, param))

        # self.named_parameters_to_optim = self.named_parameters_to_optim[1:-1] if self.args.isllama else self.named_parameters_to_optim[2:]

        # # args.resume = f'/home/Qitao/project/ZO_quant_new/qat_init_parameters/{self.args.model_nick_name}.pth'
        # # # args.resume = f'/home/Qitao/project/ZO_quant_new/omni_parameters/opt-6.7b-w4a4.pth'
        # # # args.resume = '/home/Qitao/project/ZO_quant_new/omni_parameters/Llama-2-7b-w4a4.pth'
        # # args.resume = '/home/Qitao/project/ZO_quant_new/act_scales/Llama-2-7b-w2a16.pth'
        # # # # args.resume = '/home/Qitao/project/OmniQuant-main-raw/log/opt-2.7b-w2a16/omni_parameters.pth'
        # args.resume = '/home/Qitao/project/ZO_quant_new/act_scales/llama-7b-w2a16.pth'
        # # # # # # # args.resume = f'/home/Qitao/project/ZO_quant_new/act_scales/Llama-2-7b-w2a16.pth'
        # # # # # # # args.resume = f'/home/Qitao/project/ZO_quant_new/act_scales/Llama-2-13b-w2a16.pth'
        # # args.resume = '/home/Qitao/project/ZO_quant_new/act_scales/opt-2.7b-w2a16g128.pth'
        # # args.resume = '/home/Qitao/project/ZO_quant_new/qat_init_parameters/Llama-2-7b-hf.pth'
        # args.resume = '/home/Qitao/project/OmniQuant-main-raw/log/Llama-2-7b-chat-w4a4g128/omni_parameters.pth'
        # # args.resume = '/home/Qitao/project/ZO_quant_new/omni_parameters/Llama-2-7b-w4a4.pth'
        # if args.resume:
        #     omni_parameters = torch.load(args.resume)
        #     for i in range(len(layers)):
        #         layers[i].load_state_dict(omni_parameters[i], strict=False)

        # state = torch.load('/home/Qitao/project/ZO_quant_new/Llama-2-7b-hf-W4A4-good.pth')
        # print(model.load_state_dict(state, strict=False))

        # if self.a_train_bits != 16:
        #     self.smooth_ln_pre_training(model)
        #     pass
        # if self.a_train_bits == 16:
        #     for layer in self.qlinears:
        #         layer.use_act_quant = False

        # state = torch.load('/home/Qitao/project/ZO_quant_new/llama-7b-W4A4.pth')
        # print(model.load_state_dict(state, strict=False))


        # for layer in layers:
        #     smooth_and_quant_inplace(layer, self.isllama, self.a_train_bits)
        # for linear in self.qlinears:
        #     if linear.smooth.__name__ not in ['q_smooth', 'v_smooth']:
        #         # linear.to_fp8()
        #         linear.replaced = True
        # torch.cuda.empty_cache()

        # if self.a_train_bits != 16:
        #     self.smooth_ln_pre_training(model)
        # else:
        #     for layer in self.qlinears:
        #         layer.use_act_quant = False


        # self.perturb_qlinear(True)

        # state = torch.load('/home/Qitao/project/ZO_quant_new/opt-6.7b-W4A4_with_omni_init.pth')
        self.target_layer_idx = None
        self.omniquant(args, train_dataloader, logger)
        # self.smooth_and_quant(model, extra=True)
        # set_quant_state(model, weight_quant=True, act_quant=True)

        # self.perturb_qlinear(False)
        best_eval_results = evaluate(model, self.args)
        # self.perturb_qlinear(True)
        logger.info("Best dev result: {}".format(best_eval_results))

        # self.perturb_qlinear(True)
        best_eval_results = {'wikitext2': 15.3806858062744142, 'c4': 16.195674896240234}
        best_eval_results['wikitext2'] = 100
        best_eval_results['c4'] = 100


        self.loss_list = []
        self.accuracy = []

        self.z = {}
        self.loss_pairs = []


        lr_lambda = lambda step: 1 - self.state.global_step / (2 * args.max_steps)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        # self.lr_scheduler = MultiStepLR(
        #     self.optimizer,
        #     milestones=args.num_steps_before_decay,  # Number of steps after which LR will change
        #     gamma=0.5,  # Multiplicative factor of learning rate decay
        # )
        self.hessian_trace = defaultdict(float)
        self.hessian_decay = 0.05

        self.seeds = []
        self.projected_grads = []


        # optimizer = torch.optim.AdamW([param for name, param in self.named_smoothing_to_optim], lr=args.learning_rate)

        # dataset ='wikitext2'
        # dataloader, testloader = get_loaders(
        #     dataset,
        #     seed=args.seed,
        #     model=args.model,
        #     seqlen=model.seqlen,
        # )
        # testenc = testloader.input_ids

        for epoch in range(epochs_trained, num_train_epochs):

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            times = []
            for step, inputs in enumerate(epoch_iterator):

                # batch = testenc[:, (step * model.seqlen): ((step + 1) * model.seqlen)].to(model.device)

                # for i in range(10000):
                # tr_loss_step, loss_list = self.zo_step(model, inputs)
                # # print(f"Step {i}, loss: {loss_list}", 'grad:', self.projected_grads[0])
                # self.zo_update()
                    # if (i + 1) % 800 == 0:
                    #     for linears in self.qlinears:
                    #         linears.zo_eps = linears.zo_eps / 2
                    #     self.args.learning_rate = self.args.learning_rate / 2



                # time1 = time.time()
                # inputs = self._prepare_inputs(inputs)
                # inputs["use_cache"] = False
                # outputs = model.model.decoder(inputs['input_ids']) if not self.args.isllama else model.model(
                #     inputs['input_ids'])
                # hidden_states = outputs[0]
                # logits = model.lm_head(hidden_states)
                # shift_logits = logits[:, :-1, :]
                # shift_labels = inputs['labels'][:, 1:]
                # loss = self.loss_fct(
                #     shift_logits.reshape(-1, shift_logits.size(-1)),
                #     shift_labels.reshape(-1),
                # )
                #
                # loss.backward()
                #
                # optimizer.step()
                # optimizer.zero_grad()
                #
                # times.append(time.time() - time1)
                # print('time:', np.mean(times))




                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self.args.eval_steps = 1

                log_step = 10
                if self.state.global_step % log_step == 0:
                    logger.info(
                        {'loss': round(np.mean(self.loss_list[-10:]), 4), 'epoch': epoch, 'lr': self._get_learning_rate()})

                # self.loss_list.append(tr_loss_step.item())

                if (self.state.global_step) % self.args.eval_steps == 0:

                        # with torch.no_grad():
                        #     self.smooth_and_quant(model)
                        #     use_temp_parameters(model)
                        #     set_quant_state(model, weight_quant=False, act_quant=True)
                        #
                        #     metrics = self.local_evaluate([], self.eval_dataset)
                        #     metrics["global_step"] = self.state.global_step
                        #     logger.info(f"Eval results: {metrics}")

                        # # wiki_eval
                        # with torch.no_grad():
                        #
                        #     self.perturb_qlinear(False)
                        #     tmp_eval_results = evaluate(model, self.args)
                        #     self.perturb_qlinear(True)
                        #     logger.info(f'W{self.w_train_bits}A{self.a_train_bits}:{tmp_eval_results}')
                        #
                        # if sum(tmp_eval_results.values()) <= sum(best_eval_results.values()):
                        #     best_eval_results = tmp_eval_results
                        #     logger.info("Best dev result: {}".format(best_eval_results))
                        #     self.best_model_ckpt = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

                        # import code
                        # code.interact(local=locals())
                        # mmlu eval
                        lm_eval_model = HFLM(pretrained=model, batch_size=16)
                        # task_manager = lm_eval.tasks.TaskManager()
                        results = lm_eval.simple_evaluate(  # call simple_evaluate
                            model=lm_eval_model,
                            tasks=['mmlu', ],
                            # tasks=['hellaswag', 'arc_easy', 'piqa', 'arc_challenge', 'winogrande'],
                            # tasks=['arc_easy'],
                            num_fewshot=5,
                            cache_requests=True,
                            limit=40,
                        )
                        accs = []
                        for task, metrics in results["results"].items():
                            acc = metrics.get("acc_norm,none")
                            if acc is None:
                                acc = metrics.get("acc,none")
                            accs.append(acc)
                            print(f"{task}: {acc * 100:.2f}%")
                        avg_acc = np.mean(accs) * 100
                        logger.info(f"Average Acc: {avg_acc:.2f}%")
                        exit()

                        if avg_acc >= self.best_objective:
                            self.best_objective = avg_acc
                            logger.info("Best dev result: {}".format(self.best_objective))
                            self.best_model_ckpt = {k: v.detach().half().cpu() for k, v in self.model.state_dict().items()}




                if self.state.global_step % 2000 == 0 or self.state.global_step >= max_steps - 1:
                    logger.info(f"Saving model checkpoint at step {self.state.global_step}")
                    torch.save(self.best_model_ckpt, args.model.split('/')[1] + f'-W{self.w_train_bits}A{self.a_train_bits}_alpaca.pth')

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # return TrainOutput(self.state.global_step, train_loss, metrics)

    # def scale_init(self, model, cache, inps):
    #
    #     layers = model.model.layers
    #     loss_func = torch.nn.MSELoss()
    #     lr = [1, 0.1] + [0.001] * (len(layers) - 2)
    #     zo_eps = 0.005
    #     bs = 8
    #     cache["attention_mask"] = cache["attention_mask"].repeat(bs, 1, 1, 1)
    #     torch.cuda.empty_cache()
    #     quant_inputs = inps
    #     fp_inputs = copy.deepcopy(inps)
    #     with torch.no_grad():
    #         # for l in range(32):
    #         for l in range(len(layers)):
    #             layer = layers[l]
    #             alpha_qkv = self.named_alpha_to_optim[2 * l][1]
    #             alpha_fc = self.named_alpha_to_optim[2 * l + 1][1]
    #
    #             set_quant_state(model, weight_quant=False, act_quant=False)
    #             use_parameters(model)
    #             for j in range(fp_inputs.shape[0] // bs - 1):
    #                 fp_inputs[j * bs:(j + 1) * bs, :] = \
    #                     layer(fp_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                           position_ids=cache["position_ids"])[0]
    #             set_quant_state(model, weight_quant=False, act_quant=True)
    #
    #             epoch = 5 if l <= 3 else 0
    #             for i in range(epoch):
    #                 # loss_list = []
    #                 # if i == 0:
    #                 #     for j in range(inps.shape[0] // bs - 1):
    #                 #         self.named_smoothing_to_optim[2 * l][1].data = (
    #                 #                 self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                 #             1 - alpha_qkv)).clamp(min=1e-5)
    #                 #         self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                 #                 self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                 #             1 - alpha_fc)).clamp(min=1e-5)
    #                 #         smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                 #                                    isllama=self.args.isllama)
    #                 #         loss = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                 #                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #                 #         loss_list.append(loss.item())
    #                 #     print(f'layer {l}', 'before training', f'loss {np.mean(loss_list)}')
    #
    #                 loss = []
    #                 for j in range(inps.shape[0]//bs - 1):
    #
    #                     z_qkv = torch.normal(0, 1, size=alpha_qkv.data.size(), device=alpha_qkv.data.device,
    #                                          dtype=alpha_qkv.data.dtype)
    #                     z_fc = torch.normal(0, 1, size=alpha_fc.data.size(), device=alpha_fc.data.device,
    #                                         dtype=alpha_fc.data.dtype)
    #
    #                     z_qkv = z_qkv / z_qkv.norm()
    #
    #                     zo_eps = 1
    #
    #                     for _ in range(30):
    #                         self.named_smoothing_to_optim[2 * l][1].data = (
    #                                 self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                             1 - alpha_qkv)).clamp(min=1e-5)
    #                         smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                    isllama=self.args.isllama)
    #                         loss0 = loss_func(
    #                             layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                   position_ids=cache["position_ids"])[0].float(),
    #                             fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #
    #                         alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
    #                         self.named_smoothing_to_optim[2 * l][1].data = (
    #                                     self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                                 1 - alpha_qkv)).clamp(min=1e-5)
    #                         smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                    isllama=self.args.isllama)
    #                         loss1 = loss_func(
    #                             layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                   position_ids=cache["position_ids"])[0].float(),
    #                             fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #                         alpha_qkv.data = alpha_qkv.data - 2 * zo_eps * z_qkv
    #                         self.named_smoothing_to_optim[2 * l][1].data = (
    #                                 self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                             1 - alpha_qkv)).clamp(min=1e-5)
    #                         smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                    isllama=self.args.isllama)
    #                         loss2 = loss_func(
    #                             layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                   position_ids=cache["position_ids"])[0].float(),
    #                             fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #                         alpha_qkv.data = torch.ones(alpha_qkv.shape, device=alpha_qkv.device, dtype=alpha_qkv.dtype) * 0.5
    #                         Hv = (loss1 + loss2 - 2 * loss0) / zo_eps ** 2 * z_qkv
    #
    #                         # (Hv / Hv.norm() - z_qkv).norm()
    #                         # ((loss1 + loss2 - 2 * loss0) / zo_eps ** 2).item()
    #
    #                         z_qkv = Hv / Hv.norm()
    #
    #
    #
    #                     alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data + zo_eps * z_fc
    #
    #                     self.named_smoothing_to_optim[2 * l][1].data = (self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(1 - alpha_qkv)).clamp(min=1e-5)
    #                     self.named_smoothing_to_optim[2 * l + 1][1].data = (self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(1 - alpha_fc)).clamp(min=1e-5)
    #                     # from quantize.utils import smooth_and_quant_temporary
    #                     smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,isllama=self.args.isllama)
    #
    #                     loss1 = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #                     alpha_qkv.data = alpha_qkv.data - 2 * zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data - 2 * zo_eps * z_fc
    #
    #                     self.named_smoothing_to_optim[2 * l][1].data = (
    #                             self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                         1 - alpha_qkv)).clamp(min=1e-5)
    #                     self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                             self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                         1 - alpha_fc)).clamp(min=1e-5)
    #                     smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                isllama=self.args.isllama)
    #                     loss2 = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #
    #                     grad = ((loss1 - loss2) / (2 * zo_eps)).item()
    #
    #                     alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data + zo_eps * z_fc
    #
    #                     alpha_qkv.data = alpha_qkv.data - (lr[l]) * grad * z_qkv
    #                     alpha_fc.data = alpha_fc.data - (lr[l]) * grad * z_fc
    #
    #                     # loss.append(loss1.item())
    #                     # if loss1 < loss2:
    #                     #     alpha_qkv.data = alpha_qkv.data - zo_eps * z_qkv
    #                     #     alpha_fc.data = alpha_fc.data - zo_eps * z_fc
    #                     #     loss.append(loss1.item())
    #                     # else:
    #                     #     loss.append(loss1.item())
    #                     #     continue
    #
    #                     loss.append(loss1.item())
    #
    #
    #                 print(f'layer {l}', f'epoch {i}', f'loss {np.mean(loss)}')
    #
    #             # alpha_qkv = 0.5
    #             # alpha_fc = 0.5
    #             self.named_smoothing_to_optim[2 * l][1].data = (
    #                     self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                 1 - alpha_qkv)).clamp(min=1e-5)
    #             self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                     self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                 1 - alpha_fc)).clamp(min=1e-5)
    #             smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits, isllama=self.args.isllama)
    #             loss = []
    #             for j in range(quant_inputs.shape[0]//bs - 1):
    #                 out = layer(quant_inputs[j*bs:(j+1)*bs, :], attention_mask=cache["attention_mask"], position_ids=cache["position_ids"])[0]
    #                 loss.append(loss_func(out.float(), fp_inputs[j*bs:(j+1)*bs, :].float()).item())
    #                 quant_inputs[j * bs:(j + 1) * bs, :] = out
    #                 # quant_inputs[j*bs:(j+1)*bs, :] = layer(quant_inputs[j*bs:(j+1)*bs, :], attention_mask=cache["attention_mask"], position_ids=cache["position_ids"])[0]
    #             print(f'layer {l}', f'loss {np.mean(loss)}')
    #
    #             # 清空显存
    #             torch.cuda.empty_cache()

    def scale_init(self, model, train_dataloader):

        layers = model.model.layers
        loss_func = torch.nn.MSELoss()
        lr = 0.001
        zo_eps = 0.01
        bs = 8
        with torch.no_grad():
            # for l in range(32):
            for epoch in range(10):
                length = len(train_dataloader)
                loss_list = []
                for i, batch in enumerate(train_dataloader):
                    seed = random.randint(0, 10000)
                    torch.manual_seed(seed)

                    inps = batch

                    for idx, (name, param) in enumerate(self.named_alpha_to_optim):
                        z = torch.normal(0, 1, size=param.data.size(), device=param.data.device,
                                         dtype=param.data.dtype)
                        param.data = param.data + zo_eps * z
                        self.named_smoothing_to_optim[idx][1].data = (
                            self.act4alpha[idx].pow(param.data) / self.weight4alpha[idx].pow(
                        1 - param.data)).clamp(min=1e-5)
                    self.smooth_and_quant(model)
                    loss1 = self.zo_forward(model, inps)

                    torch.manual_seed(seed)
                    for idx, (name, param) in enumerate(self.named_alpha_to_optim):
                        z = torch.normal(0, 1, size=param.data.size(), device=param.data.device,
                                         dtype=param.data.dtype)
                        param.data = param.data - 2 * zo_eps * z
                        self.named_smoothing_to_optim[idx][1].data = (
                            self.act4alpha[idx].pow(param.data) / self.weight4alpha[idx].pow(
                        1 - param.data)).clamp(min=1e-5)
                    self.smooth_and_quant(model)
                    loss2 = self.zo_forward(model, inps)

                    grad = ((loss1 - loss2) / (2 * zo_eps)).item()

                    torch.manual_seed(seed)
                    for idx, (name, param) in enumerate(self.named_alpha_to_optim):
                        z = torch.normal(0, 1, size=param.data.size(), device=param.data.device,
                                         dtype=param.data.dtype)
                        param.data = param.data + zo_eps * z

                        param.data = torch.clip(param.data - lr * grad * z, min=0.1, max=0.9)

                    if i >= 10:
                        break

                    loss_list.append(loss1.item())
                    print(f'progress: {i} / {length}')

                print(f'epoch {epoch}', f'loss {np.mean(loss_list)}')

            # alpha_qkv = self.named_alpha_to_optim[2 * l][1]
            # alpha_fc = self.named_alpha_to_optim[2 * l + 1][1]
            #
            # z_qkv = torch.normal(0, 1, size=alpha_qkv.data.size(), device=alpha_qkv.data.device,
            #                      dtype=alpha_qkv.data.dtype)
            # z_fc = torch.normal(0, 1, size=alpha_fc.data.size(), device=alpha_fc.data.device,
            #                     dtype=alpha_fc.data.dtype)
            #
            # alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
            # alpha_fc.data = alpha_fc.data + zo_eps * z_fc
            #
            # self.named_smoothing_to_optim[2 * l][1].data = (
            #         self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
            #     1 - alpha_qkv)).clamp(min=1e-5)
            # self.named_smoothing_to_optim[2 * l + 1][1].data = (
            #         self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
            #     1 - alpha_fc)).clamp(min=1e-5)
            # smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
            #                            isllama=self.args.isllama)






            # 清空显存
            torch.cuda.empty_cache()

    # def scale_init(self, model, cache, inps):
    #
    #     layers = model.model.layers
    #     loss_func = torch.nn.MSELoss()
    #     lr = [1, 0.1] + [0.001] * (len(layers) - 2)
    #     zo_eps = 0.005
    #     bs = 8
    #     cache["attention_mask"] = cache["attention_mask"].repeat(bs, 1, 1, 1)
    #     quant_inputs = inps
    #     fp_inputs = copy.deepcopy(inps)
    #     with torch.no_grad():
    #         # for l in range(32):
    #         for l in range(len(layers)):
    #             layer = layers[l]
    #             alpha_qkv = self.named_alpha_to_optim[2 * l][1]
    #             alpha_fc = self.named_alpha_to_optim[2 * l + 1][1]
    #
    #             set_quant_state(model, weight_quant=False, act_quant=False)
    #             use_parameters(model)
    #             for j in range(fp_inputs.shape[0] // bs - 1):
    #                 fp_inputs[j * bs:(j + 1) * bs, :] = \
    #                     layer(fp_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                           position_ids=cache["position_ids"])[0]
    #             set_quant_state(model, weight_quant=False, act_quant=True)
    #
    #             epoch = 5 if l <= 3 else 0
    #             for i in range(epoch):
    #                 loss_list = []
    #                 if i == 0:
    #                     for j in range(inps.shape[0] // bs - 1):
    #                         self.named_smoothing_to_optim[2 * l][1].data = (
    #                                 self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                             1 - alpha_qkv)).clamp(min=1e-5)
    #                         self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                                 self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                             1 - alpha_fc)).clamp(min=1e-5)
    #                         smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                    isllama=self.args.isllama)
    #                         loss = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #                         loss_list.append(loss.item())
    #                     print(f'layer {l}', 'before training', f'loss {np.mean(loss_list)}')
    #
    #                 loss = []
    #                 for j in range(inps.shape[0]//bs - 1):
    #
    #                     z_qkv = torch.normal(0, 1, size=alpha_qkv.data.size(), device=alpha_qkv.data.device,
    #                                          dtype=alpha_qkv.data.dtype)
    #                     z_fc = torch.normal(0, 1, size=alpha_fc.data.size(), device=alpha_fc.data.device,
    #                                         dtype=alpha_fc.data.dtype)
    #
    #                     alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data + zo_eps * z_fc
    #
    #                     self.named_smoothing_to_optim[2 * l][1].data = (
    #                             self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                         1 - alpha_qkv)).clamp(min=1e-5)
    #                     self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                             self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                         1 - alpha_fc)).clamp(min=1e-5)
    #                     smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                isllama=self.args.isllama)
    #
    #
    #                     loss1 = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #                     alpha_qkv.data = alpha_qkv.data - 2 * zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data - 2 * zo_eps * z_fc
    #
    #                     self.named_smoothing_to_optim[2 * l][1].data = (
    #                             self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                         1 - alpha_qkv)).clamp(min=1e-5)
    #                     self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                             self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                         1 - alpha_fc)).clamp(min=1e-5)
    #                     smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits,
    #                                                isllama=self.args.isllama)
    #                     loss2 = loss_func(layer(quant_inputs[j * bs:(j + 1) * bs, :], attention_mask=cache["attention_mask"],
    #                                             position_ids=cache["position_ids"])[0].float(), fp_inputs[j * bs:(j + 1) * bs, :].float())
    #
    #
    #                     grad = ((loss1 - loss2) / (2 * zo_eps)).item()
    #
    #                     alpha_qkv.data = alpha_qkv.data + zo_eps * z_qkv
    #                     alpha_fc.data = alpha_fc.data + zo_eps * z_fc
    #
    #                     alpha_qkv.data = alpha_qkv.data - (lr[l]) * grad * z_qkv
    #                     alpha_fc.data = alpha_fc.data - (lr[l]) * grad * z_fc
    #
    #
    #                     # loss.append(loss1.item())
    #                     # if loss1 < loss2:
    #                     #     alpha_qkv.data = alpha_qkv.data - zo_eps * z_qkv
    #                     #     alpha_fc.data = alpha_fc.data - zo_eps * z_fc
    #                     #     loss.append(loss1.item())
    #                     # else:
    #                     #     loss.append(loss1.item())
    #                     #     continue
    #
    #                     loss.append(loss1.item())
    #
    #
    #                 print(f'layer {l}', f'epoch {i}', f'loss {np.mean(loss)}')
    #
    #             # alpha_qkv = 0.5
    #             # alpha_fc = 0.5
    #             self.named_smoothing_to_optim[2 * l][1].data = (
    #                     self.act4alpha[2 * l].pow(alpha_qkv) / self.weight4alpha[2 * l].pow(
    #                 1 - alpha_qkv)).clamp(min=1e-5)
    #             self.named_smoothing_to_optim[2 * l + 1][1].data = (
    #                     self.act4alpha[2 * l + 1].pow(alpha_fc) / self.weight4alpha[2 * l + 1].pow(
    #                 1 - alpha_fc)).clamp(min=1e-5)
    #             smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits, isllama=self.args.isllama)
    #             loss = []
    #             for j in range(quant_inputs.shape[0]//bs - 1):
    #                 out = layer(quant_inputs[j*bs:(j+1)*bs, :], attention_mask=cache["attention_mask"], position_ids=cache["position_ids"])[0]
    #                 loss.append(loss_func(out.float(), fp_inputs[j*bs:(j+1)*bs, :].float()).item())
    #                 quant_inputs[j * bs:(j + 1) * bs, :] = out
    #                 # quant_inputs[j*bs:(j+1)*bs, :] = layer(quant_inputs[j*bs:(j+1)*bs, :], attention_mask=cache["attention_mask"], position_ids=cache["position_ids"])[0]
    #             print(f'layer {l}', f'loss {np.mean(loss)}')
    #
    #             # 清空显存
    #             torch.cuda.empty_cache()

    def omniquant(
            self,
            args,
            train_dataloader,
            logger=None,
    ):

        model = self.model
        is_llama = args.isllama
        if is_llama:
            layers = model.model.layers if type(model) != DataParallel else model.module.model.layers
        else:
            layers = model.model.decoder.layers if type(model) != DataParallel else model.module.model.decoder.layers
        if self.target_layer_idx is None:
            # self.target_layer_idx = [0, 1, 2, 3]
            # 2 epoch, ppl 15.4, args.let_lr = 5e-3, args.lwc_lr = 5e-3, init=3
            # self.target_layer_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.target_layer_idx = list(range(len(layers)))
            # self.target_layer_idx = [0, 1, 2, 3, 4, 5]
            # self.target_layer_idx = [4, 5]
        logger.info("Starting ...")

        set_quant_state(self.model, weight_quant=False, act_quant=False)
        # move embedding layer and first layer to target device
        dev = model.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        use_shift = True
        if is_llama:
            use_shift = False  # deactivate channel-wise shifting for llama model and weight-only quantization

        for idx in self.target_layer_idx:
            layer = layers[idx]
            params = get_omni_parameters(layer, use_shift)
            for param in params:
                param.requires_grad = True

        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
        inps = torch.zeros(
            (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {"i": 0}

        # catch the first layer input
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.is_llama = False

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += inp.shape[0]
                cache["attention_mask"] = kwargs["attention_mask"]
                if self.is_llama:
                    cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        start_idx = min(self.target_layer_idx)
        layers[start_idx] = Catcher(layers[start_idx])
        layers[start_idx].is_llama = is_llama

        with torch.no_grad():
            for batch in train_dataloader:
                data = batch['input_ids'].cuda()
                for j in range(data.shape[0]):
                    if cache["i"] >= args.nsamples:
                        break
                    try:
                        model(data[j,].unsqueeze(0))
                    except ValueError:
                        pass

        # move embedding layer and first layer to cpu
        layers[start_idx] = layers[start_idx].module

        attention_mask = cache["attention_mask"]

        args.batch_size = 1

        if attention_mask is not None:
            attention_mask_batch = attention_mask.repeat(args.batch_size,1, 1, 1).float()
        else:
            logger.info(
                "No attention mask caught from the first layer."
                " Seems that model's attention works without a mask."
            )
            attention_mask_batch = None

        loss_func = torch.nn.MSELoss()
        if is_llama:
            position_ids = cache["position_ids"]
        else:
            position_ids = None

        quant_inps = inps
        fp_inps = copy.deepcopy(inps)
        args.aug_loss = True #opt false, llama2 true
        fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None  # take output of quantization model as input

        for i in self.target_layer_idx:

            logger.info(f"=== Start quantize layer {i} ===")
            qlayer = layers[i].to(dev)

            # init smooth parameters
            set_quant_state(qlayer, weight_quant=False,
                            act_quant=True)  # weight will be manually quantized before forward

            args.epochs = 2
            args.let_lr = 1e-2 if is_llama else 5e-3 # opt 5e-3
            args.lwc_lr = 5e-3 if is_llama else 1e-2 # llama 5e-3, opt 1e-2
            args.batch_size = 1
            if args.epochs > 0:
                with torch.no_grad():
                    qlayer.float()  # required for AMP training
                # create optimizer
                optimizer = torch.optim.AdamW(
                    [{"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                     {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}], weight_decay=0.0)
                loss_scaler = NativeScalerWithGradNormCount()
                # loss_scaler = torch.cuda.amp.GradScaler()

                set_quant_state(qlayer, weight_quant=False,
                                act_quant=False)  # deactivate quantization for loss calculation
                with torch.no_grad():
                    with traincast():
                        for j in range(args.nsamples // args.batch_size):
                            index = j * args.batch_size
                            fp_inps[index:index + args.batch_size, ] = \
                                qlayer(fp_inps[index:index + args.batch_size, ], attention_mask=attention_mask_batch,
                                       position_ids=position_ids)[0]
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,
                                                      position_ids=position_ids)[0]

                set_quant_state(qlayer, weight_quant=False, act_quant=True)
                for epochs in range(args.epochs):
                    loss_list = []
                    norm_list = []
                    for j in range(args.nsamples // args.batch_size):
                        index = j * args.batch_size
                        # obtain output of quantization model
                        with traincast():
                            smooth_and_quant_temporary(qlayer, self.w_train_bits, self.a_train_bits, is_llama)

                            # inps[index:index + args.batch_size, ].requires_grad_()
                            quant_out = qlayer(quant_inps[index:index + args.batch_size, ], attention_mask=attention_mask_batch,
                                   position_ids=position_ids)[0]

                            loss = loss_func(fp_inps[index:index + args.batch_size, ], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index + args.batch_size, ], quant_out)

                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")

                        loss_list.append(loss.detach().cpu())
                        optimizer.zero_grad()
                        # loss_scaler._scaler.scale(loss).backward(create_graph=False, retain_graph=False)
                        norm = loss_scaler(loss, optimizer, parameters=get_omni_parameters(qlayer, use_shift))
                        norm = norm.cpu()
                        norm_list.append(norm.data)

                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    logger.info(
                        f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024 ** 2} ")

                del optimizer

            qlayer.half()

            smooth_and_quant_temporary(qlayer, self.w_train_bits, self.a_train_bits, is_llama)
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    set_quant_state(qlayer, weight_quant=False, act_quant=True)
                    for j in range(args.nsamples):
                        quant_inps[j] = \
                            qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    # set_quant_state(qlayer, weight_quant=False, act_quant=False)
                    # for j in range(args.nsamples):
                    #     fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0),
                    #                    attention_mask=attention_mask_batch,
                    #                    position_ids=position_ids)[0]

            layers[i].to('cpu')
            clear_temp_variable(layers[i])
            torch.cuda.empty_cache()

        del inps
        del quant_inps
        del fp_inps
        params = get_omni_parameters(model, use_shift)
        for param in params:
            param.requires_grad = False
        clear_temp_variable(model)
        torch.cuda.empty_cache()
        model.config.use_cache = use_cache
        self.target_layer_idx = None
        set_quant_state(self.model, weight_quant=True, act_quant=True)

        model = model.cuda()

        return model

    def scale_update(self, model):
        layers = model.model.decoder.layers

        alpha = 0.5
        layer_name_prefix = "model.decoder.layers"
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
        self.scale_range = []
        for i in range(len(layers)):
            layer = layers[i]
            for name, module in layer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = self.act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device='cuda',
                                                                                        dtype=torch.float16).clamp(
                                min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)

                            layer.qkv_smooth_scale.data = scale

    def smooth_ln_pre_training(self, model):

        if self.args.isllama:
            layers = model.model.layers
        else:
            layers = model.model.decoder.layers

        for layer in layers:
            smooth_ln(layer, isllama=self.args.isllama)


    def smooth_and_quant(self, model, extra=False):

        if self.args.isllama:
            layers = model.model.layers if type(model) != DataParallel else model.module.model.layers
        else:
            layers = model.model.decoder.layers if type(model) != DataParallel else model.module.model.decoder.layers
        idx = 0
        for layer in layers:
            smooth_and_quant_temporary(layer, self.w_train_bits, self.a_train_bits, isllama=self.args.isllama, extra=extra, idx=idx)

        # model.model.norm = model.model.norm.to(layer.mlp.down_proj.temp_weight.device)

    ############## MeZO ##############

    def change_quantizer_bits(self, model, eval):

        layers = model.model.decoder.layers if not self.isllama else model.model.layers
        for layer in layers:
            for name, module in layer.named_modules():
                if isinstance(module, QuantLinear):
                    if eval:
                        module.weight_quantizer.change_n_bits(self.w_eval_bits)
                        module.act_quantizer.change_n_bits(self.a_eval_bits)
                        module.use_temporary_parameter = False
                    else:
                        module.weight_quantizer.change_n_bits(self.w_train_bits)
                        module.act_quantizer.change_n_bits(self.a_train_bits)
                    # module.use_temporary_parameter = True

    def zo_Hessian_step_update(self, model, inputs, zo_learning_rate, Hessian_smooth):

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        random_seed = np.random.randint(1000000000)
        with torch.no_grad():
            loss_original = self.zo_forward(model, inputs)

            # first function evaluation
            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # second function evaluation
            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix,
                                                              scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            model = self.efficient_Hessian_perturb_parameters(model, random_seed, self.Hessian_matrix, scaling_factor=1)

            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                Hessian_temp = (z * z).float()
                Hessian_trace = (torch.abs(loss1 + loss2 - 2 * loss_original) * Hessian_temp / (
                        2 * self.args.zo_eps * self.args.zo_eps)).mean()
                # Hessian_estimator = (torch.abs(loss1+loss2-2 * loss_original)* Hessian_temp * Hessian_smooth /(2 * self.args.zo_eps*self.args.zo_eps))

                # self.Hessian_matrix[name] = ((1-Hessian_smooth) * self.Hessian_matrix[name] +  Hessian_estimator)

                grad = (loss1 - loss2) / (2 * self.args.zo_eps) * z / torch.sqrt(self.Hessian_matrix[name])
                param.data = param.data - zo_learning_rate * (grad + self.args.weight_decay * param.data)

            # print(torch.sqrt(self.Hessian_matrix['model.decoder.layers.11.self_attn.q_proj.weight']).max().item())

            # loss_out = self.zo_forward(model, inputs)

        return loss1


    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1.0):

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        # with torch.no_grad():
        #     for name, param in self.named_parameters_to_optim:
        #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
        #                          dtype=torch.float32, generator=gen)
        #         tmp32 = param.detach().float()
        #         tmp32.add_(scaling_factor * self.args.zo_eps * z)
        #         param.copy_(tmp32.to(param.dtype))
                # param.data.add_(scaling_factor * self.args.zo_eps * z)

        with torch.no_grad():
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                param.data.add_(scaling_factor * self.args.zo_eps * z)
            # for name, param in self.named_smoothing_to_optim:
            #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
            #                      dtype=param.data.dtype)
            #     param.data.add_(scaling_factor * self.args.zo_eps * z)
                # param.data = param.data + scaling_factor * self.args.zo_eps * z


            # for name, param in self.named_smoothing_to_optim:
            #     if 'scale' in name:
            #         scale = torch.abs(param)
            #         z = torch.normal(mean=torch.zeros_like(param), std=scale)
            #
            #         param.data.add_(scaling_factor * self.scale_zo_eps * z)

            # if not use_z:
            #     self.z[name] = z.detach()


    @staticmethod
    def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None,
                                     return_dict=None, **kwargs):
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
            outputs = self.forward(input_ids=input_ids, **kwargs)

        if labels is None:
            return outputs
        logits = outputs.logits

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
        shift_labels[shift_labels == self.config.pad_token_id] = -100

        # Apply option len (do not calculate loss on the non-option part)
        if option_len is not None:
            for _i, _len in enumerate(option_len):
                shift_labels[_i, :-_len] = -100

        # Calculate the loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if num_options is not None:
            # Train as a classification tasks
            log_probs = F.log_softmax(shift_logits, dim=-1)
            mask = shift_labels != -100  # Option part
            shift_labels[~mask] = 0  # So that it doesn't mess up with indexing

            selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
                -1)  # (bsz x num_options, len)
            selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)  # (bsz x num_options)

            if any([x != num_options[0] for x in num_options]):
                # Multi choice tasks with different number of options
                loss = 0
                start_id = 0
                count = 0
                while start_id < len(num_options):
                    end_id = start_id + num_options[start_id]
                    _logits = selected_log_probs[start_id:end_id].unsqueeze(0)  # (1, num_options)
                    _labels = labels[start_id:end_id][0].unsqueeze(0)  # (1)
                    loss = loss_fct(_logits, _labels) + loss
                    count += 1
                    start_id = end_id
                loss = loss / count
            else:
                num_options = num_options[0]
                selected_log_probs = selected_log_probs.view(-1, num_options)  # (bsz, num_options)
                labels = labels.view(-1, num_options)[:, 0]  # Labels repeat so we only take the first one
                loss = loss_fct(selected_log_probs, labels)
        else:
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            with self.compute_loss_context_manager():
                with torch.no_grad():
                    inputs = self._prepare_inputs(inputs)
                    # inputs["use_cache"] = False
                    outputs = model.model.decoder(inputs['input_ids']) if not self.args.isllama else model.model(inputs['input_ids'])
                    hidden_states = outputs[0]
                    logits = model.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = inputs['labels'][:, 1:]
                    loss = self.loss_fct(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                    )
                    # loss = self.compute_loss(model, inputs)
                # loss = self.forward_wrap_with_option_len(model, **inputs, return_dict=True).loss
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def update_scales(self, model, tokenizer, train_dataloader, num_samples=128):

        model.eval()
        device = next(model.parameters()).device
        act_scales = {}
        calibration_dataset = train_dataloader

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, QuantLinear):
                hooks.append(
                    m.register_forward_hook(functools.partial(stat_input_hook, name=name))
                )

        with torch.no_grad():
            # for i in tqdm(range(min(num_samples, len(calibration_dataset)))):
            for i, data in tqdm(enumerate(calibration_dataset)):
                self.zo_forward(model, data)
                if i > (num_samples // data['input_ids'].shape[0]):
                    break
                print(f"Processed {i + 1}/{num_samples} samples for scale estimation.")

        for h in hooks:
            # h = h.cpu()
            h.remove()

        return act_scales

    def get_alpha(self, model, train_dataloader, num_samples=512):

        beta = 0.9
        max_alpha = 1
        min_alpha_list = [max_alpha * (beta ** i) for i in range(1, 15)]
        # min_alpha_list = [3, 1, 0.8, 0.6, 0.4]

        layers = model.model.decoder.layers
        best_loss_gap = 100000
        best_min_alpha = None
        best_iter = None

        for iter in range(len(min_alpha_list)):

            min_alpha = min_alpha_list[iter]
            loss_gaps = []

            for i in range(len(layers)):
                layer = layers[i]
                for name, module in layer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in self.pairs.keys():
                            if key in name:
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                act = self.act_scales[f"{self.layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                        dtype=torch.float16).clamp(min=1e-5)
                                shift = self.act_shifts[f"{self.layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                                   dtype=torch.float16)
                                alpha = min_alpha + (act - act.min()) / (act.max() - act.min()) * (max_alpha - min_alpha)

                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)

                                # if hasattr(layer, f"{self.pairs[key]}_smooth_shift"):
                                getattr(layer, f"{self.pairs[key]}_smooth_shift").data = shift
                                getattr(layer, f"{self.pairs[key]}_smooth_scale").data = scale

                                # else:
                                #     layer.register_parameter(f"{self.pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                                #     layer.register_parameter(f"{self.pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

            with torch.no_grad():
                for i, inputs in enumerate(train_dataloader):

                    self.smooth_and_quant(model)
                    loss1 = self.zo_forward(model, inputs)
                    self.change_quantizer_bits(model, eval=True)
                    self.smooth_and_quant(model)
                    true_loss = self.zo_forward(model, inputs)
                    self.change_quantizer_bits(model, eval=False)

                    loss_gaps.append((loss1 - true_loss).abs().item())

                    if i > (num_samples // inputs['input_ids'].shape[0]):
                        break

            loss_gap = np.mean(loss_gaps)
            if loss_gap < best_loss_gap:
                best_loss_gap = loss_gap
                best_min_alpha = min_alpha
                best_iter = iter

            print('iter:', iter, 'best_iter:', best_iter,
                  'min_alpha:', min_alpha, 'best_min_alpha:', best_min_alpha,
                  'loss gap:', loss_gap, 'best loss gap:', best_loss_gap)



        return best_min_alpha



    def test(self, model, inputs, iter=5):

        loss = []
        for i in range(5):
            for j in range(iter):

                self.zo_random_seed = np.random.randint(1000000000)

                self.zo_perturb_parameters(scaling_factor=1.0)
                self.smooth_and_quant(model)
                loss1 = self.zo_forward(model, inputs)
                self.change_quantizer_bits(model, eval=True)
                self.smooth_and_quant(model)
                true_loss = self.zo_forward(model, inputs)
                self.change_quantizer_bits(model, eval=False)
                self.zo_perturb_parameters(scaling_factor=-1.0)

                loss.append((loss1 - true_loss).abs().item())

        print('loss gap:', np.mean(loss), loss)

    def reverse_qlinear(self):

        for qlinear in self.qlinears:
            qlinear.scaling_factor = - qlinear.scaling_factor

    def perturb_qlinear(self, perturb):

        for qlinear in self.qlinears:
            qlinear.perturb_forward = perturb

    def zo_step(self, model, inputs, num_trials=1):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        # self.scale_zo_eps = 0.01

        self.seeds = []
        self.projected_grads = []

        # self.zo_perturb_parameters(scaling_factor=1.0)

        for i in range(num_trials):

            self.zo_random_seed = np.random.randint(1000000000)

            # self.perturb_qlinear(False)
            # loss0 = self.zo_forward(model, inputs)
            # self.perturb_qlinear(True)

            torch.manual_seed(self.zo_random_seed)
            loss1 = self.zo_forward(model, inputs)
            self.reverse_qlinear()

            torch.manual_seed(self.zo_random_seed)
            loss2 = self.zo_forward(model, inputs)
            self.reverse_qlinear()

            # assert torch.isnan(loss1).sum() == 0, 'nan detected at iteration {}'.format(self.state.global_step)

            self.projected_grad = np.clip(((loss1 - loss2) / (2 * self.args.zo_eps)).item(), -0.2 / self.args.zo_eps, 0.2 / self.args.zo_eps)

            self.seeds.append(self.zo_random_seed)
            self.projected_grads.append(self.projected_grad)


        return loss1, None
        # return loss1, [loss0.item(), loss1.item(), loss2.item()]

    def zo_perturb_smooth_parameters(self, random_seed=None, scaling_factor=1.0):

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_smooth_random_seed)

        with torch.no_grad():
            for i in range(len(self.named_alpha_to_optim)):
                alpha = self.named_alpha_to_optim[i][1]
                z = torch.normal(0, 1, size=alpha.data.size(), device=alpha.data.device,
                                 dtype=alpha.data.dtype)
                alpha.data.add_(scaling_factor * self.scale_zo_eps * z)
                self.named_smoothing_to_optim[i][1].data = (self.act4alpha[i].pow(alpha) / self.weight4alpha[i].pow(1 - alpha)).clamp(min=1e-5)

            # for name, param in self.named_smoothing_to_optim:
            #     scale = torch.abs(param)
            #     z = torch.normal(mean=torch.zeros_like(param), std=scale)
            #     param.data.add_(scaling_factor * self.scale_zo_eps * z)

    def zo_smooth_step(self, model, inputs):

        self.zo_smooth_random_seed = np.random.randint(1000000000)
        self.scale_zo_eps = 0.01

        self.zo_perturb_smooth_parameters(scaling_factor=1.0)
        self.smooth_and_quant(model)
        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_smooth_parameters(scaling_factor=-2.0)
        self.smooth_and_quant(model)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad_scale = ((loss1 - loss2) / (2 * self.scale_zo_eps)).item()

        self.zo_perturb_smooth_parameters(scaling_factor=1.0)

        return loss1

    def zo_smooth_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_smooth_random_seed)
        self.scale_lr = 1e-3

        for i in range(len(self.named_alpha_to_optim)):
            alpha = self.named_alpha_to_optim[i][1]
            z = torch.normal(0, 1, size=alpha.data.size(), device=alpha.data.device,
                             dtype=alpha.data.dtype)
            alpha.data.sub_(self.scale_lr * self.projected_grad_scale * z)

        # for name, param in self.named_smoothing_to_optim:
        #     # if 'scale' in name:
        #     # scale = torch.abs(param)
        #     # z = torch.normal(mean=torch.zeros_like(param), std=scale)
        #     z = torch.normal(0, 1, size=alpha.data.size(), device=alpha.data.device,
        #                      dtype=alpha.data.dtype)
        #     param.data.sub_(self.scale_lr * self.projected_grad_scale * z)
        #     param.data = torch.clip(param.data, min=0.0664)

    # def zo_update(self, model):
    #     """
    #     Update the parameters with the estimated gradients.
    #     """
    #
    #     # Reset the random seed for sampling zs
    #     torch.manual_seed(self.zo_random_seed)
    #     # self.scale_lr = 1e-3
    #
    #     for i, (name, param) in enumerate(self.named_parameters_to_optim):
    #         # Resample z
    #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
    #         param.data.sub_(self._get_learning_rate() * self.projected_grad * z)
    #
    #     self.lr_scheduler.step()

    def zo_update(self):
        q = len(self.seeds)
        assert q == len(self.projected_grads)
        # lr = self._get_learning_rate()
        lr = self.args.learning_rate

        for seed_i, pg_i in zip(self.seeds, self.projected_grads):
            torch.manual_seed(seed_i)

            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data.sub_(lr * (pg_i * z / q))

        # for name, param in self.named_parameters_to_optim:
        #
        #     # 累加多个扰动对应的更新方向
        #     upd = torch.zeros_like(param.data)
        #
        #     # 为了不同参数张量在同一个seed下也能稳定复现各自的 z，
        #     # 我们把参数名的hash混入seed中（避免所有参数拿到同一个z）。
        #     name_hash = (hash(name) & 0xFFFFFFFF)
        #
        #     for seed_i, pg_i in zip(self.seeds, self.projected_grads):
        #         gen = torch.Generator(device=param.data.device)
        #         gen.manual_seed((seed_i ^ name_hash) & 0xFFFFFFFF)
        #
        #         z_i = torch.normal(
        #             mean=0.0, std=1.0,
        #             size=param.data.size(),
        #             generator=gen,
        #             device=param.data.device,
        #             dtype=param.data.dtype
        #         )
        #         # 累加：pg_i * z_i
        #         upd.add_(pg_i * z_i)
        #
        #     # 多扰动平均 + 一次性更新
        #     param.data.sub_(lr * (upd / q))

        # self.lr_scheduler.step()

    # def zo_update(self, model):
    #     """
    #     Update the parameters with the estimated gradients.
    #     """
    #
    #     # Reset the random seed for sampling zs
    #     torch.manual_seed(self.zo_random_seed)
    #     # self.scale_lr = 1e-3
    #
    #     for i, (name, param) in enumerate(self.named_parameters_to_optim):
    #         # Resample z
    #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
    #         param.data.sub_(self._get_learning_rate() * self.projected_grad * z)
    #
    #     self.lr_scheduler.step()

    def layer_selection(self, trace_list, min_len=3, max_len=6):

        n = len(trace_list)
        max_obj = float('-inf')
        best_start = best_end = -1

        prefix_sum = [0]
        for val in trace_list:
            prefix_sum.append(prefix_sum[-1] + val)

        for i in range(n):
            start_j = i + min_len - 1
            end_j = n - 1 if max_len is None else min(n - 1, i + max_len - 1)

            for j in range(start_j, end_j + 1):
                length = j - i + 1
                total = prefix_sum[j + 1] - prefix_sum[i]
                avg = total / length
                obj = avg
                if obj > max_obj:
                    max_obj = obj
                    best_start, best_end = i, j

        return list(range(best_start, best_end + 1))



    ############## Misc overload functions ##############
    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args

            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = False

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(),
                                                                    train_samples, eval_sample, self.tokenizer,
                                                                    max_length=self.args.max_length,
                                                                    sfc=self.args.sfc, icl_sfc=self.args.icl_sfc,
                                                                    generation=self.task.generation,
                                                                    max_new_tokens=self.args.max_new_tokens
                                                                    )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id],
                                                          option_len=sfc_option_lens[candidate_id])

                outputs.append({"log_probs": selected_log_probs,
                                "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def local_evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                   eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                # ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                # or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        parameters = iter(parameters) if parameters is not None else None
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
