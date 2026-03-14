import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
from collections import defaultdict


FP8_DTYPE = torch.float8_e4m3fn  # or torch.float8_e5m2
COMPUTE_DTYPE = torch.float16

def q_smooth(weight, bias, scales, shifts):

    shifts = shifts[0]
    if bias is not None:
        bias = bias + weight @ shifts
    else:
        bias = weight @ shifts

    weight = weight * scales[0].view(1, -1)
    weight = weight / scales[1].view(-1, 1)

    bias = bias / scales[1].view(-1)

    return weight, bias


def k_smooth(weight, bias, scales, shifts):
    shifts = shifts[0]
    if bias is not None:
        bias = bias + weight @ shifts
    else:
        bias = weight @ shifts

    weight = weight * scales[0].view(1, -1)
    weight = weight * scales[1].view(-1, 1)

    bias = bias * scales[1].view(-1)

    return weight, bias

def v_smooth(weight, bias, scales, shifts):

    if bias is not None:
        bias = bias + weight @ shifts[0]
    else:
        bias = weight @ shifts[0]
    bias = (bias - shifts[1]) / scales[1]

    weight = weight * scales[0].view(1, -1)
    weight = weight / scales[1].view(-1, 1)

    return weight, bias

def o_smooth(weight, bias, scales, shifts):

    if bias is not None:
        bias = bias + weight@shifts[0]
    else:
        bias = weight@shifts[0]
    weight = weight * scales[0].view(1, -1)

    return weight, bias

def mlp_smooth(weight, bias, scales, shifts):

    if bias is not None:
        bias = bias + weight @ shifts[0]
    else:
        bias = weight @ shifts[0]
    weight = weight * scales[0].view(1, -1)

    return weight, bias

def mlp_down_proj_smooth(weight, bias, scales, shifts):

    return weight, bias


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
            self,
            org_module: nn.Linear,
            weight_quant_params: dict = {},
            act_quant_params: dict = {},
            disable_input_quant=False,
            isllama=False,
            zo_eps=1e-3
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        # self.register_buffer('weight',org_module.weight)
        self.register_parameter('weight', org_module.weight)
        if org_module.bias is not None:
            # self.register_buffer('bias',org_module.bias)
            self.register_parameter('bias', org_module.bias)
        else:
            self.bias = None

        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = True
        weight_quant_params['lwc'] = True
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.scale = None

    def init_smoothing(self, smooth_scale: list = [], smooth_shift: list = []):

            self.smoothing_scale = smooth_scale
            self.smoothing_shift = smooth_shift

    def forward(self, input_t: torch.Tensor):

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        # elif self.use_weight_quant:
        #     weight = self.weight_quantizer(self.weight)
        #     bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input_t = self.act_quantizer(input_t)

        out = self.fwd_func(input_t, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


#
# class QuantLinear(nn.Module):
#     """
#     Quantized Module that can perform quantized convolution or normal convolution.
#     To activate quantization, please use set_quant_state function.
#     """
#     def __init__(
#         self,
#         org_module: nn.Linear,
#         weight_quant_params: dict = {},
#         act_quant_params: dict = {},
#         disable_input_quant=False,
#         isllama=False,
#         zo_eps=1e-3
#     ):
#         super().__init__()
#         self.smoothing_scale = None
#         self.smoothing_shift = None
#
#         self.fwd_kwargs = dict()
#         self.fwd_func = F.linear
#         # self.register_buffer('weight',org_module.weight)
#         self.register_parameter('weight', org_module.weight)
#         if org_module.bias is not None:
#             # self.register_buffer('bias',org_module.bias)
#             self.register_parameter('bias', org_module.bias)
#         else:
#             self.bias = None
#
#         self.in_features = org_module.in_features
#         self.out_features = org_module.out_features
#         # de-activate the quantized forward default
#         self.use_weight_quant = False
#         self.use_act_quant = True
#         # initialize quantizer
#         self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
#         if not disable_input_quant:
#             self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
#         else:
#             self.act_quantizer = None
#
#         self.disable_input_quant = disable_input_quant
#         self.use_temporary_parameter = False
#
#         self.perturb_forward = False
#         self.zo_eps = zo_eps
#         self.scaling_factor = 1.0
#         self.smooth = None
#         self.replaced = False
#
#         self.compute_dtype = COMPUTE_DTYPE
#         self.fp8_dtype = FP8_DTYPE
#
#     def init_smoothing(self, smooth_scale: list = [], smooth_shift: list = []):
#
#         self.smoothing_scale = smooth_scale
#         self.smoothing_shift = smooth_shift
#
#     def to_fp8(self):
#
#         self.weight.data = self.weight.to(self.fp8_dtype)
#
#     def forward(self, input_t: torch.Tensor):
#
#         if self.use_temporary_parameter:
#             weight = self.temp_weight
#             bias = self.temp_bias
#         else:
#             weight = self.weight
#             bias = self.bias
#
#         weight = weight.to(self.compute_dtype)
#
#         if self.use_act_quant and not self.disable_input_quant:
#             input_t = self.act_quantizer(input_t)
#
#         out = self.fwd_func(input_t, weight, bias, **self.fwd_kwargs)
#
#         return out
#
#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant



    # def forward(self, input_t: torch.Tensor):
    #
    #     if self.use_temporary_parameter:
    #         weight = self.temp_weight
    #         bias = self.temp_bias
    #     # elif self.use_weight_quant:
    #     #     weight = self.weight_quantizer(self.weight)
    #     #     bias = self.bias
    #     else:
    #         weight = self.weight
    #         bias = self.bias
    #
    #     if self.replaced:
    #         weight = weight.to(self.compute_dtype)
    #         if self.use_act_quant:
    #             # input_t = self.act_quantizer(input_t, quant_only=True, activation=True)
    #             input_t = self.act_quantizer(input_t)
    #         out = self.fwd_func(input_t, weight, bias, **self.fwd_kwargs)
    #         return out
    #
    #     # if torch.isinf(weight).sum() != 0 or torch.isinf(input_t).sum() != 0:
    #     if self.use_act_quant:
    #         input_t = self.act_quantizer(input_t, inplace=False)
    #
    #         if self.perturb_forward:
    #             z = torch.normal(mean=0, std=1, size=weight.data.size(), device=weight.data.device, dtype=weight.data.dtype)
    #             tmp_weight, tmp_bias = self.smooth(weight + self.scaling_factor * self.zo_eps * z, bias, self.smoothing_scale, self.smoothing_shift)
    #             out = self.fwd_func(input_t, self.weight_quantizer(tmp_weight), tmp_bias, **self.fwd_kwargs)
    #         else:
    #             tmp_weight, tmp_bias = self.smooth(weight, bias, self.smoothing_scale, self.smoothing_shift)
    #             out = self.fwd_func(input_t, self.weight_quantizer(tmp_weight), tmp_bias, **self.fwd_kwargs)
    #
    #     else:
    #         if self.perturb_forward:
    #             z = torch.normal(mean=0, std=1, size=weight.data.size(), device=weight.data.device,
    #                              dtype=weight.data.dtype)
    #             tmp_weight, tmp_bias = weight + self.scaling_factor * self.zo_eps * z, bias
    #             # tmp_weight, tmp_bias = self.smooth(weight + self.scaling_factor * self.zo_eps * z, bias)
    #             out = self.fwd_func(input_t, self.weight_quantizer(tmp_weight), tmp_bias, **self.fwd_kwargs)
    #         else:
    #             tmp_weight, tmp_bias = weight, bias
    #             # tmp_weight, tmp_bias = self.smooth(weight, bias, self.smoothing_scale, self.smoothing_shift)
    #             out = self.fwd_func(input_t, self.weight_quantizer(tmp_weight), tmp_bias, **self.fwd_kwargs)
    #
    #     return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


# class QuantLinear(nn.Module):
#     """
#     Quantized Module that can perform quantized convolution or normal convolution.
#     To activate quantization, please use set_quant_state function.
#     """
#     def __init__(
#         self,
#         org_module: nn.Linear,
#         weight_quant_params: dict = {},
#         act_quant_params: dict = {},
#         disable_input_quant=False,
#         isllama=False,
#         zo_eps=1e-3
#     ):
#         super().__init__()
#         self.smoothing_scale = None
#         self.smoothing_shift = None
#
#         self.fwd_kwargs = dict()
#         self.fwd_func = F.linear
#         # self.register_buffer('weight',org_module.weight)
#         self.register_parameter('weight', org_module.weight)
#         if org_module.bias is not None:
#             # self.register_buffer('bias',org_module.bias)
#             self.register_parameter('bias', org_module.bias)
#         else:
#             self.bias = None
#
#         self.in_features = org_module.in_features
#         self.out_features = org_module.out_features
#         # de-activate the quantized forward default
#         self.use_weight_quant = False
#         self.use_act_quant = True
#         # initialize quantizer
#         self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
#         if not disable_input_quant:
#             self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
#         else:
#             self.act_quantizer = None
#
#         self.disable_input_quant = disable_input_quant
#         self.use_temporary_parameter = False
#
#         self.perturb_forward = False
#         self.zo_eps = zo_eps
#         self.scaling_factor = 1.0
#         self.smooth = None
#
#         self.weight.data = self.weight_quantizer(self.weight.data)
#
#     def init_smoothing(self, smooth_scale: list = [], smooth_shift: list = []):
#
#         self.smoothing_scale = smooth_scale
#         self.smoothing_shift = smooth_shift
#
#
#     def forward(self, input_t: torch.Tensor):
#
#         if self.use_act_quant:
#             input_t = self.act_quantizer(input_t, inplace=False)
#
#         out = self.fwd_func(input_t, self.weight, self.bias, **self.fwd_kwargs)
#
#         return out
#
#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant
