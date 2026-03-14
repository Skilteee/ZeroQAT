from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *
from quantize.omni_norm import OmniLlamaRMSNorm, OmniLayerNorm


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)


def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)


def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)


def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination


def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[
                                                                   truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)

def smooth_ln(model, isllama):

    if isllama:
        smooth_ln_replace(model.input_layernorm, model.qkv_smooth_scale, model.qkv_smooth_shift)
        smooth_ln_replace(model.post_attention_layernorm, model.fc1_smooth_scale, model.fc1_smooth_shift)
    else:
        smooth_ln_replace(model.self_attn_layer_norm, model.qkv_smooth_scale, model.qkv_smooth_shift)
        smooth_ln_replace(model.final_layer_norm, model.fc1_smooth_scale, model.fc1_smooth_shift)


def smooth_and_quant_temporary(model, w_train_bits, a_train_bits, isllama, extra=False, idx=0):
    # if args.let:
    # with torch.no_grad():

    # for name, module in model.named_parameters():
    #     if "smooth_scale" in name:
    #         module.data = truncate_number(module)

    if isllama:
        # model.self_attn.o_proj.temp_weight = model.self_attn.o_proj.weight.detach().clone()
        # model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight.detach().clone()

        if a_train_bits == 16:
            model.self_attn.q_proj.temp_weight = model.self_attn.q_proj.weight.detach().clone()
            model.self_attn.k_proj.temp_weight = model.self_attn.k_proj.weight.detach().clone()
            model.self_attn.v_proj.temp_weight = model.self_attn.v_proj.weight.detach().clone()
            model.input_layernorm.temp_weight = model.input_layernorm.weight.detach().clone()

            model.post_attention_layernorm.temp_weight = model.post_attention_layernorm.weight.detach().clone()
            model.mlp.up_proj.temp_weight = model.mlp.up_proj.weight.detach().clone()
            model.mlp.gate_proj.temp_weight = model.mlp.gate_proj.weight.detach().clone()


        else:
            smooth_ln_fcs_temporary(model.input_layernorm,
                                    [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm, [model.mlp.up_proj, model.mlp.gate_proj],
                                    model.fc1_smooth_scale, model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj, model.self_attn.o_proj,
                                   model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                 model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight.data.to(model.self_attn.q_proj.temp_weight.device)


    else:

        model.fc2.temp_weight = model.fc2.weight.data
        model.fc2.temp_bias = model.fc2.bias.data
        if a_train_bits == 16:
            model.self_attn.q_proj.temp_weight = model.self_attn.q_proj.weight.data
            model.self_attn.q_proj.temp_bias = model.self_attn.q_proj.bias.data
            model.self_attn.k_proj.temp_weight = model.self_attn.k_proj.weight.data
            model.self_attn.k_proj.temp_bias = model.self_attn.k_proj.bias.data
            model.self_attn.v_proj.temp_weight = model.self_attn.v_proj.weight.data
            model.self_attn.v_proj.temp_bias = model.self_attn.v_proj.bias.data
            model.self_attn_layer_norm.temp_weight = model.self_attn_layer_norm.weight.data
            model.self_attn_layer_norm.temp_bias = model.self_attn_layer_norm.bias.data

            model.final_layer_norm.temp_weight = model.final_layer_norm.weight.data
            model.final_layer_norm.temp_bias = model.final_layer_norm.bias.data
            model.fc1.temp_weight = model.fc1.weight.data
            model.fc1.temp_bias = model.fc1.bias.data

            model.final_layer_norm.use_temporary_parameter = True
            model.self_attn_layer_norm.use_temporary_parameter = True


        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,
                                    [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm, [model.fc1],
                                    model.fc1_smooth_scale, model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj, model.self_attn.out_proj,
                                    model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                 model.qkt_smooth_scale)

            # smooth_ln_fcs_temporary(model.self_attn_layer_norm,
            #                         [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
            #                         model.qkv_smooth_scale, model.qkv_smooth_shift)
            # smooth_ln_fcs_temporary(model.final_layer_norm, [model.fc1],
            #                         model.fc1_smooth_scale, model.fc1_smooth_shift)
            # smooth_fc_fc_temporary(model.self_attn.v_proj, model.self_attn.out_proj,
            #                         model.out_smooth_scale, model.out_smooth_shift)
            # smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
            #                      model.qkt_smooth_scale)
        model.fc2.temp_weight = model.fc2.temp_weight.to(model.fc1.temp_weight.device)
        model.fc2.temp_bias = model.fc2.temp_bias.to(model.fc1.temp_weight.device)

    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter = True


    # quant
    # for name, module in model.named_modules():
    #     if isinstance(module, QuantLinear):
    #         if hasattr(module, "temp_weight"):
    #             module.temp_weight = module.weight_quantizer(module.temp_weight)
    #         else:
    #             module.temp_weight = module.weight_quantizer(module.weight)
    #         if not hasattr(module, "temp_bias"):
    #             module.temp_bias = module.bias
    #         module.use_temporary_parameter = True


def use_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMatMul, OmniLlamaRMSNorm, OmniLayerNorm)):
            module.use_temporary_parameter = False


def use_temp_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMatMul, OmniLlamaRMSNorm, OmniLayerNorm)):
            module.use_temporary_parameter = True


def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter = False
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias


def count(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            print(torch.isnan(module.weight).sum(), torch.isinf(module.weight).sum(),
                  torch.isnan(module.temp_weight).sum(), torch.isinf(module.temp_weight).sum())


@torch.no_grad()
def smooth_and_quant_inplace(model, isllama, a_train_bits):


    ori_v_proj_weight = model.self_attn.v_proj.weight.data.clone()
    ori_q_proj_weight = model.self_attn.q_proj.weight.data.clone()
    if not isllama:
        ori_v_proj_bias = model.self_attn.v_proj.bias.data.clone()
        ori_q_proj_bias = model.self_attn.q_proj.bias.data.clone()

    if a_train_bits < 16:
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # module.weight.data = module.weight_quantizer.quantize_weight_only(module.weight)
            module.weight.data = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False




    model.self_attn.v_proj.weight.data = ori_v_proj_weight
    model.self_attn.q_proj.weight.data = ori_q_proj_weight
    if not isllama:
        model.self_attn.v_proj.bias.data = ori_v_proj_bias
        model.self_attn.q_proj.bias.data = ori_q_proj_bias
    else:
        model.self_attn.v_proj.bias = None
        model.self_attn.q_proj.bias = None


def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul, OmniLlamaRMSNorm, OmniLayerNorm)):
            m.set_quant_state(weight_quant, act_quant)

            if not weight_quant and not act_quant:
                m.use_temporary_parameter = False
            else:
                m.use_temporary_parameter = True
