import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
# from transformers.models.mistral.modeling_mistral import (
#     MistralDecoderLayer,
#     MistralRMSNorm,
# )
# from transformers.models.mixtral.modeling_mixtral import (
#     MixtralDecoderLayer,
#     MixtralRMSNorm,
# )
# from transformers.models.falcon.modeling_falcon import FalconDecoderLayer

from quantize.int_linear import QuantLinear
from models.int_opt_layer import QuantOPTDecoderLayer

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        # assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    # act_scales = torch.stack([each.to(device=device, dtype=dtype) for each in act_scales], dim=0)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # import code
    # code.interact(local=locals())

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    return scales


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    assert isinstance(ln, (LlamaRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def smooth_lm(model, scales, alpha=0.5):

    ln1 = ['self_attn_layer_norm.weight', 'self_attn_layer_norm.bias']
    attn = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight']
    ln2 = ['final_layer_norm.weight', 'final_layer_norm.bias']

    new_scales = {}
    for name, module in model.named_modules():

        if isinstance(module, OPTDecoderLayer):
            # import code
            # code.interact(local=locals())
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            scales1 = smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            for layer in ln1:
                new_scales[name + '.' + layer] = 1 / scales1
            for layer in attn:
                new_scales[name + '.' + layer] = scales1


            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            scales2 = smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            for layer in ln2:
                new_scales[name + '.' + layer] = 1 / scales2
            new_scales[name + '.fc1.weight'] = scales2

    return new_scales

# @torch.no_grad()
# def smooth_lm(model, scales, alpha=0.5):
#
#     ln1 = ['self_attn_layer_norm.weight', 'self_attn_layer_norm.bias']
#     attn = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight']
#     ln2 = ['final_layer_norm.weight', 'final_layer_norm.bias']
#
#     new_scales = {}
#     for name, module in model.named_modules():
#
#         if isinstance(module, QuantLinear):
#             attn_ln = module.self_attn_layer_norm
#             qkv = [
#                 module.self_attn.q_proj,
#                 module.self_attn.k_proj,
#                 module.self_attn.v_proj,
#             ]
#             qkv_input_scales = scales[name + ".self_attn.q_proj"]
#             scales1 = smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
#             for layer in ln1:
#                 new_scales[name + '.' + layer] = 1 / scales1
#             for layer in attn:
#                 new_scales[name + '.' + layer] = scales1
#
#
#             ffn_ln = module.final_layer_norm
#             fc1 = module.fc1
#             fc1_input_scales = scales[name + ".fc1"]
#             scales2 = smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
#             for layer in ln2:
#                 new_scales[name + '.' + layer] = 1 / scales2
#             new_scales[name + '.fc1.weight'] = scales2
#
#     return new_scales
        # elif isinstance(module, BloomBlock):
        #     attn_ln = module.input_layernorm
        #     qkv = module.self_attention.query_key_value
        #     qkv_input_scales = scales[name + ".self_attention.query_key_value"]
        #     smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
        #
        #     ffn_ln = module.post_attention_layernorm
        #     fc1 = module.mlp.dense_h_to_4h
        #     fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
        #     smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        # elif isinstance(module, (LlamaDecoderLayer)):
        #     attn_ln = module.input_layernorm  # attention forward norm
        #     qkv = [
        #         module.self_attn.q_proj,
        #         module.self_attn.k_proj,
        #         module.self_attn.v_proj,
        #     ]
        #
        #     qkv_input_scales = scales[name + ".self_attn.q_proj"]
        #     smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)
        #
        #     ffn_ln = module.post_attention_layernorm  # feed forward norm
        #     fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        #     fcs_input_scales = scales[name + ".mlp.gate_proj"]
        #
        #     smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
