"""SGLang boundary wrapper for DeepGEMM Mega mHC.

Credit: DeepSeek's DeepGEMM project provides the fused Mega mHC kernel.
https://github.com/deepseek-ai/DeepGEMM

https://github.com/sgl-project/sglang/pull/42808/changes#diff-542c35835d4581af72ec01e003befbe61b525b0485075b76ef5a669253de1ce2R1-R68
"""

import inspect

import torch

from sglang.srt.utils import is_maca

mega_mhc = None
_supports_int8 = False
if is_maca():
    try:
        from deep_gemm import mega_mhc as _deep_gemm_mega_mhc

        # Older MACA builds export the same API with a Torch reference backend.
        parameters = inspect.signature(_deep_gemm_mega_mhc).parameters
        backend = parameters.get("backend")
        if backend is not None and backend.default == "maca_c":
            mega_mhc = _deep_gemm_mega_mhc
            _supports_int8 = {"y_int8", "y_int8_scale"}.issubset(parameters)
    except (ImportError, TypeError, ValueError):
        pass


def mhc_mega_boundary(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    *,
    quantize=False,
    packed_size=None,
):
    """Fuse mHC post, the next mHC pre, and output RMSNorm in normal mode.

    With ``quantize``, return the existing RMSNorm input tuple for the MLP.
    ``packed_size`` selects the padded MoE dispatch layout.
    """
    if mega_mhc is None or (quantize and not _supports_int8):
        return None

    rows, hidden = x.shape
    hc = residual.shape[1]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_post = torch.empty((rows, hc, 1), device=x.device, dtype=torch.float32)
    next_comb = torch.empty_like(comb)
    quant_outputs = {}
    if quantize:
        quant = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(rows, device=x.device, dtype=torch.float32)
        quant_outputs = {"y_int8": quant, "y_int8_scale": scales}

    mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=None,
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=hc,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=None,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normalized,
        backend="maca_c",
        **quant_outputs,
    )
    layer_input = normalized
    if quantize:
        scales = scales.view(rows, 1)
        if packed_size is None:
            layer_input = (quant, scales, normalized)
        else:
            packed = torch.empty((rows, packed_size), device=x.device, dtype=x.dtype)
            packed_bytes = packed.view(torch.uint8)
            packed_bytes[:, :hidden].copy_(quant.view(torch.uint8))
            packed_bytes[:, hidden : hidden + 4].copy_(scales.view(torch.uint8))
            packed_bytes[:, hidden + 4 :].zero_()
            layer_input = (packed, normalized, (quant, scales))
    return updated, layer_input, next_post.squeeze(-1), next_comb
