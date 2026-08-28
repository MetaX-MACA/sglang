from typing import Optional, Union

import torch
from sgl_kernel.utils import _to_tensor_scalar_tuple

try:
    import flashinfer.sampling as _flashinfer_sampling

    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False


# def _top_k_renorm_probs_internal(
#     probs: torch.Tensor,
#     maybe_top_k_arr: Optional[torch.Tensor],
#     top_k_val: int,
# ) -> torch.Tensor:
#     probs = probs.float()
#     maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
#     renorm_probs = torch.empty_like(probs)
#     torch.ops.sgl_kernel.top_k_renorm_probs.default(
#         probs, renorm_probs, maybe_top_k_arr, top_k_val
#     )
#     return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    if probs.device.type == "musa" or not _has_flashinfer:
        return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))
    return _flashinfer_sampling.top_k_renorm_probs(probs, top_k)


# top_k_renorm_prob = top_k_renorm_probs


# def _top_p_renorm_probs_internal(
#     probs: torch.Tensor,
#     maybe_top_p_arr: Optional[torch.Tensor],
#     top_p_val: float,
# ) -> torch.Tensor:
#     probs = probs.float()
#     maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
#     renorm_probs = torch.empty_like(probs)
#     torch.ops.sgl_kernel.top_p_renorm_probs.default(
#         probs, renorm_probs, maybe_top_p_arr, top_p_val
#     )
#     return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    if probs.device.type == "musa" or not _has_flashinfer:
        return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))
    return _flashinfer_sampling.top_p_renorm_probs(probs, top_p)


# top_p_renorm_prob = top_p_renorm_probs
