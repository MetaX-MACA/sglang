"""Fused EPLB logical->physical remap + padded-row mask (方案3).

Replaces, per MoE layer per forward pass:

    physical = map[topk_ids]                     # index_elementwise_kernel
    physical = physical.to(topk_ids.dtype)       # copy_cast (int64 map) / internal
    _mask_topk_ids_padded_region(...)            # mask_topk_ids_padded_region

with a single Triton kernel.  Measured on MACA C550 (ab_trace_off/on, 43 layers):
those three kernels accounted for  index_elementwise +43/step, copy_cast +86/step,
mask_topk_ids_padded_region +40/step  =>  +189 kernel/step total.
At the measured 0.19 ms of GPU idle per extra kernel launch, this is ~24 ms/step.

Semantics copied from the reference path
----------------------------------------
* `_topk_ids_logical_to_physical_static` (expert_location_dispatch.py:105):
      out = map[topk_ids]           # map is the *per-rank* dispatch table
* `_mask_topk_ids_padded_region` (topk.py:1738) with default fill_value=-1:
      rows >= num_token_non_padded  ->  -1
* `num_token_non_padded is None`  ->  no masking at all.

The map is indexed with `topk_ids`, which may contain -1 on padded rows; the table
then aliases its last entry (that is why upstream re-masks *after* the remap).  We
clamp the index to [0, num_logical-1] so such rows can never cause an OOB read;
their result is overwritten with -1 immediately afterwards.

Enable with SGLANG_EPLB_FUSED_REMAP=1 (module-level flag, restart to change).
"""
import os

import torch
import triton
import triton.language as tl

_ENABLED = os.getenv("SGLANG_EPLB_FUSED_REMAP", "0").strip().lower() not in (
    "", "0", "false", "no", "off",
)
# One-shot kill switch: if the Triton kernel ever fails to compile/run on this
# platform we log once and fall back to the reference path for the rest of the
# process instead of crashing the server inside the forward pass.
_BROKEN = False


@triton.jit
def _eplb_remap_mask_kernel(
    topk_ids_ptr,
    map_ptr,
    out_ptr,
    n_pad_ptr,
    numel,
    stride_row,
    NUM_LOGICAL: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    row = offs // TOP_K
    col = offs % TOP_K
    # topk_ids may be a non-contiguous last-dim slice (routed_cols), so use strides.
    v = tl.load(topk_ids_ptr + row * stride_row + col, mask=m, other=0)
    # -1 (padded rows) or any garbage must not read out of bounds.
    v = tl.minimum(tl.maximum(v, 0), NUM_LOGICAL - 1)
    phys = tl.load(map_ptr + v, mask=m, other=0)
    if HAS_PAD:
        n_pad = tl.load(n_pad_ptr)
        phys = tl.where(row < n_pad, phys, -1)
    tl.store(out_ptr + offs, phys, mask=m)


def eplb_remap_mask(topk_ids: torch.Tensor, mapping: torch.Tensor,
                    num_token_non_padded=None) -> torch.Tensor:
    """Core kernel. Returns a NEW contiguous int32 tensor of the same shape."""
    T, K = topk_ids.shape
    out = torch.empty((T, K), dtype=torch.int32, device=topk_ids.device)
    numel = topk_ids.numel()
    if numel == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _eplb_remap_mask_kernel[grid](
        topk_ids,
        mapping,
        out,
        num_token_non_padded if num_token_non_padded is not None else out,
        numel,
        topk_ids.stride(0),
        NUM_LOGICAL=mapping.numel(),
        TOP_K=K,
        HAS_PAD=(num_token_non_padded is not None),
        BLOCK=BLOCK,
    )
    return out


def try_eplb_remap_mask(topk_ids, dispatch_info, num_token_non_padded):
    """Return the fused result, or None to let the reference path run."""
    global _BROKEN
    if not _ENABLED or _BROKEN:
        return None
    if dispatch_info is None:
        return None
    if dispatch_info.ep_dispatch_algorithm != "static":
        return None
    mapping = dispatch_info.partial_logical_to_rank_dispatch_physical_map
    if mapping is None:
        return None
    if not topk_ids.is_cuda or topk_ids.dtype != torch.int32:
        return None
    if topk_ids.dim() != 2 or topk_ids.stride(-1) != 1:
        return None
    if mapping.dtype != torch.int32 or mapping.device != topk_ids.device:
        return None
    if mapping.dim() != 1 or mapping.numel() == 0:
        return None
    if topk_ids.numel() == 0:
        return None
    if num_token_non_padded is not None:
        if not torch.is_tensor(num_token_non_padded):
            return None
        if num_token_non_padded.device != topk_ids.device:
            return None
    try:
        return eplb_remap_mask(topk_ids, mapping, num_token_non_padded)
    except Exception as exc:  # noqa: BLE001 - must never break the forward pass
        _BROKEN = True
        import logging

        logging.getLogger(__name__).error(
            "[eplb] fused remap kernel failed, permanently falling back to the "
            "reference path: %s: %s", type(exc).__name__, exc,
        )
        return None
