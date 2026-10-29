"""The index logits of the DeepSeek V4.1 ratio-1/2 index layers, by the layout of
the index K they read: the flattened K of a prefill chunk (tile by tile under a
memory budget), the paged pool, or the published blocks of a sparse table."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch

from sglang.srt.layers.attention.dsv4.index_cache_kind import dsv4_index_cache_kind
from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align
from sglang.srt.utils.common import is_maca

from .candidate_table import CANDIDATE_BLOCK_SIZE


def _align_score_rows(logits: torch.Tensor, align: int) -> torch.Tensor:
    """Pad the row stride of ``logits`` to a multiple of ``align`` elements.

    The fused top-k kernels load score rows with 16-byte vectorized reads
    and reject a row stride that is not a multiple of 4 fp32 elements. The
    tilelang int8 logits backends size the output by the tile's max window
    (or ``max_seq_len``) instead of the caller's aligned width, so the
    stride is not guaranteed to be aligned: copy only when it is not.
    """
    stride = logits.stride(0)
    if stride % align == 0:
        return logits
    padded = logits.new_empty((logits.shape[0], ceil_align(stride, align)))
    padded[:, : logits.shape[1]].copy_(logits)
    return padded


def flat_index_logits_rows_per_tile(
    rows: int, width: int, *, heads: int, budget_bytes: int
) -> int:
    """Rows per fp32 logits tile within ``budget_bytes``, at the kernel's row
    alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=budget_bytes,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)


def flat_index_logits_tiles(
    *,
    q: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    context_lengths: list[int],
    budget_bytes: int,
    width_align: int = 4,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """``(rows, logits)`` per row tile, each fp32 tile within ``budget_bytes``:
    ``logits[i, j]`` scores query row ``rows.start + i`` against ``kv[starts + j]``,
    garbage past the row's ``lengths``; the width is ``max(context_lengths)``
    aligned to ``width_align``. ``q`` is the ``(q_fp4, q_sf)`` tuple of the fp4
    cache kind, the int8 tensor of the int8 kind."""


    kind = dsv4_index_cache_kind()
    if kind.is_fp4:
        from deep_gemm import fp8_fp4_mqa_logits
        rows, heads = q[0].shape[0], q[0].shape[1]
    else:
        from deep_gemm import int8_mqa_logits
        rows, heads = q.shape[0], q.shape[1]
    width = ceil_align(max(context_lengths, default=0), width_align)
    if kind.is_int8 and width % 4:
        width = ceil_align(width, 4)
    if rows == 0 or width == 0:
        return
    rows_per_chunk = flat_index_logits_rows_per_tile(
        rows, width, heads=heads, budget_bytes=budget_bytes
    )
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        tile_starts = starts[tile]
        if kind.is_fp4:
            yield (
                tile,
                fp8_fp4_mqa_logits(
                    (q[0][tile], q[1][tile]),
                    kv,
                    weights[tile],
                    tile_starts,
                    tile_starts + lengths[tile],
                    False,
                    width,
                ),
            )
            continue
        elif kind.is_int8 and is_maca():
            logits = int8_mqa_logits(
                q[tile],
                kv,
                weights[tile],
                tile_starts,
                tile_starts + lengths[tile],
                False,
                backend="maca_c",
            )
            yield tile, _align_score_rows(logits, width_align)


def deep_gemm_fp4_paged_mqa_logits(
    q: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
) -> torch.Tensor:
    """DeepGEMM paged logits; no hadamard, the reference does not apply one. ``q``
    is the ``(q_fp4, q_sf)`` tuple of the fp4 cache kind, the int8 tensor
    of the int8 kind."""
    from deep_gemm import fp8_fp4_paged_mqa_logits
    from deep_gemm import int8_paged_mqa_logits

    sl = seq_lens.to(torch.int32)
    if sl.dim() == 1:
        sl = sl.unsqueeze(-1)
    kind = dsv4_index_cache_kind()
    if kind.is_fp4:
        return fp8_fp4_paged_mqa_logits(
            q,
            k_cache,
            weights,
            sl,
            page_table,
            deep_gemm_metadata,
            max_seq_len,
            False,
        )
    elif kind.is_int8 and is_maca():
        return int8_paged_mqa_logits(
            q,
            k_cache,
            weights,
            sl,
            page_table,
            deep_gemm_metadata,
            max_seq_len,
            False,
            backend="maca_c",
        )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    schedule: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16, ``schedule`` the
    ``build_sparse_indexer_schedule`` of the ``topk_blocks`` published blocks."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        schedule,
        topk_blocks,
        CANDIDATE_BLOCK_SIZE,
    )
