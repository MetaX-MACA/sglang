# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel object broadcast helpers for encoder extras."""

from __future__ import annotations

from typing import Any

import torch


def minimax_h3_sp_ctx() -> tuple[int, int]:
    """Return the sequence-parallel ``(world_size, rank)`` or ``(1, 0)``."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return 1, 0
    group = get_sp_group()
    return int(group.world_size), int(group.rank_in_group)


def minimax_h3_sp_broadcast_extra(batch: Any, key: str) -> None:
    """Broadcast ``batch.extra[key]`` from the sequence-parallel main rank."""
    world, rank = minimax_h3_sp_ctx()
    if world <= 1:
        return
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    group = get_sp_group()
    payload = [batch.extra.get(key)] if rank == 0 else [None]
    torch.distributed.broadcast_object_list(
        payload, src=group.ranks[0], group=group.cpu_group
    )
    if rank != 0:
        if payload[0] is None:
            raise RuntimeError(f"sp broadcast of batch.extra[{key!r}] got None")
        batch.extra[key] = payload[0]


__all__ = ["minimax_h3_sp_broadcast_extra", "minimax_h3_sp_ctx"]
