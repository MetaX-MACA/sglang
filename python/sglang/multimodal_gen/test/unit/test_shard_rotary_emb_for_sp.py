# SPDX-License-Identifier: Apache-2.0
"""SP rotary-embedding sharding must use an explicit initialization check."""

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs import base as pipeline_base


def test_uninitialized_model_parallel_returns_identity_without_sp_lookups(
    monkeypatch,
):
    def _forbidden():
        raise AssertionError("sp lookups must not run before initialization")

    monkeypatch.setattr(pipeline_base, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(pipeline_base, "get_sp_world_size", _forbidden)
    monkeypatch.setattr(pipeline_base, "get_sp_parallel_rank", _forbidden)
    emb = torch.arange(12.0).reshape(4, 3)

    assert torch.equal(pipeline_base.shard_rotary_emb_for_sp(emb), emb)


def test_initialized_sp_lookup_errors_propagate(monkeypatch):
    monkeypatch.setattr(pipeline_base, "model_parallel_is_initialized", lambda: True)

    def _broken_lookup():
        raise RuntimeError("sp group lookup failed")

    monkeypatch.setattr(pipeline_base, "get_sp_world_size", _broken_lookup)

    with pytest.raises(RuntimeError, match="sp group lookup failed"):
        pipeline_base.shard_rotary_emb_for_sp(torch.zeros(4, 3))


def test_initialized_sp_shards_with_last_row_padding(monkeypatch):
    monkeypatch.setattr(pipeline_base, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(pipeline_base, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_base, "get_sp_parallel_rank", lambda: 1)
    emb = torch.arange(6.0).reshape(3, 2)

    sharded = pipeline_base.shard_rotary_emb_for_sp(emb)

    assert sharded.shape == (2, 2)
    assert torch.equal(sharded[0], emb[2])
    # 3 rows padded to 4 by repeating the last row.
    assert torch.equal(sharded[1], emb[2])
