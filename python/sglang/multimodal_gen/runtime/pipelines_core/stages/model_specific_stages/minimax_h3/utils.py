# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import torch


def _not_implemented(stage_name: str) -> None:
    raise NotImplementedError(
        f"{stage_name} is a MiniMax H3 contract stage and has no implementation yet."
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _required_tensor(value: Any, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must be a torch.Tensor")
    return value


def _batch_sampling_input(batch: Any, field_name: str) -> Any | None:
    if batch.sampling_params is None:
        return None
    return getattr(batch.sampling_params, field_name)


@contextlib.contextmanager
def minimax_h3_scoped_encode_rng(seed: int, device: torch.device | None = None):
    """Seed torch RNGs for a deterministic sampled VAE encode without leaking state.

    The encode recipes seed the default torch generators right
    before a posterior-sampled VAE encode. Doing that on the process-global
    generators silently reseeds every other consumer in the process, so this
    helper forks the CPU RNG (plus the encode device's CUDA RNG) and restores
    both on exit. Seeding the forked generators with the same value keeps the
    encode deterministic.
    """
    devices: list[torch.device] = []
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        devices = [device]
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(int(seed))
        for forked_device in devices:
            with torch.cuda.device(forked_device):
                torch.cuda.manual_seed(int(seed))
        yield
