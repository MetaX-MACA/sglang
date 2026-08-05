# SPDX-License-Identifier: Apache-2.0
"""Argument contract for ``DiffGenerator.from_pretrained``."""

import pytest

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


def test_from_pretrained_rejects_server_args_combined_with_override_kwargs():
    with pytest.raises(ValueError, match=r"unexpected kwargs.*(num_frames|width).*"):
        DiffGenerator.from_pretrained(
            server_args={"model_path": "fake-model"},
            num_frames=8,
            width=512,
        )


def test_from_pretrained_rejects_non_server_args_type():
    with pytest.raises(TypeError, match="ServerArgs"):
        DiffGenerator.from_pretrained(server_args="not-server-args")
