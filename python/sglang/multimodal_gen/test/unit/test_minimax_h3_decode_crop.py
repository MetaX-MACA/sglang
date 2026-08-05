# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for decode target-canvas cropping."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.decoding import (
    _crop_to_target_canvas,
)


def _batch(state=None):
    extra = {}
    if state is not None:
        extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY] = state
    return SimpleNamespace(extra=extra)


class TestMiniMaxH3DecodeCrop(unittest.TestCase):
    def test_crops_tile_padding_to_target(self):
        frames = torch.arange(2 * 3 * 768 * 1344, dtype=torch.float32).reshape(
            1, 2, 3, 768, 1344
        )
        out = _crop_to_target_canvas(_batch({"latent_h": 44, "latent_w": 80}), frames)
        self.assertEqual(list(out.shape), [1, 2, 3, 704, 1280])
        # top-left content preserved (padding lands bottom/right)
        self.assertTrue(torch.equal(out, frames[..., :704, :1280]))

    def test_exact_canvas_untouched(self):
        frames = torch.zeros(1, 2, 3, 768, 1216)
        out = _crop_to_target_canvas(_batch({"latent_h": 48, "latent_w": 76}), frames)
        self.assertIs(out, frames)

    def test_legacy_path_without_state_untouched(self):
        frames = torch.zeros(1, 2, 3, 768, 1344)
        self.assertIs(_crop_to_target_canvas(_batch(), frames), frames)

    def test_smaller_than_target_fails_fast(self):
        frames = torch.zeros(1, 2, 3, 640, 1216)
        with self.assertRaisesRegex(ValueError, "smaller than target"):
            _crop_to_target_canvas(_batch({"latent_h": 44, "latent_w": 80}), frames)


if __name__ == "__main__":
    unittest.main()
