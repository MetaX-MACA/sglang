# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for MiniMax H3 visual-condition noise augmentation."""

import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)


def _patchify(latent: torch.Tensor) -> torch.Tensor:
    return minimax_h3_patchify_video_latent(latent, patch_size=[1, 2, 2])


def _expected_noise(
    clean: torch.Tensor,
    *,
    target_t: int,
    cond_frames: int,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1,
        24,
        target_t + cond_frames,
        int(clean.shape[-2]),
        int(clean.shape[-1]),
        generator=generator,
        dtype=torch.float32,
    )[:, :, : int(clean.shape[2])]
    timestep = torch.tensor(noise_aug, dtype=torch.float32)
    return _patchify(timestep * clean + (1.0 - timestep) * noise)


class TestMiniMaxH3ImgvidCondNoise(unittest.TestCase):
    def test_noise_aug_one_is_exact_noop(self):
        clean = torch.arange(2 * 96, dtype=torch.float32).reshape(2, 96)
        original = clean.clone()
        rng_before = torch.random.get_rng_state()

        out = minimax_h3_imgvid_cond_noise_aug_rows(
            clean,
            condition_shapes=[(1, 2, 4)],
            target_latent_t=3,
            imgvid_cond_num_frames=1,
            seed=17,
            noise_aug=1.0,
        )

        self.assertTrue(torch.equal(out, original))
        self.assertTrue(torch.equal(clean, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_single_condition_uses_full_shape_draw(self):
        clean = torch.linspace(-1.0, 1.0, 1 * 24 * 2 * 4 * 6).reshape(1, 24, 2, 4, 6)
        clean_rows = _patchify(clean)
        rng_before = torch.random.get_rng_state()

        actual = minimax_h3_imgvid_cond_noise_aug_rows(
            clean_rows,
            condition_shapes=[(2, 4, 6)],
            target_latent_t=3,
            imgvid_cond_num_frames=1,
            seed=17,
            noise_aug=0.999,
        )
        expected = _expected_noise(
            clean,
            target_t=3,
            cond_frames=1,
            seed=17,
            noise_aug=0.999,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_multi_condition_reseeds_and_counts_visual_blocks(self):
        image = torch.linspace(-0.5, 0.5, 1 * 24 * 1 * 4 * 4).reshape(1, 24, 1, 4, 4)
        video = torch.linspace(-1.0, 1.0, 1 * 24 * 3 * 6 * 4).reshape(1, 24, 3, 6, 4)
        clean_rows = torch.cat([_patchify(image), _patchify(video)], dim=0)

        actual = minimax_h3_imgvid_cond_noise_aug_rows(
            clean_rows,
            condition_shapes=[(1, 4, 4), (3, 6, 4)],
            target_latent_t=4,
            imgvid_cond_num_frames=2,
            seed=29,
            noise_aug=0.999,
        )
        expected = torch.cat(
            [
                _expected_noise(
                    image,
                    target_t=4,
                    cond_frames=2,
                    seed=29,
                    noise_aug=0.999,
                ),
                _expected_noise(
                    video,
                    target_t=4,
                    cond_frames=2,
                    seed=29,
                    noise_aug=0.999,
                ),
            ],
            dim=0,
        )

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
