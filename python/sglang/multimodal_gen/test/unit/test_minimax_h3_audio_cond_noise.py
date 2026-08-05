# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for MiniMax H3 audio-condition noise augmentation."""

import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
)


def _expected_noise(
    clean: torch.Tensor,
    *,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    """Expected RF mix: per-element noise from a CPU generator seeded seed+1."""

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    noise = torch.randn(
        clean.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    timestep = torch.tensor(noise_aug, dtype=torch.float32, device="cpu")
    return timestep * clean.float().cpu() + (1.0 - timestep) * noise


class TestMiniMaxH3AudioCondNoise(unittest.TestCase):
    def test_noise_aug_one_is_exact_noop(self):
        clean = torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32)
        original = clean.clone()
        rng_before = torch.random.get_rng_state()

        out = minimax_h3_audio_cond_noise_aug_rows(
            clean,
            condition_audio_t=[2],
            seed=7,
            noise_aug=1.0,
        )

        self.assertTrue(torch.equal(out, original))
        self.assertTrue(torch.equal(clean, original))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_non_negative_seed_uses_seed_plus_one_cpu_fp32(self):
        clean = torch.linspace(-1.0, 1.0, 6 * 32, dtype=torch.float32).reshape(6, 32)
        rng_before = torch.random.get_rng_state()

        actual = minimax_h3_audio_cond_noise_aug_rows(
            clean,
            condition_audio_t=[3],
            seed=7,
            noise_aug=0.4,
        )
        expected = _expected_noise(clean, seed=7, noise_aug=0.4)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_ordered_multi_condition_reseeds_each_audio_block(self):
        first = torch.linspace(-0.5, 0.5, 4 * 32, dtype=torch.float32).reshape(4, 32)
        second = torch.linspace(1.0, 2.0, 6 * 32, dtype=torch.float32).reshape(6, 32)
        clean = torch.cat([first, second], dim=0)

        actual = minimax_h3_audio_cond_noise_aug_rows(
            clean,
            condition_audio_t=[2, 3],
            seed=29,
            noise_aug=0.75,
        )
        expected = torch.cat(
            [
                _expected_noise(first, seed=29, noise_aug=0.75),
                _expected_noise(second, seed=29, noise_aug=0.75),
            ],
            dim=0,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_rows_that_do_not_match_per_condition_lengths(self):
        with self.assertRaisesRegex(ValueError, "shape-derived rows"):
            minimax_h3_audio_cond_noise_aug_rows(
                torch.zeros(5, 32),
                condition_audio_t=[2],
                seed=1,
                noise_aug=0.5,
            )


if __name__ == "__main__":
    unittest.main()
