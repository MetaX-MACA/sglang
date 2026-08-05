# SPDX-License-Identifier: Apache-2.0
"""Hermetic contract tests for the direct loop stage paths
(timestep sigma generation, latent-prep seed noise, denoising full-loop
wiring with a fake model). Numerical loop fidelity is covered by the GPU
golden in test_minimax_h3_denoise_loop.py."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args


def _t2va_canonical(seed=42):
    return minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="p",
        conditions=None,
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        seed=seed,
    )


def _batch(canonical, extra=None, sampling_params=None):
    full_extra = {MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical}
    full_extra.update(extra or {})
    return SimpleNamespace(
        extra=full_extra,
        sampling_params=sampling_params,
        latents=None,
        audio_latents=None,
        timestep=None,
        timesteps=None,
    )


class _GlobalServerArgsMixin(unittest.TestCase):
    def setUp(self):
        self._previous_server_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        self.addCleanup(set_global_server_args, self._previous_server_args)


class TestMiniMaxH3TimestepDirect(_GlobalServerArgsMixin):
    def test_sigma_generation_from_profile(self):
        stage = MiniMaxH3TimestepPreparationStage()
        batch = _batch(_t2va_canonical())
        stage.forward(batch, server_args=None)
        sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
        self.assertEqual(sorted(sigmas), ["audio", "video"])
        for modality, shift in (("video", 12.0), ("audio", 3.0)):
            values = sigmas[modality]
            self.assertEqual(len(values), 50)
            self.assertEqual(values[0], 1.0)
            self.assertEqual(values[-1], 0.0)
            self.assertEqual(
                values,
                minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=shift),
                modality,
            )


class TestMiniMaxH3LatentPrepDirect(_GlobalServerArgsMixin):
    def test_t2va_seed_noise_recipe(self):
        stage = MiniMaxH3LatentPreparationStage()
        batch = _batch(_t2va_canonical(seed=42))
        stage.forward(batch, server_args=None)
        state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
        # adaptive v2 16:9 geometry: 768x1344 -> latent 48x84 -> token grid 24x42
        self.assertEqual(state["latent_t"], 37)
        self.assertEqual(state["latent_h"], 48)
        self.assertEqual(state["latent_w"], 84)
        n_video = 37 * 24 * 42
        n_audio = state["audio_t"] * 2
        self.assertEqual(list(state["initial_video_rows"].shape), [n_video, 96])
        self.assertEqual(list(state["initial_audio_rows"].shape), [n_audio, 32])
        # Noise-seeding contract: video is drawn on the raw latent tensor then
        # patchified;
        # audio from an independent generator re-seeded with the same seed.
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
            minimax_h3_patchify_video_latent,
        )

        gen_v = torch.Generator().manual_seed(42)
        want_video = minimax_h3_patchify_video_latent(
            torch.randn(1, 24, 37, 48, 84, generator=gen_v, dtype=torch.float32),
            patch_size=[1, 2, 2],
        ).to(torch.float32)
        gen_a = torch.Generator().manual_seed(42)
        want_audio = torch.randn(n_audio, 32, generator=gen_a, dtype=torch.float32)
        self.assertTrue(torch.equal(state["initial_video_rows"], want_video))
        self.assertTrue(torch.equal(state["initial_audio_rows"], want_audio))

    def test_t2va_auto_geometry_needs_no_keyframe_rows(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=None,
            target={
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            seed=42,
        )

        batch = _batch(canonical)
        MiniMaxH3LatentPreparationStage().forward(batch, server_args=None)

        state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
        self.assertEqual((state["latent_h"], state["latent_w"]), (48, 84))

    def test_non_negative_seed_keeps_deterministic_manual_seed_recipe(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
            minimax_h3_patchify_video_latent,
        )

        stage = MiniMaxH3LatentPreparationStage()
        batch = _batch(_t2va_canonical(seed=7))
        stage.forward(batch, server_args=None)
        state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]

        gen_v = torch.Generator().manual_seed(7)
        want_video = minimax_h3_patchify_video_latent(
            torch.randn(
                1,
                24,
                state["latent_t"],
                state["latent_h"],
                state["latent_w"],
                generator=gen_v,
                dtype=torch.float32,
            ),
            patch_size=[1, 2, 2],
        ).to(torch.float32)
        gen_a = torch.Generator().manual_seed(7)
        want_audio = torch.randn(
            state["audio_t"] * 2,
            32,
            generator=gen_a,
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(state["initial_video_rows"], want_video))
        self.assertTrue(torch.equal(state["initial_audio_rows"], want_audio))

    def test_fl2va_uses_final_resolved_geometry(self):
        stage = MiniMaxH3LatentPreparationStage()
        canonical = minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="p",
            conditions=[
                {
                    "type": "image",
                    "uri": "file:///k.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "19:12",
                "duration_seconds": 5.1,
            },
            seed=7,
        )
        batch = _batch(
            canonical,
            extra={
                MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY: {
                    "rows": torch.zeros(2 * 912, 96),
                    "latent_h": 48,
                    "latent_w": 76,
                    "semantic_frame_indices": [0, -1],
                    "pixel_frame_indices": [0, 123],
                    "frame_count": 124,
                }
            },
        )
        stage.forward(batch, server_args=None)
        state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
        self.assertEqual(state["latent_h"], 48)
        self.assertEqual(state["latent_w"], 76)
        self.assertEqual(
            list(state["initial_video_rows"].shape),
            [state["latent_t"] * 24 * 38, 96],
        )

    def test_fl2va_deferred_without_keyframe_rows_fails_fast(self):
        stage = MiniMaxH3LatentPreparationStage()
        canonical = minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="p",
            conditions=[
                {
                    "type": "image",
                    "uri": "file:///k.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.1,
            },
        )
        with self.assertRaisesRegex(ValueError, "pre-queue resolved_v2 geometry"):
            stage.forward(_batch(canonical), server_args=None)


class _FakeDiT(torch.nn.Module):
    """Returns zero velocities with per-position shapes (loop wiring only)."""

    def forward(self, **kwargs):
        img_pos = kwargs["img_pos_info"]["position_ids"]
        audio_pos = kwargs["audio_pos_info"]["position_ids"]
        return (
            torch.zeros(int(img_pos.shape[0]), 96, device=img_pos.device),
            torch.zeros(int(audio_pos.shape[0]), 32, device=audio_pos.device),
        )


class _ToRecordingDiT(torch.nn.Module):
    """Records .to(...) invocations so residency behavior can be asserted."""

    def __init__(self):
        super().__init__()
        self.to_calls: list = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class FSDPToRecordingDiT(_ToRecordingDiT):
    """Class name mimics FSDP2's fully_shard class swap (FSDP<Cls>)."""


class TestResolveDenoiseModelResidency(unittest.TestCase):
    def test_plain_module_is_moved_to_device(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoising import (
            _resolve_denoise_model,
        )

        module = _ToRecordingDiT()
        device = torch.device("cpu")
        resolved = _resolve_denoise_model(module, device)
        self.assertIs(resolved, module)
        self.assertEqual(module.to_calls, [((device,), {})])
        self.assertFalse(resolved.training)

    def test_fsdp_module_is_never_moved(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoising import (
            _resolve_denoise_model,
        )

        module = FSDPToRecordingDiT()
        resolved = _resolve_denoise_model(module, torch.device("cpu"))
        self.assertIs(resolved, module)
        self.assertEqual(module.to_calls, [])
        self.assertFalse(resolved.training)

    def test_wrapper_model_attribute_is_unwrapped(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoising import (
            _resolve_denoise_model,
        )

        inner = FSDPToRecordingDiT()
        wrapper = SimpleNamespace(model=inner)
        resolved = _resolve_denoise_model(wrapper, torch.device("cpu"))
        self.assertIs(resolved, inner)
        self.assertEqual(inner.to_calls, [])


@unittest.skipUnless(torch.cuda.is_available(), "full-loop wiring needs CUDA")
class TestMiniMaxH3DenoisingFullLoop(_GlobalServerArgsMixin):
    def test_t2va_full_loop_wiring(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoising import (
            MiniMaxH3DenoisingStage,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.timestep_preparation import (
            MiniMaxH3TimestepPreparationStage,
        )

        canonical = _t2va_canonical()
        embeddings = {
            "positive": {
                "hidden_states": torch.zeros(5, 5120, dtype=torch.bfloat16),
                "text_len": 5,
                "text_token_tags": torch.ones(5, dtype=torch.long),
            }
        }
        batch = _batch(
            canonical,
            extra={MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: embeddings},
            sampling_params=SimpleNamespace(),
        )
        MiniMaxH3LatentPreparationStage().forward(batch, server_args=None)
        MiniMaxH3TimestepPreparationStage().forward(batch, server_args=None)
        # shrink to 2 steps for wiring speed
        sig = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
        sig["video"] = sig["video"][:3]
        sig["audio"] = sig["audio"][:3]
        stage = MiniMaxH3DenoisingStage(transformer=_FakeDiT())
        stage.forward(batch, server_args=None)
        state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
        self.assertEqual(
            list(batch.latents.shape),
            [1, 24, state["latent_t"], state["latent_h"], state["latent_w"]],
        )
        self.assertEqual(list(batch.audio_latents.shape), [2, 32, state["audio_t"]])


if __name__ == "__main__":
    unittest.main()
