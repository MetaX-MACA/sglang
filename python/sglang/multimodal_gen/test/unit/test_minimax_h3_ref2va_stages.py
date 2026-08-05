# SPDX-License-Identifier: Apache-2.0
"""Hermetic ref2va stage wiring contracts with fakes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args


class _FakeAudioVae:
    """AudioEncoding wiring fake: stage should pass itself to the encoder."""


def _video_vae_arch_config():
    return SimpleNamespace(
        latent_channels=24,
        latents_mean=[0.0] * 24,
        latents_std=[1.0] * 24,
    )


def _audio_vae_arch_config():
    return SimpleNamespace(
        latent_channels=32,
        latents_mean=[0.0] * 32,
        latents_std=[1.0] * 32,
    )


class TestMiniMaxH3Ref2vaStageWiring(unittest.TestCase):
    def setUp(self):
        self._previous = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        self.addCleanup(set_global_server_args, self._previous)

    def _canonical(self):
        return minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                {"type": "image", "uri": "file:///ref.png", "role": "reference"},
                {"type": "audio", "uri": "file:///ref.mp3", "role": "reference"},
            ],
            target={"short_edge": 768, "aspect_ratio": "16:9"},
        )

    def test_audio_stage_routes_reference_encode(self):
        from unittest import mock

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.audio_encoding import (
            MiniMaxH3AudioEncodingStage,
        )

        stage = MiniMaxH3AudioEncodingStage(
            audio_vae=_FakeAudioVae(),
            vae_arch_config=_audio_vae_arch_config(),
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: self._canonical()},
            sampling_params=None,
        )
        sentinel = {
            "rows": torch.zeros(4, 32),
            "ref_audio_t": 2,
            "duration_seconds": 0.05,
        }
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_encode_reference_audio_rows",
                return_value=sentinel,
            ) as encode,
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.material_io."
                "minimax_h3_localize_material_uri",
                return_value="/ref.mp3",
            ),
        ):
            stage.forward(batch, server_args=None)
        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[1], "/ref.mp3")
        payload = batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY]
        self.assertEqual(payload["ref_audio_t"], 2)
        self.assertEqual(len(payload["audios"]), 1)
        self.assertEqual(payload["audios"][0]["condition_index"], 1)

    def test_audio_stage_preserves_multi_audio_order(self):
        from unittest import mock

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.audio_encoding import (
            MiniMaxH3AudioEncodingStage,
        )

        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                {"type": "image", "uri": "file:///ref.png", "role": "reference"},
                {"type": "audio", "uri": "file:///a.wav", "role": "reference"},
                {"type": "audio", "uri": "file:///b.wav", "role": "reference"},
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 8.7,
            },
        )
        stage = MiniMaxH3AudioEncodingStage(
            audio_vae=_FakeAudioVae(),
            vae_arch_config=_audio_vae_arch_config(),
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
            sampling_params=None,
        )
        outputs = [
            {"rows": torch.ones(2, 32), "ref_audio_t": 1, "duration_seconds": 0.1},
            {"rows": torch.ones(4, 32), "ref_audio_t": 2, "duration_seconds": 0.2},
        ]
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_encode_reference_audio_rows",
                side_effect=outputs,
            ) as encode,
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.material_io."
                "minimax_h3_localize_material_uri",
                side_effect=["/a.wav", "/b.wav"],
            ),
        ):
            stage.forward(batch, server_args=None)

        self.assertEqual(
            [call.args[1] for call in encode.call_args_list],
            ["/a.wav", "/b.wav"],
        )
        payload = batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY]
        self.assertEqual(
            [entry["condition_index"] for entry in payload["audios"]],
            [1, 2],
        )
        self.assertEqual(sorted(payload), ["audios"])

    def test_audio_stage_keeps_silent_video_visual_only(self):
        from unittest import mock

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.audio_encoding import (
            MiniMaxH3AudioEncodingStage,
        )

        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                {"type": "video", "uri": "file:///with-audio.mp4", "role": "reference"},
                {"type": "video", "uri": "file:///silent.mp4", "role": "reference"},
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 8.7,
            },
        )
        stage = MiniMaxH3AudioEncodingStage(
            audio_vae=_FakeAudioVae(),
            vae_arch_config=_audio_vae_arch_config(),
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
            sampling_params=None,
        )
        encoded = {
            "rows": torch.ones(4, 32),
            "ref_audio_t": 2,
            "duration_seconds": 0.2,
        }
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_encode_reference_audio_rows",
                return_value=encoded,
            ) as encode,
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_reference_video_has_audio",
                side_effect=[True, False],
            ),
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.material_io."
                "minimax_h3_localize_material_uri",
                side_effect=["/with-audio.mp4", "/silent.mp4"],
            ),
        ):
            stage.forward(batch, server_args=None)

        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[1], "/with-audio.mp4")
        entries = batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY]["audios"]
        self.assertEqual([entry["condition_index"] for entry in entries], [0, 1])
        self.assertEqual(entries[0]["ref_audio_t"], 2)
        self.assertEqual(entries[1]["ref_audio_t"], 0)
        self.assertEqual(list(entries[1]["rows"].shape), [0, 32])

    def test_video_audio_routes_to_audio_and_visual_stages(self):
        from unittest import mock

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.audio_encoding import (
            MiniMaxH3AudioEncodingStage,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.visual_encoding import (
            MiniMaxH3VisualEncodingStage,
        )

        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                {"type": "video_audio", "uri": "file:///ref.mp4", "role": "reference"}
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 8.7,
            },
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
            sampling_params=None,
        )
        audio_stage = MiniMaxH3AudioEncodingStage(
            audio_vae=_FakeAudioVae(),
            vae_arch_config=_audio_vae_arch_config(),
        )
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_encode_reference_audio_rows",
                return_value={
                    "rows": torch.zeros(6, 32),
                    "ref_audio_t": 3,
                    "duration_seconds": 0.3,
                },
            ) as audio_encode,
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.material_io."
                "minimax_h3_localize_material_uri",
                return_value="/ref.mp4",
            ),
        ):
            audio_stage.forward(batch, server_args=None)

        audio_encode.assert_called_once()
        self.assertEqual(audio_encode.call_args.args[1], "/ref.mp4")
        self.assertEqual(
            batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY]["audios"][0][
                "material_chain"
            ],
            "video_audio.reference_preserve",
        )

        visual_stage = MiniMaxH3VisualEncodingStage(
            video_vae=object(),
            vae_arch_config=_video_vae_arch_config(),
        )
        prepared = {
            "prepared_path": "/prepared.mp4",
            "original_path": "/ref.mp4",
            "target_frame_count": 209,
            "condition_index": 0,
            "material_chain": "video_audio.reference_preserve",
            "videos": [
                {
                    "prepared_path": "/prepared.mp4",
                    "original_path": "/ref.mp4",
                    "target_frame_count": 209,
                    "condition_index": 0,
                    "material_chain": "video_audio.reference_preserve",
                }
            ],
        }
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_prepared_reference_videos",
                return_value=prepared,
            ),
            mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_encode_reference_video_rows",
                return_value=(torch.zeros(8, 96), 2, 4, 4),
            ) as video_encode,
        ):
            visual_stage.forward(batch, server_args=None)

        video_encode.assert_called_once()
        video_payload = batch.extra[MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY]
        self.assertEqual(video_payload["condition_index"], 0)
        self.assertEqual(
            video_payload["videos"][0]["material_chain"],
            "video_audio.reference_preserve",
        )

    def test_presentation_hermetic_labels(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_ref2va_presentation,
        )

        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [7 + (hash(w) % 100) for w in text.split()]}

            def convert_tokens_to_ids(self, token):
                return {
                    "<|vision_start|>": 1,
                    "<|vision_end|>": 2,
                    "<|image_pad|>": 3,
                    "<|video_pad|>": 5,
                    "!": 4,
                }[token]

        ids, tags = minimax_h3_ref2va_presentation(
            _Tok(),
            prompt="a b c",
            condition_labels=[("image", 1), ("audio", 1)],
            image_token_count=4,
        )
        # label(2) + vision(6) + label(2) + prompt(3)
        self.assertEqual(int(ids.shape[0]), 2 + 6 + 2 + 3)
        self.assertEqual(tags[2:8].unique().tolist(), [0])
        self.assertEqual(tags[:2].unique().tolist(), [1])
        self.assertEqual(tags[8:].unique().tolist(), [1])

    def test_riva_presentation_supports_multiple_images_and_videos(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_ref2va_video_presentation,
        )

        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [7 + (hash(w) % 100) for w in text.split()]}

            def convert_tokens_to_ids(self, token):
                return {
                    "<|vision_start|>": 1,
                    "<|vision_end|>": 2,
                    "<|image_pad|>": 3,
                    "<|video_pad|>": 5,
                    "!": 4,
                }[token]

        ids, tags = minimax_h3_ref2va_video_presentation(
            _Tok(),
            prompt="a b",
            condition_labels=[
                ("image", 1),
                ("audio", 1),
                ("video", 1),
                ("image", 2),
                ("audio", 2),
                ("video", 2),
                ("audio", 3),
            ],
            image_token_count=[2, 3],
            video_block_token_counts=[[4, 5], [6]],
            video_block_timestamps=[[0.25, 0.75], [1.25]],
        )

        self.assertEqual(int(ids.shape[0]), 52)
        self.assertEqual(int((tags == 0).sum()), 30)
        self.assertEqual(int((tags == 1).sum()), 22)


if __name__ == "__main__":
    unittest.main()
