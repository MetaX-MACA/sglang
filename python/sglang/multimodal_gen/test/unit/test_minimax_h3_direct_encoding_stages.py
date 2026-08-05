# SPDX-License-Identifier: Apache-2.0
"""Hermetic contract tests for the direct-encoding stage paths.

Fake encoder/tokenizer/vae verify wiring only (presentation build, extra-key
population, residency sequencing, fail-fast branches); the numerical recipes
are covered by the GPU golden tests (test_minimax_h3_qwen3vl_encoder,
test_minimax_h3_keyframe_encoding).
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
    minimax_h3_prepare_for_queue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.text_encoding import (
    MiniMaxH3TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.visual_encoding import (
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLHFEncoder,
    _retain_selected_lm_layer,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args


def _video_vae_arch_config():
    return SimpleNamespace(
        latent_channels=24,
        latents_mean=[0.0] * 24,
        latents_std=[1.0] * 24,
    )


class TestMiniMaxH3Qwen3VLEncoder(unittest.TestCase):
    def test_retain_selected_layer_matches_pre_norm_hidden_state_contract(self):
        language_model = SimpleNamespace(
            layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(64)]),
            norm=torch.nn.LayerNorm(1),
            config=SimpleNamespace(num_hidden_layers=64),
        )
        model = SimpleNamespace(language_model=language_model)

        _retain_selected_lm_layer(model, MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)

        self.assertEqual(len(language_model.layers), 50)
        self.assertIsInstance(language_model.norm, torch.nn.Identity)
        self.assertEqual(language_model.config.num_hidden_layers, 50)

    def test_encode_uses_backbone_output_without_hidden_history_or_logits(self):
        class _Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.kwargs = None

            def forward(self, **kwargs):
                self.kwargs = kwargs
                seq_len = int(kwargs["input_ids"].shape[1])
                return SimpleNamespace(
                    last_hidden_state=torch.zeros(
                        1, seq_len, 5120, dtype=torch.bfloat16
                    )
                )

        encoder = MiniMaxH3Qwen3VLHFEncoder.__new__(MiniMaxH3Qwen3VLHFEncoder)
        encoder.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        encoder.hidden_dim = 5120
        encoder.device = torch.device("cpu")
        encoder.image_token_id = 151655
        encoder.video_token_id = 151656
        encoder.model = _Backbone()

        hidden = encoder.encode_ids(torch.tensor([1, 2, 3], dtype=torch.long))

        self.assertEqual(list(hidden.shape), [3, 5120])
        self.assertFalse(encoder.model.kwargs["output_hidden_states"])
        self.assertTrue(encoder.model.kwargs["return_dict"])
        self.assertFalse(encoder.model.kwargs["use_cache"])


class _FakeTokenizer:
    """Whitespace tokenizer with a deterministic vocabulary."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.calls.append(text)
        pieces = text.split()
        if not pieces and text:
            pieces = [text]
        return {"input_ids": [7 + (hash(w) % 1000) for w in pieces]}

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": 1,
            "<|vision_end|>": 2,
            "<|image_pad|>": 3,
            "<|video_pad|>": 5,
            "!": 4,
        }[token]


class _FakeQwenEncoder:
    def __init__(self):
        self.calls = []
        self.residency = []
        self.on_device = True
        self.hf_model_path = "/fake/qwen3vl"

    def load_to_device(self):
        self.on_device = True
        self.residency.append("load")

    def offload_to_cpu(self):
        self.on_device = False
        self.residency.append("offload")

    def encode_ids(
        self,
        input_ids,
        *,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        assert self.on_device, "encode while offloaded"
        self.calls.append(
            {
                "ids": input_ids,
                "has_image": pixel_values is not None,
                "has_video": pixel_values_videos is not None,
                "image_grid_shape": (
                    None if image_grid_thw is None else list(image_grid_thw.shape)
                ),
                "video_grid_shape": (
                    None if video_grid_thw is None else list(video_grid_thw.shape)
                ),
            }
        )
        return torch.zeros(int(input_ids.shape[0]), 5120, dtype=torch.bfloat16)


class _FakeImageProcessor:
    merge_size = 1

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros(len(images), 3, 2, 2),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2] for _ in images], dtype=torch.long
            ),
        }


class _FakeProcessor:
    def __init__(self):
        self.image_processor = _FakeImageProcessor()

    def video_processor(self, *, videos, do_sample_frames, return_tensors):
        assert do_sample_frames is False
        assert return_tensors == "pt"
        return {
            "pixel_values_videos": torch.zeros(len(videos), 3, 2, 2),
            "video_grid_thw": torch.tensor(
                [[2, 2, 2] for _ in videos], dtype=torch.long
            ),
        }


def _batch(canonical, extra=None):
    full_extra = {MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical}
    full_extra.update(extra or {})
    return SimpleNamespace(extra=full_extra, sampling_params=None)


def _prequeue_batch(canonical):
    batch = _batch(canonical)
    minimax_h3_prepare_for_queue(batch)
    return batch


class _GlobalServerArgsMixin(unittest.TestCase):
    """unittest counterpart of the pytest autouse fixture in conftest.py."""

    def setUp(self):
        self._previous_server_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        self.addCleanup(set_global_server_args, self._previous_server_args)


class TestMiniMaxH3TextEncodingDirect(_GlobalServerArgsMixin):
    def test_text_sink_rejects_reversed_keyframes_even_when_cached(self):
        plan = SimpleNamespace(
            task="fl2va",
            prompt="p",
            materials=[
                SimpleNamespace(material_chain="image.target_canvas", frame_index=index)
                for index in (-1, 0)
            ],
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: {"positive": {}}}
        )
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=None,
            tokenizer=None,
            processor=_FakeProcessor(),
        )

        with self.assertRaisesRegex(ValueError, "ordered keyframe signature"):
            stage._encode_from_plan(batch, plan)

    def test_t2va_direct_encode_populates_embeddings(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="a warthog kneels",
            conditions=None,
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )
        encoder = _FakeQwenEncoder()
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=encoder, tokenizer=_FakeTokenizer(), processor=_FakeProcessor()
        )
        batch = _batch(canonical)
        stage.forward(batch, server_args=None)
        emb = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]
        self.assertEqual(list(emb), ["positive"])
        self.assertEqual(emb["positive"]["text_len"], len("a warthog kneels".split()))
        hidden = emb["positive"]["hidden_states"]
        self.assertEqual(list(hidden.shape), [emb["positive"]["text_len"], 5120])
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertEqual(len(encoder.calls), 1)
        self.assertFalse(any(c["has_image"] for c in encoder.calls))
        # residency: loaded before encode, offloaded after
        self.assertEqual(encoder.residency, ["load", "offload"])
        self.assertFalse(encoder.on_device)

    def test_direct_encode_requires_encode_ids(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=None,
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=object(),
            tokenizer=_FakeTokenizer(),
            processor=_FakeProcessor(),
        )
        with self.assertRaisesRegex(TypeError, "encode_ids"):
            stage.forward(_batch(canonical), server_args=None)

    def test_idempotent_when_embeddings_present(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=None,
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )
        encoder = _FakeQwenEncoder()
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=encoder, tokenizer=_FakeTokenizer(), processor=_FakeProcessor()
        )
        sentinel = {"positive": {}}
        batch = _batch(canonical, extra={MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY: sentinel})
        stage.forward(batch, server_args=None)
        self.assertIs(batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY], sentinel)
        self.assertEqual(encoder.calls, [])

    def test_fl2va_text_encode_uses_first_last_positive_qwen_vision(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            last = Path(tmp) / "last.png"
            Image.new("RGB", (1216, 768)).save(str(first))
            Image.new("RGB", (1216, 768)).save(str(last))
            canonical = minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="animate the frame",
                conditions=[
                    {
                        "type": "image",
                        "uri": f"file://{first}",
                        "role": "keyframe",
                        "frame_index": 0,
                    },
                    {
                        "type": "image",
                        "uri": f"file://{last}",
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
            encoder = _FakeQwenEncoder()
            tokenizer = _FakeTokenizer()
            stage = MiniMaxH3TextEncodingStage(
                text_encoder=encoder, tokenizer=tokenizer, processor=_FakeProcessor()
            )
            batch = _prequeue_batch(canonical)
            stage.forward(batch, server_args=None)

        emb = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]
        self.assertEqual(list(emb), ["positive"])
        self.assertEqual(len(encoder.calls), 1)
        image_call = encoder.calls[0]
        self.assertTrue(image_call["has_image"])
        self.assertEqual(image_call["image_grid_shape"], [2, 3])
        self.assertFalse(
            any(
                call.startswith("Use the provided keyframe pictures")
                for call in tokenizer.calls
            )
        )

    def test_i2va_and_l2va_text_encode_use_one_anchor_image(self):
        from PIL import Image

        for semantic_index in (0, -1):
            with (
                self.subTest(semantic_index=semantic_index),
                tempfile.TemporaryDirectory() as tmp,
            ):
                image_path = Path(tmp) / "anchor.png"
                Image.new("RGB", (1216, 768)).save(image_path)
                canonical = minimax_h3_validate_canonical_request(
                    task="fl2va",
                    prompt="animate the frame",
                    conditions=[
                        {
                            "type": "image",
                            "uri": f"file://{image_path}",
                            "role": "keyframe",
                            "frame_index": semantic_index,
                        }
                    ],
                    target={
                        "short_edge": 768,
                        "aspect_ratio": "auto",
                        "duration_seconds": 5.1,
                    },
                )
                encoder = _FakeQwenEncoder()
                tokenizer = _FakeTokenizer()
                stage = MiniMaxH3TextEncodingStage(
                    text_encoder=encoder,
                    tokenizer=tokenizer,
                    processor=_FakeProcessor(),
                )
                batch = _prequeue_batch(canonical)
                stage.forward(batch, server_args=None)

                self.assertEqual(len(encoder.calls), 1)
                self.assertEqual(encoder.calls[0]["image_grid_shape"], [1, 3])
                self.assertFalse(
                    any(
                        call.startswith("Use the provided keyframe pictures")
                        for call in tokenizer.calls
                    )
                )
                self.assertIn("<Picture 1>: ", tokenizer.calls)
                self.assertIn("animate the frame", tokenizer.calls)

    def test_ref2va_text_encode_handles_multi_image_video_mix(self):
        import numpy as np

        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="keep motion",
            conditions=[
                {"type": "image", "uri": "file:///a.png", "role": "reference"},
                {"type": "video", "uri": "file:///v0.mp4", "role": "reference"},
                {"type": "audio", "uri": "file:///a.wav", "role": "reference"},
                {"type": "video_audio", "uri": "file:///v1.mp4", "role": "reference"},
                {"type": "video", "uri": "file:///v2.mp4", "role": "reference"},
                {"type": "image", "uri": "file:///b.png", "role": "reference"},
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 8.7,
            },
        )
        encoder = _FakeQwenEncoder()
        tokenizer = _FakeTokenizer()
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=encoder, tokenizer=tokenizer, processor=_FakeProcessor()
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_prepared_reference_image",
                return_value={"images": [{"image": object()}, {"image": object()}]},
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_prepared_reference_videos",
                return_value={
                    "videos": [
                        {
                            "prepared_path": "/v0.mp4",
                            "condition_index": 1,
                            "input_has_audio": True,
                        },
                        {
                            "prepared_path": "/v1.mp4",
                            "condition_index": 3,
                            "input_has_audio": False,
                        },
                        {
                            "prepared_path": "/v2.mp4",
                            "condition_index": 4,
                            "input_has_audio": False,
                        },
                    ]
                },
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.minimax_h3.reference_encoding."
                "minimax_h3_sample_reference_video_frames",
                side_effect=[
                    {"frames": [frame], "block_timestamps": [0.25, 0.75]},
                    {"frames": [frame], "block_timestamps": [1.25, 1.75]},
                    {"frames": [frame], "block_timestamps": [2.25, 2.75]},
                ],
            ),
        ):
            batch = _batch(canonical)
            stage.forward(batch, server_args=None)

        embeddings = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]
        self.assertEqual(list(embeddings), ["positive"])
        self.assertEqual(len(encoder.calls), 1)
        positive = encoder.calls[0]
        self.assertEqual(positive["image_grid_shape"], [2, 3])
        self.assertEqual(positive["video_grid_shape"], [3, 3])
        self.assertTrue(positive["has_image"])
        self.assertIn("<Audio 1>: ", tokenizer.calls)
        self.assertIn("<Audio 2>: ", tokenizer.calls)
        self.assertIn("<Audio 3>: ", tokenizer.calls)
        self.assertNotIn("<Audio 4>: ", tokenizer.calls)
        self.assertIn("<Video 1>: ", tokenizer.calls)
        self.assertIn("<Video 2>: ", tokenizer.calls)
        self.assertIn("<Video 3>: ", tokenizer.calls)

    def _ref2va_video_canonical(self, uris):
        return minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="keep motion",
            conditions=[
                {"type": "video", "uri": uri, "role": "reference"} for uri in uris
            ],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 8.7,
            },
        )

    def test_ref2va_text_encode_requires_input_has_audio_probe(self):
        canonical = self._ref2va_video_canonical(["file:///v0.mp4"])
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=_FakeQwenEncoder(),
            tokenizer=_FakeTokenizer(),
            processor=_FakeProcessor(),
        )
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.reference_encoding."
            "minimax_h3_prepared_reference_videos",
            return_value={
                "videos": [{"prepared_path": "/v0.mp4", "condition_index": 0}]
            },
        ):
            with self.assertRaisesRegex(
                ValueError, r"video 0 is missing 'input_has_audio'"
            ):
                stage.forward(_batch(canonical), server_args=None)

    def test_ref2va_text_encode_requires_probe_for_every_plain_video(self):
        canonical = self._ref2va_video_canonical(["file:///v0.mp4", "file:///v1.mp4"])
        stage = MiniMaxH3TextEncodingStage(
            text_encoder=_FakeQwenEncoder(),
            tokenizer=_FakeTokenizer(),
            processor=_FakeProcessor(),
        )
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.reference_encoding."
            "minimax_h3_prepared_reference_videos",
            return_value={
                "videos": [
                    {
                        "prepared_path": "/v0.mp4",
                        "condition_index": 0,
                        "input_has_audio": True,
                    }
                ]
            },
        ):
            with self.assertRaisesRegex(KeyError, r"condition 1"):
                stage.forward(_batch(canonical), server_args=None)


class _FakeVAEModel:
    def __init__(self):
        self.parallel_tiling = True
        self.tiling_during_encode = None
        self.encode_calls = 0

    def encode_images(self, image, use_fp16_latent):
        assert use_fp16_latent is True
        self.encode_calls += 1
        self.tiling_during_encode = self.parallel_tiling
        w, h = image.size
        return torch.arange(24 * (h // 16) * (w // 16), dtype=torch.float32).reshape(
            1, 24, 1, h // 16, w // 16
        )[None]


class _FakeVAE(torch.nn.Module):
    """nn.Module wrapper: keyframe encode checks/restores parameter dtype."""

    def __init__(self):
        super().__init__()
        self.model = _FakeVAEModel()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        )


class TestMiniMaxH3VisualEncodingDirect(_GlobalServerArgsMixin):
    def test_visual_sink_rejects_middle_keyframe_even_when_cached(self):
        plan = SimpleNamespace(
            task="fl2va",
            materials=[
                SimpleNamespace(
                    material_chain="image.target_canvas",
                    frame_index=index,
                    condition_index=condition_index,
                )
                for condition_index, index in enumerate((0, 12))
            ],
        )
        batch = SimpleNamespace(
            extra={MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY: {"rows": object()}}
        )
        stage = MiniMaxH3VisualEncodingStage(
            video_vae=None,
            vae_arch_config=_video_vae_arch_config(),
        )

        with self.assertRaisesRegex(ValueError, "ordered keyframe signature"):
            stage._encode_keyframes_from_plan(batch, plan, [0, 1])

    def _canonical_fl2va(self, png_path, frame_indices=(0, -1)):
        return minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="p",
            conditions=[
                {
                    "type": "image",
                    "uri": f"file://{png_path}",
                    "role": "keyframe",
                    "frame_index": frame_index,
                }
                for frame_index in frame_indices
            ],
            # short_edge matches the 768-high test canvas so auto-canvas
            # resolution is the identity (canvas prep covered in
            # test_minimax_h3_canvas.py)
            target={
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.1,
            },
        )

    def test_fl2va_direct_keyframe_encode(self):
        from PIL import Image

        vae = _FakeVAE()
        stage = MiniMaxH3VisualEncodingStage(
            video_vae=vae,
            vae_arch_config=_video_vae_arch_config(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "k.png"
            Image.new("RGB", (1216, 768)).save(str(png))
            batch = _prequeue_batch(self._canonical_fl2va(png))
            stage.forward(batch, server_args=None)
        payload = batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY]
        rows = payload["rows"]
        self.assertEqual(list(rows.shape), [2 * 912, 96])
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(payload["latent_h"], 48)
        self.assertEqual(payload["latent_w"], 76)
        # parallel tiling scoped off during encode, restored afterwards
        self.assertFalse(vae.model.tiling_during_encode)
        self.assertTrue(vae.model.parallel_tiling)
        # fp16 weights upcast to fp32 for encoding, then restored
        self.assertEqual(next(vae.parameters()).dtype, torch.float16)

    def test_i2va_and_l2va_direct_keyframe_encode_single_anchor(self):
        from PIL import Image

        for semantic_index, pixel_index in ((0, 0), (-1, 123)):
            with (
                self.subTest(semantic_index=semantic_index),
                tempfile.TemporaryDirectory() as tmp,
            ):
                vae = _FakeVAE()
                stage = MiniMaxH3VisualEncodingStage(
                    video_vae=vae,
                    vae_arch_config=_video_vae_arch_config(),
                )
                png = Path(tmp) / "anchor.png"
                Image.new("RGB", (1216, 768)).save(png)
                batch = _prequeue_batch(self._canonical_fl2va(png, (semantic_index,)))
                stage.forward(batch, server_args=None)

            payload = batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY]
            self.assertEqual(list(payload["rows"].shape), [912, 96])
            self.assertEqual(vae.model.encode_calls, 1)
            self.assertEqual(payload["semantic_frame_indices"], [semantic_index])
            self.assertEqual(payload["pixel_frame_indices"], [pixel_index])
            self.assertEqual(len(payload["keyframes"]), 1)

    def test_fl2va_direct_keyframe_encode_preserves_first_last_order(self):
        from PIL import Image

        vae = _FakeVAE()
        stage = MiniMaxH3VisualEncodingStage(
            video_vae=vae,
            vae_arch_config=_video_vae_arch_config(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("first.png", "last.png")]
            for path in paths:
                Image.new("RGB", (1216, 768)).save(str(path))
            canonical = minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[
                    {
                        "type": "image",
                        "uri": f"file://{path}",
                        "role": "keyframe",
                        "frame_index": frame_index,
                    }
                    for path, frame_index in zip(paths, (0, -1))
                ],
                target={
                    "short_edge": 768,
                    "aspect_ratio": "auto",
                    "duration_seconds": 5.1,
                },
            )
            batch = _prequeue_batch(canonical)
            stage.forward(batch, server_args=None)

        payload = batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY]
        self.assertEqual(list(payload["rows"].shape), [2 * 912, 96])
        self.assertEqual(vae.model.encode_calls, 2)
        self.assertEqual(payload["semantic_frame_indices"], [0, -1])
        self.assertEqual(payload["pixel_frame_indices"], [0, 123])
        self.assertEqual(len(payload["keyframes"]), 2)
        self.assertEqual(
            [item["frame_index"] for item in payload["keyframes"]],
            [0, -1],
        )
        self.assertEqual(
            [item["resolved_frame_index"] for item in payload["keyframes"]],
            [0, 123],
        )

    def test_t2va_plan_skips_visual_encode(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=None,
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )
        stage = MiniMaxH3VisualEncodingStage(
            video_vae=SimpleNamespace(model=None),
            vae_arch_config=_video_vae_arch_config(),
        )
        batch = _batch(canonical)
        stage.forward(batch, server_args=None)
        self.assertNotIn(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY, batch.extra)

    def test_reference_video_encode_requires_non_empty_videos_list(self):
        stage = MiniMaxH3VisualEncodingStage(
            video_vae=object(),
            vae_arch_config=_video_vae_arch_config(),
        )
        legacy_single_payload = {
            "prepared_path": "/prepared.mp4",
            "condition_index": 0,
            "material_chain": "video.reference_preserve",
        }
        for prepared in (legacy_single_payload, {"videos": []}):
            with self.subTest(prepared=prepared):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.reference_encoding."
                    "minimax_h3_prepared_reference_videos",
                    return_value=prepared,
                ):
                    with self.assertRaisesRegex(ValueError, r"non-empty 'videos' list"):
                        stage._encode_reference_video(
                            SimpleNamespace(extra={}), plan=None
                        )


if __name__ == "__main__":
    unittest.main()
