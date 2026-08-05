# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for keyframe target-canvas preparation (P5-3)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.canvas import (
    MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY,
    minimax_h3_cover_crop_plan,
    minimax_h3_prepare_keyframe_canvas,
    minimax_h3_prepared_keyframes,
    minimax_h3_stretch_keyframe_canvas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
    MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
    minimax_h3_prepare_for_queue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)


def _prequeue_batch(canonical):
    batch = SimpleNamespace(
        extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
        num_inference_steps=1,
        num_outputs_per_prompt=1,
    )
    return batch, minimax_h3_prepare_for_queue(batch)


class TestCoverCrop(unittest.TestCase):
    def test_plan_matches_reference_math(self):
        plan = minimax_h3_cover_crop_plan(
            source_width=1280,
            source_height=808,
            target_width=1216,
            target_height=768,
            allow_upscale=False,
        )
        # scale = max(1216/1280, 768/808) = 0.9504950495...
        self.assertAlmostEqual(plan["scale"], 768 / 808, places=9)
        self.assertEqual(plan["resized_size"], (1217, 768))
        self.assertEqual(plan["crop_box"], (0, 0, 1216, 768))

    def test_upscale_refused_by_default(self):
        with self.assertRaisesRegex(ValueError, "upscale"):
            minimax_h3_cover_crop_plan(
                source_width=640,
                source_height=404,
                target_width=1216,
                target_height=768,
                allow_upscale=False,
            )

    def test_identity_passthrough(self):
        from PIL import Image

        img = Image.new("RGB", (1216, 768), (10, 20, 30))
        out = minimax_h3_prepare_keyframe_canvas(
            img, target_width=1216, target_height=768
        )
        self.assertEqual(out.size, (1216, 768))
        self.assertEqual(out.getpixel((0, 0)), (10, 20, 30))

    def test_prepare_produces_target_size(self):
        from PIL import Image

        img = Image.new("RGB", (1280, 808), (10, 20, 30))
        out = minimax_h3_prepare_keyframe_canvas(
            img, target_width=1216, target_height=768
        )
        self.assertEqual(out.size, (1216, 768))

    def test_fl_first_stretch_differs_from_last_cover_crop(self):
        from PIL import Image

        image = Image.new("RGB", (4, 2))
        image.putdata(
            [
                (255, 0, 0),
                (255, 0, 0),
                (0, 0, 255),
                (0, 0, 255),
            ]
            * 2
        )
        first = minimax_h3_stretch_keyframe_canvas(image, target_width=4, target_height=4)
        last = minimax_h3_prepare_keyframe_canvas(
            image,
            target_width=4,
            target_height=4,
            allow_upscale=True,
        )

        self.assertEqual(first.size, (4, 4))
        self.assertEqual(last.size, (4, 4))
        self.assertNotEqual(first.tobytes(), last.tobytes())


class TestPreparedKeyframeCaching(unittest.TestCase):
    def test_single_i2va_and_l2va_images_are_semantic_anchors_and_stretch(self):
        from PIL import Image

        for semantic_index in (0, -1):
            with (
                self.subTest(semantic_index=semantic_index),
                tempfile.TemporaryDirectory() as tmp,
            ):
                image_path = Path(tmp) / "anchor.png"
                Image.new("RGB", (640, 404), "red").save(image_path)
                canonical = minimax_h3_validate_canonical_request(
                    task="fl2va",
                    prompt="p",
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
                batch, plan = _prequeue_batch(canonical)
                with (
                    patch(
                        "sglang.multimodal_gen.runtime.pipelines_core.stages."
                        "model_specific_stages.minimax_h3.canvas."
                        "minimax_h3_stretch_keyframe_canvas",
                        wraps=minimax_h3_stretch_keyframe_canvas,
                    ) as stretch,
                    patch(
                        "sglang.multimodal_gen.runtime.pipelines_core.stages."
                        "model_specific_stages.minimax_h3.canvas."
                        "minimax_h3_prepare_keyframe_canvas",
                        wraps=minimax_h3_prepare_keyframe_canvas,
                    ) as follower,
                ):
                    prepared = minimax_h3_prepared_keyframes(batch, plan)

                self.assertEqual(stretch.call_count, 1)
                follower.assert_not_called()
                self.assertEqual(prepared["semantic_frame_indices"], [semantic_index])
                self.assertEqual(
                    prepared["pixel_frame_indices"],
                    [0 if semantic_index == 0 else prepared["frame_count"] - 1],
                )
                self.assertEqual(len(prepared["images"]), 1)
                self.assertIs(prepared["image"], prepared["images"][0]["image"])

    def test_canvas_sink_rejects_unsupported_or_stale_signatures(self):
        def material(frame_index: int, resolved_frame_index: int):
            return SimpleNamespace(
                material_chain="image.target_canvas",
                frame_index=frame_index,
                resolved_frame_index=resolved_frame_index,
            )

        for signature in ((1,), (0, 1), (-1, 0), (0, 0), (0, -1, -1)):
            plan = SimpleNamespace(
                task="fl2va",
                materials=[material(index, index) for index in signature],
                shape={
                    "geometry": "resolved_v2",
                    "width": 1216,
                    "height": 768,
                    "frame_count": 124,
                },
            )
            with (
                self.subTest(signature=signature),
                self.assertRaisesRegex(ValueError, "ordered frame_index signatures"),
            ):
                minimax_h3_prepared_keyframes(SimpleNamespace(extra={}), plan)

        stale = SimpleNamespace(
            task="fl2va",
            materials=[material(-1, 0)],
            shape={
                "geometry": "resolved_v2",
                "width": 1216,
                "height": 768,
                "frame_count": 124,
            },
        )
        with self.assertRaisesRegex(ValueError, "resolved_frame_index"):
            minimax_h3_prepared_keyframes(SimpleNamespace(extra={}), stale)

    def test_exif_portrait_geometry_reaches_final_fl_preparation(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.jpg"
            last = Path(tmp) / "last.png"
            source = Image.new("RGB", (40, 20), "red")
            exif = source.getexif()
            exif[274] = 6
            source.save(first, exif=exif)
            Image.new("RGB", (20, 40), "blue").save(last)
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
                    for path, frame_index in ((first, 0), (last, -1))
                ],
                target={
                    "short_edge": 768,
                    "aspect_ratio": "auto",
                    "duration_seconds": 5.0,
                },
            )
            batch, plan = _prequeue_batch(canonical)
            prepared = minimax_h3_prepared_keyframes(batch, plan)

        facts = batch.extra[MINIMAX_H3_PROBE_FACTS_EXTRA_KEY][0]
        self.assertEqual((facts["coded_width"], facts["coded_height"]), (40, 20))
        self.assertEqual((facts["display_width"], facts["display_height"]), (20, 40))
        self.assertEqual((plan.shape["width"], plan.shape["height"]), (704, 1440))
        self.assertEqual(
            [entry["image"].size for entry in prepared["images"]],
            [(704, 1440), (704, 1440)],
        )

    def test_prepared_once_and_cached(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "k.png"
            Image.new("RGB", (1280, 808)).save(str(png))
            canonical = minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[
                    {
                        "type": "image",
                        "uri": f"file://{png}",
                        "role": "keyframe",
                        "frame_index": 0,
                    },
                    {
                        "type": "image",
                        "uri": f"file://{png}",
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
            batch, plan = _prequeue_batch(canonical)
            prepared = minimax_h3_prepared_keyframes(batch, plan)
            self.assertEqual(prepared["canvas_width"], 1216)
            self.assertEqual(prepared["canvas_height"], 768)
            self.assertEqual(prepared["image"].size, (1216, 768))
            self.assertIn(MINIMAX_H3_PREPARED_KEYFRAMES_EXTRA_KEY, batch.extra)
            self.assertIs(minimax_h3_prepared_keyframes(batch, plan), prepared)

    def test_prepares_first_and_last_on_first_keyframe_shared_canvas(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("first.png", "last.png")]
            # The first and last sources are smaller than the resolved canvas.
            # FL prepares both on one canvas: first stretches directly and the
            # last cover-resizes/center-crops.
            for path, size in zip(
                paths,
                ((640, 404), (600, 400)),
            ):
                Image.new("RGB", size).save(str(path))
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
            batch, plan = _prequeue_batch(canonical)
            prepared = minimax_h3_prepared_keyframes(batch, plan)

        self.assertEqual(
            (prepared["canvas_width"], prepared["canvas_height"]),
            (1216, 768),
        )
        self.assertEqual(
            [item["image"].size for item in prepared["images"]],
            [(1216, 768)] * 2,
        )
        self.assertEqual(prepared["semantic_frame_indices"], [0, -1])
        self.assertEqual(prepared["pixel_frame_indices"], [0, 123])
        self.assertEqual(
            [item["condition_index"] for item in prepared["images"]], [0, 1]
        )
        self.assertEqual(
            [item["resolved_frame_index"] for item in prepared["images"]],
            [0, 123],
        )
        self.assertEqual(prepared["frame_count"], 124)


if __name__ == "__main__":
    unittest.main()
