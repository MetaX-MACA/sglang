# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    MiniMaxH3ResolvedPlan,
    minimax_h3_plan_from_batch,
    minimax_h3_resolve_plan,
    minimax_h3_resolve_spatial_shape,
)


def _canonical(task, conditions, target, **kw):
    return minimax_h3_validate_canonical_request(
        task=task,
        prompt=kw.pop("prompt", "p"),
        conditions=conditions,
        target=target,
        **kw,
    )


class TestAdaptShapeV1(unittest.TestCase):
    def test_finite_bucket_goldens(self):
        cases = (
            ((21, 9), (1536, 672), "area"),
            ((16, 9), (1344, 768), "area"),
            ((4, 3), (1024, 768), "short_edge"),
            ((1, 1), (768, 768), "short_edge"),
            ((3, 4), (768, 1024), "short_edge"),
            ((9, 16), (768, 1344), "area"),
        )
        for (source_w, source_h), expected_size, expected_mode in cases:
            with self.subTest(ratio=f"{source_w}:{source_h}"):
                resolved = minimax_h3_resolve_spatial_shape(
                    width=source_w, height=source_h
                )
                self.assertEqual((resolved["width"], resolved["height"]), expected_size)
                self.assertEqual(resolved["size_mode"], expected_mode)
                self.assertEqual(resolved["geometry"], "resolved_v2")
                self.assertEqual(resolved["shape_policy_version"], "adapt_shape_v1")
                self.assertEqual(resolved["max_pixels"], MINIMAX_H3_MAX_PIXELS)
                self.assertEqual(resolved["multiple"], 32)
                self.assertEqual(resolved["rounding"], "nearest")

    def test_flexible_ratios_and_reciprocal_symmetry(self):
        cases = (
            ((887, 495), (1376, 768)),
            ((2, 1), (1440, 704)),
            ((3, 1), (1760, 576)),
            ((4, 1), (2016, 512)),
            ((1205, 2124), (768, 1344)),
        )
        for (width, height), expected in cases:
            with self.subTest(ratio=f"{width}:{height}"):
                resolved = minimax_h3_resolve_spatial_shape(width=width, height=height)
                self.assertEqual((resolved["width"], resolved["height"]), expected)
                transposed = minimax_h3_resolve_spatial_shape(width=height, height=width)
                self.assertEqual(
                    (transposed["width"], transposed["height"]),
                    (expected[1], expected[0]),
                )

    def test_ratio_range_is_inclusive_one_to_four(self):
        for width, height in ((4, 1), (1, 4)):
            minimax_h3_resolve_spatial_shape(width=width, height=height)
        for width, height in ((401, 100), (100, 401), (5, 1), (1, 5)):
            with (
                self.subTest(ratio=f"{width}:{height}"),
                self.assertRaisesRegex(ValueError, "inclusive range 1:4 to 4:1"),
            ):
                minimax_h3_resolve_spatial_shape(width=width, height=height)

    def test_nearest_grid_uses_soft_not_hard_area_cap(self):
        resolved = minimax_h3_resolve_spatial_shape(width=887, height=495)
        self.assertEqual((resolved["width"], resolved["height"]), (1376, 768))
        self.assertGreater(resolved["width"] * resolved["height"], MINIMAX_H3_MAX_PIXELS)
        self.assertEqual(resolved["width"] % 32, 0)
        self.assertEqual(resolved["height"] % 32, 0)

    def test_only_768_base_short_edge_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "must be 768"):
            minimax_h3_resolve_spatial_shape(width=16, height=9, base_short_edge=720)


class TestMiniMaxH3ResolvedPlan(unittest.TestCase):
    def test_plan_defensively_rejects_empty_tampered_fl_condition_set(self):
        canonical = _canonical(
            "fl2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
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
            {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.1},
        )
        canonical["conditions"] = []

        with self.assertRaisesRegex(ValueError, "requires one or two ordered"):
            minimax_h3_resolve_plan(canonical)

    def test_t2va_plan_explicit_geometry(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "t2va",
                None,
                {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
                seed=0,
            )
        )
        self.assertEqual(plan.task, "t2va")
        self.assertEqual(plan.materials, ())
        self.assertEqual(plan.encoders["visual"], [])
        self.assertEqual(plan.encoders["audio"], [])
        self.assertEqual(plan.condition_mask, {})
        self.assertEqual(plan.seed, 0)
        shape = plan.shape
        self.assertEqual(shape["geometry"], "resolved_v2")
        self.assertEqual(shape["height"], 768)
        self.assertEqual(shape["width"], 1344)
        self.assertEqual(shape["shape_policy_version"], "adapt_shape_v1")
        self.assertEqual(shape["base_short_edge"], 768)
        self.assertEqual(shape["effective_short_edge"], 768)
        self.assertEqual(shape["size_mode"], "area")
        self.assertEqual(shape["frame_count"], 124)  # ceil alignment of 5s at 24fps
        self.assertEqual(shape["video_latent_t"], 37)  # (124-5)//17*5+2

    def test_t2va_auto_geometry_resolves_to_policy_default(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "t2va",
                None,
                {
                    "short_edge": 768,
                    "aspect_ratio": "auto",
                    "duration_seconds": 5.0,
                },
            )
        )

        self.assertEqual(plan.shape["geometry"], "resolved_v2")
        self.assertEqual(plan.shape["geometry_source"], "policy_default")
        self.assertEqual((plan.shape["height"], plan.shape["width"]), (768, 1344))

    def test_fl2va_plan_masks_and_explicit_geometry(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "fl2va",
                [
                    {
                        "type": "image",
                        "uri": "file:///b.png",
                        "role": "keyframe",
                        "frame_index": 0,
                    },
                    {
                        "type": "image",
                        "uri": "file:///z.png",
                        "role": "keyframe",
                        "frame_index": -1,
                    },
                ],
                {
                    "short_edge": 768,
                    "aspect_ratio": "1344:768",
                    "duration_seconds": 5.1,
                },
            )
        )
        self.assertEqual(plan.shape["geometry"], "resolved_v2")
        self.assertEqual(plan.shape["geometry_source"], "explicit_target")
        self.assertEqual(plan.shape["width"], 1344)
        self.assertEqual(plan.shape["height"], 768)
        self.assertEqual(plan.condition_mask["semantic_frame_indices"], [0, -1])
        self.assertEqual(plan.condition_mask["pixel_frame_indices"], [0, 123])
        self.assertEqual(
            [m.material_chain for m in plan.materials],
            ["image.target_canvas"] * 2,
        )
        self.assertEqual(plan.encoders["visual"], [0, 1])
        self.assertEqual(plan.encoders["audio"], [])
        self.assertEqual([m.frame_index for m in plan.materials], [0, -1])
        self.assertEqual([m.resolved_frame_index for m in plan.materials], [0, 123])

    def test_fl2va_plan_accepts_single_first_or_last_keyframe(self):
        for semantic_index, resolved_index in ((0, 0), (-1, 123)):
            with self.subTest(frame_index=semantic_index):
                plan = minimax_h3_resolve_plan(
                    _canonical(
                        "fl2va",
                        [
                            {
                                "type": "image",
                                "uri": f"file:///frame-{semantic_index}.png",
                                "role": "keyframe",
                                "frame_index": semantic_index,
                            }
                        ],
                        {
                            "short_edge": 768,
                            "aspect_ratio": "16:9",
                            "duration_seconds": 5.1,
                        },
                    )
                )

                self.assertEqual(
                    plan.condition_mask["semantic_frame_indices"],
                    [semantic_index],
                )
                self.assertEqual(
                    plan.condition_mask["pixel_frame_indices"],
                    [resolved_index],
                )
                self.assertEqual(len(plan.materials), 1)
                self.assertEqual(plan.encoders["visual"], [0])

    def test_ref2va_plan_encoder_routing(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "ref2va",
                [
                    {"type": "audio", "uri": "file:///a.wav", "role": "reference"},
                    {"type": "image", "uri": "file:///b.png", "role": "reference"},
                ],
                {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0},
            )
        )
        self.assertEqual(plan.encoders["audio"], [0])
        self.assertEqual(plan.encoders["visual"], [1])
        self.assertEqual(plan.condition_mask, {})
        self.assertEqual(
            [m.material_chain for m in plan.materials],
            ["audio", "image.reference_preserve"],
        )
        self.assertEqual(plan.shape["geometry_source"], "policy_default")
        self.assertEqual(plan.shape["geometry"], "resolved_v2")
        self.assertEqual((plan.shape["height"], plan.shape["width"]), (768, 1344))
        self.assertGreater(plan.shape["audio_latent_t"], 0)

    def test_ref2va_explicit_aspect_and_deferred_duration(self):
        # ref2va uses one of the six finite target buckets; duration may still
        # derive from reference audio.
        plan = minimax_h3_resolve_plan(
            _canonical(
                "ref2va",
                [
                    {"type": "image", "uri": "file:///b.png", "role": "reference"},
                    {"type": "audio", "uri": "file:///a.mp3", "role": "reference"},
                ],
                {"short_edge": 768, "aspect_ratio": "16:9"},
            )
        )
        self.assertEqual(plan.shape["geometry"], "resolved_v2")
        self.assertEqual(plan.shape["height"], 768)
        self.assertEqual(plan.shape["width"], 1344)
        self.assertEqual(plan.shape["temporal"], "deferred_from_audio_reference")
        self.assertNotIn("frame_count", plan.shape)

    def test_ref2va_video_audio_routes_to_visual_and_audio(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "ref2va",
                [
                    {
                        "type": "video_audio",
                        "uri": "file:///va.mp4",
                        "role": "reference",
                    },
                ],
                {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 8.7,
                },
            )
        )

        self.assertEqual(plan.encoders["visual"], [0])
        self.assertEqual(plan.encoders["audio"], [0])
        self.assertEqual(
            [m.material_chain for m in plan.materials],
            ["video_audio.reference_preserve"],
        )

    def test_ref2va_reference_combinations_preserve_order_and_routing(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "ref2va",
                [
                    {"type": "audio", "uri": "file:///a.wav", "role": "reference"},
                    {"type": "video", "uri": "file:///v0.mp4", "role": "reference"},
                    {
                        "type": "video_audio",
                        "uri": "file:///va0.mp4",
                        "role": "reference",
                    },
                    {"type": "image", "uri": "file:///i0.png", "role": "reference"},
                    {"type": "video", "uri": "file:///v1.mp4", "role": "reference"},
                ],
                {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 8.7,
                },
            )
        )

        self.assertEqual(plan.encoders["audio"], [0, 1, 2, 4])
        self.assertEqual(plan.encoders["visual"], [1, 2, 3, 4])
        self.assertEqual(
            [m.material_chain for m in plan.materials],
            [
                "audio",
                "video.reference_preserve",
                "video_audio.reference_preserve",
                "image.reference_preserve",
                "video.reference_preserve",
            ],
        )

    def test_plan_has_single_distilled_branch_and_standard_schedule_defaults(self):
        plan = minimax_h3_resolve_plan(
            _canonical(
                "t2va",
                None,
                {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
            )
        )
        self.assertEqual([b["name"] for b in plan.branches], ["cond_1"])
        self.assertNotIn("negative_prompt", plan.encoders["qwen"])
        self.assertNotIn("uncond_drop_conditions", plan.encoders["qwen"])
        self.assertEqual(plan.default_flow_shift, 12.0)
        self.assertEqual(plan.default_audio_flow_shift, 3.0)
        self.assertIsNone(plan.flow_shift)
        self.assertIsNone(plan.audio_flow_shift)

    def test_plan_preserves_single_sided_flow_shift_overrides(self):
        target = {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        }
        cases = (
            ({"flow_shift": 8.5}, 8.5, None),
            ({"audio_flow_shift": 2.5}, None, 2.5),
        )
        for overrides, expected_video_shift, expected_audio_shift in cases:
            with self.subTest(overrides=overrides):
                plan = minimax_h3_resolve_plan(
                    _canonical("t2va", None, target, **overrides)
                )
                self.assertEqual(plan.default_flow_shift, 12.0)
                self.assertEqual(plan.default_audio_flow_shift, 3.0)
                self.assertEqual(plan.flow_shift, expected_video_shift)
                self.assertEqual(plan.audio_flow_shift, expected_audio_shift)

    def test_tampered_middle_keyframe_fails_strict_fl_signature_first(self):
        with self.assertRaisesRegex(
            ValueError, r"frame_index \[0\], \[-1\], or \[0, -1\]"
        ):
            minimax_h3_resolve_plan(
                {
                    "schema": "minimax_h3.request/v1",
                    "task": "fl2va",
                    "prompt": "p",
                    "conditions": [
                        {
                            "type": "image",
                            "uri": "file:///a.png",
                            "role": "keyframe",
                            "frame_index": 120,
                        },
                        {
                            "type": "image",
                            "uri": "file:///z.png",
                            "role": "keyframe",
                            "frame_index": -1,
                        },
                    ],
                    "target": {
                        "short_edge": 768,
                        "aspect_ratio": "auto",
                        "duration_seconds": 5.0,
                    },
                }
            )

    def test_tampered_reverse_and_explicit_last_fails_strict_fl_signature(self):
        with self.assertRaisesRegex(
            ValueError, r"frame_index \[0\], \[-1\], or \[0, -1\]"
        ):
            minimax_h3_resolve_plan(
                {
                    "schema": "minimax_h3.request/v1",
                    "task": "fl2va",
                    "prompt": "p",
                    "conditions": [
                        {
                            "type": "image",
                            "uri": "file:///a.png",
                            "role": "keyframe",
                            "frame_index": -1,
                        },
                        {
                            "type": "image",
                            "uri": "file:///b.png",
                            "role": "keyframe",
                            "frame_index": 123,
                        },
                    ],
                    "target": {
                        "short_edge": 768,
                        "aspect_ratio": "auto",
                        "duration_seconds": 5.1,
                    },
                }
            )

    def test_plan_from_batch_resolves_once_and_caches(self):
        from types import SimpleNamespace

        canonical = _canonical(
            "ref2va",
            [{"type": "image", "uri": "file:///b.png", "role": "reference"}],
            {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.1},
        )
        batch = SimpleNamespace(extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical})
        plan = minimax_h3_plan_from_batch(batch)
        self.assertIsInstance(plan, MiniMaxH3ResolvedPlan)
        self.assertIs(batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY], plan)
        self.assertIs(minimax_h3_plan_from_batch(batch), plan)
        # legacy batches without canonical request resolve to None
        self.assertIsNone(minimax_h3_plan_from_batch(SimpleNamespace(extra={})))

    def test_incomplete_canonical_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "missing 'target'"):
            minimax_h3_resolve_plan(
                {"schema": "x", "task": "t2va", "prompt": "p", "conditions": []}
            )


if __name__ == "__main__":
    unittest.main()
