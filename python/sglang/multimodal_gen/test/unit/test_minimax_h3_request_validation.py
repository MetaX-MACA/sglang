# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)


def _target(**overrides):
    base = {
        "short_edge": 768,
        "aspect_ratio": "auto",
        "duration_seconds": 5.0,
    }
    base.update(overrides)
    return base


def _keyframe(frame_index, uri="file:///kf.png"):
    return {"type": "image", "uri": uri, "role": "keyframe", "frame_index": frame_index}


def _reference(cond_type, uri="file:///ref.bin"):
    return {"type": cond_type, "uri": uri, "role": "reference"}


class TestMiniMaxH3RequestValidation(unittest.TestCase):
    def test_t2va_minimal_ok_and_seed_zero_legal(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="a cat",
            conditions=None,
            target=_target(aspect_ratio="16:9"),
            seed=0,
        )
        self.assertEqual(canonical["task"], "t2va")
        self.assertEqual(canonical["conditions"], [])
        self.assertEqual(canonical["seed"], 0)

    def test_canonical_seed_must_be_non_negative_and_fit_signed_int64(self):
        for seed, message in ((-1, "non-negative"), (1 << 63, "signed int64")):
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(ValueError, message),
            ):
                minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="a cat",
                    conditions=None,
                    target=_target(aspect_ratio="16:9"),
                    seed=seed,
                )

    def test_target_drops_explicit_latent_shape_object_and_null(self):
        for explicit_shape in (
            {
                "out_T": 62,
                "out_H": 48,
                "out_W": 84,
                "out_audio_T": 347,
            },
            None,
        ):
            with self.subTest(explicit_latent_shape=explicit_shape):
                canonical = minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="a cat",
                    conditions=None,
                    target=_target(
                        aspect_ratio="16:9",
                        explicit_latent_shape=explicit_shape,
                    ),
                )
                self.assertNotIn("explicit_latent_shape", canonical["target"])

    def test_flow_shifts_are_normalized_and_preserved_independently(self):
        cases = (
            ({"flow_shift": 9}, {"flow_shift": 9.0}),
            ({"audio_flow_shift": 4}, {"audio_flow_shift": 4.0}),
            (
                {"flow_shift": 9, "audio_flow_shift": 4},
                {"flow_shift": 9.0, "audio_flow_shift": 4.0},
            ),
            ({"flow_shift": None, "audio_flow_shift": None}, {}),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                canonical = minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="a cat",
                    conditions=None,
                    target=_target(aspect_ratio="16:9"),
                    **overrides,
                )
                for name in ("flow_shift", "audio_flow_shift"):
                    if name in expected:
                        self.assertEqual(canonical[name], expected[name])
                    else:
                        self.assertNotIn(name, canonical)

    def test_flow_shifts_must_be_positive_finite_numbers(self):
        invalid_values = (
            True,
            False,
            "3",
            0,
            -0.1,
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for name in ("flow_shift", "audio_flow_shift"):
            for value in invalid_values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"{name} must be a number|{name} must be a positive finite number",
                    ),
                ):
                    minimax_h3_validate_canonical_request(
                        task="t2va",
                        prompt="a cat",
                        conditions=None,
                        target=_target(aspect_ratio="16:9"),
                        **{name: value},
                    )

    def test_minimax_h3_drops_fps_from_target_parameter(self):
        for fps in (12, 24):
            with self.subTest(fps=fps):
                canonical = minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="a cat",
                    conditions=None,
                    target=_target(aspect_ratio="16:9", fps=fps),
                )
                self.assertNotIn("fps", canonical["target"])

    def test_t2va_rejects_conditions(self):
        with self.assertRaisesRegex(ValueError, r"conditions must be empty"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="p",
                conditions=[_reference("image")],
                target=_target(aspect_ratio="16:9"),
            )

    def test_unknown_tasks_and_missing_prompt(self):
        for task in ("foo2va", "bar2va"):
            with (
                self.subTest(task=task),
                self.assertRaisesRegex(ValueError, "unknown minimax_h3 task"),
            ):
                minimax_h3_validate_canonical_request(
                    task=task, prompt="p", conditions=None, target=_target()
                )
        with self.assertRaisesRegex(ValueError, "prompt"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="",
                conditions=None,
                target=_target(aspect_ratio="16:9"),
            )

    def test_fl2va_accepts_first_last_and_first_plus_last_variants(self):
        for frame_indices in ([0], [-1], [0, -1]):
            with self.subTest(frame_indices=frame_indices):
                canonical = minimax_h3_validate_canonical_request(
                    task="fl2va",
                    prompt="p",
                    conditions=[
                        _keyframe(index, f"file:///frame-{index}.png")
                        for index in frame_indices
                    ],
                    target=_target(),
                )
                self.assertEqual(canonical["task"], "fl2va")
                self.assertEqual(
                    [condition["frame_index"] for condition in canonical["conditions"]],
                    frame_indices,
                )

    def test_fl2va_duration_target_preserves_first_last_sentinel(self):
        # 5.1s * 24fps rounds to 122 requested frames, then ceil-aligns to 124.
        target = {
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.1,
        }
        canonical = minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="p",
            conditions=[_keyframe(0), _keyframe(-1)],
            target=target,
        )
        self.assertEqual(
            [condition["frame_index"] for condition in canonical["conditions"]],
            [0, -1],
        )

    def test_fl2va_rejects_middle_multi_reverse_and_wrong_indices(self):
        invalid_indices = (
            [52],
            [0, 52],
            [0, 52, -1],
            [-1, 0],
            [0, 0],
            [0, 106],
            [-2, -1],
        )
        for frame_indices in invalid_indices:
            with (
                self.subTest(frame_indices=frame_indices),
                self.assertRaisesRegex(
                    ValueError,
                    r"at most 2 entries|frame_index \[0\], \[-1\], or "
                    r"\[0, -1\]|already bound|17n\+5",
                ),
            ):
                minimax_h3_validate_canonical_request(
                    task="fl2va",
                    prompt="p",
                    conditions=[_keyframe(index) for index in frame_indices],
                    target=_target(),
                )

    def test_fl2va_rejects_missing_or_non_integer_frame_index(self):
        with self.assertRaisesRegex(ValueError, r"frame_index must be an integer"):
            minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[
                    {"type": "image", "uri": "file:///a", "role": "keyframe"},
                    _keyframe(-1),
                ],
                target=_target(),
            )
        with self.assertRaisesRegex(ValueError, r"frame_index must be an integer"):
            minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[_keyframe(True), _keyframe(-1)],
                target=_target(),
            )

    def test_fl2va_rejects_wrong_condition_type_or_role(self):
        with self.assertRaisesRegex(ValueError, r"conditions\[0\].*does not allow"):
            minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[
                    {
                        "type": "video",
                        "uri": "file:///v",
                        "role": "keyframe",
                        "frame_index": 0,
                    },
                    _keyframe(-1),
                ],
                target=_target(),
            )
        with self.assertRaisesRegex(ValueError, r"conditions\[1\].*does not allow"):
            minimax_h3_validate_canonical_request(
                task="fl2va",
                prompt="p",
                conditions=[_keyframe(0), _reference("image")],
                target=_target(),
            )

    def test_ref2va_mixed_types_ok_but_keyframe_rejected(self):
        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                _reference("image"),
                _reference("audio", uri="file:///a.wav"),
                _reference("audio", uri="file:///b.wav"),
            ],
            target=_target(),
        )
        self.assertEqual(len(canonical["conditions"]), 3)
        with self.assertRaisesRegex(ValueError, r"conditions\[0\].*does not allow"):
            minimax_h3_validate_canonical_request(
                task="ref2va",
                prompt="p",
                conditions=[_keyframe(0)],
                target=_target(),
            )
        with self.assertRaisesRegex(ValueError, r"frame_index is not allowed"):
            minimax_h3_validate_canonical_request(
                task="ref2va",
                prompt="p",
                conditions=[{**_reference("image"), "frame_index": 3}],
                target=_target(),
            )

    def test_ref2va_video_reference_accepted(self):
        # ref2va video (image + video reference) is accepted.
        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[_reference("image"), _reference("video")],
            target=_target(),
        )
        self.assertEqual(
            [c["type"] for c in canonical["conditions"]], ["image", "video"]
        )

    def test_ref2va_video_audio_families_accepted(self):
        rva = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[_reference("video_audio", uri="file:///va.mp4")],
            target=_target(aspect_ratio="16:9"),
        )
        self.assertEqual([c["type"] for c in rva["conditions"]], ["video_audio"])

        rirva = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                _reference("image", uri="file:///subject.png"),
                _reference("video_audio", uri="file:///va.mp4"),
            ],
            target=_target(aspect_ratio="16:9"),
        )
        self.assertEqual(
            [c["type"] for c in rirva["conditions"]],
            ["image", "video_audio"],
        )

    def test_ref2va_accepts_supported_reference_combinations(self):
        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[
                _reference("audio", uri="file:///a.wav"),
                _reference("video", uri="file:///v0.mp4"),
                _reference("video", uri="file:///v1.mp4"),
                _reference("video_audio", uri="file:///va0.mp4"),
                _reference("video_audio", uri="file:///va1.mp4"),
                _reference("image", uri="file:///i0.png"),
                _reference("image", uri="file:///i1.png"),
            ],
            target=_target(aspect_ratio="16:9"),
        )
        self.assertEqual(
            [c["type"] for c in canonical["conditions"]],
            [
                "audio",
                "video",
                "video",
                "video_audio",
                "video_audio",
                "image",
                "image",
            ],
        )

    def test_canonical_request_drops_extra_kwargs(self):
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=[],
            target=_target(),
            uncond_drop_conditions=["text"],
            negative_prompt="avoid blur",
            arbitrary_compatibility_value={"ignored": True},
        )
        self.assertNotIn("uncond_drop_conditions", canonical)
        self.assertNotIn("negative_prompt", canonical)
        self.assertNotIn("arbitrary_compatibility_value", canonical)

    def test_canonical_request_drops_removed_keyframe_instruction(self):
        for value in (True, False, None):
            with self.subTest(value=value):
                canonical = minimax_h3_validate_canonical_request(
                    task="fl2va",
                    prompt="p",
                    conditions=[_keyframe(0), _keyframe(-1)],
                    target=_target(),
                    add_keyframe_instruction=value,
                )
                self.assertNotIn("add_keyframe_instruction", canonical)

    def test_canonical_request_drops_removed_time_spec(self):
        for value in (
            None,
            {},
            {"mode": "time_shift_only", "num_steps": 50},
            {"video": {"shift_scale": 12}, "audio": {"shift_scale": 3}},
        ):
            with self.subTest(value=value):
                canonical = minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="p",
                    conditions=None,
                    target=_target(aspect_ratio="16:9"),
                    time_spec=value,
                )
                self.assertNotIn("time_spec", canonical)

    def test_ref2va_deferred_duration_rejects_multiple_audio_sources(self):
        with self.assertRaisesRegex(ValueError, "multiple audio-bearing"):
            minimax_h3_validate_canonical_request(
                task="ref2va",
                prompt="p",
                conditions=[
                    _reference("video", uri="file:///v.mp4"),
                    _reference("audio", uri="file:///a.wav"),
                ],
                target={"short_edge": 768, "aspect_ratio": "16:9"},
            )

    def test_ref2va_finite_bucket_aspect_ratio_allowed(self):
        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="p",
            conditions=[_reference("image"), _reference("audio")],
            target={"short_edge": 768, "aspect_ratio": "16:9"},
        )
        self.assertEqual(canonical["target"]["aspect_ratio"], "16:9")
        self.assertNotIn("num_frames", canonical["target"])
        self.assertNotIn("duration_seconds", canonical["target"])

    def test_ref2va_deferred_duration_requires_audio_reference(self):
        with self.assertRaisesRegex(ValueError, "audio\\s+reference to derive"):
            minimax_h3_validate_canonical_request(
                task="ref2va",
                prompt="p",
                conditions=[_reference("image")],
                target={"short_edge": 768, "aspect_ratio": "16:9"},
            )

    def test_t2va_and_ref2va_accept_only_finite_buckets_or_auto(self):
        ratios = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "auto")
        for task in ("t2va", "ref2va"):
            for ratio in ratios:
                with self.subTest(task=task, ratio=ratio):
                    canonical = minimax_h3_validate_canonical_request(
                        task=task,
                        prompt="p",
                        conditions=(
                            None
                            if task == "t2va"
                            else [_reference("image"), _reference("audio")]
                        ),
                        target={
                            "short_edge": 768,
                            "aspect_ratio": ratio,
                            "duration_seconds": 5.0,
                        },
                    )
                    self.assertEqual(canonical["target"]["aspect_ratio"], ratio)

            with (
                self.subTest(task=task, ratio="7:4"),
                self.assertRaisesRegex(ValueError, "must be 'auto' or one of"),
            ):
                minimax_h3_validate_canonical_request(
                    task=task,
                    prompt="p",
                    conditions=(
                        None
                        if task == "t2va"
                        else [_reference("image"), _reference("audio")]
                    ),
                    target={
                        "short_edge": 768,
                        "aspect_ratio": "7:4",
                        "duration_seconds": 5.0,
                    },
                )

    def test_fl2va_explicit_aspect_ratio_allowed(self):
        canonical = minimax_h3_validate_canonical_request(
            task="fl2va",
            prompt="p",
            conditions=[_keyframe(0), _keyframe(-1)],
            target=_target(aspect_ratio="1344:768"),
        )
        self.assertEqual(canonical["target"]["aspect_ratio"], "1344:768")

    def test_target_requires_duration_and_drops_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "duration_seconds is required"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="p",
                conditions=None,
                target={"short_edge": 768, "aspect_ratio": "16:9"},
            )
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="p",
            conditions=None,
            target=_target(aspect_ratio="16:9", num_frames=121, resolved_shape=[1, 2]),
        )
        self.assertNotIn("num_frames", canonical["target"])
        self.assertNotIn("resolved_shape", canonical["target"])

    def test_duration_range_is_5_to_15_seconds(self):
        for duration in (4.999, 15.001):
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, r"must be in \[5, 15\]"),
            ):
                minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="p",
                    conditions=None,
                    target=_target(aspect_ratio="16:9", duration_seconds=duration),
                )

    def test_target_short_edge_is_strictly_768(self):
        for short_edge in (1, 720, 736, 800, 1024):
            with (
                self.subTest(short_edge=short_edge),
                self.assertRaisesRegex(ValueError, r"target.short_edge must be 768"),
            ):
                minimax_h3_validate_canonical_request(
                    task="t2va",
                    prompt="p",
                    conditions=None,
                    target=_target(aspect_ratio="16:9", short_edge=short_edge),
                )

    def test_conditions_order_preserved(self):
        canonical = minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="<Picture 1> then <Audio 1>",
            conditions=[
                _reference("image", uri="file:///a.png"),
                _reference("audio", uri="file:///b.wav"),
            ],
            target=_target(),
        )
        self.assertEqual(
            [c["uri"] for c in canonical["conditions"]],
            ["file:///a.png", "file:///b.wav"],
        )


if __name__ == "__main__":
    unittest.main()
