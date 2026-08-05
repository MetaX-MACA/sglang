# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DEFAULT_BRANCHES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_CANONICAL_MATERIAL_CHAINS,
    MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
    MINIMAX_H3_CONDITION_ROLE_REFERENCE,
    MINIMAX_H3_FINITE_ASPECT_RATIOS,
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
    MINIMAX_H3_TASK_PROFILES,
    minimax_h3_task_profile,
)


class TestMiniMaxH3TaskProfiles(unittest.TestCase):
    def test_v1_task_rows_present(self):
        self.assertEqual(
            sorted(MINIMAX_H3_TASK_PROFILES),
            ["fl2va", "ref2va", "t2va"],
        )

    def test_unknown_task_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "unknown minimax_h3 task"):
            minimax_h3_task_profile("foo2va")

    def test_roles_never_mix_within_a_task(self):
        for profile in MINIMAX_H3_TASK_PROFILES.values():
            roles = {rule.role for rule in profile.condition_rules}
            self.assertLessEqual(len(roles), 1, profile.task)

    def test_t2va_takes_no_conditions(self):
        profile = minimax_h3_task_profile("t2va")
        self.assertFalse(profile.conditions_required)
        self.assertEqual(profile.condition_rules, ())
        self.assertFalse(profile.aspect_ratio_forced_auto)
        self.assertEqual(profile.geometry_source, "explicit_target")
        self.assertEqual(profile.auto_aspect_ratio, "16:9")
        self.assertEqual(profile.auto_geometry_source, "policy_default")
        self.assertEqual(
            MINIMAX_H3_FINITE_ASPECT_RATIOS,
            ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        )

    def test_unknown_task_has_no_alias(self):
        with self.assertRaisesRegex(ValueError, "unknown minimax_h3 task"):
            minimax_h3_task_profile("bar2va")

    def test_fl2va_is_one_or_two_keyframe_images_with_supported_signatures(self):
        profile = minimax_h3_task_profile("fl2va")
        self.assertTrue(profile.conditions_required)
        self.assertEqual(profile.min_condition_count, 1)
        self.assertEqual(profile.max_condition_count, 2)
        self.assertEqual(
            MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
            ((0,), (-1,), (0, -1)),
        )
        self.assertFalse(profile.aspect_ratio_forced_auto)
        self.assertEqual(profile.geometry_source, "explicit_target")
        self.assertIsNone(profile.auto_aspect_ratio)
        self.assertEqual(profile.auto_geometry_source, "first_keyframe")
        rule = profile.rule_for(
            role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME, condition_type="image"
        )
        self.assertTrue(rule.requires_frame_index)
        self.assertTrue(rule.visual_tokenizer_encode)
        self.assertEqual(rule.material_chain, "image.target_canvas")
        with self.assertRaisesRegex(ValueError, "does not allow"):
            profile.rule_for(
                role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME, condition_type="video"
            )
        with self.assertRaisesRegex(ValueError, "does not allow"):
            profile.rule_for(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE, condition_type="image"
            )

    def test_ref2va_reference_routing(self):
        profile = minimax_h3_task_profile("ref2va")
        # Explicit aspect ratios allowed; duration may derive from a
        # reference audio. Video refs (ref2va video) are enabled.
        self.assertFalse(profile.aspect_ratio_forced_auto)
        self.assertEqual(profile.geometry_source, "explicit_target")
        self.assertEqual(profile.auto_aspect_ratio, "16:9")
        self.assertEqual(profile.auto_geometry_source, "policy_default")
        self.assertTrue(profile.duration_from_audio_reference)
        self.assertTrue(profile.video_reference_supported)
        image = profile.rule_for(
            role=MINIMAX_H3_CONDITION_ROLE_REFERENCE, condition_type="image"
        )
        video = profile.rule_for(
            role=MINIMAX_H3_CONDITION_ROLE_REFERENCE, condition_type="video"
        )
        video_audio = profile.rule_for(
            role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
            condition_type="video_audio",
        )
        audio = profile.rule_for(
            role=MINIMAX_H3_CONDITION_ROLE_REFERENCE, condition_type="audio"
        )
        self.assertEqual(image.material_chain, "image.reference_preserve")
        self.assertEqual(video.material_chain, "video.reference_preserve")
        self.assertEqual(video_audio.material_chain, "video_audio.reference_preserve")
        self.assertEqual(audio.material_chain, "audio")
        self.assertTrue(image.visual_tokenizer_encode)
        self.assertTrue(video.visual_tokenizer_encode)
        self.assertTrue(video_audio.visual_tokenizer_encode)
        # The reference video's own soundtrack is the audio reference
        self.assertTrue(video.audio_tokenizer_encode)
        self.assertTrue(video_audio.audio_tokenizer_encode)
        self.assertTrue(audio.audio_tokenizer_encode)
        self.assertFalse(audio.visual_tokenizer_encode)
        for rule in profile.condition_rules:
            self.assertFalse(rule.requires_frame_index)

    def test_material_chains_and_task_schedule_defaults_are_consistent(self):
        for task, profile in MINIMAX_H3_TASK_PROFILES.items():
            for rule in profile.condition_rules:
                self.assertIn(
                    rule.material_chain,
                    set(MINIMAX_H3_CANONICAL_MATERIAL_CHAINS),
                )
            self.assertEqual(
                [dict(branch) for branch in profile.branches],
                [dict(branch) for branch in MINIMAX_H3_DEFAULT_BRANCHES],
            )
            expected_shifts = (12.0, 3.0)
            self.assertEqual(profile.default_flow_shift, expected_shifts[0])
            self.assertEqual(profile.default_audio_flow_shift, expected_shifts[1])
            self.assertIn("transformer", profile.required_components)
            self.assertIn("processor", profile.required_components)
            # scheduler intentionally absent: model_index carries
            # scheduler=null; sigma schedules are generated in
            # TimestepPreparation from the task profile.
            self.assertNotIn("scheduler", profile.required_components)


if __name__ == "__main__":
    unittest.main()
