# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the ref2va packed sequence builder."""

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    MINIMAX_H3_AUDIO_REF_COND_ID,
    MINIMAX_H3_IMGVID_COND_ID,
    minimax_h3_packed_sequence_ref2va_blocks,
)

# Ref2va golden case geometry:
# target 1344x768 -> latent 84x48, out_T=112, out_audio_T=631;
# ref image 1024x768 intrinsic -> latent 64x48; ref audio T=582.
CASE = dict(
    latent_t=112,
    latent_h=48,
    latent_w=84,
    audio_t=631,
    ref_blocks=[
        {"kind": "image", "latent_h": 64, "latent_w": 48},
        {"kind": "audio", "ref_audio_t": 582},
    ],
)


class TestMiniMaxH3Ref2vaPackedSequence(unittest.TestCase):
    def test_hermetic_structure(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(text_len=97, **CASE)
        self.assertEqual(int(built["seq_len"]) % 64, 0)
        self.assertEqual(int(built["img_pos"].shape[0]), 32 * 24 + 112 * 24 * 42)
        self.assertEqual(int(built["audio_pos"].shape[0]), 582 * 2 + 631 * 2)
        self.assertEqual(int((~built["update_mask"]).sum()), 32 * 24)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 582 * 2)
        tags = built["token_tags"]
        self.assertEqual(tags[built["audio_pos"]].unique().tolist(), [2])
        self.assertEqual(tags[built["img_pos"]].unique().tolist(), [0])

    def test_multi_audio_blocks_preserve_order(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=10,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 3},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )

        self.assertEqual(int((~built["update_mask"]).sum()), 4)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 10)
        self.assertTrue((built["input_ids"][10:14] == MINIMAX_H3_IMGVID_COND_ID).all())
        self.assertTrue((built["input_ids"][14:24] == MINIMAX_H3_AUDIO_REF_COND_ID).all())
        audio_pos = built["audio_pos"][:10]
        audio_t = built["img_position_ids"][audio_pos, 0]
        self.assertEqual(audio_t[:3].tolist(), [11.0, 12.0, 13.0])
        self.assertEqual(audio_t[6:8].tolist(), [14.0, 15.0])

    def test_video_audio_only_layout(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=7,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
        )

        ref_video_rows = 2 * 2 * 2
        self.assertEqual(int((~built["update_mask"]).sum()), ref_video_rows)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 6)
        self.assertTrue((built["input_ids"][7:13] == MINIMAX_H3_AUDIO_REF_COND_ID).all())
        self.assertTrue((built["input_ids"][13:21] == MINIMAX_H3_IMGVID_COND_ID).all())
        target_audio_t0 = built["img_position_ids"][built["audio_pos"][6], 0]
        # Video reference rows consume the temporal RoPE span (8 1/3
        # slots for two latent frames), even when their audio span is shorter.
        self.assertAlmostEqual(float(target_audio_t0), 7.0 + 25.0 / 3.0)

    def test_image_plus_video_audio_layout(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=7,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
        )

        self.assertEqual(int((~built["update_mask"]).sum()), 4 + 8)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 6)
        self.assertTrue((built["input_ids"][7:11] == MINIMAX_H3_IMGVID_COND_ID).all())
        self.assertTrue((built["input_ids"][11:17] == MINIMAX_H3_AUDIO_REF_COND_ID).all())
        target_audio_t0 = built["img_position_ids"][built["audio_pos"][6], 0]
        self.assertAlmostEqual(float(target_audio_t0), 8.0 + 25.0 / 3.0)

    def test_silent_video_advances_target_temporal_origin(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 10,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
        )

        image_rows = 4
        video_rows = 8
        silent_video_t0 = built["img_position_ids"][
            built["img_pos"][image_rows + video_rows], 0
        ]
        target_video_t0 = built["img_position_ids"][
            built["img_pos"][image_rows + 2 * video_rows], 0
        ]
        target_audio_t0 = built["img_position_ids"][built["audio_pos"][2 * 10], 0]
        silent_video_span = (5.0 / 3.0) * (1 + 4)

        self.assertAlmostEqual(
            float(target_video_t0 - silent_video_t0), silent_video_span
        )
        self.assertAlmostEqual(float(target_audio_t0), float(target_video_t0))

    def test_multi_image_video_and_audio_mix_layout(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video",
                    "ref_audio_t": 2,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {"kind": "audio", "ref_audio_t": 1},
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
        )

        ref_visual_rows = 4 + 8 + 4 + 8
        ref_audio_rows = (2 + 1 + 3) * 2
        self.assertEqual(int((~built["update_mask"]).sum()), ref_visual_rows)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), ref_audio_rows)
        self.assertTrue((built["input_ids"][5:9] == MINIMAX_H3_IMGVID_COND_ID).all())
        self.assertTrue((built["input_ids"][9:13] == MINIMAX_H3_AUDIO_REF_COND_ID).all())
        self.assertTrue((built["input_ids"][13:21] == MINIMAX_H3_IMGVID_COND_ID).all())
        self.assertTrue((built["input_ids"][21:23] == MINIMAX_H3_AUDIO_REF_COND_ID).all())


if __name__ == "__main__":
    unittest.main()
