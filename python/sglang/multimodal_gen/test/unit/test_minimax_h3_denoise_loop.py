# SPDX-License-Identifier: Apache-2.0
"""Hermetic contracts for the cfg-distilled MiniMax H3 denoise loop."""

import unittest
from types import SimpleNamespace

import torch


class _RecordingModel:
    def __init__(self):
        self.img_counts = []
        self.timesteps = []
        self.video_inputs = []

    def __call__(self, **kwargs):
        img_count = int(kwargs["img_pos_info"]["position_ids"].shape[0])
        audio_count = int(kwargs["audio_pos_info"]["position_ids"].shape[0])
        self.img_counts.append(img_count)
        self.timesteps.append(
            kwargs["unique_timesteps"][kwargs["inverse_indices"]].detach().cpu()
        )
        self.video_inputs.append(
            kwargs["x"][0]
            .index_select(0, kwargs["img_pos_info"]["position_ids"])
            .detach()
            .cpu()
        )
        device = kwargs["x"].device
        return (
            torch.zeros(img_count, 96, device=device),
            torch.zeros(audio_count, 32, device=device),
        )


class TestMiniMaxH3DenoiseLoopCpu(unittest.TestCase):
    def test_fl2va_dit_sink_rejects_stale_keyframe_payloads(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoising import (
            _validate_fl2va_keyframe_payload,
        )

        plan = SimpleNamespace(task="fl2va")
        valid = {
            "rows": torch.zeros(8, 96),
            "latent_h": 4,
            "latent_w": 4,
            "semantic_frame_indices": [0, -1],
            "pixel_frame_indices": [0, 4],
            "frame_count": 5,
            "keyframes": [
                {"frame_index": 0, "resolved_frame_index": 0},
                {"frame_index": -1, "resolved_frame_index": 4},
            ],
        }
        _validate_fl2va_keyframe_payload(plan, valid)

        for semantic_index, pixel_index in ((0, 0), (-1, 4)):
            single = {
                "rows": torch.zeros(4, 96),
                "latent_h": 4,
                "latent_w": 4,
                "semantic_frame_indices": [semantic_index],
                "pixel_frame_indices": [pixel_index],
                "frame_count": 5,
                "keyframes": [
                    {
                        "frame_index": semantic_index,
                        "resolved_frame_index": pixel_index,
                    }
                ],
            }
            _validate_fl2va_keyframe_payload(plan, single)

        invalid = []
        middle = {**valid, "semantic_frame_indices": [0, 2, -1]}
        invalid.append((middle, "semantic_frame_indices"))
        reverse = {**valid, "semantic_frame_indices": [-1, 0]}
        invalid.append((reverse, "semantic_frame_indices"))
        wrong_pixel = {**valid, "pixel_frame_indices": [0, 3]}
        invalid.append((wrong_pixel, "pixel_frame_indices"))
        wrong_rows = {**valid, "rows": torch.zeros(4, 96)}
        invalid.append((wrong_rows, "rows do not match"))
        wrong_entries = {**valid, "keyframes": [{"frame_index": 0}]}
        invalid.append((wrong_entries, "one encoded keyframe per semantic anchor"))
        stale_resolved = {
            **valid,
            "keyframes": [
                {"frame_index": 0, "resolved_frame_index": 0},
                {"frame_index": -1, "resolved_frame_index": 3},
            ],
        }
        invalid.append((stale_resolved, "matching resolved_frame_index"))
        for payload, message in invalid:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                _validate_fl2va_keyframe_payload(plan, payload)

        with self.assertRaisesRegex(ValueError, "only valid.*fl2va"):
            _validate_fl2va_keyframe_payload(SimpleNamespace(task="t2va"), valid)

    def test_text_and_padding_timesteps_follow_video_step(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            MINIMAX_H3_PAD_ID,
            minimax_h3_packed_sequence,
        )

        packed = minimax_h3_packed_sequence(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            include_keyframe_cond=False,
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=packed["token_tags"],
            device=torch.device("cpu"),
        )
        video_t = 0.75
        audio_t = 0.625
        kwargs = branch.forward_kwargs(
            video_rows=torch.zeros(int(branch.img_pos.numel()), 96),
            audio_rows=torch.zeros(int(branch.audio_pos.numel()), 32),
            t_video=video_t,
            t_audio=audio_t,
            imgvid_cond_timestep=0.9,
            audio_ref_cond_timestep=1.0,
        )
        timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        padding_pos = torch.nonzero(
            packed["input_ids"] == MINIMAX_H3_PAD_ID, as_tuple=False
        ).view(-1)

        self.assertTrue(
            torch.allclose(
                timesteps[packed["text_pos"]],
                torch.full((int(packed["text_pos"].numel()),), video_t),
            )
        )
        self.assertTrue(
            torch.allclose(
                timesteps[padding_pos],
                torch.full((int(padding_pos.numel()),), video_t),
            )
        )
        self.assertTrue(
            torch.allclose(
                timesteps[branch.img_pos],
                torch.full((int(branch.img_pos.numel()),), video_t),
            )
        )
        self.assertTrue(
            torch.allclose(
                timesteps[branch.audio_pos],
                torch.full((int(branch.audio_pos.numel()),), audio_t),
            )
        )

    def test_positive_branch_keeps_keyframe_cond_rows(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence,
        )

        device = torch.device("cpu")
        packed = minimax_h3_packed_sequence(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=packed["token_tags"],
            device=device,
        )

        initial_video = torch.zeros(int(branch.img_pos.shape[0]), 96)
        initial_audio = torch.zeros(int(branch.audio_pos.shape[0]), 32)
        n_cond = int((~branch.update_mask).sum())
        cond_rows = torch.arange(n_cond * 96, dtype=torch.float32).reshape(n_cond, 96)
        model = _RecordingModel()

        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=cond_rows,
            sigmas_video=[1.0, 0.5, 0.0],
            sigmas_audio=[1.0, 0.5, 0.0],
            device=device,
        )

        self.assertEqual(model.img_counts, [int(branch.img_pos.shape[0])] * 2)
        self.assertEqual(len(model.video_inputs), 2)
        for video_input in model.video_inputs:
            self.assertTrue(torch.equal(video_input[:n_cond], cond_rows))

    def test_single_endpoint_keyframe_cond_rows_survive_each_step(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence,
        )

        for semantic_index in (0, -1):
            with self.subTest(semantic_index=semantic_index):
                packed = minimax_h3_packed_sequence(
                    text_len=3,
                    latent_t=2,
                    latent_h=4,
                    latent_w=4,
                    audio_t=2,
                    include_keyframe_cond=True,
                    keyframe_frame_indices=[semantic_index],
                    frame_count=5,
                )
                branch = MiniMaxH3DenoiseBranch(
                    packed=packed,
                    text_embeddings=torch.zeros(3, 5120),
                    token_tags=packed["token_tags"],
                    device=torch.device("cpu"),
                )
                n_cond = int((~branch.update_mask).sum())
                cond_rows = torch.arange(n_cond * 96, dtype=torch.float32).reshape(
                    n_cond, 96
                )
                model = _RecordingModel()

                minimax_h3_denoise_loop(
                    model=model,
                    positive=branch,
                    initial_video_rows=torch.zeros(int(branch.img_pos.shape[0]), 96),
                    initial_audio_rows=torch.zeros(int(branch.audio_pos.shape[0]), 32),
                    keyframe_cond_rows=cond_rows,
                    sigmas_video=[1.0, 0.5, 0.0],
                    sigmas_audio=[1.0, 0.5, 0.0],
                    device=torch.device("cpu"),
                )

                self.assertEqual(len(model.video_inputs), 2)
                for video_input in model.video_inputs:
                    self.assertTrue(torch.equal(video_input[:n_cond], cond_rows))

    def test_condition_timesteps_follow_noise_aug_floor(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence_ref2va_blocks,
        )

        device = torch.device("cpu")
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=packed["token_tags"],
            device=device,
        )
        initial_video = torch.zeros(int(branch.img_pos.shape[0]), 96)
        initial_audio = torch.zeros(int(branch.audio_pos.shape[0]), 32)
        cond_rows = torch.zeros(int((~branch.update_mask).sum()), 96)
        audio_ref_rows = torch.zeros(int((~branch.audio_update_mask).sum()), 32)
        model = _RecordingModel()

        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=cond_rows,
            audio_ref_rows=audio_ref_rows,
            sigmas_video=[1.0, 0.1, 0.0],
            sigmas_audio=[1.0, 0.2, 0.0],
            device=device,
            imgvid_cond_noise_aug_for_inference=0.6,
            audio_cond_noise_aug_for_inference=0.4,
        )

        video_cond_pos = branch.img_pos[~branch.update_mask]
        audio_ref_pos = branch.audio_pos[~branch.audio_update_mask]

        def expected(pos: torch.Tensor, value: float) -> torch.Tensor:
            return torch.full((int(pos.numel()),), value, dtype=torch.float32)

        step0 = model.timesteps[0]
        step1 = model.timesteps[1]
        self.assertTrue(
            torch.allclose(step0[video_cond_pos], expected(video_cond_pos, 0.6))
        )
        self.assertTrue(
            torch.allclose(step1[video_cond_pos], expected(video_cond_pos, 0.9))
        )
        self.assertTrue(
            torch.allclose(step0[audio_ref_pos], expected(audio_ref_pos, 0.4))
        )
        self.assertTrue(
            torch.allclose(step1[audio_ref_pos], expected(audio_ref_pos, 0.8))
        )

    def test_positive_path_forwards_once_per_step(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence,
        )

        device = torch.device("cpu")
        positive = minimax_h3_packed_sequence(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            include_keyframe_cond=False,
        )
        pos_branch = MiniMaxH3DenoiseBranch(
            packed=positive,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=positive["token_tags"],
            device=device,
        )

        n_video = int(pos_branch.img_pos.shape[0])
        n_audio = int(pos_branch.audio_pos.shape[0])
        initial_video = torch.zeros(n_video, 96)
        initial_audio = torch.zeros(n_audio, 32)
        num_steps = 3
        model = _RecordingModel()

        minimax_h3_denoise_loop(
            model=model,
            positive=pos_branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=None,
            sigmas_video=[1.0, 0.6, 0.3, 0.0],
            sigmas_audio=[1.0, 0.6, 0.3, 0.0],
            device=device,
        )

        self.assertEqual(len(model.img_counts), num_steps)
        self.assertTrue(all(c == n_video for c in model.img_counts))

    def test_step_profiler_context_wraps_every_step(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
            minimax_h3_packed_sequence,
        )

        device = torch.device("cpu")
        positive = minimax_h3_packed_sequence(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            include_keyframe_cond=False,
        )
        pos_branch = MiniMaxH3DenoiseBranch(
            packed=positive,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=positive["token_tags"],
            device=device,
        )
        initial_video = torch.zeros(int(pos_branch.img_pos.shape[0]), 96)
        initial_audio = torch.zeros(int(pos_branch.audio_pos.shape[0]), 32)

        events: list[tuple[str, int]] = []

        class _RecordingContext:
            def __init__(self, step_index: int):
                self.step_index = step_index

            def __enter__(self):
                events.append(("enter", self.step_index))
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                events.append(("exit", self.step_index))
                return False

        forward_steps: list[int] = []
        model = _RecordingModel()

        def counting_model(**kwargs):
            forward_steps.append(len([e for e in events if e[0] == "enter"]) - 1)
            return model(**kwargs)

        minimax_h3_denoise_loop(
            model=counting_model,
            positive=pos_branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=None,
            sigmas_video=[1.0, 0.6, 0.3, 0.0],
            sigmas_audio=[1.0, 0.6, 0.3, 0.0],
            device=device,
            step_profiler=_RecordingContext,
        )

        # one enter/exit pair per step, in order, and the model forward for
        # step i runs strictly inside context i
        self.assertEqual(
            events,
            [(kind, i) for i in range(3) for kind in ("enter", "exit")],
        )
        self.assertEqual(forward_steps, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
