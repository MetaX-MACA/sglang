# SPDX-License-Identifier: Apache-2.0
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.decoding import (
    MiniMaxH3DecodingStage,
    _canonical_output_audio_waveform,
    _canonical_visual_video_frames,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    MiniMaxH3EulerAncestralEta0SchedulerAdapter,
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SUPPORTED_FPS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    MiniMaxH3ShapePlanner,
    minimax_h3_align_frame_count,
    minimax_h3_frame_count_from_video_latent_t,
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


class _FakeVisualProcessor:
    def revert_tensor(self, tensor):
        return tensor[:, :3] + 1.0


class _FakeVisualModel:
    def __init__(self):
        self.processor = _FakeVisualProcessor()


class _FakeVideoVAE:
    def __init__(self, *, flatten_video_decode: bool = False):
        self.model = _FakeVisualModel()
        self.flatten_video_decode = flatten_video_decode
        self.decode_args = None
        self.decode_latent = None
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def decode_base(self, latent, *, frame_num=None, process_image=False):
        self.decode_args = (frame_num, process_image)
        self.decode_latent = latent.detach().clone()
        if process_image:
            if frame_num is None:
                raise ValueError("frame_num is required for fake image decode")
            return torch.zeros(
                (
                    int(latent.shape[0]),
                    int(latent.shape[1]),
                    int(frame_num),
                    int(latent.shape[3]),
                    int(latent.shape[4]),
                ),
                dtype=latent.dtype,
                device=latent.device,
            )
        frame_count = minimax_h3_frame_count_from_video_latent_t(int(latent.shape[2]))
        decoded = torch.zeros(
            (
                int(latent.shape[0]),
                int(latent.shape[1]),
                frame_count,
                int(latent.shape[3]),
                int(latent.shape[4]),
            ),
            dtype=latent.dtype,
            device=latent.device,
        )
        if self.flatten_video_decode:
            decoded = decoded.transpose(1, 2).reshape(
                int(latent.shape[0]) * frame_count,
                int(latent.shape[1]),
                int(latent.shape[3]),
                int(latent.shape[4]),
            )
        return decoded


class _FakeAudioVAE:
    sample_rate = 32000

    def __init__(self):
        self.decode_shape = None
        self.decode_latent = None
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def decode(self, latent):
        self.decode_shape = tuple(latent.shape)
        self.decode_latent = latent.detach().clone()
        return latent[:, :1, :]


class _FakeDenoisingTransformer:
    def __init__(self):
        self.calls = []

    def forward_denoising_step(
        self,
        *,
        input_visual_latent,
        input_audio_latent,
        timestep,
        visual_delta,
        audio_delta,
    ):
        self.calls.append(
            {
                "input_visual_latent": input_visual_latent,
                "input_audio_latent": input_audio_latent,
                "timestep": timestep,
                "visual_delta": visual_delta,
                "audio_delta": audio_delta,
            }
        )
        return {
            "noise_pred_visual": input_visual_latent + visual_delta,
            "noise_pred_audio": input_audio_latent + audio_delta,
        }


class _FakeDenoisingScheduler:
    def __init__(self):
        self.calls = []

    def step_denoising(
        self,
        *,
        input_visual_latent,
        input_audio_latent,
        timestep,
        noise_pred_visual,
        noise_pred_audio,
        step_scale,
    ):
        self.calls.append(
            {
                "input_visual_latent": input_visual_latent,
                "input_audio_latent": input_audio_latent,
                "timestep": timestep,
                "noise_pred_visual": noise_pred_visual,
                "noise_pred_audio": noise_pred_audio,
                "step_scale": step_scale,
            }
        )
        return {
            "output_visual_latent": input_visual_latent
            + noise_pred_visual * step_scale,
            "output_audio_latent": input_audio_latent + noise_pred_audio * step_scale,
        }


class _FakeDirectQwen3VLEncoder:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def encode_qwen3vl_encoder_inputs(
        self,
        encoder_input,
        *,
        runtime_config,
        tokenizer,
        batch,
        server_args,
    ):
        self.calls.append(
            {
                "encoder_input": encoder_input,
                "runtime_config": runtime_config,
                "tokenizer": tokenizer,
                "batch": batch,
                "server_args": server_args,
            }
        )
        return self.output


def _make_unit_server_args(*, disable_autocast=False):
    dit_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        boundary_ratio=None,
        arch_config=SimpleNamespace(in_channels=16, patch_size=2),
    )
    vae_config = SimpleNamespace(
        vae_tiling=False,
        arch_config=SimpleNamespace(
            vae_scale_factor=8,
            spatial_compression_ratio=8,
            z_dim=16,
            latent_channels=24,
            scale_factor_spatial=8,
            latents_mean=[0.0] * 24,
            latents_std=[1.0] * 24,
        ),
        get_vae_scale_factor=lambda: 8,
    )
    audio_vae_config = SimpleNamespace(
        arch_config=SimpleNamespace(
            latent_channels=32,
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        )
    )
    pipeline_config = SimpleNamespace(
        dit_config=dit_config,
        vae_config=vae_config,
        audio_vae_config=audio_vae_config,
        flow_shift=None,
        dit_precision="bf16",
        vae_precision="fp32",
        audio_vae_precision="fp32",
        get_latent_dtype=lambda dtype: dtype,
    )
    return SimpleNamespace(
        attention_backend=None,
        attention_backend_config=None,
        comfyui_mode=False,
        disable_autocast=disable_autocast,
        enable_cfg_parallel=False,
        enable_layerwise_nvtx_marker=False,
        enable_torch_compile=False,
        model_loaded={},
        model_paths={},
        pipeline_config=pipeline_config,
    )


class TestMiniMaxH3Contract(unittest.TestCase):
    def setUp(self):
        self._previous_global_server_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())

    def tearDown(self):
        set_global_server_args(self._previous_global_server_args)

    def test_audio_path_rejected_by_default_pipeline(self):
        params = SamplingParams(audio_path="input.wav")

        with self.assertRaisesRegex(ValueError, "audio_path is not supported"):
            params._validate_with_pipeline_config(
                PipelineConfig(task_type=ModelTaskType.T2I)
            )

    def test_minimax_h3_accepts_audio_path(self):
        params = MiniMaxH3SamplingParams(audio_path="input.wav")

        params._validate_with_pipeline_config(MiniMaxH3PipelineConfig())

    def test_minimax_h3_defaults_match_service_contract(self):
        params = MiniMaxH3SamplingParams()
        config = MiniMaxH3PipelineConfig()

        self.assertEqual(
            (params.height, params.fps, params.num_inference_steps), (512, 24, 50)
        )
        self.assertTrue(config.accepts_audio_input())
        self.assertEqual(config.vae_config.arch_config.latent_channels, 24)
        self.assertEqual(config.audio_vae_config.arch_config.latent_channels, 32)
        self.assertIsNone(config.vae_config.arch_config.latents_mean)
        self.assertIsNone(config.vae_config.arch_config.latents_std)
        self.assertIsNone(config.audio_vae_config.arch_config.latents_mean)
        self.assertIsNone(config.audio_vae_config.arch_config.latents_std)

    def test_minimax_h3_pipeline_wires_runtime_stage_order(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        output_video_vae = object()
        pipeline.modules = {
            "processor": object(),
            "video_vae": output_video_vae,
            "audio_vae": object(),
        }
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        pipeline._disagg_role = RoleType.MONOLITHIC

        server_args = _make_unit_server_args()
        pipeline.create_pipeline_stages(server_args)

        self.assertEqual(
            [stage.__class__.__name__ for stage in pipeline.stages],
            [
                "InputValidationStage",
                "MiniMaxH3TextEncodingStage",
                "MiniMaxH3VisualEncodingStage",
                "MiniMaxH3AudioEncodingStage",
                "MiniMaxH3LatentPreparationStage",
                "MiniMaxH3TimestepPreparationStage",
                "MiniMaxH3DenoisingStage",
                "MiniMaxH3DecodingStage",
            ],
        )
        self.assertEqual(
            MiniMaxH3Pipeline._required_config_modules,
            [
                "processor",
                "text_encoder",
                "tokenizer",
                "video_vae",
                "audio_vae",
                # scheduler intentionally absent: model_index carries
                # scheduler=null; sigma schedules are generated in
                # TimestepPreparation from the task profile.
                "transformer",
            ],
        )
        visual_stage = pipeline.stages[2]
        audio_stage = pipeline.stages[3]
        decode_stage = pipeline.stages[-1]
        self.assertIs(visual_stage.video_vae, output_video_vae)
        self.assertIs(
            visual_stage.vae_arch_config,
            server_args.pipeline_config.vae_config.arch_config,
        )
        self.assertIs(
            audio_stage.vae_arch_config,
            server_args.pipeline_config.audio_vae_config.arch_config,
        )
        self.assertIs(decode_stage.video_vae, output_video_vae)

    def test_minimax_h3_rejects_every_disaggregation_role_at_startup(self):
        for role in (
            RoleType.ENCODER,
            RoleType.DENOISER,
            RoleType.DECODER,
            RoleType.SERVER,
        ):
            with self.subTest(role=role):
                server_args = _make_unit_server_args()
                server_args.disagg_role = role
                with self.assertRaisesRegex(ValueError, "monolithic deployment"):
                    MiniMaxH3Pipeline("/unused", server_args)

        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.validate_disagg_role(RoleType.MONOLITHIC)

    def test_minimax_h3_server_args_reject_disaggregation_before_launch(self):
        for role in (
            RoleType.ENCODER,
            RoleType.DENOISER,
            RoleType.DECODER,
            RoleType.SERVER,
        ):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, "monolithic deployment"):
                    ServerArgs(
                        model_path="/unused",
                        pipeline_config=MiniMaxH3PipelineConfig(),
                        disagg_role=role,
                    )

        server_args = ServerArgs.__new__(ServerArgs)
        server_args.pipeline_config = MiniMaxH3PipelineConfig()
        server_args.disagg_role = RoleType.MONOLITHIC
        server_args._validate_disagg_capability()

    def test_load_modules_skips_null_component_before_unpacking(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.model_path = "/tmp/minimax_h3-legacy-model"
        pipeline._disagg_role = RoleType.MONOLITHIC
        pipeline._required_config_modules = [
            "text_encoder",
            "tokenizer",
            "optional_probe",
            "video_vae",
            "audio_vae",
            "transformer",
        ]
        pipeline._extra_config_module_map = {}
        pipeline.memory_usages = {}
        model_index = {
            "_class_name": "MiniMaxH3Pipeline",
            "_diffusers_version": "0.0.0",
            "text_encoder": ["transformers", "TextEncoder"],
            "tokenizer": ["transformers", "Tokenizer"],
            "optional_probe": None,
            "video_vae": ["diffusers", "VisualVAE"],
            "audio_vae": ["diffusers", "AudioVAE"],
            "transformer": ["diffusers", "Transformer"],
        }
        preloaded = {
            name: object()
            for name in (
                "text_encoder",
                "tokenizer",
                "video_vae",
                "audio_vae",
                "transformer",
            )
        }

        with mock.patch.object(pipeline, "_load_config", return_value=model_index):
            loaded = pipeline.load_modules(
                _make_unit_server_args(), loaded_modules=preloaded
            )

        self.assertEqual(loaded, preloaded)
        self.assertNotIn("optional_probe", pipeline.required_config_modules)

    def test_minimax_h3_decoder_rejects_unsupported_task(self):
        stage = MiniMaxH3DecodingStage(_FakeVideoVAE(), _FakeAudioVAE())
        batch = Req(
            sampling_params=SamplingParams(),
            latents=torch.zeros((1, 24, 1, 8, 8)),
            audio_latents=torch.zeros((2, 32, 64)),
        )
        batch.extra["minimax_h3_canonical_request"] = {"task": "foo2va"}

        with self.assertRaisesRegex(ValueError, "unsupported MiniMax H3 decoder task"):
            stage.forward(batch, _make_unit_server_args(disable_autocast=True))

    def test_minimax_h3_decoding_stage_keeps_tasks_on_generic_decoder(self):
        for task in ("t2va", "fl2va", "ref2va"):
            with self.subTest(task=task):
                output_decoder = _FakeVideoVAE()
                stage = MiniMaxH3DecodingStage(output_decoder, _FakeAudioVAE())
                batch = Req(
                    sampling_params=SamplingParams(),
                    latents=torch.zeros((1, 24, 1, 8, 8)),
                    audio_latents=torch.zeros((2, 32, 64)),
                )
                batch.extra["minimax_h3_canonical_request"] = {"task": task}

                output = stage.forward(
                    batch, _make_unit_server_args(disable_autocast=True)
                )

                self.assertEqual(output_decoder.eval_calls, 1)
                self.assertEqual(output_decoder.decode_args, (None, False))

    def test_minimax_h3_decoding_stage_decodes_vae_outputs(self):
        video_vae = _FakeVideoVAE()
        audio_vae = _FakeAudioVAE()
        stage = MiniMaxH3DecodingStage(video_vae, audio_vae)
        batch = Req(
            sampling_params=SamplingParams(),
            latents=torch.zeros((1, 24, 1, 8, 8), dtype=torch.float32),
            audio_latents=torch.zeros((2, 32, 64), dtype=torch.float32),
        )

        output = stage.forward(batch, _make_unit_server_args(disable_autocast=True))

        self.assertEqual(video_vae.decode_args, (None, False))
        self.assertEqual(video_vae.eval_calls, 1)
        self.assertEqual(audio_vae.decode_shape, (2, 32, 64))
        self.assertEqual(audio_vae.eval_calls, 1)
        self.assertTrue(torch.equal(output.output, torch.ones((1, 3, 1, 8, 8))))
        # The audio VAE's raw [C, 1, L] shape stays in evidence, while the output
        # boundary exposes explicit [1, C, L] for per-sample selection.
        self.assertEqual(tuple(output.audio.shape), (1, 2, 64))
        self.assertEqual(output.audio_sample_rate, 32000)

    def test_minimax_h3_decoding_stage_decodes_full_latent_t_to_frame_count(self):
        video_vae = _FakeVideoVAE()
        audio_vae = _FakeAudioVAE()
        stage = MiniMaxH3DecodingStage(video_vae, audio_vae)
        batch = Req(
            sampling_params=SamplingParams(),
            latents=torch.zeros((1, 24, 62, 8, 8), dtype=torch.float32),
            audio_latents=torch.zeros((2, 32, 348), dtype=torch.float32),
        )

        output = stage.forward(batch, _make_unit_server_args(disable_autocast=True))

        self.assertEqual(minimax_h3_frame_count_from_video_latent_t(62), 209)
        self.assertEqual(video_vae.decode_args, (None, False))
        self.assertEqual(tuple(output.output.shape), (1, 3, 209, 8, 8))

    def test_minimax_h3_decoding_stage_accepts_flat_video_decoder_output(self):
        video_vae = _FakeVideoVAE(flatten_video_decode=True)
        audio_vae = _FakeAudioVAE()
        stage = MiniMaxH3DecodingStage(video_vae, audio_vae)
        batch = Req(
            sampling_params=SamplingParams(),
            latents=torch.zeros((1, 24, 2, 8, 8), dtype=torch.float32),
            audio_latents=torch.zeros((2, 32, 64), dtype=torch.float32),
        )

        output = stage.forward(batch, _make_unit_server_args(disable_autocast=True))

        self.assertEqual(video_vae.decode_args, (None, False))
        self.assertEqual(tuple(output.output.shape), (1, 3, 5, 8, 8))

    def test_minimax_h3_output_audio_rejects_audio_vae_batch_mismatch(self):
        raw_audio = torch.zeros((2, 2, 64), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, r"must have shape \[C, 1, L\]"):
            _canonical_output_audio_waveform(raw_audio, batch_size=1)

    def test_minimax_h3_output_audio_rejects_multiple_visual_samples(self):
        raw_audio = torch.zeros((2, 1, 64), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "only supports one generated sample"):
            _canonical_output_audio_waveform(raw_audio, batch_size=2)

    def test_minimax_h3_output_audio_canonicalizes_audio_vae_stereo_channels(self):
        raw_audio = torch.arange(2 * 1 * 4, dtype=torch.float32).reshape(2, 1, 4)

        output_audio = _canonical_output_audio_waveform(raw_audio, batch_size=1)

        self.assertEqual(tuple(output_audio.shape), (1, 2, 4))
        self.assertTrue(torch.equal(output_audio[0, 0], raw_audio[0, 0]))
        self.assertTrue(torch.equal(output_audio[0, 1], raw_audio[1, 0]))

    def test_minimax_h3_flat_visual_decode_preserves_multi_batch_projection(self):
        flat_frames = torch.arange(2 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(
            2 * 5, 3, 2, 2
        )

        output_frames = _canonical_visual_video_frames(flat_frames, batch_size=2)

        self.assertEqual(tuple(output_frames.shape), (2, 3, 5, 2, 2))
        self.assertTrue(torch.equal(output_frames[0, :, 0], flat_frames[0]))
        self.assertTrue(torch.equal(output_frames[1, :, 0], flat_frames[5]))

    def test_minimax_h3_decoding_stage_rejects_missing_latents(self):
        stage = MiniMaxH3DecodingStage(_FakeVideoVAE(), _FakeAudioVAE())
        batch = Req(
            sampling_params=SamplingParams(),
            audio_latents=torch.zeros((2, 32, 64), dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "batch.latents"):
            stage.forward(batch, _make_unit_server_args(disable_autocast=True))

    def test_minimax_h3_decoding_stage_reverse_normalizes_latents(self):
        video_vae = _FakeVideoVAE()
        audio_vae = _FakeAudioVAE()
        stage = MiniMaxH3DecodingStage(video_vae, audio_vae)
        server_args = _make_unit_server_args(disable_autocast=True)
        server_args.pipeline_config.vae_config.arch_config.latents_mean = [2.0] * 24
        server_args.pipeline_config.vae_config.arch_config.latents_std = [3.0] * 24
        server_args.pipeline_config.audio_vae_config.arch_config.latents_mean = [
            5.0
        ] * 32
        server_args.pipeline_config.audio_vae_config.arch_config.latents_std = [
            7.0
        ] * 32
        batch = Req(
            sampling_params=SamplingParams(),
            latents=torch.ones((1, 24, 1, 2, 2), dtype=torch.float32),
            audio_latents=torch.ones((2, 32, 4), dtype=torch.float32),
        )

        stage.forward(batch, server_args)

        self.assertTrue(
            torch.equal(video_vae.decode_latent, torch.full_like(batch.latents, 5.0))
        )
        self.assertTrue(
            torch.equal(
                audio_vae.decode_latent,
                torch.full_like(batch.audio_latents, 12.0),
            )
        )

    def test_minimax_h3_video_packed_token_roundtrip(self):
        latent = torch.arange(2 * 3 * 2 * 4 * 6, dtype=torch.float32).reshape(
            2,
            3,
            2,
            4,
            6,
        )
        rows = minimax_h3_patchify_video_latent(latent, patch_size=[1, 2, 3])

        self.assertEqual(tuple(rows.shape), (16, 18))
        self.assertTrue(
            torch.equal(
                minimax_h3_unpatchify_video_tokens(
                    rows,
                    latent_shape=[2, 2, 2, 3],
                    patch_size=[1, 2, 3],
                ),
                latent,
            )
        )

    def test_minimax_h3_audio_packed_token_roundtrip(self):
        latent = torch.arange(2 * 32 * 4, dtype=torch.float32).reshape(2, 32, 4)
        rows = minimax_h3_pack_audio_latent(latent)

        self.assertEqual(tuple(rows.shape), (8, 32))
        self.assertTrue(
            torch.equal(
                minimax_h3_unpack_audio_tokens(
                    rows,
                    audio_t=8,
                    audio_channel=2,
                ),
                latent,
            )
        )

    def test_minimax_h3_packed_token_order_regression_slice(self):
        # Sparse regression-pin values for the packed condition rows. The
        # (coordinate, column) pairs follow the current SGLang convention.
        video_samples = [
            (0, 0, 0, 0, 0.4176744222640991),
            (0, 0, 1, 1, 0.576472818851471),
            (0, 1, 0, 2, -0.16939601302146912),
            (0, 1, 1, 3, -1.836861491203308),
            (1, 0, 0, 4, 1.6779203414916992),
            (1, 0, 1, 5, 0.7319522500038147),
            (5, 1, 0, 22, -0.8070425987243652),
            (5, 1, 1, 23, 0.8907957673072815),
            (6, 0, 0, 24, -0.9199880957603455),
            (6, 0, 1, 25, 0.594915509223938),
            (11, 1, 0, 46, 0.3619807958602905),
            (11, 1, 1, 47, 0.46610361337661743),
            (12, 0, 0, 48, -0.19812093675136566),
            (12, 0, 1, 49, -2.2618699073791504),
            (17, 1, 0, 70, -0.29397493600845337),
            (17, 1, 1, 71, -0.10580660402774811),
            (18, 0, 0, 72, 1.852187991142273),
            (18, 0, 1, 73, 0.043832600116729736),
            (23, 1, 0, 94, -1.129878044128418),
            (23, 1, 1, 95, -0.9405509233474731),
        ]
        video_latent = torch.zeros((1, 24, 1, 2, 2), dtype=torch.float32)
        for channel, ph, pw, _, value in video_samples:
            video_latent[0, channel, 0, ph, pw] = value

        video_rows = minimax_h3_patchify_video_latent(
            video_latent,
            patch_size=[1, 2, 2],
        )

        self.assertEqual(tuple(video_rows.shape), (1, 96))
        for _, _, _, column, value in video_samples:
            self.assertTrue(
                torch.equal(
                    video_rows[0, column],
                    torch.tensor(value, dtype=torch.float32),
                )
            )

        audio_samples = [
            (0, 0, 0, 0, 0.4176744222640991),
            (0, 0, 1, 0, 0.576472818851471),
            (0, 0, 2, 0, 1.6964287757873535),
            (0, 0, 3, 0, 0.04151688143610954),
            (0, 0, 4, 0, 2.0606374740600586),
            (0, 0, 5, 0, 0.7819227576255798),
            (0, 0, 30, 0, -0.2627560794353485),
            (0, 0, 31, 0, 0.9887282848358154),
            (0, 1, 0, 1, -0.3538547456264496),
            (0, 1, 1, 1, 1.0957027673721313),
            (0, 1, 2, 1, -0.10468527674674988),
            (0, 1, 3, 1, -0.22903171181678772),
            (0, 1, 4, 1, -0.32040369510650635),
            (0, 1, 5, 1, 1.276181936264038),
            (0, 1, 30, 1, -1.0377836227416992),
            (0, 1, 31, 1, 1.005703330039978),
            (1, 0, 0, 2, 0.8394279479980469),
            (1, 0, 1, 2, 0.6378944516181946),
            (1, 0, 2, 2, -0.36443907022476196),
            (1, 0, 3, 2, 1.2430633306503296),
            (1, 0, 4, 2, -0.3817988336086273),
            (1, 0, 5, 2, -0.45278555154800415),
            (1, 0, 30, 2, -0.30015870928764343),
            (1, 0, 31, 2, 1.2217679023742676),
            (1, 1, 0, 3, 0.6834432482719421),
            (1, 1, 1, 3, -0.8571628332138062),
            (1, 1, 2, 3, -1.588662028312683),
            (1, 1, 3, 3, 0.07421813160181046),
            (1, 1, 4, 3, -1.608457088470459),
            (1, 1, 5, 3, -0.06989705562591553),
            (1, 1, 30, 3, 0.6673216819763184),
            (1, 1, 31, 3, 1.1349895000457764),
        ]
        audio_latent = torch.zeros((2, 32, 2), dtype=torch.float32)
        for channel, step, latent_channel, _, value in audio_samples:
            audio_latent[channel, latent_channel, step] = value

        audio_rows = minimax_h3_pack_audio_latent(audio_latent)

        self.assertEqual(tuple(audio_rows.shape), (4, 32))
        for _, _, latent_channel, row, value in audio_samples:
            self.assertTrue(
                torch.equal(
                    audio_rows[row, latent_channel],
                    torch.tensor(value, dtype=torch.float32),
                )
            )

    def test_minimax_h3_external_dit_transformer_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp)
            marker_path = component_dir / "external-module-executed"
            (component_dir / "fake_external_dit.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "Path(__file__).with_name('external-module-executed').write_text('executed')",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (component_dir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3ExternalDiTTransformer",
                        "_diffusers_version": "0.32.2",
                        "external_module": "fake_external_dit.py",
                        "external_factory": "build_minimax_h3_dit",
                        "patch_size": [1, 1, 1],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            original_sys_path = list(sys.path)
            for trust_remote_code in (False, True):
                with self.subTest(trust_remote_code=trust_remote_code):
                    server_args = _make_unit_server_args()
                    server_args.pipeline_config = MiniMaxH3PipelineConfig()
                    server_args.trust_remote_code = trust_remote_code
                    with mock.patch(
                        "sglang.multimodal_gen.runtime.loader.component_loaders."
                        "transformer_loader.resolve_transformer_safetensors_to_load",
                        return_value=[str(component_dir / "unused.safetensors")],
                    ):
                        with mock.patch(
                            "sglang.multimodal_gen.runtime.loader.component_loaders."
                            "transformer_loader.ModelRegistry.resolve_model_cls",
                            side_effect=RuntimeError(
                                "normal transformer registry reached"
                            ),
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError, "normal transformer registry reached"
                            ):
                                TransformerLoader().load_customized(
                                    str(component_dir),
                                    server_args,
                                    "transformer",
                                )
                    self.assertFalse(marker_path.exists())
                    self.assertEqual(sys.path, original_sys_path)

    def test_minimax_h3_scheduler_loader_builds_eta0_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp)
            (component_dir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3EulerAncestralEta0SchedulerAdapter",
                        "_diffusers_version": "0.32.2",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            scheduler = SchedulerLoader().load_customized(
                str(component_dir),
                _make_unit_server_args(),
            )

            self.assertIsInstance(scheduler, MiniMaxH3EulerAncestralEta0SchedulerAdapter)

    def test_minimax_h3_scheduler_loader_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            component_dir = Path(tmp)
            (component_dir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3EulerAncestralEta0SchedulerAdapter",
                        "_diffusers_version": "0.32.2",
                        "unexpected": True,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not accept config fields"):
                SchedulerLoader().load_customized(
                    str(component_dir),
                    _make_unit_server_args(),
                )

    def test_minimax_h3_euler_eta0_scheduler_adapter_matches_rectified_flow_formula(self):
        scheduler = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
        visual_latent = torch.zeros((1, 24, 1, 2, 2), dtype=torch.float32)
        audio_latent = torch.ones((2, 32, 2), dtype=torch.float32)
        visual_v = torch.full_like(visual_latent, 2.0)
        audio_v = torch.full_like(audio_latent, 4.0)
        timestep = torch.tensor([0.5], dtype=torch.float32)

        visual_x0 = minimax_h3_rf_v_to_x0(visual_latent, visual_v, timestep)
        audio_x0 = minimax_h3_rf_v_to_x0(audio_latent, audio_v, timestep)
        self.assertTrue(torch.equal(visual_x0, visual_latent + visual_v * 0.5))
        self.assertTrue(torch.equal(audio_x0, audio_latent + audio_v * 0.5))
        self.assertTrue(
            torch.equal(
                minimax_h3_euler_eta0_step(
                    visual_latent,
                    visual_x0,
                    sigma_curr=0.5,
                    sigma_next=0.25,
                ),
                visual_latent + visual_v * 0.25,
            )
        )

        out = scheduler.step_denoising(
            input_visual_latent=visual_latent,
            input_audio_latent=audio_latent,
            timestep=timestep,
            noise_pred_visual=visual_v,
            noise_pred_audio=audio_v,
            sigma_curr=0.5,
            sigma_next=0.25,
            audio_timestep=torch.tensor([0.25], dtype=torch.float32),
            audio_sigma_curr=0.75,
            audio_sigma_next=0.5,
        )

        self.assertTrue(
            torch.equal(out["output_visual_latent"], visual_latent + visual_v * 0.25)
        )
        self.assertTrue(
            torch.equal(out["output_audio_latent"], audio_latent + audio_v * 0.25)
        )

    def test_minimax_h3_rf_v_to_x0_rejects_invalid_inputs(self):
        latent = torch.zeros((1, 2), dtype=torch.float32)
        velocity = torch.ones_like(latent)

        bad_velocity = velocity.clone()
        bad_velocity[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "v must be finite"):
            minimax_h3_rf_v_to_x0(
                latent,
                bad_velocity,
                torch.tensor([1.0], dtype=torch.float32),
            )

        with self.assertRaisesRegex(ValueError, r"timestep must be in \[0, 1\]"):
            minimax_h3_rf_v_to_x0(
                latent,
                velocity,
                torch.tensor([1.25], dtype=torch.float32),
            )

    def test_minimax_h3_euler_eta0_step_preserves_half_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            state = torch.zeros((4,), dtype=dtype)
            denoised = torch.ones_like(state)

            out = minimax_h3_euler_eta0_step(
                state,
                denoised,
                sigma_curr=0.5,
                sigma_next=0.25,
            )

            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.equal(out, torch.full_like(state, 0.5)))

    def test_minimax_h3_euler_eta0_step_rejects_invalid_inputs(self):
        state = torch.zeros((2,), dtype=torch.float32)
        denoised = torch.ones_like(state)

        bad_state = state.clone()
        bad_state[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "state must be finite"):
            minimax_h3_euler_eta0_step(
                bad_state,
                denoised,
                sigma_curr=0.5,
                sigma_next=0.25,
            )

        with self.assertRaisesRegex(ValueError, "sigma_curr must be non-negative"):
            minimax_h3_euler_eta0_step(
                state,
                denoised,
                sigma_curr=-0.5,
                sigma_next=0.25,
            )

        with self.assertRaisesRegex(
            ValueError,
            "sigma_next must be 0 when sigma_curr is 0",
        ):
            minimax_h3_euler_eta0_step(
                state,
                denoised,
                sigma_curr=0.0,
                sigma_next=0.25,
            )

    def test_minimax_h3_scheduler_rejects_timestep_sigma_mismatch(self):
        scheduler = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
        latent = torch.zeros((1, 2), dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError,
            "video_sigma_curr must equal 1 - video_timestep",
        ):
            scheduler.step_denoising(
                input_visual_latent=latent,
                input_audio_latent=latent,
                timestep=torch.tensor([0.5], dtype=torch.float32),
                noise_pred_visual=torch.ones_like(latent),
                noise_pred_audio=torch.ones_like(latent),
                sigma_curr=1.0,
                sigma_next=0.75,
            )

    def test_minimax_h3_target_shape_rules_match_serving_contract(self):
        frame_count = minimax_h3_align_frame_count(200)

        self.assertEqual(frame_count, 209)
        self.assertEqual(minimax_h3_frame_count_from_video_latent_t(57), 192)
        self.assertEqual(minimax_h3_frame_count_from_video_latent_t(62), 209)
        planner = MiniMaxH3ShapePlanner()
        self.assertEqual(planner.video_latent_t(frame_count), 62)
        self.assertEqual(
            planner.audio_latent_t(frame_count / MINIMAX_H3_SUPPORTED_FPS), 348
        )

    def test_minimax_h3_time_shift_sigmas_match_serving_contract(self):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=3, shift_scale=2.0)

        self.assertEqual(len(sigmas), 3)
        self.assertAlmostEqual(sigmas[0], 1.0)
        self.assertAlmostEqual(sigmas[1], 2.0 / 3.0)
        self.assertAlmostEqual(sigmas[2], 0.0)


if __name__ == "__main__":
    unittest.main()
