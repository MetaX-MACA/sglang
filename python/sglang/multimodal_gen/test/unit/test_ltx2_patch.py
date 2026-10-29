import contextlib
import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.runtime.models.vaes import ltx_2_vae
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    denoising_av,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)


class TestLTX2GuidanceBranches(unittest.TestCase):
    def _run_step(self, *, do_cfg=True, env=None, cfg_scale=None, cfg_world_size=1):
        with patch.dict(os.environ, env or {}, clear=True):
            params = LTX23SamplingParams(
                video_rescale_scale=0.0, audio_rescale_scale=0.0
            )
            if cfg_scale is not None:
                params.video_cfg_scale = params.audio_cfg_scale = cfg_scale
            extra = params.build_request_extra()
            original_guider = dict(extra["ltx2_stage1_guider_params"])
            latents = torch.ones(1, 2, 3)
            batch = SimpleNamespace(
                extra=extra,
                do_classifier_free_guidance=do_cfg,
                prompt_embeds=[latents],
                audio_prompt_embeds=[latents],
                negative_prompt_embeds=[latents] if do_cfg else [],
                negative_audio_prompt_embeds=[latents] if do_cfg else [],
                ltx2_num_image_tokens=1,
            )
            ctx = SimpleNamespace(
                stage="stage1",
                latents=latents,
                audio_latents=latents.clone(),
                audio_scheduler=object(),
                scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.5])),
                is_ltx23_variant=True,
                use_ltx23_legacy_one_stage=False,
                use_ltx23_hq_timestep_semantics=False,
                denoise_mask=None,
                clean_latent=None,
                last_denoised_video=None,
                last_denoised_audio=None,
                reserved_frames_mask=None,
                z=None,
            )
            stage = object.__new__(LTX2DenoisingStage)
            args = SimpleNamespace(
                enable_cfg_parallel=cfg_world_size > 1,
                pipeline_class_name="LTX2TwoStagePipeline",
            )
            stage.server_args = args
            stage.sampler_name = "euler"
            stage._prepare_ltx2_model_inputs = lambda *a: SimpleNamespace(
                latent_model_input=ctx.latents
            )
            stage._build_ltx2_base_model_kwargs = lambda *a: {
                "hidden_states": ctx.latents,
                "audio_hidden_states": ctx.audio_latents,
            }
            stage._get_ltx_prompt_attention_mask = lambda *a, **kw: None
            stage._ltx2_model_forward_context = lambda *a: contextlib.nullcontext()
            stage._ltx2_call_current_model = lambda c, s, kw: s.current_model(**kw)
            stage.post_forward_for_ti2v_task = lambda b, s, m, x, z: x
            model = Mock(
                side_effect=lambda **kw: (
                    torch.full_like(kw["hidden_states"], 0.25),
                    torch.full_like(kw["audio_hidden_states"], 0.5),
                )
            )
            step = SimpleNamespace(step_index=0, current_model=model)
            with (
                patch(
                    LTX2DenoisingStage.__module__
                    + ".get_classifier_free_guidance_rank",
                    return_value=0,
                ),
                patch(
                    LTX2DenoisingStage.__module__
                    + ".get_classifier_free_guidance_world_size",
                    return_value=cfg_world_size,
                ),
            ):
                stage._run_denoising_step(ctx, step, batch, args)
            self.assertEqual(extra["ltx2_stage1_guider_params"], original_guider)
            self.assertTrue(torch.isfinite(ctx.latents).all())
            self.assertTrue(torch.isfinite(ctx.audio_latents).all())
            return model.call_count

    def test_default_keeps_four_guidance_forwards(self):
        self.assertEqual(self._run_step(), 4)

    def test_no_cfg_does_not_read_missing_negative_embeddings(self):
        self.assertEqual(self._run_step(do_cfg=False), 3)

    def test_positive_only_runs_one_forward(self):
        self.assertEqual(
            self._run_step(
                do_cfg=False,
                env={
                    "LTX2_DISABLE_NEG": "1",
                    "LTX2_DISABLE_STG": "true",
                    "LTX2_DISABLE_MODALITY": "ON",
                },
            ),
            1,
        )

    def test_unit_cfg_scale_skips_negative_forward(self):
        self.assertEqual(self._run_step(cfg_scale=1.0), 3)

    def test_negative_env_switch_skips_forward_with_cfg_enabled(self):
        self.assertEqual(self._run_step(env={"LTX2_DISABLE_NEG": "1"}), 3)

    def test_positive_only_rejects_excess_cfg_ranks(self):
        with self.assertRaisesRegex(ValueError, "exceeds guidance pass count"):
            self._run_step(
                cfg_world_size=2,
                env={
                    "LTX2_DISABLE_NEG": "1",
                    "LTX2_DISABLE_STG": "1",
                    "LTX2_DISABLE_MODALITY": "1",
                },
            )


class TestLTX2RefinementGroups(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(enable_cfg_parallel=True)
        self.sp = SimpleNamespace(
            ranks=[3],
            world_size=1,
            rank_in_group=0,
            device_group=object(),
            cpu_group=object(),
            ulysses_group=object(),
            ulysses_world_size=1,
            ulysses_rank=0,
        )
        self.cfg = SimpleNamespace(
            ranks=[2, 3],
            world_size=2,
            rank_in_group=1,
            device_group=object(),
            cpu_group=object(),
        )
        self.dec = SimpleNamespace(**vars(self.cfg))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name, value in (
            ("_REFINEMENT_SP_OVER_CFG", True),
            ("model_parallel_is_initialized", lambda: True),
            ("get_sp_group", lambda: self.sp),
            ("get_cfg_group", lambda: self.cfg),
            ("get_decode_parallel_group_coordinator", lambda: self.dec),
        ):
            stack.enter_context(patch.object(denoising_av, name, value))

    def test_group_state_is_restored_after_exception(self):
        sp_saved, cfg_saved = dict(vars(self.sp)), dict(vars(self.cfg))
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with denoising_av._sp_over_cfg_group(self.args):
                self.assertEqual(self.sp.ranks, [2, 3])
                self.assertEqual(self.sp.ulysses_world_size, 2)
                self.assertEqual(self.sp.ulysses_rank, 1)
                self.assertEqual(self.cfg.ranks, [3])
                self.assertEqual(self.cfg.world_size, 1)
                self.assertFalse(self.args.enable_cfg_parallel)
                raise RuntimeError("inference failed")
        self.assertEqual(vars(self.sp), sp_saved)
        self.assertEqual(vars(self.cfg), cfg_saved)
        self.assertTrue(self.args.enable_cfg_parallel)

    def test_existing_sp_topology_is_preserved(self):
        self.sp.world_size = 2
        self.assertFalse(denoising_av._can_use_refinement_sp_over_cfg(self.args))
        with denoising_av._sp_over_cfg_group(self.args):
            self.assertTrue(self.args.enable_cfg_parallel)
            self.assertEqual(self.cfg.world_size, 2)

    def test_decode_group_must_match_cfg_rank_order(self):
        self.dec.ranks = [3, 2]
        self.assertFalse(denoising_av._can_use_refinement_sp_over_cfg(self.args))


class TestLTX2TileParallelDecode(unittest.TestCase):
    def _vae(self):
        vae = SimpleNamespace(
            use_parallel_decode=True,
            spatial_compression_ratio=1,
            tile_sample_min_height=4,
            tile_sample_min_width=4,
            tile_sample_stride_height=3,
            tile_sample_stride_width=3,
            decoder=lambda x, temb, causal: x.clone() + x.mean(),
        )
        for name in (
            "_should_use_tile_parallel_decode",
            "tiled_decode",
            "blend_v",
            "blend_h",
        ):
            setattr(
                vae,
                name,
                MethodType(getattr(ltx_2_vae.AutoencoderKLLTX2Video, name), vae),
            )
        return vae

    def test_uneven_tiles_match_serial_blending_on_both_ranks(self):
        z = torch.arange(112, dtype=torch.float32).reshape(1, 1, 2, 7, 8)
        vae = self._vae()
        expected = vae.tiled_decode(z, None, return_dict=False)[0]
        tiles = [
            vae.decoder(z[:, :, :, i : i + 4, j : j + 4], None, causal=None)
            for i in range(0, 7, 3)
            for j in range(0, 8, 3)
        ]
        flat = [
            torch.cat([tile.flatten() for tile in tiles[rank::2]]) for rank in range(2)
        ]

        def all_gather(outputs, value, group):
            for rank, output in enumerate(outputs):
                if value.dtype == torch.int64:
                    output.fill_(flat[rank].numel())
                else:
                    output.zero_()
                    output[: flat[rank].numel()].copy_(flat[rank])

        with (
            patch.dict(os.environ, {"SGLANG_LTX2_TILE_PARALLEL_DECODE": "1"}),
            patch.object(ltx_2_vae, "model_parallel_is_initialized", return_value=True),
            patch.object(ltx_2_vae, "get_decode_parallel_world_size", return_value=2),
            patch.object(
                ltx_2_vae,
                "get_decode_parallel_group_coordinator",
                return_value=SimpleNamespace(device_group=object()),
            ),
            patch.object(
                ltx_2_vae, "disable_spatial_parallel_decode", contextlib.nullcontext
            ),
            patch.object(ltx_2_vae.dist, "all_gather", side_effect=all_gather),
        ):
            for rank in range(2):
                with patch.object(
                    ltx_2_vae, "get_decode_parallel_rank", return_value=rank
                ):
                    actual = ltx_2_vae.AutoencoderKLLTX2Video.parallel_tiled_decode(
                        vae, z, return_dict=False
                    )[0]
                    self.assertTrue(torch.equal(actual, expected))

    def test_single_rank_uses_serial_fallback_without_collectives(self):
        vae = self._vae()
        z = torch.ones(1, 1, 2, 7, 8)
        with (
            patch.dict(os.environ, {"SGLANG_LTX2_TILE_PARALLEL_DECODE": "1"}),
            patch.object(ltx_2_vae, "model_parallel_is_initialized", return_value=True),
            patch.object(ltx_2_vae, "get_decode_parallel_world_size", return_value=1),
            patch.object(ltx_2_vae.dist, "all_gather") as gather,
        ):
            actual = ltx_2_vae.AutoencoderKLLTX2Video.parallel_tiled_decode(
                vae, z, return_dict=False
            )[0]
            self.assertTrue(
                torch.equal(actual, vae.tiled_decode(z, None, return_dict=False)[0])
            )
            gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
