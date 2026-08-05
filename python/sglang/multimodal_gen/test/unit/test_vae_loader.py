import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_2_TI2V_5B_Config,
    Wan2_2_I2V_A14B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import (
    remote_code,
    vae_loader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    RemoteComponentLoadError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.remote_code import (
    _remote_vae_package_name,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _backfill_ltx2_audio_vae_latent_stats,
    _should_use_channels_last_3d,
)
from sglang.multimodal_gen.runtime.models.registry import (
    ModelRegistry,
    UnsupportedCustomArchitecture,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae


class _FakeServerArgs:
    def __init__(self, pipeline_config, num_gpus=1):
        self.pipeline_config = pipeline_config
        self.num_gpus = num_gpus


def _write_remote_module(root: Path, relative_path: str, source: str) -> None:
    module_path = root / relative_path
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


class TestVAELoader(unittest.TestCase):
    def test_component_names_use_video_vae_without_legacy_alias(self):
        self.assertEqual(
            vae_loader.VAELoader.component_names,
            ["vae", "audio_vae", "video_vae"],
        )

    def test_minimax_h3_component_config_injects_latent_contract(self):
        loader = vae_loader.VAELoader()
        for component_name, expected_channels, config_attr in (
            ("video_vae", 24, "vae_config"),
            ("audio_vae", 32, "audio_vae_config"),
        ):
            with self.subTest(component_name=component_name):
                pipeline_config = MiniMaxH3PipelineConfig()
                server_args = SimpleNamespace(
                    model_paths={},
                    pipeline_config=pipeline_config,
                    trust_remote_code=True,
                    vae_cpu_offload=False,
                )
                component_config = {
                    "_class_name": "MiniMaxH3VideoVAE",
                    "latent_channels": expected_channels,
                    "latents_mean": [
                        float(index) for index in range(expected_channels)
                    ],
                    "latents_std": [
                        float(index + 1) for index in range(expected_channels)
                    ],
                }
                with (
                    patch.object(
                        vae_loader,
                        "get_diffusers_component_config",
                        return_value=component_config,
                    ),
                    patch.object(
                        loader,
                        "should_offload",
                        side_effect=RuntimeError("stop after config injection"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "stop after config injection"),
                ):
                    loader.load_customized("/unused", server_args, component_name)

                arch_config = getattr(pipeline_config, config_attr).arch_config
                self.assertEqual(arch_config.latent_channels, expected_channels)
                self.assertEqual(
                    arch_config.latents_mean, component_config["latents_mean"]
                )
                self.assertEqual(
                    arch_config.latents_std, component_config["latents_std"]
                )

    def test_minimax_h3_component_config_rejects_missing_latent_stats(self):
        for component_name, expected_channels, missing_field in (
            ("video_vae", 24, "latents_mean"),
            ("audio_vae", 32, "latents_std"),
        ):
            with self.subTest(
                component_name=component_name,
                missing_field=missing_field,
            ):
                config = {
                    "_class_name": "MiniMaxH3VideoVAE",
                    "latent_channels": expected_channels,
                    "latents_mean": [0.0] * expected_channels,
                    "latents_std": [1.0] * expected_channels,
                }
                del config[missing_field]
                self._assert_minimax_h3_config_rejected(
                    component_name,
                    config,
                    rf"MiniMax H3 {component_name} config.json missing {missing_field}",
                )

    def test_minimax_h3_component_config_rejects_invalid_latent_stats(self):
        invalid_cases = (
            ("latents_mean", "not-a-list", "must be a list of numbers"),
            (
                "latents_mean",
                [False, *([0.0] * 23)],
                "must be a list of numbers",
            ),
            ("latents_std", [1.0] * 23, "must contain exactly 24 values"),
        )
        for field_name, invalid_value, error_pattern in invalid_cases:
            with self.subTest(field_name=field_name):
                config = {
                    "_class_name": "MiniMaxH3VideoVAE",
                    "latent_channels": 24,
                    "latents_mean": [0.0] * 24,
                    "latents_std": [1.0] * 24,
                }
                config[field_name] = invalid_value
                self._assert_minimax_h3_config_rejected(
                    "video_vae",
                    config,
                    error_pattern,
                )

    def test_minimax_h3_component_config_rejects_latent_channel_mismatch(self):
        config = {
            "_class_name": "MiniMaxH3AudioVAE",
            "latent_channels": 24,
            "latents_mean": [0.0] * 32,
            "latents_std": [1.0] * 32,
        }

        self._assert_minimax_h3_config_rejected(
            "audio_vae",
            config,
            "MiniMax H3 audio_vae latent_channels must be 32, got 24",
        )

    def test_minimax_h3_component_config_rejects_non_finite_means(self):
        for invalid_mean in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_mean=invalid_mean):
                config = {
                    "_class_name": "MiniMaxH3VideoVAE",
                    "latent_channels": 24,
                    "latents_mean": [invalid_mean, *([0.0] * 23)],
                    "latents_std": [1.0] * 24,
                }
                self._assert_minimax_h3_config_rejected(
                    "video_vae",
                    config,
                    "latents_mean values must be finite",
                )

    def test_minimax_h3_component_config_rejects_invalid_standard_deviations(self):
        for invalid_std in (
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -1.0,
        ):
            with self.subTest(invalid_std=invalid_std):
                config = {
                    "_class_name": "MiniMaxH3AudioVAE",
                    "latent_channels": 32,
                    "latents_mean": [0.0] * 32,
                    "latents_std": [invalid_std, *([1.0] * 31)],
                }
                self._assert_minimax_h3_config_rejected(
                    "audio_vae",
                    config,
                    "latents_std values must be finite and greater than zero",
                )

    def _assert_minimax_h3_config_rejected(
        self,
        component_name,
        config,
        error_pattern,
    ):
        loader = vae_loader.VAELoader()
        server_args = SimpleNamespace(
            model_paths={},
            pipeline_config=MiniMaxH3PipelineConfig(),
            trust_remote_code=True,
            vae_cpu_offload=False,
            resolve_component_attention_backend=Mock(return_value=(None, None)),
        )
        with (
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value=config,
            ),
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=0.0,
            ),
            patch.object(loader, "load_native") as load_native,
            self.assertRaisesRegex(RemoteComponentLoadError, error_pattern),
        ):
            loader.load("/unused", server_args, component_name, "diffusers")
        load_native.assert_not_called()

    def test_malformed_auto_map_fails_closed_before_component_configuration(self):
        loader = vae_loader.VAELoader()
        server_args = SimpleNamespace(trust_remote_code=True)
        with (
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value={"_class_name": "NativeClass", "auto_map": []},
            ),
            self.assertRaisesRegex(RemoteComponentLoadError, "Invalid remote VAE"),
        ):
            loader.load_customized("/path/that/does/not/exist", server_args, "vae")

    def test_instantiation_failure_preserves_successfully_imported_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            component_root = Path(temp_dir)
            _write_remote_module(
                component_root,
                "modeling_vae.py",
                """
                class RemoteVAE:
                    attempts = 0

                    @classmethod
                    def from_pretrained(cls, *args, **kwargs):
                        cls.attempts += 1
                        if cls.attempts == 1:
                            raise RuntimeError("first instantiation fails")
                        return cls()

                    def to(self, *args, **kwargs):
                        return self
                """,
            )
            package_name = _remote_vae_package_name(component_root)
            vae_config = SimpleNamespace(update_model_arch=Mock())
            server_args = SimpleNamespace(
                model_paths={},
                pipeline_config=SimpleNamespace(
                    vae_config=vae_config,
                    vae_precision="fp32",
                ),
                revision=None,
                trust_remote_code=True,
                vae_cpu_offload=False,
            )
            loader = vae_loader.VAELoader()

            def component_config(**_kwargs):
                return {
                    "_class_name": "IgnoredNativeClass",
                    "auto_map": {"AutoModel": "modeling_vae.RemoteVAE"},
                }

            try:
                with (
                    patch.object(
                        vae_loader,
                        "get_diffusers_component_config",
                        side_effect=component_config,
                    ),
                    patch.object(
                        vae_loader,
                        "resolve_component_precision",
                        return_value=torch.float32,
                    ),
                    patch.object(
                        loader, "target_device", return_value=torch.device("cpu")
                    ),
                    patch.object(
                        vae_loader, "_should_use_channels_last_3d", return_value=False
                    ),
                    patch.object(
                        vae_loader.current_platform,
                        "optimize_vae",
                        side_effect=lambda vae: vae,
                    ),
                ):
                    with self.assertRaisesRegex(
                        RemoteComponentLoadError, "Failed to load remote VAE"
                    ):
                        loader.load_customized(str(component_root), server_args, "vae")

                    self.assertIn(package_name, sys.modules)
                    self.assertIn(package_name, remote_code._REMOTE_VAE_FINDERS)
                    remote_class = sys.modules[f"{package_name}.modeling_vae"].RemoteVAE
                    self.assertEqual(remote_class.attempts, 1)

                    loaded_vae = loader.load_customized(
                        str(component_root), server_args, "vae"
                    )

                self.assertIs(type(loaded_vae), remote_class)
                self.assertEqual(remote_class.attempts, 2)
            finally:
                remote_code._remove_remote_vae_package(package_name)

    def test_remote_component_failure_does_not_fall_back_to_native(self):
        loader = vae_loader.VAELoader()
        server_args = Mock()
        server_args.resolve_component_attention_backend.return_value = (None, None)
        remote_error = RemoteComponentLoadError("remote load failed")

        with (
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=0.0,
            ),
            patch.object(loader, "load_customized", side_effect=remote_error),
            patch.object(loader, "load_native") as load_native,
            self.assertRaisesRegex(RemoteComponentLoadError, "remote load failed"),
        ):
            loader.load("/tmp/model", server_args, "audio_vae", "diffusers")

        load_native.assert_not_called()

    def test_unsupported_custom_architecture_falls_back_to_native(self):
        loader = vae_loader.VAELoader()
        server_args = Mock()
        server_args.resolve_component_attention_backend.return_value = (None, None)
        unsupported_error = UnsupportedCustomArchitecture(
            "Unsupported model architecture: FakeVAE. Registered architectures: []"
        )
        native_component = Mock()
        native_component.to.return_value = native_component

        with (
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=0.0,
            ),
            patch.object(loader, "load_customized", side_effect=unsupported_error),
            patch.object(
                loader, "load_native", return_value=native_component
            ) as load_native,
            patch.object(loader, "target_device", return_value=torch.device("cpu")),
        ):
            component, _ = loader.load(
                "/tmp/model", server_args, "audio_vae", "diffusers"
            )

        load_native.assert_called_once()
        self.assertIs(component, native_component)

    def test_customized_loader_bug_propagates_without_native_fallback(self):
        loader = vae_loader.VAELoader()
        server_args = Mock()
        server_args.resolve_component_attention_backend.return_value = (None, None)
        customized_bug = RuntimeError("weight key drift in customized VAE")

        with (
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=0.0,
            ),
            patch.object(loader, "load_customized", side_effect=customized_bug),
            patch.object(loader, "load_native") as load_native,
            self.assertRaisesRegex(RuntimeError, "weight key drift"),
        ):
            loader.load("/tmp/model", server_args, "audio_vae", "diffusers")

        load_native.assert_not_called()

    def test_registry_raises_typed_unsupported_architecture(self):
        with self.assertRaises(UnsupportedCustomArchitecture) as ctx:
            ModelRegistry.resolve_model_cls("NotARealArchitecture")

        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIn(
            "Unsupported model architecture: NotARealArchitecture",
            str(ctx.exception),
        )

    def test_backfill_ltx2_audio_vae_latent_stats_maps_official_keys(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([3.0, 4.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_does_not_override_existing(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
            "latents_mean": torch.tensor([5.0, 6.0]),
            "latents_std": torch.tensor([7.0, 8.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([7.0, 8.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_skips_non_audio_vae(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0]),
            "per_channel_statistics.std-of-means": torch.tensor([2.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "vae")

        self.assertNotIn("latents_mean", loaded)
        self.assertNotIn("latents_std", loaded)

    def test_channels_last_3d_defaults_true_for_qwen_image_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertTrue(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_fast_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(FastWan2_2_TI2V_5B_Config(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(Wan2_2_I2V_A14B_Config(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_can_be_disabled_by_env(self):
        with (
            patch.dict(
                "os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "false"}
            ),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_can_be_enabled_by_env(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "true"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_auto_uses_model_policy(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "auto"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            wan_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            ltx_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)

            self.assertTrue(_should_use_channels_last_3d(wan_args, "video_vae"))
            self.assertFalse(_should_use_channels_last_3d(ltx_args, "video_vae"))

    def test_channels_last_3d_skips_non_video_vae_components(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "audio_vae"))

    def test_channels_last_3d_skips_unsupported_platforms(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=False),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_skips_non_cuda_platforms(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=False),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertIs(out, x)

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_uses_channels_last_3d_on_cuda(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=True),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))


if __name__ == "__main__":
    unittest.main()
