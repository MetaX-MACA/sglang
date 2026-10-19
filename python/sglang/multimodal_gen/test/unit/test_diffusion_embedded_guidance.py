import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2PipelineConfig,
    FluxPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage


class TestEmbeddedGuidance(unittest.TestCase):
    def test_guidance_units_match_model_family(self):
        for config_cls, expected in [
            (HunyuanConfig, 6000.0),
            (FastHunyuanConfig, 6000.0),
            (FluxPipelineConfig, 6000.0),
            (Flux2PipelineConfig, 6.0),
            (WanT2V480PConfig, 6.0),
        ]:
            with self.subTest(config=config_cls.__name__):
                stage = DenoisingStage.__new__(DenoisingStage)
                stage.server_args = SimpleNamespace(pipeline_config=config_cls())
                result = stage._build_guidance(2, torch.float32, "cpu", 6.0)
                torch.testing.assert_close(result, torch.full((2,), expected))

    def test_hunyuan_preserves_requested_dtype_and_scale(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        stage.server_args = SimpleNamespace(pipeline_config=HunyuanConfig())
        result = stage._build_guidance(1, torch.bfloat16, "cpu", 4.0)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(
            result, torch.full((1,), 4000.0, dtype=torch.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
