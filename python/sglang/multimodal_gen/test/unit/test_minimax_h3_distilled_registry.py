# SPDX-License-Identifier: Apache-2.0
"""Offline registry coverage for MiniMax H3's single cfg-distilled path."""

import unittest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


def _resolve_by_detector(hf_id: str):
    lowered = hf_id.lower()
    for model_id, detector in registry._MODEL_NAME_DETECTORS:
        if detector(lowered):
            return registry._CONFIG_REGISTRY.get(model_id)
    return None


class TestMiniMaxH3SingleRegistryPath(unittest.TestCase):
    def test_every_minimax_h3_model_id_resolves_to_the_single_config_pair(self):
        model_ids = (
            "MiniMaxAI/MiniMax-H3-FL2VA",
            "MiniMaxAI/minimax_h3",
            "MiniMaxH3Pipeline",
            "some-org/minimax_h3-v1",
            "/models/minimax_h3-dev",
            "/MODELS/MINIMAX_H3-CFG-DISTILL",
        )
        for model_id in model_ids:
            with self.subTest(model_id=model_id):
                info = _resolve_by_detector(model_id)
                self.assertIsNotNone(info)
                self.assertIs(info.pipeline_config_cls, MiniMaxH3PipelineConfig)
                self.assertIs(info.sampling_param_cls, MiniMaxH3SamplingParams)

    def test_registry_contains_exactly_one_minimax_h3_config_pair(self):
        matches = [
            info
            for info in registry._CONFIG_REGISTRY.values()
            if info.pipeline_config_cls is MiniMaxH3PipelineConfig
        ]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].sampling_param_cls, MiniMaxH3SamplingParams)

    def test_pipeline_and_sampling_configs_are_cfg_distilled_only(self):
        pipeline = MiniMaxH3PipelineConfig()
        sampling = MiniMaxH3SamplingParams()

        self.assertFalse(hasattr(pipeline, "is_cfg_distilled"))
        self.assertEqual(sampling.guidance_scale, 1.0)
        self.assertIsNone(sampling.negative_prompt)
        with self.assertRaisesRegex(TypeError, "guidance_scale"):
            MiniMaxH3SamplingParams(guidance_scale=2.0)


if __name__ == "__main__":
    unittest.main()
