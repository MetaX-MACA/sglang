# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class MiniMaxH3PipelineConfig(PipelineConfig):
    """MiniMax H3 pipeline configuration.

    Text encoding intentionally uses the native MiniMaxH3Qwen3VLHFEncoder path
    rather than the base ``text_encoder_configs`` loader contract.
    """

    task_type: ModelTaskType = ModelTaskType.TI2V
    dit_config: MiniMaxH3DiTConfig = field(default_factory=MiniMaxH3DiTConfig)
    vae_config: MiniMaxH3VideoVAEConfig = field(default_factory=MiniMaxH3VideoVAEConfig)
    audio_vae_config: MiniMaxH3AudioVAEConfig = field(
        default_factory=MiniMaxH3AudioVAEConfig
    )
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    audio_vae_precision: str = "fp32"
    output_audio_sample_rate: int | None = 32000
    output_audio_channels: int | None = 2
    output_av_drift_tolerance_s: float | None = 0.25

    def accepts_audio_input(self) -> bool:
        return True

    def supports_disaggregation(self) -> bool:
        return False

    @property
    def requires_audio_output(self) -> bool:
        return True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(auto_dit_layerwise_offload=True)


__all__ = ["MiniMaxH3PipelineConfig"]
