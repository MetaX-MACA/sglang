# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_contract import (
    validate_minimax_h3_vae_latent_stats,
)


@dataclass
class MiniMaxH3VideoVAEArchConfig(VAEArchConfig):
    latent_channels: int = 24
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16
    vae_clip_length: int = 17
    vae_token_drop: int = 3


@dataclass
class MiniMaxH3VideoVAEConfig(VAEConfig):
    arch_config: MiniMaxH3VideoVAEArchConfig = field(
        default_factory=MiniMaxH3VideoVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True
    use_tiling: bool = True
    use_parallel_tiling: bool = True

    def post_init(self) -> None:
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )


__all__ = ["MiniMaxH3VideoVAEArchConfig", "MiniMaxH3VideoVAEConfig"]
