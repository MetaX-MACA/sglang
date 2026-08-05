# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 HF Qwen3VL text/vision encoder.

Implements the tensor-encoding recipe:

- plain bf16 ``from_pretrained()`` (no device_map), then retain only the
  layers needed for layer-50 output
- cuDNN SDP is enabled during encode via
  ``torch.backends.cuda.enable_cudnn_sdp(True)``
- forward the multimodal backbone with all-ones attention_mask and
  mm_token_type_ids derived from ``config.image_token_id`` (position_ids
  omitted: model computes rope internally)
- return the unnormalized output after decoder layer 49; the encoder
  output contract is ``hidden_states[50]``, without retaining every
  intermediate hidden state or computing unused language-model logits

The encoder is pure-tensor (presentation building lives in
``pipelines_core/.../minimax_h3/presentation.py``). Supports persistent CPU
offload for single-GPU residency (offload after use, never free).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.sp_broadcast import minimax_h3_sp_ctx
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120

logger = init_logger(__name__)


def _enable_cudnn_sdp() -> None:
    # Encoding runs with cuDNN SDP enabled.
    torch.backends.cuda.enable_cudnn_sdp(True)


def _retain_selected_lm_layer(model: torch.nn.Module, selected_layer: int) -> None:
    """Trim a Qwen3VL backbone to the requested pre-norm hidden-state index.

    The MiniMax H3 contract names the unnormalized output after the
    first ``N`` decoder layers ``hidden_states[N]``. It consumes layer 50 only,
    so keeping later layers, the final norm, every intermediate hidden state,
    and the causal-LM head is both unnecessary and prohibitively expensive for
    long Ref2VA presentations. This explicit definition also avoids depending
    on Transformers' version-specific ``output_hidden_states`` tuple indexing.
    """

    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise TypeError("Qwen3VL backbone must expose language_model.layers")
    original_num_layers = len(layers)
    if not 0 < selected_layer <= original_num_layers:
        raise ValueError(
            "selected Qwen3VL layer must be in "
            f"[1, {original_num_layers}], got {selected_layer}"
        )
    language_model.layers = layers[:selected_layer]
    language_model.norm = torch.nn.Identity()
    if hasattr(language_model, "config"):
        language_model.config.num_hidden_layers = selected_layer
    logger.info(
        "MiniMax H3 Qwen3VL retained %d/%d decoder layers for hidden_states[%d]",
        selected_layer,
        original_num_layers,
        selected_layer,
    )


class MiniMaxH3Qwen3VLHFEncoderStub:
    """Placeholder for non-main sequence-parallel ranks."""

    def __init__(self, *, hf_model_path: str) -> None:
        self.hf_model_path = hf_model_path

    def load_to_device(self) -> None:
        return

    def offload_to_cpu(self) -> None:
        return

    def encode_ids(self, *args, **kwargs):
        raise RuntimeError(
            "MiniMax H3 Qwen3VL encoder is stubbed on this sp rank; encode "
            "must run on the sp main rank"
        )


class MiniMaxH3Qwen3VLHFEncoder:
    """Frozen HF Qwen3VL-32B forward, layer-50 hidden-state extractor."""

    def __init__(
        self,
        *,
        hf_model_path: str,
        device: str = "cuda",
    ) -> None:
        from transformers import Qwen3VLForConditionalGeneration

        self.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
        self.hf_model_path = hf_model_path
        self.device = torch.device(device)
        # Residency contract: weights live on CPU between uses; the
        # stage calls load_to_device() before encode_ids() and
        # offload_to_cpu() afterwards. Constructed on CPU so pipeline init
        # never co-resides the 64G encoder with the 62G DiT on one GPU.
        causal_lm = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_model_path, dtype=torch.bfloat16, trust_remote_code=False
        ).eval()
        self.image_token_id = int(causal_lm.config.image_token_id)
        self.video_token_id = int(causal_lm.config.video_token_id)
        # Consumers read hidden_states[50], never logits. Drop
        # the CausalLM wrapper/head and later decoder layers while everything
        # is still on CPU, before the first load_to_device().
        self.model = causal_lm.model
        _retain_selected_lm_layer(self.model, self.selected_lm_layer)
        del causal_lm

    @classmethod
    def load_component(
        cls,
        *,
        component_model_path: str,
        component_name: str,
        config: dict[str, Any],
    ) -> MiniMaxH3Qwen3VLHFEncoder | MiniMaxH3Qwen3VLHFEncoderStub:
        """Load a flat Qwen3VL HF snapshot for a registered component."""
        if component_name != "text_encoder":
            raise ValueError(
                f"{cls.__name__} can only be used for component text_encoder, "
                f"got {component_name}"
            )
        config_path = os.path.join(component_model_path, "config.json")
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or not any(
            isinstance(item, str) and item.startswith("Qwen3VL")
            for item in architectures
        ):
            raise ValueError(
                f"{config_path} must be a Qwen3VL HF snapshot config "
                f"(architectures=[Qwen3VL*...]), got architectures={architectures!r}"
            )

        sp_world, sp_rank = minimax_h3_sp_ctx()
        if sp_world > 1 and sp_rank != 0:
            logger.info(
                "MiniMax H3 Qwen3VL encoder stubbed on sp rank %d (main-rank only)",
                sp_rank,
            )
            return MiniMaxH3Qwen3VLHFEncoderStub(hf_model_path=component_model_path)
        return cls(hf_model_path=component_model_path)

    @torch.inference_mode()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a 1-D token stream; returns [seq_len, hidden_dim] bf16 CPU.

        pixel_values/image_grid_thw must be provided together for image
        presentations (fl2va) and omitted for text-only (t2va);
        pixel_values_videos/video_grid_thw likewise for video-frame
        presentations (ref2va video).
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )
        if next(self.model.parameters()).device.type != self.device.type:
            raise RuntimeError(
                "encoder is offloaded; call load_to_device() before encode_ids()"
            )
        _enable_cudnn_sdp()
        ids = input_ids.to(self.device, torch.long)[None]
        kwargs: dict = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
        }
        if pixel_values is not None or pixel_values_videos is not None:
            # get_rope_index routes grid consumption by token type:
            # {1: image_grid_thw, 2: video_grid_thw}. Marking video pads as 1
            # makes video blocks consume
            # image grid entries -> StopIteration.
            mm_types = torch.zeros_like(ids, dtype=torch.int32)
            mm_types[ids == self.image_token_id] = 1
            mm_types[ids == self.video_token_id] = 2
            kwargs["mm_token_type_ids"] = mm_types
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            kwargs["image_grid_thw"] = image_grid_thw.to(self.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            kwargs["video_grid_thw"] = video_grid_thw.to(self.device, torch.long)
        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state[0].to(torch.bfloat16)
        if list(hidden.shape) != [int(ids.shape[1]), self.hidden_dim]:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)} for "
                f"seq_len={int(ids.shape[1])}"
            )
        return hidden.cpu()

    def offload_to_cpu(self) -> None:
        """Move weights to CPU (persistent residency; never freed)."""
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def load_to_device(self) -> None:
        self.model.to(self.device)


EntryClass = MiniMaxH3Qwen3VLHFEncoder

__all__ = [
    "MiniMaxH3Qwen3VLHFEncoder",
    "MiniMaxH3Qwen3VLHFEncoderStub",
]
