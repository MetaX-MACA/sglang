# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-token DiT.

Native SGLang implementation of the MiniMax H3 audio-video DiT. The forward
contract accepts packed inference keyword arguments and returns packed logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()
_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32

MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})

# AdaLN modality count: token tags carry -1 for padding and 0/1/2 for
# video/text/audio tokens (padding is clamped to 0 before the embedding
# lookup and masked out afterwards).
MINIMAX_H3_ADALN_MODALITY_NUM = 3


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)


def _ulysses_ctx() -> tuple[int, int]:
    """(world_size, rank) of the Ulysses sequence-parallel group.

    Returns (1, 0) when model parallelism is not initialized (unit tests /
    single-process debug paths init tp=1 sp=1 which also yields ws=1).
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_rank,
        get_ulysses_parallel_world_size,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return 1, 0
    return get_ulysses_parallel_world_size(), get_ulysses_parallel_rank()


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> nn.RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply per-index affine modulation: x * (1 + scale[idx]) + shift[idx].
    return (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Apply the per-index gated residual: x + gate[idx] * other.
    return (x + gate.index_select(0, indices) * other).to(dtype)


class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate the first rot_dim head dims; pass the rest through.

    x: [T, heads, head_dim]; freqs: [T, rot_dim]. In the unfused path, cos/sin
    are cast to the activation dtype before the elementwise math.
    """
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)  # [T, 1, rot_dim]
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=False,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32.

        The sinusoidal embedding stays fp32 throughout and concatenates cosine
        values before sine values.
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=_FP32_DTYPE, device=t.device)
            / half
        )
        args = t.to(_FP32_DTYPE)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden, _ = self.proj_in(t_freq)
        hidden = nn.functional.silu(hidden)
        out, _ = self.proj_out(hidden)
        return out


def _sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Segment-wise SDPA equivalent of the non-causal varlen FA call.

    Mirrors the generic attention layer's semantics: FA is the fast path,
    SDPA is the correctness fallback when the platform resolves another
    backend. Segments are delimited by ``cu_seqlens`` exactly like the
    varlen kernel, so attention never crosses packed-document boundaries.
    """
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = torch.nn.functional.scaled_dot_product_attention(
            seg_q,
            seg_k,
            seg_v,
            scale=softmax_scale,
        )
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self._supported_attention_backends = arch._supported_attention_backends
        self._attention_backend: AttentionBackendEnum | None = None
        self.qkv_proj = ColumnParallelLinear(
            arch.hidden_size,
            inner_dim * 3,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        self._install_qkv_weight_loader(arch)
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )

    def _install_qkv_weight_loader(self, arch: MiniMaxH3DiTArchConfig) -> None:
        base_loader = self.qkv_proj.weight.weight_loader

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # The grouped checkpoint layout is
            # [num_query_groups, q_per_group + k + v] before splitting.
            # MiniMax H3 uses MHA, so checkpoint rows are per-head [q, k, v],
            # while SGLang stores [q_all, k_all, v_all].
            reordered = _reorder_grouped_qkv_to_qkv(
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )
            base_loader(param, reordered)

        self.qkv_proj.weight.weight_loader = _weight_loader

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        sp_seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(self.num_heads * self.head_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)

        sp_active = sp_seq_lens is not None and len(sp_seq_lens) > 1
        if sp_active:
            from sglang.multimodal_gen.runtime.layers.usp import (
                _usp_input_all_to_all_varlen,
                _usp_output_all_to_all_varlen,
            )

            q = _usp_input_all_to_all_varlen(q[None], sp_seq_lens, head_dim=2)[0]
            k = _usp_input_all_to_all_varlen(k[None], sp_seq_lens, head_dim=2)[0]
            v = _usp_input_all_to_all_varlen(v[None], sp_seq_lens, head_dim=2)[0]

        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            flash_attn as _fa_backend,
        )

        if self._attention_backend is None:
            # Resolve through the shared selector once per module: the
            # platform picks the backend and the flash-attention generation
            # for this GPU (fa_ver: FA3 on Hopper, FA4 on Blackwell) and
            # honors --attention-backend overrides, same as other models.
            # q.dtype is the actual kernel input dtype; get_compute_dtype()
            # depends on thread-local mixed-precision state that the MiniMax H3
            # denoise path (autocast disabled) never sets, which would fall
            # back to float32 and wrongly disqualify FA.
            self._attention_backend = get_attn_backend(
                self.head_dim,
                q.dtype,
                supported_attention_backends=self._supported_attention_backends,
            ).get_enum()

        if self._attention_backend is AttentionBackendEnum.FA:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.softmax_scale,
                causal=False,
                ver=_fa_backend.fa_ver,
            )
            if isinstance(out, tuple):
                out = out[0]
        else:
            # Generic-layer semantics: SDPA is the correctness fallback for
            # platforms where FA is unavailable.
            out = _sdpa_varlen_attention(
                q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale
            )
        if sp_active:
            out = _usp_output_all_to_all_varlen(out[None], sp_seq_lens, head_dim=2)[0]
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            arch.hidden_size,
            arch.ffn_hidden_size * 2,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        gate, up = hidden.chunk(2, dim=-1)
        hidden = nn.functional.silu(gate) * up
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                "adaln out_features mismatch: "
                f"{out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """t_emb: [M, t_dim] -> expand_ratio tensors of [M*modality_num, H]."""
        x = nn.functional.silu(t_emb)
        x, _ = self.linear(x.to(self.linear.weight.dtype))
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Standard pre-norm transformer block without AdaLN or RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, quant_config)
        self.mlp = MiniMaxH3MLP(arch, quant_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_freqs=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(arch, quant_config)
                for _ in range(arch.token_refiner_num_layers)
            ]
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, quant_config)
        self.mlp = MiniMaxH3MLP(arch, quant_config)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            quant_config,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        sp_seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        """x: [T, H]; t_emb: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(t_emb)

        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(
            h, shift_msa, scale_msa, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.attn(
            h,
            rope_freqs=rope_freqs,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sp_seq_lens=sp_seq_lens,
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16_DTYPE)

        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(
            h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.mlp(h)
        return _modulate_gate(
            residual, gate_mlp, h, combined_indices, dtype=_BF16_DTYPE
        )


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()
        video_patch_dim = (
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2]
        )
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            quant_config,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [T, H] -> (video_logits [T, 96] fp32, audio_logits [T, 32] fp32).

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        """
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices, dtype=_BF16_DTYPE)
        # Preserve full precision through both final output projections.
        h = h.to(_FP32_DTYPE)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio


class MiniMaxH3DiTModel(CachableDiT):
    _fsdp_shard_conditions = _ARCH_DEFAULTS._fsdp_shard_conditions
    # Parameters mix fp32 (MINIMAX_H3_FP32_PARAM_NAMES: patch_proj, t_embedder,
    # adaln, final layer) with bf16 blocks; FSDP must all-gather in each
    # param's own dtype (see fsdp_load.maybe_load_fsdp_model).
    _fsdp_mixed_dtype_params = True
    _compile_conditions = _ARCH_DEFAULTS._compile_conditions
    _supported_attention_backends = _ARCH_DEFAULTS._supported_attention_backends
    param_names_mapping = _ARCH_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_DEFAULTS.lora_param_names_mapping

    def _validate_tp_config(self, *, arch: MiniMaxH3DiTArchConfig, tp_size: int) -> None:
        if tp_size != 1:
            raise ValueError(
                "MiniMaxH3DiTModel supports TP=1 only. Packed qkv/fc1 TP "
                "requires per-logical-matrix sharding before enabling TP>1."
            )
        if arch.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if arch.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if arch.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if arch.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")

    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self._validate_tp_config(arch=arch, tp_size=get_tp_world_size())

        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=quant_config,
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(arch, quant_config)
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch, quant_config)
        self.blocks = nn.ModuleList(
            [MiniMaxH3DiTBlock(arch, quant_config) for _ in range(arch.num_layers)]
        )
        self.final_layer = MiniMaxH3FinalLayer(arch, quant_config)
        self._mark_missing_params_required()

    def _mark_missing_params_required(self) -> None:
        for _, param in self.named_parameters():
            param.missing_param_init = "error"

    def post_load_weights(self) -> None:
        for name, param in self.named_parameters():
            if name in MINIMAX_H3_FP32_PARAM_NAMES and param.dtype != _FP32_DTYPE:
                raise ValueError(
                    f"{name} must stay fp32 after load, got {param.dtype}."
                )
        for name, buffer in self.named_buffers():
            if name in MINIMAX_H3_FP32_BUFFER_NAMES and buffer.dtype != _FP32_DTYPE:
                raise ValueError(
                    f"{name} must stay fp32 after load, got {buffer.dtype}."
                )

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if isinstance(pos_info, dict):
            ids = pos_info.get("position_ids")
        else:
            ids = getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        if isinstance(psp, dict):
            value = psp.get(field)
        else:
            value = getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build packed multimodal embeddings for the TP=1/SP=1 inference path.

        Returns (decoder_input [S, H] bf16, t_emb [M, t_dim] fp32).
        """
        # Latent embedders stay fp32 in and out; their outputs are cast to the
        # bf16 sequence dtype only during indexed scattering.
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(_FP32_DTYPE)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = (
            audio_x.view(-1, audio_x.shape[-1])
            .index_select(0, audio_pos)
            .to(_FP32_DTYPE)
        )
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        text_rows = text_embeddings_selected.to(device=device, dtype=_BF16_DTYPE)
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
        )

        embeddings = torch.zeros(
            (seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE
        )
        embeddings.index_add_(
            0, text_pos, text_embed.to(_BF16_DTYPE)[: text_pos.shape[0]]
        )
        embeddings.index_add_(
            0, img_pos, video_embed.to(_BF16_DTYPE)[: img_pos.shape[0]]
        )
        embeddings.index_add_(
            0, audio_pos, audio_embed.to(_BF16_DTYPE)[: audio_pos.shape[0]]
        )

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward.

        Keyword names follow the checkpoint's serving contract.
        Returns `(video_logits, audio_logits)` from rows selected by
        `img_pos_for_infer_output_info` and `audio_pos_info`, with condition
        rows zeroed by update masks.
        """
        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        update_mask = _required_kwarg(kwargs, "update_mask")
        token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(
            torch.int32
        )
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(
            refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        refiner_max = int(
            self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q")
        )

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags.shape[0] != seq_len:
            raise ValueError(
                "token_tags must cover the full packed sequence "
                f"({seq_len}), got {token_tags.shape[0]}."
            )
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(
                f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}"
            )
        device = x.device

        # Compute RoPE frequencies over the full packed sequence.
        rope_freqs = self.rope(img_position_ids).to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            seq_len=seq_len,
            device=device,
        )

        combined_indices = (
            inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)
        ).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        # With Ulysses sequence parallelism, shard rows across the group for
        # the block stack. Attention trades sequence for heads internally;
        # everything else is row-local. Embedding/refiner above and the final
        # layer below run replicated, which is cheap relative to 50 blocks.
        sp_ws, sp_rank = _ulysses_ctx()
        sp_seq_lens: list[int] | None = None
        block_rope = rope_freqs
        block_combined = combined_indices
        if sp_ws > 1:
            if seq_len % sp_ws:
                raise ValueError(
                    f"packed seq_len {seq_len} not divisible by ulysses "
                    f"world size {sp_ws}"
                )
            if self.num_attention_heads % sp_ws:
                raise ValueError(
                    f"num heads {self.num_attention_heads} not divisible by "
                    f"ulysses world size {sp_ws}"
                )
            local = seq_len // sp_ws
            shard = slice(sp_rank * local, (sp_rank + 1) * local)
            hidden = hidden[shard]
            block_rope = rope_freqs[shard]
            block_combined = combined_indices[shard]
            sp_seq_lens = [local] * sp_ws
        for block in self.blocks:
            hidden = block(
                hidden,
                t_emb=t_emb,
                combined_indices=block_combined,
                rope_freqs=block_rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                sp_seq_lens=sp_seq_lens,
            )
        if sp_ws > 1:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_group,
            )

            hidden = get_sp_group().all_gather(hidden, dim=0)

        video_logits, audio_logits = self.final_layer(
            hidden,
            t_emb=t_emb,
            inverse_indices=inverse_indices,
        )

        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length mismatch: "
                    f"{update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "_reorder_grouped_qkv_to_qkv",
]
