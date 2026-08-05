# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiTModel,
    MiniMaxH3Rope,
    _apply_rope,
    _modulate_gate,
    _modulate_scale_shift,
    _reorder_grouped_qkv_to_qkv,
    _sdpa_varlen_attention,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.server.accuracy_utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


class TestMiniMaxH3DiTContract(unittest.TestCase):
    def test_param_mappings_are_empty_for_native_key_checkpoints(self):
        # MiniMax H3 checkpoints ship with sglang-native key names, so both
        # mapping dicts must stay empty and the mapping fn must be the
        # identity for native checkpoint keys.
        arch = MiniMaxH3DiTArchConfig()
        self.assertEqual(arch.param_names_mapping, {})
        self.assertEqual(arch.reverse_param_names_mapping, {})
        forward = get_param_names_mapping(arch.param_names_mapping)
        native_keys = (
            "video_patch_proj.weight",
            "audio_patch_proj.bias",
            "condition_proj.weight",
            "time_embedder.proj_in.weight",
            "time_embedder.proj_out.bias",
            "rope.inv_freq",
            "token_refiner.blocks.0.attn.qkv_proj.weight",
            "token_refiner.blocks.0.attn.out_proj.weight",
            "token_refiner.blocks.0.mlp.fc1.weight",
            "token_refiner.blocks.0.mlp.fc2.weight",
            "token_refiner.final_norm.weight",
            "blocks.49.attn.qkv_proj.weight",
            "blocks.49.attn.out_proj.weight",
            "blocks.49.attn.q_norm.weight",
            "blocks.49.attn.k_norm.weight",
            "blocks.49.mlp.fc1.weight",
            "blocks.49.mlp.fc2.weight",
            "blocks.49.adaln_proj.linear.bias",
            "blocks.49.norm1.weight",
            "blocks.49.norm2.weight",
            "final_layer.norm.weight",
            "final_layer.adaln_proj.linear.weight",
            "final_layer.video_out.bias",
            "final_layer.audio_out.weight",
        )
        for key in native_keys:
            self.assertEqual(forward(key), (key, None, None))

    def test_qkv_reorder_from_grouped_to_qkv(self):
        weight = torch.arange(12, dtype=torch.float32).reshape(12, 1)
        actual = _reorder_grouped_qkv_to_qkv(
            weight,
            num_query_groups=2,
            heads_per_group=1,
            head_dim=2,
        )
        expected = torch.tensor(
            [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
            dtype=torch.float32,
        ).reshape(12, 1)
        torch.testing.assert_close(actual, expected)

    def test_flash_attention_backend_is_the_only_supported_backend(self):
        arch = MiniMaxH3DiTArchConfig()

        self.assertEqual(arch._supported_attention_backends, {AttentionBackendEnum.FA})

    def test_tp_gt_one_fails_fast(self):
        model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
        with self.assertRaisesRegex(ValueError, "TP=1 only"):
            model._validate_tp_config(arch=MiniMaxH3DiTArchConfig(), tp_size=2)

    def test_fp32_dtype_policy_on_meta_model(self):
        expected_fp32 = set(MINIMAX_H3_FP32_PARAM_NAMES) | set(MINIMAX_H3_FP32_BUFFER_NAMES)
        _ensure_single_process_parallel_runtime()
        with torch.device("meta"):
            model = MiniMaxH3DiTModel(
                config=MiniMaxH3DiTConfig(),
                hf_config={},
                quant_config=None,
            )
        for name, tensor in model.state_dict().items():
            if name in expected_fp32:
                self.assertEqual(tensor.dtype, torch.float32, name)
            elif tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.bfloat16, name)


class TestMiniMaxH3DiTForwardMath(unittest.TestCase):
    def test_sdpa_varlen_fallback_matches_naive_reference(self):
        torch.manual_seed(0)
        heads, dim = 2, 8
        cu = torch.tensor([0, 5, 6, 13], dtype=torch.int32)
        total = int(cu[-1])
        q = torch.randn(total, heads, dim)
        k = torch.randn(total, heads, dim)
        v = torch.randn(total, heads, dim)
        scale = dim**-0.5

        out = _sdpa_varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=scale)

        bounds = cu.tolist()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            seg_q = q[start:stop].transpose(0, 1)
            seg_k = k[start:stop].transpose(0, 1)
            seg_v = v[start:stop].transpose(0, 1)
            attn = torch.softmax(seg_q @ seg_k.transpose(-1, -2) * scale, dim=-1)
            ref = (attn @ seg_v).transpose(0, 1)
            self.assertTrue(
                torch.allclose(out[start:stop], ref, atol=1e-6),
                f"segment [{start}:{stop}] diverges from naive attention",
            )

    def test_rope_freq_layout_cat_thw_twice(self):
        rope = MiniMaxH3Rope(inv_freq_len=2)
        with torch.no_grad():
            rope.inv_freq.copy_(torch.tensor([1.0, 0.5]))
        pos = torch.tensor([[[2.0, 3.0, 5.0]]])  # [1, S=1, 3]
        freqs = rope(pos)
        # per axis: pos * inv_freq -> t=[2,1], h=[3,1.5], w=[5,2.5]; cat twice
        expected = torch.tensor([[2.0, 1.0, 3.0, 1.5, 5.0, 2.5] * 2])
        torch.testing.assert_close(freqs, expected)

    def test_apply_rope_partial_rotation_and_zero_identity(self):
        x = torch.randn(3, 2, 8)
        freqs = torch.zeros(3, 4)
        torch.testing.assert_close(_apply_rope(x, freqs), x)
        # quarter turn on the rotated half flips per rotate_half convention
        freqs = torch.full((3, 4), torch.pi / 2)
        out = _apply_rope(x, freqs)
        x1, x2 = x[..., :2], x[..., 2:4]
        torch.testing.assert_close(out[..., :2], -x2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out[..., 2:4], x1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out[..., 4:], x[..., 4:])

    def test_modulate_math_matches_reference_formulas(self):
        x = torch.randn(4, 3)
        shift = torch.randn(2, 3)
        scale = torch.randn(2, 3)
        gate = torch.randn(2, 3)
        other = torch.randn(4, 3)
        idx = torch.tensor([0, 1, 1, 0])
        got = _modulate_scale_shift(x, shift, scale, idx, dtype=torch.float32)
        expected = x * (1.0 + scale[idx]) + shift[idx]
        torch.testing.assert_close(got, expected)
        got = _modulate_gate(x, gate, other, idx, dtype=torch.float32)
        torch.testing.assert_close(got, x + gate[idx] * other)

    def test_multi_adaln_combined_index_row_layout(self):
        # MultiAdalnLinearProjCompressed layout: view(M*modality, expand*H)
        # rows ordered [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...].
        m, modality, expand, hidden = 2, 3, 6, 4
        flat = torch.arange(m * modality * expand * hidden, dtype=torch.float32)
        per_condition = flat.view(m, modality * expand * hidden)
        rows = per_condition.view(m * modality, expand * hidden)
        chunks = rows.chunk(expand, dim=-1)
        inverse_indices = torch.tensor([0, 1, 1])
        token_tags = torch.tensor([2, 0, -1])
        combined = inverse_indices * modality + token_tags.clamp(min=0)
        torch.testing.assert_close(combined, torch.tensor([2, 3, 3]))
        picked = chunks[0].index_select(0, combined)
        torch.testing.assert_close(picked[0], rows[2, :hidden])
        torch.testing.assert_close(picked[1], rows[3, :hidden])


if __name__ == "__main__":
    unittest.main()
