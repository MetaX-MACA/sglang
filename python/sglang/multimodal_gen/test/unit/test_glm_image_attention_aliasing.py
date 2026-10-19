import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.glm_image import GlmImageAttention


class _Projection(torch.nn.Module):
    def forward(self, inputs):
        return inputs.clone(), None


class _InplaceRotary(torch.nn.Module):
    def forward(self, query, key, **kwargs):
        outputs = []
        for tensor, scale in [(query, 2), (key, 3)]:
            batch, tokens, heads, dim = tensor.shape
            flat = tensor.reshape(batch * tokens, heads * dim).contiguous()
            flat.mul_(scale)
            outputs.append(flat.view(batch, tokens, heads, dim))
        return tuple(outputs)


class _Attention(torch.nn.Module):
    def forward(self, query, key, value, **kwargs):
        self.key = key.clone()
        return query


class TestGlmImageAttentionAliasing(unittest.TestCase):
    def _check_attention(self, batch_size):
        attention = GlmImageAttention.__new__(GlmImageAttention)
        torch.nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.num_local_kv_heads = 2
        attention.to_q = _Projection()
        attention.to_k = _Projection()
        attention.to_v = _Projection()
        attention.to_out = torch.nn.ModuleList([_Projection()])
        attention.norm_q = None
        attention.norm_k = None
        attention.rotary_emb = _InplaceRotary()
        attention.attn = _Attention()

        text = torch.arange(batch_size * 2 * 4).reshape(batch_size, 2, 4).float()
        image = torch.arange(batch_size * 3 * 4).reshape(batch_size, 3, 4).float()
        image = image + 100
        output, text_output = attention(
            image, text, image_rotary_emb=(torch.ones(3, 1), torch.zeros(3, 1))
        )

        self.assertTrue(torch.equal(output, image * 2))
        self.assertTrue(torch.equal(text_output, text))
        expected_key = torch.cat([text, image * 3], dim=1).unflatten(2, (2, 2))
        self.assertTrue(torch.equal(attention.attn.key, expected_key))

    def test_single_batch_inplace_rope_preserves_text_prefix(self):
        self._check_attention(1)

    def test_multiple_batches_preserve_text_prefix(self):
        self._check_attention(2)


if __name__ == "__main__":
    unittest.main()
