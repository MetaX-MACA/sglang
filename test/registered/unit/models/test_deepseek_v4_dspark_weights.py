"""Quantized DSpark vocabulary weights must load rather than remain initialized."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDSparkVocabularyWeights(CustomTestCase):
    def make_model(self, own_vocab=True):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(n_routed_experts=0)
        model.num_fused_shared_experts = 0
        model.num_stages = 3
        model.uses_own_vocab_modules = own_vocab
        model.confidence_head = None
        model.embed_tokens = torch.nn.Embedding(7, 4)
        model.lm_head = torch.nn.Linear(4, 7, bias=False)
        model.lm_head.register_parameter(
            "weight_scale", torch.nn.Parameter(torch.zeros(7, 1))
        )
        for param in model.parameters():
            param.data.zero_()
        return model

    def test_loads_checkpoint_vocabulary_and_quantization_scale(self):
        """MTP-local vocabulary tensors previously fell through as unexpected keys."""
        model = self.make_model()
        embedding = torch.arange(28, dtype=torch.float32).reshape(7, 4)
        head = embedding + 100
        scale = torch.full((7, 1), 0.25)
        weights = [
            ("mtp.0.emb.tok_emb.weight", embedding),
            ("mtp.2.head.weight", head),
            ("mtp.2.head.weight_scale", scale),
            ("mtp.1.head.weight", torch.full_like(head, -1)),
        ]
        with envs.SGLANG_DSPARK_SEPARATE_EMBED_AND_LMHEAD.override(True):
            model.load_weights(weights)
        torch.testing.assert_close(model.embed_tokens.weight, embedding)
        torch.testing.assert_close(model.lm_head.weight, head)
        torch.testing.assert_close(model.lm_head.weight_scale, scale)

    def test_rejects_missing_quantized_head_scale(self):
        """Missing scales must fail loading before an uninitialized head is used."""
        model = self.make_model()
        with (
            envs.SGLANG_DSPARK_SEPARATE_EMBED_AND_LMHEAD.override(True),
            self.assertRaisesRegex(ValueError, "lm_head.weight_scale"),
        ):
            model.load_weights(
                [
                    ("mtp.0.emb.tok_emb.weight", torch.ones(7, 4)),
                    ("mtp.2.head.weight", torch.ones(7, 4)),
                ]
            )

    def test_rejects_sharing_when_checkpoint_owns_vocabulary(self):
        """A rotated draft vocabulary cannot silently reuse target vocabulary."""
        model = self.make_model(own_vocab=False)
        with (
            envs.SGLANG_DSPARK_SEPARATE_EMBED_AND_LMHEAD.override(False),
            self.assertRaisesRegex(ValueError, "SEPARATE_EMBED_AND_LMHEAD"),
        ):
            model.load_weights([("mtp.0.emb.tok_emb.weight", torch.ones(7, 4))])


if __name__ == "__main__":
    unittest.main()
