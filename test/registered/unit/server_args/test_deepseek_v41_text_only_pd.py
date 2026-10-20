"""V4.1 language-model-only PD configuration validation."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook, model_hook
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV41TextOnlyPDPolicy(CustomTestCase):
    def validate(
        self,
        model_type="deepseek_v41",
        mode="prefill",
        arch="DeepseekV4ForCausalLM",
        **flags,
    ):
        cfg = NS(
            language_model_only=True,
            encoder_only=False,
            language_only=False,
            enable_prefix_mm_cache=False,
            enable_broadcast_mm_inputs_process=False,
            mm_enable_dp_encoder=False,
            disaggregation_mode=mode,
        )
        for name, value in flags.items():
            setattr(cfg, name, value)
        model = NS(hf_config=NS(model_type=model_type, architectures=[arch]))
        args = NS(
            LANGUAGE_MODEL_ONLY_ARCHITECTURES=ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES
        )
        with (
            patch.object(model_hook, "resolving_view", return_value=cfg),
            patch.object(model_hook, "model_config_of", return_value=model),
        ):
            model_hook.handle_language_model_only(args)

    def test_v41_modes(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                self.validate(mode=mode)

    def test_other_models_still_reject_pd(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.validate(model_type="cosmos3", arch="Cosmos3ForConditionalGeneration")

    def test_encoder_options_still_rejected(self):
        for flag in (
            "encoder_only",
            "language_only",
            "enable_prefix_mm_cache",
            "enable_broadcast_mm_inputs_process",
            "mm_enable_dp_encoder",
        ):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(ValueError, "cannot be combined"),
            ):
                self.validate(**{flag: True})

    def test_unknown_arch_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.validate(arch="UnknownArchitecture")


class TestDSV41CPPDPolicy(CustomTestCase):
    def validate(self, *, enable_prefill_cp, enable_dp_attention, **overrides):
        cfg = NS(
            enable_encoder_swa_bounded_replay=False,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            pp_size=1,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            dp_size=1,
            enable_dp_attention=enable_dp_attention,
            enable_prefill_cp=enable_prefill_cp,
            attn_cp_size=1,
            dcp_size=1,
            cp_strategy="interleave",
            tp_size=4,
            cuda_graph_config=NS(
                prefill=NS(backend=Backend.DISABLED, max_seq_len=None)
            ),
            enable_decoder_swa_bounded_replay=False,
        )
        cfg.__dict__.update(overrides)
        model = NS(hf_config=NS(model_type="deepseek_v41"))
        with (
            patch.object(deepseek_v4_hook, "resolving_view", return_value=cfg),
            patch.object(deepseek_v4_hook, "model_config_of", return_value=model),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=RaggedVerifyMode.STATIC,
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(NS())

    def test_prefill_cp_allows_internal_dp_attention_path(self):
        self.validate(enable_prefill_cp=True, enable_dp_attention=True)

    def test_decode_dp_attention_allowed(self):
        for dp in (1, 2, 4, 8):
            self.validate(
                enable_prefill_cp=False, enable_dp_attention=True,
                disaggregation_mode="decode", tp_size=8, dp_size=dp,
            )

    def test_prefill_cp_resolution(self):
        for cp in (1, 8):
            self.validate(
                enable_prefill_cp=True, enable_dp_attention=True,
                tp_size=8, attn_cp_size=cp,
            )

    def test_unsupported_cp_rejected(self):
        for changes in (
            dict(dp_size=2), dict(disaggregation_mode="decode"),
            dict(cp_strategy="other"), dict(attn_cp_size=2),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "DSpark PD requires"
            ):
                self.validate(
                    enable_prefill_cp=True, enable_dp_attention=True, **changes
                )


if __name__ == "__main__":
    unittest.main()
