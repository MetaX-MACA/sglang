import unittest
from types import SimpleNamespace
from unittest import mock

import transformers

from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)


class TestTextEncoderClassResolution(unittest.TestCase):
    """load_native must not load encoder-decoder text encoders via AutoModel.

    AutoModel maps T5/UMT5 model types to the full seq2seq class
    (T5Model/UMT5Model), whose forward needs decoder inputs and raises when the
    module is used purely as a text encoder.
    """

    server_args = SimpleNamespace(trust_remote_code=False, revision=None)

    def _resolve(self, is_encoder_decoder, architectures):
        config = SimpleNamespace(
            is_encoder_decoder=is_encoder_decoder, architectures=architectures
        )
        with mock.patch.object(
            transformers.AutoConfig, "from_pretrained", return_value=config
        ):
            return TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )

    def test_umt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["UMT5EncoderModel"]), transformers.UMT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["UMT5Model"]), transformers.UMT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["UMT5ForConditionalGeneration"]),
            transformers.UMT5EncoderModel,
        )

    def test_t5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["T5EncoderModel"]), transformers.T5EncoderModel
        )
        self.assertIs(self._resolve(True, ["T5Model"]), transformers.T5EncoderModel)
        self.assertIs(
            self._resolve(True, ["T5ForConditionalGeneration"]),
            transformers.T5EncoderModel,
        )

    def test_mt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["MT5EncoderModel"]), transformers.MT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["MT5Model"]), transformers.MT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["MT5ForConditionalGeneration"]),
            transformers.MT5EncoderModel,
        )

    def test_non_encoder_decoder_keeps_automodel(self):
        # e.g. CLIP/Mistral/Qwen text encoders are not encoder-decoder.
        self.assertIs(self._resolve(False, ["CLIPTextModel"]), transformers.AutoModel)

    def test_unknown_architecture_falls_back_to_automodel(self):
        self.assertIs(self._resolve(True, ["NotARealClass"]), transformers.AutoModel)

    def test_config_load_failure_falls_back_to_automodel(self):
        with mock.patch.object(
            transformers.AutoConfig,
            "from_pretrained",
            side_effect=OSError("no config"),
        ):
            cls = TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )
        self.assertIs(cls, transformers.AutoModel)


if __name__ == "__main__":
    unittest.main()


class TestRegisteredEncoderDispatch(unittest.TestCase):
    """Registry-backed encoders must opt in via ``load_component``.

    ``load_customized`` may only short-circuit into the model registry when
    the resolved class actually defines the ``load_component`` classmethod
    (e.g. MiniMaxH3Qwen3VLHFEncoder). Every other registered text-encoder
    architecture (UMT5EncoderModel, CLIPTextModel, LlamaModel, ...) must keep
    flowing through the legacy per-encoder-config path; anything else aborts
    server startup for those models with an AttributeError.
    """

    def _load_customized(self, model_cls):
        from sglang.multimodal_gen.runtime.loader.component_loaders import (
            text_encoder_loader as tel,
        )

        loader = TextEncoderLoader()
        loader.component_architecture = "FakeRegisteredEncoder"
        legacy_sentinel = RuntimeError("legacy path reached")
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace())
        with mock.patch.object(
            tel.ModelRegistry,
            "registered_models",
            {"FakeRegisteredEncoder": object()},
        ), mock.patch.object(
            tel.ModelRegistry,
            "resolve_model_cls",
            return_value=(model_cls, None),
        ), mock.patch.object(
            tel, "get_diffusers_component_config", return_value={}
        ), mock.patch.object(
            tel, "get_config", side_effect=legacy_sentinel
        ) as legacy_probe:
            try:
                result = loader.load_customized(
                    "dummy/path", server_args, "text_encoder"
                )
            except RuntimeError as exc:
                if exc is not legacy_sentinel:
                    raise
                result = legacy_sentinel
        return result, legacy_probe

    def test_registered_arch_without_load_component_uses_legacy_path(self):
        class _PlainEncoder:  # no load_component -> legacy path
            pass

        result, legacy_probe = self._load_customized(_PlainEncoder)

        self.assertEqual(legacy_probe.call_count, 1)
        self.assertIsInstance(result, RuntimeError)

    def test_registered_arch_with_load_component_short_circuits(self):
        loaded = object()

        class _SelfLoadingEncoder:
            @classmethod
            def load_component(cls, *, component_model_path, component_name, config):
                assert component_model_path == "dummy/path"
                assert component_name == "text_encoder"
                return loaded

        result, legacy_probe = self._load_customized(_SelfLoadingEncoder)

        self.assertIs(result, loaded)
        legacy_probe.assert_not_called()
