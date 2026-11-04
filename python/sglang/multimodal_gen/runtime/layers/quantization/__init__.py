# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Literal, get_args

from importlib import import_module

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)

# Backport the lazy-import approach in upstream PR #42249, retaining this
# branch's kitchen_int8 backend and custom registration API.
_CONFIG_IMPORTS = {
    "AutoRoundConfig": ("auto_round", "AutoRoundConfig"),
    "BitsAndBytesConfig": ("bitsandbytes", "BitsAndBytesConfig"),
    "KitchenInt8Config": ("configs.kitchen_int8_config", "KitchenInt8Config"),
    "Fp8Config": ("fp8", "Fp8Config"),
    "ModelOptFp8DiffusionConfig": ("modelopt_fp8", "ModelOptFp8Config"),
    "ModelOptFp4Config": ("modelopt_quant", "ModelOptFp4Config"),
    "ModelOptFp8Config": ("modelopt_quant", "ModelOptFp8Config"),
    "ModelSlimConfig": ("modelslim", "ModelSlimConfig"),
    "Mxfp4Config": ("mxfp4", "Mxfp4Config"),
    "NPUMXFP4Config": ("mxfp4_npu", "NPUMXFP4Config"),
    "MXFP8Config": ("mxfp8", "MXFP8Config"),
}


def __getattr__(name: str):
    if name not in _CONFIG_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = _CONFIG_IMPORTS[name]
    config = getattr(import_module(f"{__name__}.{module_name}"), class_name)
    globals()[name] = config
    return config


QuantizationMethods = Literal[
    "auto-round",
    "fp8",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "bitsandbytes",
    "modelslim",
    "mxfp8",
    "mxfp4",
    "mxfp4_npu",
    "kitchen_int8",
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {
    "auto-round": "AutoRoundConfig",
    "modelopt": "ModelOptFp8DiffusionConfig",
    "modelopt_fp8": "ModelOptFp8Config",
    "modelopt_fp4": "ModelOptFp4Config",
    "bitsandbytes": "BitsAndBytesConfig",
    "modelslim": "ModelSlimConfig",
    "fp8": "Fp8Config",
    "mxfp4": "Mxfp4Config",
    "mxfp8": "MXFP8Config",
    "mxfp4_npu": "NPUMXFP4Config",
    "kitchen_int8": "KitchenInt8Config",
}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.


    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists."
            )
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    config = _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]
    if isinstance(config, str):
        return globals()[config] if config in globals() else __getattr__(config)
    return config



__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
