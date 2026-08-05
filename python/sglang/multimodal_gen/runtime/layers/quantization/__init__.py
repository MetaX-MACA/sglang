# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes import (
    BitsAndBytesConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
    ModelOptFp8Config as ModelOptFp8DiffusionConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelslim import ModelSlimConfig
from sglang.multimodal_gen.runtime.layers.quantization.mxfp4 import Mxfp4Config
from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_npu import (
    NPUMXFP4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8_npu import MXFP8Config

QuantizationMethods = Literal[
    "fp8",
    "modelopt",
    "modelopt_fp8",
    "modelopt_fp4",
    "bitsandbytes",
    "modelslim",
    "mxfp8",
    "mxfp4",
    "mxfp4_npu",
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

_BUILTIN_METHOD_TO_CONFIG: dict[str, type[QuantizationConfig]] = {
    "modelopt": ModelOptFp8DiffusionConfig,
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "bitsandbytes": BitsAndBytesConfig,
    "modelslim": ModelSlimConfig,
    "fp8": Fp8Config,
    "mxfp4": Mxfp4Config,
    "mxfp8": MXFP8Config,
    "mxfp4_npu": NPUMXFP4Config,
}

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type[QuantizationConfig]] = {}


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

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    return _BUILTIN_METHOD_TO_CONFIG[quantization]


__all__ = [
    "BitsAndBytesConfig",
    "Fp8Config",
    "ModelOptFp8DiffusionConfig",
    "ModelOptFp8Config",
    "ModelOptFp4Config",
    "ModelSlimConfig",
    "Mxfp4Config",
    "NPUMXFP4Config",
    "MXFP8Config",
    "QuantizationMethods",
    "QuantizationConfig",
    "register_quantization_config",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
