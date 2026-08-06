from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, List, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_GAMMA = 7
SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")
_QWEN3_5_FAMILY_TEXT_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})
_QWEN3_5_FAMILY_TARGET_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
_QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT = ContextVar(
    "qwen3_5_family_dspark_draft_config", default=False
)


def is_qwen3_5_family_target(model_runner) -> bool:
    text_config = model_runner.model_config.hf_text_config
    architectures = model_runner.model_config.hf_config.architectures or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    return (
        text_config is not None
        and text_config.model_type in _QWEN3_5_FAMILY_TEXT_MODEL_TYPES
        and any(
            architecture in _QWEN3_5_FAMILY_TARGET_ARCHITECTURES
            for architecture in architectures
        )
    )


def is_qwen3_5_family_dspark_enabled(model_runner) -> bool:
    return (
        envs.SGLANG_ENABLE_DSPARK_QWEN3_5_FAMILY.get()
        and model_runner.spec_algorithm.is_dspark()
        and is_qwen3_5_family_target(model_runner)
    )


@contextmanager
def qwen3_5_family_draft_config_context(enabled: bool):
    token = _QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT.set(bool(enabled))
    try:
        yield
    finally:
        _QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT.reset(token)


def qwen3_5_family_draft_config_enabled() -> bool:
    return _QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT.get()


class DSparkLengthContract(msgspec.Struct, frozen=True):
    gamma: int

    @property
    def num_draft_positions(self) -> int:
        return int(self.gamma)

    @property
    def verify_num_draft_tokens(self) -> int:
        return int(self.gamma) + 1

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self.verify_num_draft_tokens

    def validate(self) -> None:
        if int(self.gamma) < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")


def make_dspark_length_contract(*, gamma: int) -> DSparkLengthContract:
    contract = DSparkLengthContract(gamma=int(gamma))
    contract.validate()
    return contract


def dspark_gamma_from_num_draft_tokens(num_draft_tokens: int) -> int:
    gamma = int(num_draft_tokens) - 1
    if gamma < 1:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {num_draft_tokens}."
        )
    return gamma


class DSparkDraftConfig(msgspec.Struct, frozen=True):
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    gamma: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]
    markov_rank: int
    markov_head_type: Optional[str]

    def resolve_gamma(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.gamma if self.gamma is not None else default

    def require_markov(self) -> bool:
        return int(self.markov_rank) > 0


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        text_config = config.get("text_config", None)
        if text_config is not None:
            return text_config
        transformer_layer_config = config.get("transformer_layer_config", None)
        return (
            transformer_layer_config if transformer_layer_config is not None else config
        )
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    transformer_layer_config = getattr(config, "transformer_layer_config", None)
    if transformer_layer_config is not None:
        return transformer_layer_config
    return config


def _get_dspark_config(config: Any) -> dict:
    cfg = _cfg_get(config, "dspark_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    try:
        return dict(cfg)
    except Exception:
        return {}

def _get_speculators_config(config: Any) -> dict:
    cfg = _cfg_get(config, "speculators_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    try:
        return dict(cfg)
    except Exception:
        return {}

def _get_speculators_gamma(config: Any) -> Optional[int]:
    speculators_cfg = _get_speculators_config(config)
    proposal_methods = speculators_cfg.get("proposal_methods") or []
    default_method = speculators_cfg.get("default_proposal_method")

    selected_method = None
    for method in proposal_methods:
        if not isinstance(method, dict) and not hasattr(method, "__dict__"):
            continue
        if (
            default_method is None
            or _cfg_get(method, "proposal_type", None) == default_method
        ):
            selected_method = method
            break
    if selected_method is None and proposal_methods:
        selected_method = next(
            (
                method
                for method in proposal_methods
                if isinstance(method, dict) or hasattr(method, "__dict__")
            ),
            None,
        )
    if selected_method is None:
        return None

    speculative_tokens = _cfg_get(selected_method, "speculative_tokens", None)
    if speculative_tokens is None:
        return None
    gamma = int(speculative_tokens)
    if gamma < 1:
        raise ValueError(
            "DSpark speculators_config speculative_tokens must be positive, "
            f"got {gamma}."
        )
    return gamma

def parse_dspark_draft_config(
    *, draft_hf_config: Any, enable_qwen3_5_family_aliases: Optional[bool] = None
) -> DSparkDraftConfig:
    base = parse_dflash_draft_config(draft_hf_config=draft_hf_config)

    dspark_cfg = _get_dspark_config(draft_hf_config)
    text_config = _get_text_config(draft_hf_config)

    prefixed_block_size = _cfg_get(draft_hf_config, "dspark_block_size", None)
    prefixed_markov_rank = _cfg_get(draft_hf_config, "dspark_markov_rank", None)
    prefixed_markov_head_type = _cfg_get(
        draft_hf_config, "dspark_markov_head_type", None
    )
    prefixed_noise_token_id = _cfg_get(draft_hf_config, "dspark_noise_token_id", None)
    prefixed_target_layer_ids = _cfg_get(
        draft_hf_config, "dspark_target_layer_ids", None
    )
    uses_prefixed = any(
        value is not None
        for value in (
            prefixed_block_size,
            prefixed_markov_rank,
            prefixed_noise_token_id,
            prefixed_target_layer_ids,
        )
    )

    raw_markov_rank = (
        prefixed_markov_rank
        if prefixed_markov_rank is not None
        else dspark_cfg.get(
            "markov_rank",
            _cfg_get(
                text_config, "markov_rank", _cfg_get(draft_hf_config, "markov_rank", 0)
            ),
        )
    )
    markov_rank = int(raw_markov_rank) if raw_markov_rank is not None else 0
    if markov_rank < 0:
        raise ValueError(f"DSpark markov_rank must be >= 0, got {markov_rank}.")

    markov_head_type = (
        prefixed_markov_head_type
        if prefixed_markov_head_type is not None
        else dspark_cfg.get(
            "markov_head_type",
            _cfg_get(
                text_config,
                "markov_head_type",
                _cfg_get(draft_hf_config, "markov_head_type", None),
            ),
        )
    )
    if markov_rank > 0 and markov_head_type is None and not uses_prefixed:
        raise ValueError(
            "DSpark requires markov_head_type when markov_rank > 0, got None."
        )
    if markov_head_type is not None:
        markov_head_type = str(markov_head_type).lower()
        if markov_head_type not in SUPPORTED_DSPARK_MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unsupported DSpark markov_head_type={markov_head_type!r}. "
                f"Supported: {SUPPORTED_DSPARK_MARKOV_HEAD_TYPES}."
            )

    raw_mask_token_id = (
        prefixed_noise_token_id
        if prefixed_noise_token_id is not None
        else dspark_cfg.get(
            "mask_token_id",
            _cfg_get(
                text_config,
                "mask_token_id",
                _cfg_get(draft_hf_config, "mask_token_id", base.mask_token_id),
            ),
        )
    )
    mask_token_id = int(raw_mask_token_id) if raw_mask_token_id is not None else None
    if mask_token_id is not None and mask_token_id < 0:
        raise ValueError(
            f"DSpark mask_token_id must be non-negative, got {mask_token_id}."
        )

    speculators_gamma = _get_speculators_gamma(draft_hf_config)
    if prefixed_block_size is not None:
        gamma = int(prefixed_block_size)
    elif speculators_gamma is not None:
        gamma = speculators_gamma
    else:
        gamma = base.block_size

    if prefixed_target_layer_ids is not None:
        if not isinstance(prefixed_target_layer_ids, (list, tuple)) or not len(
            prefixed_target_layer_ids
        ):
            raise ValueError(
                "DSpark dspark_target_layer_ids must be a non-empty list of ints, "
                f"got {prefixed_target_layer_ids!r}."
            )
        target_layer_ids: Optional[List[int]] = [
            int(x) for x in prefixed_target_layer_ids
        ]
    else:
        target_layer_ids = base.target_layer_ids

    enable_qwen3_5_family_aliases = (
        _QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT.get()
        if enable_qwen3_5_family_aliases is None
        else bool(enable_qwen3_5_family_aliases)
    )
    raw_target_num_layers = (
        _cfg_get(draft_hf_config, "target_num_hidden_layers", None)
        if enable_qwen3_5_family_aliases
        else None
    )
    if raw_target_num_layers is not None:
        try:
            num_target_layers = int(raw_target_num_layers)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "DSpark target_num_hidden_layers must be an integer, "
                f"got {raw_target_num_layers!r}."
            ) from e
    else:
        num_target_layers = base.num_target_layers
    if num_target_layers is not None and int(num_target_layers) <= 0:
        raise ValueError(
            "DSpark target_num_hidden_layers must be positive, "
            f"got {num_target_layers}."
        )

    return DSparkDraftConfig(
        num_hidden_layers=base.num_hidden_layers,
        num_target_layers=num_target_layers,
        gamma=gamma,
        target_layer_ids=target_layer_ids,
        mask_token=base.mask_token,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
    )


def make_qwen3_5_dspark_dflash_backbone_config(
    *,
    draft_hf_config: Any,
    dspark_config: DSparkDraftConfig,
    enable_qwen3_5_family_aliases: Optional[bool] = None,
) -> Any:
    """Adapt Qwen3.5-family DSpark aliases before DFlash backbone initialization."""
    enable_qwen3_5_family_aliases = (
        _QWEN3_5_FAMILY_DRAFT_CONFIG_CONTEXT.get()
        if enable_qwen3_5_family_aliases is None
        else bool(enable_qwen3_5_family_aliases)
    )
    if not enable_qwen3_5_family_aliases:
        return draft_hf_config
    if dspark_config.num_target_layers is None:
        return draft_hf_config

    try:
        num_target_layers = int(dspark_config.num_target_layers)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "DSpark num_target_layers must be an integer before DFlash backbone "
            f"initialization, got {dspark_config.num_target_layers!r}."
        ) from e
    if num_target_layers <= 0:
        raise ValueError(
            "DSpark num_target_layers must be positive before DFlash backbone "
            f"initialization, got {num_target_layers}."
        )

    raw_target_num_layers = _cfg_get(draft_hf_config, "target_num_hidden_layers", None)
    if (
        raw_target_num_layers is None
        and _cfg_get(draft_hf_config, "num_target_layers", None) is not None
    ):
        return draft_hf_config

    if raw_target_num_layers is not None:
        try:
            target_num_layers = int(raw_target_num_layers)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "DSpark target_num_hidden_layers must be an integer before DFlash "
                f"backbone initialization, got {raw_target_num_layers!r}."
            ) from e
        if target_num_layers <= 0:
            raise ValueError(
                "DSpark target_num_hidden_layers must be positive before DFlash "
                f"backbone initialization, got {target_num_layers}."
            )
        if target_num_layers != num_target_layers:
            raise ValueError(
                "DSpark parsed num_target_layers does not match "
                "target_num_hidden_layers before DFlash backbone initialization: "
                f"parsed={num_target_layers}, target_num_hidden_layers="
                f"{target_num_layers}."
            )

    if isinstance(draft_hf_config, dict):
        backbone_config = dict(draft_hf_config)
        backbone_config["num_target_layers"] = num_target_layers
    else:
        backbone_config = copy.copy(draft_hf_config)
        backbone_config.num_target_layers = num_target_layers
    return backbone_config
