# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import logging
from typing import List, Optional, TypeVar

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

AUTO_TRUNCATE_INPUT_TOKENS_ENV = "SGLANG_AUTO_TRUNCATE_INPUT_TOKENS"

_T = TypeVar("_T")


def get_auto_truncate_input_tokens() -> Optional[int]:
    """Return the configured head-tail input-token limit, if enabled."""
    max_input_tokens = envs.SGLANG_AUTO_TRUNCATE_INPUT_TOKENS.get()
    if max_input_tokens is None:
        return None
    if max_input_tokens <= 0:
        logger.warning(
            "%s=%r is not positive. Ignoring server-side input truncation.",
            AUTO_TRUNCATE_INPUT_TOKENS_ENV,
            max_input_tokens,
        )
        return None
    return max_input_tokens


def truncate_sequence_head_tail(values: List[_T], max_items: int) -> bool:
    """In-place head-tail truncation, preserving exactly max_items items."""
    if max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")
    if len(values) <= max_items:
        return False

    head_len = max_items // 2
    tail_len = max_items - head_len
    values[:] = values[:head_len] + values[-tail_len:]
    return True


def truncate_input_ids_head_tail(
    input_ids: List[int], max_input_tokens: int
) -> bool:
    return truncate_sequence_head_tail(input_ids, max_input_tokens)
