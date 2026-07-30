from __future__ import annotations

import bisect
import logging
from enum import Enum
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RaggedVerifyMode(str, Enum):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY_MODE.get()
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_RAGGED_VERIFY_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}"
    )


def ragged_verify_compact_enabled() -> bool:
    return read_ragged_verify_mode() == RaggedVerifyMode.COMPACT


def round_up_grid(total: int, grid: Sequence[int]) -> int:
    if not grid:
        raise ValueError("round_up_grid requires a non-empty grid")
    if total > grid[-1]:
        raise ValueError(
            f"total {total} exceeds max grid tier {grid[-1]}; "
            "the caller must reject this batch before selecting a graph tier"
        )
    index = bisect.bisect_left(grid, total)
    return grid[index]


def build_graph_num_tokens_grid(
    *,
    capture_bs: Sequence[int],
    num_tokens_per_bs: int,
    token_grid: Sequence[int],
) -> list[int]:
    if num_tokens_per_bs < 1:
        raise ValueError(f"num_tokens_per_bs must be >= 1, got {num_tokens_per_bs}")
    uniform_tiers = {bs * num_tokens_per_bs for bs in capture_bs}
    merged = uniform_tiers.union(token_grid)
    if any(tier < 1 for tier in merged):
        raise ValueError(
            f"graph num-tokens grid tiers must be >= 1, got {sorted(merged)}"
        )
    return sorted(merged)


def parse_compact_graph_verify_tiers(
    raw: Sequence[str], max_verify_len: int
) -> tuple[int, ...]:
    """Normalize average verify-length tiers for compact CUDA Graph capture."""
    if max_verify_len < 1:
        raise ValueError(f"max_verify_len must be >= 1, got {max_verify_len}")

    value = ",".join(raw).strip().lower()
    if value in ("", "auto"):
        seeds = tuple(
            tier for tier in (1, 2, 4, 6, max_verify_len) if tier <= max_verify_len
        )
    elif value == "full":
        seeds = tuple(range(1, max_verify_len + 1))
    else:
        try:
            seeds = tuple(
                int(part.strip()) for part in value.split(",") if part.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "SGLANG_DSPARK_COMPACT_GRAPH_VERIFY_TIERS must be auto, full, "
                f"or comma-separated integers, got {raw!r}."
            ) from exc

    tiers = tuple(sorted(set(seeds + (max_verify_len,))))
    if not tiers or tiers[0] < 1 or tiers[-1] > max_verify_len:
        raise ValueError(
            "SGLANG_DSPARK_COMPACT_GRAPH_VERIFY_TIERS must resolve inside "
            f"[1, {max_verify_len}], got raw={raw!r}, tiers={tiers!r}."
        )
    return tiers


class RaggedVerifyGraphSpec(msgspec.Struct, frozen=True):
    """One compact target-verify CUDA Graph shape.

    ``num_tokens`` is the physical flattened token count M, ``num_slots`` is
    the padded request count S, and ``average_verify_len`` is the configured
    tier that produced M = S * average_verify_len.
    """

    num_tokens: int
    num_slots: int
    average_verify_len: int


class RaggedVerifyGraphGrid(msgspec.Struct, frozen=True):
    """Sparse two-dimensional CUDA Graph grid for DSpark compact verify."""

    specs: tuple[RaggedVerifyGraphSpec, ...]
    verify_tiers: tuple[int, ...]
    max_verify_len: int

    @classmethod
    def from_capture_bs(
        cls,
        *,
        capture_bs: Sequence[int],
        max_verify_len: int,
        tier_config: Sequence[str],
    ) -> RaggedVerifyGraphGrid:
        verify_tiers = parse_compact_graph_verify_tiers(tier_config, max_verify_len)
        slots = tuple(sorted(set(int(bs) for bs in capture_bs)))
        if not slots or slots[0] < 1:
            raise ValueError(
                f"capture_bs must contain positive request tiers, got {capture_bs!r}"
            )
        specs = tuple(
            sorted(
                {
                    RaggedVerifyGraphSpec(
                        num_tokens=num_slots * average_verify_len,
                        num_slots=num_slots,
                        average_verify_len=average_verify_len,
                    )
                    for num_slots in slots
                    for average_verify_len in verify_tiers
                },
                key=lambda spec: (spec.num_tokens, spec.num_slots),
            )
        )
        return cls(
            specs=specs,
            verify_tiers=verify_tiers,
            max_verify_len=max_verify_len,
        )

    def select(
        self, *, runtime_bs: int, semantic_tokens: int
    ) -> Optional[RaggedVerifyGraphSpec]:
        if runtime_bs < 1 or semantic_tokens < runtime_bs:
            return None
        return next(
            (
                spec
                for spec in self.specs
                if spec.num_slots >= runtime_bs
                and spec.num_tokens >= semantic_tokens
                and spec.num_tokens <= runtime_bs * self.max_verify_len
            ),
            None,
        )

    def physical_tokens(
        self, *, runtime_bs: int, semantic_tokens: int
    ) -> Optional[int]:
        spec = self.select(runtime_bs=runtime_bs, semantic_tokens=semantic_tokens)
        return spec.num_tokens if spec is not None else None

    def get(
        self, *, num_tokens: int, num_slots: int
    ) -> Optional[RaggedVerifyGraphSpec]:
        return next(
            (
                spec
                for spec in self.specs
                if spec.num_tokens == num_tokens and spec.num_slots == num_slots
            ),
            None,
        )


class RaggedVerifyLayout(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    graph_num_tokens: int
    extend_start_loc: torch.Tensor
    qo_indptr_device: torch.Tensor
    verify_lens_cpu: Optional[list[int]] = None
    total_verify_tokens: Optional[int] = None
    qo_indptr_host: Optional[torch.Tensor] = None
    kv_indptr_host: Optional[torch.Tensor] = None
    kv_lens_host: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    max_kv_len: Optional[int] = None
    graph_num_slots: Optional[int] = None

    def __post_init__(self) -> None:
        if self.verify_lens_cpu is None:
            return
        if not self.verify_lens_cpu:
            raise ValueError("RaggedVerifyLayout requires at least one request")
        min_verify_len = min(self.verify_lens_cpu)
        if self.graph_num_slots is None:
            if min_verify_len < 1:
                raise ValueError(
                    "every semantic request must verify the anchor "
                    f"(verify_len >= 1), got {self.verify_lens_cpu}"
                )
        elif min_verify_len < 0:
            raise ValueError(
                "graph runtime verify lengths must be non-negative, got "
                f"{self.verify_lens_cpu}"
            )
        if self.total_verify_tokens != sum(self.verify_lens_cpu):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} != "
                f"sum(verify_lens_cpu) {sum(self.verify_lens_cpu)}"
            )
        if not (self.total_verify_tokens <= self.graph_num_tokens):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} exceeds "
                f"graph_num_tokens {self.graph_num_tokens}"
            )
        if self.graph_num_slots is not None:
            if self.graph_num_slots != len(self.verify_lens_cpu):
                raise ValueError(
                    f"graph_num_slots {self.graph_num_slots} != layout rows "
                    f"{len(self.verify_lens_cpu)}"
                )
            if self.total_verify_tokens != self.graph_num_tokens:
                raise ValueError(
                    "graph runtime layout must fill the selected token tier: "
                    f"total={self.total_verify_tokens}, graph={self.graph_num_tokens}"
                )

    @property
    def bs(self) -> int:
        return int(self.verify_lens.shape[0])

    @property
    def runtime_num_slots(self) -> int:
        return self.graph_num_slots or self.bs

    @classmethod
    def _assemble_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        verify_lens_cpu: Optional[list[int]] = None,
        total_verify_tokens: Optional[int] = None,
        graph_num_slots: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        from sglang.srt.speculative.dspark_components.kernels.qo_indptr import (
            BuildQoIndptr,
        )

        verify_lens = verify_lens.to(torch.int32)
        indptr = BuildQoIndptr.execute(verify_lens=verify_lens)
        return cls(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            extend_start_loc=indptr.extend_start_loc,
            qo_indptr_device=indptr.qo_indptr,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
            graph_num_slots=graph_num_slots,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        verify_lens_cpu: list[int],
        total_verify_tokens: int,
        graph_num_tokens: int,
        device: torch.device,
        graph_num_slots: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32, device=device)
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
            graph_num_slots=graph_num_slots,
        )

    @classmethod
    def from_verify_lens_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        total_verify_tokens: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            total_verify_tokens=total_verify_tokens,
        )

    @classmethod
    def from_verify_lens(
        cls,
        *,
        verify_lens_cpu: Sequence[int],
        device: torch.device,
        grid: Sequence[int],
        graph_num_tokens_floor: int = 0,
        graph_num_slots: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        bucket_input = max(total_verify_tokens, graph_num_tokens_floor)
        graph_num_tokens = round_up_grid(total=bucket_input, grid=grid)

        return cls._assemble(
            verify_lens_cpu=verify_lens_list,
            total_verify_tokens=total_verify_tokens,
            graph_num_tokens=graph_num_tokens,
            graph_num_slots=graph_num_slots,
            device=device,
        )

    @classmethod
    def uniform(
        cls,
        *,
        bs: int,
        num_draft_tokens: int,
        device: torch.device,
        grid: Sequence[int],
    ) -> RaggedVerifyLayout:
        return cls.from_verify_lens(
            verify_lens_cpu=[num_draft_tokens] * bs,
            device=device,
            grid=grid,
        )

    def padded_to_bucket(self, *, padded_bs: int) -> RaggedVerifyLayout:
        from sglang.srt.speculative.dspark_components.kernels.padded_to_bucket import (
            PaddedToBucket,
        )

        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
        )

        return RaggedVerifyLayout._assemble_device(
            verify_lens=padded,
            graph_num_tokens=self.graph_num_tokens,
            total_verify_tokens=self.graph_num_tokens,
            graph_num_slots=(
                padded_bs if self.graph_num_slots is not None else None
            ),
        )

    def padded_for_graph(
        self,
        *,
        spec: RaggedVerifyGraphSpec,
        max_verify_len: int,
        align_real_tokens: bool,
    ) -> RaggedVerifyLayout:
        """Build an exact ``(M, S)`` runtime layout without GPU synchronization.

        The original rows are semantic requests. Additional rows are graph-only
        virtual requests. Padding capacity is assigned to virtual rows first by
        default; alignment mode assigns it to real rows first so confidence-ranked
        verification can consume the selected graph tier.
        """
        if self.total_verify_tokens is None:
            raise ValueError("padded_for_graph requires a semantic token count")
        if max_verify_len < 1:
            raise ValueError(f"max_verify_len must be >= 1, got {max_verify_len}")
        semantic_bs = self.bs
        if spec.num_slots < semantic_bs:
            raise ValueError(
                f"graph slots {spec.num_slots} cannot hold {semantic_bs} requests"
            )
        if spec.num_tokens < self.total_verify_tokens:
            raise ValueError(
                f"graph tokens {spec.num_tokens} cannot hold "
                f"{self.total_verify_tokens} semantic tokens"
            )
        if spec.num_tokens > spec.num_slots * max_verify_len:
            raise ValueError(
                f"graph shape (M={spec.num_tokens}, S={spec.num_slots}) exceeds "
                f"per-request width {max_verify_len}"
            )

        if self.verify_lens_cpu is None:
            return self._padded_for_graph_device(
                spec=spec,
                max_verify_len=max_verify_len,
                align_real_tokens=align_real_tokens,
            )

        verify_lens = list(self.verify_lens_cpu) + [0] * (spec.num_slots - semantic_bs)
        if max(verify_lens, default=0) > max_verify_len:
            raise ValueError(
                f"semantic verify lengths exceed max {max_verify_len}: "
                f"{self.verify_lens_cpu}"
            )
        remaining = spec.num_tokens - sum(verify_lens)
        real_rows = list(range(semantic_bs))
        virtual_rows = list(range(semantic_bs, spec.num_slots))
        row_order = (
            real_rows + virtual_rows if align_real_tokens else virtual_rows + real_rows
        )
        for row in row_order:
            if remaining == 0:
                break
            add = min(max_verify_len - verify_lens[row], remaining)
            verify_lens[row] += add
            remaining -= add
        if remaining != 0:
            raise ValueError(
                f"cannot pack graph padding into (M={spec.num_tokens}, "
                f"S={spec.num_slots}, W={max_verify_len}); remaining={remaining}"
            )

        return RaggedVerifyLayout._assemble(
            verify_lens_cpu=verify_lens,
            total_verify_tokens=spec.num_tokens,
            graph_num_tokens=spec.num_tokens,
            graph_num_slots=spec.num_slots,
            device=self.verify_lens.device,
        )

    def _padded_for_graph_device(
        self,
        *,
        spec: RaggedVerifyGraphSpec,
        max_verify_len: int,
        align_real_tokens: bool,
    ) -> RaggedVerifyLayout:
        verify_lens = self.verify_lens.to(torch.int64)
        if spec.num_slots > self.bs:
            verify_lens = torch.nn.functional.pad(
                verify_lens, (0, spec.num_slots - self.bs)
            )
        capacity = (max_verify_len - verify_lens).clamp_min(0)
        if not align_real_tokens and spec.num_slots > self.bs:
            row_order = torch.cat(
                [
                    torch.arange(
                        self.bs,
                        spec.num_slots,
                        dtype=torch.int64,
                        device=verify_lens.device,
                    ),
                    torch.arange(
                        self.bs,
                        dtype=torch.int64,
                        device=verify_lens.device,
                    ),
                ]
            )
            ordered_lens = verify_lens[row_order]
            ordered_capacity = capacity[row_order]
        else:
            row_order = None
            ordered_lens = verify_lens
            ordered_capacity = capacity

        cumulative = torch.cumsum(ordered_capacity, dim=0)
        previous = cumulative - ordered_capacity
        needed = torch.tensor(
            spec.num_tokens - self.total_verify_tokens,
            dtype=torch.int64,
            device=verify_lens.device,
        )
        add = torch.minimum((needed - previous).clamp_min(0), ordered_capacity)
        ordered_lens = ordered_lens + add
        if row_order is not None:
            verify_lens = torch.empty_like(ordered_lens)
            verify_lens[row_order] = ordered_lens
        else:
            verify_lens = ordered_lens

        return RaggedVerifyLayout._assemble_device(
            verify_lens=verify_lens.to(torch.int32),
            graph_num_tokens=spec.num_tokens,
            total_verify_tokens=spec.num_tokens,
            graph_num_slots=spec.num_slots,
        )


def build_capture_verify_lens(
    *,
    num_tokens: int,
    num_slots: int,
    num_draft_tokens: int,
) -> list[int]:
    if num_slots < 1 or num_tokens < num_slots:
        raise ValueError(
            f"capture layout needs 1 <= num_slots <= num_tokens, got "
            f"num_slots={num_slots}, num_tokens={num_tokens}"
        )
    if num_tokens > num_slots * num_draft_tokens:
        raise ValueError(
            f"capture layout cannot pack num_tokens={num_tokens} into "
            f"{num_slots} rows of at most {num_draft_tokens} tokens"
        )
    base = num_tokens // num_slots
    rem = num_tokens - base * num_slots
    return [base + 1] * rem + [base] * (num_slots - rem)


class RaggedTargetVerifyGeometry(msgspec.Struct):
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: Optional[int]


def build_ragged_target_verify_geometry(
    *,
    seq_lens: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> RaggedTargetVerifyGeometry:
    cache_seqlens_int32 = (seq_lens + layout.verify_lens).to(torch.int32)
    cu_seqlens_q = layout.qo_indptr_device.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seq_len_q = (
        max(layout.verify_lens_cpu) if layout.verify_lens_cpu is not None else None
    )
    return RaggedTargetVerifyGeometry(
        cache_seqlens_int32=cache_seqlens_int32,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
    )
