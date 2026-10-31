"""C4 verify boundary routing and accepted-token commit on CPU."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    finalize_hisparse_spec,
    prepare_hisparse_spec_verify,
)
from sglang.srt.mem_cache.allocator.hisparse import HiSparseSpecLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiSparseC4Spec(CustomTestCase):
    def test_verify_routes_boundaries_and_ignores_invalid_padding(self):
        hot, width = 64, 5
        reqs = torch.tensor([1, 2, 3, 4, 999])
        prefix = torch.tensor([8, 9, 10, 11, -99])
        table = torch.arange(5 * 128).view(5, 128)
        tokens = torch.full((2, 5, 128), -1, dtype=torch.int32)
        mapping = torch.full((100,), -1, dtype=torch.int64)
        out_locs = torch.arange(25).view(5, width) + 10
        prepare_hisparse_spec_verify(
            req_pool_indices=reqs,
            prefix_lens=prefix,
            hisparse_out_locs=out_locs.flatten(),
            req_to_device_buffer=table,
            device_buffer_tokens=tokens,
            num_real_reqs=torch.tensor([4], dtype=torch.int32),
            logical_to_device_mapping=mapping,
            verify_width=width,
            hot_buffer_size=hot,
        )
        expected = torch.full_like(mapping, -1)
        expected[0] = 0
        for row, start in enumerate(prefix[:4].tolist()):
            count = start // 4
            for step in range(width):
                if (start + step + 1) % 4 == 0:
                    expected[out_locs[row, step]] = table[
                        row + 1, hot + 1 + (start + step + 1) // 4 - 1 - count
                    ]
        torch.testing.assert_close(mapping, expected)
        self.assertTrue((tokens[:, 0] == -1).all())
        torch.testing.assert_close(
            tokens[:, 4, hot + 1 : hot + 3],
            torch.tensor([[2, 3], [2, 3]], dtype=torch.int32),
        )

    def test_commit_backs_up_only_accepted_c4_and_clears_provisional_routes(self):
        hot, width = 64, 5
        reqs = torch.tensor([1, 2])
        prefix = torch.tensor([255, 256])
        commits = torch.tensor([5, 0], dtype=torch.int32)
        locs = torch.arange(3 * 128, dtype=torch.int32).view(1, 3, 128).repeat(2, 1, 1)
        locs[1] += 1000  # Independent per-layer rotations must survive planning.
        tokens = torch.full_like(locs, -1)
        host = torch.arange(3 * 100).view(3, 100)
        req_to_token = torch.arange(3 * 300, dtype=torch.int32).view(3, 300)
        mapping = torch.ones(1000, dtype=torch.int64)
        calls = []

        def capture(*args, **kwargs):
            calls.append((args[2].clone(), args[3].clone(), kwargs["item_size_bytes"]))

        with patch(
            "sglang.kernels.ops.kvcache.hisparse.transfer_cache_dsv4_mla",
            side_effect=capture,
        ):
            finalize_hisparse_spec(
                device_ptrs=torch.zeros(2, dtype=torch.uint64),
                host_ptrs=torch.ones(2, dtype=torch.uint64),
                req_pool_indices=reqs,
                prefix_lens=prefix,
                commit_lens=commits,
                req_to_host_pool=host,
                device_buffer_locs=locs,
                req_to_token=req_to_token,
                device_buffer_tokens=tokens,
                logical_to_device_mapping=mapping,
                verify_width=width,
                hot_buffer_size=hot,
                item_size_bytes=1024,
            )
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            calls[0][0],
            locs[:, 1, hot + 1 : hot + 3]
            .to(torch.int64)
            .repeat(1, 2)
            .index_fill(1, torch.tensor([2, 3]), -1),
        )
        torch.testing.assert_close(calls[0][1], torch.tensor([163, 164, -1, -1]))
        self.assertEqual(calls[0][2], 1024)
        torch.testing.assert_close(
            tokens[:, 1, 63:65], torch.tensor([[63, 64], [63, 64]], dtype=torch.int32)
        )
        self.assertTrue((tokens[:, 2] == -1).all())
        for row, positions in [(1, [255, 259]), (2, [259])]:
            for pos in positions:
                self.assertEqual(mapping[req_to_token[row, pos] // 4], 0)

    def test_c4_scratch_budget_includes_page_rounding_and_verify_tail(self):
        layout = HiSparseSpecLayout(
            hot_slots=128, page_size=64, compress_ratio=4, verify_width=5, top_k=33
        )
        self.assertEqual(layout.scratch_slots, 192)
        self.assertEqual(layout.speculative_slots, 2)
        self.assertEqual(layout.metadata_width(3) % 2, 0)
        self.assertGreater(layout.fixed_state_bytes(num_layers=2, num_req_slots=3), 0)
        with self.assertRaises(ValueError):
            HiSparseSpecLayout(
                hot_slots=128, page_size=64, compress_ratio=4, verify_width=256, top_k=1
            )


if __name__ == "__main__":
    unittest.main()
