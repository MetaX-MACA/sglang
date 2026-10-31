"""Physical C4 cache transfers must preserve BF16 rows and split FP8 scales."""

import unittest

import torch

from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "C4 transfer needs a GPU")
class TestHiSparseC4Transfer(CustomTestCase):
    def test_bf16_and_fp8_per_layer_transfer_with_rejected_slots(self):
        for row_bytes in (1024, 584):
            with self.subTest(row_bytes=row_bytes):
                page_bytes = 64 * 1024 if row_bytes == 1024 else 37440
                # Integer byte patterns catch copying only a row prefix or using
                # the FP8 scale offset when the source is a plain BF16 row.
                src = [
                    (
                        (torch.arange(page_bytes * 3, device="cuda") + layer * 19) % 251
                    ).to(torch.uint8)
                    for layer in range(2)
                ]
                dst = [torch.full_like(buf, 255) for buf in src]
                src_idx = torch.tensor(
                    [[63, 64, 129, -1], [1, 126, 128, 10]], device="cuda"
                )
                dst_idx = torch.tensor(
                    [[130, 65, 62, 0], [66, 127, 2, -1]], device="cuda"
                )
                transfer_cache_dsv4_mla(
                    torch.tensor(
                        [buf.data_ptr() for buf in src],
                        dtype=torch.uint64,
                        device="cuda",
                    ),
                    torch.tensor(
                        [buf.data_ptr() for buf in dst],
                        dtype=torch.uint64,
                        device="cuda",
                    ),
                    src_idx,
                    dst_idx,
                    item_size_bytes=row_bytes,
                )
                for layer in range(2):
                    expected = torch.full_like(src[layer], 255).cpu()
                    source = src[layer].cpu()
                    for source_idx, target_idx in zip(
                        src_idx[layer].tolist(), dst_idx[layer].tolist()
                    ):
                        if source_idx < 0 or target_idx < 0:
                            continue
                        if row_bytes == 1024:
                            expected[target_idx * 1024 : (target_idx + 1) * 1024] = (
                                source[source_idx * 1024 : (source_idx + 1) * 1024]
                            )
                        else:
                            for run_offset, size in ((0, 576), (64 * 576, 8)):
                                source_offset = (
                                    source_idx // 64 * page_bytes
                                    + run_offset
                                    + source_idx % 64 * size
                                )
                                target_offset = (
                                    target_idx // 64 * page_bytes
                                    + run_offset
                                    + target_idx % 64 * size
                                )
                                expected[target_offset : target_offset + size] = source[
                                    source_offset : source_offset + size
                                ]
                    torch.testing.assert_close(dst[layer].cpu(), expected)

    def test_registered_host_pointer_round_trip(self):
        from sglang.srt.mem_cache.pool_host.common import (
            HostTensorAllocator,
            _cuda_host_unregister,
            alloc_with_host_register,
            make_kernel_ptr_table,
        )

        for row_bytes in (1024, 584):
            page_bytes = 65536 if row_bytes == 1024 else 37440
            host = alloc_with_host_register(
                (2, 3 * page_bytes), torch.uint8, "cpu", True, HostTensorAllocator()
            )
            try:
                pattern = (torch.arange(host.numel()).reshape_as(host) % 251).to(
                    torch.uint8
                )
                host.copy_(pattern)
                layers = [host[layer] for layer in range(2)]
                host_ptrs = make_kernel_ptr_table(
                    layers, "cuda", host_memory_registered=True
                )
                device = [
                    torch.full((3 * page_bytes,), 255, dtype=torch.uint8, device="cuda")
                    for _ in layers
                ]
                device_ptrs = make_kernel_ptr_table(
                    device, "cuda", host_memory_registered=False
                )
                src = torch.tensor([[63, 64, 129], [1, 126, 128]], device="cuda")
                dst = torch.tensor([[130, 65, 62], [66, 127, 2]], device="cuda")

                def reference_copy(target, source, source_ids, target_ids):
                    for layer in range(2):
                        for source_id, target_id in zip(
                            source_ids[layer].tolist(), target_ids[layer].tolist()
                        ):
                            runs = (
                                ((0, 1024),)
                                if row_bytes == 1024
                                else ((0, 576), (64 * 576, 8))
                            )
                            for run, size in runs:
                                source_offset = (
                                    source_id // 64 * page_bytes
                                    + run
                                    + source_id % 64 * size
                                )
                                target_offset = (
                                    target_id // 64 * page_bytes
                                    + run
                                    + target_id % 64 * size
                                )
                                target[layer, target_offset : target_offset + size] = (
                                    source[layer, source_offset : source_offset + size]
                                )

                expected_device = torch.full_like(pattern, 255)
                reference_copy(expected_device, pattern, src.cpu(), dst.cpu())
                transfer_cache_dsv4_mla(
                    host_ptrs, device_ptrs, src, dst, item_size_bytes=row_bytes
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    torch.stack([buf.cpu() for buf in device]), expected_device
                )
                host.zero_()
                expected_host = torch.zeros_like(pattern)
                reference_copy(expected_host, expected_device, dst.cpu(), src.cpu())
                transfer_cache_dsv4_mla(
                    device_ptrs, host_ptrs, dst, src, item_size_bytes=row_bytes
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(host, expected_host)
            finally:
                _cuda_host_unregister(host)


if __name__ == "__main__":
    unittest.main()
