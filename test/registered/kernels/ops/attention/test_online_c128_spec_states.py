"""Online C128 ragged prefix banks and accepted-prefix commits."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.online_c128_mtp import (
    _jit_online_c128_mtp_module,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestOnlineC128SpecStates(CustomTestCase):
    def test_ragged_prefix_banks_and_reordered_commit(self):
        torch.manual_seed(2)
        device = "cuda"
        stride, max_verify, dim = 4, 4, 512
        seq = torch.tensor([0, 126, 128], device=device, dtype=torch.int64)
        reqs = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
        lengths = torch.tensor([2, 3, 0], device=device, dtype=torch.int32)
        starts = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
        inputs = torch.randn((5, 2 * dim), device=device)
        ape = torch.randn((128, dim), device=device)
        state = torch.full((stride * (max_verify + 1), dim * 3), -17.0, device=device)
        state[2, :dim] = 0.25
        state[2, dim : 2 * dim] = 3
        state[2, 2 * dim :] = 0.5
        before = state.clone()
        expected = state.cpu().clone()
        data, positional = inputs.cpu(), ape.cpu()
        for row, (seq_before, req, count, start) in enumerate(
            zip([0, 126, 128], [1, 2, 3], [2, 3, 0], [0, 2, 5])
        ):
            m, z, v = (
                expected[req].reshape(3, dim).clone()
                if seq_before % 128
                else torch.zeros((3, dim))
            )
            for step in range(count):
                pos = (seq_before + step) % 128
                kv, score = data[start + step].chunk(2)
                score = score + positional[pos]
                if pos == 0:
                    m, z, v = score, torch.ones_like(score), kv
                else:
                    new_m = torch.maximum(m, score)
                    old_z = z * (m - new_m).exp()
                    new_z = (score - new_m).exp()
                    v = (v * old_z + kv * new_z) / (old_z + new_z)
                    m, z = new_m, old_z + new_z
                if (seq_before + step + 1) % 128:
                    expected[req + (step + 1) * stride] = torch.cat((m, z, v))
        mapping = torch.zeros((stride, 256), device=device, dtype=torch.int32)
        pending = torch.full((stride,), -1, device=device, dtype=torch.int64)
        module = _jit_online_c128_mtp_module(dim, seq.dtype, reqs.dtype, state.dtype)
        module.mark_pending(seq, reqs, pending, 3, stride)
        module.write_prefix_states(
            inputs,
            seq,
            reqs,
            lengths,
            starts,
            mapping,
            ape,
            state,
            3,
            max_verify,
            stride,
        )
        torch.testing.assert_close(state.cpu(), expected, atol=2e-5, rtol=2e-5)
        # Request 2 accepts across the chunk boundary; request 1 rolls back its second draft.
        commit_reqs = torch.tensor([2, 1, 3], device=device, dtype=torch.int64)
        final_seq = torch.tensor([129, 1, 128], device=device, dtype=torch.int64)
        module.commit_pending(
            final_seq,
            commit_reqs,
            mapping,
            pending,
            state,
            3,
            max_verify,
            stride,
            stride,
        )
        torch.testing.assert_close(state[2], state[2 + 3 * stride])
        torch.testing.assert_close(state[1], state[1 + stride])
        torch.testing.assert_close(state[3], before[3])


if __name__ == "__main__":
    unittest.main()
