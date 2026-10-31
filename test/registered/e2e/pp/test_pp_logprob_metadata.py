"""PP metadata must not restore a sender's CUDA storage during unpickling.

Nested top-logprob tensors previously bypassed the tensor transport and opened
the sender's device on receiving PP ranks, hanging the first logprob request.
"""

import pickle
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import TensorMetadata, _split_tensor_dict
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestPPLogprobMetadata(CustomTestCase):
    def _roundtrip(self, result):
        payload = get_logprob_dict_from_result(result)
        metadata, tensors = _split_tensor_dict(payload)
        locations = []
        restore = torch.serialization.default_restore_location

        def cpu_metadata_only(storage, location):
            self.assertEqual(location, "cpu", "PP metadata restored a sender GPU")
            locations.append(location)
            return restore(storage, location)

        with patch(
            "torch.serialization.default_restore_location",
            side_effect=cpu_metadata_only,
        ):
            received = pickle.loads(pickle.dumps(metadata))
        self.assertTrue(locations, "Exercise real tensor-storage unpickling")
        tensor_iter = iter(tensors)
        received = {
            key: next(tensor_iter) if isinstance(value, TensorMetadata) else value
            for key, value in received
        }
        return get_logprob_from_pp_outputs(PPProxyTensors(received)), payload, tensors

    def test_cuda_logprobs_use_cpu_metadata_and_tensor_transport(self):
        top = torch.tensor([-0.25, -1.5], device="cuda")
        ids = torch.tensor([3, 7], device="cuda")
        next_logprobs = torch.tensor([-0.25], device="cuda")
        span_source = torch.tensor([[-0.5, -1.0], [-2.0, -3.0]], device="cuda")
        span = torch.empty((2, 2), pin_memory=True)
        copy_stream = torch.cuda.Stream()
        copy_stream.wait_stream(torch.cuda.current_stream())
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            span.copy_(span_source, non_blocking=True)
            copy_done.record()
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=next_logprobs,
            next_token_top_logprobs_val=[top, []],
            next_token_top_logprobs_idx=[ids, []],
            next_token_token_ids_logprobs_val=[top[:1], None],
            next_token_token_ids_logprobs_idx=[ids[:1], None],
            input_top_logprobs_val=[[span]],
            input_logprobs_copy_done=copy_done,
        )
        result = GenerationBatchResult(
            logits_output=output,
            extend_input_len_per_req=[2],
            extend_logprob_start_len_per_req=[0],
        )
        (decoded, lengths, starts), payload, tensors = self._roundtrip(result)
        self.assertIs(payload["next_token_logprobs"], next_logprobs)
        self.assertTrue(any(value is next_logprobs for value in tensors))
        self.assertEqual(decoded.next_token_top_logprobs_val[0].device.type, "cpu")
        torch.testing.assert_close(decoded.next_token_top_logprobs_val[0], top.cpu())
        torch.testing.assert_close(decoded.next_token_top_logprobs_idx[0], ids.cpu())
        torch.testing.assert_close(
            decoded.next_token_token_ids_logprobs_val[0], top[:1].cpu()
        )
        torch.testing.assert_close(
            decoded.next_token_token_ids_logprobs_idx[0], ids[:1].cpu()
        )
        self.assertEqual(decoded.next_token_top_logprobs_val[1], [])
        self.assertIsNone(decoded.next_token_token_ids_logprobs_val[1])
        self.assertEqual(decoded.input_top_logprobs_val, [[[-0.5, -1.0], [-2.0, -3.0]]])
        self.assertIsNone(output.input_logprobs_copy_done)
        self.assertEqual((lengths, starts), ([2], [0]))

    def test_nested_metadata_keeps_containers_and_empty_values(self):
        nested = ({"rows": [torch.tensor([1.0]), None, []], "empty": ()},)
        output = LogitsProcessorOutput(
            next_token_logits=None, next_token_top_logprobs_val=nested
        )
        (decoded, lengths, starts), _, _ = self._roundtrip(
            GenerationBatchResult(logits_output=output)
        )
        received = decoded.next_token_top_logprobs_val
        self.assertIsInstance(received, tuple)
        self.assertIsInstance(received[0], dict)
        self.assertIsInstance(received[0]["rows"], list)
        torch.testing.assert_close(received[0]["rows"][0], torch.tensor([1.0]))
        self.assertIsNone(received[0]["rows"][1])
        self.assertEqual(received[0]["rows"][2], [])
        self.assertEqual(received[0]["empty"], ())
        self.assertIsNone(decoded.input_top_logprobs_val)
        self.assertIsNone(lengths)
        self.assertIsNone(starts)


if __name__ == "__main__":
    unittest.main()
