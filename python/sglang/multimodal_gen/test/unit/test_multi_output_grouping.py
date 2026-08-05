import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    expand_request_outputs,
    normalize_output_seeds,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


class CountingDedupStage(PipelineStage):
    deduplicated_output_fields = ("prompt_embeds",)
    deduplicated_tensor_tree_output_fields = ("timesteps",)
    deduplicated_deepcopy_output_fields = ("scheduler",)
    deduplicated_extra_tensor_tree_output_keys = ("mu",)

    def __init__(self):
        self.server_args = SimpleNamespace(
            comfyui_mode=True, enable_layerwise_nvtx_marker=False
        )
        self.forward_calls = 0

    def build_dedup_fingerprint(self, batch: Req, server_args):
        return batch.prompt

    def forward(self, batch: Req, server_args) -> Req:
        self.forward_calls += 1
        value = float(self.forward_calls)
        batch.prompt_embeds = [torch.tensor([value])]
        batch.timesteps = torch.tensor([value])
        batch.scheduler = {"state": [value]}
        batch.extra["mu"] = torch.tensor([value])
        return batch


class CountingLatentStage(LatentPreparationStage):
    def __init__(self):
        self.server_args = SimpleNamespace(
            comfyui_mode=True, enable_layerwise_nvtx_marker=False
        )
        self.prepare_group_calls = 0
        self.forward_calls = 0

    def build_dedup_fingerprint(self, batch: Req, server_args):
        return batch.prompt

    def _prepare_grouped_latents(
        self,
        batches: list[Req],
        server_args,
    ) -> Req:
        self.prepare_group_calls += 1
        first_batch = batches[0]
        first_batch.latents = torch.arange(len(batches), dtype=torch.float32).reshape(
            len(batches), 1, 1
        )
        first_batch.latent_ids = first_batch.latents + 10
        first_batch.raw_latent_shape = first_batch.latents.shape
        return first_batch

    def forward(self, batch: Req, server_args) -> Req:
        self.forward_calls += 1
        batch.latents = torch.tensor([[[100.0 + self.forward_calls]]])
        return batch


class TestMultiOutputGrouping(unittest.TestCase):
    def test_normalize_output_seeds_from_int(self):
        self.assertEqual(
            normalize_output_seeds(10, num_outputs_per_prompt=3),
            [10, 11, 12],
        )

    def test_normalize_output_seeds_from_per_prompt_list(self):
        self.assertEqual(
            normalize_output_seeds([3, 5], num_outputs_per_prompt=2),
            [3, 5],
        )

    def test_normalize_output_seeds_from_total_list(self):
        self.assertEqual(
            normalize_output_seeds(
                [1, 2, 3, 4],
                num_outputs_per_prompt=2,
                num_prompts=2,
                prompt_index=1,
            ),
            [3, 4],
        )

    def test_normalize_output_seeds_rejects_mismatched_list(self):
        with self.assertRaisesRegex(ValueError, r"seed list length"):
            normalize_output_seeds(
                [1, 2, 3],
                num_outputs_per_prompt=2,
                num_prompts=2,
                prompt_index=0,
            )

    def test_expand_request_outputs_splits_seed_and_output_name(self):
        req = Req(
            sampling_params=SamplingParams(
                request_id="rid",
                prompt="p",
                output_path="/tmp",
                output_file_name="image.png",
                num_outputs_per_prompt=2,
                seed=[100, 101],
            )
        )

        outputs = expand_request_outputs(req)

        self.assertEqual([item.seed for item in outputs], [100, 101])
        self.assertEqual([item.num_outputs_per_prompt for item in outputs], [1, 1])
        self.assertEqual(
            [item.output_file_name for item in outputs],
            ["image_0.png", "image_1.png"],
        )
        self.assertEqual(
            [item.request_id for item in outputs],
            ["rid:0", "rid:1"],
        )

    def test_split_batched_latents_uses_original_batched_tensor(self):
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        src = Req(sampling_params=SamplingParams(prompt="p"))
        dst = Req(sampling_params=SamplingParams(prompt="p"))
        src.latents = torch.tensor([[[1.0]], [[2.0]]])
        src.latent_ids = torch.tensor([[[10.0]], [[20.0]]])

        stage._split_batched_latents(src, [src, dst])

        self.assertTrue(torch.equal(src.latents, torch.tensor([[[1.0]]])))
        self.assertTrue(torch.equal(dst.latents, torch.tensor([[[2.0]]])))
        self.assertTrue(torch.equal(src.latent_ids, torch.tensor([[[10.0]]])))
        self.assertTrue(torch.equal(dst.latent_ids, torch.tensor([[[20.0]]])))

    def test_declarative_stage_dedup_runs_equivalent_request_once(self):
        stage = CountingDedupStage()
        reqs = [
            Req(sampling_params=SamplingParams(prompt="same")),
            Req(sampling_params=SamplingParams(prompt="same")),
            Req(sampling_params=SamplingParams(prompt="same")),
        ]

        results = stage.run_grouped_requests(reqs, SimpleNamespace())

        self.assertEqual(stage.forward_calls, 1)
        self.assertEqual(results, reqs)
        for req in reqs:
            self.assertTrue(torch.equal(req.prompt_embeds[0], torch.tensor([1.0])))
            self.assertTrue(torch.equal(req.timesteps, torch.tensor([1.0])))
            self.assertEqual(req.scheduler, {"state": [1.0]})
            self.assertTrue(torch.equal(req.extra["mu"], torch.tensor([1.0])))

        self.assertIsNot(reqs[0].prompt_embeds, reqs[1].prompt_embeds)
        self.assertIs(reqs[0].prompt_embeds[0], reqs[1].prompt_embeds[0])
        self.assertIsNot(reqs[0].timesteps, reqs[1].timesteps)
        self.assertIsNot(reqs[0].scheduler, reqs[1].scheduler)
        self.assertIsNot(reqs[0].extra["mu"], reqs[1].extra["mu"])

    def test_declarative_stage_dedup_runs_distinct_fingerprints_separately(self):
        stage = CountingDedupStage()
        reqs = [
            Req(sampling_params=SamplingParams(prompt="a")),
            Req(sampling_params=SamplingParams(prompt="b")),
        ]

        stage.run_grouped_requests(reqs, SimpleNamespace())

        self.assertEqual(stage.forward_calls, 2)
        self.assertTrue(torch.equal(reqs[0].prompt_embeds[0], torch.tensor([1.0])))
        self.assertTrue(torch.equal(reqs[1].prompt_embeds[0], torch.tensor([2.0])))

    def test_latent_grouped_path_batches_equivalent_requests_once(self):
        stage = CountingLatentStage()
        reqs = [
            Req(sampling_params=SamplingParams(prompt="same")),
            Req(sampling_params=SamplingParams(prompt="same")),
        ]

        results = stage.run_grouped_requests(reqs, SimpleNamespace())

        self.assertEqual(results, reqs)
        self.assertEqual(stage.prepare_group_calls, 1)
        self.assertEqual(stage.forward_calls, 0)
        self.assertTrue(torch.equal(reqs[0].latents, torch.tensor([[[0.0]]])))
        self.assertTrue(torch.equal(reqs[1].latents, torch.tensor([[[1.0]]])))
        self.assertTrue(torch.equal(reqs[0].latent_ids, torch.tensor([[[10.0]]])))
        self.assertTrue(torch.equal(reqs[1].latent_ids, torch.tensor([[[11.0]]])))

    def test_dynamic_batch_slice_error_on_batch_sized_value_propagates(self):
        scheduler = Scheduler.__new__(Scheduler)

        class _UnsliceableBatchValue:
            shape = (2, 4)

            def __getitem__(self, _item):
                raise TypeError("slicing unsupported")

        with self.assertRaisesRegex(RuntimeError, r"'output'.*_UnsliceableBatchValue"):
            scheduler._slice_batched_value(
                _UnsliceableBatchValue(), 0, 1, 2, field_name="output"
            )

    def test_dynamic_batch_scalar_metadata_is_still_deepcopied(self):
        scheduler = Scheduler.__new__(Scheduler)
        scalar_metadata = {"peak_memory_mb": 1.0}

        result = scheduler._slice_batched_value(
            scalar_metadata, 0, 1, 2, field_name="metrics"
        )

        self.assertEqual(result, scalar_metadata)
        self.assertIsNot(result, scalar_metadata)

    def test_count_first_dim_propagates_malformed_shape(self):
        class _MalformedShape:
            shape = 5  # not sized: len(shape) raises

        with self.assertRaises(TypeError):
            Scheduler._count_first_dim(_MalformedShape())

    def test_count_first_dim_keeps_genuine_scalars_and_zero_dim(self):
        self.assertIsNone(Scheduler._count_first_dim(None))
        self.assertIsNone(Scheduler._count_first_dim(1.5))
        self.assertIsNone(Scheduler._count_first_dim(torch.tensor(3.0)))
        self.assertEqual(Scheduler._count_first_dim(torch.zeros(2, 1)), 2)


if __name__ == "__main__":
    unittest.main()
