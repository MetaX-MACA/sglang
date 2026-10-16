import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import safe_open, save_file

from sglang.multimodal_gen.runtime.loader import rank_local_checkpoint as checkpoint
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
    MixedRowParallelLinear,
)


class TestRankLocalMixedRowCheckpoint(unittest.TestCase):
    def test_mixed_row_shards_match_custom_weight_loader_on_every_rank(self):
        weight = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)
        bias = torch.arange(3, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights.safetensors")
            save_file({"proj.weight": weight, "proj.bias": bias}, path)
            for rank in range(4):
                with self.subTest(rank=rank):
                    # Exercise the production weight loader without initializing
                    # a GPU process group or allocating a full Hunyuan model.
                    projection = MixedRowParallelLinear.__new__(MixedRowParallelLinear)
                    torch.nn.Module.__init__(projection)
                    projection.input_sizes = [8, 24]
                    projection.tp_size = 4
                    projection.tp_rank = rank
                    projection.weight = torch.nn.Parameter(torch.empty(3, 8))
                    projection.weight.input_dim = 1
                    projection.weight.weight_loader = projection.weight_loader
                    projection.bias = torch.nn.Parameter(torch.empty(3))
                    projection.bias.weight_loader = projection.weight_loader
                    projection.weight_loader(projection.weight, weight)
                    projection.weight_loader(projection.bias, bias)
                    model = torch.nn.Module()
                    model.add_module("proj", projection)
                    with (
                        patch.object(checkpoint, "get_tp_world_size", return_value=4),
                        patch.object(checkpoint, "get_tp_rank", return_value=rank),
                        patch.object(torch.distributed, "get_rank", return_value=rank),
                    ):
                        result = checkpoint.try_load_rank_local_tp_state_dict(
                            model, [path], lambda name: (name, None, None)
                        )
                    self.assertIsNotNone(result)
                    state, _ = result
                    expected = torch.cat(
                        [
                            weight[:, rank * 2 : (rank + 1) * 2],
                            weight[:, 8 + rank * 6 : 8 + (rank + 1) * 6],
                        ],
                        dim=1,
                    )
                    torch.testing.assert_close(state["proj.weight"].tensor, expected)
                    torch.testing.assert_close(
                        state["proj.weight"].tensor, projection.weight
                    )
                    torch.testing.assert_close(state["proj.bias"].tensor, bias)

    def test_fused_outputs_and_plain_rows_keep_existing_partition_rules(self):
        matrix = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights.safetensors")
            save_file({"weight": matrix}, path)
            source = checkpoint.SafetensorsSource(
                path, "weight", tuple(matrix.shape), "F32", None, None
            )
            with safe_open(path, framework="pt", device="cpu") as handle:
                for rank in range(2):
                    with self.subTest(rank=rank):
                        output_shard = checkpoint.read_tp_local_tensor(
                            [source],
                            {path: handle},
                            0,
                            rank,
                            2,
                            output_sizes=[4, 4, 4],
                        )
                        expected = torch.cat(
                            [
                                matrix[start + rank * 2 : start + (rank + 1) * 2]
                                for start in [0, 4, 8]
                            ],
                            dim=0,
                        )
                        torch.testing.assert_close(output_shard, expected)
                        input_shard = checkpoint.read_tp_local_tensor(
                            [source],
                            {path: handle},
                            1,
                            rank,
                            2,
                        )
                        torch.testing.assert_close(
                            input_shard, matrix[:, rank * 4 : (rank + 1) * 4]
                        )


if __name__ == "__main__":
    unittest.main()
