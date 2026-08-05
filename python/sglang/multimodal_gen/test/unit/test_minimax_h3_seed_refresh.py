from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import expand_request_outputs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def test_expand_minimax_h3_outputs_refreshes_canonical_seed() -> None:
    req = Req(
        sampling_params=MiniMaxH3SamplingParams(
            request_id="rid",
            prompt="p",
            output_path="/tmp",
            output_file_name="video.mp4",
            num_outputs_per_prompt=2,
            seed=[7, 9],
            task="t2va",
            conditions=[],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )
    )
    req.sampling_params.apply_request_extra(req)

    outputs = expand_request_outputs(req)

    assert [item.seed for item in outputs] == [7, 9]
    assert [item.extra["minimax_h3_canonical_request"]["seed"] for item in outputs] == [
        7,
        9,
    ]
