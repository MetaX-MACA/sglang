import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sglang.multimodal_gen.runtime.entrypoints.openai.utils as openai_utils
import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import expand_request_outputs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


class _SchedulerClient:
    def __init__(self, result: OutputBatch):
        self.result = result
        self.requests = None

    async def forward(self, requests):
        self.requests = requests
        return self.result


@pytest.fixture(autouse=True)
def _server_args(monkeypatch):
    server_args = SimpleNamespace(
        batching_max_size=1,
        pipeline_config=SimpleNamespace(requires_audio_output=False),
    )
    monkeypatch.setattr(openai_utils, "get_global_server_args", lambda: server_args)


def _req(
    output_path: Path,
    *,
    request_id: str,
    seed: int,
    num_outputs_per_prompt: int = 1,
) -> Req:
    return Req(
        sampling_params=SamplingParams(
            data_type=DataType.VIDEO,
            prompt="prompt",
            request_id=request_id,
            seed=seed,
            output_path=str(output_path.parent),
            output_file_name=output_path.name,
            num_outputs_per_prompt=num_outputs_per_prompt,
            save_output=True,
        )
    )


@pytest.mark.parametrize("num_outputs", [2, 3, 10])
def test_expanded_raw_output_fallback_uses_each_request_path_and_audio(
    tmp_path,
    monkeypatch,
    num_outputs,
):
    parent = _req(
        tmp_path / "unused" / "video.mp4",
        request_id="parent-request",
        seed=100,
        num_outputs_per_prompt=num_outputs,
    )
    requests = expand_request_outputs(parent)
    for output_index, request in enumerate(requests):
        request.output_path = str(tmp_path / f"output-{output_index}")

    outputs = [f"frames-{output_index}" for output_index in range(num_outputs)]
    audio = torch.stack(
        [torch.full((2, 4), float(output_index)) for output_index in range(num_outputs)]
    )
    client = _SchedulerClient(
        OutputBatch(output=outputs, audio=audio, audio_sample_rate=32000)
    )
    materialized = []

    def fake_post_process_sample(
        sample,
        _data_type,
        _fps,
        _save_output,
        save_file_path,
        **_kwargs,
    ):
        materialized.append((sample, save_file_path))
        return []

    monkeypatch.setattr(output_utils, "post_process_sample", fake_post_process_sample)

    output_paths, _ = asyncio.run(
        openai_utils.process_generation_batch(client, requests)
    )

    expected_paths = [request.output_file_path(1, 0) for request in requests]
    assert output_paths == expected_paths
    assert client.requests is requests
    assert [request.request_id for request in client.requests] == [
        f"parent-request:{output_index}" for output_index in range(num_outputs)
    ]
    assert [request.seed for request in client.requests] == list(
        range(100, 100 + num_outputs)
    )
    assert [path for _, path in materialized] == expected_paths
    assert [sample[0] for sample, _ in materialized] == outputs
    for output_index, (sample, _) in enumerate(materialized):
        torch.testing.assert_close(sample[1], audio[output_index])


def test_single_request_native_multioutput_keeps_basename_numbering(
    tmp_path,
    monkeypatch,
):
    request = _req(
        tmp_path / "native" / "video.mp4",
        request_id="native-request",
        seed=7,
    )
    client = _SchedulerClient(OutputBatch(output=["a", "b", "c"]))
    materialized_paths = []

    def fake_post_process_sample(
        _sample,
        _data_type,
        _fps,
        _save_output,
        save_file_path,
        **_kwargs,
    ):
        materialized_paths.append(save_file_path)
        return []

    monkeypatch.setattr(output_utils, "post_process_sample", fake_post_process_sample)

    output_paths, _ = asyncio.run(
        openai_utils.process_generation_batch(client, request)
    )

    expected_paths = [request.output_file_path(3, index) for index in range(3)]
    assert output_paths == expected_paths
    assert materialized_paths == expected_paths


@pytest.mark.parametrize("raw_output_count", [2, 4])
def test_expanded_raw_output_count_mismatch_fails_before_writes(
    tmp_path,
    monkeypatch,
    raw_output_count,
):
    requests = [
        _req(
            tmp_path / f"output-{index}" / "video.mp4",
            request_id=f"request-{index}",
            seed=index,
        )
        for index in range(3)
    ]
    client = _SchedulerClient(
        OutputBatch(output=[object() for _ in range(raw_output_count)])
    )

    def fail_save_outputs(*_args, **_kwargs):
        raise AssertionError("save_outputs must not run for a count mismatch")

    monkeypatch.setattr(openai_utils, "save_outputs", fail_save_outputs)

    with pytest.raises(
        RuntimeError,
        match=rf"returned {raw_output_count} raw outputs for 3 expanded requests",
    ):
        asyncio.run(openai_utils.process_generation_batch(client, requests))

    assert list(tmp_path.rglob("*.mp4")) == []


def test_raw_output_fallback_failure_removes_only_new_outputs(
    tmp_path,
    monkeypatch,
):
    requests = [
        _req(
            tmp_path / f"output-{index}" / "video.mp4",
            request_id=f"request-{index}",
            seed=index,
        )
        for index in range(3)
    ]
    output_paths = [Path(request.output_file_path(1, 0)) for request in requests]
    output_paths[0].parent.mkdir(parents=True)
    output_paths[0].write_bytes(b"preexisting-output")
    client = _SchedulerClient(OutputBatch(output=["a", "b", "c"]))

    def fail_on_last_output(
        _sample,
        _data_type,
        _fps,
        _save_output,
        save_file_path,
        **_kwargs,
    ):
        path = Path(save_file_path)
        if path != output_paths[0]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"new-output")
        if path == output_paths[-1]:
            raise RuntimeError("third output failed")
        return []

    monkeypatch.setattr(output_utils, "post_process_sample", fail_on_last_output)

    with pytest.raises(RuntimeError, match="third output failed"):
        asyncio.run(openai_utils.process_generation_batch(client, requests))

    assert output_paths[0].read_bytes() == b"preexisting-output"
    assert not output_paths[1].exists()
    assert not output_paths[2].exists()
