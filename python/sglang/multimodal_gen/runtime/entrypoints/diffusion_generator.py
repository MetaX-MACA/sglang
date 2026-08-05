# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating images/videos using
diffusion models.
"""

import dataclasses
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from contextlib import ExitStack
from typing import Any, List, Union

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GenerationResult,
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.launch_server import launch_server
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.server_warmup import (
    run_sync_client_warmup,
    should_run_explicit_client_warmup,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    GREEN,
    RESET,
    init_logger,
    log_batch_completion,
    log_generation_timer,
)
from sglang.multimodal_gen.runtime.utils.trace_wrapper import (
    init_diffusion_tracing,
    trace_req,
)

logger = init_logger(__name__)


def get_video_model_adapter(pipeline_config):
    """Resolve a video adapter lazily so importing DiffGenerator stays light."""

    from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
        get_video_model_adapter as _resolve_video_model_adapter,
    )

    return _resolve_video_model_adapter(pipeline_config)


# TODO: move to somewhere appropriate
try:
    # Set the start method to 'spawn' to avoid CUDA errors in forked processes.
    # This must be done at the top level of the module, before any CUDA context
    # or other processes are initialized.
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method can only be set once per program execution.
    pass


class DiffGenerator:
    """
    A unified class for generating images/videos using diffusion models.

    This class provides a simple interface for image/video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    """

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        """
        Initialize the generator.

        Args:
            server_args: The inference arguments
        """
        self.server_args = server_args
        self.port_args = PortArgs.from_server_args(server_args)

        # The executor is now a client to the Scheduler service
        self.local_scheduler_process: list[mp.Process] | None = None
        self.owns_scheduler_client: bool = False

    @classmethod
    def from_pretrained(
        cls,
        local_mode: bool = True,
        **kwargs,
    ) -> "DiffGenerator":
        """
        Create a DiffGenerator from a pretrained model.

        Priority level: Default pipeline config < User's pipeline config < User's kwargs
        """
        server_args = kwargs.pop("server_args", None)
        if server_args is not None:
            if kwargs:
                raise ValueError(
                    "server_args cannot be combined with additional override "
                    f"kwargs; unexpected kwargs: {sorted(kwargs)}"
                )
            if isinstance(server_args, dict):
                server_args = ServerArgs.from_kwargs(**server_args)
            elif not isinstance(server_args, ServerArgs):
                raise TypeError(
                    "server_args must be a ServerArgs instance or a dict, got "
                    f"{type(server_args).__name__}"
                )
        else:
            server_args = ServerArgs.from_kwargs(**kwargs)

        return cls.from_server_args(server_args, local_mode=local_mode)

    @classmethod
    def from_server_args(
        cls, server_args: ServerArgs, local_mode: bool = True
    ) -> "DiffGenerator":
        """
        Create a DiffGenerator with the specified arguments.

        Args:
            server_args: The inference arguments

        Returns:
            The created DiffGenerator
        """
        instance = cls(
            server_args=server_args,
        )
        init_diffusion_tracing(server_args, "DiffGenerator")

        logger.info(f"Local mode: {local_mode}")
        if local_mode:
            instance.local_scheduler_process = instance._start_local_server_if_needed()
            instance.owns_scheduler_client = True
            instance._run_client_warmup_if_needed()
        else:
            # In remote mode, we just need to connect and check.
            sync_scheduler_client.initialize(server_args)
            instance._check_remote_scheduler()
            instance.owns_scheduler_client = True
        return instance

    def _start_local_server_if_needed(
        self,
    ) -> list[mp.Process]:
        """Check if a local server is running; if not, start it and return the process handles."""
        # First, we need a client to test the server. Initialize it temporarily.
        sync_scheduler_client.initialize(self.server_args)

        processes = launch_server(self.server_args, launch_http_server=False)

        return processes

    def _run_client_warmup_if_needed(self) -> None:
        if not should_run_explicit_client_warmup(self.server_args):
            return

        run_sync_client_warmup(self.server_args, sync_scheduler_client.forward)

    def _check_remote_scheduler(self):
        """Check if the remote scheduler is accessible."""
        if not sync_scheduler_client.ping():
            raise ConnectionError(
                f"Could not connect to remote scheduler at "
                f"{self.server_args.scheduler_endpoint} with `local mode` as False. "
                "Please ensure the server is running."
            )
        logger.info(
            f"Successfully connected to remote scheduler at "
            f"{self.server_args.scheduler_endpoint}."
        )

    @staticmethod
    def _resolve_image_paths_per_prompt(
        prompts: list[str], image_paths: str | list[str] | None
    ) -> list[str | list[str] | None]:
        if len(prompts) <= 1:
            return [image_paths]

        if not isinstance(image_paths, list) or len(image_paths) <= 1:
            return [image_paths for _ in prompts]

        if len(image_paths) != len(prompts):
            raise ValueError(
                "When using multiple prompts with multiple input images, "
                "provide either one shared image or exactly one image per prompt."
            )

        return [[image_path] for image_path in image_paths]

    def generate(
        self,
        sampling_params_kwargs: dict | None = None,
        external_trace_header: dict[str, str] | None = None,
    ) -> GenerationResult | list[GenerationResult] | None:
        """Generate image(s)/video(s) based on the given prompt(s).

        Each parent request owns one complete admission/dispatch/validation/
        cleanup lifecycle.  Keeping that sequence together is important for
        MiniMax H3: localized condition media and resolved shape facts must stay
        alive until every expanded output has been delivered.
        """
        if sampling_params_kwargs is None:
            sampling_params_kwargs = {}
        adapter = get_video_model_adapter(self.server_args.pipeline_config)
        prompts = self._resolve_prompts(
            sampling_params_kwargs.get("prompt"),
            sampling_params_kwargs.get("prompt_path"),
        )
        user_output_file_name = sampling_params_kwargs.get("output_file_name")

        if len(prompts) > 1 and user_output_file_name is not None:
            raise ValueError(
                "Cannot use multiple prompts with a fixed output_file_name. "
                "Either remove --output-file-name or use a single prompt."
            )

        sampling_params_orig = SamplingParams.from_user_sampling_params_args(
            self.server_args.model_path,
            server_args=self.server_args,
            **sampling_params_kwargs,
        )
        # Model-specific admission happens before request-owned resources exist.
        adapter.validate_sampling_params(sampling_params_orig)

        image_paths_per_prompt = self._resolve_image_paths_per_prompt(
            prompts, sampling_params_orig.image_path
        )
        results: list[GenerationResult] = []
        failures: list[str] = []
        total_start_time = time.perf_counter()
        global_output_index = 0

        for prompt_index, prompt in enumerate(prompts):
            sampling_params = dataclasses.replace(
                sampling_params_orig,
                prompt=prompt,
                output_file_name=user_output_file_name,
                image_path=image_paths_per_prompt[prompt_index],
            )
            requested_output_count = int(sampling_params.num_outputs_per_prompt)
            # ``dataclasses.replace`` drops non-field attributes; restore the
            # explicit-field marker consumed by input validation stages.
            sampling_params._explicit_fields = getattr(
                sampling_params_orig, "_explicit_fields", set()
            ) | {"prompt", "output_file_name", "image_path"}
            sampling_params._set_output_file_name()
            if len(prompts) > 1:
                # ``SamplingParams`` derives a filename from the prompt and
                # current second. Identical prompts in one batch would
                # otherwise overwrite one another; parent request IDs also
                # need a stable prompt component before output expansion.
                sampling_params.request_id = (
                    f"{sampling_params.request_id}:{prompt_index}"
                    if sampling_params.request_id is not None
                    else f"prompt:{prompt_index}"
                )
                if user_output_file_name is None:
                    base, extension = os.path.splitext(sampling_params.output_file_name)
                    sampling_params.output_file_name = (
                        f"{base}_{prompt_index}{extension}"
                    )

            req = prepare_request(
                server_args=self.server_args,
                sampling_params=sampling_params,
                external_trace_header=external_trace_header,
            )
            requests: list[Req] = []
            try:
                # Admission is deliberately once per parent, before output
                # expansion. Children share the resolved plan but own derived
                # per-output resources.
                adapter.prepare_for_queue_sync(req)
                dispatch_batch = adapter.expand_for_offline_dispatch(
                    req,
                    num_prompts=len(prompts),
                    prompt_index=prompt_index,
                )
                requests = (
                    list(dispatch_batch)
                    if isinstance(dispatch_batch, list)
                    else [dispatch_batch]
                )
                preexisting_paths = {
                    os.path.abspath(path)
                    for output_req in requests
                    for path in (output_req.output_file_path(1, 0),)
                    if path is not None and os.path.lexists(path)
                }
            except Exception as exc:
                self._cleanup_offline_requests(adapter, req, requests)
                logger.error(
                    "Generation admission failed: %s",
                    req.request_id,
                    exc_info=True,
                )
                failures.append(
                    f"prompt {prompt_index} ({req.request_id}): admission "
                    f"failed: {exc!r}"
                )
                global_output_index += requested_output_count
                continue

            group_results = self._process_request_group(
                adapter=adapter,
                requests=requests,
                parent_req=req,
                preexisting_paths=preexisting_paths,
                global_output_index=global_output_index,
                failures=failures,
            )
            results.extend(group_results)
            # Use the requested cardinality even if admission/dispatch failed,
            # so later prompt_index values remain globally stable.
            global_output_index += requested_output_count

        total_gen_time = time.perf_counter() - total_start_time
        if self.server_args.batching_max_size > 1:
            log_batch_completion(logger, len(results), total_gen_time)
        self._log_summary(results)

        if not results:
            if failures:
                # Per-prompt isolation supports partial success, but a run
                # where every prompt failed must not exit successfully with
                # zero outputs.
                raise RuntimeError(
                    f"all {len(prompts)} prompt(s) produced no successful "
                    f"outputs; first error: {failures[0]}; all errors: "
                    f"{'; '.join(failures)}"
                )
            return None
        return results[0] if len(results) == 1 else results

    @staticmethod
    def _cleanup_offline_requests(
        adapter: Any,
        parent_req: Req | None,
        requests: list[Req],
    ) -> None:
        """Clean child-owned derived media, then the parent source closure."""

        seen: set[int] = set()
        for child in requests:
            if parent_req is not None and child is parent_req:
                continue
            child_id = id(child)
            if child_id in seen:
                continue
            seen.add(child_id)
            try:
                adapter.cleanup_request_sync(child)
            except Exception:
                logger.warning(
                    "Failed to clean offline child request resources",
                    exc_info=True,
                )
        if parent_req is not None and id(parent_req) not in seen:
            try:
                adapter.cleanup_request_sync(parent_req)
            except Exception:
                logger.warning(
                    "Failed to clean offline request resources",
                    exc_info=True,
                )

    def _process_request_group(
        self,
        *,
        adapter: Any,
        requests: list[Req],
        parent_req: Req,
        preexisting_paths: set[str],
        global_output_index: int,
        failures: list[str],
    ) -> list[GenerationResult]:
        """Dispatch one already-admitted parent and publish all-or-nothing outputs."""

        generated_output_paths: set[str] = set()
        preserve_generated_outputs = False
        preexisting_backups: dict[str, str] = {}

        def _elapsed_generation_time(timer: Any) -> float:
            start_time = getattr(timer, "start_time", None)
            if isinstance(start_time, (int, float)) and start_time > 0:
                return max(0.0, time.perf_counter() - start_time)
            return float(getattr(timer, "duration", 0.0) or 0.0)

        def _record_generated_paths(paths: Any) -> list[str]:
            normalized: list[str] = []
            for raw_path in paths or ():
                if raw_path is None:
                    continue
                path = os.path.abspath(os.fspath(raw_path))
                generated_output_paths.add(path)
                normalized.append(path)
            return normalized

        def _backup_existing_path(raw_path: Any) -> None:
            if raw_path is None:
                return
            path = os.path.abspath(os.fspath(raw_path))
            if path in preexisting_backups or not os.path.lexists(path):
                return
            fd, backup_path = tempfile.mkstemp(
                prefix="sglang_diffusion_output_backup_", suffix=".bak"
            )
            os.close(fd)
            preexisting_backups[path] = backup_path
            try:
                shutil.copy2(path, backup_path)
            except Exception:
                preexisting_backups.pop(path, None)
                try:
                    os.remove(backup_path)
                except OSError:
                    pass
                raise
            preexisting_paths.add(path)

        try:
            # Snapshot destinations before a potentially non-atomic saver runs.
            if self.server_args.pipeline_config.requires_audio_output or any(
                bool(getattr(request, "save_output", False)) for request in requests
            ):
                for path in list(preexisting_paths):
                    _backup_existing_path(path)
            _record_generated_paths(
                request.output_file_path(1, 0) for request in requests
            )

            timer_prompt = [request.prompt for request in requests]
            logger.info("Processing %d grouped request(s)", len(requests))
            with ExitStack() as stack:
                for request in requests:
                    stack.enter_context(trace_req(request.trace_ctx))
                timer = stack.enter_context(log_generation_timer(logger, timer_prompt))
                output_batch = self._send_to_scheduler_and_wait_for_response(requests)
                if output_batch.error:
                    raise RuntimeError(str(output_batch.error))
                if (
                    output_batch.output is None
                    and output_batch.output_file_paths is None
                ):
                    logger.error("Received empty output from scheduler")
                    failures.append(
                        f"request group {parent_req.request_id}: scheduler "
                        "returned empty output"
                    )
                    stack.close()
                    return []

                if self.server_args.pipeline_config.requires_audio_output:
                    output_file_paths = [
                        str(path) for path in (output_batch.output_file_paths or [])
                    ]
                    samples_out: list[Any] = []
                    audios_out: list[Any] = []
                    frames_out: list[Any] = []
                    if not output_file_paths:
                        if output_batch.output is None:
                            raise RuntimeError(
                                "strict AV delivery produced no file output"
                            )
                        self._validate_output_count(
                            len(output_batch.output), len(requests)
                        )
                        output_file_paths = save_outputs(
                            output_batch.output,
                            requests[0].data_type,
                            requests[0].fps,
                            True,
                            lambda index: requests[index].output_file_path(1, 0),
                            audio=output_batch.audio,
                            audio_sample_rate=output_batch.audio_sample_rate,
                            samples_out=samples_out,
                            audios_out=audios_out,
                            frames_out=frames_out,
                            output_compression=requests[0].output_compression,
                            enable_frame_interpolation=False,
                            enable_upscaling=False,
                            strict_audio_mux=True,
                            output_audio_sample_rate=self.server_args.pipeline_config.output_audio_sample_rate,
                            output_audio_channels=self.server_args.pipeline_config.output_audio_channels,
                            output_av_drift_tolerance_s=self.server_args.pipeline_config.output_av_drift_tolerance_s,
                        )
                    _record_generated_paths(output_file_paths)
                    self._validate_output_count(len(output_file_paths), len(requests))
                    adapter.validate_final_outputs_sync(
                        output_file_paths,
                        parent_req,
                    )
                    generation_time = _elapsed_generation_time(timer)
                    group_results = [
                        GenerationResult(
                            **self._result_common(
                                request,
                                output_batch,
                                generation_time,
                                index,
                            ),
                            samples=(
                                samples_out[index] if index < len(samples_out) else None
                            ),
                            frames=(
                                frames_out[index] if index < len(frames_out) else None
                            ),
                            audio=(
                                audios_out[index] if index < len(audios_out) else None
                            ),
                            prompt_index=global_output_index + index,
                            output_file_path=path,
                        )
                        for index, (request, path) in enumerate(
                            zip(requests, output_file_paths)
                        )
                    ]
                    # No child is published until every output validates.
                    stack.close()
                    preserve_generated_outputs = True
                    return group_results

                if requests[0].save_output and requests[0].return_file_paths_only:
                    output_file_paths = output_batch.output_file_paths or []
                    _record_generated_paths(output_file_paths)
                    self._validate_output_count(len(output_file_paths), len(requests))
                    generation_time = _elapsed_generation_time(timer)
                    group_results = [
                        GenerationResult(
                            **self._result_common(
                                requests[index],
                                output_batch,
                                generation_time,
                                index,
                            ),
                            prompt_index=global_output_index + index,
                            output_file_path=path,
                        )
                        for index, path in enumerate(output_file_paths)
                    ]
                    stack.close()
                    preserve_generated_outputs = True
                    return group_results

                if requests[0].data_type == DataType.MESH:
                    output_file_paths = output_batch.output_file_paths or []
                    _record_generated_paths(output_file_paths)
                    self._validate_output_count(len(output_file_paths), len(requests))
                    generation_time = _elapsed_generation_time(timer)
                    group_results = [
                        GenerationResult(
                            **self._result_common(
                                requests[index],
                                output_batch,
                                generation_time,
                                index,
                            ),
                            prompt_index=global_output_index + index,
                            output_file_path=path,
                        )
                        for index, path in enumerate(output_file_paths)
                    ]
                    stack.close()
                    preserve_generated_outputs = True
                    return group_results

                self._validate_output_count(len(output_batch.output), len(requests))
                samples_out: list[Any] = []
                audios_out: list[Any] = []
                frames_out: list[Any] = []
                save_outputs(
                    output_batch.output,
                    requests[0].data_type,
                    requests[0].fps,
                    requests[0].save_output,
                    lambda index: requests[index].output_file_path(1, 0),
                    audio=output_batch.audio,
                    audio_sample_rate=output_batch.audio_sample_rate,
                    samples_out=samples_out,
                    audios_out=audios_out,
                    frames_out=frames_out,
                    output_compression=requests[0].output_compression,
                    enable_frame_interpolation=requests[0].enable_frame_interpolation,
                    frame_interpolation_exp=requests[0].frame_interpolation_exp,
                    frame_interpolation_scale=requests[0].frame_interpolation_scale,
                    frame_interpolation_model_path=requests[
                        0
                    ].frame_interpolation_model_path,
                    enable_upscaling=requests[0].enable_upscaling,
                    upscaling_model_path=requests[0].upscaling_model_path,
                    upscaling_scale=requests[0].upscaling_scale,
                    strict_audio_mux=self.server_args.pipeline_config.requires_audio_output,
                    output_audio_sample_rate=self.server_args.pipeline_config.output_audio_sample_rate,
                    output_audio_channels=self.server_args.pipeline_config.output_audio_channels,
                    output_av_drift_tolerance_s=self.server_args.pipeline_config.output_av_drift_tolerance_s,
                )
                generation_time = _elapsed_generation_time(timer)
                group_results = [
                    GenerationResult(
                        **self._result_common(
                            requests[index],
                            output_batch,
                            generation_time,
                            index,
                        ),
                        samples=samples_out[index],
                        frames=frames_out[index],
                        audio=audios_out[index],
                        prompt_index=global_output_index + index,
                        output_file_path=requests[index].output_file_path(1, 0),
                    )
                    for index in range(len(samples_out))
                ]
                stack.close()
                preserve_generated_outputs = True
                return group_results
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            failures.append(f"request group {parent_req.request_id}: {exc!r}")
            return []
        finally:
            self._cleanup_offline_requests(adapter, parent_req, requests)
            if not preserve_generated_outputs:
                for path in generated_output_paths:
                    if path in preexisting_paths:
                        continue
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.warning(
                            "Failed to remove incomplete offline output %s",
                            path,
                            exc_info=True,
                        )
                for original_path, backup_path in preexisting_backups.items():
                    try:
                        shutil.copy2(backup_path, original_path)
                    except OSError:
                        logger.warning(
                            "Failed to restore pre-existing offline output %s",
                            original_path,
                            exc_info=True,
                        )
            for backup_path in preexisting_backups.values():
                try:
                    os.remove(backup_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Failed to remove offline output backup %s",
                        backup_path,
                        exc_info=True,
                    )

    def _resolve_prompts(
        self,
        prompt: str | list[str] | None,
        prompt_path: str | None = None,
    ) -> list[str]:
        """Collect prompts from the argument or from a prompt file."""
        path = prompt_path or self.server_args.prompt_file_path
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Prompt text file not found: {path}")
            with open(path, encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if not prompts:
                raise ValueError(f"No prompts found in file: {path}")
            logger.info("Found %d prompts in %s", len(prompts), path)
            return prompts

        if prompt is None:
            return [" "]
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    def _log_summary(self, results: list[GenerationResult]) -> None:
        if not results:
            return
        if self.server_args.warmup:
            total_duration_ms = results[0].metrics.get("total_duration_ms", 0)
            logger.info(
                f"Warmed-up request processed in {GREEN}%.2f{RESET} seconds (with warmup excluded)",
                total_duration_ms / 1000.0,
            )

        peak_memories = [r.peak_memory_mb for r in results if r.peak_memory_mb]
        if peak_memories:
            logger.info(
                f"Memory usage - Max peak: {max(peak_memories):.2f} MB, "
                f"Avg peak: {sum(peak_memories) / len(peak_memories):.2f} MB"
            )

    @staticmethod
    def _result_common(
        req: Req,
        output_batch: OutputBatch,
        generation_time: float,
        output_index: int | None = None,
    ) -> dict[str, Any]:
        metrics = output_batch.metrics
        if (
            output_index is not None
            and output_batch.metrics_list is not None
            and output_index < len(output_batch.metrics_list)
        ):
            metrics = output_batch.metrics_list[output_index]
        return dict(
            prompt=req.prompt,
            size=(req.height, req.width, req.num_frames),
            generation_time=generation_time,
            peak_memory_mb=output_batch.peak_memory_mb,
            metrics=metrics.to_dict() if metrics else {},
            trajectory_latents=output_batch.trajectory_latents,
            trajectory_timesteps=output_batch.trajectory_timesteps,
            rollout_trajectory_data=output_batch.rollout_trajectory_data,
            trajectory_decoded=output_batch.trajectory_decoded,
        )

    @staticmethod
    def _validate_output_count(output_count: int, request_count: int) -> None:
        if output_count != request_count:
            raise RuntimeError(
                f"Expected {request_count} outputs, got {output_count} from scheduler"
            )

    def _send_to_scheduler_and_wait_for_response(self, batch: list[Req]) -> OutputBatch:
        """
        Sends a request to the scheduler and waits for a response.
        """
        return sync_scheduler_client.forward(batch)

    # LoRA
    def _send_lora_request(self, req: Any, success_msg: str, failure_msg: str):
        response = sync_scheduler_client.forward(req)
        if response.error is None:
            logger.info(success_msg)
            return response
        else:
            error_msg = response.error
            raise RuntimeError(f"{failure_msg}: {error_msg}")

    def set_lora(
        self,
        lora_nickname: Union[str, List[str]],
        lora_path: Union[str, None, List[Union[str, None]]] = None,
        target: Union[str, List[str]] = "all",
        strength: Union[float, List[float]] = 1.0,
        merge_mode: str | None = None,
    ) -> None:
        """
        Set LoRA adapter(s) for the specified transformer(s).
        Supports both single LoRA (backward compatible) and multiple LoRA adapters.

        Args:
            lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
            lora_path: Path(s) to the LoRA adapter(s). Can be a string, None, or a list of strings/None.
            target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
                Valid values:
                - "all": Apply to all transformers (default)
                - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
                - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
                - "critic": Apply only to the critic model
            strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
            merge_mode: Optional LoRA merge mode: "auto", "merge", or "dynamic".
        """
        req = SetLoraReq(
            lora_nickname=lora_nickname,
            lora_path=lora_path,
            target=target,
            strength=strength,
            merge_mode=merge_mode,
        )
        nickname_str, target_str, strength_str = format_lora_message(
            lora_nickname, target, strength
        )

        self._send_lora_request(
            req,
            f"Successfully set LoRA adapter(s): {nickname_str} (target: {target_str}, strength: {strength_str})",
            "Failed to set LoRA adapter",
        )

    def unmerge_lora_weights(self, target: str = "all") -> None:
        """
        Unmerge LoRA weights from the base model.

        Args:
            target: Which transformer(s) to unmerge.
        """
        req = UnmergeLoraWeightsReq(target=target)
        self._send_lora_request(
            req,
            f"Successfully unmerged LoRA weights (target: {target})",
            "Failed to unmerge LoRA weights",
        )

    def merge_lora_weights(self, target: str = "all", strength: float = 1.0) -> None:
        """
        Merge LoRA weights into the base model.

        Args:
            target: Which transformer(s) to merge.
            strength: LoRA strength for merge, default 1.0.
        """
        req = MergeLoraWeightsReq(target=target, strength=strength)
        self._send_lora_request(
            req,
            f"Successfully merged LoRA weights (target: {target}, strength: {strength})",
            "Failed to merge LoRA weights",
        )

    def list_loras(self) -> dict:
        """List loaded LoRA adapters and current application status per module."""
        output = self._send_lora_request(
            req=ListLorasReq(),
            success_msg="Successfully listed LoRA adapters",
            failure_msg="Failed to list LoRA adapters",
        )
        # _send_lora_request already raises on error, so output.error is always None here
        return output.output or {}

    def _ensure_lora_state(
        self,
        lora_path: str | None,
        lora_nickname: str | None = None,
        merge_lora: bool = True,
    ) -> None:
        """
        Ensure the LoRA state matches the desired configuration.

        Note: This method does not cache client-side state. The server handles
        idempotent operations, so redundant calls are safe but may have minor overhead.
        """
        if lora_path is None:
            # Unmerge all LoRA weights when no lora_path is provided
            self.unmerge_lora_weights()
            return

        lora_nickname = lora_nickname or self.server_args.lora_nickname

        # Set the LoRA adapter (server handles idempotent logic)
        self.set_lora(lora_nickname, lora_path)

        # Merge or unmerge based on the merge_lora flag
        if merge_lora:
            self.merge_lora_weights()
        else:
            self.unmerge_lora_weights()

    def generate_with_lora(
        self,
        prompt: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        lora_path: str | None = None,
        lora_nickname: str | None = None,
        merge_lora: bool = True,
        **kwargs,
    ):
        self._ensure_lora_state(
            lora_path=lora_path, lora_nickname=lora_nickname, merge_lora=merge_lora
        )
        return self.generate(
            sampling_params_kwargs=dict(
                prompt=prompt,
                sampling_params=sampling_params,
                **kwargs,
            )
        )

    def shutdown(self):
        """
        Shutdown the generator.
        If in local mode, it also shuts down the scheduler server.
        """
        # sends the shutdown command to the server
        if self.local_scheduler_process and self.owns_scheduler_client:
            try:
                sync_scheduler_client.forward(ShutdownReq(), timeout_ms=5000)
            except Exception:
                pass

        if self.local_scheduler_process:
            for process in self.local_scheduler_process:
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(
                        f"Local worker {process.name} did not terminate gracefully, forcing."
                    )
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
            self.local_scheduler_process = None

        if self.owns_scheduler_client:
            sync_scheduler_client.close()
            self.owns_scheduler_client = False

    def _force_shutdown_local_processes(self) -> None:
        local_scheduler_process = getattr(self, "local_scheduler_process", None)
        log = globals().get("logger")
        if local_scheduler_process:
            for process in local_scheduler_process:
                if process.is_alive():
                    if log is not None:
                        log.warning(
                            f"Local worker {process.name} did not terminate gracefully, forcing."
                        )
                    process.terminate()
            for process in local_scheduler_process:
                process.join(timeout=1)
                if process.is_alive():
                    if log is not None:
                        log.warning(
                            f"Local worker {process.name} did not terminate after terminate(), killing."
                        )
                    process.kill()
                    process.join(timeout=1)
            self.local_scheduler_process = None

        if getattr(self, "owns_scheduler_client", False):
            try:
                client = globals().get("sync_scheduler_client")
                if client is not None:
                    client.close()
            finally:
                self.owns_scheduler_client = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        owns_scheduler_client = bool(getattr(self, "owns_scheduler_client", False))
        local_scheduler_process = getattr(self, "local_scheduler_process", None)
        log = globals().get("logger")
        if owns_scheduler_client:
            if log is not None:
                log.warning(
                    "Generator was garbage collected without being shut down. "
                    "Forcing local server and client cleanup."
                )
            self._force_shutdown_local_processes()
        elif local_scheduler_process:
            if log is not None:
                log.warning(
                    "Generator was garbage collected without being shut down. "
                    "Forcing local server cleanup."
                )
            self._force_shutdown_local_processes()
