# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    flatten_extra_params,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_adapter import (
    COMMON_MULTIPART_FORM_FIELDS,
    BaseVideoModelAdapter,
    get_video_model_adapter,
    known_video_model_fields,
    validate_adapter_field_claims,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


def _extra_value(request: VideoGenerationsRequest, name: str) -> Any:
    return (request.model_extra or {}).get(name)


def _parse_form_extra_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return value


def _multipart_extra_form_keys() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *VideoGenerationsRequest.model_fields,
                *COMMON_MULTIPART_FORM_FIELDS,
                *sorted(known_video_model_fields()),
            )
        )
    )


def _filter_multipart_declared_fields(
    extra_from_form: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep only protocol and registered model fields from multipart extras.

    Starlette exposes every submitted form key.  The video request model uses
    ``extra='allow'`` for JSON compatibility, but multipart has to build the
    model explicitly; retaining only declared fields keeps unrelated form
    controls out of the request while preserving the same lowering semantics.
    """

    declared = set(VideoGenerationsRequest.model_fields)
    declared.update(_multipart_extra_form_keys())
    return {key: value for key, value in extra_from_form.items() if key in declared}


def _merge_multipart_extra_form_fields(
    raw_form: Any,
    extra_from_form: Dict[str, Any],
) -> None:
    for key in _multipart_extra_form_keys():
        if key in raw_form and key not in extra_from_form:
            extra_from_form[key] = _parse_form_extra_value(raw_form[key])


def _multipart_video_extras(
    raw_form: Any,
    *,
    extra_body: Any,
    extra_params: Any,
) -> Dict[str, Any]:
    """Build multipart extras once for the early task gate and request model."""

    extra_from_form: Dict[str, Any] = {}
    if extra_body:
        try:
            extra_from_form = flatten_extra_params(json.loads(extra_body))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_body is not valid JSON"
            ) from exc
    if extra_params:
        try:
            extra_from_form.update(
                flatten_extra_params({"extra_params": json.loads(extra_params)})
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_params is not valid JSON"
            ) from exc
    _merge_multipart_extra_form_fields(raw_form, extra_from_form)
    flatten_extra_params(extra_from_form)
    return _filter_multipart_declared_fields(extra_from_form)


def _multipart_declared_request_fields(
    raw_form: Any,
    extra_from_form: Dict[str, Any],
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """Preserve presence for optional declared fields in multipart requests."""

    declared: Dict[str, Any] = {}
    for field_name, field_value in values.items():
        if field_name not in raw_form and field_name not in extra_from_form:
            continue
        raw_value = (
            field_value if field_value is not None else extra_from_form.get(field_name)
        )
        declared[field_name] = _parse_form_extra_value(raw_value)
    return declared


def _json_video_payload(body: Any) -> Dict[str, Any]:
    """Flatten OpenAI extra wrappers without losing task presence semantics."""

    payload: Dict[str, Any] = dict(body or {})
    extra = payload.pop("extra_body", None)
    if isinstance(extra, str):
        extra = json.loads(extra)
    if isinstance(extra, dict):
        payload.update(flatten_extra_params(extra))
    extra_json = payload.pop("extra_json", None)
    if isinstance(extra_json, str):
        extra_json = json.loads(extra_json)
    if isinstance(extra_json, dict):
        payload.update(flatten_extra_params(extra_json))
    flatten_extra_params(payload)
    return payload


def _reject_unsupported_cosmos3_modes(
    req: VideoGenerationsRequest, model_path: str | None
) -> None:
    BaseVideoModelAdapter().validate_transport_options(req, model_path=model_path)


def _video_job_from_batch(
    request_id: str,
    req: VideoGenerationsRequest,
    batch: Req,
    adapter: BaseVideoModelAdapter,
) -> Dict[str, Any]:
    size_str = f"{batch.width}x{batch.height}"
    seconds = int(round((batch.num_frames or 0) / float(batch.fps or 24)))
    job = {
        "id": request_id,
        "object": "video",
        "model": req.model or "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": os.path.abspath(batch.output_file_path()),
    }
    job.update(adapter.project_queued_job_fields(batch))
    return job


async def _save_first_input_image(
    image_sources,
    request_id: str,
    uploads_dir: str,
    *,
    prefer_remote_source: bool = False,
) -> str | None:
    """Save the first input image from a list of sources and return its path."""
    image_list = merge_image_input_list(image_sources)
    if not image_list:
        return None
    image = image_list[0]

    os.makedirs(uploads_dir, exist_ok=True)
    filename = image.filename if hasattr(image, "filename") else "url_image"
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(
        image, target_path, prefer_remote_source=prefer_remote_source
    )


def _cleanup_generated_outputs(paths: list[str]) -> None:
    """Remove outputs from a failed generation when they were materialized."""
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except (OSError, TypeError) as cleanup_error:
            logger.warning(
                "failed to remove generated video output %s: %s",
                path,
                cleanup_error,
            )


async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    adapter: BaseVideoModelAdapter | None = None,
    temp_dirs: list[str] | None = None,
    temp_resources: list[Any] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    generated_output_paths: list[str] = []
    try:
        adapter = adapter or BaseVideoModelAdapter()
        dispatch_batch = adapter.expand_for_dispatch(batch)
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, dispatch_batch
        )
        generated_output_paths = list(save_file_path_list)
        if not save_file_path_list:
            raise RuntimeError("video generation produced no output files")
        final_media_fields = await adapter.validate_final_outputs(
            save_file_path_list,
            batch,
        )

        cloud_urls: list[str] = []
        if cloud_storage.is_enabled():
            for path in save_file_path_list:
                cloud_url = await cloud_storage.upload_and_cleanup(path)
                if not cloud_url:
                    # Preserve the legacy local fallback when the first upload
                    # fails and every output still exists. Once any earlier
                    # output has uploaded and been cleaned, falling back would
                    # publish an incomplete local result and must fail closed.
                    if not cloud_urls and output_persistent:
                        break
                    raise RuntimeError(f"failed to upload generated video {path}")
                cloud_urls.append(cloud_url)
        if not cloud_urls and not output_persistent:
            raise RuntimeError(
                "generated video has no durable destination; configure an "
                "output path or enable cloud storage"
            )

        persistent_paths = (
            [os.path.abspath(path) for path in save_file_path_list]
            if not cloud_urls and output_persistent
            else None
        )
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_urls[0] if cloud_urls else None,
            "urls": cloud_urls or None,
            "file_path": persistent_paths[0] if persistent_paths else None,
            "file_paths": persistent_paths,
            "num_outputs": len(save_file_path_list),
            **final_media_fields,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except asyncio.CancelledError:
        if adapter is not None and adapter.strict_file_delivery:
            _cleanup_generated_outputs(generated_output_paths)
        await asyncio.shield(
            VIDEO_STORE.update_fields(
                job_id,
                {
                    "status": "failed",
                    "error": {"message": "Video generation cancelled"},
                    "url": None,
                    "urls": None,
                    "file_path": None,
                    "file_paths": None,
                    "num_outputs": None,
                },
            )
        )
        raise
    except Exception as e:
        if adapter is not None and adapter.strict_file_delivery:
            _cleanup_generated_outputs(generated_output_paths)
        logger.error("video job %s failed (%s)", job_id, type(e).__name__)
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": str(e)},
                "url": None,
                "urls": None,
                "file_path": None,
                "file_paths": None,
                "num_outputs": None,
            },
        )
    finally:
        try:
            if adapter is not None:
                await asyncio.shield(adapter.cleanup_request(batch))
        finally:
            for resource in temp_resources or []:
                resource.cleanup()
            for td in temp_dirs or []:
                shutil.rmtree(td, ignore_errors=True)


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    num_outputs_per_prompt: Optional[int] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    max_sequence_length: Optional[int] = Form(None),
    flow_shift: Optional[float] = Form(None),
    enable_teacache: Optional[bool] = Form(None),
    enable_frame_interpolation: Optional[bool] = Form(None),
    frame_interpolation_exp: Optional[int] = Form(None),
    frame_interpolation_scale: Optional[float] = Form(None),
    frame_interpolation_model_path: Optional[str] = Form(None),
    enable_upscaling: Optional[bool] = Form(None),
    upscaling_model_path: Optional[str] = Form(None),
    upscaling_scale: Optional[int] = Form(None),
    output_quality: Optional[str] = Form(None),
    output_compression: Optional[int] = Form(None),
    output_path: Optional[str] = Form(None),
    extra_params: Optional[str] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    is_multipart = "multipart/form-data" in content_type
    request_id = generate_request_id()

    server_args = get_global_server_args()
    adapter = get_video_model_adapter(server_args.pipeline_config)
    task_type = server_args.pipeline_config.task_type

    # MiniMax H3 task identity controls the canonical pipeline. Validate it before
    # creating request-owned directories, saving uploads, or scheduling work.
    raw_form: Any = None
    extra_from_form: Dict[str, Any] | None = None
    payload: Dict[str, Any] | None = None
    try:
        if is_multipart:
            raw_form = await request.form()
            extra_from_form = _multipart_video_extras(
                raw_form,
                extra_body=extra_body,
                extra_params=extra_params,
            )
            adapter.validate_task_gate(
                extra_from_form.get("task"),
                provided="task" in extra_from_form,
            )
        else:
            try:
                body = await request.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="request body is not valid JSON"
                ) from exc
            payload = _json_video_payload(body)
            adapter.validate_task_gate(
                payload.get("task"),
                provided="task" in payload,
            )
    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Resolve input upload directory using the historical image/video API
    # behavior: honor input_save_path, otherwise keep a request temp directory.
    temp_dirs: list[str] = []
    temp_resources: list[Any] = []
    input_save_path = getattr(server_args, "input_save_path", None)
    if input_save_path is not None:
        uploads_dir = input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        input_resource = tempfile.TemporaryDirectory(prefix="sglang_input_")
        temp_resources.append(input_resource)
        uploads_dir = input_resource.name
        temp_dirs.append(uploads_dir)

    output_persistent = True

    def cleanup_request_resources() -> None:
        for resource in temp_resources:
            resource.cleanup()
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if is_multipart:
        if not prompt:
            cleanup_request_resources()
            raise HTTPException(status_code=400, detail="prompt is required")
        image_sources = merge_image_input_list(input_reference, reference_url)
        if task_type.requires_image_input() and not image_sources:
            cleanup_request_resources()
            raise HTTPException(
                status_code=400,
                detail="input_reference or reference_url is required for image-to-video generation",
            )
        assert extra_from_form is not None
        assert raw_form is not None

        def form_value(name: str, value: Any) -> Any:
            return value if value is not None else extra_from_form.get(name)

        request_field_names = set(VideoGenerationsRequest.model_fields)
        extra_request_fields = {
            key: value
            for key, value in extra_from_form.items()
            if key not in request_field_names
        }
        fps_val = form_value("fps", fps)
        num_frames_val = form_value("num_frames", num_frames)
        declared_model_fields = _multipart_declared_request_fields(
            raw_form,
            extra_from_form,
            {
                "width": None,
                "height": None,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": None,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": None,
                "perf_dump_path": None,
            },
        )

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=None,
            reference_url=reference_url,
            model=form_value("model", model),
            n=form_value("n", n),
            num_outputs_per_prompt=form_value(
                "num_outputs_per_prompt", num_outputs_per_prompt
            ),
            seconds=form_value("seconds", seconds) or 4,
            size=form_value("size", size),
            fps=fps_val,
            num_frames=num_frames_val,
            seed=form_value("seed", seed),
            generator_device=form_value("generator_device", generator_device),
            num_inference_steps=form_value("num_inference_steps", num_inference_steps),
            max_sequence_length=form_value("max_sequence_length", max_sequence_length),
            flow_shift=form_value("flow_shift", flow_shift),
            enable_teacache=form_value("enable_teacache", enable_teacache),
            enable_frame_interpolation=form_value(
                "enable_frame_interpolation", enable_frame_interpolation
            ),
            frame_interpolation_exp=form_value(
                "frame_interpolation_exp", frame_interpolation_exp
            ),
            frame_interpolation_scale=form_value(
                "frame_interpolation_scale", frame_interpolation_scale
            ),
            frame_interpolation_model_path=form_value(
                "frame_interpolation_model_path", frame_interpolation_model_path
            ),
            enable_upscaling=form_value("enable_upscaling", enable_upscaling),
            upscaling_model_path=form_value(
                "upscaling_model_path", upscaling_model_path
            ),
            upscaling_scale=form_value("upscaling_scale", upscaling_scale),
            output_compression=form_value("output_compression", output_compression),
            output_quality=form_value("output_quality", output_quality),
            output_path=form_value("output_path", output_path),
            diffusers_kwargs=form_value("diffusers_kwargs", None),
            **declared_model_fields,
            **extra_request_fields,
        )
        try:
            validate_adapter_field_claims(req, adapter)
            adapter.validate_transport_options(req, model_path=server_args.model_path)
            input_path = await _save_first_input_image(
                image_sources,
                request_id,
                uploads_dir,
                prefer_remote_source=input_save_path is None,
            )
        except Exception as e:
            cleanup_request_resources()
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            ) from e
        req.input_reference = input_path
        req.reference_url = None
    else:
        try:
            assert payload is not None
            has_image_input = payload.get("reference_url") or payload.get(
                "input_reference"
            )
            if task_type.requires_image_input() and not has_image_input:
                cleanup_request_resources()
                raise HTTPException(
                    status_code=400,
                    detail="input_reference or reference_url is required for image-to-video generation",
                )
            image_source = payload.get("reference_url") or payload.get(
                "input_reference"
            )
            preflight_req = VideoGenerationsRequest(**payload)
            validate_adapter_field_claims(preflight_req, adapter)
            adapter.validate_transport_options(
                preflight_req,
                model_path=server_args.model_path,
            )
            # Preserve the legacy behavior: URL/data references are localized
            # only when supplied through reference_url; local input_reference
            # paths are passed through to the selected adapter unchanged.
            if payload.get("reference_url"):
                try:
                    input_path = await _save_first_input_image(
                        image_source,
                        request_id,
                        uploads_dir,
                        prefer_remote_source=input_save_path is None,
                    )
                except Exception as e:
                    cleanup_request_resources()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    ) from e
                payload["input_reference"] = input_path
            req = VideoGenerationsRequest(**payload)
        except HTTPException:
            cleanup_request_resources()
            raise
        except Exception as e:
            cleanup_request_resources()
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    try:
        validate_adapter_field_claims(req, adapter)
        adapter.validate_transport_options(req, model_path=server_args.model_path)
    except (TypeError, ValueError) as exc:
        for resource in temp_resources:
            resource.cleanup()
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Resolve per-request output_path override without imposing a shared path
    # allowlist. The cloud-only fallback remains the historical behavior.
    effective_output_path = req.output_path or getattr(server_args, "output_path", None)
    if effective_output_path is None:
        if not cloud_storage.is_enabled():
            for resource in temp_resources:
                resource.cleanup()
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Video delivery is unavailable: configure an output path "
                    "or enable cloud storage"
                ),
            )
        output_resource = tempfile.TemporaryDirectory(prefix="sglang_output_")
        temp_resources.append(output_resource)
        output_tmp = output_resource.name
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False

    req.output_path = effective_output_path

    logger.debug(
        "video request admitted: id=%s model=%s task=%s",
        request_id,
        req.model,
        _extra_value(req, "task"),
    )

    batch: Req | None = None
    try:
        sampling_params = adapter.lower_sampling_params(request_id, req)
        trace_headers = extract_trace_headers(request.headers)
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
            external_trace_header=trace_headers,
        )
        await adapter.prepare_for_queue(batch)
        job = _video_job_from_batch(request_id, req, batch, adapter)
    except (ValueError, TypeError) as e:
        if batch is not None:
            await adapter.cleanup_request(batch)
        for resource in temp_resources:
            resource.cleanup()
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        if batch is not None:
            await adapter.cleanup_request(batch)
        for resource in temp_resources:
            resource.cleanup()
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
        if "max_sequence_length" in req.diffusers_kwargs:
            batch.max_sequence_length = req.diffusers_kwargs["max_sequence_length"]
        if "flow_shift" in req.diffusers_kwargs:
            batch.flow_shift = req.diffusers_kwargs["flow_shift"]

    try:
        await VIDEO_STORE.upsert(request_id, job)
        asyncio.create_task(
            _dispatch_job_async(
                request_id,
                batch,
                adapter=adapter,
                temp_dirs=temp_dirs or None,
                temp_resources=temp_resources or None,
                output_persistent=output_persistent,
            )
        )
    except Exception:
        await adapter.cleanup_request(batch)
        for resource in temp_resources:
            resource.cleanup()
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=order != "asc")

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []
    if limit is not None:
        jobs = jobs[:limit]
    return VideoListResponse(data=[VideoResponse(**job) for job in jobs])


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(**job)


# TODO: support aborting a job.
@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.pop(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...),
    variant: Optional[str] = Query(None),
    output_index: int = Query(0, ge=0),
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if variant not in (None, "video"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video content variant: {variant!r}",
        )

    urls = job.get("urls") or ([job["url"]] if job.get("url") else [])
    if urls:
        if output_index >= len(urls):
            raise HTTPException(status_code=404, detail="Video output not found")
        raise HTTPException(
            status_code=400,
            detail=(
                "Video has been uploaded to cloud storage. "
                f"Please use the cloud URL: {urls[output_index]}"
            ),
        )

    file_paths = job.get("file_paths") or (
        [job["file_path"]] if job.get("file_path") else []
    )
    if output_index >= len(file_paths):
        detail = (
            "Generation is still in-progress"
            if job.get("status") in {"queued", "in_progress"}
            else "Video output not found"
        )
        raise HTTPException(status_code=404, detail=detail)
    file_path = file_paths[output_index]
    if not file_path or not os.path.exists(file_path):
        status = job.get("status")
        detail = (
            "Generation is still in-progress"
            if status in {"queued", "in_progress"}
            else "Video output not found"
        )
        raise HTTPException(status_code=404, detail=detail)

    media_type = "video/mp4"
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
