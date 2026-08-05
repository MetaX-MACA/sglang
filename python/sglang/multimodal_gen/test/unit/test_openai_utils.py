# SPDX-License-Identifier: Apache-2.0

import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    adjust_output_quality,
    flatten_extra_params,
)


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_flatten_extra_params_rejects_invalid_json_with_400():
    with pytest.raises(HTTPException) as exc_info:
        flatten_extra_params({"extra_params": "{not valid json"})

    assert exc_info.value.status_code == 400
    assert "extra_params" in exc_info.value.detail


def test_flatten_extra_params_promotes_valid_json_fields():
    payload = flatten_extra_params({"extra_params": '{"task": "t2va"}'})

    assert payload["task"] == "t2va"


def test_flatten_extra_params_keeps_non_string_payloads_unchanged():
    payload = flatten_extra_params({"guardrails": True})

    assert payload == {"guardrails": True, "use_guardrails": True}


def test_adjust_output_quality_maps_known_levels_and_keeps_none_default():
    assert adjust_output_quality(None) is None
    assert adjust_output_quality("high") == 90
    assert adjust_output_quality("default", DataType.VIDEO) == 50
    assert adjust_output_quality("default", DataType.IMAGE) == 75


def test_adjust_output_quality_rejects_unknown_value_with_400():
    with pytest.raises(HTTPException) as exc_info:
        adjust_output_quality("ultra")

    assert exc_info.value.status_code == 400
    for allowed in ("maximum", "high", "medium", "low", "default"):
        assert allowed in exc_info.value.detail
