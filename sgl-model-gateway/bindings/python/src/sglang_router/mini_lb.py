# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import ipaddress
import logging
import os
import random
import time
import urllib
import warnings
from http import HTTPStatus
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang.srt.utils.common import get_bool_env_var
from sglang_router.router_args import RouterArgs

try:
    from sglang.srt.observability.trace import (
        process_tracing_init,
        trace_get_remote_propagate_context,
        trace_req_finish,
        trace_req_start,
        trace_set_thread_info,
        trace_slice_end,
        trace_slice_start,
    )

    trace_package_imported = True
except ImportError:
    trace_package_imported = False

logger = logging.getLogger(__name__)

_DEBUG_SHORTEST_QUEUE = get_bool_env_var("SGLANG_DEBUG_SHORTEST_QUEUE")

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error

def _format_counts(counts: List[int]) -> str:
    return "[" + ",".join(str(count) for count in counts) + "]"

def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address

class MinCount:
    def __init__(self, num: int):
        assert num > 0
        self.mutex = asyncio.Lock()
        self.counts = [0] * num
        self.estimated_end_times = [1.0] * num

    async def inc_min(self):
        async with self.mutex:
            min_val, min_weight = self.counts[0], self.estimated_end_times[0]
            min_val_idx = 0
            for i, (v, w) in enumerate(zip(self.counts, self.estimated_end_times)):
                if v == 0:
                    min_val_idx = i
                    break
                if v < min_val or (v == min_val and w < min_weight):
                    min_val_idx, min_val, min_weight = i, v, w
            self.counts[min_val_idx] += 1
            self.estimated_end_times[min_val_idx] = time.perf_counter()
        return min_val_idx

    async def dec(self, index: int):
        async with self.mutex:
            assert self.counts[index] > 0
            self.counts[index] -= 1


def get_server_dp_size(url: str) -> int:
    error_count = 0
    while True:
        try:
            if error_count == 1:
                logger.info(f"Retrying to get dp size from {url}")
            response = requests.get(f"{url}/get_server_info", timeout=(1, 10))
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            error_count += 1
            time.sleep(1)
    response.raise_for_status()
    info = response.json()
    return info["dp_size"]


def pack_room_id(p_id: int, d_id: int) -> int:
    """Encode (p_id, d_id) using the same 10+10+43 layout as sglang backend."""
    _prefill_width = 10
    _decode_width = 10
    _random_width = 43
    if not (0 <= p_id < (1 << _prefill_width) and 0 <= d_id < (1 << _decode_width)):
        raise ValueError("p_id and d_id must each be in [0, 1023].")
    random_part = random.randrange(1 << _random_width)
    return (random_part << (_prefill_width + _decode_width)) | (p_id << _decode_width) | d_id


def unpack_room_id(room_id: int):
    _prefill_width = 10
    _decode_width = 10
    p_id = (room_id >> _decode_width) & ((1 << _prefill_width) - 1)
    d_id = room_id & ((1 << _decode_width) - 1)
    return p_id, d_id

class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.shortest_queue = router_args.enable_shortest_queue
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.otlp_traces_endpoint = router_args.otlp_traces_endpoint
        self.enable_trace = router_args.enable_trace
        if self.enable_trace and not trace_package_imported:
            logger.warning(
                "Tracing is not supported in this environment. Please install sglang."
            )
            self.enable_trace = False

        if self.shortest_queue:
            prefill_dp_sizes = [get_server_dp_size(url) for url in self.prefill_urls]
            decode_dp_sizes = [get_server_dp_size(url) for url in self.decode_urls]
            logger.info(f"Prefill dp sizes: {prefill_dp_sizes}, decode dp sizes: {decode_dp_sizes}")
            assert len(set(prefill_dp_sizes)) == 1, (
                f"All prefill servers must have the same dp size, got {prefill_dp_sizes}"
            )
            assert len(set(decode_dp_sizes)) == 1, (
                f"All decode servers must have the same dp size, got {decode_dp_sizes}"
            )
            self.prefill_dp_size = prefill_dp_sizes[0]
            self.decode_dp_size = decode_dp_sizes[0]
            self.prefill_min_count = MinCount(sum(prefill_dp_sizes))
            self.decode_min_count = MinCount(sum(decode_dp_sizes))

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        if self.enable_trace:
            process_tracing_init(self.otlp_traces_endpoint, "sglang")
            trace_set_thread_info("Mini lb")
        uvicorn.run(app, host=self.host, port=self.port)

    async def _ensure_dp_sizes(self):
        if self.prefill_dp_size is not None:
            return
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prefill_urls[0]}/server_info") as resp:
                info = await resp.json()
                self.prefill_dp_size = len(info.get("internal_states", [1]))
            async with session.get(f"{self.decode_urls[0]}/server_info") as resp:
                info = await resp.json()
                self.decode_dp_size = len(info.get("internal_states", [1]))
        logger.info(
            f"[MiniLB] DP sizes: prefill={self.prefill_dp_size}, decode={self.decode_dp_size}"
        )

    def _fork_dp_requests(self, request):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["routed_dp_rank"] = p_rank
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    def select_pair(self):
        assert len(self.prefill_urls) > 0, "No prefill servers available"
        assert len(self.decode_urls) > 0, "No decode servers available"
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    async def select_pair_shortest(self):
        assert len(self.prefill_urls) > 0, "No prefill servers available"
        assert len(self.decode_urls) > 0, "No decode servers available"
        p_id = await self.prefill_min_count.inc_min()
        d_id = await self.decode_min_count.inc_min()
        room_id = pack_room_id(p_id, d_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.prefill_min_count.mutex:
                async with self.decode_min_count.mutex:
                    logger.info(
                        f"inc: {room_id=}, {p_id=}, {d_id=}, "
                        f"prefill={self.prefill_min_count.counts}, decode={self.decode_min_count.counts}"
                    )
        return (
            self.prefill_urls[p_id // self.prefill_dp_size],
            self.prefill_bootstrap_ports[p_id // self.prefill_dp_size],
            self.decode_urls[d_id // self.decode_dp_size],
            room_id,
        )

    async def completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        p_id, d_id = unpack_room_id(room_id)
        await self.prefill_min_count.dec(p_id)
        await self.decode_min_count.dec(d_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.prefill_min_count.mutex:
                async with self.decode_min_count.mutex:
                    logger.info(
                        f"dec: {room_id=}, {p_id=}, {d_id=}, "
                        f"prefill={self.prefill_min_count.counts}, decode={self.decode_min_count.counts}"
                    )

    async def prefill_completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        p_id, _ = unpack_room_id(room_id)
        await self.prefill_min_count.dec(p_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.prefill_min_count.mutex:
                logger.info(
                    f"dec prefill: {room_id=}, {p_id=}, prefill={_format_counts(self.prefill_min_count.counts)}"
                )

    async def decode_completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        _, d_id = unpack_room_id(room_id)
        await self.decode_min_count.dec(d_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.decode_min_count.mutex:
                logger.info(
                    f"dec decode: {room_id=}, {d_id=}, decode={_format_counts(self.decode_min_count.counts)}"
                )

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint, server_room_id=None
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            prefill_req, decode_req, expected_decode_dp_rank = self._fork_dp_requests(
                modified_request
            )
        else:
            prefill_req = modified_request
            decode_req = modified_request

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:

            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                session.post(f"{decode_server}/{endpoint}", json=decode_req),
            ]

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:

                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            if expected_decode_dp_rank is not None:
                actual = ret_json.get("meta_info", {}).get("dp_rank")
                if actual != expected_decode_dp_rank:
                    return ORJSONResponse(
                        content={
                            "error": f"DP rank mismatch: expected {expected_decode_dp_rank}, got {actual}"
                        },
                        status_code=500,
                    )

            await self.completed(server_room_id)
            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate", server_room_id=None
    ):

        if self.test_external_dp_routing:
            warnings.warn("--test-external-dp-routing is not supported with streaming")

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            prefill_released = False
            bootstrap_room_list = []
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=self.timeout
                    )  # Add timeout for request reliability
                ) as session:
                    # Create the tasks for both prefill and decode requests
                    headers = {}
                    if self.enable_trace:
                        bootstrap_room_list = (
                            modified_request["bootstrap_room"]
                            if isinstance(modified_request["bootstrap_room"], list)
                            else [modified_request["bootstrap_room"]]
                        )
                        trace_context = trace_get_remote_propagate_context(
                            bootstrap_room_list
                        )
                        headers = {"trace_context": trace_context}

                    tasks = [
                        session.post(
                            f"{prefill_server}/{endpoint}",
                            json=modified_request,
                            headers=headers,
                        ),
                        session.post(
                            f"{decode_server}/{endpoint}",
                            json=modified_request,
                            headers=headers,
                        ),
                    ]

                    for bootstrap_room in bootstrap_room_list:
                        trace_slice_end(
                            "mini_lb_launch", bootstrap_room, auto_next_anon=True
                        )
                    # Wait for both responses to complete. Since this is streaming, they return immediately.
                    prefill_response, decode_response = await asyncio.gather(*tasks)

                    if modified_request.get("return_logprob", False):
                        prefill_chunks = []
                        async for chunk in prefill_response.content:
                            prefill_chunks.append(chunk)

                        first_prefill_chunk = (
                            prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                        )
                        first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                        async for chunk in decode_response.content:
                            # Note: This is inefficient
                            # merge prefill input_token_logprobs, output_token_logprobs to decode
                            decoded_chunk = chunk.decode("utf-8")
                            if (
                                decoded_chunk
                                and decoded_chunk.startswith("data:")
                                and "[DONE]" not in decoded_chunk
                            ):
                                ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                                ret_json["meta_info"]["input_token_logprobs"] = (
                                    first_prefill_chunk_json["meta_info"][
                                        "input_token_logprobs"
                                    ]
                                    + ret_json["meta_info"]["input_token_logprobs"]
                                )

                                yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                            else:
                                yield chunk
                    else:
                        first = True
                        async for chunk in decode_response.content.iter_chunked(
                            AIOHTTP_STREAM_READ_CHUNK_SIZE
                        ):
                            if first:
                                first = False
                                await lb.prefill_completed(server_room_id)
                                prefill_released = True
                            yield chunk

                    for bootstrap_room in bootstrap_room_list:
                        trace_slice_end(
                            "wait_PD_finish",
                            bootstrap_room,
                            thread_finish_flag=True,
                        )
                        trace_req_finish(bootstrap_room)

            finally:
                if prefill_released:
                    await lb.decode_completed(server_room_id)
                else:
                    await lb.completed(server_room_id)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


# TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
@app.get("/server_info")
@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    server_room_id = None
    if lb.shortest_queue:
        prefill_server, bootstrap_port, decode_server, server_room_id = await lb.select_pair_shortest()
    else:
        prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": server_room_id if server_room_id is not None else _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request, prefill_server, decode_server, "generate", server_room_id=server_room_id
        )
    else:
        return await lb.generate(
            modified_request, prefill_server, decode_server, "generate", server_room_id=server_room_id
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    server_room_id = None
    if lb.shortest_queue:
        prefill_server, bootstrap_port, decode_server, server_room_id = await lb.select_pair_shortest()
    else:
        prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": server_room_id if server_room_id is not None else _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
            server_room_id=server_room_id,
        )
    else:
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
            server_room_id=server_room_id,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    bootstrap_room = random.randint(0, 2**63 - 1)
    if lb.enable_trace:
        trace_req_start(bootstrap_room, bootstrap_room, role="router")
        trace_slice_start("mini_lb_launch", bootstrap_room)
    return bootstrap_room


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
