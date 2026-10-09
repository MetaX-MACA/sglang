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
from datetime import datetime
from http import HTTPStatus
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

AIOHTTP_STREAM_READ_CHUNK_SIZE = 64 * 1024

_room_id_prefill_width = 10
_room_id_decode_width = 10
_room_id_random_width = 43
_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).lower()
    if value not in ("true", "1", "false", "0"):
        if name not in _warned_bool_env_var_keys:
            logging.getLogger("pdlb").warning(
                f"Invalid boolean value for {name}: {value!r}; treating it as false"
            )
            _warned_bool_env_var_keys.add(name)
        return False
    return value in ("true", "1")

def pack_room_id(prefill_id, decode_id, random_len=43):
    """
    Encode to an integer without losing leading zeros:
    - prefill_instance: stored in the highest digits
    - decode_instance:  stored in the middle digits
    """
    if not (0 <= prefill_id < (1 << _room_id_prefill_width) and 0 <= decode_id < (1 << _room_id_decode_width)):
        raise ValueError("prefill_instance and decode_instance must be between 0 and 1023.")

    random_part = random.randrange(1 << _room_id_random_width)
    value = (random_part << (_room_id_prefill_width + _room_id_decode_width)) | (prefill_id << _room_id_decode_width) | decode_id
    return value


def unpack_room_id(value):
    """
    Decode the integer using arithmetic operations.
    Returns:
      prefill_instance (int),
      decode_instance  (int),
    """
    return ((value >> _room_id_decode_width) & (1 << _room_id_prefill_width) - 1), ((value >> 0) & (1 << _room_id_decode_width) - 1)


_TTFT_PREFILL_RETURN_FIRST_TOKEN = get_bool_env_var("SGLANG_TTFT_PREFILL_RETURN_FIRST_TOKEN")

_DEBUG_TTFT = get_bool_env_var("SGLANG_DEBUG_TTFT")
_DEBUG_SHORTEST_QUEUE = get_bool_env_var("SGLANG_DEBUG_SHORTEST_QUEUE")


def current_datetime():
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{now.microsecond:06d}"


@dataclasses.dataclass
class PDRegistryRequest:
    """A request to register a machine itself to the LB."""

    mode: str
    registry_url: str
    bootstrap_port: Optional[int] = None

    def __post_init__(self):
        if self.mode == "prefill" and self.bootstrap_port is None:
            raise ValueError("Bootstrap port must be set in PREFILL mode.")
        elif self.mode == "decode" and self.bootstrap_port is not None:
            raise ValueError("Bootstrap port must not be set in DECODE mode.")
        elif self.mode not in ["prefill", "decode"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'prefill' or 'decode'."
            )


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    # 注意这里的 %f 是微秒（6 位），通过字符串切片保留前 3 位表示毫秒
    class MilliSecondFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            from datetime import datetime
            ct = self.converter(record.created)
            if datefmt:
                s = datetime.fromtimestamp(record.created).strftime(datefmt)
                return s[:-3]  # 去掉最后三位微秒，保留毫秒
            else:
                t = datetime.fromtimestamp(record.created)
                return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    formatter = MilliSecondFormatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%f",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


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

@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


def get_server_dp_size(url: str) -> int:
    error_count = 0
    while True:
        try:
            if error_count == 1:
                logger.info(f"Retrying to get dp size from {url}")
            response = requests.get(f"{url}/server_info", timeout=(1, 10))
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            error_count += 1
            time.sleep(1)
    response.raise_for_status()
    info = response.json()
    return info["dp_size"]


class MiniLoadBalancer:
    def __init__(
        self,
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        shortest_queue: bool,
        client_timeout: int = 10800,
        test_external_dp_routing: bool = False,
        enable_trace: bool = False,
        otlp_traces_endpoint: str = "localhost:4317",
    ):
        self.shortest_queue = shortest_queue
        self.client_timeout = client_timeout
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
        self.test_external_dp_routing = test_external_dp_routing
        self.enable_trace = enable_trace
        self.trace_req_context_cls = None
        self.prefill_dp_size = None
        self.decode_dp_size = None
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        if self.enable_trace:
            try:
                from sglang.srt.observability.trace import (
                    TraceReqContext,
                    process_tracing_init,
                    trace_set_thread_info,
                )

                process_tracing_init(otlp_traces_endpoint, "sglang")
                trace_set_thread_info("Mini lb")
                self.trace_req_context_cls = TraceReqContext
            except (ImportError, RuntimeError) as exc:
                logger.warning(f"Tracing is disabled: {exc}")
                self.enable_trace = False

        if self.shortest_queue:
            prefill_dp_sizes = [get_server_dp_size(p.url) for p in prefill_configs]
            decode_dp_sizes = [get_server_dp_size(d) for d in decode_servers]
            logger.info(f"Prefill dp sizes: {prefill_dp_sizes}, decode dp sizes: {decode_dp_sizes}")
            assert len(set(prefill_dp_sizes)) == 1, f"All prefill servers must have the same dp size, got {prefill_dp_sizes}"
            assert len(set(decode_dp_sizes)) == 1, f"All decode servers must have the same dp size, got {decode_dp_sizes}"
            self.prefill_dp_size = prefill_dp_sizes[0]
            self.decode_dp_size = decode_dp_sizes[0]
            self.prefill_min_count = MinCount(sum(prefill_dp_sizes))
            self.decode_min_count = MinCount(sum(decode_dp_sizes))


    async def _ensure_dp_sizes(self):
        if self.prefill_dp_size is not None and self.decode_dp_size is not None:
            return

        async def get_dp_size(session, url):
            async with session.get(f"{url}/server_info") as response:
                response.raise_for_status()
                info = await response.json()
            dp_size = info.get("dp_size")
            if dp_size is None:
                dp_size = len(info.get("internal_states", []))
            if not isinstance(dp_size, int) or dp_size <= 0:
                raise RuntimeError(f"Invalid DP size from {url}/server_info: {dp_size}")
            return dp_size

        async with aiohttp.ClientSession() as session:
            self.prefill_dp_size, self.decode_dp_size = await asyncio.gather(
                get_dp_size(session, self.prefill_servers[0]),
                get_dp_size(session, self.decode_servers[0]),
            )
        logger.info(
            f"External DP routing sizes: prefill={self.prefill_dp_size}, "
            f"decode={self.decode_dp_size}"
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

    @staticmethod
    def _get_response_dp_ranks(response):
        responses = response if isinstance(response, list) else [response]
        return [item.get("meta_info", {}).get("dp_rank") for item in responses]

    def _start_trace(self, request):
        if not self.enable_trace:
            return None, {}

        bootstrap_room = request.get("bootstrap_room")
        if isinstance(bootstrap_room, list):
            bootstrap_room = bootstrap_room[0] if bootstrap_room else None
        trace_ctx = self.trace_req_context_cls(
            rid=bootstrap_room,
            bootstrap_room=bootstrap_room,
            role="router",
            module_name="mini_lb",
        )
        trace_ctx.trace_req_start()
        trace_ctx.trace_slice_start("mini_lb_launch", level=1)
        trace_state = trace_ctx.__getstate__()
        return trace_ctx, trace_state.get("root_span_context", {})

    @staticmethod
    def _start_trace_wait(trace_ctx):
        if trace_ctx is None:
            return False
        trace_ctx.trace_slice_end("mini_lb_launch", level=1)
        trace_ctx.trace_slice_start("wait_PD_finish", level=1)
        return True

    @staticmethod
    def _finish_trace(trace_ctx, wait_started):
        if trace_ctx is None:
            return
        if wait_started:
            trace_ctx.trace_slice_end(
                "wait_PD_finish", level=1, thread_finish_flag=True
            )
        trace_ctx.trace_req_finish()

    async def select_pair(self):
        if self.shortest_queue:
            p_id = await self.prefill_min_count.inc_min()
            d_id = await self.decode_min_count.inc_min()
            room_id = pack_room_id(p_id, d_id)
            p_c = self.prefill_configs[p_id // self.prefill_dp_size]
            d_c = self.decode_servers[d_id // self.decode_dp_size]
            if _DEBUG_SHORTEST_QUEUE:
                async with self.prefill_min_count.mutex:
                    async with self.decode_min_count.mutex:
                        logger.info(f"inc: {room_id=}, {p_id=}, {d_id=}, prefill={self.prefill_min_count.counts}, decode={self.decode_min_count.counts}")
        else:
            room_id = None
            p_c = random.choice(self.prefill_configs)
            d_c = random.choice(self.decode_servers)

        return p_c.url, p_c.bootstrap_port, d_c, room_id

    async def completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        p_id, d_id = unpack_room_id(room_id)
        await self.prefill_min_count.dec(p_id)
        await self.decode_min_count.dec(d_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.prefill_min_count.mutex:
                async with self.decode_min_count.mutex:
                    logger.info(f"dec: {room_id=}, {p_id=}, {d_id=}, prefill={self.prefill_min_count.counts}, decode={self.decode_min_count.counts}")

    async def prefill_completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        p_id, _ = unpack_room_id(room_id)
        await self.prefill_min_count.dec(p_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.prefill_min_count.mutex:
                logger.info(
                    f"dec prefill: {room_id=}, {p_id=}, "
                    f"prefill={_format_counts(self.prefill_min_count.counts)}"
                )

    async def decode_completed(self, room_id):
        if not self.shortest_queue or room_id is None:
            return
        _, d_id = unpack_room_id(room_id)
        await self.decode_min_count.dec(d_id)
        if _DEBUG_SHORTEST_QUEUE:
            async with self.decode_min_count.mutex:
                logger.info(
                    f"dec decode: {room_id=}, {d_id=}, "
                    f"decode={_format_counts(self.decode_min_count.counts)}"
                )

    async def generate(
        self,
        modified_request,
        prefill_server,
        decode_server,
        endpoint,
        server_room_id=None,
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        expected_decode_dp_rank = None
        trace_ctx, headers = self._start_trace(modified_request)
        trace_wait_started = False

        try:
            if self.test_external_dp_routing:
                await self._ensure_dp_sizes()
                prefill_req, decode_req, expected_decode_dp_rank = (
                    self._fork_dp_requests(modified_request)
                )
            else:
                prefill_req = modified_request
                decode_req = modified_request

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.client_timeout)
            ) as session:
                tasks = [
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=prefill_req,
                        headers=headers,
                    ),
                    session.post(
                        f"{decode_server}/{endpoint}",
                        json=decode_req,
                        headers=headers,
                    ),
                ]
                trace_wait_started = self._start_trace_wait(trace_ctx)
                # Prefill should complete before decode, but both responses are needed.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if "return_logprob" in modified_request:
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()

                    # Merge prefill input logprobs into the decode response.
                    if "meta_info" in ret_json:
                        if "input_token_logprobs" in ret_json["meta_info"]:
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                prefill_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                else:
                    ret_json = await decode_response.json()

                if expected_decode_dp_rank is not None:
                    actual_dp_ranks = self._get_response_dp_ranks(ret_json)
                    if any(
                        rank != expected_decode_dp_rank for rank in actual_dp_ranks
                    ):
                        return ORJSONResponse(
                            content={
                                "error": (
                                    "DP rank mismatch: expected "
                                    f"{expected_decode_dp_rank}, got {actual_dp_ranks}"
                                )
                            },
                            status_code=500,
                        )

                return ORJSONResponse(
                    content=ret_json,
                    status_code=decode_response.status,
                )
        finally:
            await self.completed(server_room_id)
            self._finish_trace(trace_ctx, trace_wait_started)

    async def generate_stream(
        self,
        modified_request,
        prefill_server,
        decode_server,
        endpoint="generate",
        server_room_id=None,
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        if self.test_external_dp_routing:
            warnings.warn("--test-external-dp-routing is not supported with streaming")

        async def stream_results():
            prefill_release = False
            trace_ctx = None
            trace_wait_started = False
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.client_timeout),
                    read_bufsize=10 * 1024 * 1024,
                ) as session:
                    # Create the tasks for both prefill and decode requests
                    if _DEBUG_TTFT:
                        logger.info(
                            f"room_id: {server_room_id}, "
                            f"mini_lb_start: {current_datetime()}"
                        )
                    trace_ctx, headers = self._start_trace(modified_request)
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
                    trace_wait_started = self._start_trace_wait(trace_ctx)
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
                        if _TTFT_PREFILL_RETURN_FIRST_TOKEN:
                            first_chunk_origin = None
                            async for chunk in prefill_response.content:
                                if first_chunk_origin is None:
                                    first_chunk_origin = chunk

                            await self.prefill_completed(server_room_id)
                            prefill_release = True
                            if _DEBUG_TTFT:
                                logger.info(
                                    f"room_id: {server_room_id}, "
                                    f"mini_lb_first: {current_datetime()}"
                                )

                            first_chunk = first_chunk_origin.decode("utf-8")[5:].strip("\n")
                            first_text = orjson.loads(first_chunk)["text"]
                            yield first_chunk_origin
                            async for chunk in decode_response.content:
                                if first_text is None:
                                    yield chunk
                                else:
                                    # TODO: 校验 Decode 返回的第一个 chunk 的 text 前缀是否和 first_text 匹配
                                    first_text = None
                        else:
                            first = True
                            async for chunk in decode_response.content.iter_chunked(
                                AIOHTTP_STREAM_READ_CHUNK_SIZE
                            ):
                                if first:
                                    first = False
                                    await self.prefill_completed(server_room_id)
                                    prefill_release = True
                                    if _DEBUG_TTFT:
                                        logger.info(
                                            f"room_id: {server_room_id}, "
                                            f"mini_lb_first: {current_datetime()}"
                                        )

                                yield chunk
            finally:
                if prefill_release:
                    await self.decode_completed(server_room_id)
                else:
                    await self.completed(server_room_id)
                self._finish_trace(trace_ctx, trace_wait_started)


        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
load_balancer: MiniLoadBalancer | None = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/server_info")
async def server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    all_internal_states = []
    async with aiohttp.ClientSession() as session:
        for server in prefill_servers:
            server_info = await session.get(f"{server}/server_info")
            prefill_infos.append(await server_info.json())
        for server in decode_servers:
            server_info = await session.get(f"{server}/server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            all_internal_states.extend(info_json.get("internal_states", []))

    if not all_internal_states:
        all_internal_states = [
            {
                "last_gen_throughput": 0.0,
                "avg_spec_accept_length": None,
            }
        ]
    return {
        "internal_states": all_internal_states,
        "prefill": prefill_infos,
        "decode": decode_infos,
    }


@app.get("/model_info")
async def model_info():
    if not load_balancer or not load_balancer.prefill_servers:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no prefill server registered",
        )

    target_server = load_balancer.prefill_servers[0]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{target_server}/model_info") as response:
                if response.status != HTTPStatus.OK:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server}. "
                            f"Status: {response.status}, response: {error_text}"
                        ),
                    )
                return ORJSONResponse(content=await response.json())
    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Failed to get model info from {target_server}: {exc}",
        ) from exc


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    """
    room_id:
        | 63                           ... 8 |  7 6 5 4 3 2 1 0   |
        | high bit: random num for unique id | 0 ~ 143, decode id |

    后续可使用 secrets.randbelow产生更加安全的随机数
    """
    prefill_server, bootstrap_port, decode_server, server_room_id = (
        await load_balancer.select_pair()
    )

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
                "bootstrap_room": (
                    server_room_id
                    if server_room_id is not None
                    else _generate_bootstrap_room()
                ),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            "generate",
            server_room_id=server_room_id,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            "generate",
            server_room_id=server_room_id,
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server, server_room_id = (
        await load_balancer.select_pair()
    )

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": (
                server_room_id
                if server_room_id is not None
                else _generate_bootstrap_room()
            ),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
            server_room_id=server_room_id,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
            server_room_id=server_room_id,
        )


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)

# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
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


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.prefill_configs.append(
            PrefillConfig(obj.registry_url, obj.bootstrap_port)
        )
        load_balancer.prefill_servers.append(obj.registry_url)
        logger.info(
            f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}"
        )
    elif obj.mode == "decode":
        load_balancer.decode_servers.append(obj.registry_url)
        logger.info(f"Registered decode server: {obj.registry_url}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(
        f"#Prefill servers: {len(load_balancer.prefill_configs)}, "
        f"#Decode servers: {len(load_balancer.decode_servers)}"
    )

    return Response(status_code=200)


def run(
    prefill_configs,
    decode_addrs,
    host,
    port,
    shortest_queue: bool,
    client_timeout: int = 10800,
    test_external_dp_routing: bool = False,
    enable_trace: bool = False,
    otlp_traces_endpoint: str = "localhost:4317",
):
    global load_balancer
    load_balancer = MiniLoadBalancer(
        prefill_configs,
        decode_addrs,
        shortest_queue,
        client_timeout,
        test_external_dp_routing,
        enable_trace,
        otlp_traces_endpoint,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--prefill", type=str, default=[], nargs="+", help="URLs for prefill servers"
    )
    parser.add_argument(
        "--decode", type=str, default=[], nargs="+", help="URLs for decode servers"
    )
    parser.add_argument(
        "--prefill-bootstrap-ports",
        type=int,
        nargs="+",
        help="Bootstrap ports for prefill servers",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    parser.add_argument(
        "--enable-shortest-queue", action="store_true", help="Enable shortest queue scheduling"
    )
    parser.add_argument(
        "--client-timeout", type=int, default=10800, help="Timeout (seconds) for aiohttp ClientSession (default: 10800)"
    )
    parser.add_argument(
        "--test-external-dp-routing",
        action="store_true",
        help=(
            "Randomly assign routed_dp_rank/disagg_prefill_dp_rank and verify "
            "the decode response"
        ),
    )
    parser.add_argument(
        "--enable-trace",
        action="store_true",
        help="Enable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--otlp-traces-endpoint",
        type=str,
        default="localhost:4317",
        help="OpenTelemetry collector endpoint in <host>:<port> format",
    )
    args = parser.parse_args()

    bootstrap_ports = args.prefill_bootstrap_ports
    if bootstrap_ports is None:
        bootstrap_ports = [None] * len(args.prefill)
    elif len(bootstrap_ports) == 1:
        bootstrap_ports = bootstrap_ports * len(args.prefill)
    else:
        if len(bootstrap_ports) != len(args.prefill):
            raise ValueError(
                "Number of prefill URLs must match number of bootstrap ports"
            )

    prefill_configs = [
        PrefillConfig(url, port) for url, port in zip(args.prefill, bootstrap_ports)
    ]

    run(
        prefill_configs,
        args.decode,
        args.host,
        args.port,
        args.enable_shortest_queue,
        args.client_timeout,
        args.test_external_dp_routing,
        args.enable_trace,
        args.otlp_traces_endpoint,
    )
