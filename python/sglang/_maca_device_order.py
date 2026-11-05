# SPDX-License-Identifier: Apache-2.0
"""Early, dependency-free MACA device ordering for supported CLI launches.

Explicit visibility selects the input devices; mx-cmi selects their order.
Keep this module independent of torch,
srt.utils and ServerArgs: importing them can discover GPUs before injection.
The tool owns topology search. Its independent axes describe the Attention
layout; overlapping MoE EP and decode-CP groups are validated separately.
"""

import argparse
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_ENV = "MACA_VISIBLE_DEVICES"
_TOOL_TIMEOUT = 30
_logger = logging.getLogger(__name__)
_configured = None


class DeviceOrderError(ValueError):
    """A device ordering request cannot be used safely."""


class _DeviceOrderToolNotFound(DeviceOrderError):
    """Automatic ordering is unavailable; preserve existing visibility."""


@dataclass(frozen=True)
class _Request:
    # Tool dimensions are independent; SGLang CP/EP share outer TP devices.
    dimensions: tuple
    topology: tuple
    count: int
    ep_size: int = 1
    dcp_size: int = 1

    @property
    def visible(self):
        return ",".join(map(str, range(self.count)))


def _diagnostic_text(text):
    """Use neutral tool names in external diagnostics and log messages."""
    labels = {
        "mx-cmi": "device-order tool",
        "mx-smi": "device-inventory tool",
    }
    return re.sub(
        "|".join(re.escape(name) for name in labels),
        lambda match: labels[match.group().lower()],
        text,
        flags=re.IGNORECASE,
    )


def _log(level, message):
    # Package initialization precedes the normal CLI logging setup. Do not
    # reconfigure the root logger or leave a duplicate handler behind.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [device-order] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    old_level, old_propagate = _logger.level, _logger.propagate
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    try:
        _logger.log(level, _diagnostic_text(message))
    finally:
        _logger.removeHandler(handler)
        _logger.setLevel(old_level)
        _logger.propagate = old_propagate


def _launch_args():
    original = getattr(sys, "orig_argv", [])
    if "-c" in original or "--multiprocessing-fork" in original:
        return None
    executable = Path(sys.argv[0]).name if sys.argv else ""
    if executable in ("sglang", "sglang.exe") and sys.argv[1:2] == ["serve"]:
        return sys.argv[2:]
    # Under -m, package initialization happens while argv[0] is still "-m".
    if "-m" in original:
        index = original.index("-m")
        if original[index + 1 : index + 2] == ["sglang.launch_server"]:
            return sys.argv[1:]
    return None


def _is_maca():
    try:
        return "metax" in importlib.metadata.version("torch").lower()
    except importlib.metadata.PackageNotFoundError:
        return False


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DeviceOrderError(message)


def _parse_args(argv, selection_only=False):
    parser = _Parser(add_help=False, allow_abbrev=False)
    # These are the 0.5.21 CLI's explicit aliases plus its unambiguous --tp/--dp
    # prefixes. ServerArgs still performs the complete parse/validation later.
    integers = {
        "tp_size": (1, ["--tensor-parallel-size", "--tp"]),
        "dp_size": (1, ["--data-parallel-size", "--dp"]),
        "pp_size": (1, ["--pipeline-parallel-size"]),
        "dcp_size": (1, ["--decode-context-parallel-size"]),
        "attn_cp_size": (1, ["--attention-context-parallel-size"]),
        "ep_size": (1, ["--expert-parallel-size", "--ep"]),
        "nnodes": (1, []),
        "node_rank": (0, []),
        "base_gpu_id": (0, []),
        "gpu_id_step": (1, []),
        "moe_dp_size": (1, ["--moe-data-parallel-size"]),
    }
    for name, (default, aliases) in ({} if selection_only else integers).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            *aliases,
            dest=name,
            type=int,
            default=default,
        )
    for name in (
        ()
        if selection_only
        else ("enable_dp_attention", "enable_prefill_cp", "use_ray")
    ):
        parser.add_argument("--" + name.replace("_", "-"), action="store_true")
    parser.add_argument("--model-path", "--model", dest="model_path")
    parser.add_argument("--model-type", default="auto")
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--encoder-only", action="store_true")
    # Consume exact non-topology options that share guarded prefixes below.
    # ServerArgs owns their value validation; they do not alter device groups.
    for option in (
        "--pp-max-micro-batch-size",
        "--pp-async-batch-depth",
        "--dcp-comm-backend",
        "--ep-num-redundant-experts",
        "--ep-dispatch-algorithm",
        "--eplb-algorithm",
        "--eplb-rebalance-num-iterations",
        "--eplb-rebalance-layers-per-chunk",
        "--eplb-min-rebalancing-utilization-threshold",
    ):
        parser.add_argument(option)
    parser.add_argument("--dcp-replicate-q-proj", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--enable-dp-attention-local-control-broadcast", action="store_true"
    )
    # Match serve's dispatch preprocessing before parsing the model flags.
    args = []
    model_type = "auto"
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--model-type":
            if index + 1 == len(argv):
                raise DeviceOrderError("--model-type requires a value.")
            model_type = argv[index + 1]
            index += 2
        elif token.startswith("--model-type="):
            model_type = token.split("=", 1)[1]
            index += 1
        else:
            args.append(token)
            index += 1
    if args and not args[0].startswith("-"):
        args = ["--model-path", args[0], *args[1:]]
    model_count = sum(
        token.split("=", 1)[0] in ("--model-path", "--model") for token in args
    )
    parsed, unknown = parser.parse_known_args(["--model-type", model_type, *args])
    parsed.multiple_models = model_count > 1
    if selection_only:
        return parsed
    for token in unknown:
        option = token.split("=", 1)[0]
        # Never silently use a default for an unrecognized topology prefix.
        if option.startswith(
            (
                "--tp",
                "--dp",
                "--pp",
                "--ep",
                "--attn-cp",
                "--attention-context",
                "--tensor-parallel",
                "--data-parallel",
                "--pipeline-parallel",
                "--expert-parallel",
                "--nn",
                "--node-r",
                "--base-gpu",
                "--gpu-id",
                "--enable-prefill-c",
                "--enable-dp-att",
                "--moe-dp",
                "--decode-context",
                "--enable-dsa-prefill-context",
                "--enable-nsa-prefill-context",
                "--dsa-prefill-cp-mode",
                "--nsa-prefill-cp-mode",
                "--prefill-cp-mode",
                "--elastic-ep",
                "--max-ep-size",
                "--enable-attn-cp-v2",
                "--dcp",
                "--moe-data-parallel",
                "--dwdp",
                "--use-r",
            )
        ):
            raise DeviceOrderError(
                f"Unsupported or ambiguous topology option {option!r}; use its full name."
            )
    return parsed


def _local_model_config(args):
    if not args.model_path:
        return {}
    path = Path(args.model_path)
    try:
        with (path / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        return config if isinstance(config, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_llm(args, model_config):
    if args.model_type == "llm":
        return True
    if args.model_type != "auto" or not args.model_path:
        return False
    # serve's backend detection reads the first model flag, whereas argparse
    # uses the last. Leave ambiguous dispatch untouched by the automatic path.
    if args.multiple_models:
        return False
    # Do not import the diffusion registry or third-party backend factories.
    # The first demo auto-detects local HF causal-LM directories only.
    if (
        importlib.metadata.entry_points(group="sglang.serve_backends")
        or os.environ.get("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY")
        or (Path(args.model_path) / "model_index.json").exists()
    ):
        return False
    architectures = model_config.get("architectures", [])
    return isinstance(architectures, list) and any(
        isinstance(name, str) and name.endswith("ForCausalLM") for name in architectures
    )


def _validate_explicit_cp(args):
    # Prefill CP's model/backend-specific defaults cannot be resolved without
    # loading GPU-dependent code. Do not duplicate the model resolver here.
    if args.enable_prefill_cp and args.attn_cp_size == 1:
        raise DeviceOrderError(
            "This automatic-ordering demo requires explicit --attn-cp-size "
            "with --enable-prefill-cp (for TP8CP8, use --attn-cp-size 8)."
        )


def _build_request(args):
    t, d, c, p, e, k, n = (
        args.tp_size,
        args.dp_size,
        args.attn_cp_size,
        args.pp_size,
        args.ep_size,
        args.dcp_size,
        args.nnodes,
    )
    if min(t, d, c, p, e, k, n) < 1 or not 0 <= args.node_rank < n:
        raise DeviceOrderError(
            "Parallel sizes must be positive and node_rank < nnodes."
        )
    if args.base_gpu_id != 0 or args.gpu_id_step != 1:
        raise DeviceOrderError(
            "Automatic ordering requires base_gpu_id=0 and gpu_id_step=1."
        )
    if args.use_ray or args.moe_dp_size != 1:
        raise DeviceOrderError("Ray and MoE DP are outside this adapter.")
    if d > 1 and p > 1:
        raise DeviceOrderError("Simultaneous DP and PP are outside this demo.")
    if d > 1 and not args.enable_dp_attention:
        raise DeviceOrderError(
            "This demo requires --enable-dp-attention when dp_size > 1."
        )
    if t % (d * c):
        raise DeviceOrderError("tp_size must be divisible by dp_size * attn_cp_size.")
    # EP spans ranks within one PP stage, not the full TP * PP device pool.
    if t % e:
        raise DeviceOrderError("ep_size must divide tp_size within each PP stage.")
    a = t // (d * c)
    # Decode CP subdivides an existing Attention TP group. It does not reduce
    # Attention TP or add workers, unlike a separate Attention CP axis.
    if a % k:
        raise DeviceOrderError(
            "dcp_size must divide Attention TP (tp_size / dp_size / attn_cp_size)."
        )
    if n <= p:
        if p % n:
            raise DeviceOrderError("PP stages must divide evenly between nodes.")
        local_p, local_t = p // n, t
    else:
        if n % p or t % (n // p):
            raise DeviceOrderError("Nodes per PP stage must divide tp_size evenly.")
        local_p, local_t = 1, t // (n // p)
    # A node owns a contiguous, aligned TP-rank interval. Only represent
    # rectangular local groups; never invent a truncated communication group.
    if local_t % (c * a) == 0:
        local_d, local_c, local_a = local_t // (c * a), c, a
    elif (c * a) % local_t == 0 and local_t % a == 0:
        local_d, local_c, local_a = 1, local_t // a, a
    elif a % local_t == 0:
        local_d, local_c, local_a = 1, 1, local_t
    else:
        raise DeviceOrderError(
            "This node's TP/CP/DP groups are not a supported local grid."
        )
    # Keep the established Attention ordering. Passing EP/DCP as independent
    # axes would duplicate ranks or replace the actual Attention groups.
    dimensions = (local_a, local_c, local_p, local_d, 1)
    return _Request(
        dimensions=dimensions,
        topology=(t, d, c, p, n, args.node_rank),
        count=local_t * local_p,
        ep_size=e,
        dcp_size=k,
    )


_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _parse_tool_output(stdout):
    """Collect KV fields without interpreting or applying the tool's output."""
    fields = {}
    for line in stdout.splitlines():
        match = _ASSIGNMENT.fullmatch(line.strip())
        if not match:
            continue
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        # Only consumers of a key need to reject ambiguous or invalid values.
        fields.setdefault(key, []).append(value)
    return fields


def _parse_order(stdout, count):
    assignments = _parse_tool_output(stdout).get(_ENV, [])
    if len(assignments) != 1:
        raise DeviceOrderError(
            f"The device-order tool must return exactly one {_ENV} assignment."
        )
    value = assignments[0]
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise DeviceOrderError(
            "The device-order tool returned an invalid device-order assignment."
        )
    ids = [int(item) for item in value.split(",")]
    if len(ids) != count or len(set(ids)) != count:
        raise DeviceOrderError(
            f"The device-order tool must return {count} distinct device IDs."
        )
    # Scheme 2 may change the numeric set. Container/UMD enforce the allocation
    # boundary; do not reinterpret these IDs as host ranks or shell commands.
    return value


def _diagnostics(stdout, stderr):
    # Preserve both streams: some tool versions report failures on stdout.
    def decode(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    return "".join(
        f"\n{name}:\n{_diagnostic_text(decode(value).strip())}"
        for name, value in (("stdout", stdout), ("stderr", stderr))
        if value and decode(value).strip()
    )


def _run_command(command):
    tool_label = _diagnostic_text(Path(command[0]).name)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_TOOL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeviceOrderError(
            f"{tool_label} timed out after {_TOOL_TIMEOUT}s."
            + _diagnostics(exc.stdout, exc.stderr)
        ) from exc
    except FileNotFoundError as exc:
        if Path(command[0]).name == "mx-cmi" and not Path(command[0]).exists():
            raise _DeviceOrderToolNotFound(
                f"{tool_label} is unavailable: {_diagnostic_text(str(exc))}"
            ) from exc
        raise DeviceOrderError(
            f"Cannot run {tool_label}: {_diagnostic_text(str(exc))}"
        ) from exc
    except OSError as exc:
        raise DeviceOrderError(
            f"Cannot run {tool_label}: {_diagnostic_text(str(exc))}"
        ) from exc
    if result.returncode:
        raise DeviceOrderError(
            f"{tool_label} exited with status {result.returncode}."
            + _diagnostics(result.stdout, result.stderr)
        )
    return result


def _manual_visible(value, count):
    # Empty/malformed input must not trigger the tool's all-device fallback.
    if not re.fullmatch(r"\s*[0-9]+(?:\s*,\s*[0-9]+)*\s*", value):
        raise DeviceOrderError(
            "MACA_VISIBLE_DEVICES must be a nonempty comma-separated list "
            "of nonnegative device IDs."
        )
    ids = [int(item) for item in value.split(",")]
    if len(ids) != count or len(set(ids)) != count:
        raise DeviceOrderError(
            f"MACA_VISIBLE_DEVICES must contain {count} distinct device IDs "
            "for this node."
        )
    return ",".join(map(str, ids))


def _get_device_count():
    """Count logical devices without initializing the parent GPU runtime.

    C588 mx-smi -L enumerates physical cards; each available C588 card exposes
    two MACA device IDs. Other models retain the existing one-ID-per-row rule.
    mx-cmi validates the requested IDs and owns their topology/order.
    """
    result = _run_command(["mx-smi", "-L"])
    ids = []
    die_count = 0
    for line in result.stdout.splitlines():
        if not line.lstrip().startswith("GPU#"):
            continue
        match = re.fullmatch(r"\s*GPU#([0-9]+)\s+(\S+)\s+\S+\s+Available\b.*", line)
        if not match and re.fullmatch(
            r"\s*GPU#([0-9]+)\s+\S+\s+\S+\s+Not Available\b.*", line
        ):
            continue
        if not match:
            raise DeviceOrderError(
                "Cannot determine available device IDs from the device-inventory tool."
                + _diagnostics(result.stdout, result.stderr)
            )
        ids.append(int(match[1]))
        die_count += 2 if match[2] in ("MXC588", "C588") else 1
    if not ids or sorted(ids) != list(range(len(ids))):
        raise DeviceOrderError(
            "The device-inventory tool must list a nonempty, contiguous set "
            "of available device IDs "
            "starting at 0." + _diagnostics(result.stdout, result.stderr)
        )
    if result.stderr.strip():
        _log(
            logging.INFO, "Device-inventory tool diagnostics:\n" + result.stderr.strip()
        )
    return die_count


def _run_tool(request, visible, tool_path="mx-cmi"):
    command = [tool_path, "device_order"]
    for name, size in zip(("tp", "cp", "pp", "dp", "ep"), request.dimensions):
        command.extend(("--" + name, str(size)))
    command.extend(("--visible-devices", visible))
    result = _run_command(command)
    try:
        order = _parse_order(result.stdout, request.count)
    except DeviceOrderError as exc:
        raise DeviceOrderError(
            str(exc) + _diagnostics(result.stdout, result.stderr)
        ) from exc
    if result.stderr.strip():
        _log(logging.INFO, "Device-order tool diagnostics:\n" + result.stderr.strip())
    return order


def maybe_configure_maca_devices():
    """Configure a MACA CLI launch before any GPU-related imports.

    Both explicit and automatically selected devices are ordered by mx-cmi.
    Ordinary imports, worker startup and other backends leave visibility alone.
    """
    global _configured
    if _configured is not None and _configured[0] == os.getpid():
        return
    argv = _launch_args()
    if argv is None or any(arg in ("-h", "--help", "--version") for arg in argv):
        return
    if not _is_maca():
        return
    try:
        args = _parse_args(argv, selection_only=True)
        # Encoder DP has its own worker layout, outside the LLM adapter.
        if args.encoder_only:
            return
        if args.device not in (None, "cuda", "maca"):
            return
        if args.model_type not in ("auto", "llm"):
            return
        model_config = _local_model_config(args)
        if not _is_llm(args, model_config):
            _log(
                logging.INFO,
                "Automatic ordering skipped: backend not identified as a supported LLM.",
            )
            return
        tool_path = shutil.which("mx-cmi")
        if tool_path is None:
            raise _DeviceOrderToolNotFound(
                "The device-order tool was not found in PATH."
            )
        # Config merging needs the full parser and is deferred in this demo;
        # do not mistake YAML topology for defaults and allocate the wrong GPUs.
        if args.config is not None:
            raise DeviceOrderError(
                "Device ordering with --config is not supported yet. "
                "Use explicit CLI parallel parameters."
            )
        args = _parse_args(argv)
        _validate_explicit_cp(args)
        request = _build_request(args)
        manual = _ENV in os.environ
        if manual:
            visible = _manual_visible(os.environ[_ENV], request.count)
        else:
            available = _get_device_count()
            if available < request.count:
                raise DeviceOrderError(
                    f"Insufficient visible devices: required_local={request.count}, "
                    f"available={available}."
                )
            visible = request.visible
        order = _run_tool(request, visible, tool_path)
        os.environ[_ENV] = order
        _configured = (os.getpid(), request, order)
        _log(
            logging.INFO,
            f"source={'manual' if manual else 'auto'} required_local={request.count} "
            f"ordering_basis=attention ep_size={request.ep_size} "
            f"moe_tp_size={args.tp_size // request.ep_size} "
            f"dcp_size={request.dcp_size} tool_dims={request.dimensions} "
            f"input_devices={visible} "
            f"MACA_VISIBLE_DEVICES={order}",
        )
    except _DeviceOrderToolNotFound as exc:
        visibility = (
            f"preserving {_ENV}={os.environ[_ENV]!r}"
            if _ENV in os.environ
            else f"leaving {_ENV} unset"
        )
        _log(
            logging.ERROR,
            f"{exc} Automatic device ordering is skipped; {visibility}; "
            "continuing with the existing startup behavior.",
        )
    except DeviceOrderError as exc:
        _log(logging.ERROR, str(exc))
        raise SystemExit(1) from None


def validate_maca_device_order(server_args):
    """Reject model overrides that changed the early layout, before workers."""
    global _configured
    if _configured is None or _configured[0] != os.getpid():
        return
    _, expected, order = _configured
    try:
        actual = _build_request(server_args)
        # Backend resolution can normalize EP (e.g. DeepEP uses outer TP).
        # Permit that known normalization without changing Attention ordering;
        # other EP changes and all DCP/layout changes remain unexpected.
        if actual.ep_size not in (expected.ep_size, server_args.tp_size):
            raise DeviceOrderError(
                f"Resolved ep_size changed from {expected.ep_size} "
                f"to {actual.ep_size}; only normalization to tp_size is supported "
                "after device ordering."
            )
        if (
            actual != replace(expected, ep_size=actual.ep_size)
            or os.environ.get(_ENV) != order
        ):
            raise DeviceOrderError(
                "Resolved parallel layout or visibility differs from automatic "
                "device ordering; refusing to create workers."
            )
        if actual.ep_size != expected.ep_size:
            _log(
                logging.INFO,
                f"Resolved ep_size {expected.ep_size} -> {actual.ep_size}; "
                f"moe_tp_size={server_args.tp_size // actual.ep_size}; "
                "keeping ordering_basis=attention "
                "and MACA_VISIBLE_DEVICES.",
            )
            _configured = (os.getpid(), actual, order)
    except DeviceOrderError as exc:
        _log(logging.ERROR, str(exc))
        raise SystemExit(1) from None
