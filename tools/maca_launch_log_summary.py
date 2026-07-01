#!/usr/bin/env python3
"""Summarize service, benchmark, or distributed runtime logs into JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

METRICS = ["latency", "throughput", "tokens", "port"]
ERROR_RE = re.compile(
    r"(error|failed|timeout|traceback|exception|segmentation fault|core dumped)", re.I
)
NUMBER_RE = re.compile(
    r"(?P<key>[A-Za-z_/-]+)\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?)"
)


def parse(path: Path) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, 1):
                if ERROR_RE.search(line):
                    errors.append({"line": lineno, "text": line.strip()})
                for match in NUMBER_RE.finditer(line):
                    key = match.group("key").lower()
                    if not METRICS or any(token in key for token in METRICS):
                        values.append(
                            {
                                "line": lineno,
                                "metric": key,
                                "value": float(match.group("value")),
                            }
                        )
    except OSError as exc:
        return {
            "path": str(path),
            "metric_count": 0,
            "error_count": 1,
            "metrics": [],
            "errors": [{"line": None, "text": f"cannot read log: {exc}"}],
        }
    return {
        "path": str(path),
        "metric_count": len(values),
        "error_count": len(errors),
        "metrics": values,
        "errors": errors,
    }


def self_test() -> None:
    with TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "log_summary_sample.log"
        sample.write_text("latency_ms=12.5\nERROR timeout\n", encoding="utf-8")
        data = parse(sample)
    if data["metric_count"] != 1 or data["error_count"] != 1:
        raise RuntimeError(f"self-test failed: {data}")
    print(
        json.dumps(
            {"ok": True, "metric_count": data["metric_count"]}, ensure_ascii=False
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps([parse(Path(p)) for p in args.logs], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
