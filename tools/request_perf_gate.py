#!/usr/bin/env python3
"""Compare baseline and current performance JSON and fail on regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

METRIC = "throughput"
TOLERANCE = 0.05


def _json_payloads(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        payloads: list[object] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc
        return payloads


def _append_metric(rows: dict[str, float], name: object, value: object) -> None:
    try:
        rows[str(name)] = float(value)
    except (TypeError, ValueError):
        return


def _collect_metrics(payload: object, metric: str, rows: dict[str, float]) -> None:
    if isinstance(payload, list):
        for item in payload:
            _collect_metrics(item, metric, rows)
        return
    if not isinstance(payload, dict):
        return
    if "name" in payload and metric in payload:
        _append_metric(rows, payload["name"], payload[metric])
        return
    for name, value in payload.items():
        if isinstance(value, dict):
            if metric in value:
                _append_metric(rows, name, value[metric])
        else:
            _append_metric(rows, name, value)


def load(path: Path, metric: str = METRIC) -> dict[str, float]:
    rows: dict[str, float] = {}
    for payload in _json_payloads(path):
        _collect_metrics(payload, metric, rows)
    return rows


def compare(
    baseline: dict[str, float],
    current: dict[str, float],
    metric: str = METRIC,
    tolerance: float = TOLERANCE,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failed = False
    for name, old in sorted(baseline.items()):
        if name not in current:
            rows.append({"name": name, "status": "missing-current"})
            failed = True
            continue
        new = current[name]
        ratio = (new - old) / old if old else 0.0
        status = "regression" if ratio < -tolerance else "ok"
        failed = failed or status != "ok"
        rows.append(
            {
                "name": name,
                "baseline": old,
                "current": new,
                "delta_ratio": ratio,
                "status": status,
            }
        )
    return {"ok": not failed, "metric": metric, "tolerance": tolerance, "rows": rows}


def self_test() -> None:
    data_ok = compare({"case": 100.0}, {"case": 99.0})
    data_regression = compare({"case": 100.0}, {"case": 90.0})
    data_missing = compare({"case": 100.0}, {})
    with TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "request_perf.jsonl"
        sample.write_text(
            '{"name": "prefill", "throughput": 10}\n'
            '{"decode": {"throughput": 20}}\n',
            encoding="utf-8",
        )
        loaded = load(sample)
    if (
        not data_ok["ok"]
        or data_regression["ok"]
        or data_missing["ok"]
        or loaded != {"prefill": 10.0, "decode": 20.0}
    ):
        raise RuntimeError(
            {
                "ok_case": data_ok,
                "regression_case": data_regression,
                "missing_case": data_missing,
                "loaded": loaded,
            }
        )
    print(json.dumps({"ok": True, "rows": len(data_ok["rows"])}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?")
    parser.add_argument("current", nargs="?")
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.baseline or not args.current:
        parser.error("baseline and current are required unless --self-test is used")
    result = compare(
        load(Path(args.baseline), args.metric),
        load(Path(args.current), args.metric),
        metric=args.metric,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
