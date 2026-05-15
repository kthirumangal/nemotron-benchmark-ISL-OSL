#!/usr/bin/env python3
"""
Run the Ao1 benchmark across multiple precision/deployment profiles.

The matrix runner expects each profile to expose an OpenAI-compatible
/v1/chat/completions endpoint. For hosted NVIDIA API, precision is managed by
the service. For self-hosted NIM/vLLM, run one endpoint per precision profile
and list each endpoint in the matrix CSV.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Optional


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def value(row: dict[str, str], key: str, default: str) -> str:
    raw = row.get(key, "")
    return raw.strip() if raw and raw.strip() else default


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "profile"


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def as_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarize_csv(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {
            "completed": "0",
            "errors": "0",
            "p50_ttft_s": "",
            "p90_ttft_s": "",
            "p99_ttft_s": "",
            "p50_total_latency_s": "",
            "p90_total_latency_s": "",
            "p99_total_latency_s": "",
            "p50_decode_tok_s": "",
            "p90_decode_tok_s": "",
            "p99_decode_tok_s": "",
            "meets_targets_runs": "0",
            "total_runs": "0",
        }

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") != "ok"]

    def series(key: str) -> list[float]:
        return [
            parsed
            for row in ok
            if (parsed := as_float(row.get(key, ""))) is not None
        ]

    def fmt(number: Optional[float]) -> str:
        return "" if number is None else f"{number:.3f}"

    ttft = series("ttft_s")
    total = series("total_latency_s")
    decode = series("decode_tokens_per_s")
    meets_targets = [row for row in ok if truthy(row.get("meets_targets", ""))]

    return {
        "completed": str(len(ok)),
        "errors": str(len(errors)),
        "p50_ttft_s": fmt(percentile(ttft, 0.50)),
        "p90_ttft_s": fmt(percentile(ttft, 0.90)),
        "p99_ttft_s": fmt(percentile(ttft, 0.99)),
        "p50_total_latency_s": fmt(percentile(total, 0.50)),
        "p90_total_latency_s": fmt(percentile(total, 0.90)),
        "p99_total_latency_s": fmt(percentile(total, 0.99)),
        "p50_decode_tok_s": fmt(percentile(decode, 0.50)),
        "p90_decode_tok_s": fmt(percentile(decode, 0.90)),
        "p99_decode_tok_s": fmt(percentile(decode, 0.99)),
        "meets_targets_runs": str(len(meets_targets)),
        "total_runs": str(len(ok)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run precision-profile benchmark matrix.")
    parser.add_argument("--matrix", default="precision_matrix.example.csv")
    parser.add_argument("--prompt-dir", default="of1-testprompts")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ttft-target-s", type=float, default=2.0)
    parser.add_argument("--total-latency-target-s", type=float, default=5.0)
    parser.add_argument("--throughput-target-tok-s", type=float, default=200.0)
    parser.add_argument("--default-max-tokens", default="1024")
    parser.add_argument("--default-runs", default="3")
    parser.add_argument("--default-concurrency", default="1")
    parser.add_argument("--default-timeout-s", default="180")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = pathlib.Path(args.matrix)
    if not matrix_path.exists():
        print(f"Matrix file not found: {matrix_path}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = pathlib.Path(args.output_dir or pathlib.Path("results") / f"precision-matrix-{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with matrix_path.open(newline="") as handle:
        profiles = [
            row
            for row in csv.DictReader(handle)
            if row.get("label", "").strip() and not row.get("label", "").strip().startswith("#")
        ]

    summary_rows: list[dict[str, str]] = []
    for row in profiles:
        label = value(row, "label", "profile")
        precision = value(row, "precision", label)
        out_csv = output_dir / f"{sanitize(label)}.csv"

        command = [
            sys.executable,
            "benchmark_nano.py",
            "--prompt-dir",
            args.prompt_dir,
            "--base-url",
            value(row, "base_url", "https://integrate.api.nvidia.com/v1"),
            "--model",
            value(row, "model", "nvidia/nemotron-3-nano-30b-a3b"),
            "--precision-label",
            precision,
            "--api-key-env",
            value(row, "api_key_env", "NVIDIA_API_KEY"),
            "--max-tokens",
            value(row, "max_tokens", args.default_max_tokens),
            "--runs",
            value(row, "runs", args.default_runs),
            "--concurrency",
            value(row, "concurrency", args.default_concurrency),
            "--timeout-s",
            value(row, "timeout_s", args.default_timeout_s),
            "--temperature",
            value(row, "temperature", "0.0"),
            "--ttft-target-s",
            str(args.ttft_target_s),
            "--total-latency-target-s",
            str(args.total_latency_target_s),
            "--throughput-target-tok-s",
            str(args.throughput_target_tok_s),
            "--output",
            str(out_csv),
        ]

        if truthy(row.get("enable_thinking", "")):
            command.append("--enable-thinking")
        if truthy(row.get("allow_missing_api_key", "")):
            command.append("--allow-missing-api-key")

        print(f"\n=== Running {label} ({precision}) ===", flush=True)
        completed = subprocess.run(command, text=True)
        profile_summary = summarize_csv(out_csv)

        p90_ttft = as_float(profile_summary["p90_ttft_s"])
        p90_total = as_float(profile_summary["p90_total_latency_s"])
        p50_decode = as_float(profile_summary["p50_decode_tok_s"])
        p90_ttft_pass = p90_ttft is not None and p90_ttft <= args.ttft_target_s
        p90_total_pass = p90_total is not None and p90_total <= args.total_latency_target_s
        p50_decode_pass = p50_decode is not None and p50_decode >= args.throughput_target_tok_s

        summary_rows.append(
            {
                "label": label,
                "precision": precision,
                "base_url": value(row, "base_url", "https://integrate.api.nvidia.com/v1"),
                "model": value(row, "model", "nvidia/nemotron-3-nano-30b-a3b"),
                "exit_code": str(completed.returncode),
                "output_csv": str(out_csv),
                "ttft_target_s": str(args.ttft_target_s),
                "total_latency_target_s": str(args.total_latency_target_s),
                "throughput_target_tok_s": str(args.throughput_target_tok_s),
                "p90_ttft_pass": str(p90_ttft_pass),
                "p90_total_latency_pass": str(p90_total_pass),
                "p50_decode_throughput_pass": str(p50_decode_pass),
                "meets_p90_targets": str(p90_ttft_pass and p90_total_pass and p50_decode_pass),
                **profile_summary,
            }
        )

    summary_path = output_dir / "summary.csv"
    fields = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote matrix summary: {summary_path}")
    return 0 if all(row["meets_p90_targets"] == "True" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
