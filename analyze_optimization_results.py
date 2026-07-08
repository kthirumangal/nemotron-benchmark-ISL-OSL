#!/usr/bin/env python3
"""Rank benchmark variants by latency while preserving accuracy."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Any


def as_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze optimization summary CSV.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--category", default="recommender")
    parser.add_argument("--min-accuracy-pass-rate", type=float, default=0.95)
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = pathlib.Path(args.summary).expanduser()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [
        row
        for row in rows
        if row.get("category", "") in {args.category, "ALL"}
        and as_float(row.get("p50_total_latency_s")) is not None
    ]
    passing = [
        row
        for row in filtered
        if (as_float(row.get("accuracy_pass_rate")) or 0.0)
        >= args.min_accuracy_pass_rate
    ]
    ranked = sorted(
        passing or filtered,
        key=lambda row: (
            as_float(row.get("p50_total_latency_s")) or 999999.0,
            -(as_float(row.get("accuracy_pass_rate")) or 0.0),
        ),
    )

    if not passing:
        print(
            "No rows met the requested accuracy threshold; "
            "showing the best latency rows regardless of accuracy."
        )

    fields = [
        "optimization_label",
        "max_tokens",
        "accuracy_pass_rate",
        "target_pass_rate",
        "p50_total_latency_s",
        "p90_total_latency_s",
        "median_output_tokens",
        "p50_decode_tok_s",
        "p50_ttft_s",
    ]
    print("\t".join(fields))
    for row in ranked[: max(1, args.top)]:
        print("\t".join(row.get(field, "") for field in fields))

    if ranked:
        best = ranked[0]
        print()
        print(
            "Best: "
            f"{best.get('optimization_label', '')} "
            f"max_tokens={best.get('max_tokens', '')} "
            f"p50={best.get('p50_total_latency_s', '')}s "
            f"accuracy={best.get('accuracy_pass_rate', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
