#!/usr/bin/env python3
"""
Combine benchmark result folders into side-by-side summary/detail CSVs.

Use this when you benchmark one endpoint at a time on a single-GPU Brev
instance. Each benchmark run creates results/precision-matrix-*/summary.csv.
This script scans those folders and writes:

- results/combined-summary.csv
- results/combined-details.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_detail_path(repo_root: pathlib.Path, summary_csv: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    if path.is_absolute():
        return path

    candidates = [
        repo_root / path,
        summary_csv.parent / path,
        summary_csv.parent / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def scan_results(repo_root: pathlib.Path, results_dir: pathlib.Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []

    for summary_csv in sorted(results_dir.glob("precision-matrix-*/summary.csv")):
        run_id = summary_csv.parent.name
        for row in read_csv(summary_csv):
            enriched = {
                "result_run": run_id,
                "summary_csv": str(summary_csv),
                **row,
            }
            summary_rows.append(enriched)

            detail_ref = row.get("output_csv", "")
            if not detail_ref:
                continue

            detail_path = resolve_detail_path(repo_root, summary_csv, detail_ref)
            for detail in read_csv(detail_path):
                detail_rows.append(
                    {
                        "result_run": run_id,
                        "summary_csv": str(summary_csv),
                        "detail_csv": str(detail_path),
                        **detail,
                    }
                )

    return summary_rows, detail_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine benchmark result folders.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary-output", default="results/combined-summary.csv")
    parser.add_argument("--details-output", default="results/combined-details.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path.cwd()
    results_dir = pathlib.Path(args.results_dir)
    summary_rows, detail_rows = scan_results(repo_root, results_dir)

    write_csv(pathlib.Path(args.summary_output), summary_rows)
    write_csv(pathlib.Path(args.details_output), detail_rows)

    print(f"Wrote {args.summary_output} ({len(summary_rows)} rows)")
    print(f"Wrote {args.details_output} ({len(detail_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
