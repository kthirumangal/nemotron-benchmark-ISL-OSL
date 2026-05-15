#!/usr/bin/env python3
"""
Benchmark NVIDIA Nemotron Nano against the Audience of One prompt set.

Measures client-observed:
- TTFT: time from request start to first streamed content token/chunk
- Total latency: time from request start to stream completion
- Output tokens: provider usage if available, otherwise estimated from text
- Decode throughput: output tokens per second after TTFT

The script uses NVIDIA's OpenAI-compatible NIM endpoint by default:
https://integrate.api.nvidia.com/v1/chat/completions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


@dataclass
class RunResult:
    prompt_file: str
    run_index: int
    concurrency: int
    model: str
    max_tokens: int
    enable_thinking: bool
    input_chars: int
    estimated_input_tokens: int
    ttft_s: Optional[float]
    total_latency_s: Optional[float]
    output_tokens: Optional[int]
    output_tokens_source: str
    decode_tokens_per_s: Optional[float]
    e2e_tokens_per_s: Optional[float]
    output_chars: int
    status: str
    error: str


def load_prompt(path: pathlib.Path) -> list[dict[str, str]]:
    messages = json.loads(path.read_text())
    if not isinstance(messages, list):
        raise ValueError(f"{path} is not a chat-message array")

    clean_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{path} has unsupported role: {role}")
        clean_messages.append({"role": role, "content": content})

    # The repo includes a final empty assistant message as an output placeholder.
    # Do not send it to the model; the API will create the assistant turn.
    while clean_messages and clean_messages[-1]["role"] == "assistant" and not clean_messages[-1]["content"]:
        clean_messages.pop()

    return clean_messages


def estimate_tokens(text: str) -> int:
    # Conservative-enough planning estimate for Nemotron-style tokenizers.
    # Exact counts require the model tokenizer; this avoids adding dependencies.
    return max(1, round(len(text) / 4.1))


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


def parse_sse_line(line: bytes) -> Optional[dict[str, Any]]:
    decoded = line.decode("utf-8", errors="replace").strip()
    if not decoded or not decoded.startswith("data:"):
        return None

    data = decoded[len("data:") :].strip()
    if data == "[DONE]":
        return {"done": True}

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def extract_usage(chunk: dict[str, Any]) -> Optional[dict[str, Any]]:
    usage = chunk.get("usage")
    return usage if isinstance(usage, dict) else None


def call_streaming_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    prompt_file: str,
    run_index: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout_s: int,
) -> RunResult:
    input_text = "\n".join(message["content"] for message in messages)
    input_chars = len(input_text)
    estimated_input_tokens = estimate_tokens(input_text)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    output_parts: list[str] = []
    ttft_s: Optional[float] = None
    usage: Optional[dict[str, Any]] = None
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                now = time.perf_counter()
                chunk = parse_sse_line(raw_line)
                if not chunk:
                    continue
                if chunk.get("done"):
                    break

                usage = extract_usage(chunk) or usage
                text = extract_delta_text(chunk)
                if text:
                    if ttft_s is None:
                        ttft_s = now - start
                    output_parts.append(text)

        total_latency_s = time.perf_counter() - start
        output_text = "".join(output_parts)

        if usage and usage.get("completion_tokens") is not None:
            output_tokens = int(usage["completion_tokens"])
            output_tokens_source = "provider_usage"
        else:
            output_tokens = estimate_tokens(output_text)
            output_tokens_source = "estimated_chars"

        if ttft_s is not None and total_latency_s > ttft_s:
            decode_tokens_per_s = output_tokens / (total_latency_s - ttft_s)
        else:
            decode_tokens_per_s = None

        e2e_tokens_per_s = output_tokens / total_latency_s if total_latency_s > 0 else None

        return RunResult(
            prompt_file=prompt_file,
            run_index=run_index,
            concurrency=concurrency,
            model=model,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            input_chars=input_chars,
            estimated_input_tokens=estimated_input_tokens,
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
            output_tokens=output_tokens,
            output_tokens_source=output_tokens_source,
            decode_tokens_per_s=decode_tokens_per_s,
            e2e_tokens_per_s=e2e_tokens_per_s,
            output_chars=len(output_text),
            status="ok",
            error="",
        )

    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return error_result(
            prompt_file,
            run_index,
            concurrency,
            model,
            max_tokens,
            enable_thinking,
            input_chars,
            estimated_input_tokens,
            f"HTTP {exc.code}: {body[:500]}",
        )
    except Exception as exc:
        return error_result(
            prompt_file,
            run_index,
            concurrency,
            model,
            max_tokens,
            enable_thinking,
            input_chars,
            estimated_input_tokens,
            repr(exc),
        )


def error_result(
    prompt_file: str,
    run_index: int,
    concurrency: int,
    model: str,
    max_tokens: int,
    enable_thinking: bool,
    input_chars: int,
    estimated_input_tokens: int,
    error: str,
) -> RunResult:
    return RunResult(
        prompt_file=prompt_file,
        run_index=run_index,
        concurrency=concurrency,
        model=model,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        input_chars=input_chars,
        estimated_input_tokens=estimated_input_tokens,
        ttft_s=None,
        total_latency_s=None,
        output_tokens=None,
        output_tokens_source="none",
        decode_tokens_per_s=None,
        e2e_tokens_per_s=None,
        output_chars=0,
        status="error",
        error=error,
    )


def write_csv(path: pathlib.Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RunResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def print_summary(results: list[RunResult]) -> None:
    ok = [result for result in results if result.status == "ok"]
    errors = [result for result in results if result.status != "ok"]

    print("\nSummary")
    print("-------")
    print(f"completed: {len(ok)}")
    print(f"errors:    {len(errors)}")

    if not ok:
        return

    def values(name: str) -> list[float]:
        return [
            value
            for result in ok
            if (value := getattr(result, name)) is not None
        ]

    for label, field in [
        ("TTFT s", "ttft_s"),
        ("Total latency s", "total_latency_s"),
        ("Decode tok/s", "decode_tokens_per_s"),
        ("E2E tok/s", "e2e_tokens_per_s"),
    ]:
        series = values(field)
        if not series:
            continue
        print(
            f"{label:16}"
            f" p50={statistics.median(series):.3f}"
            f" p90={percentile(series, 0.90):.3f}"
            f" p99={percentile(series, 0.99):.3f}"
            f" min={min(series):.3f}"
            f" max={max(series):.3f}"
        )

    output_tokens = [result.output_tokens for result in ok if result.output_tokens is not None]
    if output_tokens:
        print(
            f"{'Output tokens':16}"
            f" median={statistics.median(output_tokens):.0f}"
            f" min={min(output_tokens):.0f}"
            f" max={max(output_tokens):.0f}"
        )

    if errors:
        print("\nFirst errors")
        for result in errors[:5]:
            print(f"- {result.prompt_file} run {result.run_index}: {result.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Nemotron 3 Nano 30B-A3B with Ao1 prompts.")
    parser.add_argument("--prompt-dir", default="of1-testprompts")
    parser.add_argument("--base-url", default=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Nemotron 3 Nano to emit reasoning traces. Defaults off for structured website output benchmarks.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing API key. Set {args.api_key_env}=<your NVIDIA API key>.", file=sys.stderr)
        return 2

    prompt_dir = pathlib.Path(args.prompt_dir)
    prompt_paths = sorted(prompt_dir.glob("*.json"))
    if not prompt_paths:
        print(f"No JSON prompts found in {prompt_dir}", file=sys.stderr)
        return 2

    prompts = [(path, load_prompt(path)) for path in prompt_paths]
    tasks = []
    for run_index in range(args.runs):
        for path, messages in prompts:
            tasks.append((path, messages, run_index))

    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                call_streaming_chat,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                prompt_file=path.name,
                run_index=run_index,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                timeout_s=args.timeout_s,
            )
            for path, messages, run_index in tasks
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.status == "ok":
                ttft = f"{result.ttft_s:.3f}s" if result.ttft_s is not None else "n/a"
                total = (
                    f"{result.total_latency_s:.3f}s"
                    if result.total_latency_s is not None
                    else "n/a"
                )
                tok_s = (
                    f"{result.decode_tokens_per_s:.1f}"
                    if result.decode_tokens_per_s is not None
                    else "n/a"
                )
                print(
                    f"ok {result.prompt_file} run={result.run_index} "
                    f"ttft={ttft} total={total} decode_tok_s={tok_s}"
                )
            else:
                print(
                    f"error {result.prompt_file} run={result.run_index}: "
                    f"{result.error[:160]}",
                    file=sys.stderr,
                )

    results.sort(key=lambda item: (item.prompt_file, item.run_index))

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = pathlib.Path("results") / f"nano-{stamp}.csv"

    write_csv(output_path, results)
    print_summary(results)
    print(f"\nWrote CSV: {output_path}")
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
