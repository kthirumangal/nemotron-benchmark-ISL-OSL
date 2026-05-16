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
    precision_label: str
    max_tokens: int
    enable_thinking: bool
    chat_template_kwargs_enabled: bool
    system_reasoning_effort: str
    input_chars: int
    estimated_input_tokens: int
    ttft_s: Optional[float]
    total_latency_s: Optional[float]
    output_tokens: Optional[int]
    output_tokens_source: str
    decode_tokens_per_s: Optional[float]
    e2e_tokens_per_s: Optional[float]
    ttft_target_s: float
    total_latency_target_s: float
    throughput_target_tok_s: float
    ttft_pass: bool
    total_latency_pass: bool
    throughput_pass: bool
    meets_targets: bool
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


def apply_system_reasoning_effort(
    messages: list[dict[str, str]], effort: str
) -> list[dict[str, str]]:
    effort = effort.strip().lower()
    if not effort:
        return messages

    prefix = f"Reasoning: {effort}\n\n"
    updated = [dict(message) for message in messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = prefix + message["content"]
            return updated

    return [{"role": "system", "content": prefix.strip()}, *updated]


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
    precision_label: str,
    messages: list[dict[str, str]],
    prompt_file: str,
    run_index: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    omit_chat_template_kwargs: bool,
    system_reasoning_effort: str,
    ttft_target_s: float,
    total_latency_target_s: float,
    throughput_target_tok_s: float,
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
    }
    if not omit_chat_template_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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

        if not output_text:
            return RunResult(
                prompt_file=prompt_file,
                run_index=run_index,
                concurrency=concurrency,
                model=model,
                precision_label=precision_label,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                chat_template_kwargs_enabled=not omit_chat_template_kwargs,
                system_reasoning_effort=system_reasoning_effort,
                input_chars=input_chars,
                estimated_input_tokens=estimated_input_tokens,
                ttft_s=None,
                total_latency_s=total_latency_s,
                output_tokens=output_tokens,
                output_tokens_source=output_tokens_source,
                decode_tokens_per_s=None,
                e2e_tokens_per_s=(
                    output_tokens / total_latency_s if total_latency_s > 0 else None
                ),
                ttft_target_s=ttft_target_s,
                total_latency_target_s=total_latency_target_s,
                throughput_target_tok_s=throughput_target_tok_s,
                ttft_pass=False,
                total_latency_pass=total_latency_s <= total_latency_target_s,
                throughput_pass=False,
                meets_targets=False,
                output_chars=0,
                status="error",
                error=(
                    "No streamed visible content captured. The endpoint may have "
                    "emitted only hidden/reasoning tokens, non-content SSE deltas, "
                    "or an empty completion."
                ),
            )

        if ttft_s is not None and total_latency_s > ttft_s:
            decode_tokens_per_s = output_tokens / (total_latency_s - ttft_s)
        else:
            decode_tokens_per_s = None

        if decode_tokens_per_s is None:
            return RunResult(
                prompt_file=prompt_file,
                run_index=run_index,
                concurrency=concurrency,
                model=model,
                precision_label=precision_label,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                chat_template_kwargs_enabled=not omit_chat_template_kwargs,
                system_reasoning_effort=system_reasoning_effort,
                input_chars=input_chars,
                estimated_input_tokens=estimated_input_tokens,
                ttft_s=ttft_s,
                total_latency_s=total_latency_s,
                output_tokens=output_tokens,
                output_tokens_source=output_tokens_source,
                decode_tokens_per_s=None,
                e2e_tokens_per_s=(
                    output_tokens / total_latency_s if total_latency_s > 0 else None
                ),
                ttft_target_s=ttft_target_s,
                total_latency_target_s=total_latency_target_s,
                throughput_target_tok_s=throughput_target_tok_s,
                ttft_pass=False,
                total_latency_pass=total_latency_s <= total_latency_target_s,
                throughput_pass=False,
                meets_targets=False,
                output_chars=len(output_text),
                status="error",
                error="Could not compute decode throughput because TTFT was not captured.",
            )

        e2e_tokens_per_s = output_tokens / total_latency_s if total_latency_s > 0 else None
        ttft_pass = ttft_s is not None and ttft_s <= ttft_target_s
        total_latency_pass = total_latency_s <= total_latency_target_s
        throughput_pass = (
            decode_tokens_per_s is not None
            and decode_tokens_per_s >= throughput_target_tok_s
        )

        return RunResult(
            prompt_file=prompt_file,
            run_index=run_index,
            concurrency=concurrency,
            model=model,
            precision_label=precision_label,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            chat_template_kwargs_enabled=not omit_chat_template_kwargs,
            system_reasoning_effort=system_reasoning_effort,
            input_chars=input_chars,
            estimated_input_tokens=estimated_input_tokens,
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
            output_tokens=output_tokens,
            output_tokens_source=output_tokens_source,
            decode_tokens_per_s=decode_tokens_per_s,
            e2e_tokens_per_s=e2e_tokens_per_s,
            ttft_target_s=ttft_target_s,
            total_latency_target_s=total_latency_target_s,
            throughput_target_tok_s=throughput_target_tok_s,
            ttft_pass=ttft_pass,
            total_latency_pass=total_latency_pass,
            throughput_pass=throughput_pass,
            meets_targets=ttft_pass and total_latency_pass and throughput_pass,
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
            precision_label,
            max_tokens,
            enable_thinking,
            not omit_chat_template_kwargs,
            system_reasoning_effort,
            input_chars,
            estimated_input_tokens,
            ttft_target_s,
            total_latency_target_s,
            throughput_target_tok_s,
            f"HTTP {exc.code}: {body[:500]}",
        )
    except Exception as exc:
        return error_result(
            prompt_file,
            run_index,
            concurrency,
            model,
            precision_label,
            max_tokens,
            enable_thinking,
            not omit_chat_template_kwargs,
            system_reasoning_effort,
            input_chars,
            estimated_input_tokens,
            ttft_target_s,
            total_latency_target_s,
            throughput_target_tok_s,
            repr(exc),
        )


def error_result(
    prompt_file: str,
    run_index: int,
    concurrency: int,
    model: str,
    precision_label: str,
    max_tokens: int,
    enable_thinking: bool,
    chat_template_kwargs_enabled: bool,
    system_reasoning_effort: str,
    input_chars: int,
    estimated_input_tokens: int,
    ttft_target_s: float,
    total_latency_target_s: float,
    throughput_target_tok_s: float,
    error: str,
) -> RunResult:
    return RunResult(
        prompt_file=prompt_file,
        run_index=run_index,
        concurrency=concurrency,
        model=model,
        precision_label=precision_label,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        chat_template_kwargs_enabled=chat_template_kwargs_enabled,
        system_reasoning_effort=system_reasoning_effort,
        input_chars=input_chars,
        estimated_input_tokens=estimated_input_tokens,
        ttft_s=None,
        total_latency_s=None,
        output_tokens=None,
        output_tokens_source="none",
        decode_tokens_per_s=None,
        e2e_tokens_per_s=None,
        ttft_target_s=ttft_target_s,
        total_latency_target_s=total_latency_target_s,
        throughput_target_tok_s=throughput_target_tok_s,
        ttft_pass=False,
        total_latency_pass=False,
        throughput_pass=False,
        meets_targets=False,
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

    meets_targets = [result for result in ok if result.meets_targets]
    print(
        f"{'Target pass':16}"
        f" {len(meets_targets)}/{len(ok)} runs"
        f" (TTFT <= {ok[0].ttft_target_s:.3f}s,"
        f" total <= {ok[0].total_latency_target_s:.3f}s,"
        f" decode >= {ok[0].throughput_target_tok_s:.1f} tok/s)"
    )

    if errors:
        print("\nFirst errors")
        for result in errors[:5]:
            print(f"- {result.prompt_file} run {result.run_index}: {result.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible chat model with Ao1 prompts.")
    parser.add_argument("--prompt-dir", default="of1-testprompts")
    parser.add_argument("--base-url", default=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--precision-label", default=os.getenv("NVIDIA_PRECISION_LABEL", "hosted-managed"))
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Allow requests without an Authorization header for local/self-hosted OpenAI-compatible endpoints.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Nemotron 3 Nano to emit reasoning traces. Defaults off for structured website output benchmarks.",
    )
    parser.add_argument(
        "--omit-chat-template-kwargs",
        action="store_true",
        help="Do not send chat_template_kwargs. Use this for runtimes/providers that reject vLLM/Nemotron-specific extra fields.",
    )
    parser.add_argument(
        "--system-reasoning-effort",
        choices=["", "low", "medium", "high"],
        default="",
        help="Optionally prepend 'Reasoning: <effort>' to the system prompt, useful for GPT-OSS comparisons.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--ttft-target-s", type=float, default=2.0)
    parser.add_argument("--total-latency-target-s", type=float, default=5.0)
    parser.add_argument("--throughput-target-tok-s", type=float, default=200.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not args.allow_missing_api_key:
        print(f"Missing API key. Set {args.api_key_env}=<your NVIDIA API key>.", file=sys.stderr)
        return 2

    prompt_dir = pathlib.Path(args.prompt_dir)
    prompt_paths = sorted(prompt_dir.glob("*.json"))
    if not prompt_paths:
        print(f"No JSON prompts found in {prompt_dir}", file=sys.stderr)
        return 2

    prompts = [
        (path, apply_system_reasoning_effort(load_prompt(path), args.system_reasoning_effort))
        for path in prompt_paths
    ]
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
                precision_label=args.precision_label,
                messages=messages,
                prompt_file=path.name,
                run_index=run_index,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                omit_chat_template_kwargs=args.omit_chat_template_kwargs,
                system_reasoning_effort=args.system_reasoning_effort,
                ttft_target_s=args.ttft_target_s,
                total_latency_target_s=args.total_latency_target_s,
                throughput_target_tok_s=args.throughput_target_tok_s,
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
