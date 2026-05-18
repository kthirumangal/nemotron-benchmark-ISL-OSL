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
    api_reasoning_effort: str
    force_visible_output: bool
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
    measurement_mode: str
    visible_output_captured: bool
    ttft_measured: bool
    decode_throughput_measured: bool
    measurement_quality: str
    streamed_chunks: int
    content_chunks: int
    reasoning_chunks: int
    reasoning_chars: int
    debug_trace_path: str
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


def apply_force_visible_output(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = (
        "Return the answer in final visible assistant content. Do not leave "
        "message content empty. Do not emit only reasoning/analysis tokens. "
        "For this benchmark, the first useful user-visible token must be "
        "streamed as normal assistant content."
    )
    updated = [dict(message) for message in messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = instruction + "\n\n" + message["content"]
            return updated
    return [{"role": "system", "content": instruction}, *updated]


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


VISIBLE_TEXT_FIELDS = ("content", "text", "output_text")
REASONING_TEXT_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_from_value(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "value", "output_text"):
            if key in value:
                parts.append(text_from_value(value[key]))
        return "".join(parts)
    return ""


def extract_stream_text(chunk: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    choices = chunk.get("choices") or []
    if not choices:
        return "", "", [], []

    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    visible_fields: list[str] = []
    reasoning_fields: list[str] = []

    for choice in choices:
        for container_name in ("delta", "message"):
            container = choice.get(container_name) or {}
            if not isinstance(container, dict):
                continue
            for field in VISIBLE_TEXT_FIELDS:
                text = text_from_value(container.get(field))
                if text:
                    visible_parts.append(text)
                    visible_fields.append(f"{container_name}.{field}")
            for field in REASONING_TEXT_FIELDS:
                text = text_from_value(container.get(field))
                if text:
                    reasoning_parts.append(text)
                    reasoning_fields.append(f"{container_name}.{field}")

        text = text_from_value(choice.get("text"))
        if text:
            visible_parts.append(text)
            visible_fields.append("choice.text")

    visible_text = "".join(visible_parts)
    reasoning_text = "".join(reasoning_parts)

    return visible_text, reasoning_text, visible_fields, reasoning_fields


def extract_usage(chunk: dict[str, Any]) -> Optional[dict[str, Any]]:
    usage = chunk.get("usage")
    return usage if isinstance(usage, dict) else None


def write_debug_trace(
    *,
    debug_dir: str,
    prompt_file: str,
    run_index: int,
    records: list[dict[str, Any]],
) -> str:
    if not debug_dir:
        return ""

    path = pathlib.Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / f"{pathlib.Path(prompt_file).stem}-run{run_index}.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(trace_path)


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
    api_reasoning_effort: str,
    force_visible_output: bool,
    extra_body_json: str,
    capture_reasoning_as_output: bool,
    stream_debug_dir: str,
    measurement_mode: str,
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
    if api_reasoning_effort:
        payload["reasoning_effort"] = api_reasoning_effort
    if extra_body_json:
        try:
            extra_body = json.loads(extra_body_json)
        except json.JSONDecodeError as exc:
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
                api_reasoning_effort,
                force_visible_output,
                measurement_mode,
                input_chars,
                estimated_input_tokens,
                ttft_target_s,
                total_latency_target_s,
                throughput_target_tok_s,
                f"Invalid --extra-body-json: {exc}",
            )
        if not isinstance(extra_body, dict):
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
                api_reasoning_effort,
                force_visible_output,
                measurement_mode,
                input_chars,
                estimated_input_tokens,
                ttft_target_s,
                total_latency_target_s,
                throughput_target_tok_s,
                "--extra-body-json must decode to a JSON object.",
            )
        payload.update(extra_body)

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
    reasoning_parts: list[str] = []
    debug_records: list[dict[str, Any]] = []
    ttft_s: Optional[float] = None
    usage: Optional[dict[str, Any]] = None
    streamed_chunks = 0
    content_chunks = 0
    reasoning_chunks = 0
    used_reasoning_as_output = False
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

                streamed_chunks += 1
                usage = extract_usage(chunk) or usage
                visible_text, reasoning_text, visible_fields, reasoning_fields = (
                    extract_stream_text(chunk)
                )
                timing_text = visible_text
                timing_source = "visible" if visible_text else ""
                if capture_reasoning_as_output and not visible_text and reasoning_text:
                    timing_text = reasoning_text
                    timing_source = "reasoning"
                    used_reasoning_as_output = True
                if timing_text:
                    if ttft_s is None:
                        ttft_s = now - start
                    output_parts.append(timing_text)
                if visible_text:
                    content_chunks += 1
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    reasoning_chunks += 1
                if stream_debug_dir:
                    debug_records.append(
                        {
                            "t_rel_s": now - start,
                            "visible_chars": len(visible_text),
                            "reasoning_chars": len(reasoning_text),
                            "timing_chars": len(timing_text),
                            "timing_source": timing_source,
                            "visible_fields": sorted(set(visible_fields)),
                            "reasoning_fields": sorted(set(reasoning_fields)),
                            "usage_present": extract_usage(chunk) is not None,
                            "chunk": chunk,
                        }
                    )

        total_latency_s = time.perf_counter() - start
        output_text = "".join(output_parts)
        reasoning_text = "".join(reasoning_parts)
        debug_trace_path = write_debug_trace(
            debug_dir=stream_debug_dir,
            prompt_file=prompt_file,
            run_index=run_index,
            records=debug_records,
        )

        if usage and usage.get("completion_tokens") is not None:
            output_tokens = int(usage["completion_tokens"])
            output_tokens_source = "provider_usage"
        else:
            output_tokens = estimate_tokens(output_text)
            output_tokens_source = "estimated_chars"

        if not output_text:
            is_lenient = measurement_mode == "lenient"
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
                api_reasoning_effort=api_reasoning_effort,
                force_visible_output=force_visible_output,
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
                measurement_mode=measurement_mode,
                visible_output_captured=False,
                ttft_measured=False,
                decode_throughput_measured=False,
                measurement_quality=(
                    "reasoning_only_no_visible_output"
                    if reasoning_text
                    else "usage_only_no_visible_output"
                ),
                streamed_chunks=streamed_chunks,
                content_chunks=content_chunks,
                reasoning_chunks=reasoning_chunks,
                reasoning_chars=len(reasoning_text),
                debug_trace_path=debug_trace_path,
                status="ok" if is_lenient else "error",
                error="" if is_lenient else (
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
            is_lenient = measurement_mode == "lenient"
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
                api_reasoning_effort=api_reasoning_effort,
                force_visible_output=force_visible_output,
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
                measurement_mode=measurement_mode,
                visible_output_captured=content_chunks > 0,
                ttft_measured=ttft_s is not None,
                decode_throughput_measured=False,
                measurement_quality="missing_decode_throughput",
                streamed_chunks=streamed_chunks,
                content_chunks=content_chunks,
                reasoning_chunks=reasoning_chunks,
                reasoning_chars=len(reasoning_text),
                debug_trace_path=debug_trace_path,
                status="ok" if is_lenient else "error",
                error=(
                    ""
                    if is_lenient
                    else "Could not compute decode throughput because TTFT was not captured."
                ),
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
            api_reasoning_effort=api_reasoning_effort,
            force_visible_output=force_visible_output,
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
            measurement_mode=measurement_mode,
            visible_output_captured=content_chunks > 0,
            ttft_measured=True,
            decode_throughput_measured=True,
            measurement_quality=(
                "complete_reasoning_as_output"
                if used_reasoning_as_output
                else "complete"
            ),
            streamed_chunks=streamed_chunks,
            content_chunks=content_chunks,
            reasoning_chunks=reasoning_chunks,
            reasoning_chars=len(reasoning_text),
            debug_trace_path=debug_trace_path,
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
            api_reasoning_effort,
            force_visible_output,
            measurement_mode,
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
            api_reasoning_effort,
            force_visible_output,
            measurement_mode,
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
    api_reasoning_effort: str,
    force_visible_output: bool,
    measurement_mode: str,
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
        api_reasoning_effort=api_reasoning_effort,
        force_visible_output=force_visible_output,
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
        measurement_mode=measurement_mode,
        visible_output_captured=False,
        ttft_measured=False,
        decode_throughput_measured=False,
        measurement_quality="request_error",
        streamed_chunks=0,
        content_chunks=0,
        reasoning_chunks=0,
        reasoning_chars=0,
        debug_trace_path="",
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
            f" n={len(series)}/{len(ok)}"
        )

    output_tokens = [result.output_tokens for result in ok if result.output_tokens is not None]
    if output_tokens:
        print(
            f"{'Output tokens':16}"
            f" median={statistics.median(output_tokens):.0f}"
            f" min={min(output_tokens):.0f}"
            f" max={max(output_tokens):.0f}"
        )

    visible_output = [result for result in ok if result.visible_output_captured]
    ttft_measured = [result for result in ok if result.ttft_measured]
    decode_measured = [result for result in ok if result.decode_throughput_measured]
    complete_measurement = [
        result for result in ok if result.measurement_quality == "complete"
    ]
    print(
        f"{'Measurement':16}"
        f" visible_output={len(visible_output)}/{len(ok)}"
        f" ttft={len(ttft_measured)}/{len(ok)}"
        f" decode={len(decode_measured)}/{len(ok)}"
        f" complete={len(complete_measurement)}/{len(ok)}"
    )
    streamed = [result for result in ok if result.streamed_chunks]
    content = [result for result in ok if result.content_chunks]
    reasoning = [result for result in ok if result.reasoning_chunks]
    if streamed:
        print(
            f"{'Stream fields':16}"
            f" content_chunks={len(content)}/{len(ok)}"
            f" reasoning_chunks={len(reasoning)}/{len(ok)}"
            f" streamed_chunks_total={sum(result.streamed_chunks for result in ok)}"
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

    quality_notes = [
        result
        for result in ok
        if result.measurement_quality not in {"complete", ""}
    ]
    if quality_notes:
        print("\nFirst measurement notes")
        for result in quality_notes[:5]:
            print(
                f"- {result.prompt_file} run {result.run_index}: "
                f"{result.measurement_quality}"
            )


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
    parser.add_argument(
        "--api-reasoning-effort",
        choices=["", "low", "medium", "high"],
        default="",
        help=(
            "Send a request-level reasoning_effort parameter. Use this for "
            "GPT-OSS NIM/vLLM endpoints that support reasoning_effort directly."
        ),
    )
    parser.add_argument(
        "--force-visible-output",
        action="store_true",
        help=(
            "Prepend a system instruction asking the model to stream final visible "
            "assistant content instead of emitting only reasoning/analysis tokens."
        ),
    )
    parser.add_argument(
        "--extra-body-json",
        default="",
        help=(
            "JSON object merged into the chat/completions request body. Use this "
            "for provider-specific GPT-OSS serving parameters."
        ),
    )
    parser.add_argument(
        "--capture-reasoning-as-output",
        action="store_true",
        help=(
            "Diagnostic mode only: if no visible content is present, count streamed "
            "reasoning_content/reasoning text as output for timing. Do not use this "
            "for final visible-output latency comparisons."
        ),
    )
    parser.add_argument(
        "--stream-debug-dir",
        default="",
        help=(
            "Optional directory for per-run JSONL traces of streamed chunks, including "
            "which fields carried visible or reasoning text."
        ),
    )
    parser.add_argument(
        "--measurement-mode",
        choices=["strict", "lenient"],
        default="strict",
        help=(
            "strict marks rows without visible streamed content/TTFT/decode throughput as errors. "
            "lenient keeps completed HTTP responses as ok and records missing metrics as not measured."
        ),
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

    prompts = []
    for path in prompt_paths:
        messages = load_prompt(path)
        messages = apply_system_reasoning_effort(messages, args.system_reasoning_effort)
        if args.force_visible_output:
            messages = apply_force_visible_output(messages)
        prompts.append((path, messages))
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
                api_reasoning_effort=args.api_reasoning_effort,
                force_visible_output=args.force_visible_output,
                extra_body_json=args.extra_body_json,
                capture_reasoning_as_output=args.capture_reasoning_as_output,
                stream_debug_dir=args.stream_debug_dir,
                measurement_mode=args.measurement_mode,
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
