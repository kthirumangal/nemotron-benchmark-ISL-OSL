#!/usr/bin/env python3
"""
Run Arco PromptFoo-style benchmark prompts against an OpenAI-compatible endpoint.

The runner captures one long-form CSV across prompt category, model, precision,
GPU hardware, run index, latency, throughput, and assertion-based accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import yaml


DEFAULT_CONFIGS = ("classification.yaml", "reasoning.yaml", "recommender.yaml")
VISIBLE_TEXT_FIELDS = ("content", "text", "output_text")
REASONING_TEXT_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")

JS_EVAL_SCRIPT = r"""
const fs = require("fs");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const output = payload.output;
const expr = payload.expr;
let pass = false;
let error = "";
try {
  pass = !!eval(expr);
} catch (e) {
  error = String((e && e.message) || e);
}
process.stdout.write(JSON.stringify({pass, error}));
"""


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    config_file: str
    prompt_template_file: str
    description: str
    threshold: float
    vars: dict[str, Any]
    assertions: list[dict[str, Any]]
    default_assertions: list[dict[str, Any]]
    messages: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Arco prompt categories and write one combined CSV."
    )
    parser.add_argument("--arco-repo", required=True, help="Path to aem-growth-arco-benchmark")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="PromptFoo YAML configs to run, relative to --arco-repo.",
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", default="", help="Model ID exposed by /v1/models")
    parser.add_argument("--deployment-label", default="", help="Human-readable deployment name")
    parser.add_argument("--precision-label", default="", help="Precision/profile label")
    parser.add_argument("--requested-precision-label", default="", help="Requested precision/profile label from the experiment matrix")
    parser.add_argument("--detected-precision-label", default="", help="Detected precision/profile label from the serving logs")
    parser.add_argument("--gpu-label", default="", help="Override detected GPU label")
    parser.add_argument("--gpu-count", default="", help="Override detected GPU count")
    parser.add_argument("--gpu-memory-total-mb", default="", help="Override detected GPU memory total in MB")
    parser.add_argument("--experiment-id", default="", help="Stable run grouping ID")
    parser.add_argument("--optimization-label", default="", help="Label for prompt/perf variant")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--allow-missing-api-key", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--ttft-target-s", type=float, default=2.0)
    parser.add_argument("--total-latency-target-s", type=float, default=5.0)
    parser.add_argument("--throughput-target-tok-s", type=float, default=200.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--omit-chat-template-kwargs", action="store_true")
    parser.add_argument("--api-reasoning-effort", default="")
    parser.add_argument("--force-visible-output", action="store_true")
    parser.add_argument("--capture-reasoning-as-output", action="store_true")
    parser.add_argument("--system-suffix", default="", help="Extra instruction appended to the system prompt")
    parser.add_argument("--system-suffix-file", default="", help="File containing extra instruction appended to the system prompt")
    parser.add_argument("--extra-body-json", default="")
    parser.add_argument("--measurement-mode", choices=("strict", "lenient"), default="strict")
    parser.add_argument("--stream-debug-dir", default="")
    parser.add_argument("--output", default="", help="Combined CSV output path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV")
    parser.add_argument("--dry-run", action="store_true", help="Load cases but do not call endpoint")
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4.1)) if text else 0


def render_template(text: str, variables: dict[str, Any]) -> str:
    text = text.replace("{% raw %}", "").replace("{% endraw %}", "")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in variables:
            return str(variables.get(key, ""))
        return match.group(0)

    return re.sub(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}", replace, text)


def load_yaml(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prompt_file_from_config(arco_repo: pathlib.Path, config: dict[str, Any]) -> pathlib.Path:
    prompts = config.get("prompts") or []
    if not prompts:
        raise ValueError("Config has no prompts entry")
    raw = str(prompts[0])
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    return arco_repo / raw


def render_messages(
    message_templates: list[dict[str, Any]], variables: dict[str, Any]
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in message_templates:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported prompt role: {role}")
        rendered = render_template(content, variables)
        messages.append({"role": role, "content": rendered})

    while messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages.pop()
    return messages


def load_cases(arco_repo: pathlib.Path, config_files: list[str]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for config_file in config_files:
        config_path = arco_repo / config_file
        config = load_yaml(config_path)
        prompt_path = prompt_file_from_config(arco_repo, config)
        message_templates = load_yaml(prompt_path)
        if not isinstance(message_templates, list):
            raise ValueError(f"Prompt template is not a message list: {prompt_path}")

        category = pathlib.Path(config_file).stem
        default_assertions = (
            config.get("defaultTest", {}).get("assert", [])
            if isinstance(config.get("defaultTest"), dict)
            else []
        )
        for test in config.get("tests", []):
            variables = test.get("vars") or {}
            if not isinstance(variables, dict):
                variables = {}
            threshold = float(test.get("threshold", 1.0))
            assertions = test.get("assert") or []
            cases.append(
                BenchmarkCase(
                    category=category,
                    config_file=config_file,
                    prompt_template_file=str(prompt_path.relative_to(arco_repo)),
                    description=str(test.get("description", "")),
                    threshold=threshold,
                    vars=dict(variables),
                    assertions=list(assertions),
                    default_assertions=list(default_assertions),
                    messages=render_messages(message_templates, dict(variables)),
                )
            )
    return cases


def detect_gpu_metadata(
    gpu_label: str, gpu_count: str, gpu_memory_total_mb: str = ""
) -> dict[str, str]:
    metadata = {
        "gpu_count": gpu_count,
        "gpu_names": gpu_label,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if gpu_label and gpu_count and gpu_memory_total_mb:
        return metadata

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return metadata

    names: list[str] = []
    memories: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        names.append(parts[0])
        memories.append(parts[1] if len(parts) > 1 else "")

    if not gpu_count:
        metadata["gpu_count"] = str(len(names))
    if not gpu_label:
        metadata["gpu_names"] = " | ".join(names)
    if not gpu_memory_total_mb:
        metadata["gpu_memory_total_mb"] = " | ".join(memories)
    return metadata


def apply_force_visible_output(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = (
        "Return the answer in final visible assistant content. Do not leave "
        "message content empty. Do not emit only hidden reasoning tokens."
    )
    updated = [dict(message) for message in messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = instruction + "\n\n" + message["content"]
            return updated
    return [{"role": "system", "content": instruction}, *updated]


def apply_system_suffix(messages: list[dict[str, str]], suffix: str) -> list[dict[str, str]]:
    if not suffix.strip():
        return messages
    instruction = "\n\n## Additional Benchmark Instruction\n" + suffix.strip()
    updated = [dict(message) for message in messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = message["content"].rstrip() + instruction
            return updated
    return [{"role": "system", "content": suffix.strip()}, *updated]


def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_from_value(item) for item in value)
    if isinstance(value, dict):
        return "".join(
            text_from_value(value.get(key))
            for key in ("text", "content", "value", "output_text")
            if key in value
        )
    return ""


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


def extract_stream_text(chunk: dict[str, Any]) -> tuple[str, str]:
    choices = chunk.get("choices") or []
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for choice in choices:
        for container_name in ("delta", "message"):
            container = choice.get(container_name) or {}
            if not isinstance(container, dict):
                continue
            for field in VISIBLE_TEXT_FIELDS:
                text = text_from_value(container.get(field))
                if text:
                    visible_parts.append(text)
            for field in REASONING_TEXT_FIELDS:
                text = text_from_value(container.get(field))
                if text:
                    reasoning_parts.append(text)
        text = text_from_value(choice.get("text"))
        if text:
            visible_parts.append(text)
    return "".join(visible_parts), "".join(reasoning_parts)


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    in_quote = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_quote:
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return text[start:].strip()


def normalize_output_for_accuracy(category: str, output: str) -> str:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", output)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    if category in {"classification", "reasoning"}:
        return extract_first_json_object(cleaned)
    return cleaned


def eval_js_assertion(output: str, expression: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["node", "-e", JS_EVAL_SCRIPT],
            input=json.dumps({"output": output, "expr": expression}),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False, "node_not_available"
    except subprocess.TimeoutExpired:
        return False, "assertion_timeout"

    if proc.returncode != 0:
        return False, proc.stderr.strip()[:300]
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, "invalid_assertion_result"
    return bool(result.get("pass")), str(result.get("error", ""))


def evaluate_accuracy(category: str, output: str, case: BenchmarkCase) -> dict[str, Any]:
    normalized = normalize_output_for_accuracy(category, output)
    assertions = [*case.default_assertions, *case.assertions]
    total_weight = 0.0
    passed_weight = 0.0
    passed = 0
    failed = 0
    unsupported = 0
    failed_descriptions: list[str] = []
    json_valid = False

    for assertion in assertions:
        assertion_type = assertion.get("type", "")
        description = str(assertion.get("description", assertion_type))
        weight = float(assertion.get("weight", 1))
        total_weight += weight
        ok = False
        error = ""

        if assertion_type == "is-json":
            try:
                json.loads(normalized)
                ok = True
                json_valid = True
            except json.JSONDecodeError as exc:
                error = str(exc)
        elif assertion_type == "javascript":
            ok, error = eval_js_assertion(normalized, str(assertion.get("value", "")))
        else:
            unsupported += 1
            error = f"unsupported_assertion_type:{assertion_type}"

        if ok:
            passed += 1
            passed_weight += weight
        else:
            failed += 1
            failed_descriptions.append(
                description if not error else f"{description} ({error})"
            )

    score = passed_weight / total_weight if total_weight else 0.0
    return {
        "accuracy_score": score,
        "accuracy_pass": score >= case.threshold,
        "accuracy_threshold": case.threshold,
        "accuracy_passed_assertions": passed,
        "accuracy_failed_assertions": failed,
        "accuracy_unsupported_assertions": unsupported,
        "accuracy_total_assertions": len(assertions),
        "json_valid": json_valid,
        "accuracy_failed_descriptions": " | ".join(failed_descriptions[:8]),
        "normalized_output_chars": len(normalized),
    }


def write_debug_trace(path: str, records: list[dict[str, Any]]) -> str:
    if not path:
        return ""
    trace_path = pathlib.Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(trace_path)


def call_endpoint(
    *,
    args: argparse.Namespace,
    case: BenchmarkCase,
    run_index: int,
    api_key: str,
    experiment_id: str,
    gpu_metadata: dict[str, str],
) -> dict[str, Any]:
    messages = [dict(message) for message in case.messages]
    if args.system_suffix:
        messages = apply_system_suffix(messages, args.system_suffix)
    if args.force_visible_output:
        messages = apply_force_visible_output(messages)

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.omit_chat_template_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    if args.api_reasoning_effort:
        payload["reasoning_effort"] = args.api_reasoning_effort
    if args.extra_body_json:
        payload.update(json.loads(args.extra_body_json))

    input_text = "\n".join(message["content"] for message in messages)
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
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
    usage: Optional[dict[str, Any]] = None
    ttft_s: Optional[float] = None
    streamed_chunks = 0
    content_chunks = 0
    reasoning_chunks = 0
    start = time.perf_counter()

    base_row = {
        "experiment_id": experiment_id,
        "optimization_label": args.optimization_label,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "deployment_label": args.deployment_label,
        "model": args.model,
        "precision_label": args.precision_label,
        "requested_precision_label": args.requested_precision_label,
        "detected_precision_label": args.detected_precision_label,
        "base_url": args.base_url,
        **gpu_metadata,
        "category": case.category,
        "config_file": case.config_file,
        "prompt_template_file": case.prompt_template_file,
        "prompt_description": case.description,
        "prompt_query": str(case.vars.get("query", "")),
        "run_index": run_index,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "chat_template_kwargs_enabled": not args.omit_chat_template_kwargs,
        "api_reasoning_effort": args.api_reasoning_effort,
        "force_visible_output": args.force_visible_output,
        "measurement_mode": args.measurement_mode,
        "input_chars": len(input_text),
        "estimated_input_tokens": estimate_tokens(input_text),
        "ttft_target_s": args.ttft_target_s,
        "total_latency_target_s": args.total_latency_target_s,
        "throughput_target_tok_s": args.throughput_target_tok_s,
    }

    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            for raw_line in response:
                now = time.perf_counter()
                chunk = parse_sse_line(raw_line)
                if not chunk:
                    continue
                if chunk.get("done"):
                    break
                streamed_chunks += 1
                usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else usage
                visible_text, reasoning_text = extract_stream_text(chunk)
                timing_text = visible_text
                if (
                    args.capture_reasoning_as_output
                    and not timing_text
                    and reasoning_text
                ):
                    timing_text = reasoning_text
                if timing_text:
                    if ttft_s is None:
                        ttft_s = now - start
                    output_parts.append(timing_text)
                if visible_text:
                    content_chunks += 1
                if reasoning_text:
                    reasoning_chunks += 1
                    reasoning_parts.append(reasoning_text)
                if args.stream_debug_dir:
                    debug_records.append(
                        {
                            "t_rel_s": now - start,
                            "visible_chars": len(visible_text),
                            "reasoning_chars": len(reasoning_text),
                            "usage_present": isinstance(chunk.get("usage"), dict),
                            "chunk": chunk,
                        }
                    )
        total_latency_s = time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {
            **base_row,
            **empty_metrics(),
            "status": "error",
            "error": f"HTTP {exc.code}: {body}",
        }
    except Exception as exc:
        return {**base_row, **empty_metrics(), "status": "error", "error": str(exc)}

    output_text = "".join(output_parts)
    reasoning_text = "".join(reasoning_parts)
    if usage and usage.get("completion_tokens") is not None:
        output_tokens = int(usage["completion_tokens"])
        output_tokens_source = "provider_usage"
    else:
        output_tokens = estimate_tokens(output_text)
        output_tokens_source = "estimated_chars"

    decode_tokens_per_s: Optional[float] = None
    if ttft_s is not None and total_latency_s > ttft_s:
        decode_tokens_per_s = output_tokens / (total_latency_s - ttft_s)
    e2e_tokens_per_s = output_tokens / total_latency_s if total_latency_s > 0 else None

    has_output = bool(output_text)
    accuracy = evaluate_accuracy(case.category, output_text, case) if has_output else {
        "accuracy_score": 0.0,
        "accuracy_pass": False,
        "accuracy_threshold": case.threshold,
        "accuracy_passed_assertions": 0,
        "accuracy_failed_assertions": len(case.default_assertions) + len(case.assertions),
        "accuracy_unsupported_assertions": 0,
        "accuracy_total_assertions": len(case.default_assertions) + len(case.assertions),
        "json_valid": False,
        "accuracy_failed_descriptions": "no_visible_output",
        "normalized_output_chars": 0,
    }

    ttft_pass = ttft_s is not None and ttft_s <= args.ttft_target_s
    total_latency_pass = total_latency_s <= args.total_latency_target_s
    throughput_pass = (
        decode_tokens_per_s is not None
        and decode_tokens_per_s >= args.throughput_target_tok_s
    )
    complete_measurement = has_output and ttft_s is not None and decode_tokens_per_s is not None
    measurement_quality = "complete" if complete_measurement else "partial"
    if not has_output:
        measurement_quality = "no_visible_output"

    debug_trace_path = ""
    if args.stream_debug_dir:
        debug_path = (
            pathlib.Path(args.stream_debug_dir)
            / case.category
            / f"{slugify(case.description)}-run{run_index}.jsonl"
        )
        debug_trace_path = write_debug_trace(str(debug_path), debug_records)

    status = "ok"
    error = ""
    if args.measurement_mode == "strict" and not complete_measurement:
        status = "error"
        error = "Incomplete streaming measurement"

    return {
        **base_row,
        "ttft_s": fmt(ttft_s),
        "total_latency_s": fmt(total_latency_s),
        "output_tokens": output_tokens,
        "output_tokens_source": output_tokens_source,
        "decode_tokens_per_s": fmt(decode_tokens_per_s),
        "e2e_tokens_per_s": fmt(e2e_tokens_per_s),
        "ttft_pass": ttft_pass,
        "total_latency_pass": total_latency_pass,
        "throughput_pass": throughput_pass,
        "meets_latency_targets": ttft_pass and total_latency_pass and throughput_pass,
        "meets_all_targets": ttft_pass
        and total_latency_pass
        and throughput_pass
        and bool(accuracy["accuracy_pass"]),
        "output_chars": len(output_text),
        "output_excerpt": compact(output_text, 500),
        "visible_output_captured": content_chunks > 0,
        "ttft_measured": ttft_s is not None,
        "decode_throughput_measured": decode_tokens_per_s is not None,
        "measurement_quality": measurement_quality,
        "streamed_chunks": streamed_chunks,
        "content_chunks": content_chunks,
        "reasoning_chunks": reasoning_chunks,
        "reasoning_chars": len(reasoning_text),
        "debug_trace_path": debug_trace_path,
        **accuracy,
        "status": status,
        "error": error,
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "ttft_s": "",
        "total_latency_s": "",
        "output_tokens": "",
        "output_tokens_source": "",
        "decode_tokens_per_s": "",
        "e2e_tokens_per_s": "",
        "ttft_pass": False,
        "total_latency_pass": False,
        "throughput_pass": False,
        "meets_latency_targets": False,
        "meets_all_targets": False,
        "output_chars": "",
        "output_excerpt": "",
        "visible_output_captured": False,
        "ttft_measured": False,
        "decode_throughput_measured": False,
        "measurement_quality": "",
        "streamed_chunks": "",
        "content_chunks": "",
        "reasoning_chunks": "",
        "reasoning_chars": "",
        "debug_trace_path": "",
        "accuracy_score": "",
        "accuracy_pass": False,
        "accuracy_threshold": "",
        "accuracy_passed_assertions": "",
        "accuracy_failed_assertions": "",
        "accuracy_unsupported_assertions": "",
        "accuracy_total_assertions": "",
        "json_valid": False,
        "accuracy_failed_descriptions": "",
        "normalized_output_chars": "",
    }


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def compact(text: str, limit: int) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    return one_line[:limit]


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "case"


FIELDNAMES = [
    "experiment_id",
    "optimization_label",
    "timestamp_utc",
    "host",
    "deployment_label",
    "model",
    "precision_label",
    "requested_precision_label",
    "detected_precision_label",
    "base_url",
    "gpu_count",
    "gpu_names",
    "gpu_memory_total_mb",
    "cuda_visible_devices",
    "category",
    "config_file",
    "prompt_template_file",
    "prompt_description",
    "prompt_query",
    "run_index",
    "concurrency",
    "max_tokens",
    "temperature",
    "enable_thinking",
    "chat_template_kwargs_enabled",
    "api_reasoning_effort",
    "force_visible_output",
    "measurement_mode",
    "input_chars",
    "estimated_input_tokens",
    "ttft_s",
    "total_latency_s",
    "output_tokens",
    "output_tokens_source",
    "decode_tokens_per_s",
    "e2e_tokens_per_s",
    "ttft_target_s",
    "total_latency_target_s",
    "throughput_target_tok_s",
    "ttft_pass",
    "total_latency_pass",
    "throughput_pass",
    "meets_latency_targets",
    "meets_all_targets",
    "output_chars",
    "output_excerpt",
    "visible_output_captured",
    "ttft_measured",
    "decode_throughput_measured",
    "measurement_quality",
    "streamed_chunks",
    "content_chunks",
    "reasoning_chunks",
    "reasoning_chars",
    "debug_trace_path",
    "accuracy_score",
    "accuracy_pass",
    "accuracy_threshold",
    "accuracy_passed_assertions",
    "accuracy_failed_assertions",
    "accuracy_unsupported_assertions",
    "accuracy_total_assertions",
    "json_valid",
    "accuracy_failed_descriptions",
    "normalized_output_chars",
    "status",
    "error",
]


def write_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> int:
    args = parse_args()
    arco_repo = pathlib.Path(args.arco_repo).expanduser().resolve()
    if not arco_repo.exists():
        print(f"Arco repo not found: {arco_repo}", file=sys.stderr)
        return 2

    cases = load_cases(arco_repo, args.configs)
    if args.dry_run:
        print(f"Loaded {len(cases)} cases from {arco_repo}")
        for category in sorted({case.category for case in cases}):
            count = sum(1 for case in cases if case.category == category)
            print(f"- {category}: {count} cases")
        return 0

    if not args.base_url:
        print("--base-url is required unless --dry-run is set.", file=sys.stderr)
        return 2
    if not args.model:
        print("--model is required unless --dry-run is set.", file=sys.stderr)
        return 2
    if not args.output:
        print("--output is required unless --dry-run is set.", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key and not args.allow_missing_api_key:
        print(
            f"Missing API key. Set {args.api_key_env} or pass --allow-missing-api-key.",
            file=sys.stderr,
        )
        return 2

    if args.extra_body_json:
        try:
            parsed_extra = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            print(f"Invalid --extra-body-json: {exc}", file=sys.stderr)
            return 2
        if not isinstance(parsed_extra, dict):
            print("--extra-body-json must decode to an object.", file=sys.stderr)
            return 2

    if args.system_suffix_file:
        suffix_path = pathlib.Path(args.system_suffix_file).expanduser()
        if not suffix_path.exists():
            print(f"--system-suffix-file not found: {suffix_path}", file=sys.stderr)
            return 2
        file_suffix = suffix_path.read_text(encoding="utf-8")
        args.system_suffix = (
            args.system_suffix.rstrip() + "\n\n" + file_suffix.strip()
            if args.system_suffix.strip()
            else file_suffix.strip()
        )

    experiment_id = args.experiment_id or time.strftime("arco-%Y%m%d-%H%M%S")
    gpu_metadata = detect_gpu_metadata(
        args.gpu_label, args.gpu_count, args.gpu_memory_total_mb
    )
    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not args.append or not output_path.exists() or output_path.stat().st_size == 0

    tasks = [(case, run_index) for run_index in range(args.runs) for case in cases]
    completed = 0
    errors = 0
    accuracy_passes = 0

    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(
                    call_endpoint,
                    args=args,
                    case=case,
                    run_index=run_index,
                    api_key=api_key,
                    experiment_id=experiment_id,
                    gpu_metadata=gpu_metadata,
                )
                for case, run_index in tasks
            ]
            for future in as_completed(futures):
                row = future.result()
                write_row(writer, row)
                handle.flush()
                if row.get("status") == "ok":
                    completed += 1
                else:
                    errors += 1
                if truthy(row.get("accuracy_pass", "")):
                    accuracy_passes += 1
                print(
                    "ok" if row.get("status") == "ok" else "error",
                    row.get("category", ""),
                    row.get("prompt_description", ""),
                    f"run={row.get('run_index', '')}",
                    f"ttft={row.get('ttft_s', '')}",
                    f"total={row.get('total_latency_s', '')}",
                    f"decode={row.get('decode_tokens_per_s', '')}",
                    f"accuracy={row.get('accuracy_score', '')}",
                )

    total = completed + errors
    print()
    print("Summary")
    print("-------")
    print(f"rows:          {total}")
    print(f"completed:     {completed}")
    print(f"errors:        {errors}")
    print(f"accuracy pass: {accuracy_passes}/{total}")
    print(f"wrote:         {output_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
