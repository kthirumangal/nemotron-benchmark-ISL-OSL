#!/usr/bin/env python3
"""
Sequentially run Arco benchmarks across NIM model containers.

For each enabled model profile this script:
1. pulls the NIM image
2. starts a clean container
3. waits for readiness
4. discovers the served model ID from /v1/models when requested
5. runs benchmark_arco.py and appends to one CSV
6. stops/removes the container before moving to the next profile
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import benchmark_arco


@dataclass(frozen=True)
class ModelProfile:
    enabled: bool
    label: str
    image: str
    precision_label: str
    served_model_id: str
    port: int
    tensor_parallel_size: str
    passthrough_args: str
    cache_dir: str
    extra_env: dict[str, str]
    benchmark_extra_args: str


@dataclass(frozen=True)
class NimModelProfile:
    profile_id: str
    description: str
    section: str
    required_gb_per_gpu: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull/run NIM containers one at a time and benchmark Arco prompts."
    )
    parser.add_argument("--matrix", default="arco_nim_models.example.csv")
    parser.add_argument("--arco-repo", required=True)
    parser.add_argument("--output", default="results/arco-all-experiments.csv")
    parser.add_argument("--summary-output", default="results/arco-by-prompt-summary.csv")
    parser.add_argument("--category-summary-output", default="results/arco-category-summary.csv")
    parser.add_argument("--model-summary-output", default="results/arco-model-summary.csv")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--ngc-api-key-env", default="NGC_API_KEY")
    parser.add_argument("--ngc-registry", default="nvcr.io")
    parser.add_argument("--skip-docker-login", action="store_true")
    parser.add_argument("--benchmark-api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--container-prefix", default="arco-bench")
    parser.add_argument("--start-port", type=int, default=8001)
    parser.add_argument("--startup-timeout-s", type=int, default=1800)
    parser.add_argument("--health-interval-s", type=int, default=15)
    parser.add_argument("--docker-shm-size", default="64GB")
    parser.add_argument("--gpu-count", default="auto")
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--ttft-target-s", type=float, default=2.0)
    parser.add_argument("--total-latency-target-s", type=float, default=5.0)
    parser.add_argument("--throughput-target-tok-s", type=float, default=200.0)
    parser.add_argument("--configs", nargs="+", default=list(benchmark_arco.DEFAULT_CONFIGS))
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep Docker images after each profile. Default removes the just-used image.",
    )
    parser.add_argument(
        "--keep-model-cache",
        action="store_true",
        help="Keep per-profile NIM cache after each profile. Default removes it.",
    )
    parser.add_argument(
        "--docker-prune-after-profile",
        action="store_true",
        help="Also run Docker prune commands after each profile.",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--auto-resolve-nim-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "List NIM model profiles from the pulled image and select the exact "
            "profile ID matching precision_label and tensor_parallel_size."
        ),
    )
    parser.add_argument("--resolve-amd64-digest", action="store_true", default=True)
    parser.add_argument(
        "--no-resolve-amd64-digest",
        action="store_false",
        dest="resolve_amd64_digest",
        help="Do not retry pulls through the linux/amd64 manifest digest.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_float(value: Any) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def fmt_summary(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "profile"


def run_cmd(
    cmd: list[str],
    *,
    check: bool = True,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in redact_command(cmd)))
    return subprocess.run(
        cmd,
        check=check,
        timeout=timeout,
        capture_output=capture,
        text=True,
    )


def redact_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in cmd:
        if redact_next:
            if part.startswith("NGC_API_KEY="):
                redacted.append("NGC_API_KEY=<redacted>")
            elif part.startswith("NVIDIA_API_KEY="):
                redacted.append("NVIDIA_API_KEY=<redacted>")
            else:
                redacted.append(part)
            redact_next = False
            continue
        redacted.append(part)
        if part in {"-e", "--env"}:
            redact_next = True
    return redacted


def docker_available() -> bool:
    return shutil.which("docker") is not None


def docker_login(registry: str, api_key: str) -> None:
    print(f"+ docker login {registry} -u '$oauthtoken' --password-stdin")
    proc = subprocess.run(
        ["docker", "login", registry, "-u", "$oauthtoken", "--password-stdin"],
        input=api_key + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Docker login failed for {registry}: {stderr}")
    print(f"Docker login succeeded for {registry}.")


def load_profiles(matrix_path: pathlib.Path, start_port: int) -> list[ModelProfile]:
    profiles: list[ModelProfile] = []
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            label = (row.get("label") or "").strip()
            image = (row.get("image") or "").strip()
            if not label or not image or label.startswith("#"):
                continue
            port_raw = (row.get("port") or "").strip()
            port = int(port_raw) if port_raw else start_port + index
            extra_env_raw = (row.get("extra_env_json") or "").strip()
            extra_env: dict[str, str] = {}
            if extra_env_raw:
                parsed = json.loads(extra_env_raw)
                if not isinstance(parsed, dict):
                    raise ValueError(f"extra_env_json must be an object for {label}")
                extra_env = {str(key): str(value) for key, value in parsed.items()}
            profiles.append(
                ModelProfile(
                    enabled=truthy(row.get("enabled", "true")),
                    label=label,
                    image=image,
                    precision_label=(row.get("precision_label") or "auto").strip(),
                    served_model_id=(row.get("served_model_id") or "auto").strip(),
                    port=port,
                    tensor_parallel_size=(row.get("tensor_parallel_size") or "auto").strip(),
                    passthrough_args=(row.get("passthrough_args") or "").strip(),
                    cache_dir=(row.get("cache_dir") or "").strip(),
                    extra_env=extra_env,
                    benchmark_extra_args=(row.get("benchmark_extra_args") or "").strip(),
                )
            )
    return profiles


def detect_gpu_count(override: str) -> str:
    if override and override != "auto":
        return override
    try:
        proc = run_cmd(["nvidia-smi", "-L"], check=True, timeout=10)
    except Exception:
        return "1"
    count = len([line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")])
    return str(count or 1)


def parse_positive_int(value: str) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def default_cache_root() -> pathlib.Path:
    nvme = pathlib.Path("/opt/dlami/nvme")
    if nvme.exists() and os.access(nvme, os.W_OK):
        return nvme / "nim-cache"
    return pathlib.Path.home() / "nim-cache"


def image_repo_without_tag(image: str) -> str:
    if "@" in image:
        return image.split("@", 1)[0]
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return image[:last_colon]
    return image


def resolve_linux_amd64_digest(image: str) -> Optional[str]:
    try:
        proc = run_cmd(
            ["docker", "buildx", "imagetools", "inspect", "--raw", image],
            check=True,
            timeout=120,
        )
    except Exception:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for manifest in payload.get("manifests", []):
        platform = manifest.get("platform") or {}
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            digest = manifest.get("digest")
            if digest:
                return f"{image_repo_without_tag(image)}@{digest}"
    return None


def pull_image(image: str, resolve_amd64_digest: bool) -> str:
    if resolve_amd64_digest:
        resolved = resolve_linux_amd64_digest(image)
        if resolved:
            if resolved != image:
                print(f"Resolved linux/amd64 image: {resolved}")
            run_cmd(["docker", "pull", resolved], check=True, timeout=3600, capture=False)
            return resolved

    try:
        run_cmd(["docker", "pull", image], check=True, timeout=3600, capture=False)
        return image
    except subprocess.CalledProcessError as exc:
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        if "Incorrect Repository Format" not in combined or not resolve_amd64_digest:
            raise
        print("Docker pull hit OCI index selection issue; resolving linux/amd64 digest.")
        resolved = resolve_linux_amd64_digest(image)
        if not resolved:
            raise
        run_cmd(["docker", "pull", resolved], check=True, timeout=3600, capture=False)
        return resolved


def list_nim_model_profiles(
    *,
    image_ref: str,
    ngc_api_key: str,
    gpus: str,
) -> list[NimModelProfile]:
    commands = [
        [
            "docker",
            "run",
            "--rm",
            "--runtime=nvidia",
            "--gpus",
            gpus,
            "-e",
            f"NGC_API_KEY={ngc_api_key}",
            image_ref,
            "list-model-profiles",
        ],
        [
            "docker",
            "run",
            "--rm",
            "--runtime=nvidia",
            "--gpus",
            gpus,
            "-e",
            f"NGC_API_KEY={ngc_api_key}",
            "--entrypoint",
            "nim_list_model_profiles",
            image_ref,
        ],
    ]
    last_error = ""
    for cmd in commands:
        try:
            proc = run_cmd(cmd, check=True, timeout=900, capture=True)
        except Exception as exc:
            last_error = str(exc)
            continue
        profiles = parse_nim_profile_output(proc.stdout)
        if profiles:
            return profiles
        last_error = "profile list command produced no parsable profiles"
    raise RuntimeError(f"Could not list NIM model profiles for {image_ref}: {last_error}")


def parse_nim_profile_output(output: str) -> list[NimModelProfile]:
    profiles: list[NimModelProfile] = []
    section = ""
    profile_re = re.compile(
        r"^\s*-\s+([a-fA-F0-9]{64})\s+\(([^)]+)\)\s+\[requires\s+>=\s*([0-9.]+)\s+GB/gpu\]"
    )
    for line in output.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped.startswith("- Compatible with system and runnable"):
            section = "runnable"
            continue
        if stripped.startswith("- Compatible with system but low memory"):
            section = "low_memory"
            continue
        if stripped.startswith("- Incompatible with system"):
            section = "incompatible"
            continue
        match = profile_re.match(line)
        if not match:
            continue
        profile_id, description, required = match.groups()
        profiles.append(
            NimModelProfile(
                profile_id=profile_id,
                description=description,
                section=section or "unknown",
                required_gb_per_gpu=required,
            )
        )
    return profiles


def profile_description_matches_request(
    description: str,
    *,
    precision_label: str,
    tensor_parallel_size: str,
) -> bool:
    precision = precision_label.strip().lower()
    tp = parse_positive_int(tensor_parallel_size)
    if not precision or precision == "auto" or not tp:
        return False
    prefix = f"vllm-{precision}-tp{tp}-pp1"
    return description == prefix or description.startswith(prefix + "-")


def find_nim_profile(
    profiles: list[NimModelProfile],
    *,
    requested_profile: str,
    precision_label: str,
    tensor_parallel_size: str,
) -> Optional[NimModelProfile]:
    non_lora = [
        profile
        for profile in profiles
        if "feat_lora" not in profile.description.lower()
    ]
    preferred_sections = ("runnable", "low_memory", "incompatible", "unknown")

    def sorted_candidates(candidates: list[NimModelProfile]) -> list[NimModelProfile]:
        return sorted(
            candidates,
            key=lambda profile: preferred_sections.index(profile.section)
            if profile.section in preferred_sections
            else len(preferred_sections),
        )

    requested = requested_profile.strip()
    if requested and requested.lower() not in {"auto", "default"}:
        exact = [
            profile
            for profile in non_lora
            if profile.profile_id == requested or profile.description == requested
        ]
        if exact:
            return sorted_candidates(exact)[0]
        prefix = [
            profile
            for profile in non_lora
            if profile.description.startswith(requested + "-")
        ]
        if prefix:
            return sorted_candidates(prefix)[0]

    precision_tp = [
        profile
        for profile in non_lora
        if profile_description_matches_request(
            profile.description,
            precision_label=precision_label,
            tensor_parallel_size=tensor_parallel_size,
        )
    ]
    if precision_tp:
        return sorted_candidates(precision_tp)[0]
    return None


def resolve_nim_model_profile(
    *,
    args: argparse.Namespace,
    profile: ModelProfile,
    image_ref: str,
    ngc_api_key: str,
    tensor_parallel_size: str,
) -> tuple[dict[str, str], Optional[str]]:
    env = dict(profile.extra_env)
    requested_profile = env.get("NIM_MODEL_PROFILE", "").strip()
    if not args.auto_resolve_nim_profile:
        return env, None
    if profile.precision_label.strip().lower() in {"", "auto"} and not requested_profile:
        return env, None

    print("Resolving NIM model profile from image manifest...")
    profiles = list_nim_model_profiles(
        image_ref=image_ref,
        ngc_api_key=ngc_api_key,
        gpus=args.gpus,
    )
    resolved = find_nim_profile(
        profiles,
        requested_profile=requested_profile,
        precision_label=profile.precision_label,
        tensor_parallel_size=tensor_parallel_size,
    )
    if not resolved:
        available = ", ".join(
            f"{p.description} [{p.section}]" for p in profiles if "feat_lora" not in p.description.lower()
        )
        error = (
            f"No NIM profile matched precision={profile.precision_label}, "
            f"tensor_parallel_size={tensor_parallel_size}. Available profiles: {available}"
        )
        return env, error
    if resolved.section != "runnable":
        error = (
            f"Resolved NIM profile {resolved.description} is marked {resolved.section} "
            f"on this hardware, so it will not be launched."
        )
        return env, error
    env["NIM_MODEL_PROFILE"] = resolved.profile_id
    print(
        "Resolved NIM profile: "
        f"{resolved.profile_id} ({resolved.description}) "
        f"[requires >={resolved.required_gb_per_gpu} GB/gpu]"
    )
    return env, None


def docker_rm(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"Timed out removing container: {container_name}")


def docker_image_rm(image_ref: str) -> None:
    if not image_ref:
        return
    try:
        subprocess.run(
            ["docker", "image", "rm", "-f", image_ref],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"Timed out removing Docker image: {image_ref}")


def docker_prune() -> None:
    for cmd in (
        ["docker", "container", "prune", "-f"],
        ["docker", "builder", "prune", "-f"],
    ):
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("Timed out during Docker prune.")


def safe_rmtree(path: pathlib.Path, cache_root: pathlib.Path) -> None:
    try:
        resolved = path.resolve()
        resolved_root = cache_root.resolve()
    except FileNotFoundError:
        return

    unsafe = {
        pathlib.Path("/"),
        pathlib.Path.home().resolve(),
        resolved_root,
        resolved_root.parent if resolved_root.parent != resolved_root else resolved_root,
    }
    if resolved in unsafe:
        print(f"Refusing to remove unsafe cache path: {resolved}")
        return
    if resolved_root not in resolved.parents and resolved != resolved_root:
        print(f"Refusing to remove cache outside cache root: {resolved}")
        return
    shutil.rmtree(resolved, ignore_errors=True)


def cleanup_after_profile(
    *,
    args: argparse.Namespace,
    container_name: str,
    image_ref: str,
    cache_dir: pathlib.Path,
    cache_root: pathlib.Path,
) -> None:
    if args.skip_cleanup:
        print(f"Leaving container/cache/image in place for debugging: {container_name}")
        return

    print(f"Stopping and removing container: {container_name}")
    docker_rm(container_name)

    if args.keep_model_cache:
        print(f"Keeping NIM cache: {cache_dir}")
    else:
        print(f"Removing NIM cache to reclaim disk: {cache_dir}")
        safe_rmtree(cache_dir, cache_root)

    if args.keep_images:
        print(f"Keeping Docker image: {image_ref}")
    else:
        print(f"Removing Docker image to reclaim disk: {image_ref}")
        docker_image_rm(image_ref)

    if args.docker_prune_after_profile:
        print("Pruning unused Docker containers/build cache.")
        docker_prune()


def start_container(
    *,
    profile: ModelProfile,
    image_ref: str,
    container_name: str,
    cache_dir: pathlib.Path,
    ngc_api_key: str,
    tensor_parallel_size: str,
    resolved_extra_env: dict[str, str],
    args: argparse.Namespace,
) -> str:
    docker_rm(container_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # NIM containers may run as a non-root user. The orchestrator creates the
    # host bind-mount path, so make it writable before mounting it as /opt/nim/.cache.
    cache_dir.chmod(0o777)

    env_pairs = {
        "NGC_API_KEY": ngc_api_key,
        "NIM_TENSOR_PARALLEL_SIZE": tensor_parallel_size,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        **resolved_extra_env,
    }
    if profile.passthrough_args:
        env_pairs["NIM_PASSTHROUGH_ARGS"] = profile.passthrough_args

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime=nvidia",
        "--gpus",
        args.gpus,
        "--ipc=host",
        "--shm-size",
        args.docker_shm_size,
    ]
    for key, value in env_pairs.items():
        if value:
            cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(["-v", f"{cache_dir}:/opt/nim/.cache"])
    cmd.extend(["-p", f"{profile.port}:8000"])
    cmd.append(image_ref)
    proc = run_cmd(cmd, check=True, timeout=120)
    return proc.stdout.strip()


def http_json(url: str, timeout_s: int = 5) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def is_container_running(container_name: str) -> bool:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def docker_logs(container_name: str, tail: int = 120) -> str:
    proc = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()


def wait_for_ready(profile: ModelProfile, container_name: str, args: argparse.Namespace) -> bool:
    deadline = time.time() + args.startup_timeout_s
    ready_url = f"http://localhost:{profile.port}/v1/health/ready"
    models_url = f"http://localhost:{profile.port}/v1/models"
    last_log_print = 0.0
    while time.time() < deadline:
        if not is_container_running(container_name):
            print(f"{container_name} exited before readiness.")
            print(docker_logs(container_name, tail=160))
            return False
        ready = http_json(ready_url)
        models = http_json(models_url)
        if models and isinstance(models.get("data"), list) and models["data"]:
            return True
        if ready and str(ready.get("status", "")).lower() == "ready":
            return True
        if time.time() - last_log_print > 60:
            print(f"Waiting for {profile.label} on port {profile.port}...")
            print(docker_logs(container_name, tail=20))
            last_log_print = time.time()
        time.sleep(args.health_interval_s)
    print(f"Timed out waiting for {profile.label}.")
    print(docker_logs(container_name, tail=160))
    return False


def discover_model_id(port: int) -> str:
    models = http_json(f"http://localhost:{port}/v1/models", timeout_s=10)
    data = models.get("data", []) if models else []
    if not data:
        return ""
    return str(data[0].get("id", ""))


def parse_precision_from_logs(container_name: str) -> str:
    logs = docker_logs(container_name, tail=250)
    matches = re.findall(r"Precision:\s*([A-Za-z0-9_.-]+)", logs, flags=re.IGNORECASE)
    return matches[-1] if matches else ""


def current_gpu_metadata() -> dict[str, str]:
    return benchmark_arco.detect_gpu_metadata("", "")


def write_startup_failure_row(
    *,
    args: argparse.Namespace,
    profile: ModelProfile,
    output_path: pathlib.Path,
    experiment_id: str,
    error: str,
) -> None:
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    gpu_metadata = current_gpu_metadata()
    requested_precision = "" if profile.precision_label in {"", "auto"} else profile.precision_label
    row = {
        "experiment_id": experiment_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "deployment_label": profile.label,
        "model": profile.served_model_id,
        "precision_label": profile.precision_label,
        "requested_precision_label": requested_precision,
        "detected_precision_label": "",
        "base_url": f"http://localhost:{profile.port}/v1",
        **gpu_metadata,
        "category": "startup",
        "config_file": "",
        "prompt_template_file": "",
        "prompt_description": "container startup",
        "prompt_query": "",
        "run_index": "",
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "status": "error",
        "error": error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_arco.FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in benchmark_arco.FIELDNAMES})


def write_profile_skip_row(
    *,
    args: argparse.Namespace,
    profile: ModelProfile,
    output_path: pathlib.Path,
    experiment_id: str,
    error: str,
    tensor_parallel_size: str,
) -> None:
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    gpu_metadata = current_gpu_metadata()
    requested_precision = "" if profile.precision_label in {"", "auto"} else profile.precision_label
    row = {
        "experiment_id": experiment_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "deployment_label": profile.label,
        "model": profile.served_model_id,
        "precision_label": profile.precision_label,
        "requested_precision_label": requested_precision,
        "detected_precision_label": "",
        "base_url": f"http://localhost:{profile.port}/v1",
        **gpu_metadata,
        "category": "startup",
        "config_file": "",
        "prompt_template_file": "",
        "prompt_description": f"profile skipped before container start (TP={tensor_parallel_size})",
        "prompt_query": "",
        "run_index": "",
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "status": "error",
        "error": error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_arco.FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in benchmark_arco.FIELDNAMES})


SUMMARY_FIELDNAMES = [
    "experiment_id",
    "deployment_label",
    "model",
    "precision_label",
    "requested_precision_label",
    "detected_precision_label",
    "gpu_count",
    "gpu_names",
    "gpu_memory_total_mb",
    "category",
    "prompt_description",
    "prompt_query",
    "completed_runs",
    "error_runs",
    "accuracy_pass_rate",
    "target_pass_rate",
    "p50_accuracy_score",
    "p90_accuracy_score",
    "p50_ttft_s",
    "p90_ttft_s",
    "p50_total_latency_s",
    "p90_total_latency_s",
    "p50_decode_tok_s",
    "p90_decode_tok_s",
    "p50_e2e_tok_s",
    "p90_e2e_tok_s",
    "median_output_tokens",
    "visible_output_rate",
    "ttft_measured_rate",
    "decode_measured_rate",
]


def summarize_combined_csv(raw_csv: pathlib.Path, summary_csv: pathlib.Path) -> None:
    if not raw_csv.exists():
        return
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    group_fields = [
        "experiment_id",
        "deployment_label",
        "model",
        "precision_label",
        "requested_precision_label",
        "detected_precision_label",
        "gpu_count",
        "gpu_names",
        "gpu_memory_total_mb",
        "category",
        "prompt_description",
        "prompt_query",
    ]
    for row in rows:
        if row.get("category") == "startup":
            continue
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        base = dict(zip(group_fields, key))
        ok_rows = [row for row in group_rows if row.get("status") == "ok"]
        error_rows = [row for row in group_rows if row.get("status") != "ok"]

        def series(field: str) -> list[float]:
            return [
                parsed
                for row in ok_rows
                if (parsed := as_float(row.get(field))) is not None
            ]

        def rate(field: str) -> str:
            if not group_rows:
                return ""
            passed = sum(1 for row in group_rows if truthy(row.get(field, "")))
            return fmt_summary(passed / len(group_rows))

        output_tokens = series("output_tokens")
        summary_rows.append(
            {
                **base,
                "completed_runs": len(ok_rows),
                "error_runs": len(error_rows),
                "accuracy_pass_rate": rate("accuracy_pass"),
                "target_pass_rate": rate("meets_all_targets"),
                "p50_accuracy_score": fmt_summary(percentile(series("accuracy_score"), 0.50)),
                "p90_accuracy_score": fmt_summary(percentile(series("accuracy_score"), 0.90)),
                "p50_ttft_s": fmt_summary(percentile(series("ttft_s"), 0.50)),
                "p90_ttft_s": fmt_summary(percentile(series("ttft_s"), 0.90)),
                "p50_total_latency_s": fmt_summary(percentile(series("total_latency_s"), 0.50)),
                "p90_total_latency_s": fmt_summary(percentile(series("total_latency_s"), 0.90)),
                "p50_decode_tok_s": fmt_summary(percentile(series("decode_tokens_per_s"), 0.50)),
                "p90_decode_tok_s": fmt_summary(percentile(series("decode_tokens_per_s"), 0.90)),
                "p50_e2e_tok_s": fmt_summary(percentile(series("e2e_tokens_per_s"), 0.50)),
                "p90_e2e_tok_s": fmt_summary(percentile(series("e2e_tokens_per_s"), 0.90)),
                "median_output_tokens": fmt_summary(percentile(output_tokens, 0.50)),
                "visible_output_rate": rate("visible_output_captured"),
                "ttft_measured_rate": rate("ttft_measured"),
                "decode_measured_rate": rate("decode_throughput_measured"),
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary sheet: {summary_csv} ({len(summary_rows)} rows)")


ROLLUP_FIELDNAMES = [
    "experiment_id",
    "deployment_label",
    "model",
    "precision_label",
    "requested_precision_label",
    "detected_precision_label",
    "gpu_count",
    "gpu_names",
    "gpu_memory_total_mb",
    "category",
    "completed_runs",
    "error_runs",
    "prompt_count",
    "accuracy_pass_rate",
    "target_pass_rate",
    "p50_accuracy_score",
    "p90_accuracy_score",
    "p50_ttft_s",
    "p90_ttft_s",
    "p50_total_latency_s",
    "p90_total_latency_s",
    "p50_decode_tok_s",
    "p90_decode_tok_s",
    "p50_e2e_tok_s",
    "p90_e2e_tok_s",
    "median_output_tokens",
    "visible_output_rate",
    "ttft_measured_rate",
    "decode_measured_rate",
]


def summarize_rollup_csv(
    raw_csv: pathlib.Path,
    summary_csv: pathlib.Path,
    *,
    include_category: bool,
) -> None:
    if not raw_csv.exists():
        return
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    group_fields = [
        "experiment_id",
        "deployment_label",
        "model",
        "precision_label",
        "requested_precision_label",
        "detected_precision_label",
        "gpu_count",
        "gpu_names",
        "gpu_memory_total_mb",
    ]
    if include_category:
        group_fields.append("category")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("category") == "startup":
            continue
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        base = dict(zip(group_fields, key))
        if not include_category:
            base["category"] = "ALL"
        ok_rows = [row for row in group_rows if row.get("status") == "ok"]
        error_rows = [row for row in group_rows if row.get("status") != "ok"]
        prompt_count = len(
            {
                (row.get("category", ""), row.get("prompt_description", ""))
                for row in group_rows
            }
        )

        def series(field: str) -> list[float]:
            return [
                parsed
                for row in ok_rows
                if (parsed := as_float(row.get(field))) is not None
            ]

        def rate(field: str) -> str:
            if not group_rows:
                return ""
            passed = sum(1 for row in group_rows if truthy(row.get(field, "")))
            return fmt_summary(passed / len(group_rows))

        summary_rows.append(
            {
                **base,
                "completed_runs": len(ok_rows),
                "error_runs": len(error_rows),
                "prompt_count": prompt_count,
                "accuracy_pass_rate": rate("accuracy_pass"),
                "target_pass_rate": rate("meets_all_targets"),
                "p50_accuracy_score": fmt_summary(percentile(series("accuracy_score"), 0.50)),
                "p90_accuracy_score": fmt_summary(percentile(series("accuracy_score"), 0.90)),
                "p50_ttft_s": fmt_summary(percentile(series("ttft_s"), 0.50)),
                "p90_ttft_s": fmt_summary(percentile(series("ttft_s"), 0.90)),
                "p50_total_latency_s": fmt_summary(percentile(series("total_latency_s"), 0.50)),
                "p90_total_latency_s": fmt_summary(percentile(series("total_latency_s"), 0.90)),
                "p50_decode_tok_s": fmt_summary(percentile(series("decode_tokens_per_s"), 0.50)),
                "p90_decode_tok_s": fmt_summary(percentile(series("decode_tokens_per_s"), 0.90)),
                "p50_e2e_tok_s": fmt_summary(percentile(series("e2e_tokens_per_s"), 0.50)),
                "p90_e2e_tok_s": fmt_summary(percentile(series("e2e_tokens_per_s"), 0.90)),
                "median_output_tokens": fmt_summary(percentile(series("output_tokens"), 0.50)),
                "visible_output_rate": rate("visible_output_captured"),
                "ttft_measured_rate": rate("ttft_measured"),
                "decode_measured_rate": rate("decode_throughput_measured"),
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLLUP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote rollup sheet: {summary_csv} ({len(summary_rows)} rows)")


def write_all_summaries(
    raw_csv: pathlib.Path,
    by_prompt_summary: pathlib.Path,
    category_summary: pathlib.Path,
    model_summary: pathlib.Path,
) -> None:
    summarize_combined_csv(raw_csv, by_prompt_summary)
    summarize_rollup_csv(raw_csv, category_summary, include_category=True)
    summarize_rollup_csv(raw_csv, model_summary, include_category=False)


def run_benchmark(
    *,
    args: argparse.Namespace,
    profile: ModelProfile,
    model_id: str,
    precision_label: str,
    requested_precision_label: str,
    detected_precision_label: str,
    gpu_metadata: dict[str, str],
    experiment_id: str,
    output_path: pathlib.Path,
) -> int:
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("benchmark_arco.py")),
        "--arco-repo",
        args.arco_repo,
        "--base-url",
        f"http://localhost:{profile.port}/v1",
        "--model",
        model_id,
        "--deployment-label",
        profile.label,
        "--precision-label",
        precision_label,
        "--requested-precision-label",
        requested_precision_label,
        "--detected-precision-label",
        detected_precision_label,
        "--gpu-label",
        gpu_metadata.get("gpu_names", ""),
        "--gpu-count",
        gpu_metadata.get("gpu_count", ""),
        "--gpu-memory-total-mb",
        gpu_metadata.get("gpu_memory_total_mb", ""),
        "--experiment-id",
        experiment_id,
        "--api-key-env",
        args.benchmark_api_key_env,
        "--allow-missing-api-key",
        "--runs",
        str(args.runs),
        "--concurrency",
        str(args.concurrency),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout-s",
        str(args.timeout_s),
        "--ttft-target-s",
        str(args.ttft_target_s),
        "--total-latency-target-s",
        str(args.total_latency_target_s),
        "--throughput-target-tok-s",
        str(args.throughput_target_tok_s),
        "--output",
        str(output_path),
        "--append",
        "--configs",
        *args.configs,
    ]
    if profile.benchmark_extra_args:
        cmd.extend(shlex.split(profile.benchmark_extra_args))
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    args = parse_args()
    repo_dir = pathlib.Path(__file__).resolve().parent
    matrix_path = pathlib.Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_dir / matrix_path
    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_dir / output_path
    summary_output_path = pathlib.Path(args.summary_output)
    if not summary_output_path.is_absolute():
        summary_output_path = repo_dir / summary_output_path
    category_summary_output_path = pathlib.Path(args.category_summary_output)
    if not category_summary_output_path.is_absolute():
        category_summary_output_path = repo_dir / category_summary_output_path
    model_summary_output_path = pathlib.Path(args.model_summary_output)
    if not model_summary_output_path.is_absolute():
        model_summary_output_path = repo_dir / model_summary_output_path
    arco_repo = pathlib.Path(args.arco_repo).expanduser().resolve()
    args.arco_repo = str(arco_repo)

    if not docker_available():
        print("Docker is required but was not found on PATH.", file=sys.stderr)
        return 2
    if not matrix_path.exists():
        print(f"Matrix file not found: {matrix_path}", file=sys.stderr)
        return 2
    if not arco_repo.exists():
        print(f"Arco repo not found: {arco_repo}", file=sys.stderr)
        return 2

    if not args.append and output_path.exists():
        output_path.unlink()

    profiles = load_profiles(matrix_path, args.start_port)
    gpu_count = detect_gpu_count(args.gpu_count)
    cache_root = pathlib.Path(args.cache_root).expanduser() if args.cache_root else default_cache_root()
    experiment_id = args.experiment_id or time.strftime("arco-nim-%Y%m%d-%H%M%S")

    print(f"Experiment: {experiment_id}")
    print(f"Profiles:   {sum(1 for profile in profiles if profile.enabled)} enabled")
    print(f"GPU count:  {gpu_count}")
    print(f"Cache root: {cache_root}")
    print(f"Output:     {output_path}")
    print(f"By prompt:  {summary_output_path}")
    print(f"By category:{category_summary_output_path}")
    print(f"By model:   {model_summary_output_path}")

    if args.dry_run:
        for profile in profiles:
            if profile.enabled:
                print(f"- {profile.label}: {profile.image} port={profile.port}")
        return 0

    ngc_api_key = os.environ.get(args.ngc_api_key_env, "")
    if not ngc_api_key:
        print(f"Missing {args.ngc_api_key_env}; NIM model downloads may fail.", file=sys.stderr)
        return 2
    if not args.skip_docker_login:
        try:
            docker_login(args.ngc_registry, ngc_api_key)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2

    failures = 0
    for profile in profiles:
        if not profile.enabled:
            print(f"Skipping disabled profile: {profile.label}")
            continue

        container_name = f"{args.container_prefix}-{sanitize(profile.label).lower()}"
        tensor_parallel_size = (
            gpu_count if profile.tensor_parallel_size in {"", "auto"} else profile.tensor_parallel_size
        )
        detected_gpu_count = parse_positive_int(gpu_count)
        requested_tp = parse_positive_int(tensor_parallel_size)
        if profile.cache_dir:
            requested_cache_dir = pathlib.Path(profile.cache_dir).expanduser()
            cache_dir = (
                requested_cache_dir
                if requested_cache_dir.is_absolute()
                else cache_root / requested_cache_dir
            )
        else:
            cache_dir = cache_root / sanitize(profile.label).lower()

        print()
        print("=" * 78)
        print(f"Running profile: {profile.label}")
        print("=" * 78)

        if detected_gpu_count and requested_tp and requested_tp > detected_gpu_count:
            failures += 1
            error = (
                f"Profile requires tensor_parallel_size={requested_tp}, "
                f"but only {detected_gpu_count} GPU(s) were detected."
            )
            print(error)
            write_profile_skip_row(
                args=args,
                profile=profile,
                output_path=output_path,
                experiment_id=experiment_id,
                error=error,
                tensor_parallel_size=tensor_parallel_size,
            )
            write_all_summaries(
                output_path,
                summary_output_path,
                category_summary_output_path,
                model_summary_output_path,
            )
            if not args.continue_on_error:
                return 1
            continue

        image_ref = profile.image
        try:
            if not args.skip_pull:
                image_ref = pull_image(profile.image, args.resolve_amd64_digest)
            resolved_extra_env, profile_resolution_error = resolve_nim_model_profile(
                args=args,
                profile=profile,
                image_ref=image_ref,
                ngc_api_key=ngc_api_key,
                tensor_parallel_size=tensor_parallel_size,
            )
            if profile_resolution_error:
                failures += 1
                print(profile_resolution_error)
                write_profile_skip_row(
                    args=args,
                    profile=profile,
                    output_path=output_path,
                    experiment_id=experiment_id,
                    error=profile_resolution_error,
                    tensor_parallel_size=tensor_parallel_size,
                )
                write_all_summaries(
                    output_path,
                    summary_output_path,
                    category_summary_output_path,
                    model_summary_output_path,
                )
                if not args.continue_on_error:
                    return 1
                continue
            start_container(
                profile=profile,
                image_ref=image_ref,
                container_name=container_name,
                cache_dir=cache_dir,
                ngc_api_key=ngc_api_key,
                tensor_parallel_size=tensor_parallel_size,
                resolved_extra_env=resolved_extra_env,
                args=args,
            )

            if not wait_for_ready(profile, container_name, args):
                failures += 1
                write_startup_failure_row(
                    args=args,
                    profile=profile,
                    output_path=output_path,
                    experiment_id=experiment_id,
                    error="Container did not become ready",
                )
                if not args.continue_on_error:
                    return 1
                continue

            discovered_model_id = discover_model_id(profile.port)
            model_id = (
                discovered_model_id
                if profile.served_model_id in {"", "auto"}
                else profile.served_model_id
            )
            detected_precision = parse_precision_from_logs(container_name)
            requested_precision = (
                "" if profile.precision_label in {"", "auto"} else profile.precision_label
            )
            precision_label = (
                detected_precision
                if profile.precision_label in {"", "auto"}
                else profile.precision_label
            )
            if not precision_label:
                precision_label = "unknown"

            print(f"Served model: {model_id}")
            print(f"Precision:    {precision_label}")

            gpu_metadata = current_gpu_metadata()
            print(f"GPU:          {gpu_metadata.get('gpu_count', '')} x {gpu_metadata.get('gpu_names', '')}")

            rc = run_benchmark(
                args=args,
                profile=profile,
                model_id=model_id,
                precision_label=precision_label,
                requested_precision_label=requested_precision,
                detected_precision_label=detected_precision,
                gpu_metadata=gpu_metadata,
                experiment_id=experiment_id,
                output_path=output_path,
            )
            if rc != 0:
                failures += 1
                print(f"Benchmark failed for {profile.label} with exit code {rc}.")
                if not args.continue_on_error:
                    return rc
            write_all_summaries(
                output_path,
                summary_output_path,
                category_summary_output_path,
                model_summary_output_path,
            )
        except Exception as exc:
            failures += 1
            print(f"Profile failed: {profile.label}: {exc}", file=sys.stderr)
            write_startup_failure_row(
                args=args,
                profile=profile,
                output_path=output_path,
                experiment_id=experiment_id,
                error=str(exc),
            )
            if not args.continue_on_error:
                return 1
            write_all_summaries(
                output_path,
                summary_output_path,
                category_summary_output_path,
                model_summary_output_path,
            )
        finally:
            cleanup_after_profile(
                args=args,
                container_name=container_name,
                image_ref=image_ref,
                cache_dir=cache_dir,
                cache_root=cache_root,
            )

    print()
    write_all_summaries(
        output_path,
        summary_output_path,
        category_summary_output_path,
        model_summary_output_path,
    )
    print(f"Done. Combined CSV: {output_path}")
    print(f"By-prompt CSV:     {summary_output_path}")
    print(f"Category CSV:      {category_summary_output_path}")
    print(f"Model CSV:         {model_summary_output_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
