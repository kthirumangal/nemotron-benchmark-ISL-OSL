# Runbook

This benchmark calls OpenAI-compatible chat endpoints. It does not start local model servers.

## Fastest Valid Run: Hosted NVIDIA API

Use this first.

```bash
cd ~/nemotron-benchmark-ISL-OSL
export NVIDIA_API_KEY="your_key_here"
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

The default `precision_matrix.example.csv` enables only the hosted API row.

## Automated Arco NIM Run Checks

For the AEM / Arco benchmark, prefer the Docker orchestrator. It pulls one NIM image at a time, resolves the exact runnable NIM profile for the requested precision, starts the server, waits for readiness, benchmarks, writes CSVs, then cleans up before the next profile.

A healthy profile startup looks like this:

```text
Running profile: NIM Nano 30B FP8
+ docker buildx imagetools inspect --raw <nim image>
Resolved linux/amd64 image: <image>@sha256:<digest>
+ docker pull <image>@sha256:<digest>
Status: Downloaded newer image
Resolving NIM model profile from image manifest...
+ docker run ... list-model-profiles
Resolved NIM profile: <64-char-profile-id> (vllm-<precision>-tp<tp>-pp1-<memory>)
+ docker run -d --name arco-bench-...
Waiting for <profile> on port <port>...
```

While the orchestrator is waiting, monitor from another terminal:

```bash
docker logs -f arco-bench-nim-nano-30b-fp8
curl http://localhost:8002/v1/health/ready
curl http://localhost:8002/v1/models
```

Profile container names and ports:

```text
arco-bench-nim-nano-30b-fp8     -> 8002
arco-bench-nim-nano-30b-nvfp4   -> 8003
arco-bench-gpt-oss-120b-mxfp4   -> 8004
arco-bench-nim-super-120b-fp8   -> 8005
arco-bench-nim-super-120b-nvfp4 -> 8006
```

Good readiness signs:

```text
Application startup complete
/v1/models returns a model id
/v1/health/ready returns ready
```

Expected output files:

```text
results/arco-all-experiments.csv
results/arco-by-prompt-summary.csv
results/arco-category-summary.csv
results/arco-model-summary.csv
```

## 4x A10G Clean Safe Matrix

For the most reliable 4x A10G run with current NIM images, use:

```text
arco_nim_models.a10g-clean-safe.csv
```

This runs only:

```text
NIM Nano 30B BF16 Safe
GPT-OSS 120B MXFP4 Safe
```

The matrix uses TP4, `--max-model-len 16384`, `--max-num-seqs 1`, and `--gpu-memory-utilization 0.90`. This is intentionally more conservative than the general 32K matrix because 4x A10G has very little BF16 headroom.

Expected output files for this run:

```text
results/arco-a10g-clean-safe-all.csv
results/arco-a10g-clean-safe-by-prompt.csv
results/arco-a10g-clean-safe-by-category.csv
results/arco-a10g-clean-safe-by-model.csv
```

## 4x A10G Matrix: GPT-OSS MXFP4 And Nano FP8

For the focused 4x A10G run, use:

```text
arco_nim_models.a10g-gptoss-mxfp4-nano-fp8.csv
```

This runs only:

```text
NIM Nano 30B FP8
GPT-OSS 120B MXFP4
```

The matrix uses `tensor_parallel_size=auto`, so the orchestrator should detect four A10Gs and resolve TP4 profiles from each NIM image manifest. The A10G matrix uses `--max-num-seqs 4` because the benchmark runs concurrency 1 and this leaves more memory headroom than the general matrix.

Expected output files for this run:

```text
results/arco-a10g-gptoss-mxfp4-nano-fp8-all.csv
results/arco-a10g-gptoss-mxfp4-nano-fp8-by-prompt.csv
results/arco-a10g-gptoss-mxfp4-nano-fp8-by-category.csv
results/arco-a10g-gptoss-mxfp4-nano-fp8-by-model.csv
```

## H100 Nano NVFP4 Profile Check

For a focused H100 Nano NVFP4 profile check, use:

```text
arco_nim_models.h100-nano-nvfp4.csv
```

This matrix is intentionally a compatibility check. Recent
`nemotron-3-nano:latest` manifests have marked Nano NVFP4 as incompatible on
some H100 instances while FP8 and BF16 were runnable. If that happens, the
orchestrator writes a startup skip/error row and continues. Treat successful
H100 NVFP4 numbers as image/profile-specific and record the exact NIM image
tag/digest and profile ID.

Expected output files for this run:

```text
results/arco-h100-nano-nvfp4-all.csv
results/arco-h100-nano-nvfp4-by-prompt.csv
results/arco-h100-nano-nvfp4-by-category.csv
results/arco-h100-nano-nvfp4-by-model.csv
```

## H200 GPT-OSS Standard Vs Turbo Candidate Matrix

For a focused H200 GPT-OSS comparison, use:

```text
arco_nim_models.h200-gptoss-standard-vs-turbo.csv
```

This compares:

```text
GPT-OSS 120B MXFP4 H200
GPT-OSS 120B Turbo Candidate H200
```

The Turbo candidate image is:

```text
nvcr.io/nim/openai/gpt-oss-120b-turbo:1.0.0
```

If the image is not yet released to your NGC account, or if you are testing a
private NimCraft release candidate, update the matrix row with the exact
`nvcr.io/...` image URI you were granted. The orchestrator records the
pull/startup failure and continues.

Expected output files for this run:

```text
results/arco-h200-gptoss-standard-vs-turbo-all.csv
results/arco-h200-gptoss-standard-vs-turbo-by-prompt.csv
results/arco-h200-gptoss-standard-vs-turbo-by-category.csv
results/arco-h200-gptoss-standard-vs-turbo-by-model.csv
```

## Local Model Runs

Local rows require servers to be running first:

```text
localhost:8001 -> Nemotron BF16
localhost:8002 -> Nemotron FP8
localhost:8003 -> Nemotron NVFP4
localhost:8004 -> GPT-OSS 120B
```

Check before running:

```bash
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/models
curl http://localhost:8004/v1/models
```

If you get `Connection refused`, do not enable that row.

## Measurement Modes

The matrix has a `measurement_mode` column.

```text
strict  -> missing visible streamed output / TTFT / decode throughput becomes an error
lenient -> completed responses stay ok, but missing metrics are marked as not measured
```

Keep Nano rows as `strict`. Use `lenient` for GPT-OSS if you want to preserve total latency, provider token usage, and E2E throughput even when this benchmark client does not capture visible streamed content for every run.

## GPT-OSS Streaming Debug Workflow

If GPT-OSS completes requests but TTFT/decode are blank for some prompts, do not treat that as a model failure yet. First check whether the endpoint streamed final visible content or only reasoning/non-content fields.

Run a focused GPT-OSS pass:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --base-url http://localhost:8004/v1 \
  --model openai/gpt-oss-120b \
  --precision-label MXFP4 \
  --allow-missing-api-key \
  --omit-chat-template-kwargs \
  --api-reasoning-effort low \
  --force-visible-output \
  --measurement-mode lenient \
  --max-tokens 1024 \
  --runs 3 \
  --concurrency 1 \
  --timeout-s 180 \
  --stream-debug-dir results/gpt-oss-stream-debug \
  --output results/gpt-oss-120b-mxfp4-stream-debug.csv
```

Interpretation:

```text
content_chunks > 0    -> final visible streamed content was captured and counts for TTFT/decode
reasoning_chunks > 0  -> reasoning text streamed, but it does not count as visible output
reasoning_only_no_visible_output -> server streamed/thought but did not expose final visible content
usage_only_no_visible_output     -> request completed with usage/latency but no text fields were captured
```

Use `--capture-reasoning-as-output` only to debug stream timing. Do not use it for final customer-facing TTFT/decode results.

## One-GPU Workflow

On one GPU, benchmark one local model/profile at a time.

1. Start one model server.
2. Verify `curl http://localhost:<port>/v1/models` works.
3. Edit `precision_matrix.example.csv`.
4. Set only that row to `enabled=true`.
5. Run `benchmark_precision_matrix.py`.
6. Stop the model server.
7. Repeat for the next profile.
8. Combine results:

```bash
python3 combine_results.py
```

Outputs:

```text
results/combined-summary.csv
results/combined-details.csv
```

## What Connection Refused Means

`ConnectionRefusedError` means the benchmark tried to call a local URL, but no process was listening on that port. It is not a model quality or latency result.

Fix it by either:

- starting the model server for that row, or
- setting that row to `enabled=false`.
