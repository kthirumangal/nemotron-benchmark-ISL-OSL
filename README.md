# Nemotron Benchmark ISL/OSL

This benchmark runs the `of1-testprompts` JSON chat prompts against OpenAI-compatible chat endpoints. The default model is NVIDIA `Nemotron-3-Nano-30B-A3B`, and the matrix runner also includes `openai/gpt-oss-120b`.

It measures:

- TTFT: request start to first streamed content
- Total latency: request start to stream completion
- Decode throughput: output tokens per second after TTFT
- E2E throughput: output tokens per second over the full request
- Output tokens: provider usage when returned, otherwise a character-based estimate
- Measurement coverage: whether visible streamed output, TTFT, and decode throughput were actually captured
- Target pass/fail: defaults to TTFT <= 2s, total latency <= 5s, and decode throughput >= 200 tok/s

Reasoning is disabled where the serving path supports it, and minimized for GPT-OSS:

- Nemotron rows send `chat_template_kwargs: {"enable_thinking": false}`.
- GPT-OSS rows do not prepend a `Reasoning:` instruction by default. The example matrix sends request-level `reasoning_effort=low` because the GPT-OSS NIM API exposes `low`, `medium`, and `high` reasoning effort rather than an off switch. It also adds a visible-output instruction so the benchmark can test whether final assistant content is streamed consistently.

Use `--enable-thinking` for Nemotron, `--api-reasoning-effort low|medium|high`, or `--system-reasoning-effort low|medium|high` only when you explicitly want a separate reasoning-mode comparison.

## Measurement Modes

The benchmark supports two measurement modes:

- `strict`: a completed HTTP response is marked `error` if no visible streamed content, TTFT, or decode throughput was captured.
- `lenient`: a completed HTTP response remains `ok`, but missing TTFT/decode metrics are recorded as not measured.

Use `strict` for clean apples-to-apples streaming comparisons. Use `lenient` for GPT-OSS or other runtimes where provider usage and total latency may be available even when visible streamed content is not captured by this client. The matrix examples set GPT-OSS to `lenient` and Nano rows to `strict`.

## GPT-OSS Streaming Diagnostics

GPT-OSS is a reasoning model. Depending on the NIM/vLLM serving configuration, streamed chunks can carry text in final visible fields such as `delta.content`, or reasoning fields such as `delta.reasoning_content` / `delta.reasoning`. This benchmark now records both:

- final visible content chunks count toward TTFT and decode throughput
- reasoning chunks are counted separately in `reasoning_chunks` / `reasoning_chars`
- `--stream-debug-dir <dir>` writes one JSONL trace per prompt run so you can inspect the exact streamed fields

Recommended GPT-OSS diagnostic run:

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

Use `--capture-reasoning-as-output` only as a diagnostic to prove the server is streaming reasoning text. Do not use it for final visible-output TTFT/decode comparisons.

## Start Here: How To Run

There are two different ways to run this benchmark:

### 1. Hosted NVIDIA API

This runs immediately after you set `NVIDIA_API_KEY`. You do not need to start any local model server.

Use this first to confirm the benchmark works:

```bash
export NVIDIA_API_KEY="your_key_here"
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

By default, `precision_matrix.example.csv` enables only:

```text
hosted-managed -> https://integrate.api.nvidia.com/v1
```

### 2. Local/Self-Hosted Models

These rows do not run until you start a local OpenAI-compatible model server yourself.

The CSV rows are only endpoints. They do not launch models.

```text
localhost:8001 -> Nemotron BF16 server must already be running
localhost:8002 -> Nemotron FP8 server must already be running
localhost:8003 -> Nemotron NVFP4 server must already be running
localhost:8004 -> GPT-OSS 120B server must already be running
```

Before enabling a local row, verify the server exists:

```bash
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/models
curl http://localhost:8004/v1/models
```

If `curl` returns `Connection refused`, that row is not runnable yet. Keep it `enabled=false`.

On a one-GPU Brev instance, run local models one at a time:

1. Start one local model server.
2. Verify it with `curl http://localhost:<port>/v1/models`.
3. Set only that row to `enabled=true`.
4. Run the benchmark.
5. Stop the server.
6. Repeat for the next model/profile.
7. Run `python3 combine_results.py` to compare all result folders side by side.

## Setup

Set your NVIDIA API key:

```bash
export NVIDIA_API_KEY="..."
```

Default endpoint and model:

```text
https://integrate.api.nvidia.com/v1
nvidia/nemotron-3-nano-30b-a3b
```

Override them if needed:

```bash
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_MODEL="nvidia/nemotron-3-nano-30b-a3b"
```

## Brev Notebook Workflow

Brev often opens directly into a notebook instance. You can run setup either from a terminal or from notebook cells.

Clone the repo on Brev:

```bash
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
cd nemotron-benchmark-ISL-OSL
```

If you are inside a notebook cell, use:

```python
!git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
%cd nemotron-benchmark-ISL-OSL
```

Install notebook dependencies:

```bash
python3 -m pip install -r requirements-notebook.txt
```

If Brev reports `No module named pip`, bootstrap `pip` in the active Python environment:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-notebook.txt
```

If `ensurepip` is unavailable:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
python3 -m ensurepip --upgrade
python3 -m pip install -r requirements-notebook.txt
```

If Brev is using `/home/ubuntu/.venv`, call that Python directly:

```bash
/home/ubuntu/.venv/bin/python -m ensurepip --upgrade
/home/ubuntu/.venv/bin/python -m pip install --upgrade pip
/home/ubuntu/.venv/bin/python -m pip install -r requirements-notebook.txt
```

From a notebook cell, prefix shell commands with `!`:

```python
!python3 -m ensurepip --upgrade
!python3 -m pip install --upgrade pip
!python3 -m pip install -r requirements-notebook.txt
```

As a fallback, install only the packages needed for visualization:

```bash
python3 -m pip install pandas matplotlib
```

Open:

```text
notebooks/benchmark_visualization.ipynb
```

The notebook lets you:

- Review/edit `precision_matrix.example.csv`
- Run the benchmark matrix after your endpoints are live
- Load the latest `results/precision-matrix-*/summary.csv`
- Plot p90 TTFT, p90 total latency, p50 decode throughput, and pass/fail status

If your Brev instance has one GPU, do not try to run every model endpoint at the same time. Keep only the active endpoint row set to `enabled=true` in `precision_matrix.example.csv`, run the matrix, save results, then switch to the next endpoint/profile.

After running multiple one-endpoint benchmarks, combine them for side-by-side comparison:

```bash
python3 combine_results.py
```

This writes:

```text
results/combined-summary.csv
results/combined-details.csv
```

## Run

Single-run latency check:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 1 \
  --concurrency 1
```

Use custom targets:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 5 \
  --concurrency 1 \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

Repeat runs for p50/p90:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 5 \
  --concurrency 1
```

Concurrency check:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 3 \
  --concurrency 5
```

Stress output length:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 2000 \
  --runs 3 \
  --concurrency 5
```

Optional reasoning-mode comparison, not part of the default latency matrix:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 2000 \
  --runs 3 \
  --concurrency 1 \
  --enable-thinking
```

## Precision Matrix

The hosted NVIDIA API model alias does not expose precision selection directly. To compare BF16, FP8, and NVFP4, run separate self-hosted NIM/vLLM endpoints for each precision profile, then point the matrix runner at those endpoints.

For GPT-OSS 120B, run an OpenAI-compatible runtime such as vLLM/SGLang and add the endpoint to the same matrix. GPT-OSS 120B is natively MXFP4-quantized and designed to fit on a single 80GB GPU such as an NVIDIA H100. It is not served through the OpenAI API.

Edit:

```text
precision_matrix.example.csv
```

Use `precision_matrix.example.csv` for the safe default: hosted API enabled, local endpoints disabled.

Use `precision_matrix.all-local.example.csv` when every endpoint is live and you want side-by-side results for:

- Hosted managed Nemotron 3 Nano
- Self-hosted Nemotron BF16
- Self-hosted Nemotron FP8
- Self-hosted Nemotron NVFP4
- Self-hosted GPT-OSS 120B MXFP4

Then run:

```bash
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

All-endpoint comparison:

```bash
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.all-local.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

Single-GPU comparison workflow:

1. Start one model endpoint.
2. Set only that row to `enabled=true` in `precision_matrix.example.csv`.
3. Run `benchmark_precision_matrix.py`.
4. Stop that endpoint.
5. Repeat for the next model/profile.
6. Run `python3 combine_results.py` to create combined side-by-side CSVs.

The matrix runner writes one CSV per profile plus a summary:

```text
results/precision-matrix-<timestamp>/summary.csv
```

Summary pass/fail columns:

- `p90_ttft_pass`
- `p90_total_latency_pass`
- `p50_decode_throughput_pass`
- `meets_p90_targets`

For local endpoints without API keys, set `allow_missing_api_key=true` in the matrix CSV.

The matrix CSV has an `enabled` column. Only rows with `enabled=true` run. The default file enables only the hosted NVIDIA API row and leaves local self-hosted rows disabled until you start those model servers. This avoids `ConnectionRefusedError` noise from localhost ports that are not live yet.

If the hosted row is enabled but `NVIDIA_API_KEY` is not set, the matrix runner skips it and records `skip_reason=missing NVIDIA_API_KEY` in the summary instead of failing the whole run.

Example GPT-OSS 120B vLLM server:

```bash
vllm serve openai/gpt-oss-120b --host 0.0.0.0 --port 8004
```

Then change that GPT-OSS row to `enabled=true` in `precision_matrix.example.csv`, or run the full matrix if all endpoints are available.

## Suggested Benchmark Matrix

| Scenario | ISL | OSL cap | Concurrency |
|---|---:|---:|---:|
| First-pass latency | ~16K | 512 | 1 |
| Main target | ~16K | 1024 | 1 |
| Multi-user target | ~16K | 1024 | 5 |
| Stress | ~16K | 2000 | 5 |

The five prompts are approximately `13.1K-15.4K` input tokens with the Nemotron 3 Nano tokenizer and `12.3K-14.4K` input tokens with the GPT-OSS harmony tokenizer, so `16K ISL` is the safer rounded test bucket.

Exact ISL counts calculated with the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer and the GPT-OSS harmony renderer are in:

```text
prompt_inventory.csv
```

## Output

CSV files are written under:

```text
results/
```

Key columns:

- `ttft_s`
- `total_latency_s`
- `output_tokens`
- `decode_tokens_per_s`
- `e2e_tokens_per_s`
- `measurement_mode`
- `visible_output_captured`
- `ttft_measured`
- `decode_throughput_measured`
- `measurement_quality`
- `streamed_chunks`
- `content_chunks`
- `reasoning_chunks`
- `reasoning_chars`
- `debug_trace_path`
- `status`
- `error`

## Model Notes

- `Nemotron-3-Nano-30B-A3B`: matrix rows include BF16, FP8, and NVFP4 self-hosted profiles.
- `openai/gpt-oss-120b`: matrix rows include MXFP4. The example row uses `api_reasoning_effort=low`, `force_visible_output=true`, `measurement_mode=lenient`, and `stream_debug_dir=stream-debug` so missing visible-content metrics can be debugged instead of treated as model failure.
- GPT-OSS requires the harmony response format. OpenAI-compatible runtimes such as vLLM should apply the chat format for `/v1/chat/completions`.

References:

- NVIDIA NIM supported models: https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html
- OpenAI GPT-OSS announcement: https://openai.com/index/introducing-gpt-oss
- GPT-OSS 120B Hugging Face model card: https://huggingface.co/openai/gpt-oss-120b
- OpenAI harmony response format: https://cookbook.openai.com/article/harmony
- NVIDIA GPT-OSS 120B NIM API reference: https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-120b-infer
- vLLM reasoning outputs: https://docs.vllm.ai/features/reasoning_outputs.html
