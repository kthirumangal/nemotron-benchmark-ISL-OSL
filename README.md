# Nemotron Benchmark ISL/OSL

This benchmark runs the `of1-testprompts` JSON chat prompts against OpenAI-compatible chat endpoints. The default model is NVIDIA `Nemotron-3-Nano-30B-A3B`, and the matrix runner also includes `openai/gpt-oss-120b`.

It measures:

- TTFT: request start to first streamed content
- Total latency: request start to stream completion
- Decode throughput: output tokens per second after TTFT
- E2E throughput: output tokens per second over the full request
- Output tokens: provider usage when returned, otherwise a character-based estimate
- Target pass/fail: defaults to TTFT <= 2s, total latency <= 5s, and decode throughput >= 200 tok/s

Reasoning is off / not requested by default for all benchmark rows:

- Nemotron rows send `chat_template_kwargs: {"enable_thinking": false}`.
- GPT-OSS rows do not prepend a `Reasoning:` instruction by default.

Use `--enable-thinking` for Nemotron or `--system-reasoning-effort low|medium|high` for GPT-OSS only when you explicitly want a separate reasoning-mode comparison.

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

Clone the repo on Brev:

```bash
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
cd nemotron-benchmark-ISL-OSL
```

Install notebook dependencies:

```bash
python3 -m pip install -r requirements-notebook.txt
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

Then run:

```bash
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

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

Example GPT-OSS 120B vLLM server:

```bash
vllm serve openai/gpt-oss-120b --host 0.0.0.0 --port 8004
```

Then run just that row by copying or trimming `precision_matrix.example.csv`, or run the full matrix if all endpoints are available.

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
- `status`
- `error`

## Model Notes

- `Nemotron-3-Nano-30B-A3B`: matrix rows include BF16, FP8, and NVFP4 self-hosted profiles.
- `openai/gpt-oss-120b`: matrix rows include MXFP4 with no reasoning instruction by default.
- GPT-OSS requires the harmony response format. OpenAI-compatible runtimes such as vLLM should apply the chat format for `/v1/chat/completions`.

References:

- NVIDIA NIM supported models: https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html
- OpenAI GPT-OSS announcement: https://openai.com/index/introducing-gpt-oss
- GPT-OSS 120B Hugging Face model card: https://huggingface.co/openai/gpt-oss-120b
- OpenAI harmony response format: https://cookbook.openai.com/article/harmony
