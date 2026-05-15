# Nemotron Benchmark ISL/OSL

This benchmark runs the `of1-testprompts` JSON chat prompts against NVIDIA's OpenAI-compatible NIM API for Nemotron Nano.

It measures:

- TTFT: request start to first streamed content
- Total latency: request start to stream completion
- Decode throughput: output tokens per second after TTFT
- E2E throughput: output tokens per second over the full request
- Output tokens: provider usage when returned, otherwise a character-based estimate

## Setup

Set your NVIDIA API key:

```bash
export NVIDIA_API_KEY="..."
```

Default endpoint and model:

```text
https://integrate.api.nvidia.com/v1
nvidia/llama-3.1-nemotron-nano-8b-v1
```

Override them if needed:

```bash
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_MODEL="nvidia/llama-3.1-nemotron-nano-8b-v1"
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

## Suggested Benchmark Matrix

| Scenario | ISL | OSL cap | Concurrency |
|---|---:|---:|---:|
| First-pass latency | ~15K | 512 | 1 |
| Main target | ~15K | 1024 | 1 |
| Multi-user target | ~15K | 1024 | 5 |
| Stress | ~15K | 2000 | 5 |

The five prompts are approximately `12.4K-14.5K` input tokens with a Nemotron/Llama tokenizer, so `15K ISL` is the practical benchmark shape.

Exact ISL counts calculated with the `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` tokenizer are in:

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
