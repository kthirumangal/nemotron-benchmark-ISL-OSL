# Nemotron Benchmark ISL/OSL

This benchmark runs the `of1-testprompts` JSON chat prompts against NVIDIA's OpenAI-compatible NIM API for `Nemotron-3-Nano-30B-A3B`.

It measures:

- TTFT: request start to first streamed content
- Total latency: request start to stream completion
- Decode throughput: output tokens per second after TTFT
- E2E throughput: output tokens per second over the full request
- Output tokens: provider usage when returned, otherwise a character-based estimate

By default the script sends `chat_template_kwargs: {"enable_thinking": false}` so the benchmark measures structured website output rather than additional reasoning traces. Use `--enable-thinking` if you explicitly want to benchmark reasoning mode.

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

Reasoning-mode comparison:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 2000 \
  --runs 3 \
  --concurrency 1 \
  --enable-thinking
```

## Suggested Benchmark Matrix

| Scenario | ISL | OSL cap | Concurrency |
|---|---:|---:|---:|
| First-pass latency | ~16K | 512 | 1 |
| Main target | ~16K | 1024 | 1 |
| Multi-user target | ~16K | 1024 | 5 |
| Stress | ~16K | 2000 | 5 |

The five prompts are approximately `13.1K-15.4K` input tokens with the Nemotron 3 Nano tokenizer, so `15K ISL` is the practical benchmark shape and `16K ISL` is the safer rounded test bucket.

Exact ISL counts calculated with the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer are in:

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
