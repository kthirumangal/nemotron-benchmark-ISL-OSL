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
