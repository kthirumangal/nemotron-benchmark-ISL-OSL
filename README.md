# Nemotron NIM Benchmark

Command-first benchmark repo for running NIM models on your GPU resource and
capturing one raw CSV plus by-prompt, by-category, and by-model summaries.

Start with:

```bash
cat GPU_QUICKSTART.md
```

## What It Captures

- TTFT
- Total latency
- Decode throughput
- E2E throughput
- Output tokens
- Accuracy pass rate
- Target pass rate
- GPU name, GPU count, and GPU memory
- Requested and detected precision

## Clone

```bash
cd ~
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
git clone https://github.com/aem-growth-adoption/aem-growth-arco-benchmark.git
cd ~/nemotron-benchmark-ISL-OSL
```

## Set NGC Key

```bash
export NGC_API_KEY="your_ngc_key_here"
```

## Build

```bash
docker build --no-cache -t arco-nim-orchestrator .
```

## Run Default Matrix

```bash
mkdir -p "$HOME/nim-cache" results

docker run --rm \
  --name arco-nim-orchestrator-run \
  --network host \
  --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/nim-cache:$HOME/nim-cache" \
  -v "$HOME/aem-growth-arco-benchmark:/arco:ro" \
  -v "$PWD/results:/results" \
  arco-nim-orchestrator \
  --arco-repo /arco \
  --matrix /bench/arco_nim_models.example.csv \
  --output /results/arco-all-experiments.csv \
  --summary-output /results/arco-by-prompt-summary.csv \
  --category-summary-output /results/arco-category-summary.csv \
  --model-summary-output /results/arco-model-summary.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## Output Files

```text
results/arco-all-experiments.csv
results/arco-by-prompt-summary.csv
results/arco-category-summary.csv
results/arco-model-summary.csv
```

## Other Run Commands

```bash
cat GPU_QUICKSTART.md
cat RUNBOOK.md
```

## Dry Runs

```bash
python3 benchmark_arco.py \
  --arco-repo ../aem-growth-arco-benchmark \
  --dry-run
```

```bash
python3 run_arco_nim_experiments.py \
  --arco-repo ../aem-growth-arco-benchmark \
  --matrix arco_nim_models.example.csv \
  --dry-run
```

## Clean Local Generated Files

```bash
rm -rf results __pycache__ .pytest_cache
```
