# Command Runbook

## Smoke Test Prompt Loading

```bash
cd ~/nemotron-benchmark-ISL-OSL
python3 benchmark_arco.py \
  --arco-repo ../aem-growth-arco-benchmark \
  --dry-run
```

## Smoke Test Matrix Loading

```bash
cd ~/nemotron-benchmark-ISL-OSL
python3 run_arco_nim_experiments.py \
  --arco-repo ../aem-growth-arco-benchmark \
  --matrix arco_nim_models.example.csv \
  --dry-run
```

## Build Orchestrator

```bash
cd ~/nemotron-benchmark-ISL-OSL
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

## Run A Focused Matrix

```bash
MATRIX=/bench/arco_nim_models.a10g-clean-safe.csv
PREFIX=arco-a10g-clean-safe
```

```bash
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
  --matrix "$MATRIX" \
  --output "/results/${PREFIX}-all.csv" \
  --summary-output "/results/${PREFIX}-by-prompt.csv" \
  --category-summary-output "/results/${PREFIX}-by-category.csv" \
  --model-summary-output "/results/${PREFIX}-by-model.csv" \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## Focused Matrix Names

```bash
arco_nim_models.example.csv
arco_nim_models.a10g-clean-safe.csv
arco_nim_models.a10g-gptoss-mxfp4-nano-fp8.csv
arco_nim_models.h100-nano-nvfp4.csv
arco_nim_models.h200-gptoss-standard-vs-turbo.csv
```

## Monitor

```bash
docker ps
nvidia-smi
du -sh "$HOME/nim-cache"/*
```

```bash
docker logs -f arco-bench-nim-nano-30b-fp8
docker logs -f arco-bench-nim-nano-30b-nvfp4
docker logs -f arco-bench-gpt-oss-120b-mxfp4
docker logs -f arco-bench-nim-super-120b-fp8
docker logs -f arco-bench-nim-super-120b-nvfp4
```

```bash
docker logs -f arco-bench-nim-nano-30b-bf16-safe
docker logs -f arco-bench-gpt-oss-120b-mxfp4-safe
```

```bash
docker logs -f arco-bench-nim-nano-30b-nvfp4-h100
docker logs -f arco-bench-gpt-oss-120b-mxfp4-h200
docker logs -f arco-bench-gpt-oss-120b-turbo-candidate-h200
```

```bash
curl http://localhost:8002/v1/health/ready && curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/health/ready && curl http://localhost:8003/v1/models
curl http://localhost:8004/v1/health/ready && curl http://localhost:8004/v1/models
curl http://localhost:8005/v1/health/ready && curl http://localhost:8005/v1/models
curl http://localhost:8006/v1/health/ready && curl http://localhost:8006/v1/models
curl http://localhost:8007/v1/health/ready && curl http://localhost:8007/v1/models
```

## Check Output

```bash
ls -lh results
head -5 results/*summary*.csv
```

## Direct Hosted API Check

```bash
export NVIDIA_API_KEY="your_key_here"
python3 benchmark_precision_matrix.py \
  --matrix precision_matrix.example.csv \
  --prompt-dir of1-testprompts \
  --ttft-target-s 2.0 \
  --total-latency-target-s 5.0 \
  --throughput-target-tok-s 200
```

## Direct Local Endpoint Check

```bash
curl http://localhost:8004/v1/models
```

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
  --output results/gpt-oss-120b-mxfp4.csv
```

## Cleanup

```bash
docker rm -f arco-nim-orchestrator-run 2>/dev/null || true
docker ps -a --format '{{.Names}}' | grep '^arco-bench-' | xargs -r docker rm -f
docker image prune -f
rm -rf "$HOME/nim-cache"/*
```
