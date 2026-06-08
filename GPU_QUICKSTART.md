# GPU Quickstart

Run these commands on a GPU resource with Docker, NVIDIA Container Toolkit, and
NGC/NVCR access.

## 1. Check GPU And Docker

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Clone Repos

```bash
cd ~
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
git clone https://github.com/aem-growth-adoption/aem-growth-arco-benchmark.git
cd ~/nemotron-benchmark-ISL-OSL
```

## 3. Set NGC Key

```bash
export NGC_API_KEY="your_ngc_key_here"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

## 4. Build Orchestrator

```bash
docker build --no-cache -t arco-nim-orchestrator .
mkdir -p "$HOME/nim-cache" results
```

## 5. Run Default Matrix

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

## 6. Run 4x A10G Safe Matrix

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
  --matrix /bench/arco_nim_models.a10g-clean-safe.csv \
  --output /results/arco-a10g-clean-safe-all.csv \
  --summary-output /results/arco-a10g-clean-safe-by-prompt.csv \
  --category-summary-output /results/arco-a10g-clean-safe-by-category.csv \
  --model-summary-output /results/arco-a10g-clean-safe-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --gpu-count 4 \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 7. Run 4x A10G GPT-OSS MXFP4 And Nano FP8 Matrix

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
  --matrix /bench/arco_nim_models.a10g-gptoss-mxfp4-nano-fp8.csv \
  --output /results/arco-a10g-gptoss-mxfp4-nano-fp8-all.csv \
  --summary-output /results/arco-a10g-gptoss-mxfp4-nano-fp8-by-prompt.csv \
  --category-summary-output /results/arco-a10g-gptoss-mxfp4-nano-fp8-by-category.csv \
  --model-summary-output /results/arco-a10g-gptoss-mxfp4-nano-fp8-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 8. Run H100 Nano NVFP4 Profile Check

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
  --matrix /bench/arco_nim_models.h100-nano-nvfp4.csv \
  --output /results/arco-h100-nano-nvfp4-all.csv \
  --summary-output /results/arco-h100-nano-nvfp4-by-prompt.csv \
  --category-summary-output /results/arco-h100-nano-nvfp4-by-category.csv \
  --model-summary-output /results/arco-h100-nano-nvfp4-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 9. Run H200 GPT-OSS Standard Vs Turbo Candidate

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
  --matrix /bench/arco_nim_models.h200-gptoss-standard-vs-turbo.csv \
  --output /results/arco-h200-gptoss-standard-vs-turbo-all.csv \
  --summary-output /results/arco-h200-gptoss-standard-vs-turbo-by-prompt.csv \
  --category-summary-output /results/arco-h200-gptoss-standard-vs-turbo-by-category.csv \
  --model-summary-output /results/arco-h200-gptoss-standard-vs-turbo-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 10. Monitor A Running Profile

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

## 11. Check Results

```bash
ls -lh results
head -5 results/arco-all-experiments.csv
head -5 results/arco-by-prompt-summary.csv
head -5 results/arco-category-summary.csv
head -5 results/arco-model-summary.csv
```

## 12. Use Your Own Prompt Repo

```bash
docker run --rm \
  --name arco-nim-orchestrator-run \
  --network host \
  --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/nim-cache:$HOME/nim-cache" \
  -v "/path/to/your/prompt-repo:/arco:ro" \
  -v "$PWD/results:/results" \
  arco-nim-orchestrator \
  --arco-repo /arco \
  --configs classification.yaml reasoning.yaml recommender.yaml \
  --matrix /bench/arco_nim_models.example.csv \
  --output /results/custom-all-experiments.csv \
  --summary-output /results/custom-by-prompt-summary.csv \
  --category-summary-output /results/custom-by-category.csv \
  --model-summary-output /results/custom-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 13. Cleanup

```bash
docker rm -f arco-nim-orchestrator-run 2>/dev/null || true
docker ps -a --format '{{.Names}}' | grep '^arco-bench-' | xargs -r docker rm -f
docker image prune -f
rm -rf "$HOME/nim-cache"/*
```
