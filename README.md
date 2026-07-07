# Nemotron NIM Benchmark

Run NIM benchmark experiments on your GPU resource and save one raw CSV plus
three summaries: by prompt, by category, and by model.

Captured fields include latency, TTFT, decode throughput, E2E throughput, output
tokens, accuracy pass rate, target pass rate, GPU details, model, and precision.

## 1. Clone

```bash
cd ~
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
git clone https://github.com/aem-growth-adoption/aem-growth-arco-benchmark.git
cd ~/nemotron-benchmark-ISL-OSL
```

## 2. Set NGC Key

```bash
export NGC_API_KEY="your_ngc_key_here"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

## 3. Build

```bash
docker build --no-cache -t arco-nim-orchestrator .
mkdir -p "$HOME/nim-cache" results
```

## 4. Run

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

## 5. Recommender Token Sweep

Use this to test whether shorter recommender output lowers E2E latency.

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
  --matrix /bench/arco_nim_models.h100-nano-fp8.csv \
  --configs recommender.yaml \
  --max-tokens-sweep 1024 768 512 384 \
  --output /results/arco-h100-nano-fp8-recommender-token-sweep-all.csv \
  --summary-output /results/arco-h100-nano-fp8-recommender-token-sweep-by-prompt.csv \
  --category-summary-output /results/arco-h100-nano-fp8-recommender-token-sweep-by-category.csv \
  --model-summary-output /results/arco-h100-nano-fp8-recommender-token-sweep-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

Compare `max_tokens`, `p50_total_latency_s`, `median_output_tokens`, and
`accuracy_pass_rate` in:

```bash
results/arco-h100-nano-fp8-recommender-token-sweep-by-category.csv
```

## 6. MiMo V2 Flash

Use this to benchmark the experimental MiMo V2 Flash NIM container.

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
  --matrix /bench/arco_nim_models.mimo-v2-flash-experimental.csv \
  --output /results/arco-mimo-v2-flash-all.csv \
  --summary-output /results/arco-mimo-v2-flash-by-prompt.csv \
  --category-summary-output /results/arco-mimo-v2-flash-by-category.csv \
  --model-summary-output /results/arco-mimo-v2-flash-by-model.csv \
  --cache-root "$HOME/nim-cache" \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

For a faster first pass, add only the recommender category:

```bash
--configs recommender.yaml
```

## 7. Watch

```bash
docker ps
nvidia-smi
du -sh "$HOME/nim-cache"/*
docker logs -f arco-nim-orchestrator-run
```

For a running model container:

```bash
docker logs -f <container_name>
```

## 8. Results

```bash
ls -lh results
head -5 results/arco-all-experiments.csv
head -5 results/arco-by-prompt-summary.csv
head -5 results/arco-category-summary.csv
head -5 results/arco-model-summary.csv
```

## 9. Use A Different Matrix

Change only the `--matrix` file and output names in the run command.

```text
/bench/arco_nim_models.example.csv
/bench/arco_nim_models.h100-nano-fp8.csv
/bench/arco_nim_models.mimo-v2-flash-experimental.csv
/bench/arco_nim_models.a10g-clean-safe.csv
/bench/arco_nim_models.a10g-gptoss-mxfp4-nano-fp8.csv
/bench/arco_nim_models.h100-nano-nvfp4.csv
/bench/arco_nim_models.h200-gptoss-standard-vs-turbo.csv
```

## 10. Use Your Own Prompts

Replace this mount:

```bash
-v "$HOME/aem-growth-arco-benchmark:/arco:ro"
```

with:

```bash
-v "/path/to/your/prompt-repo:/arco:ro"
```

If your config file names differ, add:

```bash
--configs classification.yaml reasoning.yaml recommender.yaml
```

## 11. Cleanup

```bash
docker rm -f arco-nim-orchestrator-run 2>/dev/null || true
docker ps -a --format '{{.Names}}' | grep '^arco-bench-' | xargs -r docker rm -f
docker image prune -f
rm -rf "$HOME/nim-cache"/*
```
