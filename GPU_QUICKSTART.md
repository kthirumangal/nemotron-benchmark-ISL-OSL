# GPU Resource Quickstart

Use this on any GPU resource that has Docker, NVIDIA Container Toolkit, access
to the target GPUs, and network access to NGC/NVCR. You can run it from a
terminal or from notebook cells if your environment provides a notebook UI.

## Read This First

There are two benchmark paths in this repo.

1. The legacy ISL/OSL precision matrix does not start local models for you.
2. The Arco NIM orchestrator does start NIM containers for you, waits for readiness, benchmarks, writes CSVs, then cleans up before the next profile.

For the AEM / Arco benchmark, use the automated Arco NIM orchestrator.

## Automated Arco NIM Run

The default Arco model matrix runs five profiles sequentially:

```text
NIM Nano 30B FP8
NIM Nano 30B NVFP4
GPT-OSS 120B MXFP4
NIM Super 120B FP8
NIM Super 120B NVFP4
```

The matrix specifies the desired precision, not a hardcoded manifest profile ID. For each NIM image, the orchestrator now lists the profiles from that exact container, selects the runnable non-LoRA profile matching the requested precision and detected tensor-parallel size, and injects the resolved 64-character `NIM_MODEL_PROFILE` automatically. If a requested precision cannot run on the current GPU, the orchestrator records a clear startup skip/error row and continues.

Clone both repos:

```bash
cd ~
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
git clone https://github.com/aem-growth-adoption/aem-growth-arco-benchmark.git
cd ~/nemotron-benchmark-ISL-OSL
```

To reproduce with your own traffic or prompt data, keep the same PromptFoo-style
layout as the Arco repo:

```text
classification.yaml
reasoning.yaml
recommender.yaml
prompts/*.yaml
```

Each config should contain `prompts: [file://...]`, `tests`, optional
`defaultTest.assert`, and per-test `vars` / `assert` entries. Then mount your
repo at `/arco` or pass a different `--arco-repo` path. To benchmark different
config names, pass `--configs your-config.yaml another-config.yaml` to the
orchestrator command.

Set your NGC key:

```bash
export NGC_API_KEY="your_ngc_key_here"
```

Build the orchestrator:

```bash
docker build --no-cache -t arco-nim-orchestrator .
```

Run the full matrix:

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

Expected outputs:

```text
results/arco-all-experiments.csv
results/arco-by-prompt-summary.csv
results/arco-category-summary.csv
results/arco-model-summary.csv
```

The orchestrator removes each NIM container, image, and per-profile cache after that profile unless you pass `--keep-images` or `--keep-model-cache`.

## What A Healthy Automated Run Looks Like

For each profile, the orchestrator should print this sequence:

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

That means the image pull worked, the exact NIM model profile was selected automatically, and the server is starting. First startup can take several minutes because NIM may download model artifacts into the per-profile cache.

To monitor readiness without interrupting the orchestrator, use another terminal:

```bash
docker logs -f arco-bench-nim-nano-30b-fp8
curl http://localhost:8002/v1/health/ready
curl http://localhost:8002/v1/models
```

For other profiles, replace the container name and port:

```text
arco-bench-nim-nano-30b-nvfp4  -> 8003
arco-bench-gpt-oss-120b-mxfp4 -> 8004
arco-bench-nim-super-120b-fp8 -> 8005
arco-bench-nim-super-120b-nvfp4 -> 8006
```

Good readiness signs:

```text
Application startup complete
/v1/models returns a model id
/v1/health/ready returns ready
```

After each profile, the orchestrator should write summaries and clean up:

```text
Wrote summary sheet
Wrote rollup sheet
Stopping and removing container
Removing NIM cache to reclaim disk
Removing Docker image to reclaim disk
```

## 4x A10G: Clean Safe Run

Use this matrix when you want the most reliable current A10G run:

```text
arco_nim_models.a10g-clean-safe.csv
```

It runs:

```text
NIM Nano 30B BF16 Safe
GPT-OSS 120B MXFP4 Safe
```

This matrix uses TP4, `--max-model-len 16384`, `--max-num-seqs 1`, and `--gpu-memory-utilization 0.90` to avoid the tight A10G memory failures seen with 32K context. The Arco prompts observed in the benchmark fit within this 16K serving context.

Run it with:

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
  --keep-model-cache \
  --runs 3 \
  --concurrency 1 \
  --continue-on-error
```

## 4x A10G: GPT-OSS MXFP4 And Nano FP8

Use this matrix when you only want to compare:

```text
GPT-OSS 120B MXFP4
NIM Nano 30B FP8
```

The matrix file is:

```text
arco_nim_models.a10g-gptoss-mxfp4-nano-fp8.csv
```

It keeps `tensor_parallel_size=auto`, so the orchestrator detects the 4 A10Gs and resolves matching TP4 NIM profiles automatically. The profile resolver still records a clear startup row and continues if a requested profile is not runnable on the current hardware.

Run it with:

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

## H100: Nano NVFP4 Profile Check

Use this focused matrix only when you want to test whether the currently pulled
Nano NIM exposes a runnable NVFP4 profile on your H100:

```text
arco_nim_models.h100-nano-nvfp4.csv
```

Recent `nemotron-3-nano:latest` manifests have marked Nano NVFP4 as
incompatible on some H100 instances while FP8 and BF16 were runnable. The
orchestrator will record a clear startup skip row if the NVFP4 profile is not
runnable on the exact image/hardware combination. If someone shares H100 NVFP4
numbers, ask for the exact NIM image tag/digest and profile ID used.

Run it with:

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

## H200: GPT-OSS Standard Vs Turbo Candidate

Use this matrix to compare the current GPT-OSS 120B NIM against the candidate Turbo NIM image:

```text
arco_nim_models.h200-gptoss-standard-vs-turbo.csv
```

The Turbo candidate row uses:

```text
nvcr.io/nim/openai/gpt-oss-120b-turbo:1.0.0
```

If that image is not yet visible in your NGC account, or if you are testing a
private release-candidate image from NimCraft, update the matrix row with the
exact `nvcr.io/...` image URI you were granted. The orchestrator records the
pull/startup failure and still writes the summary CSVs when access or profile
selection fails.

Run it with:

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

## Legacy ISL/OSL Precision Matrix

The local rows in `precision_matrix.example.csv` are just URLs:

```text
localhost:8001 -> Nemotron BF16
localhost:8002 -> Nemotron FP8
localhost:8003 -> Nemotron NVFP4
localhost:8004 -> GPT-OSS 120B
```

Before setting any local row to `enabled=true`, verify it:

```bash
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/models
curl http://localhost:8004/v1/models
```

If `curl` says `Connection refused`, there is no server running on that port. Leave that row disabled.

## Clone

Terminal:

```bash
git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
cd nemotron-benchmark-ISL-OSL
```

Notebook cell:

```python
!git clone https://github.com/kthirumangal/nemotron-benchmark-ISL-OSL.git
%cd nemotron-benchmark-ISL-OSL
```

## Install Notebook Dependencies

Try:

```bash
python3 -m pip install -r requirements-notebook.txt
```

If you see `No module named pip`:

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

If your GPU environment uses `/home/ubuntu/.venv`:

```bash
/home/ubuntu/.venv/bin/python -m ensurepip --upgrade
/home/ubuntu/.venv/bin/python -m pip install --upgrade pip
/home/ubuntu/.venv/bin/python -m pip install -r requirements-notebook.txt
```

Notebook-cell equivalent:

```python
!python3 -m ensurepip --upgrade
!python3 -m pip install --upgrade pip
!python3 -m pip install -r requirements-notebook.txt
```

Fallback:

```bash
python3 -m pip install pandas matplotlib
```

## Open The Notebook

Open:

```text
notebooks/benchmark_visualization.ipynb
```

Set:

```python
RUN_BENCHMARK = True
```

after your model endpoint is live.

## Run One Endpoint At A Time

If your GPU resource has one GPU, edit `precision_matrix.example.csv` and keep only the row for the endpoint you are currently running set to `enabled=true`. Leave other rows as `enabled=false`. Run the notebook/benchmark, save results, then switch the CSV row for the next model or precision profile.

The default matrix enables only the hosted NVIDIA API row. Local rows are disabled until you start those servers.

If the hosted row is enabled but `NVIDIA_API_KEY` is missing, the matrix runner skips it and records `skip_reason=missing NVIDIA_API_KEY`.

The matrix has a `measurement_mode` column. Keep Nano rows as `strict`. Use `lenient` for GPT-OSS when you want to keep completed responses, total latency, provider token usage, and E2E throughput even if this client does not capture visible streamed output for every prompt. The visualization should then use the coverage columns to show which TTFT/decode metrics are actually measured.

After you run several one-endpoint benchmarks, combine them:

```bash
python3 combine_results.py
```

Combined outputs:

```text
results/combined-summary.csv
results/combined-details.csv
```

## Hosted NVIDIA API

In a notebook cell:

```python
import os
os.environ["NVIDIA_API_KEY"] = "your_api_key_here"
```

Then run:

```bash
python3 benchmark_nano.py \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 3 \
  --concurrency 1
```

## Local OpenAI-Compatible Endpoint

For a local endpoint without an API key:

```bash
python3 benchmark_nano.py \
  --base-url http://localhost:8004/v1 \
  --model openai/gpt-oss-120b \
  --precision-label MXFP4 \
  --prompt-dir of1-testprompts \
  --max-tokens 1024 \
  --runs 3 \
  --concurrency 1 \
  --allow-missing-api-key \
  --omit-chat-template-kwargs
```
