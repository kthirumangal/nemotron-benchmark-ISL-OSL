# Brev Quickstart

Brev may open directly into a notebook environment. You can use either terminal commands or notebook cells.

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

If Brev uses `/home/ubuntu/.venv`:

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

If your Brev instance has one GPU, edit `precision_matrix.example.csv` and keep only the row for the endpoint you are currently running set to `enabled=true`. Leave other rows as `enabled=false`. Run the notebook/benchmark, save results, then switch the CSV row for the next model or precision profile.

The default matrix enables only the hosted NVIDIA API row. Local rows are disabled until you start those servers.

If the hosted row is enabled but `NVIDIA_API_KEY` is missing, the matrix runner skips it and records `skip_reason=missing NVIDIA_API_KEY`.

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
