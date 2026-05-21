FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates docker.io nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir PyYAML

WORKDIR /bench
COPY benchmark_arco.py /bench/benchmark_arco.py
COPY run_arco_nim_experiments.py /bench/run_arco_nim_experiments.py
COPY arco_nim_models.example.csv /bench/arco_nim_models.example.csv

ENTRYPOINT ["python", "/bench/run_arco_nim_experiments.py"]
CMD ["--arco-repo", "/arco", "--matrix", "/bench/arco_nim_models.example.csv", "--output", "/results/arco-all-experiments.csv", "--summary-output", "/results/arco-by-prompt-summary.csv", "--cache-root", "/opt/dlami/nvme/nim-cache", "--runs", "3", "--concurrency", "1", "--continue-on-error"]
