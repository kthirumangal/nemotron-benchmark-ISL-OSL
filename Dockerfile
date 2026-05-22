FROM docker:27-cli

RUN apk add --no-cache python3 py3-yaml nodejs

WORKDIR /bench
COPY benchmark_arco.py /bench/benchmark_arco.py
COPY run_arco_nim_experiments.py /bench/run_arco_nim_experiments.py
COPY arco_nim_models.example.csv /bench/arco_nim_models.example.csv

ENTRYPOINT ["python3", "/bench/run_arco_nim_experiments.py"]
CMD ["--arco-repo", "/arco", "--matrix", "/bench/arco_nim_models.example.csv", "--output", "/results/arco-all-experiments.csv", "--summary-output", "/results/arco-by-prompt-summary.csv", "--category-summary-output", "/results/arco-category-summary.csv", "--model-summary-output", "/results/arco-model-summary.csv", "--cache-root", "/opt/dlami/nvme/nim-cache", "--runs", "3", "--concurrency", "1", "--continue-on-error"]
