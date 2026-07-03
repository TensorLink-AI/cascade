# cascade-bench — the benchmark sidecar as one reproducible artifact:
# pinned python env (uv.lock, incl. the fragile gift-eval pin set and the
# timebench git dep) + the three benchmark datasets FROZEN at the revisions
# pinned in cascade_benchmark/datasets.py. Numbers from this image are
# comparable across rounds, machines, and time.
#
# Build (from the benchmarks/ directory):
#   docker build -t cascade-bench:<tag> .
#
# Run (checkpoint and output mounted; GPU optional):
#   docker run --rm --gpus all -v /path/to/ckpt:/ckpt:ro -v /path/to/out:/out \
#       cascade-bench:<tag> /ckpt /out/report.json --device cuda --batch-size 512
#
# Layer order is deliberate:
#   1. deps      — invalidated only by pyproject.toml/uv.lock
#   2. data      — invalidated only by datasets.py (where the revisions live)
#   3. code      — cheap, changes freely without re-downloading 4.4 GB
FROM python:3.11-slim

# git: uv needs it to install the timebench git dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.10.4

WORKDIR /app
ENV UV_LINK_MODE=copy

# 1. dependencies (locked; project itself not yet installed)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra time

# 2. benchmark data at the PINNED revisions — datasets.py is the single source
#    of truth for which revisions, so this layer rebuilds exactly when they bump.
#    (One-liner, not a heredoc: the legacy builder silently no-ops RUN heredocs.)
COPY cascade_benchmark/datasets.py /tmp/bench_datasets.py
RUN .venv/bin/python -c "\
import importlib.util, sys; \
s = importlib.util.spec_from_file_location('bd', '/tmp/bench_datasets.py'); \
m = importlib.util.module_from_spec(s); \
sys.modules['bd'] = m; \
s.loader.exec_module(m); \
[print('downloading', k, '@', m.DATASETS[k].revision, flush=True) or m.download_suite(k, '/data/' + k) for k in m.DATASETS]" \
    && du -sh /data/* && test -d /data/gift-eval && test -d /data/boom && test -d /data/time

# 3. the scorer itself
COPY cascade_benchmark ./cascade_benchmark
RUN uv sync --frozen --extra time

# suites read their data locations from these env vars (no --data-dir needed)
ENV GIFT_EVAL=/data/gift-eval \
    BOOM=/data/boom \
    CASCADE_BENCH_TIME_DATASET=/data/time

ENTRYPOINT ["uv", "run", "--frozen", "cascade-benchmark"]
