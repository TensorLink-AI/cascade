# cascade-bench — the benchmark sidecar as one reproducible artifact:
# pinned python env (uv.lock, incl. the fragile gift-eval pin set and the
# timebench git dep) + the three benchmark datasets FROZEN at the revisions
# pinned in cascade_benchmark/datasets.py. Numbers from this image are
# comparable across rounds, machines, and time.
#
# Build (from the repository root):
#   docker build -f benchmarks/Dockerfile -t cascade-bench:<tag> .
#
# Run (checkpoint, output and pinned FEV data mounted; GPU optional).
# Preserve the FEV YAML's absolute dataset paths inside the container:
#   docker run --rm --gpus all -v /path/to/ckpt:/ckpt:ro -v /path/to/out:/out \
#       -v /data/bench/fev:/data/bench/fev:ro \
#       cascade-bench:<tag> /ckpt /out/report.json --device cuda --batch-size 512 \
#       --fev-manifest /data/bench/fev/manifest.json
#
# Layer order is deliberate:
#   1. deps      — invalidated only by pyproject.toml/uv.lock
#   2. data      — invalidated only by datasets.py (where the revisions live)
#   3. code      — cheap, changes freely without re-downloading 4.4 GB
FROM python:3.11-slim AS tls-trust

# git: uv needs it to install the timebench git dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /bin/uv
# Upgrade the base image's pip through uv's patched, bundled trust store.
RUN uv pip install --system "pip==26.2.2" "certifi>=2026.10.19"

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY scripts/check_tls_trust.py /usr/local/bin/check_tls_trust.py
RUN python /usr/local/bin/check_tls_trust.py

FROM tls-trust AS benchmark
# 1. dependencies (locked; project itself not yet installed)
COPY benchmarks/pyproject.toml benchmarks/uv.lock benchmarks/README.md ./
RUN uv sync --frozen --no-install-project --extra time --extra fev \
    && .venv/bin/python /usr/local/bin/check_tls_trust.py

# 2. benchmark data at the PINNED revisions — datasets.py is the single source
#    of truth for which revisions, so this layer rebuilds exactly when they bump.
#    (One-liner, not a heredoc: the legacy builder silently no-ops RUN heredocs.)
COPY benchmarks/cascade_benchmark/datasets.py /tmp/bench_datasets.py
RUN .venv/bin/python -c "\
import importlib.util, sys; \
s = importlib.util.spec_from_file_location('bd', '/tmp/bench_datasets.py'); \
m = importlib.util.module_from_spec(s); \
sys.modules['bd'] = m; \
s.loader.exec_module(m); \
[print('downloading', k, '@', m.DATASETS[k].revision, flush=True) or m.download_suite(k, '/data/' + k) for k in m.DATASETS]" \
    && du -sh /data/* && test -d /data/gift-eval && test -d /data/boom && test -d /data/time

# 3. the scorer itself
COPY benchmarks/cascade_benchmark ./cascade_benchmark
RUN uv sync --frozen --extra time --extra fev \
    && .venv/bin/python /usr/local/bin/check_tls_trust.py

# suites read their data locations from these env vars (no --data-dir needed)
ENV GIFT_EVAL=/data/gift-eval \
    BOOM=/data/boom \
    CASCADE_BENCH_TIME_DATASET=/data/time

ENTRYPOINT ["uv", "run", "--frozen", "--extra", "time", "--extra", "fev", "cascade-benchmark"]
