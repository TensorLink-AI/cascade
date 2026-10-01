# Cascade miner image: one pinned environment for any mining strategy.
#
# Two targets share one base:
#
#   toolbox  (DEFAULT)  the miner CLI in the exact numeric stack a training pod
#                       runs: verify, score, fetch, submit, round/queue/duel, and
#                       `cascade mine` with YOUR strategy (--proposer cmd).
#                       Bring your own generator + strategy at /work. No UI, no LLM.
#   oneclick            toolbox + Claude Code + the cascade-mine skill, starting
#                       the web UI (cascade mine-ui) by default. Press Start.
#
# A MINER-side image: no operator credentials, and nothing secret is baked in.
# The wallet is a read-only mount; LIUM_API_KEY / HIPPIUS_* / ANTHROPIC_API_KEY
# come in through the environment (or /work/.env) at `docker run`. It is NOT the
# trainer-worker image (deploy/Dockerfile, contract-pinned by digest), but it
# installs the same torch build and generator-runtime pins, so `cascade verify` /
# `cascade score` see the numeric stack a pod does.
#
# Build:
#   docker build -f deploy/miner.Dockerfile -t cascade-miner .                     # toolbox
#   docker build -f deploy/miner.Dockerfile --target oneclick -t cascade-miner:oneclick .
#   CPU-only: --build-arg BASE_IMAGE=ubuntu:22.04 \
#             --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Use: docs/MINER_DOCKER.md (toolbox), docs/ONE_CLICK_MINING.md (oneclick).

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# Where the uv binary comes from. Docker Hub mirror: --build-arg UV_IMAGE=astral/uv:latest
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- #
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /bin/uv

WORKDIR /opt/cascade

# Layer order = cache order. Heavy, rarely-changing layers first: torch, the
# pinned generator runtime, then the project's dependencies resolved from
# pyproject.toml ALONE. Only then the source, so a code change rebuilds one
# thin layer instead of re-downloading the CUDA stack.
#
# Same pinned torch build as the worker image (numerics). TORCH_INDEX is
# overridable (e.g. .../whl/cpu for a CPU-only image); the default matches
# deploy/Dockerfile.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN uv venv --python 3.11 /opt/cascade/.venv \
    && uv pip install --python /opt/cascade/.venv/bin/python \
        torch==2.4.1 --index-url "$TORCH_INDEX"

# Generator-runtime allowlist, pinned exactly as in deploy/Dockerfile, so your
# generator (and the king: numba, scikit-learn, …) imports the way it does on a pod.
ARG RUNTIME_PINS="numpy==2.4.6 pandas==3.0.3 pyarrow==25.0.0 pyyaml==6.0.3 \
scipy==1.17.1 statsmodels==0.14.6 numba==0.66.0 scikit-learn==1.9.0 \
gpytorch==1.15.2 networkx==3.6.1"
RUN uv pip install --python /opt/cascade/.venv/bin/python $RUNTIME_PINS

# The miner extras' dependencies: train (score), hippius (fetch/deploy), chain
# (commit/submit; bittensor pinned). --no-sources stops pyproject's
# [tool.uv.sources] from re-resolving torch against the cu124 index: the torch
# installed above already satisfies the ==2.4.1 pin. The runtime pins are
# restated so a dependency that wants another numpy/scipy fails the build
# instead of silently moving a pin.
COPY pyproject.toml /opt/cascade/pyproject.toml
RUN uv pip install --python /opt/cascade/.venv/bin/python --no-sources \
        -r pyproject.toml --extra train --extra hippius --extra chain \
        torch==2.4.1 $RUNTIME_PINS

# The source, last. --no-deps: everything it needs is already in place.
COPY . /opt/cascade
RUN uv pip install --python /opt/cascade/.venv/bin/python --no-sources --no-deps -e .

ARG BUILD_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/TensorLink-AI/cascade" \
      org.opencontainers.image.revision="${BUILD_SHA}" \
      org.opencontainers.image.title="cascade-miner"
ENV CASCADE_MINER_BUILD_SHA=${BUILD_SHA}

ENV PATH=/opt/cascade/.venv/bin:$PATH \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CASCADE_MINE_WORKDIR=/work/mine-run \
    PORT=8765

COPY deploy/miner-entrypoint.sh /usr/local/bin/cascade-miner
RUN chmod +x /usr/local/bin/cascade-miner

WORKDIR /work
VOLUME ["/work"]
ENTRYPOINT ["/usr/local/bin/cascade-miner"]

# --------------------------------------------------------------------------- #
FROM base AS oneclick

# Claude Code for the `agent` proposer (native installer; no Node needed).
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && ln -sf /root/.local/bin/claude /usr/local/bin/claude

# The agent skill, user-level: the agent runs inside candidate dirs under /work,
# outside this repo, so a project-level .claude/ would not be found.
RUN mkdir -p /root/.claude/skills \
    && cp -r /opt/cascade/.claude/skills/cascade-mine /root/.claude/skills/cascade-mine

EXPOSE 8765
CMD ["ui"]

# --------------------------------------------------------------------------- #
# Last stage = what a plain `docker build` produces.
FROM base AS toolbox
CMD ["help"]
