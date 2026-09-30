# Cascade one-click miner image — `cascade mine` + its web UI + the agent skill.
#
# A MINER-side image. It never holds operator credentials, and nothing secret
# is baked into it: the wallet is a read-only mount, and LIUM_API_KEY /
# ANTHROPIC_API_KEY come in through the environment at `docker run`.
# It is NOT the trainer-worker image (deploy/Dockerfile, contract-pinned by digest),
# but it installs the same torch build and the same generator-runtime pins, so
# `cascade verify` / `cascade score` see the numeric stack a pod does.
#
# Build:
#   docker build -f deploy/miner.Dockerfile -t cascade-miner .
# Run (GPU strongly recommended; CPU works, slowly):
#   docker run --gpus all -p 8765:8765 -v "$PWD/mine:/work" \
#       -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
#       -e LIUM_API_KEY -e ANTHROPIC_API_KEY cascade-miner
#   → open the http://localhost:8765/?token=… URL it prints.
# Headless:
#   docker run --gpus all -v "$PWD/mine:/work" cascade-miner mine --iterations 30
#
# Full guide: docs/ONE_CLICK_MINING.md.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /opt/cascade
COPY . /opt/cascade

# Same pinned torch build as the worker image (numerics), then the miner extras:
# train (score), hippius (fetch/deploy), chain (commit/submit — bittensor pinned).
# TORCH_INDEX is overridable (e.g. https://download.pytorch.org/whl/cpu for a
# CPU-only image); the default matches deploy/Dockerfile.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN uv venv --python 3.11 /opt/cascade/.venv \
    && uv pip install --python /opt/cascade/.venv/bin/python \
        torch==2.4.1 --index-url "$TORCH_INDEX" \
    && uv pip install --python /opt/cascade/.venv/bin/python -e '.[train,hippius,chain]'

# Generator-runtime allowlist, pinned exactly as in deploy/Dockerfile, so the
# determinism check can import the king (numba, scikit-learn, …) the same way a pod does.
RUN uv pip install --python /opt/cascade/.venv/bin/python \
        numpy==2.4.6 pandas==3.0.3 pyarrow==25.0.0 pyyaml==6.0.3 \
        scipy==1.17.1 statsmodels==0.14.6 numba==0.66.0 \
        scikit-learn==1.9.0 gpytorch==1.15.2 networkx==3.6.1

# Claude Code for the `agent` proposer (native installer; no Node needed).
# Skip it with --build-arg INSTALL_CLAUDE=0 for a tune-only image.
ARG INSTALL_CLAUDE=1
RUN if [ "$INSTALL_CLAUDE" = "1" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash \
        && ln -sf /root/.local/bin/claude /usr/local/bin/claude; \
    fi

# The agent skill, user-level: the agent runs inside candidate dirs under /work,
# outside this repo, so a project-level .claude/ would not be found.
RUN mkdir -p /root/.claude/skills \
    && cp -r /opt/cascade/.claude/skills/cascade-mine /root/.claude/skills/cascade-mine

ENV PATH=/opt/cascade/.venv/bin:$PATH \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CASCADE_MINE_WORKDIR=/work/mine-run \
    PORT=8765

COPY deploy/miner-entrypoint.sh /usr/local/bin/cascade-miner
RUN chmod +x /usr/local/bin/cascade-miner

VOLUME ["/work"]
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/cascade-miner"]
CMD ["ui"]
