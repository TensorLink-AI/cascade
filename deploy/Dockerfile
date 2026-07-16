# Cascade trainer-worker image — the DATA plane only.
#
# This image runs ONE thing: `cascade.trainer.worker`, dispatched over SSH by the
# orchestrator (see cascade/trainer/remote.py). It never holds the bittensor
# wallet and bakes NO secrets — the wallet + signing stay on the trusted CPU
# orchestrator; registry/S3 creds arrive at launch (env) or per-dispatch
# (hosts.toml forward_env).
#
# Why an image (not `pip install` on each boot): a pinned torch/CUDA/cuDNN stack
# is part of the reproducibility contract. Every pod — on Shadeform, Targon, or
# Lium — and every audit re-run then shares the identical numeric stack, so the
# only variable left is the GPU SKU (pin that with [training] expected_gpu).
#
# Build (tag immutably; treat the digest as part of the contract):
#   docker build -f deploy/Dockerfile -t <registry>/cascade-worker:<tag> .
#   docker push <registry>/cascade-worker:<tag>

# cuDNN runtime base — cuDNN version is pinned here for deterministic kernels.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# Key-only root SSH; no passwords. The orchestrator's pubkey is injected at
# launch by the entrypoint (never baked in).
RUN printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\n' \
        > /etc/ssh/sshd_config.d/cascade.conf

# uv for a reproducible pinned Python + deps (matches the repo's uv toolchain).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /root/cascade
COPY . /root/cascade

# Pinned Python + the worker's extras. Torch is pinned to a specific CUDA build
# so the numeric stack is identical on every pod. Bump deliberately — a torch
# change can shift numerics and break byte-exact re-derivation of older rounds.
RUN uv venv --python 3.11 /root/cascade/.venv \
    && uv pip install --python /root/cascade/.venv/bin/python \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /root/cascade/.venv/bin/python -e '.[train,hippius]'

# Generator-runtime allowlist (chain.toml [dependencies]) — baked and PINNED so
# every miner generator finds the identical numeric stack on every pod.
# worker-v0.3.0 shipped without these: every generator importing pandas failed
# heats with generator_import_failed (2026-07-15; pods were hand-patched).
# Pins match uv.lock where present, else a frozen `uv pip compile` resolution
# against it (2026-07-16). torch stays the cu124 build installed above.
RUN uv pip install --python /root/cascade/.venv/bin/python \
        numpy==2.4.6 pandas==3.0.3 pyarrow==25.0.0 pyyaml==6.0.3 \
        scipy==1.17.1 statsmodels==0.14.6 numba==0.66.0 \
        scikit-learn==1.9.0 gpytorch==1.15.2 networkx==3.6.1

# The public contract (chain.toml) is baked in so every pod trains the identical
# recipe; the worker cd's here and load_chain_config finds it. Override per-host
# with hosts.toml `chain_toml` only if a pod must differ.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
