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
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /bin/uv

WORKDIR /root/cascade
COPY . /root/cascade

# Pinned Python + the worker's extras, installed FROM THE LOCK. `uv sync
# --frozen` installs exactly uv.lock (torch 2.4.1+cu124 from the cu124 index
# pinned in pyproject [tool.uv.sources]) and never re-resolves. Images up to
# worker-v2.1.0-rc5 / mkt-rc3 used `uv pip install -e '.[...]'`, which
# re-resolved every unpinned package at build time: ~45 packages drifted off
# the lock, including hippius_hub 0.7.2 against a locked 0.5.1, whose chunked-v2
# pushes no 0.5.1 validator could read (2026-11-28, #998). Bump deliberately: a
# lock change can shift numerics and break byte-exact re-derivation.
#   train/hippius/marketcast  the worker; wandb  per-step log mirror (no-op
#   unless enabled); generation  the generator-runtime allowlist (chain.toml
#   [dependencies]: numpy/scipy/pandas/numba/statsmodels/scikit-learn/gpytorch/
#   networkx/pyarrow, pinned in pyproject); pool-forge  pyyaml/pyarrow/pandas.
RUN uv venv --python 3.11 /root/cascade/.venv \
    && .venv/bin/python scripts/check_tls_trust.py --system-only \
    && UV_LINK_MODE=copy uv sync --frozen --no-dev \
        --extra train --extra hippius --extra wandb --extra marketcast \
        --extra generation --extra pool-forge

# Build-time guard: the venv must EQUAL uv.lock (every installed distribution
# at its locked version, nothing the lock does not know, timestral at its locked
# commit). Any drift fails the build. On success it writes the venv's lock
# marker, which the provisioner's health gate (cascade/provision/health.py
# lock_match) requires before it enforces the same check on a booted pod.
RUN .venv/bin/python -P -m cascade.shared.lockdrift --lock uv.lock \
        --write-marker /root/cascade/.venv/.cascade-lock-verified

RUN .venv/bin/python scripts/check_tls_trust.py

# The baked chain.toml is a BUILD-TIME SNAPSHOT, kept only so a pod can boot
# and run legs before the orchestrator pushes config. It is NOT authoritative:
# its [training] train_image_digest is the pin as of the build — necessarily the
# PREVIOUS image's digest, never this image's own (an image cannot know its
# post-push digest; 2026-08-28 incident: freshly rotated images self-reported
# the old pin and final workers refused with TrainImageMismatch). The pin cannot
# be blanked here either — train_image_digest folds into contract_digest, so a
# worker booting from a pin-blanked snapshot would compute a different contract
# than the orchestrator. Instead the snapshot gets a loud header, and the
# provisioner delivers truth at gate time: the deployed chain config
# (chain.deployed.toml) plus the launch-injected CASCADE_TRAIN_IMAGE_DIGEST.
#
# HARD RULE for this image: NEVER bake CASCADE_TRAIN_IMAGE_DIGEST (no ENV, no
# /etc/environment, no .bashrc, no sentinel). Any baked value is a stale
# self-claim; absence makes the worker fail with an explicit "provisioner must
# inject it" error instead of a confusing mismatch.
ARG BUILD_REF=unknown
ARG BUILD_SHA=unknown
RUN for f in chain.toml chain.testnet.toml; do \
        printf '%s\n' \
"# =============================================================================" \
"# BUILD-TIME SNAPSHOT — baked into the cascade-worker image at build time" \
"# (ref ${BUILD_REF}, commit ${BUILD_SHA}). NOT authoritative on a live pod:" \
"# the provisioner-pushed deployed chain config is the source of truth." \
"# In particular, [training] train_image_digest below is the pin AS OF THE" \
"# BUILD — for a freshly published image that is always the PREVIOUS image's" \
"# digest, never this image's own (an image cannot know its post-push digest)." \
"# Final-mode workers must run from the pushed config plus the launch-injected" \
"# CASCADE_TRAIN_IMAGE_DIGEST; this file exists only so a pod can boot before" \
"# config delivery." \
"# =============================================================================" \
        | cat - "$f" > "$f.tmp" && mv "$f.tmp" "$f"; \
    done
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
