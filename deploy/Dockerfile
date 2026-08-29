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
    && uv pip install --python /root/cascade/.venv/bin/python -e '.[train,hippius,wandb]'
    # [wandb] extra (2026-07-18): lets pod workers mirror per-step training
    # logs to wandb when [wandb] enabled=true AND WANDB_API_KEY is forwarded
    # (provision.toml forward_env). Absent either, the sink no-ops. NOTE: this
    # layer change lands in the NEXT image rotation (digest re-pin + contract
    # recompute) — it is not worth a rotation on its own.

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

# The baked chain.toml is a BUILD-TIME SNAPSHOT, kept only so a pod can boot
# and run heats before the orchestrator pushes config. It is NOT authoritative:
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
"# CASCADE_TRAIN_IMAGE_DIGEST; this file exists only so heats can boot before" \
"# config delivery." \
"# =============================================================================" \
        | cat - "$f" > "$f.tmp" && mv "$f.tmp" "$f"; \
    done
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
