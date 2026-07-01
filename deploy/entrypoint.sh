#!/usr/bin/env bash
# Trainer-worker pod entrypoint: make the container an SSH endpoint the
# orchestrator can dispatch `cascade.trainer.worker` into. No wallet, no secrets.
set -euo pipefail

# Fresh host keys per pod (not baked into the image layer).
ssh-keygen -A

# Install the orchestrator's public key — passed at launch via $SSH_PUBKEY,
# never baked in. This is the ONLY key allowed to dispatch into the pod.
if [[ -n "${SSH_PUBKEY:-}" ]]; then
    install -d -m 700 /root/.ssh
    printf '%s\n' "$SSH_PUBKEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
else
    echo "WARNING: SSH_PUBKEY unset — the orchestrator cannot dispatch to this pod" >&2
fi

# Registry/S3 creds are NOT injected here. Prefer hosts.toml `forward_env`, which
# passes them inline per-dispatch (never persisted on the pod). If you instead
# set them as launch env, expose them to the ssh command shell yourself.

exec /usr/sbin/sshd -D -e
