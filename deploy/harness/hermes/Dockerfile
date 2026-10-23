# The gauntlet OPERATOR: Hermes Agent on any OpenAI-compatible endpoint (Chutes,
# SayGM, …), woken on a timer to read the gauntlet's status and steer it.
#
# It mounts ONLY <workdir>/operator (status.json in; DIRECTIVES.md, reports/,
# STOP out). It holds no Lium key, no wallet, no pool data, and cannot promote
# or submit anything — the judge's code does that.
#
# The Hermes CLI surface used here (`pip install hermes-agent`, a `custom`
# provider with model.api_key in ~/.hermes/config.yaml, one-shot
# `hermes chat -Q -s <skill> -q`) was checked against hermes-agent 0.19.0,
# hence the pin. Re-check operator-loop.sh before bumping it.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip --cert /etc/ssl/certs/ca-certificates.crt install --no-cache-dir --upgrade "pip>=26.2.2" "certifi>=2026.10.19"

ARG HERMES_VERSION="0.19.0"
# Hermes 0.19.0 pins the compromised CA data. Preserve its tested CLI and
# override only that certificate-data pin until a patched Hermes release ships.
# pip check will report this one intentional upstream metadata conflict.
RUN pip install --no-cache-dir "hermes-agent${HERMES_VERSION:+==$HERMES_VERSION}" \
    && pip install --no-cache-dir --no-deps --upgrade certifi==2026.10.19
COPY check_tls_trust.py /tmp/check_tls_trust.py
RUN python /tmp/check_tls_trust.py

COPY skills/cascade-operator /root/.hermes/skills/cascade-operator
COPY OPERATOR_PROMPT.md /opt/operator/OPERATOR_PROMPT.md
COPY operator-loop.sh /usr/local/bin/operator-loop
RUN chmod +x /usr/local/bin/operator-loop

WORKDIR /operator
CMD ["operator-loop"]
