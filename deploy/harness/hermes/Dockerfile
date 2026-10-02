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

ARG HERMES_VERSION="0.19.0"
RUN pip install --no-cache-dir "hermes-agent${HERMES_VERSION:+==$HERMES_VERSION}"

COPY skills/cascade-operator /root/.hermes/skills/cascade-operator
COPY OPERATOR_PROMPT.md /opt/operator/OPERATOR_PROMPT.md
COPY operator-loop.sh /usr/local/bin/operator-loop
RUN chmod +x /usr/local/bin/operator-loop

WORKDIR /operator
CMD ["operator-loop"]
