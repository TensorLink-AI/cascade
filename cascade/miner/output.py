"""One output contract for every ``cascade`` command — humans and agents.

Human mode (the default) prints exactly what the commands always printed:
progress and results on stdout, problems on stderr.

``--json`` (or ``CASCADE_JSON=1``) is for scripts and coding agents: stdout
carries exactly ONE JSON document when the command ends, and every human line
goes to stderr instead, so ``cascade <cmd> --json | jq`` always parses::

    {"schema": "cascade-cli/1", "command": "verify", "ok": false,
     "exit_code": 1, "data": {...},
     "error": {"code": "verify_failed", "message": "..."}}

``error`` is null on success. ``error.code`` is a stable snake_case string an
agent can branch on; ``message`` is for people. Commands never prompt, so a
non-interactive caller can never hang.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

SCHEMA = "cascade-cli/1"

# Exit codes are part of the CLI contract (docs/MINER.md, AGENTS.md).
OK = 0
REJECTED = 1          # checked and refused: verify failed, intake said no, ...
USAGE = 2             # bad flags or local input; nothing was sent
NETWORK = 3           # chain / intake / storage unreachable
REGISTRY = 4          # generator upload or fetch failed
NEEDS_CONFIRM = 5     # a paid action without --yes (or over --max-spend); nothing was sent

EXIT_CODES = {OK: "ok", REJECTED: "rejected", USAGE: "usage", NETWORK: "network",
              REGISTRY: "registry", NEEDS_CONFIRM: "needs_confirmation"}


def json_default() -> bool:
    return os.environ.get("CASCADE_JSON", "").strip().lower() in ("1", "true", "yes")


class Reporter:
    """Collects a command's result; prints human lines as they happen."""

    def __init__(self, command: str = "", *, json_mode: bool = False):
        self.command = command
        self.json_mode = json_mode
        self.data: dict = {}
        self.error: dict | None = None

    def info(self, text: str = "") -> None:
        """A human line: stdout normally, stderr under --json."""
        print(text, file=sys.stderr if self.json_mode else sys.stdout)

    def warn(self, text: str) -> None:
        print(text, file=sys.stderr)

    def result(self, **fields) -> None:
        """Fields for the JSON document's ``data`` (ignored in human mode)."""
        self.data.update(fields)

    def fail(self, code: str, message: str, exit_code: int = REJECTED, **fields) -> int:
        """Record the error, print the message to stderr, return ``exit_code``."""
        self.error = {"code": code, "message": message, **fields}
        print(message, file=sys.stderr)
        return exit_code

    def finish(self, exit_code: int) -> int:
        if self.json_mode:
            if exit_code != OK and self.error is None:
                self.error = {"code": EXIT_CODES.get(exit_code, "failed"),
                              "message": f"exit code {exit_code}"}
            payload = {"schema": SCHEMA, "command": self.command,
                       "ok": exit_code == OK, "exit_code": exit_code,
                       "data": self.data, "error": self.error}
            try:
                encoded = json.dumps(payload, sort_keys=True, default=str, allow_nan=False)
            except ValueError:
                exit_code = NETWORK
                encoded = json.dumps({"schema": SCHEMA, "command": self.command,
                                      "ok": False, "exit_code": exit_code, "data": {},
                                      "error": {"code": "non_finite_result", "message":
                                                "Result contains non-finite values; no usable result was established."}},
                                     sort_keys=True, allow_nan=False)
            print(encoded)
        return exit_code


def reporter(args) -> Reporter:
    """The command's reporter; a plain human one for hand-built Namespaces."""
    out = getattr(args, "_reporter", None)
    if out is None:
        out = Reporter(getattr(args, "cmd", "") or "")
        with contextlib.suppress(AttributeError):
            args._reporter = out
    return out
