"""Worker-side progress record for the orchestrator's progress watchdog.

A detached leg runs with ``CASCADE_PROGRESS_FILE`` naming a file in its run
dir (cascade.trainer.remote). Every ``step`` event the trainer logs (pyramid:
step 1 and every 50 steps) rewrites it atomically with one small JSON object;
the detached poll prints it back, so the orchestrator can measure steps/s on
a pod it otherwise only sees at exit. Best-effort: a write error never
touches the run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .remote import PROGRESS_FILE_ENV


class ProgressFileSink:
    """``emit(record)`` sink: keeps the latest step record in ``path``."""

    def __init__(self, path: str | Path, clock=time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    def emit(self, record: dict) -> None:
        if not isinstance(record, dict) or record.get("event") != "step":
            return
        try:
            out = {"step": int(record.get("step", 0) or 0),
                   "tokens": int(record.get("tokens", 0) or 0),
                   "t": round(float(self.clock()), 3)}
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")),
                           encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001 — observability must never sink training
            pass


def progress_file_sink(env=None) -> ProgressFileSink | None:
    """The sink named by ``$CASCADE_PROGRESS_FILE``, or ``None`` when unset."""
    path = (os.environ if env is None else env).get(PROGRESS_FILE_ENV, "")
    return ProgressFileSink(path) if path else None
