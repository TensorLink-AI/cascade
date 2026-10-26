"""Bounded, non-authoritative generation counters; never a corpus/verdict input."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)
PROGRESS_NAME = "progress.json"
PHASES = frozenset({"waiting_cpu", "booting_cpu", "generating", "hashing", "uploading",
                    "uploaded", "ready", "reused", "generator_fault", "infra_fault"})
TERMINAL = frozenset({"ready", "reused", "generator_fault", "infra_fault"})
NUMBERS = ("points", "target_points", "n_series", "elapsed_seconds", "points_per_second",
           "cpu_seconds", "active_workers", "workers", "shards", "stored_bytes", "raw_bytes", "compression_ratio",
           "uploaded_bytes", "upload_total_bytes", "uploaded_shards", "upload_total_shards", "percent")
IDENTITY = ("round_id", "hotkey", "generator_ref", "attempt_id", "submission_attempt_id")


def public_progress(doc):
    if not isinstance(doc, dict) or doc.get("stage") not in PHASES:
        return None
    out = {k: str(doc[k])[:256] for k in IDENTITY if isinstance(doc.get(k), str)}
    for key in NUMBERS:
        value = doc.get(key)
        if type(value) in (int, float):
            try:
                if math.isfinite(value) and 0 <= value <= 1e20:
                    out[key] = value
            except OverflowError:
                pass
    if doc.get("stop_reason") in {"target", "window", "exhausted", "generator_fault", "infra_fault"}:
        out["stop_reason"] = doc["stop_reason"]
    out["stage"] = doc["stage"]
    try:
        stamp = datetime.fromisoformat(str(doc.get("as_of", "")).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return None
    except (ValueError, OverflowError):
        return None
    out.update(as_of=stamp.isoformat(), source_kind="reported_generation_not_consensus")
    return out


def atomic_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(doc, stream, allow_nan=False, sort_keys=True)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class CpuProgress:
    """Local snapshot plus a latest-only upload worker. No account credential."""

    def __init__(self, path, *, attempt_id="", url=None):
        self.path, self.attempt_id, self.url = Path(path), attempt_id, url
        self.latest = {}
        self.pending = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.thread = None
        if url:
            self.thread = threading.Thread(target=self._upload, daemon=True, name="generation-progress-upload")
            self.thread.start()

    def __call__(self, event):
        self.latest.update(event)
        self.latest.update(attempt_id=self.attempt_id, as_of=datetime.now(UTC).isoformat())
        payload = json.dumps(self.latest, allow_nan=False, sort_keys=True).encode()
        try:
            atomic_json(self.path, self.latest)
        except OSError:
            log.warning("local generation progress persistence unavailable")
        if self.url:
            with contextlib.suppress(queue.Empty):
                self.pending.get_nowait()
            with contextlib.suppress(queue.Full):
                self.pending.put_nowait(payload)

    def _upload(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                payload = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                request = Request(self.url, data=payload, method="PUT",
                                  headers={"Content-Length": str(len(payload))})
                with urlopen(request, timeout=10) as response:  # noqa: S310 — scoped operator PUT
                    if response.status >= 300:
                        raise OSError("progress upload refused")
            except Exception:  # noqa: BLE001 — progress loss never fails generation; no URL/error text
                log.warning("generation progress upload unavailable")

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=0.5)


class WandbProgress:
    """Latest-only W&B delivery off the work path; no key reaches the CPU VM."""

    def __init__(self, cfg, identity):
        self.cfg, self.identity = cfg, identity
        self.pending = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True, name="generation-wandb")
        self.thread.start()

    def emit(self, doc):
        with contextlib.suppress(queue.Empty):
            self.pending.get_nowait()
        with contextlib.suppress(queue.Full):
            self.pending.put_nowait(dict(doc))

    def _run(self):
        from ..trainer.wandb_sink import open_generation_run

        sink = open_generation_run(self.cfg, **self.identity)
        last = time.monotonic()
        while True:
            try:
                doc = self.pending.get(timeout=1)
            except queue.Empty:
                if time.monotonic() - last > 24 * 3600:
                    break
                continue
            last = time.monotonic()
            if sink:
                sink.emit({**{f"generation/{k}": doc[k] for k in NUMBERS if k in doc},
                           "generation/stage": doc["stage"],
                           **({"generation/stop_reason": doc["stop_reason"]} if "stop_reason" in doc else {})})
            if doc["stage"] in TERMINAL:
                break
        if sink:
            sink.finish()
