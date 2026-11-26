"""Lane-B public bench rows, benched on an EVAL POD — never on the trainer box (#909).

Lane B pairs the king's, every rebased candidate's (and, with suite targets,
the anchor's) four-suite public rows (``[lane_b] require_public``). The
trainer box is CPU-only with no bench data (and must never run the sidecar:
an attempt OOM'd a co-hosted validator on testnet), so every lane-B bench
runs through the validator's eval-offload path
(:func:`cascade.validator.eval_offload.run_bench_via_host_many`) on the pod
named by ``--lane-b-bench-hosts`` (an eval-hosts TOML, re-read per bench):

* **Pairing.** King, candidates and anchor are all benched here, on the same
  pod, with the same settings — exactly as they were all benched on the
  trainer before. The validators only pair the signed rows
  (:func:`cascade.eval.public_guards.pair_public_rows`: ``status`` ok, equal
  ``data_revision``, shared item ids); nothing about the bench host enters a
  verdict, so moving the bench changes no consensus rule.
* **Off the settlement path.** A full four-suite sweep is ~1 h per
  checkpoint, longer than the rolling scheduler lets a lane-B settlement hold
  lane A (``LANE_B_SETTLE_MAX_SECONDS``). So :meth:`LaneBBench.rows` never
  benches inline: it returns the cached rows of that checkpoint's CONTENT, or
  queues the bench on one background worker and returns None — the settlement
  then fails fast (its finished legs wait for the next one, as on any lane-B
  I/O failure) and lane A is not held. :meth:`LaneBBench.prebench` queues the
  king as soon as a leg has finished, so it is usually ready by settlement.
  Rows are cached on disk by content digest (a restart re-benches nothing; a
  candidate rebased again onto the same king at the next settlement has the
  same bytes, hence the same key).
* **Shared pod.** The launch config points ``--lane-b-bench-hosts`` at lane
  A's bench pod (``targon-eval.toml``). There is NO pod-side lock: this
  assumes lane A's sweeps are quiet once the other lanes launch (owner
  2026-11-25); add a lock if lane A resumes. One lane-B bench runs at a time
  (one worker).
* **Fail closed.** No hosts file, no host in it, a lock that never frees, a
  failed or incomplete sweep: nothing is cached, an ERROR is logged, and the
  bench is re-queued by the next settlement. Nothing ever runs locally.

Torch-free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("cascade.trainer.lane_b_bench")

# Lane B's sidecar allowance per checkpoint (a full four-suite sweep is ~1 h).
LANE_B_BENCH_TIMEOUT_S = 2 * 3600 + 1800
ROWS_SCHEMA = "lane-b-bench-rows/1"


def content_key(ckpt_dir: Path, settings: dict) -> str:
    """sha256 over the bench files' bytes (the eval-offload upload set) and
    the bench settings — the cache key of a checkpoint's rows."""
    from ..validator.eval_offload import bench_upload_files

    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    files = bench_upload_files(ckpt_dir)
    if not files:
        raise FileNotFoundError(f"{ckpt_dir}: no bench files")
    for f in files:
        h.update(f.name.encode() + b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def rows_complete(rows: dict | None, suites: tuple[str, ...]) -> bool:
    """Every suite present with status ok (only such rows are cached)."""
    from ..eval.benchmarks import ITEM_SUITE_KEY

    if not isinstance(rows, dict):
        return False
    return all(isinstance(rows.get(ITEM_SUITE_KEY[s]), dict)
               and rows[ITEM_SUITE_KEY[s]].get("status") == "ok" for s in suites)


class LaneBBench:
    """Eval-pod lane-B bench with a content-keyed row cache and one worker."""

    def __init__(self, *, hosts_path: Path | str | None, cache_dir: Path | str,
                 eval_cfg, fetch: Callable[[str], Path] | None = None,
                 host_fn: Callable[[], object] | None = None,
                 bench_fn: Callable[..., list] | None = None,
                 start_worker: bool = True):
        from ..eval.benchmarks import ITEM_SUITES

        self.hosts_path = Path(hosts_path) if hosts_path else None
        self.cache_dir = Path(cache_dir)
        self.suites = tuple(ITEM_SUITES)
        self.settings = {
            "suites": list(self.suites),
            "max_series": int(eval_cfg.cascade_bench_max_series),
            "data_dir": str(eval_cfg.gift_gate_data_dir or ""),
            "num_samples": 100, "batch_size": 512,       # run_item_rows' defaults, as before
        }
        self.workers = getattr(eval_cfg, "bench_workers", "1")
        self._fetch = fetch
        if host_fn is None and self.hosts_path is not None:
            from ..validator.eval_offload import make_eval_host_fn

            host_fn = make_eval_host_fn(self.hosts_path, local_fallback=False)
        self._host_fn = host_fn
        self._bench_fn = bench_fn
        self._lock = threading.Lock()
        self._mem: dict[str, dict] = {}
        self._pending: set[str] = set()
        self._ptr_keys: dict[str, str] = {}
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._start_worker = start_worker

    # ── cache ────────────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def cached(self, key: str) -> dict | None:
        with self._lock:
            if key in self._mem:
                return self._mem[key]
        p = self._path(key)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        rows = doc.get("rows") if isinstance(doc, dict) and doc.get("schema") == ROWS_SCHEMA \
            else None
        if not rows_complete(rows, self.suites):
            return None
        with self._lock:
            self._mem[key] = rows
        return rows

    def _store(self, key: str, rows: dict, label: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps({"schema": ROWS_SCHEMA, "key": key, "label": label,
                                   "settings": self.settings, "rows": rows}),
                       encoding="utf-8")
        tmp.replace(self._path(key))
        with self._lock:
            self._mem[key] = rows

    # ── public ───────────────────────────────────────────────────────────────

    def rows(self, ckpt_dir: Path) -> dict | None:
        """The cached rows of ``ckpt_dir``'s content, or None after queueing
        its bench on the eval pod (never benched here, never inline)."""
        if self.hosts_path is None:
            log.error("lane B bench: no --lane-b-bench-hosts configured — lane-B public rows "
                      "are never benched on the trainer box; settlement fails closed")
            return None
        key = content_key(Path(ckpt_dir), self.settings)
        got = self.cached(key)
        if got is not None:
            return got
        self._enqueue(key, ("dir", str(ckpt_dir)))
        log.info("lane B bench: %s not benched yet — queued on the eval pod (settlement "
                 "retries)", Path(ckpt_dir).name[-32:])
        return None

    def prebench(self, pointer: str) -> None:
        """Queue ``pointer``'s bench (fetched on the worker) unless cached."""
        if self.hosts_path is None or self._fetch is None or not pointer:
            return
        with self._lock:
            key = self._ptr_keys.get(str(pointer))
        if key is not None and self.cached(key) is not None:
            return                       # benched: no re-fetch / re-hash every tick
        self._enqueue(f"ptr:{pointer}", ("ptr", str(pointer)))

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    # ── worker ───────────────────────────────────────────────────────────────

    def _enqueue(self, tag: str, item: tuple[str, str]) -> None:
        with self._lock:
            if tag in self._pending:
                return
            self._pending.add(tag)
            if self._start_worker and (self._worker is None or not self._worker.is_alive()):
                self._worker = threading.Thread(target=self._loop, name="laneb-bench",
                                                daemon=True)
                self._worker.start()
        self._q.put((tag, item))

    def _loop(self) -> None:
        while True:
            tag, item = self._q.get()
            try:
                self.run_one(item)
            except Exception as e:  # noqa: BLE001 — a failed bench is retried by the next ask
                log.error("lane B bench: %s failed: %s", item[1][-40:], e)
            finally:
                with self._lock:
                    self._pending.discard(tag)

    def run_one(self, item: tuple[str, str]) -> dict | None:
        """Bench one queued item now (the worker's body; tests call it)."""
        kind, ref = item
        ckpt = Path(self._fetch(ref)) if kind == "ptr" else Path(ref)
        key = content_key(ckpt, self.settings)
        if kind == "ptr":
            with self._lock:
                self._ptr_keys[ref] = key
        got = self.cached(key)
        if got is not None:
            return got
        host = self._host_fn() if self._host_fn is not None else None
        if host is None:
            log.error("lane B bench: NO eval pod in %s — %s not benched (never on the trainer "
                      "box); lane B cannot settle until a bench host is listed",
                      self.hosts_path, ckpt.name[-32:])
            return None
        from ..eval.benchmarks import item_rows_from_report

        bench = self._bench_fn
        if bench is None:
            from ..validator.eval_offload import run_bench_via_host_many as bench
        s = self.settings
        log.info("lane B bench: sweeping %s on %s (suites=%s, max_series=%d)", ckpt.name[-32:],
                 getattr(host, "name", host), ",".join(self.suites), s["max_series"])
        reports = bench(host, [ckpt], suites=",".join(self.suites),
                        num_samples=s["num_samples"], batch_size=s["batch_size"],
                        max_series=s["max_series"], data_dir=(s["data_dir"] or None),
                        device="cuda", timeout_s=LANE_B_BENCH_TIMEOUT_S, workers=self.workers,
                        evidence_label=f"laneb-{ckpt.name[-24:]}")
        rows = item_rows_from_report(reports[0]) if reports and reports[0] else None
        if not rows_complete(rows, self.suites):
            log.error("lane B bench: %s on %s produced no complete four-suite rows (%s) — not "
                      "cached; re-queued by the next settlement", ckpt.name[-32:],
                      getattr(host, "name", host),
                      {k: (v or {}).get("status") for k, v in (rows or {}).items()} or "none")
            return None
        self._store(key, rows, ckpt.name)
        log.info("lane B bench: %s benched on %s — rows cached", ckpt.name[-32:],
                 getattr(host, "name", host))
        return rows
