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
* **Pointer path, on the pod** (owner rule: no checkpoint bytes on the
  trainer box). With a :class:`~cascade.trainer.lane_b_offload.LaneBPodRunner`
  (``pod=``), a queued pointer (:meth:`LaneBBench.prebench`,
  :meth:`LaneBBench.rows_remote`) is fetched ON THE POD, its content key is
  computed there by :func:`content_key` (the same key, so the on-disk row
  cache carries over), its bench files are hard-linked into a pinned dir and
  the sidecar sweeps that dir in place (:func:`bench_pinned`: the same
  ``cascade-benchmark`` command line and settings as the upload path). The
  pointer → key map is persisted next to the rows. Without a pod runner a
  pointer bench needs a box-side fetch and is refused unless
  ``CASCADE_LANE_B_LOCAL_EVAL=1`` (the lane-B offload guard).
* **Fail closed.** No hosts file, no host in it, a lock that never frees, a
  failed or incomplete sweep: nothing is cached, an ERROR is logged, and the
  bench is re-queued by the next settlement. Nothing ever runs locally.

Torch-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import shlex
import threading
import uuid
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
                 start_worker: bool = True, pod=None,
                 pinned_bench_fn: Callable[..., dict | None] | None = None):
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
        # the eval-pod runner (LaneBPodRunner): pointer benches stage on the pod
        self._pod = pod
        self._pinned_bench_fn = pinned_bench_fn
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

    def _keys_path(self) -> Path:
        return self.cache_dir / "pointer-keys.json"

    def _ptr_key(self, pointer: str) -> str | None:
        """The content key of a digest-pinned ``pointer`` (immutable bytes ⇒
        an immutable key), from memory or the persisted map."""
        with self._lock:
            key = self._ptr_keys.get(str(pointer))
        if key is not None:
            return key
        try:
            doc = json.loads(self._keys_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        key = doc.get(str(pointer)) if isinstance(doc, dict) else None
        if not isinstance(key, str):
            return None
        with self._lock:
            self._ptr_keys[str(pointer)] = key
        return key

    def note_key(self, pointer: str, key: str) -> None:
        """Record ``pointer``'s content key (memory + the persisted map)."""
        with self._lock:
            self._ptr_keys[str(pointer)] = str(key)
            snap = dict(self._ptr_keys)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                old = json.loads(self._keys_path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
            merged = {**(old if isinstance(old, dict) else {}), **snap}
            # bounded: the newest entries win (insertion order)
            merged = dict(list(merged.items())[-KEYS_KEEP:])
            tmp = self._keys_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(merged), encoding="utf-8")
            tmp.replace(self._keys_path())
        except OSError as e:
            log.warning("lane B bench: pointer key map not persisted: %s", e)

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

    def rows_remote(self, pointer: str, key: str | None = None) -> dict | None:
        """The cached rows of the checkpoint ``pointer`` names (``key``: its
        content key when the pod already computed it), or None after queueing
        its pointer bench (staged on the eval pod; never fetched here)."""
        if self.hosts_path is None:
            log.error("lane B bench: no --lane-b-bench-hosts configured — lane-B public rows "
                      "are never benched on the trainer box; settlement fails closed")
            return None
        if key:
            self.note_key(pointer, key)
        else:
            key = self._ptr_key(pointer)
        got = self.cached(key) if key else None
        if got is not None:
            return got
        self._enqueue(f"ptr:{pointer}", ("ptr", str(pointer)))
        log.info("lane B bench: %s not benched yet — queued on the eval pod (settlement "
                 "retries)", str(pointer)[-32:])
        return None

    def prebench(self, pointer: str) -> None:
        """Queue ``pointer``'s bench (staged on the eval pod by the worker)
        unless cached. Without a pod runner the worker's box-side fetch is
        refused unless ``CASCADE_LANE_B_LOCAL_EVAL=1``."""
        if self.hosts_path is None or not pointer:
            return
        if self._pod is None and self._fetch is None:
            return
        key = self._ptr_key(str(pointer))
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

    def _run_pinned(self, pointer: str) -> dict | None:
        """A pointer bench on the eval pod: stage (fetch + content key + pin)
        there, sweep the pinned dir in place, cache the rows by content."""
        from ..eval.benchmarks import item_rows_from_report

        host, key, pin = self._pod.bench_stage(pointer, self.settings)
        try:
            self.note_key(pointer, key)
            got = self.cached(key)
            if got is not None:
                return got
            s = self.settings
            label = str(pointer).rsplit("@", 1)[-1].replace(":", "-")
            log.info("lane B bench: sweeping %s on %s in place (suites=%s, max_series=%d)",
                     label[-32:], getattr(host, "name", host), ",".join(self.suites),
                     s["max_series"])
            bench = self._pinned_bench_fn or bench_pinned
            report = bench(host, pin, settings=s, workers=self.workers,
                           timeout_s=LANE_B_BENCH_TIMEOUT_S, label=f"laneb-{label[-24:]}")
            rows = item_rows_from_report(report) if report else None
            if not rows_complete(rows, self.suites):
                log.error("lane B bench: %s on %s produced no complete four-suite rows (%s) — "
                          "not cached; re-queued by the next settlement", label[-32:],
                          getattr(host, "name", host),
                          {k: (v or {}).get("status") for k, v in (rows or {}).items()}
                          or "none")
                return None
            self._store(key, rows, label)
            log.info("lane B bench: %s benched on %s — rows cached", label[-32:],
                     getattr(host, "name", host))
            return rows
        finally:
            self._pod.unpin(host, pin)

    def run_one(self, item: tuple[str, str]) -> dict | None:
        """Bench one queued item now (the worker's body; tests call it)."""
        kind, ref = item
        if kind == "ptr" and self._pod is not None:
            return self._run_pinned(ref)
        if kind == "ptr":
            # a box-side checkpoint download: the lane-B offload guard
            from .lane_b_offload import refuse_local

            refuse_local(f"bench prebench download ({str(ref)[-24:]})")
            if self._fetch is None:
                return None
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


KEYS_KEEP = 512


def _run_or_ssh(run):
    from .remote import run_ssh

    return run or (lambda argv, timeout, stdin="": run_ssh(argv, timeout, stdin))


def bench_pinned(host, remote_ckpt: str, *, settings: dict, workers, timeout_s: int,
                 label: str, run=None) -> dict | None:
    """Sweep a checkpoint that is ALREADY on ``host`` (a pinned dir) with the
    upload path's exact sidecar command line
    (:func:`~cascade.validator.eval_offload.build_bench_remote_command`, the
    same settings, the same disk-headroom check); the report, or None. Never
    raises. Removes its run dir afterwards (the pin is the caller's)."""
    from ..validator.eval_offload import (
        BENCH_DISK_HEADROOM_BYTES,
        _room_ok,
        bench_make_room_command,
        build_bench_remote_command,
        record_bench_evidence,
    )
    from .remote import build_ssh_argv

    run = _run_or_ssh(run)
    q = shlex.quote
    from ..shared.pod_data_root import base_dir
    from ..validator.eval_offload import _placement
    from .lane_b_offload import LaneBPodRunner

    # 2026-11-27: the same dirs (and data-mount placement) as the lane B
    # runner and the validator's bench offload
    base = f"{LaneBPodRunner.root_of(host)}/bench-runs/{uuid.uuid4().hex}"
    report = f"{base}/report.json"
    cache = f"{base_dir(host)}/_eval_offload/cache"
    try:
        prep = run(build_ssh_argv(host, _placement(host, f"{base_dir(host)}/_eval_offload")
                                  + _placement(host, LaneBPodRunner.root_of(host))
                                  + f"mkdir -p {q(base)} {q(cache)}"), 120)
        if prep.returncode != 0:
            log.warning("lane B bench: prep failed on %s: %s", getattr(host, "name", host),
                        (prep.stderr or "")[-200:])
            return None
        room = run(build_ssh_argv(host, bench_make_room_command(
            cache, [], BENCH_DISK_HEADROOM_BYTES)), 300)
        if not _room_ok(host, room, BENCH_DISK_HEADROOM_BYTES, "the lane-B benchmark"):
            return None
        cmd = build_bench_remote_command(
            host, remote_ckpt, report, suites=",".join(settings["suites"]),
            num_samples=settings["num_samples"], batch_size=settings["batch_size"],
            max_series=settings["max_series"], data_dir=(settings["data_dir"] or None),
            device="cuda", workers=workers, timeout_s=max(1, int(timeout_s) - 15))
        proc = run(build_ssh_argv(host, cmd), int(timeout_s))
        if proc.returncode != 0:
            log.warning("lane B bench: sidecar failed on %s (exit %s): %s",
                        getattr(host, "name", host), proc.returncode,
                        (proc.stderr or "")[-400:])
            record_bench_evidence(label, reports=[None], proc=proc,
                                  note=f"sidecar exit {proc.returncode} on {host.name}")
            return None
        cat = run(build_ssh_argv(host, f"cat {q(report)}"), 120)
        rep = json.loads(cat.stdout) if cat.returncode == 0 else None
        record_bench_evidence(label, reports=[rep], proc=proc, note=f"on {host.name}")
        return rep
    except Exception as e:  # noqa: BLE001 — a failed bench is retried by the next ask
        log.warning("lane B bench: pinned sweep errored on %s: %s", getattr(host, "name", host), e)
        return None
    finally:
        with contextlib.suppress(Exception):
            run(build_ssh_argv(host, f"rm -rf {q(base)}"), 30)
