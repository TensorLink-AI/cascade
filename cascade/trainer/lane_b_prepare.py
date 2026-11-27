"""Lane B verdict prepared AHEAD of the settlement (owner 2026-11-27).

A finished lane B leg used to wait for the next 3-hourly settlement before
anything ran: the rebase, the candidate's four-suite public bench (~40 min)
and the merge check were all queued INSIDE the settlement, so one transient
failure (eval-pod disk, a malformed pod reply) cost a whole settlement slot
and a cold bench pushed the verdict ~3 h further out.

This module runs the deterministic, settlement-independent steps as soon as
the driver records a leg finished (:meth:`LaneBDriver.tick` calls
:meth:`LaneBPrepare.submit` with :func:`prepare_specs`), on ONE background
worker, against the CURRENT era king — the king the settlement rebases onto
(the king is frozen while lane B settles). Per leg, in order:

1. ``king_rows`` — the king's public bench rows (the #909 prebench), cached by
   content key in :class:`~cascade.trainer.lane_b_bench.LaneBBench`;
2. ``rebase`` — :meth:`~cascade.trainer.lane_b_offload.LaneBPodRunner.rebase_stage`
   on the bench pod: the settlement's rebase (the same king / base / leg /
   selection / refuse_nonfinite / limits, so the same bytes) WITHOUT the
   push, its bench files pinned there, plus the judge's merge check of those
   bytes onto the king;
3. ``candidate_rows`` — the pinned candidate swept by the bench's single
   worker and cached under its content key (+ bench method).

Nothing here is published or committed. The settlement path is unchanged
and recomputes everything that depends on the settlement itself:

* the ROUND ID — ``block_seed(boundary)``, i.e. the boundary block's hash —
  names the pushed repo (``ckpt-r{rid}-expert-…``), hence the rebased
  POINTER in the expert manifest: the settlement still rebases and pushes
  (its pod pulls are warm), and the push returns the content key the bench
  cache is keyed by — so the candidate rows are a cache hit;
* the routing-staleness probes (the affinity slice is drawn at the block),
  the §16 sentinel, the verdict windows (``verdict_windows(round_id,
  block)``) and every king / candidate evaluation on them, the anchor's
  derivation, and every block-gated flag (``merge_guard_at_settlement``,
  ``acceptance_at_settlement``, ``image_record_at``) — evaluated at the
  settlement block exactly as before.

The cached merge check is reused by the self-judge only when it said
"mergeable" for the same king pointer, candidate content key, selection and
strictness (:func:`merge_check_with`); a refusal (whose text may name pod
paths) is always recomputed live. So the published expert manifest, the
judgements and every committed record are byte-identical to the
settle-only path (tests/accept/wb27/test_wb27_lane_b_prepare_ahead.py).

**Retry fast.** A failed step (pod unreachable, disk short, a malformed
reply, the heavy lock busy, a bench without complete rows) retries with
backoff (:data:`BACKOFF_S`, bounded by :data:`MAX_ATTEMPTS`) — minutes, not a
settlement. **Alert early.** The FIRST failure of each step logs an ERROR,
and :meth:`LaneBPrepare.status` (``status/v2.json`` ``lane_b.prepare``)
carries each leg's steps with state, attempts and the last error.

**Box rules.** All model work is on the eval pod (``LaneBPodRunner``, under
the pod's ``.heavy.lock``); one step at a time on one worker thread; never on
the tick thread (``submit`` only records specs); paused while a settlement
runs (the settlement's own pod steps go first); a step failure never raises
into the driver. Torch-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger("cascade.trainer.lane_b_prepare")

STEPS = ("king_rows", "rebase", "candidate_rows")
# retry backoff per failed attempt (seconds); the last value repeats
BACKOFF_S = (60, 120, 240, 480, 600)
MAX_ATTEMPTS = 8                      # ≈ 50 min of retries, then "failed"
REARM_S = 3600                        # a failed leg is re-armed after this
POLL_S = 20.0                         # the worker's wake-up interval
MERGE_CHECKS_KEEP = 256

PENDING, RUNNING, WAITING, DONE = "pending", "running", "waiting", "done"
RETRYING, FAILED, REFUSED, SKIPPED = "retrying", "failed", "refused", "skipped"
TERMINAL = frozenset({DONE, FAILED, REFUSED, SKIPPED})


def spec_id(spec: dict) -> str:
    """A prepared job's identity: every input of the rebase (any change =
    another job; the settlement's caches are keyed by the same inputs)."""
    doc = {k: spec[k] for k in ("king_pointer", "base_pointer", "leg_pointer",
                                "refuse_nonfinite", "limits", "merge_strict")}
    doc["selection"] = {str(k): sorted(int(x) for x in v)
                        for k, v in sorted(spec["selection"].items(), key=lambda kv: int(kv[0]))}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:24]


def merge_check_key(king_pointer: str, content_key: str, selection: dict,
                    strict: bool) -> str:
    sel = {str(k): sorted(int(x) for x in v)
           for k, v in sorted(selection.items(), key=lambda kv: int(kv[0]))}
    return hashlib.sha256(json.dumps([str(king_pointer), str(content_key), sel,
                                      bool(strict)], sort_keys=True).encode()).hexdigest()


class LaneBPrepare:
    """The prepare-ahead worker. ``runner``: the bench pod's
    :class:`~cascade.trainer.lane_b_offload.LaneBPodRunner` (``rebase_stage``,
    ``unpin``); ``bench``: the :class:`~cascade.trainer.lane_b_bench.LaneBBench`
    whose cache the settlement reads. ``start_worker=False`` (tests): drive it
    with :meth:`run_due`."""

    def __init__(self, runner, bench, *, clock: Callable[[], float] = time.time,
                 start_worker: bool = True, backoff_s: tuple = BACKOFF_S,
                 max_attempts: int = MAX_ATTEMPTS, poll_s: float = POLL_S):
        self.runner, self.bench = runner, bench
        self.clock = clock
        self.backoff_s = tuple(backoff_s) or (60,)
        self.max_attempts = max(1, int(max_attempts))
        self.poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._merge_checks: dict[str, str | None] = {}
        self._paused: Callable[[], bool] = lambda: False
        self._wake = threading.Event()
        self._start_worker = start_worker
        self._worker: threading.Thread | None = None

    # ── intake (tick thread: no I/O) ─────────────────────────────────────────

    def submit(self, specs: list[dict], *, paused: Callable[[], bool] | None = None) -> None:
        """Track exactly ``specs`` (one per finished leg, from
        :func:`prepare_specs`): new ones start, gone ones (settled, or the king
        moved on) are dropped; a leg failed for :data:`REARM_S` is re-armed."""
        if paused is not None:
            self._paused = paused
        now = self.clock()
        with self._lock:
            want = {spec_id(s): s for s in specs}
            for jid in [j for j in self._jobs if j not in want]:
                self._drop_locked(jid)
            for jid, s in want.items():
                job = self._jobs.get(jid)
                if job is None:
                    self._jobs[jid] = _new_job(s, now)
                elif job["state"] == FAILED and now - float(job["updated"]) >= REARM_S:
                    log.info("lane B prepare: re-arming %s after %.0fs", job["label"],
                             now - float(job["updated"]))
                    self._jobs[jid] = _new_job(s, now)
            start = bool(want) and self._start_worker and (
                self._worker is None or not self._worker.is_alive())
            if start:
                self._worker = threading.Thread(target=self._loop, name="laneb-prepare",
                                                daemon=True)
                self._worker.start()
        if want:
            self._wake.set()

    def _drop_locked(self, jid: str) -> None:
        job = self._jobs.pop(jid)
        staged = job.get("staged")
        if staged and staged.get("pin") and not job.get("pin_queued"):
            with contextlib.suppress(Exception):     # the pod prunes stale pins itself
                self.runner.unpin(staged["host"], staged["pin"])

    # ── worker ───────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            self._wake.wait(self.poll_s)
            self._wake.clear()
            try:
                self.run_due()
            except Exception as e:  # noqa: BLE001 — the worker never dies
                log.error("lane B prepare: worker pass failed: %s", e)

    def run_due(self) -> int:
        """Advance every job whose next step is due, one step at a time (the
        worker's body; tests call it). Returns the number of steps run."""
        if self._paused():
            return 0
        n = 0
        with self._lock:
            jids = list(self._jobs)
        for jid in jids:
            if self._paused():
                break
            with self._lock:
                job = self._jobs.get(jid)
            if job is None or job["state"] in TERMINAL:
                continue
            if float(job.get("next_at", 0.0)) > self.clock():
                continue
            self._advance(jid, job)
            n += 1
        return n

    def _advance(self, jid: str, job: dict) -> None:
        step = next((s for s in STEPS if job["steps"][s]["state"] not in TERMINAL), None)
        if step is None:
            self._finish(jid, job)
            return
        try:
            getattr(self, f"_step_{step}")(job)
        except Exception as e:  # noqa: BLE001 — any step failure retries with backoff
            from .lane_b_model import MergeRefused

            if isinstance(e, MergeRefused):
                self._set(job, step, REFUSED, error=str(e))
                for s in STEPS[STEPS.index(step) + 1:]:
                    self._set(job, s, SKIPPED)
                log.warning("lane B prepare: %s rebase refused (%s) — the settlement "
                            "re-decides", job["label"], e)
            else:
                self.fail(job, step, f"{type(e).__name__}: {e}")
        if all(job["steps"][s]["state"] in TERMINAL for s in STEPS):
            self._finish(jid, job)

    def _finish(self, jid: str, job: dict) -> None:
        with self._lock:
            states = {job["steps"][s]["state"] for s in STEPS}
            job["state"] = FAILED if FAILED in states else DONE
            job["updated"] = self.clock()
        if job["state"] == DONE:
            log.info("lane B prepare: %s ready ahead of the settlement (%s)", job["label"],
                     ", ".join(f"{s}={job['steps'][s]['state']}" for s in STEPS))

    # ── steps ────────────────────────────────────────────────────────────────

    def _step_king_rows(self, job: dict) -> None:
        king = job["spec"]["king_pointer"]
        if self.bench.pointer_rows(king) is not None:
            self._set(job, "king_rows", DONE)
            return
        tag = self.bench.ptr_tag(king)
        why = self.bench.failure(tag)
        if why is not None and not self.bench.queued(tag):
            raise OSError(f"king bench failed: {why}")
        self.bench.prebench(king)
        self._wait(job, "king_rows")

    def _step_rebase(self, job: dict) -> None:
        s = job["spec"]
        self._set(job, "rebase", RUNNING)
        got = self.runner.rebase_stage(
            king_pointer=s["king_pointer"], base_pointer=s["base_pointer"],
            leg_pointer=s["leg_pointer"], selection=s["selection"],
            refuse_nonfinite=bool(s["refuse_nonfinite"]), limits=bool(s["limits"]),
            bench_settings=self.bench.settings, merge_strict=bool(s["merge_strict"]))
        if got.get("content_key") is None:
            # a limits leg without its lane_b_leg record: the settlement drops it
            self._set(job, "rebase", SKIPPED, error="limits leg without its lane_b_leg record")
            self._set(job, "candidate_rows", SKIPPED)
            return
        job["staged"] = got
        job["pin_queued"] = False
        job["content_key"] = got["content_key"]
        if got.get("merge_error") is None:
            with self._lock:
                self._merge_checks[merge_check_key(
                    s["king_pointer"], got["content_key"], s["selection"],
                    bool(s["merge_strict"]))] = None
                while len(self._merge_checks) > MERGE_CHECKS_KEEP:
                    self._merge_checks.pop(next(iter(self._merge_checks)))
        job["merge_check"] = "ok" if got.get("merge_error") is None else "refused"
        self._set(job, "rebase", DONE)

    def _step_candidate_rows(self, job: dict) -> None:
        key = job.get("content_key")
        staged = job.get("staged")
        if not key:
            raise OSError("no staged candidate")
        if self.bench.cached(key) is not None:
            if staged and staged.get("pin") and not job.get("pin_queued"):
                self.runner.unpin(staged["host"], staged["pin"])
                job["pin_queued"] = True
            self._set(job, "candidate_rows", DONE)
            return
        tag = self.bench.key_tag(key)
        if self.bench.queued(tag):
            self._wait(job, "candidate_rows")
            return
        why = self.bench.failure(tag)
        if job.get("pin_queued") or staged is None or not staged.get("pin"):
            # its pin was swept (or removed): a failed bench re-stages it
            job["steps"]["rebase"]["state"] = PENDING
            job["staged"] = None
            raise OSError(f"candidate bench failed: {why or 'no rows'}")
        self.bench.bench_pin(staged["host"], key, staged["pin"], job["label"])
        job["pin_queued"] = True
        self._wait(job, "candidate_rows")

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def _set(self, job: dict, step: str, state: str, *, error: str | None = None) -> None:
        with self._lock:
            st = job["steps"][step]
            st["state"] = state
            st["updated"] = self.clock()
            if error is not None:
                st["last_error"] = str(error)[:300]
            job["updated"] = st["updated"]
            job["state"] = RUNNING if state not in TERMINAL else job["state"]

    def _wait(self, job: dict, step: str) -> None:
        self._set(job, step, WAITING)
        with self._lock:
            job["next_at"] = self.clock() + min(self.poll_s, float(self.backoff_s[0]))

    def fail(self, job: dict, step: str, error: str) -> None:
        """Record a failed attempt of ``step``: an ERROR on its first failure
        (the owner hears at once), a warning on later ones; retried after the
        backoff, ``failed`` once :attr:`max_attempts` are spent."""
        with self._lock:
            st = job["steps"][step]
            st["attempts"] = int(st.get("attempts", 0)) + 1
            n = st["attempts"]
            st["last_error"] = str(error)[:300]
            st["updated"] = self.clock()
            delay = float(self.backoff_s[min(n - 1, len(self.backoff_s) - 1)])
            exhausted = n >= self.max_attempts
            st["state"] = FAILED if exhausted else RETRYING
            if exhausted and step == "rebase":
                # nothing staged: the candidate's bench cannot run
                job["steps"]["candidate_rows"]["state"] = SKIPPED
            job["next_at"] = self.clock() + delay
            job["updated"] = st["updated"]
        if n == 1:
            log.error("lane B prepare: %s step %s FAILED (%s) — retrying in %.0fs (lane_b."
                      "prepare in status/v2.json)", job["label"], step, error, delay)
        elif exhausted:
            log.error("lane B prepare: %s step %s failed %d times (%s) — giving up until the "
                      "settlement (re-armed after %ds)", job["label"], step, n, error, REARM_S)
        else:
            log.warning("lane B prepare: %s step %s failed again (attempt %d: %s) — retrying "
                        "in %.0fs", job["label"], step, n, error, delay)

    # ── settlement side ──────────────────────────────────────────────────────

    def merge_check_cached(self, king_pointer: str, content_key: str | None,
                           selection: dict, strict: bool) -> bool:
        """True when the prepared merge check said "mergeable" for exactly
        these inputs (the only cached answer the self-judge reuses)."""
        if not content_key:
            return False
        k = merge_check_key(king_pointer, content_key, selection, strict)
        with self._lock:
            return k in self._merge_checks and self._merge_checks[k] is None

    def status(self) -> dict:
        """``status/v2.json`` ``lane_b.prepare``: per leg, its steps' state,
        attempts and last error."""
        with self._lock:
            legs = []
            for jid, job in self._jobs.items():
                legs.append({
                    "id": jid, "leg": job["label"], "state": job["state"],
                    "king_pointer": job["spec"]["king_pointer"],
                    "leg_pointer": job["spec"]["leg_pointer"],
                    "merge_check": job.get("merge_check", ""),
                    "steps": {s: {k: v for k, v in job["steps"][s].items()}
                              for s in STEPS}})
            return {"paused": bool(self._paused()), "legs": legs}


def _new_job(spec: dict, now: float) -> dict:
    return {"spec": dict(spec), "label": str(spec.get("label", "")), "state": PENDING,
            "updated": now, "next_at": 0.0,
            "steps": {s: {"state": PENDING, "attempts": 0, "last_error": "", "updated": now}
                      for s in STEPS}}


def merge_check_with(prep: LaneBPrepare | None, bench, live: Callable):
    """The self-judge's ``merge_check(king, cand, selection, strict)``: the
    prepared answer when it is "mergeable" for the candidate's content key
    (the settlement's rebase recorded ``cand`` → key in the bench), else
    ``live`` (the pod's merge check) — refusals always run live."""
    if prep is None or bench is None:
        return live

    def check(king_pointer: str, cand_pointer: str, selection: dict, strict: bool):
        key = None
        try:
            key = bench._ptr_key(str(cand_pointer))
        except Exception:  # noqa: BLE001 — a cache miss runs the live check
            key = None
        if prep.merge_check_cached(king_pointer, key, selection, strict):
            log.info("lane B: merge check of %s reused from the prepare-ahead step",
                     str(cand_pointer)[-24:])
            return None
        return live(king_pointer, cand_pointer, selection, strict)

    return check
