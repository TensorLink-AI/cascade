"""The gauntlet engine — the deterministic judge (see the package docstring).

One CYCLE:

1. **refresh** the round window (receipts + revealed snapshots; at most hourly).
   A changed window, king, chain.toml or image is a new EPOCH.
2. **baseline** (once per epoch): the king's short legs on every window round,
   plus salted king legs that measure the noise σ the stage margins derive from.
   Population members re-run G3 on the new window; the ones that no longer pass
   retire.
3. **propose**: workers edit copies of population members (or the start tree).
4. **G0–G3** for the new candidates (and any a crash left mid-gauntlet).
5. **G4** every ``g4_every_cycles``: full-contract replays of the top members
   against the receipt kings; the best passer goes to **G4.5** (pool C) and **G5**.

Everything is on disk under ``workdir`` and every stage result is written as
soon as it exists, so a killed judge resumes where it stopped. Infrastructure
faults retry and never count against a candidate; only a candidate fault
(rejected, crashed or stalled generator) kills it.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import logging
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from pathlib import Path

from ..optimize import _COPY_IGNORE
from . import stats
from .jobs import scores_from_json
from .rounds import (
    RoundRef,
    RoundWindow,
    build_window,
    fingerprint,
    local_snapshot_blocks,
    needed_snapshot_blocks,
    refresh_receipts,
    replayable_rounds,
    sync_snapshots,
    tree_digest,
)
from .workers import Proposal, build_prompt

log = logging.getLogger("cascade.miner.harness.gauntlet")

STOP_FILE = "STOP"
RESTART_FILE = "RESTART"            # the updater asks for a restart between cycles
RESTART_ACK = "RESTART.ack"         # the judge: parked, safe to recreate
def progress_bar(pr: dict, width: int = 30) -> str:
    """``dethrone [███████░░░…] 23/100  best c00005 +0.32% @G3 (target +1.00%)``."""
    score = float(pr.get("score") or 0.0)
    fill = int(round(width * score / 100.0))
    bar = "█" * fill + "░" * (width - fill)
    tail = ""
    if pr.get("id"):
        tail = (f"  best {pr['id']} {pr['estimate']:+.2%} @{pr['stage']}"
                f" (target {pr.get('target', 0.01):+.2%})")
    return f"dethrone [{bar}] {score:3.0f}/100{tail}"


# Dethrone progress: each stage's evidence can claim at most this much of 100.
PROGRESS_CAPS = (("G4.5", 100.0), ("G4", 90.0), ("G3", 70.0), ("G2", 40.0))

MAX_INFRA_RETRIES = 2
HEARTBEAT_SECONDS = 300.0
POOL_BUILD_ATTEMPTS = 3
MAX_SHORTAGE_WAITS = 6              # ~1 h of back-off when the market has no GPU


def _is_shortage(error: str) -> bool:
    """A fault that waiting fixes: no executor free (or ours cooling down)."""
    e = error.lower()
    return "available, need" in e or "not ready after" in e
REFRESH_SECONDS = 3600.0
# Statuses that are population members (parents, G4 candidates). A finalist
# or a submitted tree stays one: it is still among the best we have.
POPULATION = ("member", "finalist", "submitted")


class BudgetWait(RuntimeError):
    """The daily spend cap is reached; the judge sleeps until 00:00 UTC."""


@dataclass
class Job:
    key: str
    spec: dict
    est_hours: float


def _atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Gauntlet:
    def __init__(self, hcfg, *, chain_cfg=None, executor=None, workers=None, submitter=None,
                 index_fetch=None, text_fetch=None, list_folders=None, download=None,
                 pool_builder=None, verify_fn=None, king_hooks=None, now=time.time,
                 sleep=time.sleep) -> None:
        from ...shared.config import load_chain_config

        self.h = hcfg
        self.wd = Path(hcfg.workdir)
        self.wd.mkdir(parents=True, exist_ok=True)
        self.chain = chain_cfg or load_chain_config(hcfg.chain_toml)
        self.executor = executor
        self.workers = workers
        self.submitter = submitter
        self._fetch = {"index_fetch": index_fetch, "text_fetch": text_fetch}
        self._sync = {"list_folders": list_folders, "download": download}
        self._pool_builder = pool_builder or self._build_pool_cli
        self._verify = verify_fn or self._static_verify
        self._king_hooks = king_hooks or {}
        self._meta_cache: dict[str, dict] | None = None
        self._now, self._sleep = now, sleep
        for d in ("candidates", "epochs", "operator", "receipts", "pools"):
            (self.wd / d).mkdir(parents=True, exist_ok=True)
        self.state = _read_json(self.wd / "state.json", {}) or {
            "epoch": 0, "fingerprint": "", "window": None, "cycle": 0, "king_ready": False,
            "last_refresh": 0.0, "next_id": 1, "phase": "starting", "status": ""}

    # ------------------------------------------------------------------ disk
    def save(self) -> None:
        self.state["updated"] = self._now()
        _atomic_json(self.wd / "state.json", self.state)
        self._write_operator_status()

    def event(self, kind: str, **kw) -> None:
        with open(self.wd / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": self._now(), "kind": kind, **kw}, default=str) + "\n")

    def phase(self, text: str) -> None:
        from . import install_logging
        install_logging()
        self.state["phase"] = text
        log.info("%s", text)
        self.save()

    # Candidate metas are read from disk once, then served from memory (every
    # write goes through put()). Edit meta.json files only while the judge is down.
    def _metas(self) -> dict[str, dict]:
        if self._meta_cache is None:
            self._meta_cache = {}
            for p in sorted((self.wd / "candidates").glob("*/meta.json")):
                m = _read_json(p)
                if m:
                    self._meta_cache[m["id"]] = m
        return self._meta_cache

    def meta(self, cid: str) -> dict:
        return json.loads(json.dumps(self._metas().get(cid, {})))

    def put(self, m: dict) -> None:
        _atomic_json(self.wd / "candidates" / m["id"] / "meta.json", m)
        self._metas()[m["id"]] = json.loads(json.dumps(m))

    def all_metas(self) -> list[dict]:
        return [json.loads(json.dumps(self._metas()[k])) for k in sorted(self._metas())]

    @property
    def window(self) -> RoundWindow | None:
        w = self.state.get("window")
        return RoundWindow.from_json(w) if w else None

    @property
    def epoch(self) -> int:
        return int(self.state.get("epoch", 0))

    def tree(self, cid: str) -> Path:
        if cid == "king":
            live = self.state.get("king") if self.h.king_source == "live" else None
            return Path(live["dir"]) if live else Path(self.h.king_dir)
        if cid == "start":
            return Path(self.h.start_dir)
        return self.wd / "candidates" / cid / "tree"

    def stopped(self) -> bool:
        return (self.wd / STOP_FILE).exists() or (self.wd / "operator" / STOP_FILE).exists()

    # ----------------------------------------------------------- the window
    def _setup_extra(self) -> dict:
        ct = Path(self.h.chain_toml) if self.h.chain_toml else None
        if ct is None:
            from ...shared.config import DEFAULT_CHAIN_TOML
            ct = DEFAULT_CHAIN_TOML
        chain_digest = hashlib.sha256(ct.read_bytes()).hexdigest() if ct.is_file() else ""
        return {"king": tree_digest(self.tree("king")), "chain": chain_digest,
                "image": self.h.compute.image or self.state.get("worker_image", ""),
                "executor": self.h.compute.executor,
                "g2": self.h.stages.g2_hours, "g3": self.h.stages.g3_hours}

    def refresh_window(self, *, force: bool = False) -> bool:
        """Re-derive the window; returns True when a NEW epoch started."""
        if not force and self._now() - float(self.state.get("last_refresh", 0)) < REFRESH_SECONDS:
            return False
        self.state["last_refresh"] = self._now()
        if self.h.king_source == "live":
            from .king import refresh_live_king
            self.state["king"] = refresh_live_king(self.chain, self.wd / "king",
                                                   self.state.get("king"), **self._king_hooks)
        r = self.h.rounds
        rdir = self.wd / "receipts"
        refresh_receipts(self.chain, rdir, validator=r.validator, scan=r.receipt_scan,
                         **self._fetch)
        root = Path(r.snapshot_root)
        if r.sync_reveals:
            # A window needs at most n_a + n_b rounds; 2 rounds share a daily
            # snapshot, plus slack for the ~48h reveal lag.
            needed = needed_snapshot_blocks(rdir, newest=r.n_a + r.n_b + 4)
            if r.sync_via == "executor":
                self._sync_via_executor(needed, root)
            else:
                sync_snapshots(needed, root, repo=r.hf_repo, **self._sync)
        refs = [ref for ref in replayable_rounds(rdir, root) if self._round_ok(ref)]
        window = build_window(refs, n_a=r.n_a, n_b=r.n_b)
        if window is None:
            self.state["status"] = (f"waiting: fewer than {r.n_b + 1} replayable rounds "
                                    f"(receipts in {rdir}, snapshots in {root})")
            self.save()
            return False
        self._sync_worker_image(window)
        fp = fingerprint(window, **self._setup_extra())
        if fp == self.state.get("fingerprint"):
            self.save()
            return False
        self.state.update(epoch=self.epoch + 1, fingerprint=fp, window=window.to_json(),
                          king_ready=False, status="")
        self.event("epoch", epoch=self.epoch, rounds=[x.round_id for x in window.all])
        log.info("epoch %d: A=%s B=%s", self.epoch, [x.round_id for x in window.a],
                 [x.round_id for x in window.b])
        self.save()
        return True

    def _round_ok(self, ref: RoundRef) -> bool:
        """Can this checkout replay the round faithfully? Rebuilds its verdict
        windows and contract on the judge (CPU, cached per round), so a round
        whose draw or contract this code cannot reproduce never reaches a pod."""
        cache = self.state.setdefault("round_ok", {})
        if ref.round_id not in cache:
            from ...shared.receipt import load_receipt
            from ..replay import ReplayError, load_replay_round
            try:
                load_replay_round(self.chain, load_receipt(
                    ref.receipt_path.read_text(encoding="utf-8")), ref.snapshot_dir)
                cache[ref.round_id] = True
            except ReplayError as e:
                log.warning("round %s left out of the window: %s", ref.round_id, e)
                self.event("round_unreplayable", round_id=ref.round_id, error=str(e)[:300])
                cache[ref.round_id] = False
        return bool(cache[ref.round_id])

    def _newest_contract(self, window: RoundWindow | None = None) -> dict:
        """The newest window round's signed ``contract_body`` ({} if unknown)."""
        w = window or self.window
        if w is None or not w.all:
            return {}
        try:
            doc = json.loads(w.all[-1].receipt_path.read_text(encoding="utf-8"))
            return doc["manifest"].get("contract_body") or {}
        except (OSError, ValueError, KeyError):
            return {}

    def _sync_worker_image(self, window: RoundWindow) -> None:
        """Rent the worker image the newest round's signed contract trained on
        (unless ``[compute] image`` pins one): the repo's funded_pod_image can
        lag the live trainer by releases."""
        if self.h.compute.image or not hasattr(self.executor, "set_image"):
            return
        digest = self._newest_contract(window).get("train_image_digest", "")
        if not digest:
            return
        from .images import resolve_worker_image
        ref = resolve_worker_image(digest, cache=self.wd / "image_cache.json")
        if ref is None:
            log.warning("no published worker tag for %s; keeping %s", digest,
                        getattr(self.executor, "image", "?"))
            return
        self.executor.set_image(ref)
        self.state["worker_image"] = ref

    def _sync_via_executor(self, needed: set[int], root: Path) -> None:
        """Download missing revealed folders with a job on the executor. Blocks
        not revealed yet are retried at most every 6 h (no pod per refresh)."""
        wait = {int(k): v for k, v in self.state.get("unrevealed_until", {}).items()}
        want = sorted(b for b in needed - local_snapshot_blocks(root)
                      if wait.get(b, 0) <= self._now())
        if not want:
            return
        self.phase(f"fetching {len(want)} revealed snapshot(s) via the executor")
        res = self._run_jobs([Job("fetch-snapshots", {
            "kind": "fetch_snapshots", "inputs": {},
            "params": {"repo": self.h.rounds.hf_repo, "blocks": want},
            "outputs": {"snapshots": str(root)}}, 0.25)])["fetch-snapshots"]
        if "error" in res:
            log.warning("snapshot fetch failed: %s", res["error"])
            self.state["status"] = f"snapshot fetch failed: {str(res['error'])[:300]}"
            return
        for b in res.get("unrevealed", []):
            wait[int(b)] = self._now() + 6 * 3600
        self.state["unrevealed_until"] = {str(k): v for k, v in wait.items()
                                          if v > self._now()}
        log.info("snapshots fetched: %s; not revealed yet: %s",
                 res.get("fetched"), res.get("unrevealed"))

    # ------------------------------------------------------------ job plumbing
    def _replay_spec(self, gen: Path, ref: RoundRef, hours: float | None, *, salt: int = 0,
                     ckpt: Path | None = None) -> dict:
        spec = {"kind": "replay",
                "inputs": {"gen": str(gen), "receipt": str(ref.receipt_path),
                           "snapshot": str(ref.snapshot_dir)},
                "params": {"train_hours": hours, "seed_salt": salt}, "outputs": {}}
        if ckpt is not None:
            spec["outputs"]["checkpoint"] = str(ckpt)
        return spec

    def _run_jobs(self, jobs: list[Job]) -> dict[str, dict]:
        """Run ``jobs`` in parallel on the executor; infra faults retry up to
        :data:`MAX_INFRA_RETRIES`, GPU shortages back off. A job the spend cap
        refused comes back with ``budget_exhausted`` and arms
        :meth:`_check_budget`, so every completed (paid) result is still recorded."""
        if not jobs:
            return {}
        results: dict[str, dict] = {}
        pending = list(jobs)
        budget_hit = False
        attempt = faults = shortages = 0
        while pending:
            with ThreadPoolExecutor(max_workers=max(1, self.executor.capacity)) as pool:
                futs = {j.key: pool.submit(self.executor.run, j.spec,
                                           job_id=f"e{self.epoch}-{j.key}-a{attempt}",
                                           est_hours=j.est_hours) for j in pending}
                while not all(f.done() for f in futs.values()):
                    # Heartbeat: a full-budget batch can block for hours; the
                    # operator reads `updated` to tell a slow judge from a hung one.
                    futures_wait(list(futs.values()), timeout=HEARTBEAT_SECONDS)
                    self.save()
                done = {k: f.result() for k, f in futs.items()}
            attempt += 1
            retry, short = [], False
            for j in pending:
                res = done[j.key]
                results[j.key] = res
                if "error" not in res or res.get("candidate_fault"):
                    continue
                if res.get("budget_exhausted"):
                    budget_hit = True
                    continue
                if res.get("deterministic"):
                    continue          # the round, not the pod: another try gives the same
                log.warning("job %s infra fault (attempt %d): %s", j.key, attempt,
                            res["error"])
                self.event("infra_fault", job=j.key, attempt=attempt,
                           error=str(res["error"])[:500])
                short = short or _is_shortage(str(res["error"]))
                retry.append(j)
            pending = retry
            if not pending:
                break
            if short and shortages < MAX_SHORTAGE_WAITS and not self.stopped():
                # The market is out of the GPU (or our cooldowns): wait for
                # capacity instead of burning the fault retries in seconds.
                shortages += 1
                wait = min(900.0, 300.0 * shortages)
                self.phase(f"waiting {wait / 60:.0f} min for GPU capacity "
                           f"({len(pending)} job(s), wait {shortages}/{MAX_SHORTAGE_WAITS})")
                self._interruptible_sleep(wait)
                continue
            faults += 1
            if faults > MAX_INFRA_RETRIES:
                break
        if budget_hit:
            # The paid results above are kept; the caller finishes recording
            # them and the cycle stops at its next checkpoint.
            self._budget_hit = True
        return results

    def _check_budget(self) -> None:
        if getattr(self, "_budget_hit", False):
            self._budget_hit = False
            raise BudgetWait("spend cap reached")

    # ------------------------------------------------------------- baseline
    def _cache_path(self, ref: RoundRef, hours: float) -> Path:
        # Keyed by king AND the image it trained on: a new worker release
        # re-trains the reference instead of mixing numerics.
        king = tree_digest(self.tree("king"))[:16]
        image = (self.h.compute.image or self.state.get("worker_image", "")).split("@")[-1]
        img = image.split(":")[-1][:12] or "default"
        return self.wd / "king_cache" / f"{king}-{img}" / f"{ref.round_id}.{hours:g}h.json"

    def _king_jobs(self, refs: list[RoundRef], hours: float) -> list[Job]:
        """Short king legs still missing from the cache for ``refs`` (cached mode)."""
        if self.h.stages.reference != "cached":
            return []
        seen, jobs = set(), []
        for ref in refs:
            if ref.round_id in seen or self._cache_path(ref, hours).is_file():
                continue
            seen.add(ref.round_id)
            jobs.append(Job(f"king-{ref.round_id}-{hours:g}h",
                            self._replay_spec(self.tree("king"), ref, hours), hours))
        return jobs

    def _store_king(self, res: dict, refs: list[RoundRef], hours: float) -> None:
        for ref in refs:
            doc = res.get(f"king-{ref.round_id}-{hours:g}h")
            if doc and "scores" in doc:
                _atomic_json(self._cache_path(ref, hours), doc)

    def _king_ref(self, ref: RoundRef, tag: str, row: dict) -> tuple[float, list]:
        """``(king geomean, king per-window scores)`` the candidate in ``row`` is
        compared with on ``ref``: the cached same-budget king leg, or the
        receipt's signed scores."""
        if self.h.stages.reference == "cached":
            hours = self.h.stages.g2_hours if tag == "g2" else self.h.stages.g3_hours
            doc = _read_json(self._cache_path(ref, hours))
            if not doc or "scores" not in doc:
                raise KeyError(f"no cached king leg for round {ref.round_id} at {hours:g}h")
            return float(doc["geomean"]), scores_from_json(doc["scores"])
        from ..replay import receipt_king_scores
        return float(row["king_geomean"]), receipt_king_scores(
            self.chain, ref.receipt_path.read_text(encoding="utf-8"))

    def ensure_baseline(self) -> bool:
        """Per-epoch setup: the stage margins, then the population re-confirmed
        on the new window. Nothing is trained up front (the receipts are the
        king's scores, or its same-budget legs are trained lazily per round and
        cached). The re-confirmation has its own flag, so a cap or a crash in
        the middle of it resumes instead of skipping it."""
        s = self.h.stages
        if not self.state.get("king_ready"):
            self.state.update(king_ready=True, m2=s.g2_margin_floor, m3=s.g2_margin_floor)
            self.event("baseline", epoch=self.epoch, reference=s.reference,
                       margin=s.g2_margin_floor)
            self.save()
        if self.state.get("rebaselined_epoch") != self.epoch:
            self._rebaseline_population()
            self.state["rebaselined_epoch"] = self.epoch
            self.save()
        return True

    def _rebaseline_population(self) -> None:
        """Members re-run G3 on the new window; those that no longer pass retire."""
        members = [m for m in self.all_metas() if m.get("status") in POPULATION
                   and m.get("member_epoch") != self.epoch]
        if not members:
            return
        self.phase(f"epoch {self.epoch}: re-confirming {len(members)} population member(s)")
        for m in members:
            m.setdefault("stages", {}).pop("G3", None)
        self._g3(members)
        for m in members:
            g3 = m["stages"].get("G3", {})
            if g3.get("pass"):
                m["member_epoch"] = self.epoch
            elif g3:
                m["status"], m["reason"] = "retired", f"failed G3 in epoch {self.epoch}"
            self.put(m)
        self._trim_population()

    # ------------------------------------------------------------- proposing
    def _directives(self) -> str:
        p = self.wd / "operator" / "DIRECTIVES.md"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def members(self) -> list[dict]:
        ms = [m for m in self.all_metas() if m.get("status") in POPULATION]
        return sorted(ms, key=lambda m: -m["stages"]["G3"].get("rel", 0.0))

    def parents(self, n: int) -> list[str]:
        pinned = [ln.split(":", 1)[1].strip() for ln in self._directives().splitlines()
                  if ln.lower().startswith("parent:")]
        valid = {m["id"] for m in self.members()} | {"king", "start"}
        pool = [p for p in pinned if p in valid] or [m["id"] for m in self.members()] or ["start"]
        return [pool[i % len(pool)] for i in range(n)]

    def outcome_lines(self) -> list[str]:
        """The last outcomes with their FULL failure reasons (a truncated reason
        is how one worker repeated another's determinism bug)."""
        return [f"- {m['id']} from {m.get('parent')}: {m.get('note') or '(no note)'} "
                f"→ {self.summary(m)}" for m in self.all_metas()[-10:]]

    @staticmethod
    def summary(m: dict) -> str:
        """Worker/operator-safe outcome: G2 numbers (pool A) yes, G3+ pass/fail only."""
        st, s = m.get("status", "?"), m.get("stages", {})
        if st == "dead":
            return f"died: {m.get('reason', '?')}"
        if st in ("member", "finalist", "submitted", "retired"):
            g2 = s.get("G2", {})
            rel = f", screen {g2['rel']:+.2%}" if "rel" in g2 else ""
            return f"{st} (passed G3{rel})" if st != "retired" else f"retired ({m.get('reason')})"
        return st

    def write_knowledge(self) -> Path:
        """``<workdir>/knowledge``, regenerated from the candidate records each
        time (never edited by hand, so it cannot drift): the briefs from the
        operator folder, and ``attempts.jsonl``, one line per candidate. Workers
        see G2 numbers and G3+ pass/fail only, as everywhere else."""
        kdir = self.wd / "knowledge"
        kdir.mkdir(parents=True, exist_ok=True)
        for name in self.h.search.briefs:
            src = self.wd / "operator" / name
            if src.is_file():
                shutil.copyfile(src, kdir / name)
        rows = []
        for m in self.all_metas():
            st = m.get("stages", {})
            reached = next((k for k in ("G4.5", "G4", "G3", "G2", "G1", "G0") if k in st), None)
            rows.append({
                "id": m["id"], "parent": m.get("parent"), "epoch": m.get("epoch"),
                "change": m.get("note") or "", "status": m.get("status"),
                "furthest_stage": reached,
                "throughput_vs_king": (st.get("G1") or {}).get("ratio"),
                "screen_rel": (st.get("G2") or {}).get("rel"),
                "passed": [k for k in ("G0", "G1", "G2", "G3", "G4", "G4.5")
                           if (st.get(k) or {}).get("pass")],
                "reason": m.get("reason") or ""})
        tmp = kdir / "attempts.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(kdir / "attempts.jsonl")
        return kdir

    def propose(self, n: int) -> list[Proposal]:
        notebook = self.wd / "operator" / "NOTEBOOK.md"
        nb = notebook.read_text(encoding="utf-8") if notebook.is_file() else ""
        kdir = self.write_knowledge()
        outcomes = self.outcome_lines()
        props = []
        for parent in self.parents(n):
            cid = f"c{int(self.state['next_id']):05d}"
            self.state["next_id"] = int(self.state["next_id"]) + 1
            tree = self.wd / "candidates" / cid / "tree"
            shutil.copytree(self.tree(parent), tree, ignore=_COPY_IGNORE)
            self.put({"id": cid, "parent": parent, "epoch": self.epoch, "status": "proposed",
                      "parent_digest": tree_digest(tree), "stages": {}, "created": self._now()})
            props.append(Proposal(cid, parent, tree, build_prompt(
                directives=self._directives(), notebook=nb, outcomes=outcomes, parent=parent),
                knowledge=kdir))
        self.save()
        return props

    def accept_outcomes(self, outcomes) -> list[dict]:
        nb = self.wd / "operator" / "NOTEBOOK.md"
        out = []
        for o in outcomes:
            m = self.meta(o.id)
            m["note"], m["worker_seconds"] = o.note, round(o.seconds, 1)
            if o.lesson:
                with open(nb, "a", encoding="utf-8") as f:
                    f.write(f"- ({o.id}) {o.lesson.strip()}\n")
            if not o.ok:
                m["status"], m["reason"] = "dead", f"worker: {o.error}"
            else:
                m["status"] = "in_gauntlet"
                out.append(m)
            self.put(m)
        return out

    # ---------------------------------------------------------------- stages
    @staticmethod
    def _static_verify(tree: Path, chain) -> tuple[bool, str]:
        """Static admission checks only (layout, guard, config). Candidate code
        never runs on the judge, which holds the keys: the runtime checks
        (determinism) run in G1, on the executor."""
        from ..verify import verify_repo
        r = verify_repo(tree, chain, skip_runtime=True)
        return r.ok, "" if r.ok else r.render()[-600:]

    def _kill(self, m: dict, stage: str, reason: str) -> None:
        m["status"], m["reason"] = "dead", f"{stage}: {reason}"
        self.put(m)
        self.event("dead", id=m["id"], stage=stage, reason=reason)

    def _fault(self, m: dict, stage: str, res: dict) -> bool:
        """True when ``res`` ended the candidate (candidate fault) or stalled it."""
        if "error" not in res:
            return False
        if res.get("budget_exhausted"):
            m["status"] = "in_gauntlet"               # waits for the cap, never penalised
            self.put(m)
            return True
        if res.get("candidate_fault"):
            self._kill(m, stage, res["error"][:300])
        else:
            m["status"], m["reason"] = "stalled", f"{stage}: {res['error'][:300]}"
            self.put(m)
        return True

    def _g0(self, metas: list[dict]) -> list[dict]:
        seen = {tree_digest(self.tree("king")), tree_digest(self.tree("start"))}
        seen |= {m["digest"] for m in self.all_metas() if m.get("digest")}
        out = []
        for m in metas:
            if "G0" in m["stages"]:
                out.append(m)
                continue
            d = tree_digest(self.tree(m["id"]))
            if d == m.get("parent_digest"):
                self._kill(m, "G0", "no change")
                continue
            if d in seen:
                self._kill(m, "G0", "duplicate of an earlier tree")
                continue
            ok, why = self._verify(self.tree(m["id"]), self.chain)
            if not ok:
                self._kill(m, "G0", f"verify: {why}")
                continue
            seen.add(d)
            m["digest"], m["stages"]["G0"] = d, {"pass": True}
            self.put(m)
            out.append(m)
        return out

    def _g1(self, metas: list[dict]) -> list[dict]:
        s = self.h.stages
        todo = [m for m in metas if "G1" not in m["stages"]]
        res = self._run_jobs([Job(f"{m['id']}-G1", {
            "kind": "throughput",
            "inputs": {"gen": str(self.tree(m["id"])), "king": str(self.tree("king"))},
            "params": {"seconds": s.g1_seconds,
                       "budget_denomination": self._live_denomination()}}, 0.1)
            for m in todo])
        for m in todo:
            r = res[f"{m['id']}-G1"]
            if self._fault(m, "G1", r):
                continue
            if not r.get("verify_ok"):
                self._kill(m, "G1", f"runtime verify: {r.get('verify', '')[-800:]}")
                continue
            gen, king = r["gen"]["points_per_sec"], r["king"]["points_per_sec"]
            ratio = gen / king if king > 0 else float("inf")
            m["stages"]["G1"] = {"pass": ratio >= 1 - s.g1_throughput_tol, "ratio": ratio}
            if not m["stages"]["G1"]["pass"]:
                self._kill(m, "G1", f"{1 - ratio:.0%} slower than the king")
                continue
            self.put(m)
        return [m for m in metas if m["stages"].get("G1", {}).get("pass")]

    def _live_denomination(self) -> str | None:
        """Billing rule of the newest window round's signed contract."""
        return self._newest_contract().get("budget_denomination")

    def _g2(self, metas: list[dict]) -> list[dict]:
        w, s = self.window, self.h.stages
        todo = [m for m in metas if "G2" not in m["stages"]]

        def ref_for(m: dict) -> RoundRef:      # a fresh A round per candidate
            return w.a[int(m["id"][1:]) % len(w.a)]

        refs = [ref_for(m) for m in todo]
        res = self._run_jobs(self._king_jobs(refs, s.g2_hours) + [
            Job(f"{m['id']}-G2", self._replay_spec(self.tree(m["id"]), ref_for(m), s.g2_hours),
                s.g2_hours) for m in todo])
        self._store_king(res, refs, s.g2_hours)
        for m in todo:
            r = res[f"{m['id']}-G2"]
            if self._fault(m, "G2", r):
                continue
            ref = ref_for(m)
            if (self.h.stages.reference == "cached"
                    and not self._cache_path(ref, s.g2_hours).is_file()):
                m["status"], m["reason"] = "stalled", "G2: king leg for this round failed"
                self.put(m)
                continue
            king_geo, _ = self._king_ref(ref, "g2", r)
            rel = stats.rel_improvement(r["geomean"], king_geo)
            passed = rel >= self.state["m2"]
            explore = (not passed and random.Random(m["id"]).random() < s.g2_explore_frac)
            m["stages"]["G2"] = {"pass": passed, "explore": explore, "rel": rel,
                                 "margin": self.state["m2"], "round": ref.round_id}
            if not (passed or explore):
                self._kill(m, "G2", f"screen {rel:+.2%} < {self.state['m2']:+.2%}")
                continue
            self.put(m)
        return [m for m in metas if (g := m["stages"].get("G2", {})).get("pass")
                or g.get("explore")]

    def _g3(self, metas: list[dict]) -> list[dict]:
        w, s = self.window, self.h.stages
        todo = [m for m in metas if "G3" not in m["stages"]]
        jobs = [Job(f"{m['id']}-G3-{ref.round_id}", self._replay_spec(
            self.tree(m["id"]), ref, s.g3_hours), s.g3_hours) for m in todo for ref in w.b]
        king_jobs = self._king_jobs(list(w.b), s.g3_hours) if todo else []
        res = self._run_jobs(king_jobs + jobs)
        self._store_king(res, list(w.b), s.g3_hours)
        if todo and self.h.stages.reference == "cached" and not all(
                self._cache_path(ref, s.g3_hours).is_file() for ref in w.b):
            for m in todo:
                m["status"], m["reason"] = "stalled", "G3: a king leg for the B rounds failed"
                self.put(m)
            return []
        koth = self.chain.koth_params(w.b[-1].epoch_start_block)
        for m in todo:
            rows = [res[f"{m['id']}-G3-{ref.round_id}"] for ref in w.b]
            bad = next((r for r in rows if "error" in r), None)
            if bad is not None:
                self._fault(m, "G3", bad)
                continue
            king_s, cand_s, rels = [], [], []
            for ref, r in zip(w.b, rows, strict=True):
                king_geo, ks = self._king_ref(ref, "g3", r)
                king_s += ks
                cand_s += scores_from_json(r["scores"])
                rels.append(stats.rel_improvement(r["geomean"], king_geo))
            rel = stats.mean(rels)
            seed = int(hashlib.sha256(f"{self.epoch}:{m['id']}".encode()).hexdigest()[:12], 16)
            res_b = stats.paired_lcb(king_s, cand_s, koth, seed=seed, lcb_margin=0.0)
            passed = rel >= self.state["m3"] and res_b.lcb > 0
            m["stages"]["G3"] = {"pass": passed, "rel": rel, "lcb": res_b.lcb,
                                 "margin": self.state["m3"], "epoch": self.epoch}
            explored = m["stages"].get("G2", {}).get("explore")
            if explored:
                self.event("explore", id=m["id"], g3_pass=passed)
            if passed:
                if m.get("status") not in POPULATION:     # a finalist stays a finalist
                    m["status"] = "member"
                    self.event("member", id=m["id"], parent=m.get("parent"))
                m["member_epoch"] = self.epoch
            elif m.get("status") not in POPULATION:
                self._kill(m, "G3", "did not confirm on the newest rounds")
                continue
            self.put(m)
        return [m for m in metas if m["stages"].get("G3", {}).get("pass")]

    def _trim_population(self) -> None:
        ms = self.members()
        for m in ms[self.h.search.population_k:]:
            m["status"], m["reason"] = "retired", "population full (outranked)"
            self.put(m)

    # ------------------------------------------------------------ G4 / G4.5
    def _ckpt_dir(self, cid: str, ref: RoundRef) -> Path:
        return self.wd / "candidates" / cid / "ckpt" / ref.round_id

    def _g4(self) -> dict | None:
        """Full-contract replays of the top members not yet G4-judged this epoch;
        returns the chosen finalist. Each leg's result is kept on the candidate
        as soon as it exists, so a cap or crash never pays for the same leg twice.
        Finalists and submitted trees are never re-run."""
        s, w = self.h.stages, self.window
        finalists = [m for m in self.members() if m.get("status") == "member"
                     and (m["stages"].get("G4") or {}).get("epoch") != self.epoch]
        finalists = finalists[: s.g4_finalists]
        if not finalists:
            return None
        rounds = w.newest(s.g4_rounds)
        newest = rounds[-1]
        # Budget bound: the contract's hard wall plus staging/eval slack.
        g4h = s.g4_hours
        wall_h = (float(self.chain.training.primary_size.max_train_seconds) / 3600.0 + 0.5
                  if g4h is None else g4h + 0.25)
        jobs = [Job(f"{m['id']}-G4-{ref.round_id}", self._replay_spec(
            self.tree(m["id"]), ref, g4h,
            ckpt=self._ckpt_dir(m["id"], ref) if ref is newest else None), wall_h)
            for m in finalists for ref in rounds
            if ref.round_id not in m.get("g4_legs", {})]
        self.phase(f"G4: {len(finalists)} finalist(s), {len(jobs)} full-budget replay(s)")
        res = self._run_jobs(jobs)
        passers = []
        for m in finalists:
            legs = m.setdefault("g4_legs", {})
            for ref in rounds:
                r = res.get(f"{m['id']}-G4-{ref.round_id}")
                if r is not None and "error" not in r:
                    legs[ref.round_id] = {k: r.get(k) for k in
                                          ("geomean", "king_geomean", "verdict")}
            self.put(m)
            bad = next((r for ref in rounds
                        if (r := res.get(f"{m['id']}-G4-{ref.round_id}")) and "error" in r),
                       None)
            if bad is not None and bad.get("candidate_fault"):
                self._kill(m, "G4", bad["error"][:300])
                continue
            if any(ref.round_id not in legs for ref in rounds):
                continue                       # infra / cap: the missing legs run later
            rows = [legs[ref.round_id] for ref in rounds]
            rels = [stats.rel_improvement(r["geomean"], r["king_geomean"]) for r in rows]
            # A short (smoke) G4 leg carries no verdict: count "better than the
            # receipt king" instead. Full-contract legs use the round's own rule.
            wins = sum(1 for r, rel in zip(rows, rels, strict=True)
                       if (r.get("verdict") or {}).get("wins")
                       or (s.g4_hours is not None and rel > 0))
            g4 = {"pass": wins >= s.g4_min_wins and stats.mean(rels) > 0, "wins": wins,
                  "rounds": len(rows), "rel": stats.mean(rels), "rels": rels,
                  "epoch": self.epoch,
                  "lcbs": [(r.get("verdict") or {}).get("lcb") for r in rows]}
            m["stages"]["G4"] = g4
            self.put(m)
            self.event("g4", id=m["id"], passed=g4["pass"], wins=wins)
            if g4["pass"]:
                passers.append(m)
        self.save()
        return max(passers, key=lambda m: m["stages"]["G4"]["rel"]) if passers else None

    def _build_pool_cli(self, out: Path, as_of: str, sources: str) -> None:
        """``cascade-pool build`` with a patient per-request timeout, retried as a
        whole: the builder aborts on any request that fails 3 times, and the
        public sources do have transient failures (observed: one Open-Meteo
        archive call)."""
        argv = ["cascade-pool", "build", "--out", str(out), "--as-of", as_of, "--overwrite",
                "--timeout", "60"]
        if sources:
            argv += ["--sources", sources]
        for attempt in range(1, POOL_BUILD_ATTEMPTS + 1):
            try:
                subprocess.run(argv, check=True, timeout=7200)
                return
            except subprocess.CalledProcessError:
                if attempt == POOL_BUILD_ATTEMPTS:
                    raise
                log.warning("pool C build failed (attempt %d); retrying", attempt)
                self._interruptible_sleep(120.0 * attempt)

    def pool_c(self) -> Path | None:
        """Today's freshly built pool (built at most once a day, AFTER finalists froze)."""
        day = dt.datetime.fromtimestamp(self._now(), dt.UTC).strftime("%Y-%m-%d")
        out = self.wd / "pools" / "C" / day
        if (out / "metadata.json").is_file():
            return out
        try:
            self._pool_builder(out, day, self.h.stages.g45_sources)
        except Exception as e:  # noqa: BLE001
            log.warning("pool C build failed: %s", e)
            return None
        return out if (out / "metadata.json").is_file() else None

    def _g45(self, m: dict) -> dict:
        from ...shared.receipt import load_receipt

        newest = self.window.newest(1)[0]
        ckpt = self._ckpt_dir(m["id"], newest)
        if not self.h.stages.g45_enabled:
            return {"skipped": True}
        pool = self.pool_c()
        if pool is None or not ckpt.is_dir():
            return {"pass": False, "unavailable": "no pool C" if pool is None
                    else "no kept G4 checkpoint"}
        manifest = load_receipt(newest.receipt_path.read_text(encoding="utf-8")
                                ).load_embedded_manifest()
        king = next((e for e in manifest.entries_for_role("king")), None)
        if king is None:
            return {"pass": False, "unavailable": "newest round names no king checkpoint"}
        day_seed = int(hashlib.sha256(pool.name.encode()).hexdigest()[:12], 16)
        res = self._run_jobs([Job(f"{m['id']}-G45", {
            "kind": "eval_pool", "inputs": {"pool": str(pool), "ckpt_cand": str(ckpt)},
            "params": {"seed": day_seed, "block": newest.epoch_start_block,
                       "pointers": {"king": king.trained_pointer}}}, 0.5)])
        r = res[f"{m['id']}-G45"]
        if "error" in r:
            return {"pass": False, "unavailable": r["error"][:300]}
        king_s, cand_s = scores_from_json(r["scores"]["king"]), scores_from_json(r["scores"]["cand"])
        from ...eval.scoring import global_geomean
        rel = stats.rel_improvement(global_geomean(cand_s), global_geomean(king_s))
        rb = stats.paired_lcb(king_s, cand_s, self.chain.koth_params(newest.epoch_start_block),
                              seed=day_seed, lcb_margin=0.0)
        return {"pass": bool(rb.lcb > 0 and rel > 0), "rel": rel, "lcb": rb.lcb,
                "pool": pool.name, "n_windows": len(cand_s)}

    def finalize(self, m: dict) -> None:
        g45 = self._g45(m)
        m["stages"]["G4.5"] = g45
        m["status"] = "finalist"
        self.put(m)
        self.event("finalist", id=m["id"], g45=g45)
        if self.submitter is None:
            return
        evidence = {"g4": m["stages"]["G4"], "g45": None if g45.get("skipped") else g45,
                    "g45_required": self.h.stages.g45_enabled, "epoch": self.epoch,
                    "note": m.get("note"), "parent": m.get("parent")}
        out = self.submitter.offer(m["id"], self.tree(m["id"]), evidence)
        if out.get("action") == "submitted":
            m["status"] = "submitted"
            self.put(m)
        self.event("g5", id=m["id"], **{k: v for k, v in out.items() if k != "id"})

    # ---------------------------------------------------------------- cycle
    def cycle(self) -> str:
        """One cycle; returns ``"ran"``, ``"wait"`` or ``"budget"``."""
        try:
            self.refresh_window()
            self._check_budget()
            if self.window is None:
                return "wait"
            if not self.ensure_baseline():
                return "wait"
            self._check_budget()
            resumed = []
            for m in self.all_metas():
                if m.get("status") == "proposed":
                    # Its worker never reported (the judge died mid-proposal): the
                    # tree may be half-edited, so it is not resumed.
                    self._kill(m, "worker", "orphaned by a judge restart")
                    continue
                if m.get("status") not in ("in_gauntlet", "stalled"):
                    continue
                if m["status"] == "stalled":
                    m["stalls"] = int(m.get("stalls", 0)) + 1
                    if m["stalls"] > MAX_INFRA_RETRIES:
                        self._kill(m, "infra", f"gave up after {m['stalls']} stalls: "
                                               f"{m.get('reason', '')}")
                        continue
                if m.get("epoch") != self.epoch:
                    # Window-dependent results belong to the old epoch.
                    for st in ("G2", "G3"):
                        m["stages"].pop(st, None)
                    m["epoch"] = self.epoch
                m["status"] = "in_gauntlet"
                self.put(m)
                resumed.append(m)
            self.phase(f"cycle {self.state['cycle']}: proposing "
                       f"{self.h.search.proposals_per_cycle}")
            props = self.propose(self.h.search.proposals_per_cycle)
            fresh = self.accept_outcomes(self.workers.run(props)) if props else []
            batch = resumed + fresh
            self.phase(f"cycle {self.state['cycle']}: G0-G1 on {len(batch)}")
            batch = self._g1(self._g0(batch))
            self._check_budget()
            self.phase(f"cycle {self.state['cycle']}: G2 on {len(batch)}")
            batch = self._g2(batch)
            self._record_progress("G2")
            self._check_budget()
            self.phase(f"cycle {self.state['cycle']}: G3 on {len(batch)}")
            self._g3(batch)
            self._record_progress("G3")
            self._check_budget()
            self._trim_population()
            self.state["cycle"] = int(self.state["cycle"]) + 1
            if self.state["cycle"] % max(1, self.h.stages.g4_every_cycles) == 0:
                chosen = self._g4()
                self._check_budget()
                if chosen is not None:
                    self.finalize(chosen)
            self._record_progress("cycle")
            self.phase("idle")
            return "ran"
        except BudgetWait:
            self.state["status"] = "spend cap reached; waiting for 00:00 UTC"
            self.save()
            return "budget"
        finally:
            with contextlib.suppress(Exception):
                self.executor.reap()

    def run(self, *, max_cycles: int = 0, wait_seconds: float = 900.0,
            park_on_stop: bool = False) -> None:
        """Cycle until STOP (or ``max_cycles``). ``park_on_stop`` (the compose
        service) idles on STOP instead of exiting, so a restart policy does not
        loop: deleting STOP resumes."""
        lock = open(self.wd / ".lock", "w")  # noqa: SIM115 — held for the run's lifetime
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"another gauntlet is running on {self.wd}") from e
        try:
            # Parked for an image update (deploy/harness/updater.sh): an old
            # container restarted by its restart policy, or the new one before
            # the updater clears the handshake, waits instead of starting work.
            while (self.wd / RESTART_ACK).exists() and (self.wd / RESTART_FILE).exists():
                self.state["phase"] = "parked for an image update"
                self.save()
                self._sleep(10.0)
            n = 0
            while True:
                while not self.stopped() and (not max_cycles or n < max_cycles):
                    if self._restart_requested():
                        return
                    outcome = self.cycle()
                    n += outcome == "ran"
                    if outcome == "budget":
                        now = dt.datetime.fromtimestamp(self._now(), dt.UTC)
                        midnight = (now + dt.timedelta(days=1)).replace(
                            hour=0, minute=0, second=0, microsecond=0)
                        self._interruptible_sleep((midnight - now).total_seconds() + 60)
                    elif outcome == "wait":
                        self._interruptible_sleep(wait_seconds)
                if not (park_on_stop and self.stopped()):
                    break
                self.phase("stopped: delete STOP to resume")
                with contextlib.suppress(Exception):
                    self.executor.close()        # no pod idles (and bills) while stopped
                while self.stopped() and not (self.wd / RESTART_FILE).exists():
                    self._sleep(30.0)
                if self._restart_requested():
                    return
            if self.stopped():
                self.phase("stopped")
        finally:
            with contextlib.suppress(Exception):
                self.executor.close()
            lock.close()

    def _restart_requested(self) -> bool:
        """The updater's handshake, honoured only between cycles (no job in flight)."""
        if not (self.wd / RESTART_FILE).exists():
            return False
        (self.wd / RESTART_ACK).write_text(str(self._now()))
        self.phase("restarting for an image update")
        return True

    def _interruptible_sleep(self, seconds: float) -> None:
        end = self._now() + seconds
        while (self._now() < end and not self.stopped()
               and not (self.wd / RESTART_FILE).exists()):
            self._sleep(min(60.0, max(0.0, end - self._now())))

    # ---------------------------------------------------------- progress
    def candidate_progress(self, m: dict) -> dict:
        """``{score, stage, estimate}``: 0-100 toward beating the king.

        ``stage cap × min(1, estimate / target_improvement)`` for the DEEPEST
        stage the candidate passed, so cheap evidence can never score high
        however good its number looks (a G2 screen tops out at 40)."""
        need = max(1e-9, float(self.h.search.target_improvement))
        stages = m.get("stages", {})
        for stage, cap in PROGRESS_CAPS:
            st = stages.get(stage) or {}
            if st.get("pass") and isinstance(st.get("rel"), (int, float)):
                frac = max(0.0, min(1.0, float(st["rel"]) / need))
                return {"score": round(cap * frac, 1), "stage": stage, "estimate": st["rel"]}
        return {"score": 0.0, "stage": None, "estimate": None}

    def progress(self) -> dict:
        """The best candidate's dethrone progress this epoch (and its id)."""
        best = {"score": 0.0, "stage": None, "estimate": None, "id": None}
        for m in self.all_metas():
            if m.get("epoch") != self.epoch and m.get("member_epoch") != self.epoch:
                continue
            if m.get("status") not in POPULATION + ("in_gauntlet",):
                continue        # rejected at a later stage: its early pass no longer counts
            pr = self.candidate_progress(m)
            if pr["score"] > best["score"]:
                best = {**pr, "id": m["id"]}
        best["target"] = self.h.search.target_improvement
        return best

    def _record_progress(self, after: str = "cycle") -> None:
        pr = {"ts": self._now(), "epoch": self.epoch, "cycle": self.state.get("cycle"),
              "after": after, **self.progress()}
        self.state["progress"] = pr
        with open(self.wd / "progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(pr) + "\n")
        log.info("%s", progress_bar(pr))

    # ----------------------------------------------------------- operator
    def _write_operator_status(self) -> None:
        """``operator/status.json`` — what the operator agent may read: the
        funnel, the population (no G3+ numbers), spend, pending finalists."""
        metas = self.all_metas()
        funnel: dict[str, int] = {}
        for m in metas:
            if m.get("epoch") != self.epoch:
                continue
            key = m.get("status", "?")
            if key == "dead":
                key = f"dead@{str(m.get('reason', '?')).split(':', 1)[0]}"
            funnel[key] = funnel.get(key, 0) + 1
        w = self.window
        doc = {
            "epoch": self.epoch, "cycle": self.state.get("cycle"),
            "phase": self.state.get("phase"), "status": self.state.get("status"),
            "updated": self.state.get("updated"),
            "window": None if w is None else {
                "a_rounds": len(w.a), "b_rounds": len(w.b),
                "newest_block": w.all[-1].epoch_start_block},
            "screen_margin": self.state.get("m2"), "funnel": funnel,
            "population": [{"id": m["id"], "parent": m.get("parent"), "note": m.get("note"),
                            "screen_rel": m["stages"].get("G2", {}).get("rel")}
                           for m in metas if m.get("status") in POPULATION],
            "finalists": [{"id": m["id"], "status": m["status"], "note": m.get("note")}
                          for m in metas if m.get("status") in ("finalist", "submitted")],
            "recent": [f"{m['id']}: {self.summary(m)}" for m in metas[-15:]],
        }
        if self.executor is not None and hasattr(self.executor, "ledger"):
            doc["spend_today_usd"] = round(self.executor.ledger.spent_today(), 2)
            doc["daily_cap_usd"] = self.h.compute.daily_usd_cap
        if self.submitter is not None:
            doc["pending_submissions"] = [p["id"] for p in self.submitter.pending()]
        _atomic_json(self.wd / "operator" / "status.json", doc)
