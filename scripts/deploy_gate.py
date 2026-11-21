#!/usr/bin/env python3
"""Deploy gate: refuse an unsafe service deploy before it happens, and verify it
after. Read-only — it never restarts, kills or edits anything.

Built from the 2026-11-16 review of every logged deploy incident that cascaded
to external validators or miners (restarts with legs in flight, consensus
changes without an activation block, stale code running after a merge,
version skew, restarts inside the validator's scoring window):

  deploy_gate.py check  --service SERVICE --from REV --to REV
  deploy_gate.py verify --service SERVICE --release DIR

SERVICE is trainer, validator, intake-laned (the lane D intake, cascade/lane_d/)
or eval-host (the private eval pod running cascade/validator/private_worker.py).
``check`` classifies the diff (blast radius; every tracked path lands in an
AREA), then applies the timing guards for the service being restarted. Exit 0 = GO, 1 = REFUSE (reasons printed),
2 = usage/IO error. ``verify`` checks the running process after a restart.
``--ack`` names a REFUSE rule the owner explicitly accepted (repeatable); the
consensus rule can never be acked here — it needs release-then-activate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

SETTLEMENT_BLOCKS = 900
BLOCK_SECONDS = 12.0
PRE_SETTLEMENT_MIN = 20          # owner rule: no restart in the ~20 min before a settlement
POST_SETTLEMENT_MIN = 15         # manifest publish + validator pick-up after the boundary

# ── blast radius ─────────────────────────────────────────────────────────────

# What every validator (and satellite) computes identically: a change here that
# alters a settled round's outcome forks the fleet unless block-gated.
CONSENSUS = (
    r"^cascade/eval/", r"^cascade/audit/",
    r"^cascade/validator/(koth|evaluation|loop|weights|cascade|promotion_gate|gating|"
    r"king_check|lane_[a-z]+|windows|pool|era_gate|satellite|expert_lane|receipts)\.py$",
    r"^cascade/shared/(era|receipt|manifest|activation|netconfig|emission|decided|"
    r"chain|bench_report|calendar_launch|promotion|settlement_state)\.py$",
    r"^cascade/shared/config/",
)
# Code that runs on a rented pod: the static import closure of
# cascade/trainer/worker.py (module-level and lazy imports, every parent
# package). worker.py imports trainer.main, so the closure spans these
# packages; cascade/validator/private_worker.py (the eval host's entry point)
# reaches the same set. tests/accept/r8 recomputes the closure and fails when
# a new package enters it. The pod runs the whole baked repo (deploy/Dockerfile
# COPY . /root/cascade) and worker_code_fingerprint hashes every
# cascade/**/*.py, so over-covering here is the safe direction.
ON_POD_CODE = (
    r"^cascade/__init__\.py$",
    r"^cascade/(direct|eval|funding|generation|interface|miner|pool|provision|shared|trainer|"
    r"validator)/.*\.py$",
)
WORKER_IMAGE = (r"^deploy/Dockerfile", r"^deploy/entrypoint\.sh$", r"^\.dockerignore$",
                r"^scripts/check_tls_trust\.py$",       # RUN inside the image build
                r"^cascade/trainer/worker\.py$", r"^cascade/interface/",
                r"^uv\.lock$", r"^pyproject\.toml$", *ON_POD_CODE)
# What the eval host runs: the validator's release (private_worker + its
# imports) in a `uv sync --frozen` venv, re-synced with every validator deploy.
EVAL_HOST = (r"^cascade/validator/private_worker\.py$", *ON_POD_CODE,
             r"^uv\.lock$", r"^pyproject\.toml$", r"^\.python-version$")
AREAS = {
    "consensus": CONSENSUS,
    "worker_image": WORKER_IMAGE,
    "eval_host": EVAL_HOST,
    "shared": (r"^cascade/shared/",),
    "trainer": (r"^cascade/(trainer|provision|funding|generation|direct)/",),
    "validator": (r"^cascade/validator/",),
    "lane_d_intake": (r"^cascade/lane_d/",),            # the intake-laned service
    "pool": (r"^cascade/pool/",),                        # cascade-pool (eval pool builder)
    "miner_kit": (r"^cascade/miner/", r"^base_generator/"),
    "website": (r"^cascade/website/",),
    "chain_toml": (r"^chain(\.[a-z]+)?\.toml$", r"^decided_votes\.toml$"),
    "deploy": (r"^deploy/",),
    "benchmarks": (r"^benchmarks/",),
    "champions": (r"^champions/",),
    "ops_tooling": (r"^cascade/laned/", r"^scripts/"),  # owner-run tools, no service
    "ci": (r"^\.github/",),
}
SERVICES = ("trainer", "validator", "intake-laned", "eval-host")
# Services that serve or depend on the validator's scoring: no restart while a
# round is being judged (§10.5 S5 — the eval host runs that scoring).
SCORING_GUARDED = ("validator", "intake-laned", "eval-host")
# A consensus diff must carry its activation gate (DEC-CA-0045 typed blocks).
GATE_RE = re.compile(r"^\+.*\b[a-z0-9_]*(from_block|activation_block|_after)\b", re.M)


def classify(paths: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in paths:
        for area, pats in AREAS.items():
            if any(re.search(x, p) for x in pats):
                out.setdefault(area, []).append(p)
    return out


# ── state readers (box files; every reader tolerates absence) ────────────────

LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} ")


def _ts(line: str) -> datetime | None:
    m = LOG_TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC) if m else None


def tail(path: Path, max_bytes: int = 4 << 20) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []


BLOCK_RE = re.compile(r"\bblock (\d{6,})\b")


def last_block_sample(lines: list[str]) -> tuple[datetime, int] | None:
    for line in reversed(lines):
        m = BLOCK_RE.search(line)
        ts = _ts(line)
        if m and ts:
            return ts, int(m.group(1))
    return None


def minutes_to_next_settlement(sample: tuple[datetime, int], now: datetime) -> tuple[float, float]:
    """(minutes until the next boundary, minutes since the previous one)."""
    ts, blk = sample
    now_blk = blk + (now - ts).total_seconds() / BLOCK_SECONDS
    nxt = (int(now_blk) // SETTLEMENT_BLOCKS + 1) * SETTLEMENT_BLOCKS
    prev = nxt - SETTLEMENT_BLOCKS
    return ((nxt - now_blk) * BLOCK_SECONDS / 60, (now_blk - prev) * BLOCK_SECONDS / 60)


def validator_scoring(lines: list[str]) -> str:
    """The round the validator is still judging (a ``new manifest`` with no
    ``weights set`` / bench publish for it afterwards), or ""."""
    open_round = ""
    for line in lines:
        m = re.search(r"new manifest round=(\d+)", line)
        if m:
            open_round = m.group(1)
            continue
        if open_round and (f"round={open_round} weights set" in line
                           or f"validator-bench round={open_round}: published" in line):
            open_round = ""
    return open_round


# Funded-leg phases that hold a pod (a restart would kill or re-rent it).
POD_PHASES_FREE = {"waiting_gpu", "queued", "corpus", "generating", ""}


def legs_on_pods(work: Path) -> list[str]:
    out = []
    try:
        q = json.loads((work / "funded_queue.json").read_text())
        entries = q.get("entries", {})
        for e in (entries.values() if isinstance(entries, dict) else entries):
            if e.get("status") == "in_flight" and str(e.get("phase") or "") not in POD_PHASES_FREE:
                out.append(f"{str(e.get('hotkey', '?'))[:12]} ({e.get('phase')})")
    except (OSError, ValueError):
        pass
    try:
        st = json.loads((work / "era_state.json").read_text())
        if st.get("rents"):
            out.append(f"{len(st['rents'])} king/operator rent(s) open")
        if st.get("pending_publish"):
            out.append("a settlement manifest is pending publish")
        if st.get("verify_queue"):
            out.append(f"{len(st['verify_queue'])} harvest(s) awaiting verification")
    except (OSError, ValueError):
        pass
    return out


def freeze_window(now: datetime) -> bool:
    """Owner rule: no restarts Wed 22:00 – Thu 08:00 UTC."""
    wd, h = now.weekday(), now.hour
    return (wd == 2 and h >= 22) or (wd == 3 and h < 8)


# ── the gate ─────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    refuse: list[tuple[str, str]] = field(default_factory=list)   # (rule, why)
    notes: list[str] = field(default_factory=list)

    def no(self, rule: str, why: str) -> None:
        self.refuse.append((rule, why))


def gate(*, service: str, areas: dict[str, list[str]], diff_text: str, now: datetime,
         trainer_lines: list[str], validator_lines: list[str], work: Path,
         on_main: bool, acks: set[str]) -> Verdict:
    v = Verdict()
    if not on_main:
        v.no("provenance", "target revision is not on origin/main — releases build from main only")
    if "consensus" in areas:
        if not GATE_RE.search(diff_text):
            v.no("consensus", "touches consensus code (" + ", ".join(areas["consensus"][:4])
                 + ") with no activation block in the diff — every validator must switch at "
                   "the same block (release-then-activate); CI receipt-replay must be green")
        else:
            v.notes.append("consensus change carries an activation key — confirm the block is "
                           "after the satellite fleet upgrades, and announce it to validators")
    if "worker_image" in areas:
        v.notes.append("worker image changes: rebuild + repin from ONE source, only at a seed-epoch "
                       "boundary (a mid-epoch repin breaks king-pointer checks fleet-wide)")
    if "shared" in areas:
        v.notes.append("cascade/shared changed: restart BOTH trainer and validator onto it "
                       "(shared-code skew held a deposed king, 2026-10-02)")
    if service == "validator" and "eval_host" in areas:
        v.notes.append("the eval-host runs this code (cascade/validator/private_worker.py and its "
                       "imports, in the release venv): sync the eval host to this release first "
                       "(order eval host -> validator), then `verify --service eval-host`")
    if "lane_d_intake" in areas and service != "intake-laned":
        v.notes.append("cascade/lane_d changed: deploy the intake-laned service as well "
                       "(`check --service intake-laned`)")
    if "deploy" in areas:
        v.notes.append("deploy/ changed: pod bootstrap, systemd units and restart scripts apply "
                       "only when re-run or reinstalled, not through this restart")
    if "benchmarks" in areas:
        v.notes.append("benchmarks/ changed: bench hosts run it as its own project "
                       "(benchmarks/uv.lock, benchmarks/Dockerfile) — sync them outside a bench sweep")
    if "website" in areas:
        v.notes.append("website changed: republish it (scripts/publish_website.py) for miners")
    if "chain_toml" in areas:
        v.notes.append("chain.toml changed: diff it against the DEPLOYED file; decided votes "
                       "must reach validators via the signed overlay, not only our copy")
    if freeze_window(now):
        v.no("freeze", "Wed 22:00 – Thu 08:00 UTC: no restarts")
    sample = last_block_sample(validator_lines) or last_block_sample(trainer_lines)
    if sample is None:
        v.no("clock", "no recent chain block in the logs — cannot place the settlement window")
    else:
        to_next, since = minutes_to_next_settlement(sample, now)
        if to_next < PRE_SETTLEMENT_MIN:
            v.no("settlement", f"settlement in {to_next:.0f} min (no restarts in the last "
                               f"{PRE_SETTLEMENT_MIN} min before a boundary)")
        elif since < POST_SETTLEMENT_MIN:
            v.no("settlement", f"settlement {since:.0f} min ago — manifest publish / pick-up "
                               "still in progress")
        else:
            v.notes.append(f"next settlement in {to_next:.0f} min")
    if service in SCORING_GUARDED:
        rid = validator_scoring(validator_lines)
        if rid:
            v.no("scoring", f"validator is still judging round {rid} (wait for its weights set)")
    if service == "trainer":
        busy = legs_on_pods(work)
        if busy:
            v.no("in_flight", "work on pods would be killed or re-rented: " + "; ".join(busy))
    v.refuse = [(r, w) for r, w in v.refuse if r == "consensus" or r not in acks]
    return v


# ── verify (after the restart) ───────────────────────────────────────────────

# Box-side services and the console script their python runs; each logs
# "<service> up:" to <root>/<service>.log at start. The eval host is a rented pod
# the validator reaches over ssh per eval, so it has no box process (see
# verify_eval_host).
PATTERNS = {"trainer": "cascade-trainer", "validator": "cascade-validator",
            "intake-laned": "cascade-laned-intake"}
# ...or the package it may be launched from as ``python -P -m``.
MODULES = {"intake-laned": "cascade.lane_d"}

# validator.log evidence about the eval host (cascade/validator/eval_offload.py).
EVAL_HOST_OK = re.compile(r"private eval: (uploaded checkpoint|checkpoint \S+ already on)")
EVAL_HOST_BAD = re.compile(
    r"private eval host \S+ (compatibility probe (failed|unreadable)|cannot decode|"
    r"runs code older)|private eval \S+ failed on |PrivateEvalUnavailable|"
    r"private evaluation (worker is out of date|transport/worker failed|deadline expired|"
    r"response)")


def verify_eval_host(log: Path, *, since: datetime | None = None) -> list[str]:
    """The eval host after a sync, judged from the validator's side: since the
    sync, the validator ran a private eval on it and logged no probe or eval
    failure. No eval yet means not verified yet (widen --since-minutes or wait
    for the validator's next private eval)."""
    lines = [ln for ln in tail(log, 1 << 20) if since is None or (_ts(ln) or since) >= since]
    bad = [ln for ln in lines if EVAL_HOST_BAD.search(ln)]
    if bad:
        return [f"{len(bad)} eval-host failure line(s) in {log.name} since the sync, last: "
                f"{bad[-1][:240]}"]
    if not any(EVAL_HOST_OK.search(ln) for ln in lines):
        return [f"no private eval on the eval host in {log.name} since the sync — not verified yet"]
    return []


def _runs_module(service: str, argv: list[bytes]) -> bool:
    """``python [-P] -m <pkg>[.mod]`` for a service whose package is in MODULES."""
    pkg = MODULES.get(service)
    if not pkg:
        return False
    head = argv[1:4]
    if b"-m" not in head or head.index(b"-m") + 1 >= len(head):
        return False
    mod = head[head.index(b"-m") + 1]
    return mod == pkg.encode() or mod.startswith(pkg.encode() + b".")


def _proc_pids(service: str) -> list[int]:
    out = []
    needle = PATTERNS[service].encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        # the python interpreter running the console script — never a shell
        # that merely mentions the name (the pgrep -f self-match class)
        if len(argv) > 1 and argv[0].endswith(b"python") and (
                any(a.endswith(b"/" + needle) for a in argv[1:3]) or _runs_module(service, argv)):
            out.append(int(d.name))
    return out


def verify(service: str, release: Path, log: Path, *, since: datetime | None = None) -> list[str]:
    problems = []
    if service == "eval-host":
        return verify_eval_host(log, since=since)
    pids = _proc_pids(service)
    if len(pids) != 1:
        return [f"expected exactly one {service} process, found {len(pids)}: {pids}"]
    pid = pids[0]
    proc = Path(f"/proc/{pid}")
    env = dict(x.split("=", 1) for x in (proc / "environ").read_text("utf-8", "replace")
               .split("\0") if "=" in x)
    if env.get("PYTHONPATH") != str(release):
        problems.append(f"PYTHONPATH={env.get('PYTHONPATH')!r}, expected {release}")
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    if b"-P" not in argv[:3]:
        problems.append("not launched with python -P (cwd can shadow the release)")
    if (proc / "oom_score_adj").read_text().strip() != "-500":
        problems.append("oom_score_adj is not -500 (choom -n -500 -p PID)")
    if ".scope" not in (proc / "cgroup").read_text():
        problems.append("not in a systemd scope (systemd-run --scope ...)")
    lines = tail(log, 1 << 20)
    marker = f"{service} up:"
    ups = [ln for ln in lines if marker in ln and (since is None or (_ts(ln) or since) >= since)]
    if not ups:
        problems.append(f"no '{marker}' line in {log.name} since the restart")
    if any(ln.startswith("Traceback") for ln in lines[-400:]):
        problems.append(f"a Traceback in the last 400 lines of {log.name}")
    return problems


# ── CLI ──────────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          timeout=60, check=True).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--service", choices=SERVICES, required=True)
    c.add_argument("--from", dest="rev_from", required=True, help="running release's commit")
    c.add_argument("--to", dest="rev_to", required=True, help="commit to deploy")
    c.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    c.add_argument("--root", type=Path, default=Path(os.environ.get("CASCADE_ROOT", "/root/cascade")))
    c.add_argument("--ack", action="append", default=[],
                   help="a REFUSE rule the owner explicitly accepted (never 'consensus')")
    vp = sub.add_parser("verify")
    vp.add_argument("--service", choices=SERVICES, required=True)
    vp.add_argument("--release", type=Path, required=True,
                    help="the release dir the process must run (eval-host: judged from "
                         "validator.log instead, the release lives on the pod)")
    vp.add_argument("--root", type=Path, default=Path(os.environ.get("CASCADE_ROOT", "/root/cascade")))
    vp.add_argument("--since-minutes", type=float, default=15)
    a = ap.parse_args(argv)
    now = datetime.now(UTC)
    if a.cmd == "verify":
        log = a.root / ("validator.log" if a.service == "eval-host" else f"{a.service}.log")
        probs = verify(a.service, a.release.resolve(), log,
                       since=now - timedelta(minutes=a.since_minutes))
        for p in probs:
            print(f"VERIFY FAIL: {p}")
        print("VERIFIED" if not probs else f"{len(probs)} problem(s) — consider rolling back")
        return 1 if probs else 0
    try:
        paths = _git(a.repo, "diff", "--name-only", f"{a.rev_from}..{a.rev_to}").split()
        diff_text = _git(a.repo, "diff", "-U0", f"{a.rev_from}..{a.rev_to}")
        on_main = subprocess.run(["git", "-C", str(a.repo), "merge-base", "--is-ancestor",
                                  a.rev_to, "origin/main"], timeout=60).returncode == 0
    except (subprocess.SubprocessError, OSError) as e:
        print(f"git failed: {e}", file=sys.stderr)
        return 2
    areas = classify(paths)
    v = gate(service=a.service, areas=areas, diff_text=diff_text, now=now,
             trainer_lines=tail(a.root / "trainer.log"),
             validator_lines=tail(a.root / "validator.log"),
             work=a.root / "_train_work", on_main=on_main, acks=set(a.ack))
    print(f"{a.service} {a.rev_from[:9]}..{a.rev_to[:9]}: {len(paths)} file(s); areas: "
          + (", ".join(f"{k}({len(x)})" for k, x in areas.items()) or "none"))
    for n in v.notes:
        print(f"  note: {n}")
    for rule, why in v.refuse:
        print(f"  REFUSE [{rule}]: {why}")
    print("GO" if not v.refuse else "REFUSE")
    return 1 if v.refuse else 0


if __name__ == "__main__":
    sys.exit(main())
