"""``cascade gauntlet …`` — run, watch, steer and approve the mining gauntlet."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import load_harness_config

EXAMPLE_TOML = Path(__file__).resolve().parents[3] / "deploy" / "harness" / "harness.example.toml"


def _cfg(args):
    return load_harness_config(args.config)


def _queue_dir(h) -> Path:
    q = Path(h.workers.queue_dir)
    return q if q.is_absolute() else Path(h.workdir) / q


def build(h, *, with_workers: bool = True):
    """The wired engine for config ``h`` (executor, workers, submitter)."""
    from .executor import make_executor
    from .gauntlet import Gauntlet
    from .submit import Submitter
    from .workers import InlineWorkers, QueueWorkers

    wd = Path(h.workdir)
    workers = None
    if with_workers:
        workers = (QueueWorkers(_queue_dir(h), timeout=h.workers.agent_timeout * 2)
                   if h.workers.mode == "queue"
                   else InlineWorkers(h.workers, wd, parallel=h.search.proposals_per_cycle))
    return Gauntlet(h, executor=make_executor(h, wd), workers=workers,
                    submitter=Submitter(h.submit, wd))


def _cmd_run(args) -> int:
    h = _cfg(args)
    g = build(h)
    g.run(max_cycles=args.max_cycles or h.search.max_cycles, park_on_stop=args.park_on_stop)
    return 0


def _cmd_tick(args) -> int:
    h = _cfg(args)
    g = build(h, with_workers=False)
    new = g.refresh_window(force=True)
    ready = g.window is not None and g.ensure_baseline()
    print(json.dumps({"new_epoch": new, "epoch": g.epoch, "baseline_ready": ready,
                      "status": g.state.get("status", "")}, indent=1))
    g.executor.close()
    return 0


def _cmd_status(args) -> int:
    h = _cfg(args)
    doc = json.loads((Path(h.workdir) / "operator" / "status.json").read_text())
    if args.json:
        print(json.dumps(doc, indent=1))
        return 0
    print(f"epoch {doc['epoch']}  cycle {doc['cycle']}  phase: {doc['phase']}")
    if doc.get("status"):
        print(f"status: {doc['status']}")
    if doc.get("spend_today_usd") is not None:
        print(f"spend today: ${doc['spend_today_usd']:.2f} / ${doc['daily_cap_usd']:.2f}")
    print("funnel:", ", ".join(f"{k}={v}" for k, v in sorted(doc.get("funnel", {}).items())))
    for m in doc.get("population", []):
        print(f"  member {m['id']} (from {m['parent']}): {m.get('note') or ''}")
    for m in doc.get("finalists", []):
        print(f"  {m['status']} {m['id']}: {m.get('note') or ''}")
    for pid in doc.get("pending_submissions", []):
        print(f"  PENDING APPROVAL: {pid}  (cascade gauntlet approve {pid} --hotkey HK "
              "--confirm SUBMIT)")
    return 0


def _cmd_approve(args) -> int:
    from .submit import Submitter

    if args.confirm != "SUBMIT":
        print("refusing: pass --confirm SUBMIT (this spends the hotkey and funds a leg)",
              file=sys.stderr)
        return 2
    h = _cfg(args)
    s = Submitter(h.submit, Path(h.workdir))
    pend = {p["id"]: p for p in s.pending()}
    if args.id not in pend:
        print(f"no pending finalist {args.id}", file=sys.stderr)
        return 2
    print(json.dumps(pend[args.id]["evidence"], indent=1))
    out = s.approve(args.id, args.hotkey)
    print(json.dumps(out, indent=1))
    return 0 if out["action"] == "submitted" else 1


def _cmd_reject(args) -> int:
    from .submit import Submitter

    h = _cfg(args)
    Submitter(h.submit, Path(h.workdir)).reject(args.id)
    return 0


def _cmd_stop(args) -> int:
    h = _cfg(args)
    (Path(h.workdir) / "STOP").touch()
    print(f"stop requested: {Path(h.workdir) / 'STOP'} (the judge stops after this stage)")
    return 0


def _cmd_worker(args) -> int:
    from .workers import serve_queue

    h = _cfg(args)
    home = Path(args.home) if args.home else _queue_dir(h).parent
    serve_queue(_queue_dir(h), h.workers, home=home, once=args.once)
    return 0


def _cmd_check(args) -> int:
    """Preflight: everything that would make a run fail later, found now."""
    from ..ralph import preflight, resolve_provider

    h = _cfg(args)
    problems, notes = [], []
    if h.workers.mode == "inline" or args.workers:
        try:
            ok, msg = preflight(resolve_provider(h.workers))
            (notes if ok else problems).append(f"workers LLM: {msg}")
        except ValueError as e:
            problems.append(f"workers LLM: {e}")
    if h.compute.executor == "lium":
        key = Path(h.compute.ssh_key).expanduser()
        if not Path(str(key) + ".pub").is_file():
            problems.append(f"no SSH key pair at {key} (ssh-keygen -t ed25519 -f {key} -N '')")
        if shutil.which("lium") is None:
            problems.append("lium CLI not found (pip install lium.io)")
        for tool in ("ssh", "tar"):
            if shutil.which(tool) is None:
                problems.append(f"{tool} not found")
        notes.append(f"lium: cap ${h.compute.daily_usd_cap:.2f}/day, "
                     f"≤ ${h.compute.max_price_per_hour:.2f}/h, ≤ {h.compute.max_parallel} pods")
    root = Path(h.rounds.snapshot_root)
    n_snaps = len(list(root.glob("snapshots/*/POOL_SHA256"))) + int((root / "POOL_SHA256").is_file())
    notes.append(f"revealed snapshots under {root}: {n_snaps}"
                 + ("" if h.rounds.sync_reveals else " (sync_reveals off)"))
    if h.submit.mode == "autonomous":
        notes.append(f"submit: AUTONOMOUS, {len(h.submit.hotkeys)} hotkey(s), "
                     f"margin {h.submit.margin:.3f}, ≤ {h.submit.max_per_day}/day")
    else:
        notes.append(f"submit: {h.submit.mode}")
    for n in notes:
        print(f"ok   {n}")
    for p in problems:
        print(f"FAIL {p}")
    return 1 if problems else 0


def _cmd_init(args) -> int:
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "harness.toml"
    if dst.exists() and not args.force:
        print(f"{dst} exists (--force to overwrite)", file=sys.stderr)
        return 2
    shutil.copyfile(EXAMPLE_TOML, dst)
    op = d / "gauntlet" / "operator"
    op.mkdir(parents=True, exist_ok=True)
    (op / "DIRECTIVES.md").touch()
    (op / "NOTEBOOK.md").touch()
    print(f"wrote {dst}; edit it, then: cascade gauntlet check --config {dst}")
    return 0


def _cmd_selftest(args) -> int:
    from .selftest import run_selftest
    return run_selftest()


def add_gauntlet(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("gauntlet", help="The multi-stage mining harness (docs/GAUNTLET.md).")
    gs = p.add_subparsers(dest="gauntlet_cmd", required=True)

    def cmd(name: str, fn, help_: str):
        sp = gs.add_parser(name, help=help_)
        if name not in ("init", "selftest"):
            sp.add_argument("--config", type=Path, default=Path("harness.toml"))
        sp.set_defaults(func=fn)
        return sp

    r = cmd("run", _cmd_run, "Run the judge: propose → G0..G4.5 → G5, until STOP.")
    r.add_argument("--max-cycles", type=int, default=0)
    r.add_argument("--park-on-stop", action="store_true",
                   help="On STOP, idle (pods torn down) until STOP is deleted instead of "
                        "exiting: for a service with a restart policy (docker compose).")
    cmd("tick", _cmd_tick, "Refresh the round window and baseline the epoch, then exit.")
    s = cmd("status", _cmd_status, "Show the funnel, population, spend and pending finalists.")
    s.add_argument("--json", action="store_true")
    a = cmd("approve", _cmd_approve, "Submit a pending finalist (spends the hotkey).")
    a.add_argument("id")
    a.add_argument("--hotkey", required=True)
    a.add_argument("--confirm", default="")
    j = cmd("reject", _cmd_reject, "Drop a pending finalist.")
    j.add_argument("id")
    cmd("stop", _cmd_stop, "Ask the judge to stop after the current stage.")
    w = cmd("worker", _cmd_worker, "Drain the proposal queue (the worker container).")
    w.add_argument("--once", action="store_true")
    w.add_argument("--home", default="")
    c = cmd("check", _cmd_check, "Preflight the config, keys, LLM and Lium setup.")
    c.add_argument("--workers", action="store_true", help="Also preflight the worker LLM.")
    i = cmd("init", _cmd_init, "Write an example harness.toml and the operator files.")
    i.add_argument("--dir", default=".")
    i.add_argument("--force", action="store_true")
    cmd("selftest", _cmd_selftest,
        "One full cycle on synthetic rounds and fake compute (no GPU, network or LLM).")


def main(argv=None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="cascade gauntlet")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_gauntlet(sub)
    args = parser.parse_args(["gauntlet", *(argv if argv is not None else sys.argv[1:])])
    return int(args.func(args))
