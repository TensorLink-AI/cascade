"""Staging tier of W-E2E: one lane B, C or D cycle on TESTNET with ONE cheap
real Lium GPU (retrospective 2026-11-28 item P1b).

The CI tier (tests/accept/w_e2e/) proves the code path on fakes. This runner
proves the deployed release, the pinned images and the real hosts on testnet:

  lane b   deploy --expert-domain -> lane-b fund (1 Lium GPU, <= $1/h) -> track
           -> settlement (rebase / bench / judge / merge on the eval pod) ->
           signed expert manifest -> every validator's acceptance -> status
  lane c   lane-c upload -> commit -> fund (1 Lium GPU) -> track -> MarketCast
           judge on the mkt pod -> signed report -> cascade-audit marketcast ->
           status
  lane d   lane-d verify -> submit -> ready (commit, immediate reveal) -> track
           -> pool_d build / pin -> settlement on the eval host -> signed
           report -> cascade-audit lane-d -> status

DRY-RUN IS THE DEFAULT. A dry run reads only local state (git, chain.testnet
toml, provision.testnet.toml), parses every command it would run with the
real `cascade` CLI parser (so a renamed flag fails here, not on testnet) and
writes the plan to --report. Nothing is rented, signed, uploaded or
committed. `--spend` runs the plan; it refuses anything but `--network test`,
a non-testnet netuid, more than one GPU, a non-Lium provider and a price
cap above `--max-price` (default $1.00/h).

  .venv/bin/python -m tests.e2e_testnet.lanes --lane b                  # dry run
  .venv/bin/python -m tests.e2e_testnet.lanes --lane c --spend ...      # real run

Runbook: tests/e2e_testnet/LANES_RUNBOOK.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = ROOT / ".venv" / "bin"
TESTNET_TOML = ROOT / "chain.testnet.toml"
PROVISION_TOML = ROOT / "deploy" / "provision.testnet.toml"
TESTNET_NETUID = 259
LANES = ("b", "c", "d")
STAGES = ("preflight", "submit", "track", "settle", "verify", "status", "cleanup")
PAID = {"submit"}
# One-GPU Lium types cheap enough for a staging leg (retro P1b: 1x4090 / L40S).
CHEAP_GPUS = ("RTX4090", "L40S", "RTX3090", "A10", "RTX A6000")
TERMINAL = {"done", "failed", "withdrawn", "rejected", "spent", "cancelled", "settled"}


@dataclass
class Check:
    stage: str
    name: str
    status: str                         # PASS | FAIL | WARN | SKIP | PLAN
    detail: str = ""


@dataclass
class Report:
    lane: str
    mode: str
    started: float = field(default_factory=time.time)
    checks: list[Check] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)

    def add(self, stage: str, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(stage, name, status, detail))
        print(f"[{status:4}] {stage}/{name}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"lane": self.lane, "mode": self.mode, "started": self.started,
               "finished": time.time(), "ok": not self.failed,
               "checks": [asdict(c) for c in self.checks], "plan": self.plan}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)


# ── the plan ─────────────────────────────────────────────────────────────────


def _wallet(args) -> list[str]:
    return ["--wallet-name", args.wallet_name, "--wallet-hotkey", args.wallet_hotkey]


def _chain(args) -> list[str]:
    return ["--chain-toml", str(args.chain_toml), "--network", args.network]


def _gpu(args) -> list[str]:
    return ["--gpu", args.gpu, "--max-price", f"{args.max_price:g}",
            "--provider-key-env", "lium=LIUM_API_KEY"]


def plan_commands(args) -> list[dict]:
    """Every `cascade` command of the lane's cycle, in order: {stage, name, argv, paid}."""
    lane, intake = args.lane, args.intake
    cmds: list[tuple[str, str, list[str], bool]] = []
    if lane == "b":
        cmds += [
            ("submit", "verify_repo", ["lane-b", "verify-repo", str(args.repo_dir),
                                       "--chain-toml", str(args.chain_toml)], False),
            ("submit", "deploy", ["deploy", str(args.repo_dir), *_chain(args), *_wallet(args),
                                  "--expert-domain", args.target, "--reveal-now"], True),
            ("submit", "fund", ["lane-b", "fund", intake, "--targets", args.target,
                                "--ref", args.ref or "<ref printed by deploy>",
                                "--train-hours", "0.5", *_gpu(args), "--yes",
                                "--chain-toml", str(args.chain_toml), *_wallet(args)], True),
            ("track", "status", ["lane-b", "status", intake, "--hotkey", args.hotkey or "<hotkey>",
                                 "--chain-toml", str(args.chain_toml)], False),
        ]
    elif lane == "c":
        cmds += [
            ("submit", "check", ["lane-c", "check", str(args.repo_dir), "--target", args.target,
                                 "--chain-toml", str(args.chain_toml)], False),
            ("submit", "upload", ["lane-c", "upload", str(args.repo_dir), "--target", args.target,
                                  "--hub-repo", args.hub_repo, "--chain-toml",
                                  str(args.chain_toml)], True),
            ("submit", "commit", ["lane-c", "commit", "--target", args.target,
                                  "--ref", args.ref or "<ref printed by upload>",
                                  *_chain(args), "--reveal-now", *_wallet(args)], True),
            ("submit", "fund", ["lane-c", "fund", intake, "--target", args.target,
                                "--ref", args.ref or "<ref printed by upload>", *_gpu(args),
                                "--yes", "--chain-toml", str(args.chain_toml),
                                *_wallet(args)], True),
            ("track", "status", ["lane-c", "status", intake, "--hotkey",
                                 args.hotkey or "<hotkey>"], False),
        ]
    else:
        intake = args.lane_d_intake or intake
        cmds += [
            ("submit", "verify", ["lane-d", "verify", str(args.entry_dir),
                                  "--chain-toml", str(args.chain_toml)], False),
            ("submit", "submit", ["lane-d", "submit", intake, str(args.entry_dir),
                                  "--context", str(args.context), *_chain(args),
                                  *_wallet(args)], True),
            ("submit", "ready", ["lane-d", "ready", intake, "--digest",
                                 args.digest or "<digest printed by submit>",
                                 "--context", str(args.context), "--licence", "MIT", "--attest",
                                 *_chain(args), *_wallet(args)], True),
            ("track", "status", ["lane-d", "status", intake, "--hotkey",
                                 args.hotkey or "<hotkey>"], False),
        ]
    return [{"stage": s, "name": n, "argv": a, "paid": p} for s, n, a, p in cmds]


def parse_with_cli(argv: list[str]) -> str | None:
    """None when the real `cascade` parser accepts ``argv``, else why not."""
    import contextlib
    import io

    from cascade.miner.cli import build_parser

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            build_parser().parse_args([*argv, "--json"])
    except SystemExit as e:
        return f"cascade CLI refused the arguments (exit {e.code}): {err.getvalue().strip()[-400:]}"
    return None


# ── guards (both modes) ──────────────────────────────────────────────────────


def guard_errors(args) -> list[str]:
    """Reasons this run may not proceed (testnet, one cheap Lium GPU)."""
    errs: list[str] = []
    if args.network != "test":
        errs.append(f"--network {args.network}: the staging cycle runs on testnet only")
    try:
        doc = tomllib.loads(Path(args.chain_toml).read_text(encoding="utf-8"))
        netuid = int(doc.get("subnet", {}).get("netuid", -1))
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        errs.append(f"--chain-toml unreadable: {e}")
        netuid = -1
    if netuid != TESTNET_NETUID:
        errs.append(f"--chain-toml netuid {netuid}: not the testnet subnet ({TESTNET_NETUID})")
    if args.lane in ("b", "c"):
        if args.gpu not in CHEAP_GPUS:
            errs.append(f"--gpu {args.gpu!r}: one cheap single-GPU Lium type only {CHEAP_GPUS}")
        if not (0 < args.max_price <= args.price_ceiling):
            errs.append(f"--max-price {args.max_price}: must be in (0, {args.price_ceiling}] $/h "
                        "(an uncapped or 0 price is refused)")
    return errs


def lane_armed(args) -> tuple[bool, str]:
    doc = tomllib.loads(Path(args.chain_toml).read_text(encoding="utf-8"))
    sec = {"b": "lane_b", "c": "lane_c", "d": "lane_d"}[args.lane]
    fb = int(doc.get(sec, {}).get("from_block", 0) or 0)
    if fb <= 0:
        return False, (f"[{sec}] from_block = 0 in {args.chain_toml}: the lane is off on testnet; "
                       "arming it is a signed testnet overlay (Approve tier), not this runner")
    return True, f"[{sec}] from_block = {fb}"


# ── stages ───────────────────────────────────────────────────────────────────


def _git(*a: str) -> str:
    p = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def stage_preflight(args, rep: Report) -> None:
    for e in guard_errors(args):
        rep.add("preflight", "guard", "FAIL", e)
    ok, why = lane_armed(args)
    rep.add("preflight", "lane_armed", "PASS" if ok else ("WARN" if args.dry_run else "FAIL"), why)
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    rep.add("preflight", "release", "WARN" if dirty else "PASS",
            f"HEAD {head[:12] or '?'}{' (dirty tree: not a release)' if dirty else ''}")
    if args.expect_release and not head.startswith(args.expect_release):
        rep.add("preflight", "release_match", "FAIL",
                f"local HEAD {head[:12]} != deployed release {args.expect_release}")
    try:
        prov = tomllib.loads(PROVISION_TOML.read_text(encoding="utf-8"))
        img = str(prov.get("provisioner", {}).get("image", ""))
        rep.add("preflight", "image_pin", "PASS" if "@sha256:" in img or not img else "WARN",
                f"provision.testnet.toml image = {img or '(lium template, bootstrap mode)'}")
    except (OSError, tomllib.TOMLDecodeError) as e:
        rep.add("preflight", "image_pin", "WARN", f"provision.testnet.toml: {e}")
    if args.lane in ("b", "c"):
        have = bool(os.environ.get("LIUM_API_KEY"))
        rep.add("preflight", "lium_key", "PASS" if have else ("PLAN" if args.dry_run else "FAIL"),
                "LIUM_API_KEY present (value never read)" if have else
                "LIUM_API_KEY unset (needed for --spend; check the Lium balance before renting)")
    for c in plan_commands(args):
        why = parse_with_cli(c["argv"])
        rep.add("preflight", f"cli_{c['stage']}_{c['name']}", "FAIL" if why else "PASS",
                why or "the cascade CLI accepts this command")


def _run_cli(argv: list[str], timeout: float) -> tuple[int, dict | None, str]:
    p = subprocess.run([str(VENV_BIN / "cascade"), *argv, "--json"], capture_output=True,
                       text=True, timeout=timeout, cwd=str(ROOT))
    doc = None
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            doc = json.loads(line)
            break
        except ValueError:
            continue
    return p.returncode, doc, p.stderr[-2000:]


REF_RE = re.compile(r"[\w.\-/]+@sha256:[0-9a-f]{64}")
DIGEST_RE = re.compile(r'"(?:entry_)?digest"\s*:\s*"((?:sha256:)?[0-9a-f]{64})"')


def stage_submit(args, rep: Report) -> None:
    names = [c["name"] for c in plan_commands(args) if c["stage"] == "submit"]
    for name in names:
        # re-planned per step: deploy / upload print the ref, submit the digest
        c = next(c for c in plan_commands(args) if c["name"] == name)
        if args.dry_run:
            rep.add("submit", name, "PLAN", shlex.join(["cascade", *c["argv"]]))
            continue
        if "<" in " ".join(c["argv"]):
            rep.add("submit", name, "FAIL", "an earlier step did not print the ref / digest")
            return
        rc, doc, err = _run_cli(c["argv"], timeout=1800)
        text = json.dumps(doc) if doc else ""
        rep.add("submit", name, "PASS" if rc == 0 else "FAIL", text[:400] or err[-400:])
        if rc != 0:
            return
        if not args.ref and (m := REF_RE.search(text)):
            args.ref = m.group(0)
        if args.lane == "d" and not args.digest and (m := DIGEST_RE.search(text)):
            args.digest = m.group(1)


def _status_row(args) -> dict | None:
    c = next(c for c in plan_commands(args) if c["stage"] == "track")
    rc, doc, _ = _run_cli(c["argv"], timeout=120)
    return doc if rc == 0 else None


def stage_track(args, rep: Report) -> None:
    if args.dry_run:
        rep.add("track", "status", "PLAN", f"poll every {args.poll:g} s until a terminal status, "
                f"at most {args.leg_timeout:g} s")
        return
    deadline = time.time() + args.leg_timeout
    last = None
    while time.time() < deadline:
        doc = _status_row(args) or {}
        st = str((doc.get("data") or doc).get("status", "")) if isinstance(doc, dict) else ""
        if st != last:
            rep.add("track", "phase", "PASS", st or "(no row yet)")
            last = st
        if st in TERMINAL:
            break
        time.sleep(args.poll)
    rep.add("track", "terminal", "PASS" if last in TERMINAL else "FAIL", str(last))


VERDICT = re.compile(r'"verdict"\s*:\s*\{')


def stage_settle(args, rep: Report) -> None:
    what = {"b": "the signed lane B expert manifest (status/lane_b.json verdict)",
            "c": "the signed MarketCast report for the entry's round",
            "d": "the signed lane D settlement report"}[args.lane]
    if args.dry_run:
        rep.add("settle", "verdict", "PLAN", f"wait <= {args.settle_timeout:g} s for {what}")
        return
    deadline = time.time() + args.settle_timeout
    while time.time() < deadline:
        doc = _status_row(args)
        if doc and VERDICT.search(json.dumps(doc)):
            rep.add("settle", "verdict", "PASS", json.dumps(doc)[:600])
            return
        time.sleep(args.poll)
    rep.add("settle", "verdict", "FAIL", f"no verdict within {args.settle_timeout:g} s ({what})")


def stage_verify(args, rep: Report) -> None:
    cmd = {"b": None,
           "c": [str(VENV_BIN / "cascade-audit"), "marketcast", "--config", str(args.chain_toml),
                 "--network", args.network],
           "d": [str(VENV_BIN / "cascade-audit"), "latest", "--tier", "0", "--config",
                 str(args.chain_toml)]}[args.lane]
    if cmd is None:
        rep.add("verify", "validators", "PLAN" if args.dry_run else "SKIP",
                "every testnet validator's lane B acceptance row for the manifest "
                "(cascade-ops status / validator status/*.json)")
        return
    if args.dry_run:
        rep.add("verify", "audit", "PLAN", shlex.join(cmd))
        return
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    rep.add("verify", "audit", "PASS" if p.returncode == 0 else "FAIL",
            (p.stdout or p.stderr)[-600:])


def stage_status(args, rep: Report) -> None:
    if args.dry_run:
        c = next(c for c in plan_commands(args) if c["stage"] == "track")
        rep.add("status", "miner_view", "PLAN", shlex.join(["cascade", *c["argv"], "--json"]))
        return
    doc = _status_row(args) or {}
    shown = bool(VERDICT.search(json.dumps(doc)))
    rep.add("status", "miner_view", "PASS" if shown else "FAIL",
            "the miner CLI shows the verdict" if shown else "verdict not visible to the miner")


def stage_cleanup(args, rep: Report) -> None:
    rep.add("cleanup", "pods", "PLAN" if args.dry_run else "WARN",
            "confirm no testnet pod is left on the staging Lium key (`lium ps`); the leg's pod "
            "is torn down by the trainer at harvest")


RUNNERS = {"preflight": stage_preflight, "submit": stage_submit, "track": stage_track,
           "settle": stage_settle, "verify": stage_verify, "status": stage_status,
           "cleanup": stage_cleanup}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tests.e2e_testnet.lanes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", choices=LANES, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="(default) plan + local checks only; nothing rented or signed")
    mode.add_argument("--spend", action="store_true",
                      help="run the plan for real on testnet (rents ONE Lium GPU for b/c)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--chain-toml", type=Path, default=TESTNET_TOML)
    ap.add_argument("--network", default="test")
    ap.add_argument("--intake", default="http://127.0.0.1:8790")
    ap.add_argument("--lane-d-intake", default="")
    ap.add_argument("--wallet-name", default="miner0")
    ap.add_argument("--wallet-hotkey", default="default")
    ap.add_argument("--hotkey", default="", help="the staging miner's ss58 (for status)")
    ap.add_argument("--repo-dir", type=Path, default=ROOT / "tests" / "e2e_testnet" / "generators" / "mv32",
                    help="lane B generator tree / lane C recipe tree")
    ap.add_argument("--entry-dir", type=Path, default=ROOT / "_e2e" / "lane_d_entry",
                    help="lane D entry directory (from `cascade lane-d export`)")
    ap.add_argument("--target", default="", help="lane B domain (default energy) / lane C mkt:<suite>")
    ap.add_argument("--ref", default="", help="the committed ref, once deploy/upload printed it")
    ap.add_argument("--digest", default="", help="lane D entry digest, once submit printed it")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--hub-repo", default="cascade-testnet/e2e-lane-c")
    ap.add_argument("--gpu", default="RTX4090")
    ap.add_argument("--max-price", type=float, default=1.0, help="$/h cap on the one GPU")
    ap.add_argument("--price-ceiling", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--expect-release", default="", help="the deployed release sha (prefix)")
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--leg-timeout", type=float, default=4 * 3600)
    ap.add_argument("--settle-timeout", type=float, default=4 * 3600)
    ap.add_argument("--report", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.spend:
        args.dry_run = False
    if not args.target:
        args.target = {"b": "energy", "c": "mkt:market", "d": ""}[args.lane]
    report = args.report or ROOT / "_e2e" / f"lanes_{args.lane}_report.json"
    rep = Report(args.lane, "dry-run" if args.dry_run else "spend")
    rep.plan = plan_commands(args)
    stages = [s for s in args.stages.split(",") if s]
    unknown = [s for s in stages if s not in RUNNERS]
    if unknown:
        print(f"unknown stage(s) {unknown}; known {STAGES}", file=sys.stderr)
        return 2
    if "preflight" not in stages:
        stages.insert(0, "preflight")              # never spend without the guards
    for s in stages:
        RUNNERS[s](args, rep)
        if s == "preflight" and rep.failed and not args.dry_run:
            rep.add("preflight", "blocked", "FAIL", "a failed preflight blocks every paid stage")
            break
    rep.write(report)
    print(f"report: {report}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
