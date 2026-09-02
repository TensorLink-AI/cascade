#!/usr/bin/env python3
"""Archive the subnet's best generator (the king) into this repo, with provenance.

Read-only against the chain: resolves the highest-incentive UID's committed
generator via this repo's own CLI (``cascade fetch king``), and when its
digest differs from the archived one, replaces ``champions/king/`` and
commits with a provenance record. The fetch never submits anything.

Usage:
    scripts/sync_king.py                         # mainnet (netuid 91)
    scripts/sync_king.py --network test --chain-toml chain.testnet.toml
    scripts/sync_king.py --push                  # also push the commit

The scheduled king-sync workflow runs this hourly: each dethrone becomes one
commit, and ``git log champions/king`` is the subnet's reign timeline. The
archive is scan-gated (scripts/scan_generator.py) — the king is untrusted
competitor code; a clean scan is still not a license to run it unsandboxed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "champions" / "king"
PROVENANCE = DEST / "PROVENANCE.json"


def sh(argv, **kw):
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kw)


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade-tree", type=Path, default=REPO,
                    help="cascade checkout providing the CLI (default: this repo)")
    ap.add_argument("--python", type=Path, default=None,
                    help="interpreter with the cascade deps (default: the "
                         "tree's .venv python if present, else this one)")
    ap.add_argument("--chain-toml", default="chain.toml")
    ap.add_argument("--network", default="finney")
    ap.add_argument("--push", action="store_true", help="git push after committing")
    ap.add_argument("--allow-findings", action="store_true",
                    help="archive even with high-severity scan findings")
    args = ap.parse_args()
    if args.python is None:
        venv_py = args.cascade_tree / ".venv" / "bin" / "python"
        args.python = venv_py if venv_py.is_file() else Path(sys.executable)

    with tempfile.TemporaryDirectory(prefix="king-sync-") as td:
        out = Path(td) / "king"
        env = {**os.environ, "PYTHONPATH": str(args.cascade_tree)}
        r = subprocess.run(
            [str(args.python), "-m", "cascade.miner.cli", "fetch", "king",
             "--out", str(out), "--chain-toml", args.chain_toml,
             "--network", args.network],
            cwd=args.cascade_tree, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"fetch failed:\n{r.stderr[-800:]}", file=sys.stderr)
            return 1
        # the CLI prints the resolved ref; recover it for provenance
        ref = ""
        for line in (r.stdout + r.stderr).splitlines():
            if "@sha256:" in line:
                for tok in line.replace(",", " ").split():
                    if "@sha256:" in tok:
                        ref = tok.strip("'\".:)(")
                        break
            if ref:
                break
        digest = ref.split("@sha256:")[-1][:64] if "@sha256:" in ref else ""
        if not digest:
            # Without the digest the idempotence check cannot run: an hourly
            # cron would re-commit a fresh "unknown-ref" reign EVERY run,
            # forever, and poison future idempotence with an empty digest
            # (review 2026-09-02). Fail loudly instead — this only happens if
            # the CLI's output format changes, which a human must look at.
            print("could not parse the resolved king ref from the CLI output "
                  "— refusing to archive without provenance. CLI said:\n"
                  f"{(r.stdout + r.stderr)[-800:]}", file=sys.stderr)
            return 2

        prev = {}
        if PROVENANCE.is_file():
            try:
                prev = json.loads(PROVENANCE.read_text())
            except Exception:  # noqa: BLE001 — a torn record must not block a sync
                pass
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", str(DEST)],
            capture_output=True, text=True).stdout.strip()
        if digest and prev.get("digest") == digest and not dirty:
            # unchanged means COMMITTED-unchanged: a prior run whose commit
            # failed (e.g. no git identity) must not be skipped forever.
            print(f"king unchanged ({ref}) — nothing to commit")
            return 0

        # Static security scan BEFORE anything lands in the repo: the king is
        # untrusted competitor code; the subnet only ever runs it sandboxed,
        # but an archived copy can be run by a human on a real machine. High
        # findings block the archive unless --allow-findings; every scan is
        # recorded beside the provenance either way.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scan_generator import scan as scan_tree

        findings = scan_tree(out)
        high = [f for f in findings if f["severity"] == "high"]
        for f in findings:
            print(f"scan [{f['severity'].upper()}] {f['path']}: {f['finding']}")
        if high and not args.allow_findings:
            print(f"REFUSING to archive: {len(high)} high-severity finding(s) "
                  "— rerun with --allow-findings to archive anyway",
                  file=sys.stderr)
            return 1

        # Stage the full replacement BESIDE the destination, then swap: a
        # copy that dies mid-way (disk full) must not leave a half-replaced
        # champions/king/ behind (review 2026-09-02). Symlinks are dropped
        # from the copy UNCONDITIONALLY — the scan flags them high, but an
        # --allow-findings run (accepting some unrelated finding) must never
        # let copytree's default dereference commit HOST file contents into
        # a public repo.
        def _drop_symlinks(dirpath, names):
            return [n for n in names if (Path(dirpath) / n).is_symlink()]

        staged = DEST.parent / ".king.staging"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(out, staged, ignore=_drop_symlinks)
        (staged / "SCAN.json").write_text(json.dumps({
            "high": len(high),
            "warn": sum(1 for f in findings if f["severity"] == "warn"),
            "findings": findings,
            "note": "static scan only — NEVER run this code outside a sandbox "
                    "regardless of a clean scan",
        }, indent=2, default=str) + "\n")
        (staged / "PROVENANCE.json").write_text(json.dumps({
            "ref": ref, "digest": digest, "network": args.network,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "previous_digest": prev.get("digest", ""),
            "note": "archived by scripts/sync_king.py — read-only fetch; "
                    "verify against the on-chain commitment and the round "
                    "receipts (cascade-audit)",
        }, indent=2) + "\n")
        if DEST.exists():
            shutil.rmtree(DEST)
        staged.rename(DEST)

        sh(["git", "-C", str(REPO), "add", str(DEST)])
        # a bare box may have no git identity; commit with an explicit one
        ident = ["-c", "user.name=cascade-king-sync",
                 "-c", "user.email=chaotic.attractoor@gmail.com"]
        label = ref if ref else "unknown-ref"
        sh(["git", "-C", str(REPO), *ident, "commit", "-m",
            f"king sync ({args.network}): {label}",
            "-m", "Automated archive of the subnet's current best generator; "
                  "one commit per reign. See champions/king/PROVENANCE.json."])
        print(f"committed new king: {label}")
        if args.push:
            sh(["git", "-C", str(REPO), "push"])
            print("pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
