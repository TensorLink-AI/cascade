#!/usr/bin/env python3
"""GitHub Actions results ledger for ``scripts/laned/status`` (plan §17).

The status board needs per-WP results of the laned-accept workflow
(``{"runs": [{"workflow", "event", "mode", "wp", "head_sha", "conclusion",
"created_at"}]}``). The Actions API only reports whole workflow runs, so each
laned-accept job records its per-WP results and the push-to-main job keeps the
merged history as ``status/ci_runs.json`` on branch ``laned/build-state``
(artifacts expire; the ledger does not).

    ci_ledger.py record  --file F --wp W --mode M --event E --head-sha S
                         --conclusion C [--run-id N]
    ci_ledger.py harvest --repo OWNER/NAME --out F [--root R]
                         [--ledger-ref origin/laned/build-state] [--artifact laned-results]

``harvest`` starts from the ledger at ``--ledger-ref`` and adds the
``laned-results`` artifact of every completed laned-accept run not yet
harvested, via the ``gh`` CLI (GITHUB_TOKEN). An artifact entry is only kept
when its ``head_sha`` and ``event`` match the run that uploaded it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import lanedlib

WORKFLOW = "laned-accept"
LEDGER_PATH = "status/ci_runs.json"
MODES = ("accept", "expect-fail", "check-pr")
CONCLUSIONS = ("success", "failure")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path) -> dict:
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault("runs", [])
    data.setdefault("harvested_run_ids", [])
    return data


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["harvested_run_ids"] = sorted(set(data["harvested_run_ids"]))
    path.write_text(json.dumps(data, indent=2) + "\n")


def valid_entry(entry: object) -> bool:
    return (isinstance(entry, dict) and isinstance(entry.get("wp"), str) and entry["wp"]
            and entry.get("mode") in MODES and entry.get("conclusion") in CONCLUSIONS
            and isinstance(entry.get("head_sha"), str) and isinstance(entry.get("event"), str))


def cmd_record(args: argparse.Namespace) -> int:
    path = Path(args.file)
    data = load(path)
    data["runs"].append({
        "workflow": WORKFLOW, "event": args.event, "mode": args.mode, "wp": args.wp,
        "head_sha": args.head_sha, "conclusion": args.conclusion, "created_at": now(),
        "run_id": args.run_id,
    })
    if args.run_id is not None:
        data["harvested_run_ids"].append(args.run_id)
    save(path, data)
    print(f"ci_ledger: recorded {args.wp} {args.mode} {args.conclusion}")
    return 0


def gh(*args: str) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def cmd_harvest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    shown = lanedlib.git(root, "show", f"{args.ledger_ref}:{LEDGER_PATH}", check=False)
    data = json.loads(shown.stdout) if shown.returncode == 0 else {}
    data.setdefault("runs", [])
    data.setdefault("harvested_run_ids", [])
    done = set(data["harvested_run_ids"])
    listing = gh("api", "--paginate",
                 f"repos/{args.repo}/actions/workflows/{WORKFLOW}.yml/runs?status=completed&per_page=100",
                 "--jq", ".workflow_runs[] | [.id, .event, .head_sha, .created_at] | @tsv")
    added = 0
    for line in listing.splitlines():
        run_id_s, event, head_sha, created = line.split("\t")
        run_id = int(run_id_s)
        if run_id in done:
            continue
        artifacts = gh("api", f"repos/{args.repo}/actions/runs/{run_id}/artifacts",
                       "--jq", f'.artifacts[] | select(.name == "{args.artifact}" and (.expired | not)) | .id')
        if artifacts.strip():
            with tempfile.TemporaryDirectory(prefix="laned-ledger-") as tmp:
                gh("run", "download", str(run_id), "-R", args.repo, "-n", args.artifact, "-D", tmp)
                for f in sorted(Path(tmp).rglob("*.json")):
                    for entry in load(f)["runs"]:
                        if not valid_entry(entry) or entry["head_sha"] != head_sha or entry["event"] != event:
                            print(f"ci_ledger: run {run_id}: dropped entry {entry!r}", file=sys.stderr)
                            continue
                        entry.setdefault("created_at", created)
                        entry["workflow"] = WORKFLOW
                        entry["run_id"] = run_id
                        data["runs"].append(entry)
                        added += 1
        data["harvested_run_ids"].append(run_id)
        done.add(run_id)
    save(Path(args.out), data)
    print(f"ci_ledger: {added} new result(s); {len(data['runs'])} in the ledger")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--file", required=True)
    rec.add_argument("--wp", required=True)
    rec.add_argument("--mode", required=True, choices=MODES)
    rec.add_argument("--event", required=True)
    rec.add_argument("--head-sha", required=True)
    rec.add_argument("--conclusion", required=True, choices=CONCLUSIONS)
    rec.add_argument("--run-id", type=int)
    har = sub.add_parser("harvest")
    har.add_argument("--repo", required=True)
    har.add_argument("--out", required=True)
    har.add_argument("--root", default=".")
    har.add_argument("--ledger-ref", default="origin/laned/build-state")
    har.add_argument("--artifact", default="laned-results")
    args = ap.parse_args(argv)
    return cmd_record(args) if args.cmd == "record" else cmd_harvest(args)


if __name__ == "__main__":
    sys.exit(main())
