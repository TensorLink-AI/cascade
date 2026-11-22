#!/usr/bin/env python3
"""GitHub Actions results ledger for ``scripts/laned/status`` (plan §17).

The status board needs per-WP results of the laned-accept workflow
(``{"runs": [{"workflow", "event", "mode", "wp", "head_sha", "conclusion",
"created_at"}]}``). The Actions API only reports whole workflow runs, so each
laned-accept job records its per-WP results and the push-to-main job keeps the
merged history as ``status/ci_runs.json`` on branch ``laned/build-state``
(artifacts expire; the ledger does not).

    ci_ledger.py record  --file F --wp W --mode M --event E --head-sha S
                         --conclusion C [--run-id N] [--evidence-sha BLOB]
                         [--tests-tree TREE]
    ci_ledger.py harvest --repo OWNER/NAME --out F [--root R]
                         [--ledger-ref origin/laned/build-state] [--artifact laned-results]

``harvest`` starts from the ledger at ``--ledger-ref`` and adds the
``laned-results`` artifact of every completed laned-accept run not yet
harvested, via the ``gh`` CLI (GITHUB_TOKEN). An artifact entry is only kept
when its ``event`` matches the run that uploaded it and its ``head_sha`` is the
run's ``head_sha``. The one exception is a ``workflow_dispatch`` run: it checks
out its ``sha`` input, so its ``expect-fail`` entry carries that sha while the
run's ``head_sha`` is the dispatched ref. Such an entry is kept when its sha is
an ancestor of the run's ``head_sha`` (``git merge-base --is-ancestor``) and,
when the entry records ``run_head_sha``, that equals the run's ``head_sha``.

Dispatch runs harvested before that exception existed had their entry dropped
while their id went into ``harvested_run_ids``; ``harvest`` re-reads each
dispatch run with no entry in ``runs`` once (``rechecked_dispatch_run_ids``).

    ci_ledger.py selftest     # offline check of the harvest guard (fake gh)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -P` drops the script dir
import lanedlib  # noqa: E402

WORKFLOW = "laned-accept"
LEDGER_PATH = "status/ci_runs.json"
MODES = ("accept", "expect-fail", "verification", "tests-correction", "evidence", "check-pr", "gate")
CONCLUSIONS = ("success", "failure")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path) -> dict:
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault("runs", [])
    data.setdefault("harvested_run_ids", [])
    return data


def ledger_defaults(data: dict) -> dict:
    data.setdefault("runs", [])
    data.setdefault("harvested_run_ids", [])
    data.setdefault("rechecked_dispatch_run_ids", [])
    return data


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["harvested_run_ids"] = sorted(set(data["harvested_run_ids"]))
    if "rechecked_dispatch_run_ids" in data:
        data["rechecked_dispatch_run_ids"] = sorted(set(data["rechecked_dispatch_run_ids"]))
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
    if args.run_head_sha:
        data["runs"][-1]["run_head_sha"] = args.run_head_sha
    if args.evidence_sha:
        # the blob sha of the evidence (or gate) file the run verified
        data["runs"][-1]["evidence_sha"] = args.evidence_sha
    if args.tests_tree:
        # verification: tree sha of tests/accept/<wp>/ the run checked (status
        # finds the merge commit on main by it, squash merges included)
        data["runs"][-1]["tests_tree"] = args.tests_tree
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


def entry_ok(root: Path, entry: object, event: str, head_sha: str) -> bool:
    """True when an artifact entry belongs to the run (event, head_sha) that uploaded it."""
    if not valid_entry(entry) or entry["event"] != event:
        return False
    if entry["head_sha"] == head_sha:
        return True
    # workflow_dispatch checks out its `sha` input: the entry carries that sha,
    # the run carries the dispatched ref. Only an expect-fail result at a commit
    # already in that ref's history is accepted.
    if event != "workflow_dispatch" or entry["mode"] != "expect-fail":
        return False
    if entry.get("run_head_sha") not in (None, head_sha):
        return False
    sha = entry["head_sha"]
    if not sha or sha.startswith("-"):
        return False
    proc = lanedlib.git(root, "merge-base", "--is-ancestor", sha, head_sha, check=False)
    return proc.returncode == 0


def harvest_run(root: Path, data: dict, repo: str, artifact: str, run_id: int,
                event: str, head_sha: str, created: str, gh_fn=None) -> int:
    gh_fn = gh_fn or gh
    added = 0
    artifacts = gh_fn("api", f"repos/{repo}/actions/runs/{run_id}/artifacts",
                      "--jq", f'.artifacts[] | select(.name == "{artifact}" and (.expired | not)) | .id')
    if artifacts.strip():
        with tempfile.TemporaryDirectory(prefix="laned-ledger-") as tmp:
            gh_fn("run", "download", str(run_id), "-R", repo, "-n", artifact, "-D", tmp)
            for f in sorted(Path(tmp).rglob("*.json")):
                for entry in load(f)["runs"]:
                    if not entry_ok(root, entry, event, head_sha):
                        print(f"ci_ledger: run {run_id}: dropped entry {entry!r}", file=sys.stderr)
                        continue
                    entry.setdefault("created_at", created)
                    entry["workflow"] = WORKFLOW
                    entry["run_id"] = run_id
                    data["runs"].append(entry)
                    added += 1
    return added


def harvest(root: Path, data: dict, repo: str, artifact: str, gh_fn=None) -> int:
    gh_fn = gh_fn or gh
    ledger_defaults(data)
    done = set(data["harvested_run_ids"])
    rechecked = set(data["rechecked_dispatch_run_ids"])
    represented = {r.get("run_id") for r in data["runs"] if isinstance(r, dict)}
    listing = gh_fn("api", "--paginate",
                    f"repos/{repo}/actions/workflows/{WORKFLOW}.yml/runs?status=completed&per_page=100",
                    "--jq", ".workflow_runs[] | [.id, .event, .head_sha, .created_at] | @tsv")
    added = 0
    for line in listing.splitlines():
        run_id_s, event, head_sha, created = line.split("\t")
        run_id = int(run_id_s)
        dispatch = event == "workflow_dispatch"
        if run_id in done:
            # one-time re-read of a dispatch run whose entry the old guard dropped
            if not dispatch or run_id in rechecked or run_id in represented:
                continue
            print(f"ci_ledger: re-reading dispatch run {run_id}", file=sys.stderr)
        added += harvest_run(root, data, repo, artifact, run_id, event, head_sha, created, gh_fn)
        data["harvested_run_ids"].append(run_id)
        done.add(run_id)
        if dispatch:
            data["rechecked_dispatch_run_ids"].append(run_id)
            rechecked.add(run_id)
    return added


def cmd_harvest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    shown = lanedlib.git(root, "show", f"{args.ledger_ref}:{LEDGER_PATH}", check=False)
    data = ledger_defaults(json.loads(shown.stdout) if shown.returncode == 0 else {})
    added = harvest(root, data, args.repo, args.artifact)
    save(Path(args.out), data)
    print(f"ci_ledger: {added} new result(s); {len(data['runs'])} in the ledger")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """Offline check of the harvest guard against a throwaway git repo and a fake gh."""
    with tempfile.TemporaryDirectory(prefix="laned-ledger-selftest-") as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        env_git = ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false")

        def commit(msg: str) -> str:
            lanedlib.git(root, *env_git, "commit", "-q", "--allow-empty", "-m", msg)
            return lanedlib.git(root, "rev-parse", "HEAD").stdout.strip()

        lanedlib.git(root, "init", "-q")
        tests_sha = commit("tests-only merge")
        main_sha = commit("workflow lands")
        lanedlib.git(root, "checkout", "-q", "--orphan", "other")
        stray_sha = commit("unrelated")
        pr_sha = commit("pr head")

        def ef(sha: str, event: str, **extra: str) -> dict:
            return {"wp": "W0x", "mode": "expect-fail", "event": event, "head_sha": sha,
                    "conclusion": "success", **extra}

        # run id -> (event, run head_sha, artifact entries)
        runs = {
            1: ("workflow_dispatch", main_sha, [ef(tests_sha, "workflow_dispatch")]),
            2: ("workflow_dispatch", main_sha, [ef(stray_sha, "workflow_dispatch")]),
            3: ("pull_request", pr_sha, [ef(tests_sha, "pull_request")]),
            4: ("pull_request", pr_sha, [ef(pr_sha, "pull_request")]),
            5: ("workflow_dispatch", main_sha,
                [dict(ef(tests_sha, "workflow_dispatch"), mode="accept")]),
            6: ("workflow_dispatch", main_sha,
                [ef(tests_sha, "workflow_dispatch", run_head_sha=stray_sha)]),
            7: ("workflow_dispatch", main_sha,
                [ef(tests_sha, "workflow_dispatch", run_head_sha=main_sha)]),
        }
        downloads: list[int] = []

        def fake_gh(*args: str) -> str:
            if args[:2] == ("api", "--paginate"):
                return "".join(f"{i}\t{ev}\t{sha}\t2026-01-01T00:00:00Z\n"
                               for i, (ev, sha, _) in runs.items())
            if args[0] == "api":
                return "99\n"
            run_id = int(args[2])
            downloads.append(run_id)
            out = Path(args[args.index("-D") + 1])
            (out / "results.json").write_text(json.dumps({"runs": runs[run_id][2]}))
            return ""

        failures: list[str] = []

        def expect(cond: bool, what: str) -> None:
            if not cond:
                failures.append(what)

        # a ledger written by the old guard: dispatch run 1 harvested, its entry dropped
        data = {"runs": [], "harvested_run_ids": [1, 4]}
        added = harvest(root, data, "o/r", "laned-results", fake_gh)
        kept = {r["run_id"] for r in data["runs"]}
        expect(1 in kept, "dropped dispatch run 1 was not re-read")
        expect(downloads.count(4) == 0,
               "non-dispatch harvested runs were re-read")
        expect(kept == {1, 7}, f"kept runs {sorted(kept)}, want [1, 7]")
        expect(added == 2, f"added {added}, want 2")
        before = list(downloads)
        harvest(root, data, "o/r", "laned-results", fake_gh)
        expect(downloads == before, "a second harvest re-read runs again")
        expect(not entry_ok(root, ef(tests_sha, "push"), "push", main_sha),
               "push run accepted an entry at a different sha")
        expect(entry_ok(root, ef(main_sha, "push"), "push", main_sha),
               "push run rejected its own sha")
    for f in failures:
        print(f"ci_ledger selftest: FAIL {f}", file=sys.stderr)
    if failures:
        return 1
    print("ci_ledger selftest: ok")
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
    rec.add_argument("--run-head-sha", help="the run's own head sha when it differs (workflow_dispatch)")
    rec.add_argument("--evidence-sha", help="blob sha of the evidence/gate file the run verified")
    rec.add_argument("--tests-tree", help="verification: tree sha of tests/accept/<wp>/ the run checked")
    har = sub.add_parser("harvest")
    har.add_argument("--repo", required=True)
    har.add_argument("--out", required=True)
    har.add_argument("--root", default=".")
    har.add_argument("--ledger-ref", default="origin/laned/build-state")
    har.add_argument("--artifact", default="laned-results")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    return {"record": cmd_record, "harvest": cmd_harvest, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
