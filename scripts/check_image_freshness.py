#!/usr/bin/env python3
"""Release check: which worker-side files changed since each pinned image was built?

    python3 scripts/check_image_freshness.py                    # repo chain.toml, warn-only
    python3 scripts/check_image_freshness.py --config /root/cascade/chain.validator.toml
    python3 scripts/check_image_freshness.py --strict           # exit 1 when a pin is stale

Incident (2026-11-23 -> 26): the pinned worker image worker-v2.0.0-rc15 was
built 2026-10-31 from main 5e3fade. Main fixed lane B's all-GPU gang in #748
(2026-11-23), but no image carrying it was ever pinned, so every lane B expert
leg trained on 1 of 8 GPUs (#960). The trainer ran new code; the pods did not.

For every image pin in the config (``[round] funded_pod_image``, ``[training]``
/ ``[v2.training]`` / ``[v2]`` ``train_image_digest``, the ``[v2] *_after``
repins, ``[round] ops_continuation_image_digest``, ``[marketcast] pod_image``)
this finds the commit the image was built from (a ``-<sha>`` tag suffix, or the
pin line's own comment: "built from 6c9cab9", "rc15 = main 5e3fade"; a bare
digest borrows the commit of another line naming the same digest) and lists
the WORKER-SIDE files changed between that commit and HEAD. The worker-side
set is the cascade-ops ``image_fresh`` set (the code a pod executes for a leg).

Warn-only by default: the repo's chain.toml is the template; the deployed file
decides. Stdlib only; ``git`` reads only. Needs full history (CI checks out
with ``fetch-depth: 0``); an unknown commit is reported, never fatal.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kept in step with cascade-ops cascade_ops/image_fresh.py WORKER_PATTERNS.
WORKER_PATTERNS: tuple[str, ...] = (
    "cascade/trainer/*worker*.py",
    "cascade/trainer/loop.py",
    "cascade/trainer/*_trainer.py",
    "cascade/trainer/pyramid_*.py",
    "cascade/trainer/*_model.py",
    "cascade/trainer/model_backend.py",
    "cascade/trainer/expert_training.py",
    "cascade/trainer/direct_experts.py",
    "cascade/trainer/direct_moe.py",
    "cascade/trainer/finetune.py",
    "cascade/trainer/corpus.py",
    "cascade/trainer/stream.py",
    "cascade/trainer/continuation.py",
    "cascade/trainer/progress_file.py",
    "cascade/trainer/ckpt_digest.py",
    "cascade/trainer/gang_ranks.py",
    "cascade/trainer/marketcast_runner.py",
    "cascade/trainer/marketcast_leg.py",
    "cascade/shared/marketcast_recipe.py",
    "cascade/eval/timestral_adapter.py",
    "cascade/pool/marketcast_relay.py",
    "cascade/pool/marketcast_feeds.py",
    "cascade/shared/worker_config.py",
    "cascade/shared/lane_b_levers.py",
    "cascade/shared/direct_moe.py",
)
PIN_KEYS: tuple[tuple[str, ...], ...] = (
    ("round", "funded_pod_image"),
    ("round", "ops_continuation_image_digest"),
    ("training", "train_image_digest"),
    ("v2", "training", "train_image_digest"),
    ("v2", "train_image_digest"),
    ("v2", "train_image_digest_after"),
    ("v2", "funded_pod_image_after"),
    ("marketcast", "pod_image"),
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
TAG_SHA = re.compile(r":[\w.-]*-([0-9a-f]{7,40})@sha256:")
COMMENT_SHA = re.compile(r"(?:built from|=\s*main|from main)\s+([0-9a-f]{7,40})\b")


def is_worker_side(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in WORKER_PATTERNS)


def _get(doc: dict, keys: tuple[str, ...]):
    for k in keys:
        if not isinstance(doc, dict):
            return None
        doc = doc.get(k)
    return doc


def digest_commits(text: str) -> dict[str, str]:
    """digest -> build commit, from any line that names both."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = DIGEST.search(line)
        if not m:
            continue
        sha = TAG_SHA.search(line) or COMMENT_SHA.search(line)
        if sha:
            out.setdefault(m.group(0), sha.group(1))
    return out


def read_pins(path: Path) -> list[dict]:
    """One row per distinct pinned digest: {keys, digest, ref, commit}."""
    text = path.read_text(encoding="utf-8")
    doc = tomllib.loads(text)
    commits = digest_commits(text)
    rows: dict[str, dict] = {}
    for keys in PIN_KEYS:
        val = str(_get(doc, keys) or "").strip()
        m = DIGEST.search(val)
        if not m:
            continue
        label = f"[{'.'.join(keys[:-1])}] {keys[-1]}"
        row = rows.setdefault(m.group(0), {"keys": [], "digest": m.group(0), "ref": val,
                                           "commit": commits.get(m.group(0), "")})
        row["keys"].append(label)
        if "@" in val and "@" not in row["ref"]:
            row["ref"] = val
    return list(rows.values())


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def stale_files(root: Path, commit: str, head: str = "HEAD") -> tuple[list[str], int] | str:
    """(worker-side files changed commit..head, commits touching them) or an error."""
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
        return f"build commit {commit} is not in this checkout's history (shallow clone?)"
    diff = _git(root, "diff", "--name-only", f"{commit}..{head}")
    if diff.returncode:
        return diff.stderr.strip()[:200]
    files = sorted(p for p in diff.stdout.split() if is_worker_side(p))
    n = 0
    if files:
        log = _git(root, "rev-list", "--count", f"{commit}..{head}", "--", *files)
        n = int(log.stdout.strip() or 0)
    return files, n


def _warn(msg: str) -> None:
    # a GitHub Actions annotation in CI, plain text elsewhere
    print(("::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "WARN ") + msg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", action="append", type=Path,
                    help="chain TOML with the pins (repeatable; default: the repo chain.toml)")
    ap.add_argument("--root", type=Path, default=ROOT, help="git checkout to diff in")
    ap.add_argument("--head", default="HEAD", help="the release the pods should match")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any pin is stale")
    args = ap.parse_args(argv)
    stale = 0
    for cfg in args.config or [args.root / "chain.toml"]:
        try:
            pins = read_pins(cfg)
        except (OSError, tomllib.TOMLDecodeError) as e:
            _warn(f"{cfg}: unreadable ({e})")
            continue
        for row in pins:
            name = f"{cfg.name} {', '.join(row['keys'])} ({row['digest'][:19]}…)"
            if not row["commit"]:
                _warn(f"{name}: build commit unknown (put it in the tag or the pin "
                      "comment: 'built from <sha>')")
                continue
            got = stale_files(args.root, row["commit"], args.head)
            if isinstance(got, str):
                _warn(f"{name}: {got}")
                continue
            files, n = got
            if not files:
                print(f"OK   {name}: built from {row['commit']}, no worker-side change since")
                continue
            stale += 1
            _warn(f"{name}: built from {row['commit']}; {len(files)} worker-side file(s) "
                  f"changed in {n} commit(s) since — rebuild from the deployed SHA before relying "
                  "on a fix in them:")
            for f in files:
                print(f"       {f}")
    return 1 if (stale and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
