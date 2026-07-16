#!/usr/bin/env python3
"""Publish a built eval-pool snapshot to a PUBLIC Hugging Face dataset.

Used by .github/workflows/publish-pool.yml to REVEAL each day's held-out eval
24h after it was scored (a lagged, closed-day snapshot). Uploads the pool dir
(one ``<series_id>.npy`` per series + ``metadata.json`` + ``provenance.json``)
under ``snapshots/<as_of>/`` and refreshes the dataset card at the repo root.

The snapshot is published WITH full labels (source / domain / freq) — it is the
transparency/reproducibility artifact, deliberately not anonymised. It is a day
stale by construction, so it cannot leak the round currently being scored.

Env: HF_TOKEN (write token for the target org). Args: --pool-dir --repo-id --as-of.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
from pathlib import Path

# Force the standard LFS/HTTP upload path, not the Xet backend. Recent
# huggingface_hub auto-installs hf_xet and defaults uploads through Xet, which
# needs a SEPARATE xet-write scope on the token. A plain write token (which can
# create_repo and push LFS fine) gets 403 on .../xet-write-token/main, aborting
# the whole upload_folder. Disabling Xet sidesteps that scope entirely; the
# fallback path uses the same write token. setdefault so an explicit
# HF_HUB_DISABLE_XET=0 (opt back into Xet) is still honoured. Must be set before
# huggingface_hub is imported — its constants read this at import time.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _card(repo_id: str, as_of: str, pool_dir: Path) -> str:
    meta = json.loads((pool_dir / "metadata.json").read_text()) if (pool_dir / "metadata.json").is_file() else {}
    n = len(meta)
    dom = collections.Counter(v.get("domain", "?") for v in meta.values())
    freq = collections.Counter(v.get("freq", "?") for v in meta.values())
    dom_s = ", ".join(f"{k}={v}" for k, v in sorted(dom.items(), key=lambda x: -x[1]))
    freq_s = ", ".join(f"{k}={v}" for k, v in sorted(freq.items(), key=lambda x: -x[1]))
    # Static body kept as a plain string (it contains literal {...} in the layout
    # example, which an f-string would misread); only the stats block is templated.
    body = """---
license: cc-by-4.0
pretty_name: cascade held-out eval pool (24h-lagged reveal)
tags:
  - time-series
  - forecasting
  - benchmark
---

# cascade eval pool — lagged public reveal

Daily snapshots of the **held-out evaluation pool** used by the
[cascade](https://github.com/TensorLink-AI/cascade) subnet, built from the
[tsbench-forge](https://github.com/tensorlink-dev/TSBench-Forge) live catalog of
real public time series across the 7 GIFT-Eval domains.

**Each snapshot is revealed 24h AFTER it was used for scoring** (`as_of` is a
closed UTC day). Because the pool rotates daily, a released snapshot is always
for a round that is already scored — it lets anyone **reproduce the leaderboard**
without exposing the round currently in play.

## Layout

```
snapshots/<YYYY-MM-DD>/          one folder per revealed day (as_of)
  <series_id>.npy                float32, freshest context_length + horizon points
  metadata.json                  {series_id: {freq, seasonal_period, domain, source}}
  provenance.json                build config (context_length, horizon, builder_version)
```

Scoring is identity-agnostic: MASE uses `seasonal_period`, CRPS/WQL use the
values + a model's quantiles, and the KOTH cluster-bootstrap groups windows by
`source`. Re-cut windows of `context_length + horizon`, forecast the horizon,
and the metrics reproduce byte-for-byte.

"""
    stats = (
        f"## Latest snapshot — `{as_of}`\n\n"
        f"- **series:** {n}\n"
        f"- **domains:** {dom_s}\n"
        f"- **cadences:** {freq_s}\n\n"
        "> This is an **evaluation** set, not training data. Publishing it to train on\n"
        "> would contaminate the benchmark it exists to measure.\n"
    )
    return body + stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--as-of", required=True)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping HF publish", file=sys.stderr)
        return 0
    pool = Path(args.pool_dir)
    if not (pool / "metadata.json").is_file():
        print(f"no metadata.json under {pool} — nothing to publish", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(pool),
        path_in_repo=f"snapshots/{args.as_of}",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"eval snapshot as_of={args.as_of}",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(_card(args.repo_id, args.as_of, pool).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"refresh card (as_of={args.as_of})",
    )
    print(f"published {args.as_of} snapshot to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
