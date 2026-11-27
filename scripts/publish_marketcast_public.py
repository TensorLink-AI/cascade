#!/usr/bin/env python
"""Publish the post-embargo MarketCast bar history as a PUBLIC dataset
(owner 2026-11-27): the recorder's bars older than ``today - [marketcast]
embargo_days`` as Parquet, to the HF dataset ``tensorlink-dev/cascade-
marketcast-bars`` and the anonymously readable Hippius manifest bucket under
``marketcast/public-bars/``. Scheduled daily by
``.github/workflows/publish-mkt-public.yml``; layout, cut and index are
defined in :mod:`cascade.shared.marketcast_public`.

Every staged file is re-read and checked against the cut before any upload
(:class:`EmbargoViolation` aborts the whole run). Uploads are incremental:
a file whose content hash equals the destination's ``index.json`` entry is
skipped, so a re-run with no new day uploads nothing but the index/card.

Usage::

    python scripts/publish_marketcast_public.py --bar-store ./mkt/bars --stage ./stage \\
        [--chain-toml chain.toml] [--hf-repo R] [--no-hf] [--no-hippius] [--dry-run]

Credentials (env only): ``HF_TOKEN`` (write) for the HF dataset;
``HIPPIUS_S3_ACCESS_KEY`` / ``HIPPIUS_S3_SECRET_KEY`` (write on
``[storage] manifest_bucket``) for the Hippius mirror.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.shared import marketcast_public as P  # noqa: E402

SOURCE_NOTES = {
    "hyperliquid": ("Hyperliquid perpetuals (core dex), public Info API "
                    "`POST https://api.hyperliquid.xyz/info` (`candleSnapshot`, `fundingHistory`)"),
    "hyperliquid_xyz": ("Hyperliquid HIP-3 `xyz` builder dex perpetuals (equity indices/ETFs/"
                        "single stocks, FX, metals, commodities), same public Info API "
                        "(`dex: \"xyz\"`); prices are on-chain PERP prices (oracle-anchored, "
                        "they trade off-hours), not exchange-official quotes"),
}


# ── build ────────────────────────────────────────────────────────────────────


def public_universe(mc) -> list[tuple[str, str]]:
    """``(class, symbol)`` of every configured asset whose primary source is
    a recorded Hyperliquid feed (alpha / non-HL classes are not in the bank)."""
    out = []
    for cls, assets in sorted(mc.assets_map.items()):
        srcs = mc.sources_for(cls)
        if not srcs or srcs[0] not in P.PUBLIC_SOURCES:
            continue
        out += [(cls, a) for a in assets]
    return out


def build(bar_store: Path, stage: Path, universe, cutoff: date) -> dict[str, dict]:
    """Stage every month file of ``universe`` strictly before ``cutoff``;
    returns ``{path: entry}``."""
    entries: dict[str, dict] = {}

    def _emit(cls, sym, tf, month, cols):
        rel = P.file_path(cls, sym, tf, month)
        dest = stage / rel
        P.write_parquet(dest, cols, cutoff)
        ts = cols["timestamp"]
        entries[rel] = {
            "class": cls, "symbol": sym, "timeframe": tf, "month": month,
            "rows": int(ts.size),
            "first": datetime.fromtimestamp(int(ts.min()) / 1000, tz=UTC).isoformat(),
            "last": datetime.fromtimestamp(int(ts.max()) / 1000, tz=UTC).isoformat(),
            "content_sha256": P.content_sha256(cols),
            "sha256": P.file_sha256(dest), "bytes": dest.stat().st_size,
        }

    for cls, sym in universe:
        for tf in P.TIMEFRAMES:
            days = P.public_days(bar_store, cls, sym, tf, cutoff)
            if not days:
                continue
            for month, cols in P.split_by_month(P.read_bar_days(bar_store, cls, sym, tf, days)).items():
                _emit(cls, sym, tf, month, cols)
        for month, cols in P.split_by_month(P.read_funding(bar_store, cls, sym, cutoff)).items():
            _emit(cls, sym, P.FUNDING, month, cols)
    return entries


def verify_staged(stage: Path, entries: dict[str, dict], cutoff: date) -> None:
    """Re-read every staged file: its timestamps are before the cut and its
    bytes are the indexed ones. Raises before anything is uploaded."""
    for rel, e in entries.items():
        path = stage / rel
        P.assert_before_cutoff(P.read_parquet_timestamps_ms(path), cutoff, what=rel)
        if P.file_sha256(path) != e["sha256"]:
            raise RuntimeError(f"{rel}: staged bytes changed after indexing")


def plan_uploads(entries: dict[str, dict], remote: dict) -> list[str]:
    """Paths whose content differs from the destination's index (idempotence)."""
    have = remote.get("files", {})
    return sorted(p for p, e in entries.items()
                  if have.get(p, {}).get("content_sha256") != e["content_sha256"])


def merged_index(entries: dict[str, dict], remote: dict, *, cutoff: date, embargo_days: int,
                 now: datetime) -> dict:
    files = {**remote.get("files", {}), **entries}
    for e in files.values():                 # belt and braces: never index a post-cut row
        if date.fromisoformat(e["last"][:10]) >= cutoff:
            raise P.EmbargoViolation(f"index entry {e} reaches the embargo (cut {cutoff})")
    return {"schema": P.INDEX_SCHEMA, "embargo_days": int(embargo_days),
            "public_before": cutoff.isoformat(), "generated_at": now.isoformat(timespec="seconds"),
            "files": dict(sorted(files.items()))}


# ── dataset card ─────────────────────────────────────────────────────────────


def render_card(index: dict, mc, *, hf_repo: str, hippius_url: str) -> str:
    files = index["files"]
    by_cls: dict[str, set] = {}
    tfs: set = set()
    total = 0
    first = min((e["first"][:10] for e in files.values()), default="-")
    for e in files.values():
        by_cls.setdefault(e["class"], set()).add(e["symbol"])
        tfs.add(e["timeframe"])
        total += int(e["bytes"])
    srcs = sorted({mc.sources_for(c)[0] for c in by_cls if mc.sources_for(c)})
    universe = "\n".join(f"| {c} | {len(s)} | {', '.join(sorted(s))} |" for c, s in sorted(by_cls.items()))
    tf_list = ", ".join(t for t in [*P.TIMEFRAMES, P.FUNDING] if t in tfs) or "-"
    sources = "\n".join(f"- **{s}**: {SOURCE_NOTES.get(s, s)}" for s in srcs) or "- (none yet)"
    return f"""---
license: other
license_name: upstream-exchange-terms
license_link: https://hyperliquid.gitbook.io/hyperliquid-docs/legal
pretty_name: Cascade MarketCast bars (post-embargo)
tags: [time-series, finance, crypto, ohlcv, forecasting, bittensor]
task_categories: [time-series-forecasting]
configs:
- config_name: default
  data_files: "data/*/*/1h/*.parquet"
---

# Cascade MarketCast bars (post-embargo history)

The market bars recorded for **MarketCast**, lane C of the Cascade Bittensor
subnet, published once they are older than the training embargo. Use them to
try fine-tuning recipes locally and compare them with prior rounds.

- **30-day lag.** Only days strictly before `today - embargo_days` are here
  (`embargo_days = {index['embargo_days']}`; this build: every day before
  **{index['public_before']}**). The most recent days stay private: MarketCast
  scores each entry on windows drawn from the `embargo_eval_days`
  (= {mc.embargo_eval_days}) before its leg's dispatch, and a recipe's training
  data must end `embargo_days` before submission. Everything here is data any
  recipe may already train on; it is never the data you are scored on.
- **First day:** {first}. Updated daily. {len(files)} files, {total / 1e6:.1f} MB.
- **Mirror:** the same files and `index.json` at <{hippius_url}/index.json>
  (anonymous HTTPS, Hippius S3).

## Layout and schema

```
data/<class>/<SYMBOL>/<tf>/<YYYY-MM>.parquet      tf: {tf_list}
index.json                                        sha256, bytes, rows, first/last bar per file
```

Bar files (`1m`, `5m`, `15m`, `1h`):

| column | type | meaning |
|---|---|---|
| timestamp | timestamp[ms, UTC] | bar OPEN time |
| open, high, low, close | float64 | prices in USD (perp mark of the venue's trades) |
| volume | float64 | base-asset volume traded in the bar |

Funding files (`funding`): `timestamp` (sample time), `rate` (hourly funding
rate, float64), `premium` (float64, NaN when not reported).

A bar nobody recorded is an **absent row**, never interpolated. Only closed
bars are recorded. 4h and 1d bars (also scored) are not stored: aggregate the
1h bars (open = first, high = max, low = min, close = last, volume = sum).
Month files of the current month grow by one day per day; closed months
never change. Files are zstd-compressed Parquet with float64 preserved.

## Universe

| class | symbols | list |
|---|---|---|
{universe}

The Bittensor-alpha class is not recorded in this bank and is not published.

## Sources and terms

{sources}

Bars are recorded forward from the public API by the Cascade MarketCast
recorder (`cascade-pool mkt-record`), with the API's own history (about 5000
candles per interval) backfilled at first record. No exchange grants a
separate licence for this data: it is redistributed for research and
forecasting under the upstream venue's terms of use, which you must follow
(see the licence link). It is provided as-is, without warranty, and is not
investment advice. Binance spot (used only as a validator cross-check) is
not redistributed.

## Relation to MarketCast scoring

MarketCast fine-tunes the pinned Timestral-28M base with each miner's recipe
and scores it against the untouched base on CRPS over seven suites (returns,
OHLCV, volatility, high, low, range, liquidity) on every timeframe. The
validators read the private, signed recorder snapshot, not this dataset; the
bars here are the same recording, lagged. Guide:
https://github.com/TensorLink-AI/cascade/blob/main/docs/MINER_LANE_C_MARKETCAST.md

## Load

```python
from datasets import load_dataset
ds = load_dataset("{hf_repo}", data_files="data/crypto/BTC/1h/*.parquet", split="train")

import pandas as pd
df = pd.read_parquet("hf://datasets/{hf_repo}/data/crypto/BTC/1m/2026-10.parquet")
```

Or with the Cascade CLI (no keys; verifies every sha256):

```bash
cascade lane-c data --symbols BTC,ETH --timeframes 1m,1h --start 2026-10-01 --end 2026-10-31 --out ./bars
```
"""


# ── upload ───────────────────────────────────────────────────────────────────


def hf_remote_index(api, repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(repo, P.INDEX_NAME, repo_type="dataset", token=api.token)
    except Exception as e:  # noqa: BLE001 — first run: no index yet
        if "404" in str(e) or type(e).__name__.endswith("NotFoundError"):
            return P.empty_index()
        raise
    return P.parse_index(Path(p).read_bytes())


def upload_hf(api, repo: str, stage: Path, paths: list[str], index: dict, card: str,
              *, chunk: int = 400) -> None:
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(stage / p)) for p in paths]
    for i in range(0, len(ops), chunk):
        api.create_commit(repo, repo_type="dataset", operations=ops[i:i + chunk],
                          commit_message=f"bars before {index['public_before']} ({i + 1}-{i + len(ops[i:i + chunk])})")
    tail = [CommitOperationAdd(path_in_repo=P.INDEX_NAME,
                               path_or_fileobj=json.dumps(index, indent=1).encode()),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    api.create_commit(repo, repo_type="dataset", operations=tail,
                      commit_message=f"index + card: public before {index['public_before']}")


def hippius_remote_index(store, prefix: str) -> dict:
    from cascade.shared.hippius import ObjectNotFound

    try:
        return P.parse_index(store.get_bytes(f"{prefix}/{P.INDEX_NAME}"))
    except ObjectNotFound:
        return P.empty_index()


def _put_public(store, key: str, data: bytes, content_type: str) -> None:
    from cascade.shared.hippius import StorageError

    try:
        store.put_bytes(key, data, content_type=content_type, acl="public-read")
    except StorageError:   # the bucket convention: canned ACL refused => bucket policy serves it
        store.put_bytes(key, data, content_type=content_type)


def upload_hippius(store, prefix: str, stage: Path, paths: list[str], index: dict, card: str) -> None:
    for p in paths:
        _put_public(store, f"{prefix}/{p}", (stage / p).read_bytes(), "application/vnd.apache.parquet")
    _put_public(store, f"{prefix}/README.md", card.encode(), "text/markdown")
    _put_public(store, f"{prefix}/{P.INDEX_NAME}", json.dumps(index, indent=1).encode(),
                "application/json")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bar-store", type=Path, required=True, help="MktBarStore root (synced bars).")
    ap.add_argument("--stage", type=Path, required=True, help="Empty dir for the staged Parquet.")
    ap.add_argument("--chain-toml", type=Path, default=REPO_ROOT / "chain.toml")
    ap.add_argument("--hf-repo", default=P.HF_REPO)
    ap.add_argument("--hippius-prefix", default=P.HIPPIUS_PREFIX)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (default: today UTC; tests).")
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--no-hippius", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Build + verify only; upload nothing.")
    a = ap.parse_args(argv)

    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(a.chain_toml)
    mc = cfg.marketcast
    now = datetime.now(UTC)
    today = P.parse_day(a.today) if a.today else now.date()
    cutoff = P.embargo_cutoff(today, mc.embargo_days)
    print(f"embargo_days={mc.embargo_days} ({a.chain_toml.name}): publishing days < {cutoff}")
    if a.stage.exists() and any(a.stage.iterdir()):
        print(f"stage {a.stage} is not empty", file=sys.stderr)
        return 2
    entries = build(a.bar_store, a.stage, public_universe(mc), cutoff)
    verify_staged(a.stage, entries, cutoff)
    print(f"staged {len(entries)} files, {sum(e['bytes'] for e in entries.values()) / 1e6:.1f} MB")
    st = cfg.storage
    endpoint = str(getattr(st, "s3_endpoint", "") or "https://s3.hippius.com")
    bucket = str(getattr(st, "manifest_bucket", "") or "")
    hippius_url = P.hippius_public_url(endpoint, bucket, a.hippius_prefix)

    if not a.no_hf:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token and not a.dry_run:
            print("HF_TOKEN not set", file=sys.stderr)
            return 2
        api = HfApi(token=token)
        remote = hf_remote_index(api, a.hf_repo) if token else P.empty_index()
        todo = plan_uploads(entries, remote)
        index = merged_index(entries, remote, cutoff=cutoff, embargo_days=mc.embargo_days, now=now)
        card = render_card(index, mc, hf_repo=a.hf_repo, hippius_url=hippius_url)
        print(f"hf {a.hf_repo}: {len(todo)} new/changed of {len(entries)}")
        if not a.dry_run:
            upload_hf(api, a.hf_repo, a.stage, todo, index, card)
    if not a.no_hippius:
        if not bucket:
            print("[storage] manifest_bucket unset", file=sys.stderr)
            return 2
        from cascade.shared.hippius import S3Config, S3Store

        store = S3Store(S3Config.from_storage(st, bucket=bucket))
        remote = P.empty_index() if a.dry_run else hippius_remote_index(store, a.hippius_prefix)
        todo = plan_uploads(entries, remote)
        index = merged_index(entries, remote, cutoff=cutoff, embargo_days=mc.embargo_days, now=now)
        card = render_card(index, mc, hf_repo=a.hf_repo, hippius_url=hippius_url)
        print(f"hippius {hippius_url}: {len(todo)} new/changed of {len(entries)}")
        if not a.dry_run:
            upload_hippius(store, a.hippius_prefix, a.stage, todo, index, card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
