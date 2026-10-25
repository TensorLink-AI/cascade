#!/usr/bin/env python
"""Scrape cascade generators into the private R2 archive — kings AND everyone.

Reads the validator's public throne history (``receipts/index.json`` in the
manifest bucket) and snapshots generator code — fetched from the Hippius Hub
and packed to a deterministic tar — into a **private** S3-compatible
(Cloudflare R2) bucket, in two dirs:

* ``kings/`` — every generator that has ever held the throne, plus a
  ``kings/index.json`` "db" linking each king to its archived object.
* ``generators/<hotkey>/`` — EVERY eligible participant generator, king or
  not, grouped by the committing miner and gathered from each round's full
  signed receipt (the compact index only names the king and duel challenger),
  plus ``generators/index.json`` with one entry per (hotkey, generator).

See :mod:`cascade.shared.king_archive`.

Both dirs are content-addressed and append-only: a generator already in the
bucket is never re-fetched, only its metadata is refreshed, and already-scanned
round receipts are never re-read. Run it as often as you like — hourly is
plenty, rounds are epoch-paced.

Usage::

    # reads [storage] from chain.toml; needs S3 read creds for the manifest
    # bucket (HIPPIUS_S3_ACCESS_KEY / _SECRET_KEY), Hub pull creds
    # (HIPPIUS_HUB_TOKEN — public repos pull anonymously, but miners commit
    # private Hub repos too, and those refs fail without a token; HF_TOKEN
    # likewise for private `@hf:` refs), and R2 write creds for the archive
    # (KING_ARCHIVE_S3_ACCESS_KEY / _SECRET_KEY, or the BACKUP_S3_* pair).
    python scripts/scrape_kings.py
    python scripts/scrape_kings.py --chain-toml chain.testnet.toml
    python scripts/scrape_kings.py --dry-run          # list what would be archived
    python scripts/scrape_kings.py --kings-only       # skip the generators/ sweep
    python scripts/scrape_kings.py --generators-only  # skip the kings/ archive
    python scripts/scrape_kings.py --retry-unavailable  # re-try parked tombstones

Exit codes: 0 on success, 1 when a generator failed for a RETRYABLE reason (the
Hub was down, the network flaked — worth looking at), 2/4 on config/storage
errors. Generators whose upstream repo was deleted or made private are NOT a
failure: miners delete repos routinely, no re-run recovers them, so they are
recorded in the index with an ``unavailable`` note, parked after a few attempts,
and reported as a note on a green run. Otherwise the daily job goes permanently
red and stops telling anyone anything.

Cron (hourly is ample — rounds are epoch-paced)::

    23 * * * * cd /opt/cascade && python scripts/scrape_kings.py >> king-archive.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

_SAMPLE_N = 8


def _sample(refs) -> str:
    """A readable digest of a ref list — the full 80-odd refs are in the index."""
    refs = [str(r) for r in refs]
    head = ", ".join(refs[:_SAMPLE_N])
    extra = len(refs) - _SAMPLE_N
    return f"{head}, … and {extra} more" if extra > 0 else head


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Archive cascade generators (kings + every participant) to private R2.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be fetched + archived; write nothing.")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--kings-only", action="store_true",
                       help="Only the kings/ throne archive.")
    scope.add_argument("--generators-only", action="store_true",
                       help="Only the generators/ every-participant snapshot.")
    ap.add_argument("--retry-unavailable", action="store_true",
                    help="Re-attempt generators parked as permanently unfetchable "
                         "(deleted or private upstream). Use after a Hub/HF token "
                         "widens, or when a miner republishes a repo.")
    args = ap.parse_args(argv)

    from cascade.shared.config import load_chain_config
    from cascade.shared.env import load_env_files
    from cascade.shared.hippius import HubConfig, S3Store, StorageError, open_manifest_store
    from cascade.shared.king_archive import (
        UNAVAILABLE_MAX_ATTEMPTS,
        king_archive_config,
        sync_generators,
        sync_kings,
    )

    load_env_files()
    cfg = load_chain_config(args.chain_toml)
    storage = cfg.storage

    manifest_store = open_manifest_store(storage)
    try:
        s3cfg, endpoint, bucket = king_archive_config(storage)
    except StorageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    archive_store = S3Store(s3cfg)
    hub = HubConfig.from_storage(storage)

    common = dict(
        manifest_store=manifest_store,
        archive_store=archive_store,
        hub=hub,
        endpoint=endpoint,
        bucket=bucket,
        dry_run=args.dry_run,
        updated_at=datetime.now(UTC).isoformat(),
        retry_unavailable=args.retry_unavailable,
        log=print,
    )
    tag = "[dry-run] " if args.dry_run else ""
    failed: list[str] = []
    unavailable: list[dict] = []

    if not args.generators_only:
        print(f"{tag}king archive → {endpoint.rstrip('/')}/{bucket}/kings/")
        try:
            result = sync_kings(**common)
        except StorageError as e:
            print(f"storage error: {e}", file=sys.stderr)
            return 4
        if args.dry_run:
            print(f"\n{tag}{result.would_archive} king(s) would be archived; "
                  f"{result.skipped} already archived; {result.total_kings} total in the db.")
        else:
            print(f"\ndone: {result.archived} archived, {result.skipped} already archived, "
                  f"{result.total_kings} total king(s) in {bucket}/kings/index.json")
        failed += result.failed
        unavailable += result.unavailable

    if not args.kings_only:
        print(f"\n{tag}generator snapshots → {endpoint.rstrip('/')}/{bucket}/generators/")
        try:
            result = sync_generators(**common)
        except StorageError as e:
            print(f"storage error: {e}", file=sys.stderr)
            return 4
        if args.dry_run:
            print(f"\n{tag}{result.would_archive} generator(s) would be snapshotted; "
                  f"{result.skipped} already snapshotted; "
                  f"{result.total_generators} total in the db.")
        else:
            print(f"\ndone: {result.archived} snapshotted, {result.skipped} already "
                  f"snapshotted, {result.rounds_scanned} new round(s) scanned, "
                  f"{result.total_generators} total generator(s) in "
                  f"{bucket}/generators/index.json")
        failed += result.failed
        unavailable += result.unavailable

    if unavailable:
        counts = Counter(str(u.get("reason", "?")) for u in unavailable)
        detail = ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
        print(f"\nnote: {len(unavailable)} generator(s) unavailable upstream "
              f"({detail}) — the miner deleted or privatised the repo, so no re-run "
              f"recovers them. Recorded in the index under 'unavailable'; parked "
              f"after {UNAVAILABLE_MAX_ATTEMPTS} attempts. Re-run with "
              f"--retry-unavailable to try them again.")
        print(f"  {_sample(u.get('ref', '?') for u in unavailable)}")

    if failed:
        print(f"error: {len(failed)} generator(s) failed to archive for a "
              f"retryable reason: {_sample(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
