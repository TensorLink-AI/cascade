#!/usr/bin/env python
"""Scrape every cascade king generator into the private R2 king archive.

Reads the validator's public throne history (``receipts/index.json`` in the
manifest bucket), and for every generator that has ever held the throne, fetches
its code from the Hippius Hub and saves it — packed to a deterministic tar — to
a **private** S3-compatible (Cloudflare R2) bucket, alongside a
``kings/index.json`` "db" that links each king back to its archived object. See
:mod:`cascade.shared.king_archive`.

The archive is content-addressed and append-only: a king already in the bucket
is never re-fetched, only its throne metadata (rounds reigned) is refreshed. Run
it as often as you like — hourly is plenty, rounds are epoch-paced.

Usage::

    # reads [storage] from chain.toml; needs S3 read creds for the manifest
    # bucket (HIPPIUS_S3_ACCESS_KEY / _SECRET_KEY), Hub pull creds (public repos
    # pull anonymously), and R2 write creds for the archive
    # (KING_ARCHIVE_S3_ACCESS_KEY / _SECRET_KEY, or the BACKUP_S3_* pair).
    python scripts/scrape_kings.py
    python scripts/scrape_kings.py --chain-toml chain.testnet.toml
    python scripts/scrape_kings.py --dry-run          # list what would be archived

Cron (hourly is ample — rounds are epoch-paced)::

    23 * * * * cd /opt/cascade && python scripts/scrape_kings.py >> king-archive.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Archive every cascade king generator to private R2.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the kings that would be fetched + archived; write nothing.")
    args = ap.parse_args(argv)

    from cascade.shared.config import load_chain_config
    from cascade.shared.env import load_env_files
    from cascade.shared.hippius import HubConfig, S3Store, StorageError, open_manifest_store
    from cascade.shared.king_archive import king_archive_config, sync_kings

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

    print(f"{'[dry-run] ' if args.dry_run else ''}king archive → {endpoint.rstrip('/')}/{bucket}/kings/")
    try:
        result = sync_kings(
            manifest_store=manifest_store,
            archive_store=archive_store,
            hub=hub,
            endpoint=endpoint,
            bucket=bucket,
            dry_run=args.dry_run,
            updated_at=datetime.now(timezone.utc).isoformat(),
            log=print,
        )
    except StorageError as e:
        print(f"storage error: {e}", file=sys.stderr)
        return 4

    if args.dry_run:
        print(f"\n[dry-run] {result.would_archive} king(s) would be archived; "
              f"{result.skipped} already archived; {result.total_kings} total in the db.")
    else:
        print(f"\ndone: {result.archived} archived, {result.skipped} already archived, "
              f"{result.total_kings} total king(s) in {bucket}/kings/index.json")
    if result.failed:
        print(f"warning: {len(result.failed)} king(s) failed to archive: "
              f"{', '.join(result.failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
