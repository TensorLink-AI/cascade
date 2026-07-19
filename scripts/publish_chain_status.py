#!/usr/bin/env python
"""Publish the dashboard's live ``status/chain.json`` to the manifest bucket.

The web dashboard is a static page; between round receipts it has no live view
of the chain. This publishes that view — current block, the epoch grid, the
round-stage windows, and every revealed generator commitment — public-read, so
the page's round-stage strip and live submissions panel work (see
``cascade.shared.chain_status``).

The validator publishes the same document on its poll cadence
(``ValidatorRunner._publish_chain_status``); this script is the standalone
alternative — run it from cron or ``--loop`` on any box with chain access and
the S3 credentials (``HIPPIUS_S3_ACCESS_KEY`` / ``_SECRET_KEY``), e.g. while a
validator predating the feed is still deployed. Read-only on chain; no wallet.

Usage::

    python scripts/publish_chain_status.py                       # one shot, mainnet
    python scripts/publish_chain_status.py --chain-toml chain.testnet.toml \
        --network test --loop 60                                 # keep publishing
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from cascade.shared.chain import ChainClient, ChainError
from cascade.shared.chain_status import build_chain_status, publish_chain_status
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import open_manifest_store


def _publish_once(client: ChainClient, cfg, store) -> str:
    status = build_chain_status(
        cfg,
        current_block=client.current_block(),
        commitments=client.poll_commitments(),
        network=client.network,
        as_of=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    key = publish_chain_status(store, status)
    print(f"published {key}: block {status['current_block']:,}, "
          f"{len(status['submissions'])} submission(s)")
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish status/chain.json for the dashboard.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    ap.add_argument("--loop", type=float, default=0.0, metavar="SECONDS",
                    help="Re-publish every SECONDS (0 = publish once and exit).")
    args = ap.parse_args()

    cfg = load_chain_config(args.chain_toml)
    client = ChainClient.from_config(cfg, network=args.network)
    store = open_manifest_store(cfg.storage)

    while True:
        try:
            _publish_once(client, cfg, store)
        except (ChainError, Exception) as e:  # noqa: BLE001 — a loop must survive flakes
            print(f"publish failed: {e}", file=sys.stderr)
            if args.loop <= 0:
                return 1
        if args.loop <= 0:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
