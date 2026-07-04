#!/usr/bin/env python
"""Publish the cascade dashboard ("notebook") to the manifest bucket, public-read.

The dashboard is a single self-contained ``cascade/website/index.html`` that
reads the validator's public-read receipts — ``receipts/index.json`` (the
rolling round summary the validator maintains) and ``receipts/latest.json`` —
straight from the manifest bucket. Serving the page from that same bucket means
one public origin for everything (mirrors teutonic, whose validator re-uploads
its dashboard on restart).

Usage::

    # reads [storage] from chain.toml, needs HIPPIUS_S3_ACCESS_KEY / _SECRET_KEY
    python scripts/publish_website.py
    python scripts/publish_website.py --chain-toml chain.testnet.toml

The published objects land at the bucket root::

    s3://<manifest_bucket>/index.html
    s3://<manifest_bucket>/favicon.svg

reachable at ``<s3_endpoint>/<manifest_bucket>/index.html``. Edit the ``BUCKET``
/ ``ENDPOINTS`` constants at the top of ``index.html`` if your public read path
differs from the write endpoint.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from cascade.shared.config import load_chain_config
from cascade.shared.hippius import (
    WEBSITE_INDEX_KEY,
    S3Config,
    S3Store,
    publish_website,
)


def _read_asset(name: str) -> str:
    """Read a packaged website asset (works from a wheel or a source checkout)."""
    return resources.files("cascade.website").joinpath(name).read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the cascade dashboard to the manifest bucket.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be published and exit.")
    args = ap.parse_args()

    cfg = load_chain_config(args.chain_toml)
    bucket = cfg.storage.manifest_bucket
    s3cfg = S3Config.from_storage(cfg.storage, bucket=bucket)
    base = f"{s3cfg.endpoint.rstrip('/')}/{bucket}"

    index_html = _read_asset(WEBSITE_INDEX_KEY)
    favicon = _read_asset("favicon.svg")

    if args.dry_run:
        print(f"[dry-run] would publish {len(index_html)} B index.html + "
              f"{len(favicon)} B favicon.svg to {base}/")
        return

    store = S3Store(s3cfg)
    publish_website(store, index_html)
    try:
        store.put_text("favicon.svg", favicon,
                       content_type="image/svg+xml", acl="public-read")
    except Exception:  # noqa: BLE001 — favicon is cosmetic; the page still works
        store.put_text("favicon.svg", favicon, content_type="image/svg+xml")

    print(f"published dashboard → {base}/index.html")
    print("it reads (public-read):")
    print(f"  {base}/receipts/index.json")
    print(f"  {base}/receipts/latest.json")


if __name__ == "__main__":
    main()
