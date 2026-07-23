#!/usr/bin/env python
"""Announce subnet events to a Discord channel — the dashboard, pushed to chat.

Polls the same public-read JSON feeds the web dashboard consumes and posts a
message when something happens (see ``cascade.shared.announce``):

* round settled            — new round in ``receipts/index.json``
* stage transition         — ``status/round.json`` heat → duel → validation
* submission revealed      — new commitment in ``status/chain.json``
* benchmark refreshed      — new report in ``status/bench.json`` (if published)

Read-only and credential-free on the cascade side: every feed is an anonymous
HTTPS GET against the public manifest bucket (no chain access, no wallet, no
S3 keys), so this runs on any box. The only secret is the Discord webhook URL,
taken from ``DISCORD_WEBHOOK_URL`` (env only, never a flag — flags leak into
process lists). Create one in Discord: channel → Integrations → Webhooks.

Announced-state persists in a small JSON file (``--state``) so cron runs and
restarts never re-post; a fresh state file primes silently instead of
replaying history. At-most-once by design: state advances even if a webhook
POST fails (a missed message beats a spam loop — the dashboard remains the
record).

Usage::

    export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/…"
    python scripts/announce_events.py --loop 60          # keep announcing
    python scripts/announce_events.py                    # one check (cron-friendly)
    python scripts/announce_events.py --dry-run --loop 60  # print, don't post
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from cascade.shared.announce import (
    BENCH_STATUS_KEY,
    AnnouncerState,
    clamp_content,
    diff_events,
)
from cascade.shared.chain_status import CHAIN_STATUS_KEY, ROUND_STATUS_KEY
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import RECEIPT_INDEX_KEY


def fetch_public_json(storage: object, key: str, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET one public-read JSON doc from the manifest bucket (the
    same zero-credential path the dashboards use). Best-effort: any failure
    returns None and that feed's events simply wait for the next poll."""
    endpoint = str(getattr(storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(storage, "manifest_bucket", "") or "")
    if not endpoint.startswith(("http://", "https://")) or not bucket:
        return None
    url = f"{endpoint}/{bucket}/{key}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — a flaky feed must not kill the loop
        return None
    return doc if isinstance(doc, dict) else None


def post_discord(webhook_url: str, text: str, *, timeout: float = 10.0) -> bool:
    """POST one message to the webhook. Honors a single 429 retry-after."""
    body = json.dumps({"content": clamp_content(text)}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        webhook_url, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "cascade-announcer/1"},
    )
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout):  # noqa: S310
                return True
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                retry_after = e.headers.get("Retry-After") or "2"
                try:
                    time.sleep(min(30.0, float(retry_after)))
                except ValueError:
                    time.sleep(2.0)
                continue
            print(f"webhook post failed: HTTP {e.code}", file=sys.stderr)
            return False
        except Exception as e:  # noqa: BLE001 — network flake; drop the message
            print(f"webhook post failed: {e}", file=sys.stderr)
            return False
    return False


def load_state(path: Path) -> AnnouncerState:
    try:
        return AnnouncerState.from_json(path.read_text(encoding="utf-8"))
    except OSError:
        return AnnouncerState()


def save_state(path: Path, state: AnnouncerState) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(state.to_json(), encoding="utf-8")
    tmp.replace(path)


def check_once(storage: object, state: AnnouncerState,
               webhook_url: str | None, *, kinds: set[str]) -> AnnouncerState:
    events, new_state = diff_events(
        state,
        index_doc=fetch_public_json(storage, RECEIPT_INDEX_KEY),
        chain_doc=fetch_public_json(storage, CHAIN_STATUS_KEY),
        round_doc=fetch_public_json(storage, ROUND_STATUS_KEY),
        bench_doc=fetch_public_json(storage, BENCH_STATUS_KEY),
    )
    posted = 0
    for ev in events:
        if ev.kind not in kinds:
            continue
        if webhook_url is None:
            print(f"[dry-run] {ev.kind}: {ev.text}")
        else:
            if post_discord(webhook_url, ev.text):
                posted += 1
            time.sleep(1.0)  # stay far under Discord's webhook rate limit
    if events:
        print(f"{len(events)} event(s), {posted} posted")
    return new_state


ALL_KINDS = ("round_settled", "stage", "submission", "bench")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post cascade round/stage/submission/benchmark events to Discord.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--state", type=Path, default=Path("announcer_state.json"),
                    help="Announced-state file (prevents re-posting across runs).")
    ap.add_argument("--loop", type=float, default=0.0, metavar="SECONDS",
                    help="Re-check every SECONDS (0 = check once and exit).")
    ap.add_argument("--events", default=",".join(ALL_KINDS),
                    help=f"Comma-separated subset of {ALL_KINDS} to announce.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print events instead of posting (no webhook needed).")
    args = ap.parse_args()

    kinds = {k.strip() for k in args.events.split(",") if k.strip()}
    unknown = kinds - set(ALL_KINDS)
    if unknown:
        print(f"unknown event kind(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    webhook_url: str | None = None
    if not args.dry_run:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None
        if webhook_url is None:
            print("DISCORD_WEBHOOK_URL is not set (or use --dry-run)", file=sys.stderr)
            return 2

    storage = load_chain_config(args.chain_toml).storage
    state = load_state(args.state)

    while True:
        try:
            state = check_once(storage, state, webhook_url, kinds=kinds)
            save_state(args.state, state)
        except Exception as e:  # noqa: BLE001 — a loop must survive flakes
            print(f"check failed: {e}", file=sys.stderr)
            if args.loop <= 0:
                return 1
        if args.loop <= 0:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
