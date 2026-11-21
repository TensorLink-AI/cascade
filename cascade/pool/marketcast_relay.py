"""Relaying the forward bar recording to a MarketCast training pod.

Hyperliquid keeps only its last ~5000 candles per interval (5m ≈ 17 days,
15m ≈ 52), and a recipe's range ends at least ``embargo_days`` before its
submission, so a pod fetching the API itself gets no sub-hour candles for
it. The recording (:mod:`.marketcast_store`, mirrored to a private Hippius
bucket) holds them. The pod never holds a storage credential (DEC-CA-0053's
pattern): the trainer lists the recording's objects inside the recipe's
range and signs one time-limited GET link per object; the pod downloads
exactly those into a local :class:`~.marketcast_store.MktBarStore`, which
:func:`cascade.trainer.marketcast_leg.build_training_snapshot` reads from.

``mkt_bars.json`` = ``{"version": 1, "links": {relative path: GET link}}``,
relayed in the leg's ``--corpus-dir``.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .marketcast_feeds import FIELDS

RELAY_FILE = "mkt_bars.json"
RELAY_VERSION = 1
RELAY_EXPIRES_S = 24 * 3600        # queue + rent + a 12 h leg wall, with slack
DAY = 86400
# <class>/<asset>/[<iv>/]<YYYY-MM-DD>/<field>.npy | <class>/<asset>/funding.jsonl
_REL = re.compile(r"[a-z_]{1,32}/[A-Za-z0-9]{1,32}/(?:(?:5m|15m|1h)/)?"
                  r"\d{4}-\d\d-\d\d/(?:open|high|low|close|volume)\.npy"
                  r"|[a-z_]{1,32}/[A-Za-z0-9]{1,32}/(?:funding|ctx)\.jsonl")


def relay_rels(assets, intervals, t0: int, t1: int) -> list[str]:
    """Every recording path a range ``[t0, t1)`` of ``assets`` (``[(class,
    [asset, ...]), ...]``) at ``intervals`` (``"1m"`` = the minute bank) can
    hold, funding included."""
    days = [datetime.fromtimestamp(d, tz=UTC).strftime("%Y-%m-%d")
            for d in range(int(t0) // DAY * DAY, int(t1), DAY)]
    out = []
    for cls, syms in assets:
        for a in syms:
            out += [f"{cls}/{a}/funding.jsonl", f"{cls}/{a}/ctx.jsonl"]
            for iv in intervals:
                sub = "" if iv == "1m" else f"{iv}/"
                out += [f"{cls}/{a}/{sub}{d}/{f}.npy" for d in days for f in FIELDS]
    return sorted(out)


def relay_plan(store, prefix: str, rels, *, expires_s: int = RELAY_EXPIRES_S) -> dict:
    """``mkt_bars.json``'s body: a GET link for each of ``rels`` the
    recording under ``prefix`` actually holds (listed, so a day nobody
    recorded is simply absent — NaN on the pod, as in the store)."""
    from ..generation.transfer import presign

    want = set(rels)
    have: set[str] = set()
    roots = sorted({"/".join(r.split("/")[:2]) + "/" for r in want})
    pager = store.client().get_paginator("list_objects_v2")
    for root in roots:
        for page in pager.paginate(Bucket=store.cfg.bucket, Prefix=prefix + root):
            for o in page.get("Contents", []):
                rel = o["Key"][len(prefix):]
                if rel in want:
                    have.add(rel)
    return {"version": RELAY_VERSION,
            "links": {r: presign(store, "get_object", prefix + r, expires_s) for r in sorted(have)}}


def fetch_relay(plan: dict, dest, *, get_file=None, workers: int = 32):
    """Download a relay plan into ``dest`` and return the
    :class:`~.marketcast_store.MktBarStore` over it. Every path is checked
    against the recording's layout (no traversal, no other file kinds);
    a link that still fails after the transfer layer's retries raises."""
    from .marketcast_store import MktBarStore

    if get_file is None:
        from ..generation.transfer import get_file
    if not isinstance(plan, dict) or plan.get("version") != RELAY_VERSION:
        raise ValueError(f"{RELAY_FILE}: unsupported relay plan")
    links = plan.get("links")
    if not isinstance(links, dict):
        raise ValueError(f"{RELAY_FILE}: no links")
    root = Path(dest)
    bad = [r for r in links if not isinstance(r, str) or not _REL.fullmatch(r)]
    if bad:
        raise ValueError(f"{RELAY_FILE}: path outside the recording layout: {bad[0]!r}")

    def _one(item):
        rel, url = item
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        get_file(str(url), p)

    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        list(ex.map(_one, sorted(links.items())))
    return MktBarStore(root)


def read_relay_file(corpus_dir) -> dict | None:
    """The relay plan in a leg's ``--corpus-dir``, or None when absent."""
    if corpus_dir is None:
        return None
    p = Path(corpus_dir) / RELAY_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


__all__ = ["RELAY_EXPIRES_S", "RELAY_FILE", "RELAY_VERSION", "fetch_relay", "read_relay_file",
           "relay_plan", "relay_rels"]
