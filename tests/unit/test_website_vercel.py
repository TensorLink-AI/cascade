"""The Vercel routing config (``cascade/website/vercel.json``).

Since 2026-10-05 both dashboards read their data from the bucket DIRECTLY
first (Hippius has CORS ``*``) and use the SAME-ORIGIN Vercel proxy only as the
last-resort fallback: proxying every poll through Vercel was the site's whole
data-transfer bill. The fallback still needs a rewrite per bucket prefix, or a
Hippius outage degrades a section to its pending state — which is exactly how
``benchmarks/`` once shipped: rewrites existed for ``receipts/`` and
``status/`` only, and the Public benchmark cell stayed dashed.

These tests derive the required prefixes FROM the pages, so adding a fetch of a
new prefix fails here until the rewrite exists. Pure text/JSON; no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WEBSITE = REPO / "cascade" / "website"
PAGES = ("index.html", "stakeholders.html")

MAINNET_BUCKET = "cascade-manifests"
TESTNET_BUCKET = "cascade-testnet-manifests"


@pytest.fixture(scope="module")
def vercel() -> dict:
    return json.loads((WEBSITE / "vercel.json").read_text(encoding="utf-8"))


def fetched_prefixes() -> set[str]:
    """Every bucket prefix the pages fetch, e.g. {"receipts", "status", ...}."""
    found: set[str] = set()
    for name in PAGES:
        html = (WEBSITE / name).read_text(encoding="utf-8")
        found |= set(re.findall(r'fetchJSON\(\s*"([a-z_]+)/', html))
    return found


def test_pages_fetch_the_prefixes_we_think_they_do():
    # Guards the regex above: if a page starts fetching a prefix by some other
    # spelling, this test — not a dark tile in production — is what notices.
    assert fetched_prefixes() == {"receipts", "status", "benchmarks", "promotions", "funded", "training"}


@pytest.mark.parametrize("prefix", sorted(fetched_prefixes()))
def test_every_fetched_prefix_has_a_rewrite(vercel: dict, prefix: str):
    rules = [r for r in vercel["rewrites"] if r["source"] == f"/{prefix}/:path*"]
    assert len(rules) == 2, f"/{prefix}/ needs a testnet-host rule and a default rule"

    testnet = [r for r in rules if r.get("has")]
    default = [r for r in rules if not r.get("has")]
    assert len(testnet) == 1 and len(default) == 1

    assert testnet[0]["has"] == [{"type": "host", "value": "testnet\\..*"}]
    assert testnet[0]["destination"].endswith(f"/{TESTNET_BUCKET}/{prefix}/:path*")
    assert default[0]["destination"].endswith(f"/{MAINNET_BUCKET}/{prefix}/:path*")

    # Order matters: Vercel takes the first match, so the unconditional rule
    # must come AFTER the host-conditional one or testnet reads mainnet data.
    assert vercel["rewrites"].index(testnet[0]) < vercel["rewrites"].index(default[0])


@pytest.mark.parametrize("prefix", sorted(fetched_prefixes()))
def test_every_fetched_prefix_is_cached(vercel: dict, prefix: str):
    rules = [h for h in vercel["headers"] if h["source"] == f"/{prefix}/(.*)"]
    assert len(rules) == 1, f"/{prefix}/ needs a Cache-Control header rule"
    keys = {h["key"]: h["value"] for h in rules[0]["headers"]}
    assert "s-maxage=" in keys["Cache-Control"]


def test_scoreboard_routes_reach_the_stakeholder_page(vercel: dict):
    dests = {r["source"]: r["destination"] for r in vercel["rewrites"]}
    assert dests["/scoreboard"] == "/stakeholders.html"
    assert dests["/stakeholders"] == "/stakeholders.html"


@pytest.mark.parametrize("page", PAGES)
def test_pages_spare_the_vercel_proxy(page):
    """2026-10-05: each open tab pulled ~11 MB through Vercel every 30 s (all
    endpoints raced, ?t= cache-buster, background tabs polling). Guard the fix:
    no per-request buster, the same-origin proxy tried last, ETag reuse, slower
    polling paused while the tab is hidden."""
    html = (WEBSITE / page).read_text(encoding="utf-8")
    fetch_json = re.search(r"function fetchJSON\(.*?\n\}\n", html, re.S).group(0)
    assert "Date.now()" not in fetch_json and "\"?t=\"" not in fetch_json   # no buster literal
    assert "concat([SAME_ORIGIN])" in fetch_json
    assert 'cache: immutable ? "default" : "no-cache"' in fetch_json
    assert "_jsonMemo[path]" in fetch_json
    assert int(re.search(r"POLL_MS\s*=\s*(\d+)", html).group(1)) >= 60000
    assert "if(!document.hidden) poll();" in html
    assert '"visibilitychange"' in html


@pytest.mark.parametrize("page", PAGES)
def test_a_missing_object_is_not_retried_on_every_endpoint(page):
    """2026-10-06: the in-order fallback turned each missing bench/training doc
    (76 of 240 on the training tab) into four sequential misses, the last through
    Vercel, and the tab loaded far slower. A 403/404 is final; only network
    errors, timeouts and 5xx move on to the next endpoint."""
    html = (WEBSITE / page).read_text(encoding="utf-8")
    fetch_json = re.search(r"function fetchJSON\(.*?\n\}\n", html, re.S).group(0)
    assert "r.status===403||r.status===404" in fetch_json
    assert "miss.missing=true" in fetch_json
    assert "if(e&&e.missing) throw e;" in fetch_json
