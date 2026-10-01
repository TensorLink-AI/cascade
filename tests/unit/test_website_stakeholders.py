"""The stakeholder scoreboard (``cascade/website/stakeholders.html``).

The page is a static asset with no build step, so nothing type-checks its
constants against the code they mirror. These tests are that check: the page
derives its numbers from the public receipt trail (fine — that trail is signed
and versioned), but three things are *copied* into it and would drift silently:

* the receipt trust anchor and the rolling-index cap, which must match
  ``chain.toml`` / :mod:`cascade.shared.hippius`;
* the training-token era table, which the efficiency metric divides by and which
  is transcribed from ``[training]`` in ``chain.toml``;
* the field names it reads out of ``summarize_receipt``'s round summary — rename
  one of those in Python and the tile quietly renders ``--`` forever.

Pure text/JSON assertions; no browser, no network.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.hippius import RECEIPT_INDEX_MAX_KEEP, WEBSITE_STAKEHOLDERS_KEY
from cascade.shared.receipt import summarize_receipt

from .receipt_fixture import make_scored_receipt

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "cascade" / "website" / WEBSITE_STAKEHOLDERS_KEY


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


def _js_number(html: str, name: str) -> float:
    m = re.search(rf"var {re.escape(name)}\s*=\s*([0-9.]+)\s*;", html)
    assert m, f"{name} not found in the page"
    return float(m.group(1))


def _token_eras(html: str) -> list[dict]:
    """Parse the ``TOKEN_ERAS`` literal out of the page (keys are unquoted JS)."""
    m = re.search(r"var TOKEN_ERAS = \[(.*?)\n\];", html, re.S)
    assert m, "TOKEN_ERAS table not found"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    # quote the bare JS keys only where a key can appear ("{" or ","), never
    # inside a value (era notes contain colons)
    return [
        json.loads(re.sub(r"([{,])\s*(\w+):", r'\1"\2":', row.strip().rstrip(",")))
        for row in re.findall(r"\{[^}]*\}", body)
    ]


def test_page_exists_and_is_self_contained(html: str):
    """No external CSS/JS/font/image loads: the page must render (and its
    numbers must be checkable) with only the receipt endpoints reachable."""
    assert PAGE.is_file()
    for tag in re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html, re.I):
        src = re.search(r'(?:src|href)\s*=\s*"([^"]+)"', tag)
        if not src:
            continue
        url = src.group(1)
        assert not url.startswith(("http://", "https://", "//")), f"external asset: {tag}"
    assert "@import" not in html


def test_trust_anchor_matches_chain_toml(html: str):
    """The page collapses peer entries in favour of the anchor validator. A
    stale anchor silently demotes the real signer to "just another peer" and the
    scoreboard can disagree with the technical dashboard about who is king."""
    cfg = load_chain_config(REPO / "chain.toml")
    m = re.search(r'var RECEIPT_ANCHOR = "([^"]+)"', html)
    assert m, "RECEIPT_ANCHOR not found"
    assert m.group(1) == cfg.manifest.validator_hotkey


def test_index_cap_matches_publisher(html: str):
    """The 'since launch' framing degrades to 'since the oldest kept round' at
    the cap; the page can only say so if it knows the real cap."""
    assert _js_number(html, "INDEX_MAX_KEEP") == RECEIPT_INDEX_MAX_KEEP


def test_token_eras_current_row_matches_training_contract(html: str):
    """Efficiency = improvement ÷ tokens, and tokens come from this table (the
    receipts carry the contract digest, not its parameters). The newest era must
    equal what ``chain.toml`` currently trains under."""
    cfg = load_chain_config(REPO / "chain.toml")
    eras = _token_eras(html)
    assert eras, "no eras parsed"
    assert [e["from"] for e in eras] == sorted(e["from"] for e in eras), "eras must be chronological"
    current = eras[-1]
    assert current["ref_tps"] == cfg.training.ref_throughput_tokens_per_s
    assert current["train_h"] == pytest.approx(cfg.training.target_train_hours)
    assert current["heat_h"] == pytest.approx(cfg.round.heat_train_hours)
    # the pretraining-completion section divides the token budget by these
    assert current["batch"] == cfg.training.batch_size
    assert current["ctx"] == cfg.training.context_length


def test_toto2_pretrain_targets_are_cited_or_null(html: str):
    """The pretraining-completion target is transcribed from the official
    release's published recipe. A filled step count without a citation (or a
    zero/negative one) is an invented number — refuse it here, before it ships."""
    m = re.search(r"var TOTO2_PRETRAIN = \{(.*?)\n\};", html, re.S)
    assert m, "TOTO2_PRETRAIN table not found"
    rows = re.findall(r'"([^"]+)":\s*\{\s*steps:\s*(null|[\d_.e]+),\s*source:\s*"([^"]*)"',
                      m.group(1))
    assert rows, "no TOTO2_PRETRAIN rows parsed"
    for key, steps, source in rows:
        if steps == "null":
            continue
        assert float(steps) > 0, f"{key}: non-positive step target"
        assert source.strip(), f"{key}: step target filled without a citation"


def test_page_reads_only_fields_the_summary_publishes(html: str):
    """Every round-summary field the page consumes must exist in
    ``summarize_receipt``'s output — a rename there would blank a tile."""
    receipt, _king_scores, _chal_scores = make_scored_receipt()
    summary = summarize_receipt(receipt)
    consumed = {
        "status", "round_id", "epoch_start_block", "dethroned", "inconclusive",
        "king_geomean", "chal_geomean", "lcb", "margin", "sizes", "heat",
        "king_hotkey", "king_uid", "king_gen_ref", "warm_start",
        "chal_hotkey", "chal_uid", "chal_gen_ref", "validator_hotkey",
    }
    missing = consumed - set(summary)
    assert not missing, f"page reads fields the summary does not publish: {sorted(missing)}"
    # The heat sub-object supplies the per-round entrant count the token math
    # needs. The fixture round has no heat screen (``heat`` is None there), so
    # assert on the summariser's own construction of it.
    assert '"n_entrants"' in inspect.getsource(summarize_receipt)


def test_pending_states_are_explicit(html: str):
    """Metrics that cannot be measured yet must render an explicit pending
    state rather than a placeholder number — the whole page's credibility rests
    on never showing a figure nobody can check."""
    lower = html.lower()
    assert "awaiting first result" in lower, "public benchmark must pend explicitly"
    assert "not active yet" in lower, "carry-forward must pend explicitly"
    for phrase in ("bench_scores", "warm_start_ckpt"):
        assert phrase in html, f"page must key its pending state off {phrase}"


def test_links_to_and_from_the_technical_dashboard(html: str):
    """The two dashboards read the same trail; each must be reachable from the
    other so a stakeholder can drill into any headline number."""
    assert 'href="index.html"' in html
    index = (REPO / "cascade" / "website" / "index.html").read_text(encoding="utf-8")
    assert f'href="{WEBSITE_STAKEHOLDERS_KEY}"' in index


def _bench_entry_json_keys() -> set[str]:
    """The keys a signed bench report entry actually carries on the wire —
    taken from :meth:`BenchReport.canonical_body`, not transcribed."""
    from cascade.shared.bench_report import BenchEntry, BenchReport
    from cascade.shared.manifest import BenchScores

    six = dict(gifteval_crps=0.5, gifteval_mase=0.8, boom_crps=0.4,
               boom_mase=0.6, time_crps=0.5, time_mase=0.7)
    entry = BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="5Hk",
                       miner_uid=133, trained_pointer="metro-v1:trained:hippius:cascade/x@sha256:ab",
                       scores=BenchScores(**six))
    body = json.loads(BenchReport(round_id="1", created_block=1, entries=(entry,)).canonical_body())
    (wire,) = body["entries"]
    return set(wire) | {f"scores.{k}" for k in wire["scores"]}


def test_best_checkpoint_card_reads_the_report_wire_fields(html: str):
    """The best-checkpoint card names the checkpoint from the report entry:
    uid, hotkey and the Hub pointer. Rename one of those in
    cascade.shared.bench_report and the card would silently render a nameless
    checkpoint — this pins the names the page reads to the serialisation."""
    keys = _bench_entry_json_keys()
    for field in ("miner_uid", "miner_hotkey", "trained_pointer", "role"):
        assert field in keys, f"bench report entries no longer carry {field!r}"
        assert re.search(rf"\be\.{field}\b", html), f"the page no longer reads e.{field}"
    for k in ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase"):
        assert f"scores.{k}" in keys
        assert f'"{k}"' in html, f"the page no longer reads {k}"


def test_best_checkpoint_card_strips_the_pointer_scheme(html: str):
    """The card shows the bare Hub ``repo@digest`` (what a registry pull takes),
    so the prefix it strips must be exactly the trained-pointer scheme."""
    from cascade.shared.manifest import TRAINED_RE

    m = re.search(r'var HUB_POINTER_PREFIX = "([^"]+)"', html)
    assert m, "HUB_POINTER_PREFIX not found"
    prefix = m.group(1)
    ref = "cascade/ckpt-r1-challenger-toto2-4m-u133@sha256:" + "ab" * 32
    parsed = TRAINED_RE.match(prefix + ref)
    assert parsed and parsed.group("ref") == ref, "the trained-pointer scheme moved; update HUB_POINTER_PREFIX"


def test_best_checkpoint_card_has_explicit_pending_state(html: str):
    assert 'id="best-panel"' in html and 'id="best-answer"' in html
    assert "Awaiting first benched checkpoint" in html
    assert "renderBest()" in html
