"""W12a: `docs/LANE_D.md` (new), the miner guide for lane D (plan §11.2 first
row, the W12a row's rev3 FAQ items). Contract: README.md "docs/LANE_D.md"."""

from __future__ import annotations

import re

import pytest

from . import helpers as h

FAQ = re.compile(r"\b(?:FAQ|frequently asked)\b", re.I)
# The FAQ keys (§1.2 rate limits, §1.5 reveal delay, B10, A1) with their
# defaults read from LaneDConfig.
FAQ_KEYS = ("max_open_per_coldkey", "max_judged_per_coldkey_per_settlement",
            "max_entries_per_settlement", "max_coldkey_unresolved_settlements",
            "max_reveal_delay_days", "ready_ttl_blocks", "max_mirror_lag_hours")


def _doc() -> str:
    return h.require(h.LANE_D_MD)


def _faq() -> str:
    return h.require_sections(h.LANE_D_MD, FAQ, "FAQ")


def _need(text: str, *needles, negation=False, what=""):
    p = h.paragraph_with(text, *needles, negation=negation)
    assert p is not None, f"docs/LANE_D.md: no paragraph {what or needles!r}"
    return p


def test_the_guide_exists_with_a_lane_d_title():
    text = _doc()
    first = next((ln for ln in text.splitlines() if ln.startswith("# ")), "")
    assert h.LANE_D_HEADING.search(first), f"title {first!r} does not name lane D"


def test_checkpoint_format_names_the_allowed_files_and_bf16():
    text = _doc()
    for name in h.allowed_files():
        assert h.has_token(text, name), f"the checkpoint format does not name `{name}`"
    for token in ("pyramid_model.py", "pyramid_wrapper.py"):
        assert h.has_token(text, token), f"the code-bytes rule does not name `{token}`"
    _need(text, "BF16", "F32", what="(BF16 only; lane A's F32 refused)")


def test_contexts_are_the_armed_defaults_with_s8_later():
    text = _doc()
    for c in h.lane_d_defaults()["contexts"]:
        assert re.search(rf"\b{c}\b|\b{c // 1024}k\b", text), f"context {c} is not documented"
    _need(text, "S8", "context", what="(longer contexts at S8)")


@pytest.mark.parametrize("cmd", h.PLAN_SUBCOMMANDS)
def test_every_cli_command_is_documented(cmd):
    assert any(f"cascade lane-d {cmd}" in s for s in h.spans(_doc())), \
        f"`cascade lane-d {cmd}` is not documented in a code span or block"


def test_the_dedicated_hotkey_rule():
    _need(_doc(), "dedicated hotkey", what="(lane D uses a dedicated hotkey)")


def test_the_licence_attestation_and_takedown():
    text = _doc()
    _need(text, "Apache-2.0", "OpenMDW", "training data", "teacher",
          what="(the B13 licence attestation at ready)")
    _need(text, "takedown", what="(the Hub/HF takedown path)")


def test_the_commit_reveals_immediately():
    _need(_doc(), "reveal", re.compile(r"\bimmediate(?:ly)?\b", re.I), "commit",
          what="(D43: the commit reveals immediately)")


def test_forward_only_and_latency_by_frequency():
    _need(_doc(), "forward-only", "latency",
          re.compile(r"\b(?:hourly|daily|frequency)\b", re.I),
          what="(the forward-only rule and expected latency by frequency)")


@pytest.mark.parametrize("word", ["house", "gate", "sentinel", "court"])
def test_house_gate_sentinel_and_court_are_explained(word):
    assert re.search(rf"\b{word}\b", _doc(), re.I), f"docs/LANE_D.md never explains the {word}"


def test_there_is_a_faq():
    _faq()


@pytest.mark.parametrize("key", FAQ_KEYS)
def test_faq_documents_each_cap_and_bound_with_its_default(key):
    default = h.lane_d_defaults()[key]
    assert h.documents_default(_faq(), key, default), \
        f"the FAQ does not document `{key}` with its default {default!r}"


def test_faq_names_every_typed_skip_reason():
    faq = _faq()
    missing = [r for r in h.skip_reasons() if not h.has_token(faq, r)]
    assert not missing, f"the FAQ does not name the typed skip reasons {missing}"


def test_faq_confirmation_duel_latency():
    _need(_faq(), "confirmation", re.compile(r"\b(?:latency|settlement|later)\b", re.I),
          what="(the confirmation duel and its latency)")


def test_faq_both_clocks_and_mirror_lag():
    _need(_faq(), "first-seen", re.compile(r"\bdata[- ]time\b", re.I), "mirror lag",
          what="(both clocks and mirror lag)")


def test_faq_bf16_only():
    _need(_faq(), "BF16", what="(BF16 only)")


def test_every_named_lane_d_key_is_real():
    text = _doc()
    keys = h.lane_d_keys()
    for s in h.spans(text):
        m = re.fullmatch(r"\[lane_d\]\s*([a-z][a-z0-9_]*)(?:\s*=.*)?", s)
        if m:
            assert m.group(1) in keys, f"`{s}` names no LaneDConfig field"
