"""WB14: the lane B commit intake wire point. ``POST /v1/lane-b/fund``
(``FundingIntake.lane_b_fund``) parses the optional declared-rows field
(header ``X-Lane-B-Declared-Rows``, a JSON object keyed by MoE ordinal) with
``parse_declared_rows`` under the loaded ``[lane_b]`` fractions, refuses a bad
one with a typed 400 before anything is queued or vaulted, binds it in the
signed message, and stores the parsed rows on the lane B queue entry.

No declared rows (header absent) is today's request, byte for byte: TB21's
trainer-picked (ESFT) default."""

from __future__ import annotations

import ast
import dataclasses
import hashlib

import pytest

from . import helpers as h
from .helpers import BAD_LAYER, COUNT, MALFORMED, OUT_OF_RANGE


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


def _entry_rows(lane_b):
    e = lane_b.get(h.HK)
    assert e is not None, "no lane B entry queued"
    return getattr(e, "declared_rows", None)


def test_header_constant_is_exported():
    from cascade.funding.intake import LANE_B_DECLARED_ROWS_HEADER  # noqa: I001

    assert LANE_B_DECLARED_ROWS_HEADER == h.HEADER


def test_production_intake_is_wired_with_the_loaded_lane_b_config():
    tree = ast.parse(h.FUNDING_MAIN.read_text(encoding="utf-8"))
    kws = [kw for node in ast.walk(tree)
           if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "FundingIntake"
           for kw in node.keywords]
    assert any(kw.arg == "lane_b_cfg" for kw in kws), (
        "cascade/funding/main.py must pass lane_b_cfg (cfg.lane_b) to FundingIntake")


# ── invariant 1 / 10 pair ────────────────────────────────────────────────────


def test_inv1_lane_b_fund_unchanged_when_unarmed(tmp_path):
    """No declared rows: 202 queued, the signed message is today's
    ``canonical_fund_message`` byte for byte, the entry carries no rows and
    the queue file never mentions them."""
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref  # noqa: I001

    seen = []
    t = h.intake(tmp_path, sig=True, verify=lambda hk, m, s: seen.append(m) or s == "good")
    ts = str(int(t.clock()))
    status, body = t.intake.lane_b_fund(h.headers(**{"X-Timestamp": ts, "X-Signature": "good"}))
    assert (status, body["status"], body["targets"]) == (202, "queued", ["energy"])
    digest = hashlib.sha256(h.API_KEY.encode()).hexdigest()
    assert seen[-1] == canonical_fund_message(
        "lane-b-fund", h.HK, lane_b_signed_ref(("energy",), h.REF), ts, digest)
    assert _entry_rows(t.lane_b) is None
    assert "declared_rows" not in h.queue_text(t.qpath)


def test_inv1_lane_b_fund_armed_nonempty_lane_d_effect(tmp_path):
    """Declared rows: 202 queued, the parsed rows ride the entry (int MoE
    ordinals, sorted) and survive a reload, and the signature binds them."""
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref  # noqa: I001
    from cascade.funding.queue import LaneBQueue

    seen = []
    t = h.intake(tmp_path, sig=True, verify=lambda hk, m, s: seen.append(m) or s == "good")
    ts = str(int(t.clock()))
    sel = {"0": [3, 1, 2, 0], "4": h.rows(16, start=300)}
    status, body = t.intake.lane_b_fund(
        h.with_rows(sel, **{"X-Timestamp": ts, "X-Signature": "good"}))
    assert (status, body["status"]) == (202, "queued"), body
    want = {0: [0, 1, 2, 3], 4: h.rows(16, start=300)}
    got = _entry_rows(t.lane_b)
    assert got is not None and h.nonempty(dict(got)) == want
    again = LaneBQueue(t.qpath, clock=t.clock).get(h.HK)
    assert h.nonempty(dict(getattr(again, "declared_rows", None) or {})) == want
    # the signature covers the rows: not today's message, and rows-specific
    digest = hashlib.sha256(h.API_KEY.encode()).hexdigest()
    plain = canonical_fund_message("lane-b-fund", h.HK, lane_b_signed_ref(("energy",), h.REF),
                                   ts, digest)
    with_rows = seen[-1]
    assert with_rows != plain
    (tmp_path / "other").mkdir()
    t2 = h.intake(tmp_path / "other", sig=True,
                  verify=lambda hk, m, s: seen.append(m) or s == "good")
    other = {"0": [4, 5, 6, 7], "4": h.rows(16, start=300)}
    t2.intake.lane_b_fund(h.with_rows(other, **{"X-Timestamp": ts, "X-Signature": "good"}))
    assert seen[-1] not in (plain, with_rows)


# ── typed refusals at intake (TB13, TB15, shape, total) ──────────────────────


@pytest.mark.parametrize("sel,code", [
    ({"0": [0, 1, 2, 128]}, OUT_OF_RANGE),                 # TB13: 128-expert ordinal 0
    ({"7": h.rows(16)}, BAD_LAYER),                         # TB13: past ordinal 6
    ({"8": h.rows(16)}, BAD_LAYER),                         # TB13: block index 8
    ({"0": h.rows(3)}, COUNT),                              # TB15: 128 -> 4..19
    ({"0": h.rows(20)}, COUNT),
    ({"1": h.rows(7)}, COUNT),                              # TB15: 256 -> 8..38
    ({"2": h.rows(39)}, COUNT),
    ({"3": h.rows(15)}, COUNT),                             # TB15: 512 -> 16..76
    ({"6": h.rows(77)}, COUNT),
    ({}, COUNT),                                            # at least one row in total
    ({"0": [1, 1, 2, 3, 4]}, COUNT),                        # duplicates
    ("not json", MALFORMED),
    ("[0, 1]", MALFORMED),
])
def test_bad_declared_rows_refused_before_queueing(tmp_path, sel, code):
    t = h.intake(tmp_path)
    status, body = t.intake.lane_b_fund(h.with_rows(sel))
    assert (status, body.get("code")) == (400, code), body
    assert t.lane_b.entries() == []
    assert t.intake.vault.get(h.HK) is None


def test_tb15b_two_layers_accepted_at_intake(tmp_path):
    t = h.intake(tmp_path)
    sel = {"0": h.rows(19), "5": h.rows(16, start=496)}
    sel.update({str(i): [] for i in (1, 2, 3, 4, 6)})
    status, body = t.intake.lane_b_fund(h.with_rows(sel))
    assert status == 202, body
    assert h.nonempty(dict(_entry_rows(t.lane_b))) == {0: h.rows(19),
                                                       5: h.rows(16, start=496)}


def test_intake_clamp_uses_the_configured_fractions(tmp_path):
    """The intake reads ``min_experts_frac`` / ``max_experts_frac`` from the
    ``[lane_b]`` config it is given: at 0.0625 / 0.125, 128 experts ⇒ 8..16."""
    lb = dataclasses.replace(h.live_lane_b(), min_experts_frac=0.0625, max_experts_frac=0.125)
    t = h.intake(tmp_path, lane_b_cfg=lb)
    status, body = t.intake.lane_b_fund(h.with_rows({"0": h.rows(4)}))
    assert (status, body.get("code")) == (400, COUNT), body
    status, body = t.intake.lane_b_fund(h.with_rows({"0": h.rows(8)}))
    assert status == 202, body
    assert h.nonempty(dict(_entry_rows(t.lane_b))) == {0: h.rows(8)}
