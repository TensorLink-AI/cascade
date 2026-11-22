"""WB13: the contamination check at lane B intake (``POST /v1/lane-b/fund``).

``FundingIntake.lane_b_fund`` runs the check after the identity gate
(signature, registration, key, revealed commitment) and before anything is
queued or vaulted ("At intake, **before any compute**", §12b.1). It fetches
the submitted tree with the injected ``fetch_lane_b_repo(ref)`` and scans it
with ``scan_repo`` against the injected ``lane_b_hashset``, which must be the
artefact ``[lane_b] contamination_hashset`` pins.

- A hit is a typed ``400 benchmark_series_embedded`` that carries the
  receipt field. Nothing is queued or vaulted.
- A clean repo is queued. The reply and the queue entry carry
  ``contamination: {hashset_version, result: "clean", suite: null,
  window_id: null}``.
- Unpinned (``""``, the shipped default) is today's request byte for byte:
  no fetch and no field (TB22).
- Pinned, but the artefact or the repo is unavailable, fails closed with
  ``503``.
- D38: a suite-target entry whose tree exceeds
  ``suite_target_max_repo_mb`` is refused as ``repo_too_large``.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json

import pytest

from . import helpers as h


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


@pytest.fixture(scope="module")
def art():
    return h.artefact(1)


def _armed(art, **kw):
    return h.lane_b_cfg(pin=h.pin_of(art), **kw)


def _not_queued(t) -> None:
    assert t.lane_b.get(h.HK) is None, "a refused fund left a lane B entry"
    assert h.HK not in h.queue_text(t.qpath), "a refused fund reached the queue file"
    assert not t.vault.has(h.HK), "a refused fund vaulted the miner's key"


def _seg_repo(tmp_path):
    return h.repo(tmp_path, "seg", **{"curve.py": h.py_list("CURVE", h.segment())})


# ── wiring ───────────────────────────────────────────────────────────────────


def test_production_intake_is_wired_with_the_hash_set_and_fetch():
    tree = ast.parse(h.FUNDING_MAIN.read_text(encoding="utf-8"))
    kws = {kw.arg for node in ast.walk(tree)
           if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "FundingIntake"
           for kw in node.keywords}
    assert {"lane_b_hashset", "fetch_lane_b_repo"} <= kws, (
        "cascade/funding/main.py must pass lane_b_hashset (the pinned artefact) and "
        "fetch_lane_b_repo (the submitted tree) to FundingIntake")
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Name))}
    assert "load_pinned_hashset" in names, (
        "main.py must load the hash set through contamination_scan.load_pinned_hashset "
        "(fail closed on a pin with no matching artefact)")


# ── invariant 1 / 10 pair ────────────────────────────────────────────────────


def test_inv1_lane_b_fund_unchanged_when_unarmed(tmp_path, art):
    """``contamination_hashset = ""`` (default): even a repo that embeds a
    test window is today's fund. 202 queued, the signed message is today's,
    the repo is never fetched, and neither the reply nor the queue file
    mentions contamination."""
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref

    seen = []
    fetch = h.Fetcher(_seg_repo(tmp_path))
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(), hashset=art, fetch=fetch, sig=True,
                 verify=lambda hk, m, s: seen.append(m) or s == "good")
    ts = str(int(t.clock()))
    status, body = t.intake.lane_b_fund(h.headers(**{"X-Timestamp": ts, "X-Signature": "good"}))
    assert (status, body) == (202, {"status": "queued", "lane": "b", "targets": ["energy"]})
    digest = hashlib.sha256(h.API_KEY.encode()).hexdigest()
    assert seen[-1] == canonical_fund_message(
        "lane-b-fund", h.HK, lane_b_signed_ref(("energy",), h.REF), ts, digest)
    assert fetch.calls == []
    assert h.entry_contamination(t.lane_b) is None
    assert "contamination" not in h.queue_text(t.qpath)


def test_inv1_lane_b_fund_armed_nonempty_lane_d_effect(tmp_path, art):
    """Pinned: the same request fetches the committed ref, and a clean repo
    is queued with the receipt field on the reply and on the entry. The
    signed message doesn't change, because the check adds no request field."""
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref
    from cascade.funding.queue import LaneBQueue

    seen = []
    fetch = h.Fetcher(h.repo(tmp_path))
    t = h.intake(tmp_path, cfg=_armed(art), hashset=art, fetch=fetch, sig=True,
                 verify=lambda hk, m, s: seen.append(m) or s == "good")
    ts = str(int(t.clock()))
    status, body = t.intake.lane_b_fund(h.headers(**{"X-Timestamp": ts, "X-Signature": "good"}))
    want = h.receipt(1, "clean")
    assert status == 202 and body["status"] == "queued", (status, body)
    assert body["contamination"] == want
    digest = hashlib.sha256(h.API_KEY.encode()).hexdigest()
    assert seen[-1] == canonical_fund_message(
        "lane-b-fund", h.HK, lane_b_signed_ref(("energy",), h.REF), ts, digest)
    assert fetch.calls == [h.REF]
    assert h.entry_contamination(t.lane_b) == want
    reloaded = LaneBQueue(t.qpath, clock=t.clock)
    assert getattr(reloaded.get(h.HK), "contamination", None) == want


# ── TB10 / TB11: refused at intake, before any compute ───────────────────────


def test_tb10_exact_window_refused_at_intake(tmp_path, art):
    t = h.intake(tmp_path, cfg=_armed(art), hashset=art, fetch=h.Fetcher(_seg_repo(tmp_path)))
    status, body = t.intake.lane_b_fund(h.headers())
    assert status == 400, (status, body)
    assert body["code"] == h.EMBEDDED
    assert body["contamination"] == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
    _not_queued(t)


def test_tb11_scaled_window_in_config_json_refused(tmp_path, art):
    vals = h.tb11(h.segment())
    r = h.repo_with(tmp_path, {"config.json": json.dumps({"name": "x", "shape": vals})})
    t = h.intake(tmp_path, cfg=_armed(art), hashset=art, fetch=h.Fetcher(r))
    status, body = t.intake.lane_b_fund(h.headers())
    first = h.expected_first(art, vals)
    assert (status, body["code"]) == (400, h.EMBEDDED), body
    assert body["contamination"] == h.receipt(1, "embedded", first["suite"], first["window_id"])
    _not_queued(t)


def test_refusal_on_a_suite_target_entry(tmp_path, art):
    """Suite-target entries (``suite:<s>``, WB18) are checked the same way."""
    cfg = _armed(art, suite_targets=True)
    assert "suite:gift" in cfg.targets
    t = h.intake(tmp_path, cfg=cfg, hashset=art, fetch=h.Fetcher(_seg_repo(tmp_path)))
    status, body = t.intake.lane_b_fund(h.headers(target="suite:gift"))
    assert (status, body["code"]) == (400, h.EMBEDDED), body
    _not_queued(t)


def test_hit_rule_is_read_from_the_config(tmp_path, art):
    """``[lane_b] contamination_min_hits`` decides: TB11 has no exact hit,
    so with ``normalised = 1000`` it is clean."""
    vals = h.tb11(h.segment())
    r = h.repo_with(tmp_path, {"config.json": json.dumps({"shape": vals})})
    cfg = _armed(art, min_hits={"exact": 1, "normalised": 1000})
    t = h.intake(tmp_path, cfg=cfg, hashset=art, fetch=h.Fetcher(r))
    status, body = t.intake.lane_b_fund(h.headers())
    assert status == 202, (status, body)
    assert body["contamination"] == h.receipt(1, "clean")


# ── TB12: the hash-set version the receipt records ───────────────────────────


def test_tb12_hash_set_version_change(tmp_path):
    for n in (1, 2):
        a = h.artefact(n)
        d = tmp_path / f"v{n}"
        d.mkdir()
        t = h.intake(d, cfg=h.lane_b_cfg(pin=h.pin_of(a)), hashset=a,
                     fetch=h.Fetcher(h.repo(d)))
        status, body = t.intake.lane_b_fund(h.headers())
        assert status == 202, (status, body)
        assert body["contamination"] == h.receipt(n, "clean")
        assert h.entry_contamination(t.lane_b) == h.receipt(n, "clean")


# ── fail closed while pinned ─────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["no_artefact", "other_artefact"])
def test_pinned_without_the_pinned_artefact_fails_closed(tmp_path, art, case):
    hashset = None if case == "no_artefact" else h.artefact(2)
    fetch = h.Fetcher(h.repo(tmp_path))
    t = h.intake(tmp_path, cfg=_armed(art), hashset=hashset, fetch=fetch)
    status, body = t.intake.lane_b_fund(h.headers())
    assert status == 503, (status, body)
    assert body.get("code") != h.EMBEDDED
    _not_queued(t)


@pytest.mark.parametrize("case", ["none", "raises", "no_fetcher"])
def test_unfetchable_repo_fails_closed(tmp_path, art, case):
    fetch = {"none": h.Fetcher(None), "raises": h.Fetcher(None, raises=OSError("hub down")),
             "no_fetcher": None}[case]
    t = h.intake(tmp_path, cfg=_armed(art), hashset=art, fetch=fetch)
    status, body = t.intake.lane_b_fund(h.headers())
    assert status == 503, (status, body)
    _not_queued(t)


def test_identity_gate_runs_before_the_fetch(tmp_path, art):
    """An unauthenticated or unrevealed request never makes the intake
    fetch or scan a repo."""
    fetch = h.Fetcher(_seg_repo(tmp_path))
    (tmp_path / "sig").mkdir()
    t = h.intake(tmp_path / "sig", cfg=_armed(art), hashset=art, fetch=fetch, sig=True,
                 verify=lambda hk, m, s: False)
    status, body = t.intake.lane_b_fund(
        h.headers(**{"X-Timestamp": str(int(t.clock())), "X-Signature": "bad"}))
    assert (status, body["code"]) == (400, "bad_signature")
    (tmp_path / "rev").mkdir()
    t2 = h.intake(tmp_path / "rev", cfg=_armed(art), hashset=art, fetch=fetch, reveal=None)
    status, body = t2.intake.lane_b_fund(h.headers())
    assert (status, body["code"]) == (403, "not_revealed")
    assert fetch.calls == []


# ── D38: suite_target_max_repo_mb ────────────────────────────────────────────


def _sized_repo(tmp_path, name, total):
    r = h.repo(tmp_path, name)
    h.pad_to(r, total)
    return r


def test_d38_suite_target_repo_over_the_cap_is_refused(tmp_path, art):
    cfg = _armed(art, suite_targets=True, cap_mb=1)
    over = _sized_repo(tmp_path, "over", h.MIB + 1)
    t = h.intake(tmp_path, cfg=cfg, hashset=art, fetch=h.Fetcher(over))
    status, body = t.intake.lane_b_fund(h.headers(target="suite:boom"))
    assert status in (400, 413), (status, body)
    assert body["code"] == h.TOO_LARGE
    _not_queued(t)


def test_d38_cap_boundary_and_scope(tmp_path, art):
    """Exactly ``cap`` MiB is accepted. The cap is for suite-target entries
    only (D38: "for lane B suite-target entries"), so a domain entry with
    the same over-cap tree is queued."""
    cfg = _armed(art, suite_targets=True, cap_mb=1)
    at = _sized_repo(tmp_path, "at", h.MIB)
    (tmp_path / "a").mkdir()
    t = h.intake(tmp_path / "a", cfg=cfg, hashset=art, fetch=h.Fetcher(at))
    status, body = t.intake.lane_b_fund(h.headers(target="suite:boom"))
    assert status == 202, (status, body)
    assert body["contamination"] == h.receipt(1, "clean")
    over = _sized_repo(tmp_path, "over", h.MIB + 1)
    (tmp_path / "b").mkdir()
    t = h.intake(tmp_path / "b", cfg=cfg, hashset=art, fetch=h.Fetcher(over))
    status, body = t.intake.lane_b_fund(h.headers(target="energy"))
    assert status == 202, (status, body)


def test_d38_cap_is_read_from_the_config(tmp_path, art):
    over = _sized_repo(tmp_path, "over", h.MIB + 1)
    cfg = dataclasses.replace(_armed(art, suite_targets=True, cap_mb=1),
                              suite_target_max_repo_mb=2)
    t = h.intake(tmp_path, cfg=cfg, hashset=art, fetch=h.Fetcher(over))
    status, body = t.intake.lane_b_fund(h.headers(target="suite:boom"))
    assert status == 202, (status, body)
