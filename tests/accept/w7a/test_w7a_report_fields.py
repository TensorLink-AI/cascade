"""W7a: what a report produced by `settle_lane_d` carries (plan rev6 §3.0
W7a row, §1.5 "(rev3) The report also carries", §14.3 items 1-3, B7, B12).

End to end on the synthetic chain (no torch): W1.5's box-side pipeline on two
snapshots, the first clear at BOUNDARY_1 and the confirmation that crowns at
BOUNDARY_2. The pools carry the W5a `feeds` block, a per-series `feed` and
the B12 `first_seen_roots`, plus one §1.3 trap window. A second run uses the
W1.5 pool format (none of those keys). On main the tests fail at the import
of `validate_report` or at an `assert` on a missing field.
"""

from __future__ import annotations

import pytest

from .helpers import (
    BOUNDARY_1,
    BOUNDARY_2,
    CHALLENGER_UID,
    FEEDS,
    HOUR,
    PARAMS_FIELDS,
    ROOTS,
    canonical_rows_sha256,
    coldkey,
    finite,
    two_settlements,
    window_id,
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7a_fields")


def _built(scratch, *, rich=True):
    from cascade.validator.lane_d_report import validate_report

    built = two_settlements(scratch / ("rich" if rich else "legacy"), rich=rich)
    for s in (built["s1"], built["s2"]):
        assert s.report is not None, "an armed settlement produced no lane D report"
        validate_report(s.report)
    return built


def _expected_params(cfg) -> dict:
    """§14.3 item 2: every value a satellite needs, as the overlay states it."""
    return {
        "share": float(cfg.share), "takes_from": str(cfg.takes_from),
        "remainder": str(cfg.remainder), "horizon": int(cfg.horizon),
        "horizons": [int(cfg.horizon)], "contexts": [int(c) for c in cfg.contexts],
        "forward_margin_hours": float(cfg.forward_margin_hours),
        "max_mirror_lag_hours": float(cfg.max_mirror_lag_hours),
        "min_windows": int(cfg.min_windows), "min_clusters": int(cfg.min_clusters),
        "d_margin": float(cfg.d_margin), "alpha": float(cfg.alpha),
        "bootstrap_B": int(cfg.bootstrap_B), "king_decay": float(cfg.king_decay),
        "reward_prior_kings": int(cfg.reward_prior_kings),
        "confirm_on_king_change": str(cfg.confirm_on_king_change),
        "max_entries_per_settlement": int(cfg.max_entries_per_settlement),
        "max_judged_per_coldkey_per_settlement": int(cfg.max_judged_per_coldkey_per_settlement),
        "scoring_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "gate_suites": [str(s) for s in cfg.gate_suites],
        "tau_public": float(cfg.tau_public),
        "lowfreq_freqs": [str(f) for f in cfg.lowfreq_freqs],
    }


def test_e2e_settle_reports_validate_and_state_quantile_head(scratch):
    built = _built(scratch)
    for s, boundary in ((built["s1"], BOUNDARY_1), (built["s2"], BOUNDARY_2)):
        assert s.report["schema"] == "lane-d-report/1"
        assert s.report["boundary"] == boundary
        assert s.report["scoring_method"] == "quantile_head"


def test_e2e_legacy_pool_format_report_validates(scratch):
    built = _built(scratch, rich=False)
    r = built["s1"].report
    assert r["pool"]["first_seen_roots"] == []
    assert r["cohorts"] and all(c["first_seen_roots"] == [] for c in r["cohorts"])
    for w in r["windows"]:
        assert w["feed"] == w["source"], "a series without `feed` reports through its source"
        assert w["mirror_lag_hours"] is None, "an unlisted feed's lag is unmeasured (null)"


def test_e2e_params_carry_every_satellite_value(scratch):
    built = _built(scratch)
    want = _expected_params(built["cfg"])
    assert set(want) == set(PARAMS_FIELDS)
    for s in (built["s1"], built["s2"]):
        params = s.report["params"]
        for k, v in want.items():
            assert params[k] == v, f"params.{k} = {params.get(k)!r}, want {v!r}"


def test_e2e_rows_sha256_is_recomputable_from_rows(scratch):
    built = _built(scratch)
    for s in (built["s1"], built["s2"]):
        rows = s.report["rows"]
        assert rows
        assert s.report["rows_sha256"] == canonical_rows_sha256(rows)


def test_e2e_windows_carry_both_clocks_and_mirror_lag(scratch):
    built = _built(scratch)
    chain, cfg = built["chain"], built["cfg"]
    bound = max(chain["house"]["reveal_ts"], built["entry"]["reveal_ts"]) \
        + HOUR * cfg.forward_margin_hours
    for snap, s in ((1, built["s1"]), (2, built["s2"])):
        by_id = {window_id(x["series_id"], x["target_ts"]): x for x in chain["series"][snap]}
        windows = s.report["windows"]
        assert {w["window_id"] for w in windows} == {r["window_id"] for r in s.report["rows"]}
        assert len(windows) == 48
        for w in windows:
            src = by_id[w["window_id"]]
            assert w["min_target_ts"] == min(src["target_ts"])
            assert w["min_first_seen_ts"] == min(src["first_seen_ts"])
            assert w["min_target_ts"] > bound and w["min_first_seen_ts"] > bound
            feed = src.get("feed", src["source"])
            assert w["feed"] == feed
            want = FEEDS.get(feed, {}).get("mirror_lag_hours")
            assert w["mirror_lag_hours"] == want, (w["window_id"], w["mirror_lag_hours"])
        lags = {w["feed"]: w["mirror_lag_hours"] for w in windows}
        assert lags["feed-a"] == 1.5 and lags["src1"] is None and lags["src3"] is None


def test_e2e_first_seen_roots_and_refusal_tally_per_frozen_set(scratch):
    built = _built(scratch)
    for snap, s in ((1, built["s1"]), (2, built["s2"])):
        r = s.report
        assert r["pool"]["first_seen_roots"] == ROOTS[snap]
        assert len(r["cohorts"]) == 1
        cohort = r["cohorts"][0]
        assert cohort["first_seen_roots"] == ROOTS[snap]
        assert cohort["refused"] == {"before_margin": 1}, cohort["refused"]
        assert all(not w["window_id"].startswith("src2-trap") for w in r["windows"])


def test_e2e_judged_set_skips_and_coldkey(scratch):
    built = _built(scratch)
    dg = built["entry"]["digest"]
    for s in (built["s1"], built["s2"]):
        r = s.report
        assert r["judged"] == [dg]
        assert r["skipped"] == []
        ent = {e["digest"]: e for e in r["entries"]}[dg]
        assert ent["coldkey"] == coldkey()
        assert ent["hotkey"] == built["entry"]["hotkey"]
        assert ent["context"] == built["entry"]["context"]
    assert {e["digest"]: e for e in built["s1"].report["entries"]}[dg]["status"] \
        == "confirm_pending"
    e2 = {e["digest"]: e for e in built["s2"].report["entries"]}[dg]
    assert (e2["status"], e2["reason"]) == ("crowned", "crowned")


def test_e2e_confirmation_block(scratch):
    built = _built(scratch)
    house = built["chain"]["house"]["digest"]
    (v1,) = built["s1"].report["verdicts"]
    (v2,) = built["s2"].report["verdicts"]
    assert v1["outcome"] == "confirm_pending" and v1["cleared"] is True
    assert v1["confirmation"] == {"stage": "first", "rule": "restart",
                                  "first_set_sha256": None, "first_king_digest": None}
    first_set = built["s1"].report["cohorts"][0]["set_sha256"]
    assert v2["outcome"] == "crowned"
    assert v2["confirmation"] == {"stage": "confirm", "rule": "restart",
                                  "first_set_sha256": first_set, "first_king_digest": house}
    ids1 = set(built["s1"].report["cohorts"][0]["window_ids"])
    ids2 = set(built["s2"].report["cohorts"][0]["window_ids"])
    assert ids1 and ids2 and not ids1 & ids2, "the confirmation set must be disjoint"


def test_e2e_lowfreq_result(scratch):
    built = _built(scratch)
    cfg = built["cfg"]
    (v1,) = built["s1"].report["verdicts"]
    (v2,) = built["s2"].report["verdicts"]
    freqs = [str(f) for f in cfg.lowfreq_freqs]
    assert v1["lowfreq"] == {"result": "not_candidate", "freqs": freqs, "n_windows": 0,
                             "tau_public": float(cfg.tau_public)}
    # The crowning candidate: the fixture is hourly only, so no low-frequency window.
    assert v2["lowfreq"] == {"result": "no_windows", "freqs": freqs, "n_windows": 0,
                             "tau_public": float(cfg.tau_public)}


def test_e2e_court_slots_carry_burn_reason(scratch):
    built = _built(scratch)
    (slot1,) = built["s1"].report["pay"]["court"]
    assert slot1["uid"] is None and slot1["burn_reason"] == "house_king"
    assert slot1["digest"] == built["chain"]["house"]["digest"]
    (slot2,) = built["s2"].report["pay"]["court"]
    assert slot2["uid"] == CHALLENGER_UID and slot2["burn_reason"] is None
    assert built["s2"].pay["court_uids"] == [CHALLENGER_UID]


def test_e2e_rows_values_are_finite(scratch):
    built = _built(scratch)
    for s in (built["s1"], built["s2"]):
        for row in s.report["rows"]:
            assert finite(row["crps"]) and finite(row["mase"])
