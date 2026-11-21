"""R4: stake-weighted upgrade tracker (plan §3.0 R4, §5 row 11 S6, X6a).

``python -P -m cascade.laned.upgrade_tracker --metagraph F --notes G --cfg X
[--threshold T]`` reads a fixture metagraph (hotkey -> stake, validator_permit)
and a fixture notes map (hotkey -> readiness note text), and reports per
hotkey the stake, the release (primary segment), the ``cfg-`` segment and
whether it is on the target overlay; then the exact stake-weighted share on
the target among permit holders, and the laggards by stake. Exit 0 iff the
share >= threshold, 1 otherwise, 2 on bad input. Contract: README.md.
"""

from __future__ import annotations

import importlib.util

import pytest

from .helpers import (
    LARGEST,
    NO_PERMIT,
    OLD_CFG,
    RELEASE,
    S6_STAKES,
    TARGET,
    ZERO_STAKE,
    by_hotkey,
    laggard_names,
    netconfig,
    note_of,
    off_note,
    on_note,
    s6_metagraph,
    s6_notes,
    track,
    track_bad,
)

ORDER = ["5Bvalidator2", "5Cvalidator3", "5Dvalidator4", "5Evalidator5", "5Fvalidator6"]


# ── the module ───────────────────────────────────────────────────────────────


def test_upgrade_tracker_module_exists():
    try:
        spec = importlib.util.find_spec("cascade.laned.upgrade_tracker")
    except ModuleNotFoundError as e:
        raise AssertionError(f"cascade.laned.upgrade_tracker is missing (R4 PATHS): {e}") from None
    assert spec is not None, "cascade.laned.upgrade_tracker is missing (R4 PATHS)"


# ── exact-% golden tests: the S6 shape ───────────────────────────────────────


def test_s6_all_on_target_is_100_percent_exit_0(tmp_path):
    out = track(tmp_path, s6_metagraph(), s6_notes(), expect_rc=0)
    assert out["target"] == TARGET
    assert out["threshold"] == 0.6667
    assert out["signed_stake"] == 1_000_000
    assert out["total_stake"] == 1_000_000
    assert out["ratio"] == "1"
    assert out["pct"] == 100.0
    assert out["meets_threshold"] is True
    assert out["laggards"] == []
    assert out["signed"] == sorted(S6_STAKES)
    assert out["eligible"] == sorted(S6_STAKES)


def test_s6_largest_holder_off_is_51_percent_below_two_thirds(tmp_path):
    """Every other validator on target is still only 51%: the largest holder
    (5E2LP6 at ~49%) must be on for the S6 >= 2/3 proof."""
    out = track(tmp_path, s6_metagraph(), s6_notes(off=(LARGEST,)), expect_rc=1)
    assert out["signed_stake"] == 510_000
    assert out["total_stake"] == 1_000_000
    assert out["ratio"] == "51/100"
    assert out["pct"] == 51.0
    assert out["meets_threshold"] is False
    assert laggard_names(out) == [LARGEST]
    lag = out["laggards"][0]
    assert lag["stake"] == 490_000
    assert lag["note_state"] == "ok"
    assert lag["release"] == RELEASE
    assert lag["cfg"] is None


def test_s6_largest_plus_second_is_67_percent_exit_0(tmp_path):
    off = ("5Cvalidator3", "5Dvalidator4", "5Evalidator5", "5Fvalidator6")
    out = track(tmp_path, s6_metagraph(), s6_notes(off=off), expect_rc=0)
    assert out["ratio"] == "67/100"
    assert out["pct"] == 67.0
    assert out["meets_threshold"] is True
    assert laggard_names(out) == list(off)
    assert out["signed"] == sorted([LARGEST, "5Bvalidator2"])


def test_s6_largest_third_and_sixth_is_65_percent_exit_1(tmp_path):
    off = ("5Bvalidator2", "5Dvalidator4", "5Evalidator5")
    out = track(tmp_path, s6_metagraph(), s6_notes(off=off), expect_rc=1)
    assert out["signed_stake"] == 650_000
    assert out["ratio"] == "13/20"
    assert out["pct"] == 65.0
    assert out["meets_threshold"] is False
    assert laggard_names(out) == list(off)


def test_s6_two_smallest_off_is_89_percent_laggards_by_stake_desc(tmp_path):
    out = track(tmp_path, s6_metagraph(), s6_notes(off=("5Fvalidator6", "5Evalidator5")),
                expect_rc=0)
    assert out["ratio"] == "89/100"
    assert out["pct"] == 89.0
    assert laggard_names(out) == ["5Evalidator5", "5Fvalidator6"]
    assert [r["stake"] for r in out["laggards"]] == [70_000, 40_000]


def test_s6_per_hotkey_rows_sorted_by_stake_desc_with_uncounted(tmp_path):
    out = track(tmp_path, s6_metagraph(), s6_notes(off=(LARGEST,)), expect_rc=1)
    names = [r["hotkey"] for r in out["hotkeys"]]
    assert names == [LARGEST, NO_PERMIT, *ORDER, ZERO_STAKE]
    rows = by_hotkey(out)
    assert rows[LARGEST] == {"hotkey": LARGEST, "stake": 490_000, "validator_permit": True,
                             "counted": True, "note_state": "ok", "release": RELEASE,
                             "cfg": None, "on_target": False}
    assert rows["5Bvalidator2"] == {"hotkey": "5Bvalidator2", "stake": 180_000,
                                    "validator_permit": True, "counted": True,
                                    "note_state": "ok", "release": RELEASE, "cfg": TARGET,
                                    "on_target": True}
    assert rows[NO_PERMIT]["counted"] is False
    assert rows[NO_PERMIT]["validator_permit"] is False
    assert rows[NO_PERMIT]["on_target"] is True
    assert rows[ZERO_STAKE]["counted"] is False
    assert rows[ZERO_STAKE]["validator_permit"] is True


def test_uncounted_hotkeys_never_change_totals_or_laggards(tmp_path):
    notes = s6_notes()
    notes[NO_PERMIT] = off_note()
    notes[ZERO_STAKE] = "garbage"
    out = track(tmp_path, s6_metagraph(), notes, expect_rc=0)
    assert out["ratio"] == "1"
    assert out["total_stake"] == 1_000_000
    assert out["laggards"] == []
    assert NO_PERMIT not in out["eligible"]
    assert ZERO_STAKE not in out["eligible"]


def test_notes_for_hotkeys_outside_metagraph_are_ignored(tmp_path):
    notes = s6_notes(off=(LARGEST,))
    notes["5Zstranger"] = on_note()
    out = track(tmp_path, s6_metagraph(), notes, expect_rc=1)
    assert out["ratio"] == "51/100"
    assert "5Zstranger" not in by_hotkey(out)
    assert "5Zstranger" not in out["signed"]


# ── exact rational, fixed rounding, threshold ────────────────────────────────


def test_exact_two_thirds_rounds_to_66_6667_and_fails_default(tmp_path):
    mg = {"a": {"stake": 2, "validator_permit": True}, "b": {"stake": 1, "validator_permit": True}}
    notes = {"a": on_note(), "b": off_note()}
    out = track(tmp_path, mg, notes, expect_rc=1)
    assert out["ratio"] == "2/3"
    assert out["pct"] == 66.6667
    assert out["meets_threshold"] is False


def test_threshold_flag_compares_exactly(tmp_path):
    mg = {"a": {"stake": 2, "validator_permit": True}, "b": {"stake": 1, "validator_permit": True}}
    notes = {"a": on_note(), "b": off_note()}
    out = track(tmp_path, mg, notes, TARGET, "--threshold", "0.6666", expect_rc=0)
    assert out["threshold"] == 0.6666
    assert out["meets_threshold"] is True
    track(tmp_path, mg, notes, TARGET, "--threshold", "0.66667", expect_rc=1)


def test_threshold_equal_to_ratio_meets(tmp_path):
    mg = {"a": {"stake": 51, "validator_permit": True}, "b": {"stake": 49, "validator_permit": True}}
    notes = {"a": on_note(), "b": off_note()}
    out = track(tmp_path, mg, notes, TARGET, "--threshold", "0.51", expect_rc=0)
    assert out["ratio"] == "51/100"
    assert out["meets_threshold"] is True


def test_decimal_stakes_are_exact_as_written(tmp_path):
    """0.1 and 0.2 are taken as written (1/10, 2/10), not as binary floats."""
    mg = {"a": {"stake": 0.1, "validator_permit": True},
          "b": {"stake": 0.2, "validator_permit": True}}
    out = track(tmp_path, mg, {"a": on_note(), "b": off_note()}, expect_rc=1)
    assert out["ratio"] == "1/3"
    assert out["pct"] == 33.3333
    assert by_hotkey(out)["a"]["stake"] == 0.1

    mg["c"] = {"stake": 0.7, "validator_permit": True}
    out = track(tmp_path, mg, {"a": on_note(), "b": on_note(), "c": off_note()},
                TARGET, "--threshold", "0.3", expect_rc=0)
    assert out["ratio"] == "3/10"
    assert out["pct"] == 30.0


@pytest.mark.parametrize(("signed", "want_pct"), [
    (1, 0.0),         # 0.00005 -> half to even -> 0.0000
    (3, 0.0002),      # 0.00015 -> half to even -> 0.0002
    (1_999_999, 100.0),  # 99.99995 -> half to even -> 100.0000 (ratio stays exact)
])
def test_pct_rounds_half_even_at_four_decimals(tmp_path, signed, want_pct):
    mg = {"on": {"stake": signed, "validator_permit": True},
          "off": {"stake": 2_000_000 - signed, "validator_permit": True}}
    out = track(tmp_path, mg, {"on": on_note(), "off": off_note()}, TARGET,
                "--threshold", "0.5", expect_rc=0 if signed > 1_000_000 else 1)
    assert out["pct"] == want_pct
    assert out["ratio"] != "1"


# ── notes: release, cfg, malformed, missing ──────────────────────────────────


def test_release_is_primary_feature_and_cfg_is_reported(tmp_path):
    mg = {"a": {"stake": 3, "validator_permit": True}, "b": {"stake": 1, "validator_permit": True}}
    notes = {"a": on_note(release="rolling-era-king-r9"), "b": on_note(cfg=OLD_CFG)}
    out = track(tmp_path, mg, notes, expect_rc=0)
    rows = by_hotkey(out)
    assert rows["a"]["release"] == "rolling-era-king-r9"
    assert rows["a"]["cfg"] == TARGET
    assert rows["a"]["on_target"] is True
    assert rows["b"]["release"] == RELEASE
    assert rows["b"]["cfg"] == OLD_CFG
    assert rows["b"]["on_target"] is False
    assert out["laggards"] == [{"hotkey": "b", "stake": 1, "note_state": "ok",
                                "release": RELEASE, "cfg": OLD_CFG}]
    assert out["ratio"] == "3/4"


def test_locked_cfg_segment_counts_as_on_target(tmp_path):
    mg = {"a": {"stake": 1, "validator_permit": True}}
    notes = {"a": note_of(f"{RELEASE}:9374400:9378000", f"{TARGET}:9500000:9500900")}
    out = track(tmp_path, mg, notes, expect_rc=0)
    assert by_hotkey(out)["a"]["on_target"] is True
    assert out["ratio"] == "1"


MALFORMED = [
    "garbage",
    note_of(f"{RELEASE}:5:3", f"{TARGET}:0:0"),                    # activation <= lock
    note_of(f"{RELEASE}:0:7", f"{TARGET}:0:0"),                    # activation without lock
    note_of(f"{TARGET}:0:0", f"{TARGET}:0:0"),                     # duplicate feature
    note_of(f"{RELEASE}:9374400:9378000", f"{TARGET}:0"),          # truncated segment
    "cascade-ready:2:" + f"{RELEASE}:0:0:{TARGET}:0:0",            # future version
    "hippius.example/gen@sha256:" + "0123456789abcdef" * 4,        # a generator pointer
]


@pytest.mark.parametrize("note", MALFORMED)
def test_malformed_note_is_not_on_target_and_named(tmp_path, note):
    mg = {LARGEST: {"stake": 49, "validator_permit": True},
          "rest": {"stake": 51, "validator_permit": True}}
    out = track(tmp_path, mg, {LARGEST: note, "rest": on_note()}, expect_rc=1)
    row = by_hotkey(out)[LARGEST]
    assert row["note_state"] == "malformed"
    assert row["on_target"] is False
    assert row["release"] is None
    assert row["cfg"] is None
    assert out["laggards"] == [{"hotkey": LARGEST, "stake": 49, "note_state": "malformed",
                                "release": None, "cfg": None}]
    assert out["ratio"] == "51/100"


@pytest.mark.parametrize("how", ["absent", "null"])
def test_missing_note_is_not_on_target_and_named(tmp_path, how):
    mg = {LARGEST: {"stake": 49, "validator_permit": True},
          "rest": {"stake": 51, "validator_permit": True}}
    notes = {"rest": on_note()}
    if how == "null":
        notes[LARGEST] = None
    out = track(tmp_path, mg, notes, expect_rc=1)
    row = by_hotkey(out)[LARGEST]
    assert row["note_state"] == "missing"
    assert row["on_target"] is False
    assert row["release"] is None
    assert row["cfg"] is None
    assert laggard_names(out) == [LARGEST]


def test_laggard_ties_break_by_hotkey_ascending(tmp_path):
    mg = {"z": {"stake": 5, "validator_permit": True}, "y": {"stake": 5, "validator_permit": True},
          "x": {"stake": 90, "validator_permit": True}}
    out = track(tmp_path, mg, {"x": on_note(), "y": off_note(), "z": "garbage"}, expect_rc=0)
    assert laggard_names(out) == ["y", "z"]
    assert [r["hotkey"] for r in out["hotkeys"]] == ["x", "y", "z"]


# ── --cfg forms ──────────────────────────────────────────────────────────────


def test_cfg_bare_eight_hex_is_prefixed(tmp_path):
    out = track(tmp_path, s6_metagraph(), s6_notes(), "1a2b3c4d", expect_rc=0)
    assert out["target"] == TARGET
    assert out["ratio"] == "1"


def test_cfg_full_body_digest_maps_through_netconfig(tmp_path):
    nc = netconfig()
    body = nc.next_overlay({}, 9_500_000, None, network={"netuid": 259, "chain": "test"})
    digest = nc.body_digest(body)
    want = nc.cfg_feature_name(body)
    assert len(digest) == 64 and want == "cfg-" + digest[:8]
    mg = {"a": {"stake": 2, "validator_permit": True}, "b": {"stake": 1, "validator_permit": True}}
    notes = {"a": on_note(cfg=want), "b": on_note()}
    out = track(tmp_path, mg, notes, digest, expect_rc=1)
    assert out["target"] == want
    assert out["signed"] == ["a"]
    assert out["ratio"] == "2/3"


@pytest.mark.parametrize("bad", ["cfg-1A2B3C4D", "cfg-1a2b3c", "1a2b3c4d5", "cfg-zzzzzzzz",
                                 "", "cfg-", "v2b-69630f92"])
def test_bad_cfg_exits_2(tmp_path, bad):
    track_bad(tmp_path, s6_metagraph(), s6_notes(), bad)


# ── bad input: exit 2 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("threshold", ["0", "-0.1", "1.5", "abc", "nan"])
def test_bad_threshold_exits_2(tmp_path, threshold):
    track_bad(tmp_path, s6_metagraph(), s6_notes(), TARGET, "--threshold", threshold)


def test_threshold_one_is_allowed(tmp_path):
    track(tmp_path, s6_metagraph(), s6_notes(), TARGET, "--threshold", "1", expect_rc=0)
    track(tmp_path, s6_metagraph(), s6_notes(off=("5Fvalidator6",)), TARGET,
          "--threshold", "1", expect_rc=1)


def test_missing_metagraph_file_exits_2(tmp_path):
    track_bad(tmp_path, s6_metagraph(), s6_notes(), mg_path=str(tmp_path / "nope.json"))


@pytest.mark.parametrize("metagraph", [
    "{not json",
    '[{"hotkey": "a", "stake": 1, "validator_permit": true}]',
    '{"a": {"validator_permit": true}}',
    '{"a": {"stake": 1}}',
    '{"a": {"stake": -1, "validator_permit": true}}',
    '{"a": {"stake": "5", "validator_permit": true}}',
    '{"a": {"stake": true, "validator_permit": true}}',
    '{"a": {"stake": 1, "validator_permit": "yes"}}',
    '{"a": 5}',
])
def test_bad_metagraph_exits_2(tmp_path, metagraph):
    track_bad(tmp_path, metagraph, {"a": on_note()})


@pytest.mark.parametrize("notes", ["{not json", '["a"]', '{"a": 5}', '{"a": ["x"]}'])
def test_bad_notes_exits_2(tmp_path, notes):
    track_bad(tmp_path, {"a": {"stake": 1, "validator_permit": True}}, notes)


@pytest.mark.parametrize("metagraph", [
    {"a": {"stake": 5, "validator_permit": False}},
    {"a": {"stake": 0, "validator_permit": True}},
    {},
])
def test_no_eligible_stake_exits_2(tmp_path, metagraph):
    track_bad(tmp_path, metagraph, {"a": on_note()})
