"""R2: note-budget decoder (plan §3.0 R2, §5 row 3 S0b, T22, §13, §14.1).

``python -m cascade.laned.notes`` decodes a readiness note into segments with
their pending/decided state, simulates adding a ``cfg-<8 hex>`` overlay
segment, and reports which segments the 128-byte rule trims. Trim rule
(``cascade/shared/activation.py`` ``own_signal_payload``): the primary
segment is never trimmed; feature segments go newest (rightmost) first; the
``cfg-`` segment rides last and goes only once every other extra is gone.
``--worst-case`` judges pending segments at their longest, as the validator's
startup gate ``v2b_note_problem`` does. Contract: README.md.
"""

from __future__ import annotations

import importlib.util

import pytest

from .helpers import (
    activation,
    all_votes_open_config,
    builder_note_and_trim,
    crashloop_config,
    load_fixture,
    note_of,
    run_json,
    run_notes,
    segment,
    shipped_config,
    simulate,
)

CRASH = "crashloop_20261113.json"


def _names(rows):
    return [r["feature"] for r in rows]


def _crash_simulate(fx, note=None, **kw):
    return simulate(note or fx["note_text"], fx["head_block"], fx["cfg"]["feature"],
                    worst_case=True, **kw)


# ── the module ───────────────────────────────────────────────────────────────


def test_notes_module_exists():
    try:
        spec = importlib.util.find_spec("cascade.laned.notes")
    except ModuleNotFoundError as e:
        raise AssertionError(f"cascade.laned.notes is missing (R2 PATHS): {e}") from None
    assert spec is not None, "cascade.laned.notes is missing (R2 PATHS)"


# ── decode ───────────────────────────────────────────────────────────────────


def test_decode_crashloop_note_from_hex_segments_and_states():
    fx = load_fixture(CRASH)
    assert bytes.fromhex(fx["note_hex"]).decode("utf-8") == fx["note_text"]
    out = run_json("decode", "--hex", fx["note_hex"], "--head-block", str(fx["head_block"]),
                   expect_rc=0)
    assert out["note"] == fx["note_text"]
    assert out["bytes"] == len(fx["note_text"].encode("utf-8")) == 97
    assert out["max_bytes"] == 128
    assert out["head_block"] == fx["head_block"]
    assert out["segments"] == [
        {"feature": "rolling-era-king", "lock_block": 9374400, "activation_block": 9378000,
         "role": "primary", "state": "decided"},
        {"feature": "prelaunch-95-r2", "lock_block": 9486000, "activation_block": 9486900,
         "role": "extra", "state": "decided"},
        {"feature": "v2b-69630f92", "lock_block": 0, "activation_block": 0,
         "role": "extra", "state": "pending"},
    ]


def test_decode_accepts_0x_hex_upper_hex_and_text_identically():
    fx = load_fixture(CRASH)
    head = str(fx["head_block"])
    a = run_json("decode", "--hex", "0x" + fx["note_hex"], "--head-block", head, expect_rc=0)
    b = run_json("decode", "--text", fx["note_text"], "--head-block", head, expect_rc=0)
    c = run_json("decode", "--hex", fx["note_hex"].upper(), "--head-block", head, expect_rc=0)
    assert a == b == c


@pytest.mark.parametrize(("lock", "act", "head", "state"), [
    (0, 0, 9_500_000, "pending"),                  # unlocked: the vote is still open
    (9_490_000, 9_493_200, 9_493_199, "pending"),  # locked, activation still ahead
    (9_490_000, 9_493_200, 9_493_200, "decided"),  # activation block reached
    (9_490_000, 9_493_200, 9_600_000, "decided"),
])
def test_decode_state_rule(lock, act, head, state):
    note = note_of(segment("rolling-era-king"), segment("v2b-69630f92", lock, act))
    out = run_json("decode", "--text", note, "--head-block", str(head), expect_rc=0)
    assert out["segments"][1]["state"] == state
    assert out["segments"][0]["state"] == "pending"


def test_decode_marks_an_existing_cfg_segment():
    note = note_of(segment("rolling-era-king"), segment("lad-556b1b98"),
                   segment("cfg-0a1b2c3d", 9_500_000, 9_501_200))
    out = run_json("decode", "--text", note, "--head-block", "9500100", expect_rc=0)
    assert [(s["feature"], s["role"], s["state"]) for s in out["segments"]] == [
        ("rolling-era-king", "primary", "pending"), ("lad-556b1b98", "extra", "pending"),
        ("cfg-0a1b2c3d", "cfg", "pending")]


@pytest.mark.parametrize("bad", [
    "not-a-note",
    "cascade-ready:2:rolling-era-king:0:0",          # unknown version
    "cascade-ready:1:rolling-era-king:0",            # truncated segment
    "cascade-ready:1:rolling-era-king:5:3",          # activation before lock
    "cascade-ready:1:rolling-era-king:0:5",          # activation without lock
    "cascade-ready:1:a:0:0:a:0:0",                   # duplicated feature
])
def test_decode_refuses_malformed_notes_with_exit_2(bad):
    proc = run_notes("decode", "--text", bad, "--head-block", "9500000")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr.strip()


def test_decode_refuses_non_hex_input_with_exit_2():
    proc = run_notes("decode", "--hex", "zz12", "--head-block", "9500000")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr.strip()


# ── simulate: the 2026-11-13 crash-loop (§13) ────────────────────────────────


def test_crashloop_fixture_reproduces_the_refusal():
    fx = load_fixture(CRASH)
    out = _crash_simulate(fx, expect_rc=1)
    assert out["note"] == fx["expected_note"]
    assert f"(note without it: '{out['note']}')" in fx["logged_refusal"]
    assert out["bytes"] == 109 and out["fits"] is True
    assert out["worst_case"] is True
    assert out["cfg"] == fx["cfg"]
    assert out["trimmed"] == [{"feature": "v2b-69630f92", "role": "extra", "state": "pending"}]
    assert out["pending_trimmed"] == fx["expected_trimmed"]
    assert out["refused"] is True
    assert out["input"]["note"] == fx["note_text"]
    assert _names(out["input"]["segments"]) == list(fx["records"])


def test_crashloop_refusal_matches_the_live_code(tmp_path):
    """The real validator gate (v2b_note_problem) on the same inputs produces
    the logged refusal, and the decoder's result is the note inside it."""
    act = activation()
    fx = load_fixture(CRASH)
    cfg = crashloop_config(tmp_path)
    R = act.ActivationRecord
    roll, pre = fx["records"]["rolling-era-king"], fx["records"]["prelaunch-95-r2"]
    why = act.v2b_note_problem(
        cfg, R(feature="rolling-era-king", lock_block=roll[0], activation_block=roll[1],
               source="tally"),
        None, None,
        R(feature="prelaunch-95-r2", lock_block=pre[0], activation_block=pre[1],
          source="tally"),
        head_block=fx["head_block"])
    assert why.startswith(fx["logged_refusal"]), why
    out = _crash_simulate(fx, expect_rc=1)
    assert f"(note without it: '{out['note']}')" in why


def test_pr457_typed_v2b_note_fits_with_cfg():
    fx = load_fixture(CRASH)
    out = _crash_simulate(fx, fx["fixed_by_pr457_note_text"], expect_rc=0)
    assert out["trimmed"] == [] and out["pending_trimmed"] == [] and out["refused"] is False
    assert out["note"] == fx["fixed_by_pr457_note_text"] + ":cfg-ffffffff:9999999:9999999"


def test_without_worst_case_the_crashloop_note_fits():
    """Judged at its current (unlocked) blocks the v2b vote fits; the refusal
    is the worst-case rule, so the flag must matter."""
    fx = load_fixture(CRASH)
    out = simulate(fx["note_text"], fx["head_block"], "cfg-ffffffff", 9_999_999, 9_999_999,
                   expect_rc=0)
    assert out["worst_case"] is False and out["trimmed"] == []
    assert out["note"] == fx["note_text"] + ":cfg-ffffffff:9999999:9999999"


# ── simulate: worst case ─────────────────────────────────────────────────────


@pytest.mark.parametrize(("head", "big"), [
    (9_493_750, 9_999_999),
    (9_783_999, 9_999_999),        # head + 216000 = 9,999,999: still 7 digits
    (9_784_000, 99_999_999),       # head + 216000 reaches 8 digits
    (100_000, 9_999_999),          # never fewer than 7 digits
])
def test_worst_case_digits_follow_the_note_block_horizon(head, big):
    note = note_of(segment("rolling-era-king"), segment("v2b-69630f92"))
    out = simulate(note, head, "cfg-1234abcd", expect_rc=0, worst_case=True)
    assert out["cfg"] == {"feature": "cfg-1234abcd", "lock_block": big,
                          "activation_block": big}
    assert out["note"] == note_of(segment("rolling-era-king"),
                                  segment("v2b-69630f92", big, big),
                                  segment("cfg-1234abcd", big, big))


def test_worst_case_leaves_primary_and_decided_segments_as_they_are():
    head = 9_500_000
    note = note_of(segment("rolling-era-king"),
                   segment("prelaunch-95-r2", 9_486_000, 9_486_900),
                   segment("lad-556b1b98", 9_490_000, 9_501_200))
    out = simulate(note, head, "cfg-1234abcd", expect_rc=0, worst_case=True)
    assert out["note"] == note_of(segment("rolling-era-king"),
                                  segment("prelaunch-95-r2", 9_486_000, 9_486_900),
                                  segment("lad-556b1b98", 9_999_999, 9_999_999),
                                  segment("cfg-1234abcd", 9_999_999, 9_999_999))


# ── simulate: trim rule ──────────────────────────────────────────────────────


def test_only_decided_segments_trimmed_is_not_a_refusal():
    head = 10_000_100
    note = note_of(segment("rolling-era-king", 10_000_001, 10_000_002),
                   segment("prelaunch-95-r2", 10_000_003, 10_000_004),
                   segment("v2b-69630f92", 10_000_005, 10_000_006))
    out = simulate(note, head, "cfg-1234abcd", 10_000_101, 10_000_200, expect_rc=0)
    assert out["trimmed"] == [{"feature": "v2b-69630f92", "role": "extra", "state": "decided"}]
    assert out["pending_trimmed"] == [] and out["refused"] is False
    assert out["note"] == note_of(segment("rolling-era-king", 10_000_001, 10_000_002),
                                  segment("prelaunch-95-r2", 10_000_003, 10_000_004),
                                  segment("cfg-1234abcd", 10_000_101, 10_000_200))
    assert out["bytes"] == len(out["note"]) <= 128


def test_newest_extra_first_then_cfg_last_never_primary():
    primary = segment("p" * 80)                                   # 84 bytes
    note = note_of(primary, segment("a-1"), segment("b-2"))
    out = simulate(note, 9_500_000, "cfg-ffffffff", 99_999_999, 99_999_999, expect_rc=1)
    assert out["trimmed"] == [
        {"feature": "b-2", "role": "extra", "state": "pending"},
        {"feature": "a-1", "role": "extra", "state": "pending"},
        {"feature": "cfg-ffffffff", "role": "cfg", "state": "pending"},
    ]
    assert out["pending_trimmed"] == ["b-2", "a-1", "cfg-ffffffff"]
    assert out["note"] == note_of(primary) and out["fits"] is True


def test_primary_alone_over_budget_does_not_fit():
    primary = segment("p" * 120)
    out = simulate(note_of(primary), 9_500_000, "cfg-ffffffff", expect_rc=1)
    assert out["note"] == note_of(primary) and out["fits"] is False
    assert _names(out["trimmed"]) == ["cfg-ffffffff"] and out["refused"] is True


def test_cfg_outranks_older_extras():
    """Room for cfg- is made by trimming extras newest first, decided or not."""
    note = note_of(segment("rolling-era-king", 9_374_400, 9_378_000),
                   segment("forfeit-42b1a486", 9_165_600, 9_166_500),
                   segment("margin-v2-d30eb443", 9_190_800, 9_194_400),
                   segment("prelaunch-95-r2"))
    out = simulate(note, 9_500_000, "cfg-ffffffff", 9_999_999, 9_999_999, expect_rc=1)
    assert out["trimmed"] == [
        {"feature": "prelaunch-95-r2", "role": "extra", "state": "pending"},
        {"feature": "margin-v2-d30eb443", "role": "extra", "state": "decided"},
    ]
    assert out["note"] == note_of(segment("rolling-era-king", 9_374_400, 9_378_000),
                                  segment("forfeit-42b1a486", 9_165_600, 9_166_500),
                                  segment("cfg-ffffffff", 9_999_999, 9_999_999))
    assert out["pending_trimmed"] == ["prelaunch-95-r2"]


def test_simulate_replaces_an_existing_cfg_segment():
    note = note_of(segment("rolling-era-king"), segment("cfg-aaaaaaaa", 9_400_000, 9_403_200))
    out = simulate(note, 9_500_000, "cfg-bbbbbbbb", expect_rc=0)
    assert out["note"] == note_of(segment("rolling-era-king"), segment("cfg-bbbbbbbb"))
    assert out["trimmed"] == []
    assert out["cfg"] == {"feature": "cfg-bbbbbbbb", "lock_block": 0, "activation_block": 0}


def test_simulate_reads_text_like_hex():
    fx = load_fixture(CRASH)
    a = _crash_simulate(fx, expect_rc=1, as_hex=True)
    b = _crash_simulate(fx, expect_rc=1, as_hex=False)
    assert a == b


@pytest.mark.parametrize(("feature", "lock", "act"), [
    ("cfg-123", 0, 0),                # not 8 hex
    ("cfg-ABCDEF12", 0, 0),           # upper case is not a cfg_feature_name
    ("lad-556b1b98", 0, 0),           # not a cfg- segment
    ("cfg-1234abcd", -1, 0),          # negative block
])
def test_simulate_refuses_a_bad_cfg_segment_with_exit_2(feature, lock, act):
    proc = run_notes("simulate", "--text", note_of(segment("rolling-era-king")),
                     "--head-block", "9500000", "--cfg", feature, "--cfg-lock", str(lock),
                     "--cfg-activation", str(act))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr.strip()


# ── cross-check against the real builder ─────────────────────────────────────


def _crashloop_records(act, v2b=(0, 0)):
    fx = load_fixture(CRASH)
    R = act.ActivationRecord
    roll = fx["records"]["rolling-era-king"]
    pre = fx["records"]["prelaunch-95-r2"]
    return {
        "record": R(feature="rolling-era-king", lock_block=roll[0], activation_block=roll[1],
                    source="tally"),
        "prelaunch_record": R(feature="prelaunch-95-r2", lock_block=pre[0],
                              activation_block=pre[1], source="tally"),
        "image_decoupled_record": (R(feature="v2b-69630f92", lock_block=v2b[0],
                                     activation_block=v2b[1], source="tally")
                                   if v2b[0] else None),
    }


def _live_records(act):
    R = act.ActivationRecord
    return {
        "record": R(feature="rolling-era-king", lock_block=9133200, activation_block=9136800,
                    source="tally"),
        "forfeit_record": R(feature="forfeit-42b1a486", lock_block=9165600,
                            activation_block=9166500, source="tally"),
        "margin_v2_record": R(feature="margin-v2-d30eb443", lock_block=9190800,
                              activation_block=9194400, source="tally"),
    }


CFG_SEGS = (("cfg-ffffffff", 9_999_999, 9_999_999), ("cfg-ffffffff", 99_999_999, 99_999_999),
            ("cfg-1234abcd", 0, 0))


def _cross_check(cfg, records, head):
    act = activation()
    for feature, lock, a in CFG_SEGS:
        seg = act.ReadySignal(feature, lock_block=lock, activation_block=a)
        untrimmed, final, trimmed = builder_note_and_trim(cfg, records, seg)
        blocks = {x.feature: (x.lock_block, x.activation_block)
                  for x in (*act.parse_signals(untrimmed), seg)}
        pending = [f for f in trimmed if not (blocks[f][0] > 0 and blocks[f][1] <= head)]
        out = simulate(untrimmed, head, feature, lock, a, expect_rc=1 if pending else 0)
        assert _names(out["trimmed"]) == trimmed, (feature, lock, a, untrimmed)
        assert out["pending_trimmed"] == pending
        assert out["note"] == final, (feature, lock, a)
        assert out["bytes"] == len(final.encode("utf-8"))
        assert out["refused"] is bool(pending)


def test_cross_check_crashloop_config(tmp_path):
    act = activation()
    _cross_check(crashloop_config(tmp_path), _crashloop_records(act), 9_493_750)


def test_cross_check_all_votes_open_config(tmp_path):
    act = activation()
    cfg = all_votes_open_config(tmp_path)
    untrimmed, _, trimmed = builder_note_and_trim(
        cfg, _live_records(act), act.ReadySignal("cfg-ffffffff", 99_999_999, 99_999_999))
    assert len(act.parse_signals(untrimmed)) >= 4 and trimmed, untrimmed   # a real trim case
    _cross_check(cfg, _live_records(act), 9_493_750)


def test_cross_check_worst_case_against_the_builder(tmp_path):
    """--worst-case on the unlocked-v2b note == the builder fed the v2b record
    and the cfg- segment at 10**7-1 (what v2b_note_problem builds)."""
    act = activation()
    fx = load_fixture(CRASH)
    cfg = crashloop_config(tmp_path)
    big = 9_999_999
    seg = act.ReadySignal("cfg-ffffffff", big, big)
    untrimmed, _, _ = builder_note_and_trim(cfg, _crashloop_records(act), seg)
    assert untrimmed == fx["note_text"]                 # the fixture is the builder's note
    _, final, trimmed = builder_note_and_trim(cfg, _crashloop_records(act, (big, big)), seg)
    out = _crash_simulate(fx, untrimmed, expect_rc=1)
    assert out["note"] == final == fx["expected_note"]
    assert _names(out["trimmed"]) == trimmed == fx["expected_trimmed"]


def test_cross_check_existing_cfg_segment_is_replaced_like_the_builder(tmp_path):
    act = activation()
    cfg = all_votes_open_config(tmp_path)
    recs = _live_records(act)
    seg = act.ReadySignal("cfg-bbbbbbbb", 9_999_999, 9_999_999)
    untrimmed, final, trimmed = builder_note_and_trim(cfg, recs, seg)
    old = untrimmed + ":" + segment("cfg-aaaaaaaa", 9_400_000, 9_403_200)
    proc_out = simulate(old, 9_493_750, "cfg-bbbbbbbb", 9_999_999, 9_999_999,
                        expect_rc=1 if trimmed else 0)
    assert proc_out["note"] == final and _names(proc_out["trimmed"]) == trimmed


# ── T22 (CI part): the mainnet shape on main plus cfg- trims nothing pending ─


def test_t22_main_release_note_plus_cfg_trims_nothing():
    """Main types every decided block (rollover, forfeit, margin-v2, prelaunch,
    v2b, shared-ladder), so a node on main signals only the primary segment and
    the cfg- vote, which fits even at 8-digit worst-case blocks."""
    act = activation()
    cfg = shipped_config()
    rec = act.ActivationRecord()
    bare = act.format_signals(act.ReadySignal(act.primary_feature(cfg).name))
    for feature, lock, a in CFG_SEGS:
        seg = act.ReadySignal(feature, lock_block=lock, activation_block=a)
        final = act.own_signal_payload(cfg, rec, netconfig_segment=seg)
        assert act.note_trimmed_segments(cfg, rec, netconfig_segment=seg) == []
        out = simulate(bare, 9_600_000, feature, lock, a, expect_rc=0)
        assert out["trimmed"] == [] and out["pending_trimmed"] == []
        assert out["note"] == final
    out = simulate(bare, 9_800_000, "cfg-ffffffff", expect_rc=0, worst_case=True)
    assert out["trimmed"] == [] and out["fits"] is True


def test_t22_live_shape_without_typed_v2b_would_refuse_but_typed_fits():
    """The external-validator shape (rolling resolved late, prelaunch locked)
    at an 8-digit worst case: with v2b still signalled the cfg- segment trims
    a pending segment (refuse); with v2b typed as main ships it, nothing goes."""
    fx = load_fixture(CRASH)
    bad = simulate(fx["note_text"], 9_800_000, "cfg-ffffffff", expect_rc=1, worst_case=True)
    assert bad["pending_trimmed"] == ["v2b-69630f92"]
    good = simulate(fx["fixed_by_pr457_note_text"], 9_800_000, "cfg-ffffffff", expect_rc=0,
                    worst_case=True)
    assert good["pending_trimmed"] == [] and good["trimmed"] == []
