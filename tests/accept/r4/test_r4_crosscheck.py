"""R4 cross-check: the tracker agrees with the REAL adoption math.

Where they overlap, the tracker must equal ``cascade.shared.activation.tally``
(the stake-weighted lock-in every node runs for a ``cfg-`` feature, DEC-CA-0045
/ DEC-CA-0057): the same eligible set (permit and stake > 0), the same signed
set (a well-formed note carrying a segment named the target, at any blocks),
the same ratio and the same verdict. The tracker computes exactly; the real
tally uses floats, so ratios compare to 1e-9 and scenarios stay off the line.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from .helpers import (
    LARGEST,
    OLD_CFG,
    RELEASE,
    TARGET,
    activation,
    note_of,
    off_note,
    on_note,
    s6_metagraph,
    s6_notes,
    track,
)
from .test_r4_tracker import MALFORMED

THRESHOLD = 0.6667


def _scenarios():
    yield "s6_all_on", s6_metagraph(), s6_notes()
    yield "s6_largest_off", s6_metagraph(), s6_notes(off=(LARGEST,))
    yield "s6_two_off", s6_metagraph(), s6_notes(off=("5Bvalidator2", "5Fvalidator6"))
    notes = s6_notes(off=("5Dvalidator4",))
    notes[LARGEST] = MALFORMED[1]
    notes["5Cvalidator3"] = on_note(cfg=OLD_CFG)
    notes["5Evalidator5"] = note_of(f"{RELEASE}:9374400:9378000", f"{TARGET}:9500000:9500900")
    del notes["5Fvalidator6"]
    yield "s6_mixed", s6_metagraph(), notes
    mg = {f"5V{i:02d}": {"stake": 1000 + 37 * i, "validator_permit": i % 5 != 0}
          for i in range(1, 16)}
    nt = {}
    for i in range(1, 16):
        hk = f"5V{i:02d}"
        nt[hk] = [on_note(), off_note(), on_note(cfg=OLD_CFG), MALFORMED[i % len(MALFORMED)]][i % 4]
    yield "fifteen_mixed", mg, nt


SCENARIOS = list(_scenarios())


@pytest.mark.parametrize(("name", "metagraph", "notes"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_cross_check_matches_real_activation_tally(tmp_path, name, metagraph, notes):
    act = activation()
    validators = [act.ValidatorStake(hotkey=hk, stake=float(row["stake"]),
                                     permit=bool(row["validator_permit"]))
                  for hk, row in metagraph.items()]
    signals = {hk: n for hk, n in notes.items() if n is not None}
    t = act.tally(TARGET, validators, signals, threshold=THRESHOLD, block=0)
    assert abs(t.ratio - THRESHOLD) > 1e-6, f"{name}: scenario sits on the line"

    out = track(tmp_path, metagraph, notes, expect_rc=0 if t.locked else 1)
    assert out["signed"] == list(t.signed), name
    assert out["eligible"] == list(t.eligible), name
    assert abs(out["signed_stake"] - t.signed_stake) < 1e-6, name
    assert abs(out["total_stake"] - t.total_stake) < 1e-6, name
    assert abs(float(Fraction(out["ratio"])) - t.ratio) < 1e-9, name
    assert abs(out["pct"] / 100 - t.ratio) < 1e-6, name
    assert out["meets_threshold"] is t.locked, name
    eligible_off = sorted(set(t.eligible) - set(t.signed),
                          key=lambda hk: (-metagraph[hk]["stake"], hk))
    assert [r["hotkey"] for r in out["laggards"]] == eligible_off, name


@pytest.mark.parametrize("note", MALFORMED)
def test_malformed_fixtures_are_malformed_under_real_grammar(tmp_path, note):
    """The real ``parse_signals`` rejects each malformed fixture, and the
    tracker names it ``malformed`` (never on target)."""
    act = activation()
    assert act.parse_signals(note) is None, note
    assert act.signal_for(note, TARGET) is None, note
    mg = {"a": {"stake": 1, "validator_permit": True}, "b": {"stake": 3, "validator_permit": True}}
    out = track(tmp_path, mg, {"a": note, "b": on_note()}, expect_rc=0)
    row = {r["hotkey"]: r for r in out["hotkeys"]}["a"]
    assert row["note_state"] == "malformed"
    assert row["on_target"] is False


def test_release_matches_real_primary_segment(tmp_path):
    act = activation()
    note = on_note(release="rolling-era-king-r9")
    out = track(tmp_path, {"a": {"stake": 1, "validator_permit": True}}, {"a": note}, expect_rc=0)
    row = out["hotkeys"][0]
    assert row["release"] == act.parse_signal(note).feature
    assert row["on_target"] is (act.signal_for(note, TARGET) is not None)
