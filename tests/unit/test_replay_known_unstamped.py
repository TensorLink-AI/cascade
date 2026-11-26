"""scripts/replay_mainnet_receipts.py: the known-defect pin for receipts signed
by a pre-ab0b412 validator (rate-limited settlement push ⇒ king check recorded
but no ``emission`` block). The pin may only downgrade the exact "no reign
clock" FAIL, only when the preceding receipt verifies as pinned, and only when
the borrowed-clock re-derivation PASSes."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from cascade.audit.checks import FAIL, KING_CHECK_BENCH, PASS, SKIP, WARN, CheckResult

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replay_mainnet_receipts.py"
_spec = importlib.util.spec_from_file_location("_replay_mainnet_receipts", _SCRIPT)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


@dataclass(frozen=True)
class _Rc:
    round_id: str
    epoch_start_block: int
    emission: dict | None = None
    weights: tuple = ()
    king_check: dict | None = None
    state_post: dict | None = None


PIN = {"clock_round": "prev", "clock_boundary": 900, "reign_start_block": 100}


def _bad(**kw) -> _Rc:
    base = dict(round_id="bad", epoch_start_block=1800, emission=None, weights=(),
                king_check={"result": "fail"}, state_post={"king_since_block": 50})
    base.update(kw)
    return _Rc(**base)


def _src(**kw) -> _Rc:
    base = dict(round_id="prev", epoch_start_block=900, king_check=None,
                emission={"king_since_block": 50,
                          "ladder": {"event": "none", "reign_start_block": 100}},
                weights=(0.9, 0.1))
    base.update(kw)
    return _Rc(**base)


def test_pinned_mainnet_entry_is_the_documented_round():
    pin = R.KNOWN_UNSTAMPED_EMISSION["14666242199155766185"]
    assert pin == {"clock_round": "9550553995117568107", "clock_boundary": 9560700,
                   "reign_start_block": 9547200}


def test_pinned_mainnet_entry_2026_11_24():
    # rc37 settlement 9576000: rate-limited push (13:42:50, 88 blocks after the
    # 13:25:08 re-assert); clock 9561600 from the 9574200 receipt — the 9575100
    # settlement between them recorded no king check, no verdict, ladder none.
    pin = R.KNOWN_UNSTAMPED_EMISSION["13058784948867351860"]
    assert pin == {"clock_round": "9495195076473738382", "clock_boundary": 9574200,
                   "reign_start_block": 9561600}


def test_clock_borrowable_when_everything_verifies():
    assert R.known_clock_problem(_bad(), _src(), PIN) is None
    patched = R.with_borrowed_clock(_bad(), _src())
    assert patched.emission == {"king_since_block": 50, "ladder": {"reign_start_block": 100}}
    assert _bad().emission is None                      # the original is untouched


@pytest.mark.parametrize("rc, src, needle", [
    (_bad(emission={"ladder": {}}), _src(), "carries an emission record"),
    (_bad(weights=(1.0,)), _src(), "landed weight vector"),
    (_bad(king_check=None), _src(), "no king check"),
    (_bad(), _src(round_id="other"), "!= pinned prev"),
    (_bad(), _src(epoch_start_block=901), "boundary 901"),
    (_bad(epoch_start_block=900), _src(), "not before 900"),
    (_bad(), _src(emission={"king_since_block": 50,
                            "ladder": {"event": "none", "reign_start_block": 101}}), "101"),
    (_bad(), _src(emission={"king_since_block": 50, "ladder": {"event": "fail",
                                                              "reign_start_block": 100}}),
     "could have moved"),
    (_bad(), _src(king_check={"result": "pass"}), "could have moved"),
    (_bad(state_post={"king_since_block": 51}), _src(), "king_since_block 51"),
])
def test_clock_refused_on_any_mismatch(rc, src, needle):
    why = R.known_clock_problem(rc, src, PIN)
    assert why is not None and needle in why


def _patch(monkeypatch, *, src, rederived):
    import cascade.audit.main as M

    calls = []

    def fake_audit(rc, cfg, *, tier, client=None):
        calls.append(rc)
        if rederived is None:
            return [CheckResult("status", PASS)]
        name = rederived[2] if len(rederived) > 2 else "king-check"
        return [CheckResult(name, rederived[0], rederived[1])]

    monkeypatch.setattr(M, "audit_receipt", fake_audit)
    monkeypatch.setattr(M, "_fetch_text", lambda cfg, key: key)
    monkeypatch.setattr(M, "load_receipt", lambda text: src)
    return calls


ROWS = [{"round_id": "prev", "receipt_key": "receipts/x/round-prev.json"}]
BEFORE = [CheckResult("status", PASS), CheckResult("king-check", FAIL, R.NO_CLOCK),
          CheckResult("weights", SKIP)]


def test_downgrades_only_when_rederivation_passes(monkeypatch):
    calls = _patch(monkeypatch, src=_src(), rederived=(PASS, "fail reproduced"))
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, ROWS, cfg=None, client=None)
    kc = out[1]
    assert kc.status == WARN and "KNOWN DEFECT" in kc.detail and "fail reproduced" in kc.detail
    assert calls[0].emission["ladder"]["reign_start_block"] == 100
    assert [x.status for x in out if x.name != "king-check"] == [PASS, SKIP]


def test_rederivation_failure_keeps_fail(monkeypatch):
    _patch(monkeypatch, src=_src(), rederived=(FAIL, "recorded at 1800 but not due"))
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, ROWS, cfg=None, client=None)
    assert out[1].status == FAIL and "not due" in out[1].detail


def test_unverifiable_clock_keeps_fail(monkeypatch):
    calls = _patch(monkeypatch, src=_src(round_id="other"), rederived=(PASS, ""))
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, ROWS, cfg=None, client=None)
    assert out[1].status == FAIL and "pin rejected" in out[1].detail and not calls
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, [], cfg=None, client=None)
    assert out[1].status == FAIL and "not in the receipt index" in out[1].detail


def test_other_king_check_fails_untouched(monkeypatch):
    calls = _patch(monkeypatch, src=_src(), rederived=(PASS, ""))
    other = [CheckResult("king-check", FAIL, "the king check was due at 1800 but none")]
    assert R._replay_known_unstamped(_bad(), other, PIN, ROWS, cfg=None,
                                     client=None) == other
    assert not calls


def test_bench_rows_unfetched_warn_downgrades(monkeypatch):
    # due-ness + pool half + recorded bench means verify, signed task rows not
    # fetched ⇒ the audit's KING_CHECK_BENCH WARN; still a verified downgrade
    _patch(monkeypatch, src=_src(), rederived=(WARN, "fail reproduced; NOT replayed",
                                               KING_CHECK_BENCH))
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, ROWS, cfg=None, client=None)
    assert out[1].name == "king-check" and out[1].status == WARN
    assert "KNOWN DEFECT" in out[1].detail and "NOT replayed" in out[1].detail


@pytest.mark.parametrize("rederived", [
    None,                                               # no king-check row at all
    (FAIL, "bench half differs", KING_CHECK_BENCH),
    (WARN, "transient deferral"),                       # king-check WARN is not a PASS
])
def test_other_rederivations_keep_fail(monkeypatch, rederived):
    _patch(monkeypatch, src=_src(), rederived=rederived)
    out = R._replay_known_unstamped(_bad(), BEFORE, PIN, ROWS, cfg=None, client=None)
    assert out[1].status == FAIL and "pin rejected" in out[1].detail
