"""W7c (RV6 B3): a lost state file after a settlement fails closed, never
genesis (plan rev6 §3.2 invariant 5, T11, §1.5 "State": "protected like
`cascade_state.json` ... never hand-edited").

RV6 reproduced it on main: settle boundary(1), then boundary(2) (X
crowned), delete `lane_d_state.json`, settle boundary(3). The report showed
`king_before` = the house and `crowned = None`, and X was judged again: a
silent dethrone plus a re-score, and this validator diverged from the others.

The pin. Once any lane D settlement has been persisted (a
`lane_d_report_<boundary>.json` under `report_path`'s directory):

- a missing state file is `LaneDStateError("state_missing")`, and an
  unreadable one is typed too (`bad_json` for empty/truncated bytes,
  `state_unreadable` when the path can't be read as a file). Nothing is
  scored, gated or written;
- a state file older than a persisted report (other than the one this call
  recovers) is a rolled-back state: `state_digest_mismatch`, nothing written;
- with `expect_prior_sha256` (the last published posterior), a state whose
  digest differs is `state_digest_mismatch`, nothing written;
- recovery is only explicit: `restore_state(state_path, backup_path, *,
  expect_sha256)` installs an operator backup whose sha256 equals the last
  receipt's posterior and whose `last_report` binds the persisted report at
  its boundary.

Genesis remains only when no prior settlement artefact exists. README.md
has the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .helpers import (
    STATE_ERROR_CODES,
    Chain,
    CountingForecasts,
    boundary,
    canonical,
    chain_fingerprint,
    receipt_of,
    sha,
    state_sha,
    write_canonical,
)


def _state_error(excinfo, code: str) -> None:
    from cascade.validator.lane_d_state import LaneDStateError

    assert isinstance(excinfo.value, LaneDStateError), repr(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    assert excinfo.value.code in STATE_ERROR_CODES


def _tree(directory: Path) -> dict:
    """Every file and directory under the run's state dir, with file bytes."""
    out = {}
    for p in sorted(Path(directory).rglob("*")):
        out[str(p.relative_to(directory))] = p.read_bytes() if p.is_file() else "<dir>"
    return out


def _reference(chain: Chain) -> dict:
    ref = chain_fingerprint(chain, chain.state_path("ref"), 3)
    assert all(isinstance(r, dict) for r in ref["receipts"]), "settle_lane_d returned no receipt"
    return ref


def _refused_untouched(chain: Chain, n: int, path: Path, code: str, **extra) -> None:
    """settle boundary(n) must refuse with `code` before anything is read,
    gated or written: the state dir is byte-identical afterwards."""
    before = _tree(path.parent)
    hosts = len(chain.hosts)
    fc = CountingForecasts(chain.forecasts)
    with pytest.raises(ValueError) as e:
        chain.settle(n, path, forecasts=fc, **extra)
    _state_error(e, code)
    assert _tree(path.parent) == before, "a refused settlement writes nothing (no report, " \
        "no state)"
    assert fc.reads == 0, "a refused settlement scores nothing"
    assert all(h.calls == [] for h in chain.hosts[hosts:]), "a refused settlement gates nothing"


# ── a missing state file after a settlement ─────────────────────────────────


@pytest.mark.parametrize("settled", [1, 2], ids=["after_first_clear", "after_crowning"])
def test_rv6_b3_a_missing_state_after_a_settlement_is_refused_not_genesis(tmp_path, settled):
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    chain.run(path, settled)
    path.unlink()
    _refused_untouched(chain, settled + 1, path, "state_missing")
    assert not path.exists(), "the refusal never writes a genesis state"


def test_rv6_b3_the_reproduction_no_silent_dethrone_no_rescore(tmp_path):
    """RV6's exact sequence: b1, b2 (X crowned), state deleted, b3."""
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    chain.settle(1, path)
    s2 = chain.settle(2, path)
    assert s2.report["crowned"] == chain.x["digest"]
    path.unlink()
    try:
        s3 = chain.settle(3, path)
    except ValueError as e:
        from cascade.validator.lane_d_state import LaneDStateError

        assert isinstance(e, LaneDStateError) and e.code == "state_missing", repr(e)
        return
    raise AssertionError(
        f"boundary(3) settled from genesis after a crowning: king_before="
        f"{s3.report['king_before']['digest']!r} (house {chain.house['digest']!r}), "
        f"judged={s3.report['judged']!r}")


@pytest.mark.parametrize("defect,code", [("empty", "bad_json"), ("truncated", "bad_json"),
                                         ("directory", "state_unreadable")])
def test_rv6_b3_an_unreadable_state_after_a_settlement_is_refused(tmp_path, defect, code):
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    chain.run(path, 2)
    good = path.read_bytes()
    if defect == "empty":
        path.write_bytes(b"")
    elif defect == "truncated":
        path.write_bytes(good[:len(good) // 2])
    else:
        path.unlink()
        path.mkdir()
    _refused_untouched(chain, 3, path, code)


def test_rv6_b3_load_state_never_hides_an_unreadable_file_as_genesis(tmp_path):
    from cascade.validator.lane_d_state import load_state

    chain = Chain(tmp_path)
    path = chain.state_path("run")
    path.mkdir()
    with pytest.raises(ValueError) as e:
        load_state(path, chain.house)
    _state_error(e, "state_unreadable")


# ── genesis remains only with no prior artefact ─────────────────────────────


def test_rv6_b3_genesis_still_applies_when_no_settlement_was_persisted(tmp_path):
    from cascade.validator.lane_d_state import genesis, report_path

    chain = Chain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("fresh")
    report_path(path, boundary(1)).parent.mkdir(parents=True)   # an empty report dir
    s1 = chain.settle(1, path)
    assert receipt_of(s1) == ref["receipts"][0]
    assert receipt_of(s1)["prior_state_sha256"] == state_sha(genesis(chain.house))


def test_rv6_b3_a_lost_state_before_its_first_report_recovers_from_genesis(tmp_path,
                                                                         monkeypatch):
    """The first settlement crashed after its report landed: the only persisted
    report is the boundary being settled, so genesis is the true prior and
    the recovery matches the clean run (README "Recovery")."""
    from .helpers import crash_on_rename

    chain = Chain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("crash")
    crash_on_rename(monkeypatch, path)
    with pytest.raises(OSError):
        chain.settle(1, path)
    monkeypatch.undo()
    assert not path.exists()
    s1 = chain.settle(1, path)
    assert receipt_of(s1) == ref["receipts"][0]
    assert sha(path.read_bytes()) == ref["states"][0]


# ── a rolled-back state file ────────────────────────────────────────────────


def test_rv6_b3_a_rolled_back_state_is_refused(tmp_path):
    """The state at boundary(1) put back while boundary(2)'s report is
    persisted: settling boundary(3) on it would re-judge X and drop the crown."""
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    chain.settle(1, path)
    old = path.read_bytes()
    chain.settle(2, path)
    path.write_bytes(old)
    _refused_untouched(chain, 3, path, "state_digest_mismatch")
    assert path.read_bytes() == old


def test_rv6_b3_an_older_state_with_only_this_boundarys_report_recovers(tmp_path):
    """The state at boundary(1) with boundary(2)'s report persisted, settling
    boundary(2): indistinguishable from a crash between the two writes, so it
    is the recovery path, and it ends on the clean run's bytes."""
    chain = Chain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("run")
    chain.settle(1, path)
    old = path.read_bytes()
    chain.settle(2, path)
    path.write_bytes(old)
    s2 = chain.settle(2, path)
    assert receipt_of(s2) == ref["receipts"][1]
    assert sha(path.read_bytes()) == ref["states"][1]


# ── the caller's expected prior ─────────────────────────────────────────────


def test_rv6_b3_settle_checks_the_expected_prior(tmp_path):
    from cascade.validator.lane_d_state import genesis

    chain = Chain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("run")
    s1 = chain.settle(1, path, expect_prior_sha256=state_sha(genesis(chain.house)))
    assert receipt_of(s1) == ref["receipts"][0]
    good = receipt_of(s1)["posterior_state_sha256"]
    true_bytes = path.read_bytes()
    assert sha(true_bytes) == good
    # a canonical hand edit: a well-formed state, not the last posterior
    edited = json.loads(true_bytes)
    (dg,) = sorted(edited["entries"])
    edited["entries"][dg]["context"] = 8192
    write_canonical(path, edited)
    assert canonical(edited) == path.read_bytes() and sha(path.read_bytes()) != good
    _refused_untouched(chain, 2, path, "state_digest_mismatch", expect_prior_sha256=good)
    # the true posterior back: the expected prior matches and the chain goes on
    path.write_bytes(true_bytes)
    s2 = chain.settle(2, path, expect_prior_sha256=good)
    assert receipt_of(s2) == ref["receipts"][1]
    assert receipt_of(s2)["prior_state_sha256"] == good


def test_rv6_b3_an_expected_genesis_never_licenses_a_lost_state(tmp_path):
    from cascade.validator.lane_d_state import genesis

    chain = Chain(tmp_path)
    path = chain.state_path("run")
    chain.settle(1, path)
    path.unlink()
    _refused_untouched(chain, 2, path, "state_missing",
                       expect_prior_sha256=state_sha(genesis(chain.house)))


# ── the explicit restore ────────────────────────────────────────────────────


def _crowned_with_backup(tmp_path):
    """b1, b2 (X crowned), the operator's backup of the b2 state, then the
    live state file is lost."""
    chain = Chain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("run")
    chain.settle(1, path)
    s2 = chain.settle(2, path)
    backup = tmp_path / "backup" / "lane_d_state.json"
    backup.parent.mkdir()
    backup.write_bytes(path.read_bytes())
    path.unlink()
    return chain, ref, path, backup, receipt_of(s2)


def test_rv6_b3_restore_from_a_backup_bound_to_the_last_receipt(tmp_path):
    from cascade.validator.lane_d_state import report_path, restore_state

    chain, ref, path, backup, r2 = _crowned_with_backup(tmp_path)
    _refused_untouched(chain, 3, path, "state_missing")
    reports = _tree(report_path(path, boundary(1)).parent)
    got = restore_state(path, backup, expect_sha256=r2["posterior_state_sha256"])
    assert got == json.loads(backup.read_bytes())
    assert path.read_bytes() == backup.read_bytes()
    assert _tree(report_path(path, boundary(1)).parent) == reports, \
        "restore never touches the persisted reports"
    s3 = chain.settle(3, path)
    assert receipt_of(s3) == ref["receipts"][2]
    assert receipt_of(s3)["prior_state_sha256"] == r2["posterior_state_sha256"]
    assert s3.report["king_before"]["digest"] == chain.x["digest"]
    assert s3.report["crowned"] is None and s3.report["judged"] == []
    assert s3.pay["court_uids"][0] == chain.uid_of[chain.x["hotkey"]]
    assert sha(path.read_bytes()) == ref["states"][2]


def _restore_refused(path: Path, backup: Path, expect: str, code: str) -> None:
    from cascade.validator.lane_d_state import restore_state

    before = _tree(path.parent)
    with pytest.raises(ValueError) as e:
        restore_state(path, backup, expect_sha256=expect)
    _state_error(e, code)
    assert _tree(path.parent) == before, "a refused restore writes nothing"


def test_rv6_b3_restore_refuses_a_backup_that_is_not_the_last_posterior(tmp_path):
    from cascade.validator.lane_d_state import genesis

    chain, _, path, backup, r2 = _crowned_with_backup(tmp_path)
    _restore_refused(path, backup, r2["prior_state_sha256"], "state_digest_mismatch")
    _restore_refused(path, backup, state_sha(genesis(chain.house)), "state_digest_mismatch")
    assert not path.exists()


def test_rv6_b3_restore_refuses_a_missing_or_defective_backup(tmp_path):
    chain, _, path, backup, r2 = _crowned_with_backup(tmp_path)
    want = r2["posterior_state_sha256"]
    _restore_refused(path, backup.parent / "nope.json", want, "state_missing")
    good = backup.read_bytes()
    backup.write_bytes(good.replace(b",", b", ", 1))
    _restore_refused(path, backup, want, "not_canonical")
    backup.write_bytes(good.replace(b'"boundary":', b'"boundary":0,"boundary":', 1))
    _restore_refused(path, backup, want, "duplicate_key")
    assert not path.exists()


def test_rv6_b3_restore_refuses_a_backup_that_does_not_bind_the_persisted_report(tmp_path):
    """A canonical backup at boundary(2) whose `last_report` names another
    report body: its own sha is given, but it isn't the state that bound the
    report persisted at boundary(2)."""
    chain, _, path, backup, _ = _crowned_with_backup(tmp_path)
    forged = json.loads(backup.read_bytes())
    forged["last_report"]["report_sha256"] = "0" * 64
    write_canonical(backup, forged)
    _restore_refused(path, backup, sha(backup.read_bytes()), "report_conflict")
    assert not path.exists()


def test_rv6_b3_error_codes_include_the_lost_state_codes():
    from cascade.validator.lane_d_state import STATE_ERROR_CODES as codes

    assert {"state_missing", "state_unreadable"} <= set(codes)
