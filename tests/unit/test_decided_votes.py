"""The decided-votes drift guard (``cascade.shared.decided``, ``cascade-audit
decided``, ``decided_votes.toml``).

* The repo's own ``chain.toml`` and ``decided_votes.toml`` are IN SYNC — this
  is the CI guard: a vote that locks in and is not typed (or a pin moved on
  one side only) fails here on the next PR, whatever it is about.
* Every record row names a feature the shipped config actually computes
  (changed vote params = a fresh vote = a stale row).
* A decided, applied, untyped block is a TYPE finding carrying the exact
  lines to type; a typed block that disagrees is MISMATCH; a pin that moved
  is PIN_DRIFT; a row the config no longer computes is STALE; a decided block
  not yet applied is PENDING (never asked for early); a held row is HELD.
* A node's persisted lock-ins vs the record: UNRECORDED (add the row),
  DISAGREE (nodes differ — never typed), UNKNOWN (retired feature).
* The CLI exits 1 on drift, 0 in sync, and emits JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import cascade.shared.decided as D
from cascade.shared.activation import ActivationRecord, ActivationStore
from cascade.shared.config import load_chain_config

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def shipped():
    return load_chain_config(REPO / "chain.toml")


@pytest.fixture(scope="module")
def record():
    return D.load_decided(REPO / D.DECIDED_FILE)


def _by(findings, feature, key=None):
    return [f for f in findings if f.feature == feature and (key is None or f.key == key)]


# ── the guard itself ─────────────────────────────────────────────────────────


def test_repo_chain_toml_and_record_are_in_sync(shipped, record):
    findings = D.drift_report(shipped, record)
    bad = [f for f in findings if f.fails]
    assert not bad, "chain.toml has drifted from decided_votes.toml:\n" + D.render(bad)
    # every row's feature is one the shipped config computes by that exact
    # name — a changed param set would be a fresh vote and a stale row
    names = set(D.configured_features(shipped))
    for v in record.votes:
        assert v.feature in names, (v.feature, sorted(names))
    # the decided-and-typed ones read OK, the held ones HELD with a reason
    kinds = {f.feature: f.kind for f in findings if f.feature != "pins"}
    assert kinds["v2b-69630f92"] == D.OK
    assert kinds["forfeit-42b1a486"] == D.OK
    assert kinds["margin-v2-d30eb443"] == D.OK
    assert kinds["lad-556b1b98"] == D.OK
    assert kinds["rolling-era-king"] == D.OK          # typed 2026-11-13 (9136800)
    assert record.vote("rolling-era-king").type_it
    for held in ("prelaunch-95-r2",):
        assert kinds[held] == D.HELD
        assert record.vote(held).reason
    assert all(f.kind == D.OK for f in _by(findings, "pins"))
    assert {f.key for f in _by(findings, "pins")} == set(D.PIN_KEYS)
    assert record.as_of_block >= max(v.activation_block for v in record.votes
                                     if v.type_it or v.feature.startswith("v2b-"))


def test_every_pin_key_is_recorded(record):
    assert set(record.pins) == set(D.PIN_KEYS)


def test_record_rejects_bad_rows():
    with pytest.raises(ValueError, match="duplicate"):
        D.parse_decided('[[vote]]\nfeature="a"\nactivation_block=1\n'
                        '[[vote]]\nfeature="a"\nactivation_block=2\n')
    with pytest.raises(ValueError, match="activation_block"):
        D.parse_decided('[[vote]]\nfeature="a"\nactivation_block=0\n')
    with pytest.raises(ValueError, match="lock_block"):
        D.parse_decided('[[vote]]\nfeature="a"\nlock_block=5\nactivation_block=2\n')
    with pytest.raises(ValueError, match="unknown pin"):
        D.parse_decided('[pins]\n"round.nope" = "x"\n')


# ── findings ─────────────────────────────────────────────────────────────────


def test_untyped_decided_block_in_the_shipped_file_is_a_type_finding(tmp_path, record):
    # the file exactly as a validator loads it, with the v2b line untyped
    # (= main before PR #457): the guard names the line to type
    src = re.sub(r"(?m)^image_decoupled_from_block\s*=\s*\d+", "image_decoupled_from_block = 0",
                 (REPO / "chain.toml").read_text(encoding="utf-8"))
    p = tmp_path / "chain.toml"
    p.write_text(src, encoding="utf-8")
    cfg = load_chain_config(p)
    (f,) = _by(D.drift_report(cfg, record), "v2b-69630f92")
    assert f.kind == D.TYPE and f.fails
    assert f.key == "v2.image_decoupled_from_block"
    assert f.expected == 9493200 and f.actual == 0
    assert f.fix == "[v2]\nimage_decoupled_from_block = 9493200"
    assert "→ type:" in D.render([f]) and "image_decoupled_from_block = 9493200" in D.render([f])


def test_not_yet_applied_is_pending_not_type(shipped, record):
    cfg = replace(shipped, v2=replace(shipped.v2, image_decoupled_from_block=0))
    (f,) = _by(D.drift_report(cfg, record, as_of_block=9490000), "v2b-69630f92")
    assert f.kind == D.PENDING and not f.fails
    (f,) = _by(D.drift_report(cfg, record, as_of_block=9493200), "v2b-69630f92")
    assert f.kind == D.TYPE


def test_typed_block_that_disagrees_with_the_decision_is_mismatch(shipped, record):
    cfg = replace(shipped, v2=replace(shipped.v2, image_decoupled_from_block=9500400))
    (f,) = _by(D.drift_report(cfg, record), "v2b-69630f92")
    assert f.kind == D.MISMATCH and f.fails
    assert f.expected == 9493200 and f.actual == 9500400
    assert f.fix.endswith("image_decoupled_from_block = 9493200")


def test_held_row_is_never_asked_for(shipped, record):
    # prelaunch is open in the shipped file and HELD in the record: no TYPE
    (f,) = _by(D.drift_report(shipped, record), "prelaunch-95-r2")
    assert f.kind == D.HELD and not f.fails
    assert record.vote("prelaunch-95-r2").reason in f.detail
    # a HELD rolling row against the pre-2026-11-13 file (rollover untyped)
    import tempfile

    from tests.unit.predecision_main import predecision_text
    held = D.DecidedRecord(as_of_block=record.as_of_block, votes=(
        D.DecidedVote("rolling-era-king", 9133200, 9136800, "config", type_it=False,
                      reason="nodes disagree: 9374400→9378000"),))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.toml"
        p.write_text(predecision_text())
        untyped = load_chain_config(p)
    (f,) = _by(D.drift_report(untyped, held), "rolling-era-king")
    assert f.kind == D.HELD and not f.fails and "9374400" in f.detail


def test_typed_rollover_other_than_the_decided_one_is_a_mismatch(shipped, record):
    # the external node's own resolution (9378000) typed instead: MISMATCH
    cfg = replace(shipped, round=replace(shipped.round, rolling_from_block=9378000,
                                         epoch_blocks=900, epoch_blocks_prev=3600,
                                         epoch_activation_block=9378000))
    (f,) = _by(D.drift_report(cfg, record), "rolling-era-king")
    assert f.kind == D.MISMATCH and f.fails
    assert f.expected == 9136800 and f.actual == 9378000


def test_pin_drift(shipped, record):
    cfg = replace(shipped, round=replace(shipped.round, worker_code_fingerprint="0" * 64))
    (f,) = _by(D.drift_report(cfg, record), "pins", "round.worker_code_fingerprint")
    assert f.kind == D.PIN_DRIFT and f.fails
    assert f.fix == f'[round]\nworker_code_fingerprint = "{record.pins[f.key]}"'
    # the v2 contract's pin is read through [v2.training]
    (ok,) = _by(D.drift_report(shipped, record), "pins", "v2.training.train_image_digest")
    assert ok.kind == D.OK and ok.actual == shipped.training.v2_contract.train_image_digest


def test_stale_row_and_unknown_feature(shipped, record):
    stale = D.DecidedRecord(as_of_block=record.as_of_block, votes=(
        D.DecidedVote("v2b-deadbeef", 1, 9493200, "tally"),
        D.DecidedVote("promotion-carry", 1, 9291600, "signals")))
    findings = D.drift_report(shipped, stale)
    kinds = {f.feature: f.kind for f in findings}
    assert kinds["v2b-deadbeef"] == D.STALE
    assert kinds["promotion-carry"] == D.UNKNOWN
    # the open votes of the config with no row at all are reported as pending
    # (the rollover is typed since 2026-11-13: no longer an open vote)
    assert kinds["prelaunch-95-r2"] == D.PENDING
    assert kinds.get("rolling-era-king") in (None, D.OK)


def test_state_report(record):
    recs = [
        ActivationRecord("v2b-69630f92", 9486000, 9493200, 9486000, "tally"),          # agrees
        ActivationRecord("rolling-era-king", 9374400, 9378000, 9374400, "signals"),    # differs
        ActivationRecord("margin-v2-cafebabe", 9190800, 9194400, 9190800, "tally"),   # new vote
        ActivationRecord("promotion-carry", 9290700, 9291600, 9290700, "signals"),    # retired
        ActivationRecord("lad-556b1b98", 0, 0, 9494100, ""),                           # not locked
    ]
    kinds = {f.feature: f for f in D.state_report(recs, record)}
    assert kinds["v2b-69630f92"].kind == D.OK
    assert kinds["rolling-era-king"].kind == D.DISAGREE and not kinds["rolling-era-king"].fails
    assert kinds["rolling-era-king"].expected == 9136800
    assert kinds["rolling-era-king"].actual == 9378000
    un = kinds["margin-v2-cafebabe"]
    assert un.kind == D.UNRECORDED and un.fails
    assert un.fix == ('[[vote]]\nfeature = "margin-v2-cafebabe"\nlock_block = 9190800\n'
                      'activation_block = 9194400\nsource = "tally"')
    assert kinds["promotion-carry"].kind == D.UNKNOWN
    assert "lad-556b1b98" not in kinds


def test_read_state_records(tmp_path):
    ActivationStore(tmp_path / "activation_state.json").save(
        ActivationRecord("rolling-era-king", 9133200, 9136800, 9133200, "config"))
    ActivationStore(tmp_path / "activation_image_decoupled_state.json").save(
        ActivationRecord("v2b-69630f92", 9486000, 9493200, 9486000, "tally"))
    (tmp_path / "activation_forfeit_state.json").write_text("{not json", encoding="utf-8")
    recs = D.read_state_records(tmp_path)
    assert set(recs) == {"activation_state.json", "activation_image_decoupled_state.json"}
    assert recs["activation_state.json"].feature == "rolling-era-king"


def test_type_hint_and_key_mapping():
    hint = D.type_hint("v2b-69630f92", 9493200)
    assert "image_decoupled_from_block = 9493200" in hint and "[v2]" in hint
    assert D.DECIDED_FILE in hint
    assert D.typed_key_for("forfeit-42b1a486") == "scoring.forfeit_from_block"
    assert D.typed_key_for("lad-556b1b98") == "eval.shared_ladder_from_block"
    assert D.typed_key_for("rolling-era-king") == "round.rolling_from_block"
    assert D.typed_key_for("rolling-era-kingdom") == ""
    assert D.typed_key_for("promotion-carry") == ""
    assert "no chain.toml key" in D.type_hint("promotion-carry", 1)
    assert D.toml_line("v2.training.train_image_digest", "sha256:ab") == \
        '[v2.training]\ntrain_image_digest = "sha256:ab"'


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_cli_in_sync_and_drifted(tmp_path, capsys):
    from cascade.audit.main import main as audit_main

    rc = audit_main(["decided", "--config", str(REPO / "chain.toml")])
    out = capsys.readouterr().out
    assert rc == 0 and "in sync" in out and "[HELD] prelaunch-95-r2" in out

    src = re.sub(r"(?m)^image_decoupled_from_block\s*=\s*\d+", "image_decoupled_from_block = 0",
                 (REPO / "chain.toml").read_text(encoding="utf-8"))
    p = tmp_path / "chain.toml"
    p.write_text(src, encoding="utf-8")
    rc = audit_main(["decided", "--config", str(p), "--decided", str(REPO / D.DECIDED_FILE)])
    out = capsys.readouterr().out
    assert rc == 1 and "[TYPE] v2b-69630f92" in out and "image_decoupled_from_block = 9493200" in out

    # --state-dir: a lock-in the record lacks fails; a retired feature is informational
    ActivationStore(tmp_path / "activation_margin_v2_state.json").save(
        ActivationRecord("margin-v2-cafebabe", 9190800, 9194400, 9190800, "tally"))
    ActivationStore(tmp_path / "activation_promotion_carry_state.json").save(
        ActivationRecord("promotion-carry", 9290700, 9291600, 9290700, "signals"))
    rc = audit_main(["decided", "--config", str(REPO / "chain.toml"), "--decided",
                     str(REPO / D.DECIDED_FILE), "--state-dir", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False
    kinds = {(f["feature"], f["kind"]) for f in out["findings"]}
    assert ("margin-v2-cafebabe", D.UNRECORDED) in kinds
    assert ("promotion-carry", D.UNKNOWN) in kinds
    assert out["as_of_block"] == 9494100
