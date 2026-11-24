"""R11 (lane D ops item 13): the lane A v2 trainer preflight that qualifies a
compliant initial king for a fresh (testnet) deployment.

``cascade-trainer --preflight-v2-baseline REF`` (fetch + full sandbox verify +
local proof) existed; a fresh deployment could not learn every blocker before
its first king leg, the blocked log line printed a literal ``REF`` / ``PATH``
instead of the king's ref, and the command exited 0 while the launch stayed
blocked (no validator anchor, unqualified worker release). The fix, tooling
only (lane A rules unchanged):

* ``launch_preflight.preflight_report(cfg, ref, work_root)``: read-only, every
  blocker named, nothing fetched or written;
* ``--preflight-v2-baseline-check REF``: that report as JSON, exit 0 ready /
  3 blocked / 2 error;
* ``--preflight-v2-baseline REF`` ends with the same report: exit 3 when a
  blocker remains after qualification;
* the trainer's blocked line names the real ref and work root.

Shipped ``chain.testnet.toml`` (the fresh testnet deployment); no torch, no
network, no fetch (``qualify`` is replaced where it would fetch).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from .helpers import REPO

REF = "fixtures/testnet-baseline@sha256:" + "c" * 64


@pytest.fixture(scope="module")
def cfg_testnet():
    from cascade.shared.config import load_chain_config

    return load_chain_config(REPO / "chain.testnet.toml")


def _unpinned(cfg, *, anchor="fixture-validator"):
    """The testnet config before the dated launch pins (``check_worker_release``
    is then a no-op), with the given validator receipt anchor."""
    return replace(cfg, v2=replace(cfg.v2, from_block=0, launch_at_utc=""),
                   manifest=replace(cfg.manifest, validator_hotkey=anchor))


def _full_proof(cfg, ref, work_root, **verification):
    from cascade.trainer.launch_preflight import identity, proof_path

    report = {"ok": True, "scope": "full", "runtime_skipped": False,
              "corpus_digest": "d" * 64, **verification}
    path = proof_path(work_root, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**identity(cfg, ref), "verification": report}, sort_keys=True))
    return path


def test_report_names_every_blocker_and_writes_nothing(cfg_testnet, tmp_path):
    from cascade.trainer.launch_preflight import preflight_report

    cfg = _unpinned(cfg_testnet, anchor="")
    work = tmp_path / "work"
    r = preflight_report(cfg, REF, work)
    assert r["ready"] is False
    assert "no_validator_anchor" in r["blockers"] and "no_proof" in r["blockers"]
    assert r["accepted_by"] is None
    assert REF in r["next"] and str(work) in r["next"] and "--preflight-v2-baseline" in r["next"]
    assert not work.exists(), "the check must write nothing"


def test_report_ready_with_a_full_proof_and_never_writes_the_launch_root(cfg_testnet, tmp_path):
    from cascade.trainer.launch_preflight import preflight_report, ready

    cfg = _unpinned(cfg_testnet)
    work = tmp_path / "work"
    proof = _full_proof(cfg, REF, work)
    r = preflight_report(cfg, REF, work)
    assert r["ready"] is True and r["blockers"] == [] and r["accepted_by"] == str(proof)
    assert not (work / "v2-launch.json").exists(), "the check never writes the launch root"
    # The check agrees with what the first king leg decides.
    assert ready(cfg, REF, work) is True
    assert (work / "v2-launch.json").exists()


def test_report_refuses_stale_and_quick_proofs(cfg_testnet, tmp_path):
    from cascade.trainer.launch_preflight import preflight_report, proof_path

    cfg = _unpinned(cfg_testnet)
    work = tmp_path / "work"
    _full_proof(cfg, REF, work, scope="quick")
    r = preflight_report(cfg, REF, work)
    assert r["ready"] is False and r["blockers"] == ["proof_not_full"]
    proof = json.loads(proof_path(work, REF).read_text())
    proof["verification"]["scope"] = "full"
    proof["verifier_sha256"] = "0" * 64
    proof_path(work, REF).write_text(json.dumps(proof))
    r = preflight_report(cfg, REF, work)
    assert r["ready"] is False and r["blockers"] == ["proof_stale:verifier_sha256"]
    proof_path(work, REF).write_text("{not json")
    r = preflight_report(cfg, REF, work)
    assert r["ready"] is False and r["blockers"][0].startswith("proof_unreadable")


def test_report_names_the_unqualified_worker_release(cfg_testnet, tmp_path):
    from cascade.trainer.launch_preflight import preflight_report

    # The shipped testnet file carries [v2] from_block: the release check runs.
    r = preflight_report(replace(cfg_testnet, manifest=replace(
        cfg_testnet.manifest, validator_hotkey="fixture-validator")), REF, tmp_path / "w")
    assert r["ready"] is False
    if cfg_testnet.v2.from_block or cfg_testnet.v2.launch_at_utc:
        assert any(b.startswith("worker_release: ") for b in r["blockers"])


def test_bad_ref_is_an_identity_blocker(cfg_testnet, tmp_path):
    from cascade.trainer.launch_preflight import preflight_report

    r = preflight_report(_unpinned(cfg_testnet), "not-an-immutable-ref", tmp_path / "w")
    assert r["ready"] is False and any(b.startswith("identity: ") for b in r["blockers"])


def _args(tmp_path, *, qualify=None, check=None):
    return argparse.Namespace(preflight_v2_baseline=qualify, preflight_v2_baseline_check=check,
                              work_root=tmp_path / "work")


def test_cli_flags_parse():
    from cascade.trainer.main import _build_parser

    try:
        a = _build_parser().parse_args(["--preflight-v2-baseline-check", REF, "--work-root", "/w"])
    except SystemExit as e:                     # an unknown flag on a base without it
        raise AssertionError(f"--preflight-v2-baseline-check is not a trainer flag ({e})") from None
    assert a.preflight_v2_baseline_check == REF and a.preflight_v2_baseline is None


def test_check_flag_exit_codes(cfg_testnet, tmp_path, capsys):
    from cascade.trainer.main import PREFLIGHT_BLOCKED, PREFLIGHT_READY, _preflight_v2_baseline

    cfg = _unpinned(cfg_testnet)
    assert _preflight_v2_baseline(cfg, _args(tmp_path, check=REF)) == PREFLIGHT_BLOCKED == 3
    out = json.loads(capsys.readouterr().out)
    assert out["preflight"]["blockers"] == ["no_proof"] and "qualification" not in out
    _full_proof(cfg, REF, tmp_path / "work")
    assert _preflight_v2_baseline(cfg, _args(tmp_path, check=REF)) == PREFLIGHT_READY == 0


def test_qualify_flag_exits_blocked_when_a_blocker_remains(cfg_testnet, tmp_path, capsys,
                                                         monkeypatch):
    from cascade.trainer import launch_preflight
    from cascade.trainer.main import PREFLIGHT_BLOCKED, PREFLIGHT_ERROR, _preflight_v2_baseline

    def fake_qualify(cfg, ref, work_root):        # the fetch + sandbox verify, done
        return json.loads(_full_proof(cfg, ref, work_root).read_text())

    monkeypatch.setattr(launch_preflight, "qualify", fake_qualify)
    no_anchor = _unpinned(cfg_testnet, anchor="")
    # Qualified, but the launch is still blocked: never exit 0.
    assert _preflight_v2_baseline(no_anchor, _args(tmp_path, qualify=REF)) == PREFLIGHT_BLOCKED
    out = json.loads(capsys.readouterr().out)
    assert out["qualification"]["ref"] == REF
    assert out["preflight"]["blockers"] == ["no_validator_anchor"]
    assert _preflight_v2_baseline(_unpinned(cfg_testnet), _args(tmp_path, qualify=REF)) == 0

    def failing(cfg, ref, work_root):
        raise ValueError("v2 baseline verification failed: fixture")

    monkeypatch.setattr(launch_preflight, "qualify", failing)
    assert _preflight_v2_baseline(no_anchor, _args(tmp_path, qualify=REF)) == PREFLIGHT_ERROR


def test_blocked_log_line_names_the_king_ref_and_work_root(cfg_testnet, tmp_path, caplog):
    from types import SimpleNamespace

    from cascade.trainer.runner_generation import _GenerationMixin

    cfg = replace(cfg_testnet, v2=replace(cfg_testnet.v2, from_block=max(1, cfg_testnet.v2.from_block)),
                  manifest=replace(cfg_testnet.manifest, validator_hotkey="fixture-validator"))

    class _Runner(_GenerationMixin):
        def __init__(self):
            self.cfg, self.work_root = cfg, tmp_path / "work"

    runner = _Runner()
    gen = SimpleNamespace(hotkey="5Fixture", uid=3, ref=REF)   # rolling.ResolvedGen's fields
    with caplog.at_level(logging.ERROR):
        assert runner._v2_baseline_ready(gen, cfg.v2.from_block + 1) is False
    line = next(r.getMessage() for r in caplog.records
                if r.getMessage().startswith("v2 launch blocked before rental"))
    assert REF in line and str(tmp_path / "work") in line
    assert "--preflight-v2-baseline" in line and " REF " not in line and "PATH" not in line
    assert runner._v2_launch_preflight_status["state"] == "blocked"


def test_operator_doc_lists_the_check():
    text = (Path(REPO) / "docs" / "OPERATOR.md").read_text(encoding="utf-8")
    assert "--preflight-v2-baseline-check" in text
