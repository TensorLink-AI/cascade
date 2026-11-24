"""W13 recurring job: §11.1 "one digest, one identity" for every crowned
persisted lane D report (``python -P -m cascade.lane_d.ops_jobs king-check``).

Index-only by default (the crowned digest is in the signed
``champions/lane_d/index.json`` with its crown block); with local copies of
the vault and the three mirrors it runs ``check_one_digest`` (re-hash of the
vault and every mirror). Read-only. A crowned king missing from the index is
exit 1 with the critical ``lane_d_ops_king_unpublished`` line. Report
signatures are checked with the real ``verify_report`` interface (replaced
here by a signer-equality stand-in: building a schema-valid signed report is
W7a's fixture work, not this job's). Every import of the job is inside a test
body (a base without it fails on a missing module in W13's PATHS).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .helpers import fixture_key, king_record, vault_with_king

B = 9_000_900


def _run(argv: list[str]) -> int:
    import importlib

    try:
        return importlib.import_module("cascade.lane_d.ops_jobs").main(argv)
    except SystemExit as e:
        return int(e.code or 0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import cascade.validator.lane_d_report as rep

    signer = fixture_key("validator")
    monkeypatch.setattr(rep, "verify_report", lambda r, s: r.get("signer") == s)
    state = tmp_path / "state" / "lane_d_state.json"
    state.parent.mkdir()
    return {"tmp": tmp_path, "signer": signer, "state": state}


def _report(setup, boundary: int, crowned, signer=None) -> None:
    from cascade.validator.lane_d_state import report_path

    p = report_path(setup["state"], boundary)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"boundary": boundary, "crowned": crowned,
                             "signer": signer or setup["signer"].ss58_address}))


def _index(setup, kings: list[dict], key=None) -> Path:
    from cascade.lane_d.publish import build_index

    path = setup["tmp"] / "index.json"
    path.write_text(json.dumps(build_index(kings, key or setup["signer"])))
    return path


def _argv(setup, index: Path, *extra: str) -> list[str]:
    return ["king-check", "--state", str(setup["state"]), "--index", str(index),
            "--signer-ss58", setup["signer"].ss58_address, *extra]


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_index_check(setup, capsys):
    hk = fixture_key("miner").ss58_address
    d1, d2 = "1" * 64, "2" * 64
    _report(setup, B - 900, None)                       # no crown: not checked
    _report(setup, B, d1)
    index = _index(setup, [king_record(hk, d1, crown_block=B)])
    assert _run(_argv(setup, index)) == 0
    out = _last(capsys)
    assert out["crowned_reports"] == 1 and out["problems"] == {} and out["scope"] == "index"
    _report(setup, B + 900, d2)                         # crowned, never published
    assert _run(_argv(setup, index)) == 1
    out = _last(capsys)
    assert out["problems"] == {str(B + 900): ["index_missing"]}
    index = _index(setup, [king_record(hk, d1, crown_block=B),
                           king_record(hk, d2, crown_block=B + 1)])
    assert _run(_argv(setup, index)) == 1
    assert _last(capsys)["problems"] == {str(B + 900): ["crown_block_mismatch"]}


def test_refuses_unverified_index_and_reports(setup, capsys):
    hk = fixture_key("miner").ss58_address
    _report(setup, B, "1" * 64)
    forged = _index(setup, [king_record(hk, "1" * 64, crown_block=B)], key=fixture_key("x"))
    assert _run(_argv(setup, forged)) == 1
    assert _last(capsys)["action"] == "failed"
    index = _index(setup, [king_record(hk, "1" * 64, crown_block=B)])
    _report(setup, B, "1" * 64, signer=fixture_key("intruder").ss58_address)
    assert _run(_argv(setup, index)) == 1
    assert _last(capsys)["problems"] == {str(B): ["report_unverified"]}
    # crowned but no index at all yet
    (setup["tmp"] / "index.json").unlink()
    _report(setup, B, "1" * 64)
    assert _run(_argv(setup, index)) == 1
    assert _last(capsys)["problems"] == {str(B): ["index_missing"]}
    assert _run(["king-check"]) == 2


def test_full_check_rehashes_vault_and_mirrors(setup, capsys, tmp_path):
    from cascade.lane_d.publish import MIRRORS, publish_king
    from cascade.lane_d.storage import FilesystemStorage

    hk = fixture_key("miner").ss58_address
    vault, digest, _files = vault_with_king(tmp_path, hk)
    mirrors = {n: FilesystemStorage(tmp_path / "m" / n) for n in MIRRORS}
    publish_king(hotkey=hk, digest=digest, vault=vault, mirrors=mirrors, alert=lambda a: None)
    _report(setup, B, digest)
    index = _index(setup, [king_record(hk, digest, crown_block=B)])
    full = ["--vault-dir", str(tmp_path / "vault"),
            *[f"--mirror={n}={tmp_path / 'm' / n}" for n in MIRRORS]]
    assert _run(_argv(setup, index, *full)) == 0
    assert _last(capsys)["scope"] == "full"
    victim = next((tmp_path / "m" / "hf").rglob("weights.safetensors"))
    victim.write_bytes(b"overwritten in place")
    assert _run(_argv(setup, index, *full)) == 1
    assert _last(capsys)["problems"] == {str(B): ["hf_digest_mismatch"]}
    assert _run(_argv(setup, index, "--mirror", "nope=/x")) == 1
