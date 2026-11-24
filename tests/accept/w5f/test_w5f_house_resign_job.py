"""W5f recurring job: re-sign the house pointer when lane A's king changes
(plan §10.4 "refresh house if KING_A digest changed", §1.4 "Refresh" / "If
lane A has none, the previous house stays").

``python -P -m cascade.lane_d.ops_jobs house-pointer`` reads the lane A king
pointer from a local JSON (the validator's state, ``king_pointer``), the
installed pointer and its ``.source.json`` sidecar, and a
``laned_house_source/1`` the owner prepares for the current king. It signs
through ``lane_d_house.sign_house_pointer``. Every import of the job is inside
a test body (a base without it fails on a missing module in W5f's PATHS).
Keys are deterministic TEST keys; ``open_wallet`` is replaced by one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .helpers import fixture_key

KING_1 = "cascade/ckpt-r1-king-120m@sha256:" + "1" * 64
KING_2 = "cascade/ckpt-r2-king-120m@sha256:" + "2" * 64
HOUSE_1 = {"digest": "a" * 64, "reveal_ts": 1_700_000_000}
HOUSE_2 = {"digest": "b" * 64, "reveal_ts": 1_700_010_000}


def jobs():
    import importlib

    return importlib.import_module("cascade.lane_d.ops_jobs")


def _run(argv: list[str]) -> int:
    try:
        return jobs().main(argv)
    except SystemExit as e:              # argparse usage error
        return int(e.code or 0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    signer = fixture_key("validator")
    import cascade.shared.chain as chain

    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: signer)
    d = {"tmp": tmp_path, "signer": signer, "king": tmp_path / "validator_state.db",
         "pointer": tmp_path / "lane-d" / "house_pointer.json",
         "source": tmp_path / "lane-d" / "house_source.json"}
    d["pointer"].parent.mkdir(parents=True)
    return d


def _king(env, pointer: str) -> None:
    env["king"].write_text(json.dumps({"king_hotkey": "5King", "king_pointer": pointer}))


def _source(env, king: str, house: dict) -> None:
    env["source"].write_text(json.dumps({"schema": "laned_house_source/1",
                                         "king_pointer": king, **house}))


def _argv(env, *extra: str) -> list[str]:
    return ["house-pointer", "--king-source", str(env["king"]), "--king-field", "king_pointer",
            "--pointer", str(env["pointer"]), "--signer-ss58", env["signer"].ss58_address,
            "--house-source", str(env["source"]), *extra]


def _load(env) -> dict:
    from cascade.validator.lane_d_house import load_house_pointer

    return load_house_pointer(env["pointer"], signer_ss58=env["signer"].ss58_address)


def test_first_sign_then_idempotent(env, capsys):
    _king(env, KING_1)
    _source(env, KING_1, HOUSE_1)
    assert _run(_argv(env)) == 0
    assert _load(env) == HOUSE_1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["action"] == "signed" and out["king_pointer"] == KING_1
    before = env["pointer"].read_bytes()
    assert _run(_argv(env)) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["action"] == "current"
    assert env["pointer"].read_bytes() == before, "an unchanged king re-signs nothing"


def test_dry_run_prints_the_payload_and_writes_nothing(env, capsys):
    from cascade.validator.lane_d_house import canonical_body

    _king(env, KING_1)
    _source(env, KING_1, HOUSE_1)
    assert _run(_argv(env, "--dry-run")) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["action"] == "planned"
    assert out["payload"] == {"schema": "lane-d-house/1", **HOUSE_1,
                              "signer": env["signer"].ss58_address}
    assert out["signed_bytes"] == canonical_body(out["payload"]).decode()
    assert not env["pointer"].exists()
    assert not list(env["pointer"].parent.glob("*.source.json"))


def test_king_change_blocks_until_the_owner_prepares_the_house(env, capsys):
    _king(env, KING_1)
    _source(env, KING_1, HOUSE_1)
    assert _run(_argv(env)) == 0
    before = env["pointer"].read_bytes()
    _king(env, KING_2)                                  # lane A dethrone / promotion
    assert _run(_argv(env)) == 3, "a stale house source is an owner step"
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["action"] == "blocked" and KING_2 in out["reason"]
    assert env["pointer"].read_bytes() == before
    env["source"].unlink()
    assert _run(_argv(env)) == 3
    _source(env, KING_2, HOUSE_2)
    assert _run(_argv(env)) == 0
    assert _load(env) == HOUSE_2
    sidecar = json.loads(Path(str(env["pointer"]) + ".source.json").read_text())
    assert sidecar["king_pointer"] == KING_2 and sidecar["digest"] == HOUSE_2["digest"]


def test_no_lane_a_king_keeps_the_previous_house(env, capsys):
    _king(env, KING_1)
    _source(env, KING_1, HOUSE_1)
    assert _run(_argv(env)) == 0
    before = env["pointer"].read_bytes()
    _king(env, "")
    assert _run(_argv(env)) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["action"] == "kept"
    assert env["pointer"].read_bytes() == before


def test_fail_closed_inputs(env, capsys, monkeypatch):
    _king(env, KING_1)
    _source(env, KING_1, {"digest": "NOT-HEX", "reveal_ts": 1})
    assert _run(_argv(env)) == 1
    _source(env, KING_1, HOUSE_1)
    env["king"].write_text("{broken")
    assert _run(_argv(env)) == 1
    _king(env, KING_1)
    # A pointer the job cannot verify is never overwritten.
    env["pointer"].write_text('{"schema": "lane-d-house/1"}')
    assert _run(_argv(env)) == 1
    assert env["pointer"].read_text() == '{"schema": "lane-d-house/1"}'
    env["pointer"].unlink()
    # The wallet hotkey must be the pinned signer.
    import cascade.shared.chain as chain

    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: fixture_key("intruder"))
    assert _run(_argv(env)) == 1
    assert not env["pointer"].exists()
    assert _run(["house-pointer"]) == 2


def test_log_lines_carry_the_tags(env, tmp_path):
    log = tmp_path / "jobs.log"
    _king(env, KING_1)
    _source(env, KING_1, HOUSE_1)
    assert _run(_argv(env, "--dry-run", "--log-file", str(log))) == 0
    assert _run(_argv(env, "--log-file", str(log))) == 0
    _king(env, KING_2)
    assert _run(_argv(env, "--log-file", str(log))) == 3
    text = log.read_text()
    for tag in ("lane_d_ops_house_resign_planned", "lane_d_ops_house_resigned",
                "lane_d_ops_house_resign_blocked"):
        assert tag in text, tag
    assert " ERROR cascade.lane_d.ops: lane_d_ops_house_resign_blocked " in text
