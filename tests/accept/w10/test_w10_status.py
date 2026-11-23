"""W10: `cascade lane-d status` reads the intake's public queue
(`GET /v1/lane-d/queue`, W3); no wallet, no chain write. Contract: README.md
"status"."""

from __future__ import annotations

import pytest

from .helpers import (
    PATH_IDS,
    QUEUE,
    FakeChain,
    block_external_network,
    cli_env,
    fixture_key,
    install_chain,
    intake,
    run_cli,
)

ROWS = [
    {"hotkey": "5Habc", "digest": "a" * 64, "state": "ready", "context": 4096,
     "coldkey_at_ready": "5Cold", "ready_block": 10, "ready_expires_block": 7210,
     "weights_sha256": "b" * 64, "licence": "Apache-2.0",
     "licence_attestation_sha256": "c" * 64},
    {"hotkey": "5Hdef", "digest": "d" * 64, "state": "submitted", "context": 8192,
     "coldkey_at_ready": None, "ready_block": None, "ready_expires_block": None,
     "weights_sha256": None, "licence": None, "licence_attestation_sha256": None},
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    cli_env(monkeypatch, tmp_path)
    chain = install_chain(monkeypatch, FakeChain(fixture_key("status")))
    return {"chain": chain, "blocked": block_external_network(monkeypatch)}


def test_status_shows_the_public_queue(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey="c") as fake:
        fake.queue_rows = ROWS
        res = run_cli(capsys, caplog, "lane-d", "status", fake.base)
        assert [r["path"] for r in fake.requests] == [QUEUE]
    assert res.rc == 0, res
    assert res.data["entries"] == ROWS


def test_status_filters_to_one_hotkey(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey="c") as fake:
        fake.queue_rows = ROWS
        res = run_cli(capsys, caplog, "lane-d", "status", fake.base, "--hotkey", "5Hdef")
    assert res.rc == 0, res
    assert res.data["entries"] == [ROWS[1]]


def test_status_never_builds_a_chain_client(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey="c") as fake:
        fake.queue_rows = ROWS
        run_cli(capsys, caplog, "lane-d", "status", fake.base)
    assert env["chain"].from_config_calls == [] and env["chain"].commits == []


def test_status_of_an_unreachable_intake_is_a_network_failure(env, capsys, caplog):
    res = run_cli(capsys, caplog, "lane-d", "status", "http://127.0.0.1:9")
    assert res.rc == 3, res


def test_status_of_a_failing_intake_is_a_network_failure(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey="c") as fake:
        fake.refuse[QUEUE] = (503, "state_unavailable")
        res = run_cli(capsys, caplog, "lane-d", "status", fake.base)
    assert res.rc == 3, res
