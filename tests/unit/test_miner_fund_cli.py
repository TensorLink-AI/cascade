"""`cascade fund`: signed headers verify end-to-end against the intake."""

from __future__ import annotations

import argparse
import hashlib

from cascade.funding.intake import FundingIntake
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault
from cascade.miner.cli import (
    _cmd_fund,
    _decode_json_body,
    _intake_transport_ok,
    build_fund_headers,
)

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "cascade-gen-abc@sha256:" + "c" * 64


def _sign(msg: bytes) -> bytes:
    """Deterministic stand-in for the hotkey's sr25519 signer."""
    return hashlib.sha256(HK.encode() + msg).digest()


def _verify(hotkey: str, msg: bytes, sig_hex: str) -> bool:
    return sig_hex == hashlib.sha256(hotkey.encode() + msg).hexdigest()


def test_cli_headers_verify_against_intake(tmp_path):
    """The CLI and the intake agree byte-for-byte on the canonical message."""
    clock = lambda: 1_000_000.0  # noqa: E731
    intake = FundingIntake(
        FundedQueue(tmp_path / "q.json", clock=clock),
        PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: 42,
        require_signature=True,
        verify=_verify,
        clock=clock,
    )
    headers = build_fund_headers("fund", HK, REF, "sk-live", _sign, now=clock)
    status, body = intake.fund(headers)
    assert (status, body["status"]) == (202, "queued")
    assert intake.vault.get(HK) == "sk-live"
    # A fund signature can never authorize a withdraw (action is in the message).
    status, body = intake.withdraw(headers)
    assert (status, body["code"]) == (400, "bad_signature")
    headers_w = build_fund_headers("withdraw", HK, REF, "", _sign, now=clock)
    status, body = intake.withdraw(headers_w)
    assert (status, body["status"]) == (200, "withdrawn")


def test_fund_headers_omit_key_on_withdraw():
    h = build_fund_headers("withdraw", HK, REF, "sk-should-not-appear", _sign,
                           now=lambda: 0.0)
    assert "X-Lium-Api-Key" not in h


def _args(**kw) -> argparse.Namespace:
    base = {"intake_url": "https://intake.example", "ref": REF,
            "wallet_name": "w", "wallet_hotkey": "h", "wallet_path": None,
            "lium_key_env": "TEST_LIUM_KEY", "withdraw": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_fund_refuses_missing_env_key(monkeypatch, capsys):
    monkeypatch.delenv("TEST_LIUM_KEY", raising=False)
    assert _cmd_fund(_args()) == 2
    err = capsys.readouterr().err
    assert "TEST_LIUM_KEY" in err and "never pass the key" in err


def test_cmd_fund_refuses_plain_http_remote(monkeypatch, capsys):
    monkeypatch.setenv("TEST_LIUM_KEY", "sk")
    assert _cmd_fund(_args(intake_url="http://intake.example")) == 2
    assert "https" in capsys.readouterr().err
    # Withdraw needs no key, but the transport guard still applies first.
    monkeypatch.delenv("TEST_LIUM_KEY", raising=False)
    assert _cmd_fund(_args(intake_url="http://intake.example", withdraw=True)) == 2


def test_transport_guard_parses_hostname_not_substrings():
    # The bypass class the audit flagged: hostnames that merely CONTAIN a
    # local name must not pass.
    assert not _intake_transport_ok("http://localhost.evil.example")
    assert not _intake_transport_ok("http://127.0.0.1.attacker.example:80")
    assert not _intake_transport_ok("ftp://localhost")
    assert _intake_transport_ok("http://localhost:8790")
    assert _intake_transport_ok("http://127.0.0.1:8790")
    assert _intake_transport_ok("https://intake.example")


def test_decode_json_body_survives_proxy_html():
    assert _decode_json_body(b'{"code": "x", "message": "y"}') == {
        "code": "x", "message": "y"}
    body = _decode_json_body(b"<html><body>502 Bad Gateway</body></html>")
    assert body["code"] == "non_json_response" and "502" in body["message"]
    assert _decode_json_body(b"") == {}


def test_cascade_queue_renders_roster_and_marks_you(capsys, monkeypatch, tmp_path):
    """`cascade queue` prints the published roster (sku, capacities, seniority
    order, outcomes) and marks the caller's rows — the miner-side half of the
    funded transparency contract (the audit's funded-roster check is the
    other)."""
    import argparse

    from cascade.miner import cli as cli_mod

    doc = {"round_id": "77",
           "admission": {"cap": 8, "configured_cap": 8, "market_capacity": 17,
                         "reserve": 1, "sku": "RTX4090",
                         "sku_capacities": {"RTX4090": 17, "A6000": 1}},
           "seated": [{"hotkey": "hkME", "ref": "r", "reveal_block": 10}],
           "waiting": [{"hotkey": "hkW", "reveal_block": 20}],
           "terminal": [{"hotkey": "hkBad", "error_class": "auth"}],
           "outcomes": [{"hotkey": "hkME", "outcome": "trained"}]}
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: doc)
    monkeypatch.setattr(cli_mod, "load_chain_config",
                        lambda p: type("C", (), {"storage": None})())
    rc = cli_mod._cmd_queue(argparse.Namespace(
        chain_toml=None, round=None, intake=None, hotkey="hkME"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "RTX4090=17" in out and "A6000=1" in out
    assert "hkME  reveal=10  ← you" in out
    assert "hkBad  [auth]" in out
    assert "cascade-audit round" in out
