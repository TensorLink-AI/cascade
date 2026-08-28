"""`cascade fund`: signed headers verify end-to-end against the intake."""

from __future__ import annotations

import argparse
import hashlib

from cascade.funding.intake import FundingIntake
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault
from cascade.miner.cli import _cmd_fund, build_fund_headers

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
