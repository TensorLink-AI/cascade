"""`cascade fetch` target resolution — king / uid / hotkey / raw ref, and the
error paths, over a fake chain client (no network)."""

from __future__ import annotations

import pytest

from cascade.miner import cli
from cascade.shared.chain import Commitment

GEN = "cascade/testnet-smoothgp@sha256:" + "b2" * 32
CHAL = "iris999/ares-v3@sha256:" + "01" * 32


class _FakeClient:
    def __init__(self, commitments, king_hotkey):
        self._c = commitments
        self._king = king_hotkey

    def poll_commitments(self):
        return self._c

    def highest_incentive_hotkey(self):
        return self._king


def _commits():
    return [
        Commitment(uid=3, hotkey="5King", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{GEN}", commit_block=100),
        Commitment(uid=13, hotkey="5Chal", coldkey=None,
                   payload=f"metro-v1:gen:hippius:{CHAL}", commit_block=110),
    ]


@pytest.fixture()
def patched(monkeypatch, cfg):
    client = _FakeClient(_commits(), king_hotkey="5King")
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    from cascade.shared.chain import ChainClient
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: client))
    return cfg, client


def test_raw_ref_skips_chain(cfg):
    # a repo@digest resolves without any chain call
    ref, label = cli._resolve_fetch_ref(GEN, cfg, "test")
    assert ref == GEN and label == "cascade-testnet-smoothgp"


def test_resolve_king(patched):
    cfg, _ = patched
    ref, label = cli._resolve_fetch_ref("king", cfg, "test")
    assert ref == GEN and label == "king-uid3"


def test_resolve_by_uid(patched):
    cfg, _ = patched
    ref, label = cli._resolve_fetch_ref("13", cfg, "test")
    assert ref == CHAL and label == "uid13"


def test_resolve_by_hotkey(patched):
    cfg, _ = patched
    ref, label = cli._resolve_fetch_ref("5Chal", cfg, "test")
    assert ref == CHAL


def test_unknown_uid_errors(patched):
    cfg, _ = patched
    with pytest.raises(ValueError, match="uid 99 has no committed"):
        cli._resolve_fetch_ref("99", cfg, "test")


def test_unknown_hotkey_errors(patched):
    cfg, _ = patched
    with pytest.raises(ValueError, match="no committed generator"):
        cli._resolve_fetch_ref("5Ghost", cfg, "test")


def test_vacant_throne_errors(monkeypatch, cfg):
    client = _FakeClient(_commits(), king_hotkey=None)
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    from cascade.shared.chain import ChainClient
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: client))
    with pytest.raises(ValueError, match="no king"):
        cli._resolve_fetch_ref("king", cfg, "test")


def test_king_without_commitment_errors(monkeypatch, cfg):
    # king hotkey exists on metagraph but committed nothing this round
    client = _FakeClient([_commits()[1]], king_hotkey="5King")
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    from cascade.shared.chain import ChainClient
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: client))
    with pytest.raises(ValueError, match="no committed generator this round"):
        cli._resolve_fetch_ref("king", cfg, "test")


def test_fetch_cmd_downloads_and_reports(patched, monkeypatch, tmp_path, capsys):
    # end-to-end _cmd_fetch with the registry fetch stubbed
    cfg, _ = patched
    import cascade.shared.hippius as hip

    def fake_fetch(ref, out, hub):
        from pathlib import Path
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "generator.py").write_text("x")
        (d / "config.json").write_text("{}")
        return d

    monkeypatch.setattr(hip, "fetch_from_hub", fake_fetch)
    import types
    args = types.SimpleNamespace(target="king", out=tmp_path / "k", chain_toml=None,
                                 network="test", verify=False)
    rc = cli._cmd_fetch(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "fetched king-uid3" in out and "generator.py" in out
