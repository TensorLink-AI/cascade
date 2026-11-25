"""W-LAUNCH miner CLI: lane A is closed to new challengers from 9586800.

The intake refuses a new lane A fund (409 ``lane_ended``) and the trainer seats
no new challenger (#941). The CLI says so in its help, and a lane A ``deploy`` /
``submit`` refuses client-side BEFORE any upload, intake POST or chain commit
once the head has reached the cutoff (``--allow-closed-lane-a`` overrides).
Lane B deploys (``--expert-domain``) and a store-only submit stay open.
"""

from __future__ import annotations

import json

import pytest

from .helpers import MAINNET_LAUNCH, shipped_cfg

REF = "me/gen@sha256:" + "a" * 64


@pytest.fixture
def chain(monkeypatch):
    """A fake chain at a settable head; records commits and uploads."""
    from cascade.miner import cli
    from cascade.shared import chain as chain_mod

    state = {"head": MAINNET_LAUNCH, "commits": [], "uploads": [], "posts": []}

    class _Client:
        def current_block(self):
            return state["head"]

        def commit_submission(self, payload, *, blocks_until_reveal):
            state["commits"].append(payload)

        def wallet(self):
            raise RuntimeError("no wallet in tests")

    monkeypatch.setattr(chain_mod.ChainClient, "from_config",
                        classmethod(lambda cls, *a, **k: _Client()))
    monkeypatch.setattr(cli, "_live_cfg", lambda cfg, client: cfg)

    def _upload(args, cfg):
        state["uploads"].append(args.repo_dir)
        return 0, REF

    monkeypatch.setattr(cli, "_upload_generator", _upload)
    return state


def _said(capsys):
    got = capsys.readouterr()
    return got.out + got.err


def _deploy(tmp_path, *extra):
    return ["deploy", str(tmp_path), "--wallet-name", "w", "--wallet-hotkey", "h",
            "--hub-repo", "me/gen", "--skip-verify", "--blocks-until-reveal", "5", *extra]


def test_cutoff_resolves_per_network():
    from cascade.miner.cli import lane_a_closed_from

    assert lane_a_closed_from(shipped_cfg("chain.toml")) == MAINNET_LAUNCH
    assert lane_a_closed_from(shipped_cfg("chain.testnet.toml")) == 0


def test_lane_a_deploy_refuses_past_the_cutoff_before_upload(chain, tmp_path, capsys):
    from cascade.miner import cli

    rc = cli.main(["--json", *_deploy(tmp_path)])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["error"]["code"] == "lane_a_closed"
    assert doc["error"]["until_block"] == MAINNET_LAUNCH
    assert chain["uploads"] == [] and chain["commits"] == []

    # a lane B expert deploy is not lane A: it still commits
    assert cli.main([*_deploy(tmp_path), "--expert-domain", "energy"]) == 0
    assert len(chain["commits"]) == 1

    # before the cutoff a lane A deploy goes ahead (with a note)
    chain["head"] = MAINNET_LAUNCH - 100
    assert cli.main(_deploy(tmp_path)) == 0
    assert "takes no new challengers from block 9586800" in _said(capsys)
    assert len(chain["commits"]) == 2


def test_allow_closed_lane_a_overrides(chain, tmp_path, capsys):
    from cascade.miner import cli

    assert cli.main([*_deploy(tmp_path), "--allow-closed-lane-a"]) == 0
    assert len(chain["commits"]) == 1 and "will not be seated" in _said(capsys)


def test_lane_a_submit_refuses_before_posting_store_only_stays_open(chain, tmp_path, monkeypatch, capsys):
    from cascade.funding import providers
    from cascade.miner import cli

    def _post(req, out, timeout):
        chain["posts"].append(req.full_url)
        return 200, {"ref": REF, "commit_payload": "metro-v1:x", "funding": "none"}

    monkeypatch.setattr(cli, "post_intake", _post)
    monkeypatch.setattr(cli, "load_signing_wallet", lambda args, out: ("5Hk", lambda m: b"sig"))
    monkeypatch.setattr(providers, "credentials_from_env", lambda args: "lium-test-key")
    (tmp_path / "generator.py").write_text("x = 1\n")
    base = ["submit", str(tmp_path), "https://intake.example", "--wallet-name", "w",
            "--wallet-hotkey", "h", "--skip-verify", "--yes"]

    rc = cli.main(["--json", *base])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["error"]["code"] == "lane_a_closed"
    assert chain["posts"] == [] and chain["commits"] == []

    # store only (no key sent, nothing committed): lane B legs may name the vault ref
    monkeypatch.setattr(providers, "credentials_from_env", lambda args: "")
    assert cli.main([*base, "--no-fund", "--no-commit"]) == 0
    assert chain["posts"] == ["https://intake.example/v1/submit"] and chain["commits"] == []


@pytest.mark.parametrize("path", [(), ("deploy",), ("submit",), ("fund",)])
def test_help_says_lane_a_is_closed(path, capsys):
    from cascade.miner import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*path, "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Lane A is closed to new challengers from block 9586800" in text
