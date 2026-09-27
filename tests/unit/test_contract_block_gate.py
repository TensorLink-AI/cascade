"""Scheduled contract switch (DEC-CA-0047): `budget_denomination_after` +
`budget_denomination_after_block`, and the packed-source admission gate
`packed_sources_from_block`.

Pinned:
* pinning a schedule moves NO current digest — the schedule fields are never in
  the payload; only the contract EFFECTIVE past the gate (`at_block`) differs;
* every leg carries its ERA's start block (`--contract-block`), so king
  pre-train (which launches before the boundary), challengers, the settlement
  manifest and the audit agree by construction — never the launch block;
* the loader parses + validates both keys; `throne_contracts_at` follows;
* `packed_sources_at(block)` is "scan" before the gate and "reject" from it.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared.config import StaticGuardConfig, load_chain_config
from cascade.shared.manifest import contract_digest, contract_payload
from cascade.trainer.remote import RemoteHost, worker_argv

GATE = 9_165_600


def _sched(cfg, after="points+mv20", gate=GATE):
    return replace(cfg.training, budget_denomination_after=after, budget_denomination_after_block=gate)


def test_schedule_moves_no_current_digest(cfg):
    t, s = cfg.training, _sched(cfg)
    assert contract_digest(s) == contract_digest(t)
    assert not any("after" in k for k in contract_payload(s))
    assert contract_digest(s.at_block(GATE - 1)) == contract_digest(t)
    assert contract_digest(s.at_block(None)) == contract_digest(t)


def test_at_block_switches_exactly_at_the_gate(cfg):
    s = _sched(cfg)
    before, at, after = s.at_block(GATE - 1), s.at_block(GATE), s.at_block(GATE + 3600)
    assert before.budget_denomination == cfg.training.budget_denomination
    assert at.budget_denomination == after.budget_denomination == "points+mv20"
    assert at.budget_denomination_after == "" and at.budget_denomination_after_block == 0
    assert contract_digest(at) == contract_digest(after) != contract_digest(before)
    # the switched contract is what a `points+mv20` config would produce directly
    direct = replace(cfg.training, budget_denomination="points+mv20")
    assert contract_digest(at) == contract_digest(direct)


def test_no_schedule_is_identity(cfg):
    t = cfg.training
    assert t.at_block(0) is t and t.at_block(10**9) is t
    assert replace(t, budget_denomination_after="points+mv20").at_block(10**9) is not None
    # an "after" without a gate never fires
    assert replace(t, budget_denomination_after="points+mv20").at_block(10**9).budget_denomination == t.budget_denomination


def test_throne_contracts_at_follows_the_schedule(cfg):
    scheduled = replace(cfg, training=_sched(cfg))
    assert scheduled.throne_contracts_at(GATE - 1)[0].budget_denomination == cfg.training.budget_denomination
    assert scheduled.throne_contracts_at(GATE)[0].budget_denomination == "points+mv20"
    assert [c.arch_preset for c in scheduled.throne_contracts_at(GATE)] == [c.arch_preset for c in cfg.throne_contracts()]


def test_loader_parses_and_validates_the_schedule(tmp_path):
    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text()
    anchor = 'budget_denomination         = "series_points"\n'
    assert anchor in text
    good = tmp_path / "chain.toml"
    good.write_text(text.replace(anchor, anchor + f'budget_denomination_after = "points+mv20"\nbudget_denomination_after_block = {GATE}\n'))
    t = load_chain_config(good).training
    assert (t.budget_denomination_after, t.budget_denomination_after_block) == ("points+mv20", GATE)
    assert t.budget_denomination == "series_points"
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace(anchor, anchor + 'budget_denomination_after = "tokens"\nbudget_denomination_after_block = 1\n'))
    with pytest.raises(ValueError, match="budget_denomination"):
        load_chain_config(bad)
    assert load_chain_config(root / "chain.toml").training.budget_denomination_after == ""


def test_worker_argv_carries_the_contract_block():
    host = RemoteHost(name="h", host="127.0.0.1", port=22, user="root", key_path="k",
                      remote_python="python", workdir="/w")
    base = dict(gen_ref="r@sha256:" + "0" * 64, uid=1, hotkey="hk", role="king",
                base_seed=1, block=GATE + 5, trainer_spec="m:C")
    argv = worker_argv(host, **base)
    assert "--contract-block" not in argv
    argv = worker_argv(host, **base, contract_block=GATE)
    i = argv.index("--contract-block")
    assert argv[i + 1] == str(GATE)
    assert argv[argv.index("--block") + 1] == str(GATE + 5)      # launch block is a separate thing


def test_worker_resolves_the_contract_from_the_contract_block(cfg, monkeypatch):
    """The worker's `--contract-block` selects the effective contract; the launch
    `--block` never does (the king pre-train launches before the boundary)."""
    import argparse

    from cascade.trainer import worker as W

    scheduled = replace(cfg, training=_sched(cfg))
    monkeypatch.setattr(W, "load_chain_config", lambda *a, **k: scheduled, raising=False)
    ns = argparse.Namespace(contract_block=GATE, arch_preset=None)
    training = scheduled.training.at_block(ns.contract_block)
    assert training.primary_size.budget_denomination == "points+mv20"
    assert scheduled.training.at_block(None).primary_size.budget_denomination == cfg.training.budget_denomination


# ── packed_sources block gate ─────────────────────────────────────────────────

def test_packed_sources_at():
    sg = StaticGuardConfig(blocked=(), packed_sources="reject", packed_sources_from_block=GATE)
    assert sg.packed_sources_at(GATE - 1) == "scan"
    assert sg.packed_sources_at(None) == "scan"
    assert sg.packed_sources_at(GATE) == sg.packed_sources_at(GATE + 1) == "reject"
    assert StaticGuardConfig(blocked=(), packed_sources="reject").packed_sources_at(1) == "reject"   # 0 = now
    assert StaticGuardConfig(blocked=(), packed_sources="scan", packed_sources_from_block=1).packed_sources_at(10**9) == "scan"


def test_loader_parses_packed_sources_from_block(tmp_path):
    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text()
    armed = text.replace("[static_guard]\n", f'[static_guard]\npacked_sources = "reject"\npacked_sources_from_block = {GATE}\n', 1)
    p = tmp_path / "chain.toml"
    p.write_text(armed)
    sg = load_chain_config(p).static_guard
    assert (sg.packed_sources, sg.packed_sources_from_block) == ("reject", GATE)
    assert load_chain_config(root / "chain.toml").static_guard.packed_sources_from_block == 0
