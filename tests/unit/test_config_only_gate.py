"""Block-gated config_only dedup enforcement (``[round]
dedup_config_only_enforce`` + ``dedup_config_only_from_block``).

With the block set, identical-code / different-config entrants are only
DROPPED in rounds whose epoch boundary is at/after it; before it the tier
stays shadow-logged exactly as with the flag off.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

SOURCE = "\n".join(f"v{i} = {i} * 3 + 1" for i in range(200)) + "\n"


def _repo(root, name, config):
    d = root / name
    d.mkdir(parents=True)
    (d / "generator.py").write_text(SOURCE)
    (d / "requirements.txt").write_text("numpy\n")
    (d / "config.json").write_text(json.dumps(config))
    return d


@pytest.fixture()
def gated_runner(cfg, tmp_path, monkeypatch):
    repos = {}

    def add(name, uid, config, reveal_block):
        ref = f"{name}/gen@sha256:{name[0] * 64}"
        repos[ref] = _repo(tmp_path / "repos", name, config)
        return ResolvedGenerator(hotkey=name, uid=uid, ref=ref, reveal_block=reveal_block)

    class _Logs:
        def put_text(self, key, text, **kw):
            pass

    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: repos[str(ref)])
    monkeypatch.setattr(TrainerRunner, "logs_store", lambda self: _Logs())
    monkeypatch.setattr(TrainerRunner, "hub", lambda self: None)

    def make(**round_kw):
        kw = dict(dedup_mode="enforce", dedup_probe_mode="off", dedup_probe_series=0,
                  dedup_config_only_enforce=True, dedup_config_only_from_block=1000)
        kw.update(round_kw)
        rc = replace(cfg.round, **kw)
        return TrainerRunner(cfg=replace(cfg, round=rc), base_trainer=object(),
                             work_root=tmp_path / "work", use_sandbox=False)

    return make, add


def test_config_only_gate_parses_and_resolves(tmp_path):
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    bare = re.sub(r"^dedup_config_only_(enforce|from_block)\s*=.*$", "",
                  DEFAULT_CHAIN_TOML.read_text(), flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(bare)
    cfg = load_chain_config(p)
    assert cfg.round.dedup_config_only_enforce is False
    assert cfg.round.dedup_config_only_from_block == 0
    assert not cfg.round.config_only_enforced(10**9)

    p.write_text(bare.replace(
        "\n[round]\n",
        "\n[round]\ndedup_config_only_enforce = true\n"
        "dedup_config_only_from_block = 5000\n", 1))
    cfg = load_chain_config(p)
    assert cfg.round.config_only_enforced(5000) and cfg.round.config_only_enforced(9000)
    assert not cfg.round.config_only_enforced(4999)
    assert not cfg.round.config_only_enforced(None)      # unknown block: shadow
    ungated = replace(cfg.round, dedup_config_only_from_block=0)
    assert ungated.config_only_enforced(None) and ungated.config_only_enforced(1)

    shipped = load_chain_config(DEFAULT_CHAIN_TOML)
    if shipped.round.dedup_config_only_from_block:
        assert shipped.round.dedup_config_only_enforce is True
        assert shipped.round.dedup_config_only_from_block == shipped.round.duel_from_block


def test_config_only_drops_only_from_the_gate_block(gated_runner):
    make, add = gated_runner
    runner = make()
    a = add("alice", 3, {"w": 1.0}, reveal_block=10)
    b = add("bob", 9, {"w": 1.2}, reveal_block=11)      # same code, different config

    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=1, block=999)
    assert [c.hotkey for c in kept] == ["alice", "bob"]
    report = json.loads((runner.work_root / "1" / "dedup_report.json").read_text())
    assert report["config_only_enforce"] is False
    assert [v["tier"] for v in report["shadow"]] == ["config_only"]

    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=2, block=1000)
    assert [c.hotkey for c in kept] == ["alice"]        # earliest reveal keeps the slot
    report = json.loads((runner.work_root / "2" / "dedup_report.json").read_text())
    assert report["config_only_enforce"] is True
    assert [d["hotkey"] for d in report["dropped"]] == ["bob"]

    # no block known (an ad-hoc caller) under a set gate: shadow
    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=3)
    assert [c.hotkey for c in kept] == ["alice", "bob"]


def test_config_only_without_a_gate_drops_immediately(gated_runner):
    make, add = gated_runner
    runner = make(dedup_config_only_from_block=0)
    a = add("alice", 3, {"w": 1.0}, reveal_block=10)
    b = add("bob", 9, {"w": 1.2}, reveal_block=11)
    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=4)
    assert [c.hotkey for c in kept] == ["alice"]
