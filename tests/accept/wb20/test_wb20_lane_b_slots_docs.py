"""WB20 (plan rev6.22, owner 2026-11-24): docs/MINER.md states the lane B
slot rule — full pay until replaced (no decay, no expiry), every slot
tested in two families (declared at ``gain_margin``, undeclared at
``undeclared_gain_margin``), one miner may hold every slot — naming the real
keys and record fields (each read from the code)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MINER = REPO / "docs" / "MINER.md"


def _text() -> str:
    return " ".join(MINER.read_text(encoding="utf-8").split())


def test_miner_md_states_full_pay_until_replaced():
    from cascade.shared.config import LaneBConfig

    assert not hasattr(LaneBConfig(), "win_ttl_eras")
    t = _text()
    assert "100% of its slot every settlement, with no decay and no expiry" in t
    assert "A tie or an inconclusive result leaves the holder in place" in t
    assert "empty slots still burn" in t
    assert "since_era" in t


def test_miner_md_states_every_slot_two_families_and_bars():
    from cascade.shared.config import LaneBConfig

    c = LaneBConfig()
    t = _text()
    assert "tested on every slot" in t
    assert f"up to `[lane_b] max_domains_per_leg`, {c.max_domains_per_leg}" in t
    assert f"`gain_margin` ({c.gain_margin:.0%})" in t
    assert f"`undeclared_gain_margin` ({c.undeclared_gain_margin:.0%})" in t
    assert "max-T" in t and "one miner may hold every slot" in t
    assert "CRPS **or** MASE" in t


def test_miner_md_documents_multi_target_funding():
    import argparse

    from cascade.miner.cli import build_parser

    def sub(parser, name):
        for act in parser._actions:
            if isinstance(act, argparse._SubParsersAction) and name in act.choices:
                return act.choices[name]
        raise AssertionError(name)

    fund = sub(sub(build_parser(), "lane-b"), "fund")
    assert type(fund._option_string_actions["--target"]).__name__ == "_AppendAction"
    assert "repeat `--target`" in _text()
