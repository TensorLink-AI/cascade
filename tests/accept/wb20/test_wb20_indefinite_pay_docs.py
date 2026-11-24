"""WB20 (plan rev6.22, owner 2026-11-24): docs/MINER.md states the
indefinite lane B pay rule and multi-target legs, naming the real key, the
signed record fields and the CLI flags (each derived from the code)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MINER = REPO / "docs" / "MINER.md"


def _text() -> str:
    return MINER.read_text(encoding="utf-8")


def test_miner_md_names_the_indefinite_pay_key_and_its_default():
    from cascade.shared.config import LaneBConfig

    assert LaneBConfig().indefinite_pay_from_block == 0
    t = _text()
    assert "indefinite_pay_from_block" in t
    assert "no decay" in t and "28-era expiry" in t
    assert "tie or an inconclusive result leaves the holder" in t
    assert "empty slots still burn" in t


def test_miner_md_names_the_signed_record_fields():
    """The fields the validator stamps (and the audit re-derives) under the
    rule, as the producer writes them."""
    import inspect

    from cascade.validator import lane_b as vlb

    src = inspect.getsource(vlb)
    assert '"pay_rule"' in src and '"indefinite"' in src
    t = _text()
    assert 'pay_rule = "indefinite"' in t and "since_era" in t


def test_miner_md_documents_multi_target_funding():
    import argparse

    from cascade.miner.cli import build_parser

    def sub(parser, name):
        for act in parser._actions:
            if isinstance(act, argparse._SubParsersAction) and name in act.choices:
                return act.choices[name]
        raise AssertionError(name)

    fund = sub(sub(build_parser(), "lane-b"), "fund")
    assert fund._option_string_actions["--target"].__class__.__name__ == "_AppendAction"
    t = _text()
    assert "repeat `--target`" in t
    assert "every target that passes claims\nits own slot" in t
    assert "max_domains_per_leg" in t
