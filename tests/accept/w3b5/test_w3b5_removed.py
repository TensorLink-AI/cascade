"""W3b5 is REMOVED (plan rev6.18 A7, owner 2026-11-23 DECIDED: "The
low-frequency check A10 is removed."). Correction of the W3b5 suite: the
check's code, its report schema block, its audit rule and its config key are
deleted, and these tests pin the removal. Contract: README.md.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = REPO / "cascade" / "validator" / "schemas" / "lane_d_report_1.json"


def test_the_lowfreq_module_is_deleted():
    assert importlib.util.find_spec("cascade.validator.lane_d_lowfreq") is None, (
        "cascade/validator/lane_d_lowfreq.py (the A10 check) still exists")


def test_the_settlement_runs_no_lowfreq_check():
    from cascade.validator import lane_d

    src = inspect.getsource(lane_d)
    assert "lowfreq" not in src, "cascade/validator/lane_d.py still runs or records A10"
    assert "lowfreq_regression" not in src


def test_the_report_schema_has_no_lowfreq_block_or_reason():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    defs = schema["$defs"]
    assert not {"lowfreq", "lowfreq_lcbs", "lowfreq_result"} & set(defs), (
        "the lane-d-report/1 schema still defines the A10 lowfreq block")
    assert "lowfreq" not in defs["verdict"]["properties"]
    assert "lowfreq" not in defs["verdict"]["required"]
    assert "lowfreq_regression" not in defs["entry_reason"]["enum"]
    assert "lowfreq_freqs" not in defs["params"]["properties"]
    assert "lowfreq_freqs" not in defs["params"]["required"]


def test_the_report_validator_has_no_lowfreq_rule():
    from cascade.validator import lane_d_report as r

    assert not hasattr(r, "LOWFREQ_RESULTS"), "lane_d_report still exports LOWFREQ_RESULTS"
    assert not hasattr(r, "_check_lowfreq"), "lane_d_report still checks verdicts[].lowfreq"
    assert "lowfreq_regression" not in r.ENTRY_REASONS


def test_the_audit_has_no_lowfreq_rule():
    from cascade.audit import lane_d, lane_d_chain

    assert not hasattr(lane_d, "_lowfreq_problem"), "the audit still re-derives A10"
    for mod in (lane_d, lane_d_chain):
        src = inspect.getsource(mod)
        assert "lowfreq" not in src, f"{mod.__name__} still checks the A10 block"


def test_the_lowfreq_freqs_key_is_retired():
    from cascade.shared.config import RETIRED_KEYS
    from cascade.shared.config.lane_d import parse_lane_d

    ld = parse_lane_d({"lowfreq_freqs": ["D"]})
    assert not hasattr(ld, "lowfreq_freqs"), "LaneDConfig still parses lowfreq_freqs"
    assert "lowfreq_freqs" in (RETIRED_KEYS.get("lane_d") or ()), (
        "RETIRED_KEYS['lane_d'] does not list lowfreq_freqs (a deployed file still carries it)")
    for name in ("chain.toml", "chain.testnet.toml"):
        assert "lowfreq_freqs" not in (REPO / name).read_text(encoding="utf-8"), name
