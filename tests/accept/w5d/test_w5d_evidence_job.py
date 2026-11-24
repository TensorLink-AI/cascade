"""W5d recurring job: install a re-measured ``laned_readiness/1`` (W1 forge mode,
``--readiness-out``) and/or a ``laned_t7/1`` (the T7 pod run, an owner step)
at the paths the validator's ``--lane-d-inputs`` TOML names (plan rev6.6, §1.6;
OPERATOR "S8: long context").

``python -P -m cascade.lane_d.ops_jobs evidence`` validates with the scorer's
own ``armed_record`` (the gate the validator applies), refuses a non-pod T7 and
any install that would disarm an armed context (unless ``--allow-disarm``),
is a no-op when the pinned bytes already match, and writes nothing under
``--dry-run``. Shipped ``chain.testnet.toml``; no torch. Every import of the job
is inside a test body (a base without it fails on a missing module in W5d's
PATHS).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CHAIN = REPO / "chain.testnet.toml"


def _run(argv: list[str]) -> int:
    import importlib

    try:
        return importlib.import_module("cascade.lane_d.ops_jobs").main(argv)
    except SystemExit as e:
        return int(e.code or 0)


def _readiness(series: int, sources: int, contexts=(4096, 8192, 16384)) -> bytes:
    return json.dumps({"schema": "laned_readiness/1", "rungs": [
        {"context": c, "series": series, "sources": sources} for c in contexts]}).encode()


def _t7(variants=(4096, 8192, 16384), *, source="pod", wall_s=600.0) -> bytes:
    from cascade.eval.lane_d_t7 import build_evidence, settlement_budget, variant_row

    budget = settlement_budget(24.0)
    rows = [variant_row(host="testnet-eval", gpu="RTX 4090", variant=v, input_len=v,
                        history_min=v, history_max=v + 64, wall_s=wall_s, peak_mem_gb=8.0,
                        peak_mem_source="cuda_max_allocated", budget=budget, n_models=10,
                        n_windows=100) for v in variants]
    doc = build_evidence(rows, short_window={"base_context": 4096, "declared_context": 131072,
                                             "n_windows": 10, "max_abs_diff": 0.0},
                         tool={"name": "fixture"}, torch_version="2.4.1+cu124", source=source)
    return json.dumps(doc).encode()


@pytest.fixture
def paths(tmp_path):
    stage, pinned = tmp_path / "stage", tmp_path / "pinned"
    stage.mkdir()
    pinned.mkdir()
    return {"new_r": stage / "laned_readiness.json", "new_t": stage / "t7.json",
            "pin_r": pinned / "laned_readiness.json", "pin_t": pinned / "t7.json"}


def _argv(p, *extra, readiness=True, t7=False):
    a = ["evidence", "--chain-toml", str(CHAIN), "--pinned-readiness", str(p["pin_r"]),
         "--pinned-t7", str(p["pin_t"])]
    if readiness:
        a += ["--readiness", str(p["new_r"])]
    if t7:
        a += ["--t7", str(p["new_t"])]
    return [*a, *extra]


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_install_then_current(paths, capsys):
    paths["pin_t"].write_bytes(_t7())
    paths["new_r"].write_bytes(_readiness(100_000, 50))
    assert _run(_argv(paths, "--dry-run")) == 0
    out = _last(capsys)
    assert out["action"] == "planned" and not paths["pin_r"].exists()
    assert out["armed_before"] == [] and out["armed_after"]
    assert _run(_argv(paths)) == 0
    out = _last(capsys)
    assert out["action"] == "installed" and paths["pin_r"].read_bytes() == paths["new_r"].read_bytes()
    assert _run(_argv(paths)) == 0
    assert _last(capsys)["action"] == "current"


def test_t7_must_be_pod_evidence(paths, capsys):
    paths["new_r"].write_bytes(_readiness(100_000, 50))
    paths["new_t"].write_bytes(_t7(source="dev-cpu"))
    assert _run(_argv(paths, t7=True)) == 1
    assert _last(capsys)["action"] == "failed"
    assert not paths["pin_t"].exists() and not paths["pin_r"].exists()
    paths["new_t"].write_bytes(_t7())
    assert _run(_argv(paths, t7=True)) == 0
    assert paths["pin_t"].exists() and paths["pin_r"].exists()


def test_refuses_to_disarm_unless_allowed(paths, capsys):
    paths["pin_t"].write_bytes(_t7())
    paths["pin_r"].write_bytes(_readiness(100_000, 50))
    before = paths["pin_r"].read_bytes()
    paths["new_r"].write_bytes(_readiness(1, 1))           # a bad re-measure
    assert _run(_argv(paths)) == 1
    assert "disarm" in _last(capsys)["reason"]
    assert paths["pin_r"].read_bytes() == before
    assert _run(_argv(paths, "--allow-disarm")) == 0
    assert paths["pin_r"].read_bytes() == paths["new_r"].read_bytes()


def test_unparseable_document_is_refused(paths, capsys):
    paths["new_r"].write_bytes(b'{"schema": "laned_readiness/1", "rungs": [{"context": 5}]}')
    assert _run(_argv(paths)) == 1
    assert not paths["pin_r"].exists()
    assert _run(["evidence", "--chain-toml", str(CHAIN), "--pinned-readiness",
                 str(paths["pin_r"]), "--pinned-t7", str(paths["pin_t"])]) == 1
    assert _run(["evidence"]) == 2
