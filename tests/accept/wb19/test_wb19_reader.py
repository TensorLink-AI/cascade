"""WB19 → WB22/WB23: the trainer-side per-GPU step-rate table reader
(``cascade.funding.step_rates``). WB22's row: "trainer-side per-GPU step-rate
table from WB19"; D42b (a): ``steps = floor(train_hours × 3600 ×
measured_steps_per_s[gpu])``. Stdlib only, no torch. Contract: README.md
"The table reader".
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from .helpers import CHAIN_TOML, REPO, TABLE_SCHEMA, load_tool, make_pod_record, write_json

IMG = "sha256:" + "5d" * 32
ROW = {"gpu_name": "NVIDIA H100 80GB HBM3", "gpu_count": 8, "driver": "570.86.15",
       "cuda": "12.4", "steps_per_s": 2.2, "tokens_per_s": 2.2 * 512 * 4096,
       "overhead_seconds": 412.5, "peak_memory_bytes": 51_000_000_000,
       "memory_total_bytes": 85_899_345_920, "run_body_sha256": "ab" * 32}


def _table(**rows):
    return {"schema": TABLE_SCHEMA, "image_digest": IMG, "contract_digest": "cd" * 32,
            "contract_block": 9399600, "rows": rows or {"8xH100": dict(ROW)}}


def _sr():
    from cascade.funding import step_rates

    return step_rates


def test_reader_constants():
    sr = _sr()
    assert sr.TABLE_SCHEMA == TABLE_SCHEMA
    assert issubclass(sr.StepRateTableError, ValueError)


def test_load_bare_table(tmp_path):
    sr = _sr()
    p = write_json(tmp_path / "t.json", _table(**{"8xH100": dict(ROW),
                                                  "L40S": dict(ROW, steps_per_s=0.7)}))
    t = sr.load_step_rate_table(p)
    assert set(t) == {"8xH100", "L40S"}
    r = t["8xH100"]
    assert isinstance(r, sr.StepRate)
    assert r.sku == "8xH100" and r.gpu_name == ROW["gpu_name"] and r.gpu_count == 8
    assert r.steps_per_s == pytest.approx(2.2) and r.tokens_per_s == pytest.approx(ROW["tokens_per_s"])
    assert r.overhead_seconds == pytest.approx(412.5)
    assert r.peak_memory_bytes == ROW["peak_memory_bytes"]
    assert r.memory_total_bytes == ROW["memory_total_bytes"]
    assert t["L40S"].steps_per_s == pytest.approx(0.7)


def test_load_checks_image_pin(tmp_path):
    sr = _sr()
    p = write_json(tmp_path / "t.json", _table())
    assert "8xH100" in sr.load_step_rate_table(p, image_digest=IMG)
    with pytest.raises(sr.StepRateTableError) as ei:
        sr.load_step_rate_table(p, image_digest="sha256:" + "00" * 31 + "01")
    assert ei.value.code == "wrong_image"


@pytest.mark.parametrize("edit", [
    lambda t: t.update(schema="laned_wb19_table/0"),
    lambda t: t.pop("rows"),
    lambda t: t.update(rows={}),
    lambda t: t["rows"]["8xH100"].update(steps_per_s=0.0),
    lambda t: t["rows"]["8xH100"].update(steps_per_s=-1.0),
    lambda t: t["rows"]["8xH100"].update(steps_per_s=math.nan),
    lambda t: t["rows"]["8xH100"].update(steps_per_s=math.inf),
    lambda t: t["rows"]["8xH100"].update(steps_per_s="2.2"),
    lambda t: t["rows"]["8xH100"].update(overhead_seconds=-3.0),
    lambda t: t["rows"]["8xH100"].pop("overhead_seconds"),
    lambda t: t.update(image_digest="latest"),
])
def test_load_refuses_bad_table(tmp_path, edit):
    sr = _sr()
    t = _table()
    edit(t)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(t))          # allow_nan: a NaN reaches the reader as the token
    with pytest.raises(sr.StepRateTableError) as ei:
        sr.load_step_rate_table(p)
    assert ei.value.code == "bad_table"


def test_load_refuses_unparseable(tmp_path):
    sr = _sr()
    p = tmp_path / "t.json"
    p.write_text("{not json")
    with pytest.raises(sr.StepRateTableError) as ei:
        sr.load_step_rate_table(p)
    assert ei.value.code == "bad_table"


def test_load_evidence_requires_passed(tmp_path):
    sr = _sr()
    ev = {"schema": "laned_wb19/1", "wp": "WB19", "passed": True, "table": _table()}
    p = write_json(tmp_path / "wb19.json", ev)
    assert "8xH100" in sr.load_step_rate_table(p)
    ev["passed"] = False
    p = write_json(tmp_path / "wb19-failed.json", ev)
    with pytest.raises(sr.StepRateTableError) as ei:
        sr.load_step_rate_table(p)
    assert ei.value.code == "not_passed"


def test_rate_for_is_case_insensitive(tmp_path):
    sr = _sr()
    t = sr.load_step_rate_table(write_json(tmp_path / "t.json", _table()))
    assert sr.rate_for(t, "8xh100") is t["8xH100"]
    assert sr.rate_for(t, " 8XH100 ") is t["8xH100"]
    assert sr.rate_for(t, "4xH100") is None


@pytest.mark.parametrize("hours,want", [(3.0, math.floor(3 * 3600 * 2.2)),
                                        (0.5, math.floor(0.5 * 3600 * 2.2)),
                                        (6.0, math.floor(6 * 3600 * 2.2))])
def test_steps_for_hours_is_d42b_a(tmp_path, hours, want):
    sr = _sr()
    t = sr.load_step_rate_table(write_json(tmp_path / "t.json", _table()))
    got = sr.steps_for_hours(t["8xH100"], hours)
    assert type(got) is int and got == want


@pytest.mark.parametrize("hours", [0.0, -1.0, math.nan, math.inf])
def test_steps_for_hours_refuses_bad_hours(tmp_path, hours):
    sr = _sr()
    t = sr.load_step_rate_table(write_json(tmp_path / "t.json", _table()))
    with pytest.raises(ValueError):
        sr.steps_for_hours(t["8xH100"], hours)


def test_reader_imports_no_torch():
    code = ("import sys; import cascade.funding.step_rates as m; "
            "print('torch' in sys.modules)")
    p = subprocess.run([sys.executable, "-P", "-c", code], cwd=REPO, capture_output=True,
                       text=True, timeout=120, env={"PYTHONPATH": str(REPO), "PATH": ""})
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False", "the trainer-side reader must stay torch-free"


def test_reader_reads_combine_output(tmp_path_factory):
    """WB22 reads exactly what WB19's `combine` writes."""
    from .helpers import chain_cfg, pinned_image, run_tool

    tool = load_tool()
    sr = _sr()
    skus = list(chain_cfg().lane_b.funded_pod_skus)
    recs = [make_pod_record(tool, tmp_path_factory.mktemp(f"r-{s}"), None, s) for s in skus]
    recs.append(make_pod_record(tool, tmp_path_factory.mktemp("r-full"), None, "8xH100",
                                mode="full"))
    d = tmp_path_factory.mktemp("combine")
    files = [write_json(d / f"run{i}.json", r) for i, r in enumerate(recs)]
    out = d / "wb19.json"
    rc, _, _ = run_tool(tool, ["combine", "--chain-toml", CHAIN_TOML, "--runs", *files,
                               "--out", out], {"PATH": str(d)}, d, None)
    assert rc == 0
    table = sr.load_step_rate_table(out, image_digest=pinned_image())
    assert set(table) == set(skus)
    for r in recs[:-1]:
        assert table[r["sku"]].steps_per_s == pytest.approx(r["steps_per_s"], rel=1e-12)
    # the standalone table file combine can also write reads the same
    out2 = d / "table.json"
    rc, _, _ = run_tool(tool, ["combine", "--chain-toml", CHAIN_TOML, "--runs", *files,
                               "--out", d / "wb19b.json", "--table-out", out2],
                        {"PATH": str(d)}, d, None)
    assert rc == 0
    assert json.loads(out2.read_text()) == json.loads(out.read_text())["table"]
    assert sr.load_step_rate_table(out2) == table
