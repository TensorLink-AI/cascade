"""W3b6b, producer side: ``sweep_record_text`` is one sweep record's canonical
text; every gate result whose three records were swept binds them by sha256
(``rows``) inside the signed body; the records' exact bytes are persisted at
``<state dir>/<stem>_bench_rows/<sha256>.json`` before the report. A
settlement without a gate, or a sweep that raised, binds and writes nothing.
Contract: README.md."""

from __future__ import annotations

import hashlib
import json
import math

from . import helpers as h


def _text_fn():
    from cascade.validator import lane_d_bench

    assert hasattr(lane_d_bench, "sweep_record_text"), \
        "W3b6b: lane_d_bench.sweep_record_text(record) (README)"
    return lane_d_bench.sweep_record_text


def _bound(rep: dict) -> dict:
    (res,) = h.gate_results(rep)
    rows = res.get("rows")
    assert isinstance(rows, dict), "W3b6b: the gate result binds its three records (rows)"
    return rows


def test_record_text_is_canonical():
    text = _text_fn()
    rec = {"session": "s", "digest": "d" * 64,
           "suites": {"gift": {"status": "ok",
                               "rows": [{"mase": 1.25, "item": "g/é", "crps": 0.5}]}}}
    t = text(rec)
    assert isinstance(t, str) and t == h.json_text(rec), \
        "sorted keys, compact separators, ASCII-escaped"
    assert text(json.loads(t)) == t, "a parsed record re-serialises to the same bytes"
    nonfinite = {"session": "s", "digest": "d" * 64,
                 "suites": {"gift": {"status": "ok",
                                     "rows": [{"item": "a", "crps": math.nan, "mase": math.inf}]}}}
    assert json.loads(text(nonfinite))["suites"]["gift"]["rows"][0] == \
        {"item": "a", "crps": None, "mase": None}, "a non-finite number is written as null"


def test_gated_result_binds_its_three_records(tmp_path):
    r, rep = h.gated(tmp_path)
    (res,) = h.gate_results(rep)
    rows = _bound(rep)
    assert set(rows) == set(h.ROLES)
    assert all(isinstance(v, str) and h.HEX64.match(v) for v in rows.values())
    files = h.persisted_rows(r)
    for role, digest in rows.items():
        assert digest in files, f"{role}: its record is persisted beside the report"
        data = files[digest]
        assert hashlib.sha256(data).hexdigest() == digest, "the bytes hash to the bound digest"
        rec = json.loads(data)
        assert data.decode("ascii") == _text_fn()(rec), "the persisted bytes are the canonical text"
        assert rec["session"] == res["session"], "one host session"
    assert json.loads(files[rows["candidate"]])["digest"] == res["digest"]
    assert json.loads(files[rows["king_d"]])["digest"] == rep["king_before"]["digest"]
    from tests.accept.w7a.helpers import assert_valid

    assert_valid(rep)


def test_recorded_lcbs_are_the_paired_gate_of_the_bound_records(tmp_path):
    from cascade.validator.lane_d_bench import paired_gate

    r, rep = h.gated(tmp_path)
    (res,) = h.gate_results(rep)
    rows = _bound(rep)
    files = h.persisted_rows(r)
    recs = [json.loads(files[rows[role]]) for role in h.ROLES]
    g, cfg = rep["gate"], r.cfg.lane_d
    again = paired_gate(*recs, suite=g["suite"], alpha=float(cfg.alpha),
                        tau_public=float(cfg.tau_public), B=int(cfg.bootstrap_B),
                        seed=f"lane-d/1:gate:{rep['boundary']}:{g['block_hash']}",
                        sentinel=g["sentinel"])
    assert again["outcome"] == res["outcome"]
    assert {k: dict(v) for k, v in again["lcb"].items()} == res["lcb"], \
        "the bound bytes reproduce the signed LCBs exactly"


def test_independent_statistic_matches_the_signed_lcbs(tmp_path):
    """What a satellite recomputes (README "Re-derive"): items paired by id on
    the drawn suite, the paired LCB per reference and metric, seed
    ``cascade|lane_d|paired_gate|v1|lane-d/1:gate:<b>:<hash>|<suite>|<m>``."""
    r, rep = h.gated(tmp_path)
    (res,) = h.gate_results(rep)
    rows = _bound(rep)
    files = h.persisted_rows(r)
    g, cfg = rep["gate"], r.cfg.lane_d
    seed = f"lane-d/1:gate:{rep['boundary']}:{g['block_hash']}"

    def items(role):
        rec = json.loads(files[rows[role]])
        return {x["item"]: (x["crps"], x["mase"]) for x in rec["suites"][g["suite"]]["rows"]}

    cand = items("candidate")
    for ref in ("king_d", "house"):
        other = items(ref)
        ids = sorted(set(cand) & set(other))
        for j, m in enumerate(("crps", "mase")):
            got = h.independent_lcb([other[i][j] for i in ids], [cand[i][j] for i in ids],
                                    alpha=float(cfg.alpha), B=int(cfg.bootstrap_B),
                                    seed=f"{h.SEED_TAG}|{seed}|{g['suite']}|{m}")
            assert abs(got - res["lcb"][ref][m]) <= 1e-12, (ref, m)


def test_no_gate_binds_and_writes_nothing(tmp_path):
    r = h.rig(tmp_path)
    r.step(h.B1)
    rep1 = json.loads(r.report_file(h.B1).read_text(encoding="ascii"))
    assert all("rows" not in x for x in h.gate_results(rep1))
    assert h.persisted_rows(r) == {}, "no gate: no record file"
    r.step(h.B2)
    rep2 = json.loads(r.report_file(h.B2).read_text(encoding="ascii"))
    bound = set(_bound(rep2).values())
    assert set(h.persisted_rows(r)) == bound, "only bound records are persisted"


class _Broken(h.w8b.BenchHost):
    def __call__(self, digests, suites):
        raise ConnectionError("w3b6b: bench pod lost")


def test_raising_sweep_is_an_unbound_hold(tmp_path):
    """A bench host that raises: every gate result is bench_hold with no rows,
    and no record file is written."""
    _text_fn()
    fx = h.w8b.fixture(tmp_path)
    r = h.w8b.Rig(tmp_path, h.w8b.chain_cfg(tmp_path / "v"), fx,
                  host=_Broken(fx["house"]["digest"]))
    r.step(h.B1)
    r.step(h.B2)
    rep = json.loads(r.report_file(h.B2).read_text(encoding="ascii"))
    results = h.gate_results(rep)
    assert results, "the fixture gates X at B2"
    assert all(x["outcome"] == "bench_hold" and "rows" not in x for x in results)
    assert h.persisted_rows(r) == {}


def test_schema_declares_optional_gate_rows():
    from cascade.validator.lane_d_report import load_schema

    schema = load_schema()
    gr = schema["$defs"]["gate_result"]
    assert "rows" in gr["properties"], "W3b6b: gate_result.rows in the schema"
    assert "rows" not in gr.get("required", []), "optional and additive (report stays /1)"
    rows = gr["properties"]["rows"]
    if "$ref" in rows:
        rows = schema["$defs"][rows["$ref"].split("/")[-1]]
    assert set(rows.get("required", [])) == set(h.ROLES)
    assert rows.get("additionalProperties") is False
