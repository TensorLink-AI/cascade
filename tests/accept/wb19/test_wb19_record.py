"""WB19: finishing a pod record (`finish_record`) and assembling the evidence
(`combine`): rates, the overhead split, `leg_failed` / `non_finite`, the
per-GPU table, coverage of the candidate list and every `combine` refusal.

The pod's leg result is a fixture (``helpers.good_result``): what the real
leg path hands ``finish_record``. In-process, no torch, no GPU. Contract:
README.md "`finish_record`" and "`combine`".
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from .helpers import (
    CHAIN_TOML,
    EVIDENCE_SCHEMA,
    GIB,
    HEX64,
    TABLE_SCHEMA,
    TORCH_PIN,
    body_sha256,
    chain_cfg,
    contract_block,
    good_result,
    leg_contract_digest,
    load_tool,
    make_plan,
    make_pod_record,
    pinned_image,
    refusal_code,
    rehash,
    run_tool,
    write_json,
)

FULL_SKU = "8xH100"


@pytest.fixture(scope="module")
def lane_b_skus():
    return list(chain_cfg().lane_b.funded_pod_skus)


@pytest.fixture(scope="module")
def records_factory(tmp_path_factory, lane_b_skus):
    """A lazy builder (the tool loads in the test body, never in a fixture):
    one finished expert record per lane B SKU plus the full 8xH100 leg,
    built once per module and deep-copied per test."""
    cache: dict = {}

    def build(tool) -> dict:
        if not cache:
            for sku in lane_b_skus:
                d = tmp_path_factory.mktemp(f"rec-{sku}")
                cache[sku] = make_pod_record(tool, d, None, sku)
            d = tmp_path_factory.mktemp("rec-full")
            cache["full"] = make_pod_record(tool, d, None, FULL_SKU, mode="full")
        return copy.deepcopy(cache)

    return build


def _combine(tool, tmp_path, records, *, extra=(), capsys=None):
    files = [write_json(tmp_path / "runs" / f"run{i}.json", r) for i, r in enumerate(records)]
    out = tmp_path / "evidence" / "wb19.json"
    argv = ["combine", "--chain-toml", CHAIN_TOML, "--runs", *files, "--out", out, *extra]
    rc, so, se = run_tool(tool, argv, {"PATH": str(tmp_path)}, tmp_path, capsys)
    return rc, so, se, out


def _check(ev, name):
    hits = [c for c in ev["checks"] if c["name"] == name]
    assert len(hits) == 1, f"exactly one check named {name!r}: {ev['checks']}"
    return hits[0]


# ── finish_record ──────────────────────────────────────────────────────────

def test_finish_record_rates_and_split(tmp_path):
    tool = load_tool()
    plan = make_plan(tool, tmp_path, None, "8xH200")
    plan["source"] = "pod"
    res = good_result(plan["steps"])
    rec = tool.finish_record(copy.deepcopy(plan), res)
    c = plan["contract"]
    assert rec["source"] == "pod"
    assert rec["steps"] == plan["steps"]
    assert rec["steps_per_s"] == pytest.approx(plan["steps"] / 900.0, rel=1e-9)
    assert rec["tokens_per_s"] == pytest.approx(
        plan["steps"] * c["batch_size"] * c["context_length"] / 900.0, rel=1e-9)
    assert rec["wall"] == res["wall"]
    assert rec["peak_memory_bytes"] == res["peak_memory_bytes"]
    assert rec["finite"] is True and rec["torch"] == TORCH_PIN
    for k in ("leg_config_digest", "contract", "image_digest", "gpu", "sku", "mode",
              "selection", "contract_block", "schema", "wp"):
        assert rec[k] == plan[k], f"finishing must not change {k}"
    assert rec["body_sha256"] == body_sha256(rec)
    json.dumps(rec, allow_nan=False)


@pytest.mark.parametrize("edit", [
    lambda r: r.update(ok=False, error="worker exit 1"),
    lambda r: r.update(deadline_hit=True),
    lambda r: r.update(steps=r["steps"] - 1),
])
def test_finish_record_leg_failed(tmp_path, edit):
    tool = load_tool()
    plan = make_plan(tool, tmp_path, None, "8xH100")
    res = good_result(plan["steps"])
    edit(res)
    with pytest.raises(tool.MeasureRefused) as ei:
        tool.finish_record(plan, res)
    assert ei.value.code == "leg_failed"


@pytest.mark.parametrize("edit", [
    lambda r: r.update(final_loss=math.nan),
    lambda r: r.update(final_loss=math.inf),
    lambda r: r["wall"].update(train_seconds=math.nan),
    lambda r: r["wall"].update(train_seconds=0.0),
    lambda r: r["wall"].update(total_seconds=math.inf),
    lambda r: r["wall"]["overhead_seconds"].update(corpus=-math.inf),
])
def test_finish_record_non_finite(tmp_path, edit):
    tool = load_tool()
    plan = make_plan(tool, tmp_path, None, "8xH100")
    res = good_result(plan["steps"])
    edit(res)
    with pytest.raises(tool.MeasureRefused) as ei:
        tool.finish_record(plan, res)
    assert ei.value.code == "non_finite"


def test_finish_record_leg_failed_wins_over_non_finite(tmp_path):
    tool = load_tool()
    plan = make_plan(tool, tmp_path, None, "8xH100")
    res = good_result(plan["steps"])
    res.update(ok=False, final_loss=math.nan)
    with pytest.raises(tool.MeasureRefused) as ei:
        tool.finish_record(plan, res)
    assert ei.value.code == "leg_failed"


def test_measure_refused_codes_are_run_refusals():
    tool = load_tool()
    e = tool.MeasureRefused("leg_failed", "x")
    assert e.code == "leg_failed" and e.code in tool.RUN_REFUSALS


# ── combine: the evidence and the per-GPU table ────────────────────────────

def test_combine_passes_with_full_coverage(tmp_path, records_factory, lane_b_skus):
    tool = load_tool()
    pod_records = records_factory(tool)
    rc, _, se, out = _combine(tool, tmp_path, list(pod_records.values()))
    assert rc == 0, se
    ev = json.loads(out.read_text())
    assert ev["schema"] == EVIDENCE_SCHEMA and ev["wp"] == "WB19"
    assert ev["passed"] is True, ev["checks"]
    assert {c["name"] for c in ev["checks"]} >= {"coverage", "full_leg", "memory"}
    assert all(c["passed"] is True for c in ev["checks"])
    assert ev["body_sha256"] == body_sha256(ev)
    assert len(ev["runs"]) == len(pod_records)
    assert sorted(r["body_sha256"] for r in ev["runs"]) == sorted(
        r["body_sha256"] for r in pod_records.values()), "runs are carried verbatim"
    json.dumps(ev, allow_nan=False)

    cfg = chain_cfg()
    t = ev["table"]
    assert t["schema"] == TABLE_SCHEMA
    assert t["image_digest"] == pinned_image(cfg)
    assert t["contract_block"] == contract_block(cfg)
    assert t["contract_digest"] == leg_contract_digest(cfg, contract_block(cfg), "expert")
    assert sorted(t["rows"]) == sorted(lane_b_skus), "one row per expert run (no full leg)"
    for sku in lane_b_skus:
        rec, row = pod_records[sku], t["rows"][sku]
        o = rec["wall"]["overhead_seconds"]
        assert row["steps_per_s"] == pytest.approx(rec["steps_per_s"], rel=1e-12)
        assert row["tokens_per_s"] == pytest.approx(rec["tokens_per_s"], rel=1e-12)
        assert row["overhead_seconds"] == pytest.approx(o["boot"] + o["corpus"] + o["harvest"])
        assert row["peak_memory_bytes"] == rec["peak_memory_bytes"]
        assert row["memory_total_bytes"] == rec["gpu"]["memory_total_bytes"]
        assert row["gpu_name"] == rec["gpu"]["name"] and row["gpu_count"] == rec["gpu"]["count"]
        assert row["driver"] == rec["gpu"]["driver"] and row["cuda"] == rec["gpu"]["cuda"]
        assert row["run_body_sha256"] == rec["body_sha256"]

    full = ev["lane_a_full_leg"]
    fr = pod_records["full"]
    assert full["sku"] == FULL_SKU
    assert full["train_seconds"] == pytest.approx(fr["wall"]["train_seconds"])
    assert full["total_seconds"] == pytest.approx(fr["wall"]["total_seconds"])
    fo = fr["wall"]["overhead_seconds"]
    assert full["overhead_seconds"] == pytest.approx(fo["boot"] + fo["corpus"] + fo["harvest"])
    walls = dict(cfg.round.funded_sku_wall_seconds)
    assert full["funded_sku_wall_seconds"] == walls.get(FULL_SKU)
    assert ev["oa8_sha256"] is None


def test_combine_coverage_fails_for_a_missing_sku(tmp_path, records_factory, lane_b_skus):
    tool = load_tool()
    pod_records = records_factory(tool)
    missing = lane_b_skus[-1]
    recs = [r for k, r in pod_records.items() if k != missing]
    rc, _, se, out = _combine(tool, tmp_path, recs)
    assert rc == 0, se
    ev = json.loads(out.read_text())
    assert ev["passed"] is False
    cov = _check(ev, "coverage")
    assert cov["passed"] is False and missing in cov["detail"]


def test_combine_owner_candidates_join_coverage(tmp_path, records_factory):
    tool = load_tool()
    pod_records = records_factory(tool)
    oa8 = write_json(tmp_path / "oa8.json", {"oa": "OA8", "status": "approved",
                                             "gpu_types": ["L40S"], "cost_cap_usd": 400})
    rc, _, se, out = _combine(tool, tmp_path, list(pod_records.values()),
                              extra=["--candidates", oa8])
    assert rc == 0, se
    ev = json.loads(out.read_text())
    assert ev["passed"] is False and "L40S" in _check(ev, "coverage")["detail"]
    import hashlib

    assert ev["oa8_sha256"] == hashlib.sha256(oa8.read_bytes()).hexdigest()


def test_combine_needs_the_full_8xh100_leg(tmp_path, records_factory):
    tool = load_tool()
    pod_records = records_factory(tool)
    recs = [r for k, r in pod_records.items() if k != "full"]
    rc, _, se, out = _combine(tool, tmp_path, recs)
    assert rc == 0, se
    ev = json.loads(out.read_text())
    assert ev["passed"] is False and _check(ev, "full_leg")["passed"] is False
    assert ev["lane_a_full_leg"] is None


def test_combine_memory_check(tmp_path_factory, records_factory):
    tool = load_tool()
    pod_records = records_factory(tool)
    d = tmp_path_factory.mktemp("oom")
    big = make_pod_record(tool, d, None, "4xH100", peak=80 * GIB)   # == the device's memory
    recs = [r for k, r in pod_records.items() if k != "4xH100"] + [big]
    rc, _, se, out = _combine(tool, d, recs)
    assert rc == 0, se
    ev = json.loads(out.read_text())
    mem = _check(ev, "memory")
    assert ev["passed"] is False and mem["passed"] is False and "4xH100" in mem["detail"]


# ── combine refusals ───────────────────────────────────────────────────────

def _refuse(tool, tmp_path, records, code, capsys):
    rc, _, se, out = _combine(tool, tmp_path, records, capsys=capsys)
    assert rc == 2 and refusal_code(se) == code, f"expected REFUSED {code}: {se!r}"
    assert not out.exists(), f"a {code} refusal writes no evidence"


def test_combine_refuses_dry_run_record(tmp_path, records_factory, capsys):
    tool = load_tool()
    pod_records = records_factory(tool)
    plan = make_plan(tool, tmp_path, None, "8xH100")
    assert plan["source"] == "dry_run"
    recs = [r for k, r in pod_records.items() if k != "8xH100"] + [plan]
    _refuse(tool, tmp_path, recs, "not_pod", capsys)


@pytest.mark.parametrize("edit", [
    lambda r: r.update(steps_per_s=r["steps_per_s"] * 2),          # hand-edited, no rehash
    lambda r: r.update(schema="laned_wb19_run/0"),
    lambda r: r.pop("leg_config_digest"),
])
def test_combine_refuses_bad_run(tmp_path, records_factory, capsys, edit):
    tool = load_tool()
    pod_records = records_factory(tool)
    bad = copy.deepcopy(pod_records["8xH200"])
    edit(bad)
    if "schema" in bad and bad["schema"] != "laned_wb19_run/1":
        bad = rehash(bad)
    recs = [r for k, r in pod_records.items() if k != "8xH200"] + [bad]
    _refuse(tool, tmp_path, recs, "bad_run", capsys)


def test_combine_refuses_wrong_image(tmp_path, records_factory, capsys):
    tool = load_tool()
    pod_records = records_factory(tool)
    bad = copy.deepcopy(pod_records["8xH200"])
    bad["image_digest"] = "sha256:" + "12" * 32
    recs = [r for k, r in pod_records.items() if k != "8xH200"] + [rehash(bad)]
    _refuse(tool, tmp_path, recs, "wrong_image", capsys)


def test_combine_refuses_wrong_contract(tmp_path, records_factory, capsys):
    tool = load_tool()
    pod_records = records_factory(tool)
    bad = copy.deepcopy(pod_records["4xH200"])
    bad["contract"]["digest"] = "34" * 32
    recs = [r for k, r in pod_records.items() if k != "4xH200"] + [rehash(bad)]
    _refuse(tool, tmp_path, recs, "wrong_contract", capsys)


def test_combine_refuses_mixed_block(tmp_path, records_factory, capsys):
    tool = load_tool()
    pod_records = records_factory(tool)
    plan = make_plan(tool, tmp_path, None, "4xH200", block=contract_block() + 900)
    plan["source"] = "pod"
    other = tool.finish_record(plan, good_result(plan["steps"]))
    recs = [r for k, r in pod_records.items() if k != "4xH200"] + [other]
    _refuse(tool, tmp_path, recs, "mixed_block", capsys)


def test_combine_refuses_duplicate_sku(tmp_path, records_factory, capsys):
    tool = load_tool()
    pod_records = records_factory(tool)
    plan = make_plan(tool, tmp_path, None, "8xH100", steps=41)
    plan["source"] = "pod"
    dup = tool.finish_record(plan, good_result(plan["steps"]))
    _refuse(tool, tmp_path, [*pod_records.values(), dup], "duplicate_sku", capsys)


def test_finished_record_digests_are_hex(records_factory):
    tool = load_tool()
    for r in records_factory(tool).values():
        assert r["source"] == "pod" and r["body_sha256"] == body_sha256(r)
        assert HEX64.fullmatch(r["body_sha256"]) and HEX64.fullmatch(r["leg_config_digest"])
