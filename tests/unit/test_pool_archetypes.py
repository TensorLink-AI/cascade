"""Archetype-aware pool selection (2026-11-04, tsbench-forge request):
steering toward benchmark reference shares inside the balanced picker, the
overlap metric, the unknown bucket, provenance, the cluster floor and the
dry-run sweep. All on tiny synthetic catalogs."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cascade.pool import archetypes as A
from cascade.pool.builder import (
    BUILDER_VERSION,
    ClusterFloorError,
    PoolBuildConfig,
    build_pool,
    collect_records,
    steering_sweep,
)
from cascade.pool.source import HarvestContext, HarvestedSeries

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# Byte copy of the forge's real model file (bucket root archetypes.json, version 1,
# fitted 2026-11-03; sha256 378756e4…) — the list form aligned with ``names``.
REAL_MODEL = FIXTURES / "tsbench_forge_archetypes_v1.json"

CTX = HarvestContext(as_of=dt.date(2026, 6, 1), context_length=512, horizon=16, max_series=10_000)

MODEL = A.parse_archetype_model({
    "model_version": "arch-2026-11-01",
    "ref_share_daily_or_faster": {"A01": 0.1, "A04": 0.3, "A09": 0.6},
    "ref_share_by_bench": {"gift": {"A01": 0.2, "A04": 0.2, "A09": 0.6},
                           "boom": {"A01": 0.0, "A04": 0.5, "A09": 0.5},
                           "time": {"A01": 0.3, "A04": 0.3, "A09": 0.4}},
}, source="test", sha256="abc")


class _ListSource:
    name = "list"

    def __init__(self, items):
        self.items = items

    def harvest(self, fetch, ctx):
        yield from self.items


def _series(src, i, arch, domain="transport", freq="H"):
    base = 10 + np.sin(np.arange(600) / (5.0 + i)) + i * 0.013
    attrs = {"stat_archetype": arch} if arch else {}
    return HarvestedSeries(f"{src}__{i}", base, freq, domain, 24, source=src, attrs=attrs)


def _catalog():
    """30 feeds: 12 of A01, 12 of A04, 3 of A09, 3 untagged — the benchmark
    wants A09 heavy, the catalog is A01/A04 heavy, so a capped pool shows the
    steering plainly. Two series per feed."""
    items = []
    i = 0
    for k in range(12):
        items += [_series(f"a01_{k}", i, "A01"), _series(f"a01_{k}", i + 1, "A01")]
        i += 2
    for k in range(12):
        items += [_series(f"a04_{k}", i, "A04"), _series(f"a04_{k}", i + 1, "A04")]
        i += 2
    for k in range(3):
        items += [_series(f"a09_{k}", i, "A09"), _series(f"a09_{k}", i + 1, "A09")]
        i += 2
    for k in range(3):
        items += [_series(f"unk_{k}", i, ""), _series(f"unk_{k}", i + 1, "")]
        i += 2
    return items


def _cfg(**kw):
    return PoolBuildConfig(context_length=512, horizon=16, min_context=64,
                           selection="balanced", selection_seed="block:7", **kw)


def _mix(records):
    return Counter(A.archetype_of(r.metadata) for r in records)


# ── model + metric ───────────────────────────────────────────────────────────


def test_model_parses_flat_and_list_forms_and_normalises():
    assert MODEL.version == "arch-2026-11-01"
    assert sum(MODEL.ref_share.values()) == pytest.approx(1.0)
    assert MODEL.archetypes == ("A01", "A04", "A09")
    assert set(MODEL.ref_share_by_bench) == {"gift", "boom", "time"}
    lst = A.parse_archetype_model({"version": "v2", "archetypes": [
        {"id": "A01", "ref_share_daily_or_faster": 1, "ref_share_by_bench": {"gift": 1}},
        {"id": "A09", "ref_share_daily_or_faster": 3, "ref_share_by_bench": {"gift": 1}}]})
    assert lst.ref_share == {"A01": 0.25, "A09": 0.75}
    assert lst.ref_share_by_bench["gift"] == {"A01": 0.5, "A09": 0.5}
    with pytest.raises(A.ArchetypeModelError):
        A.parse_archetype_model({"model_version": "x"})
    with pytest.raises(A.ArchetypeModelError):
        A.parse_archetype_model({"ref_share_daily_or_faster": {"A01": 1}})
    # an "unknown" reference share is never accepted: the untagged have none
    m = A.parse_archetype_model({"model_version": "x", "ref_share_daily_or_faster": {"A01": 1, "unknown": 5}})
    assert m.ref_share == {"A01": 1.0}


def test_real_forge_model_file_parses_in_its_list_form_and_reproduces_the_published_overlaps():
    m = A.load_archetype_model(REAL_MODEL)
    assert m is not None
    assert m.version == "1" and m.fitted == "2026-11-03"
    assert m.sha256 == "378756e42dadbcdb58735ff90ff0918b25c0628c73504b233b49ad5f8e6f2be1"
    assert m.archetypes == tuple(f"A{i:02d}" for i in range(12))
    assert sum(m.ref_share.values()) == pytest.approx(1.0)
    assert set(m.ref_share_by_bench) == {"gift", "boom", "time"}
    for shares in m.ref_share_by_bench.values():
        assert set(shares) == set(m.archetypes) and sum(shares.values()) == pytest.approx(1.0)
    # the numbers in the forge's request table (A09 6.7% -> 17.4%, A04 2.0% -> 4.9%)
    assert m.ref_share["A09"] == pytest.approx(0.1735, abs=1e-4)
    assert m.published_pool_share["A09"] == pytest.approx(0.0671, abs=1e-4)
    assert m.ref_share["A04"] == pytest.approx(0.0492, abs=1e-4)
    # our overlap formula on their published pool mix gives their published
    # 0.811 (daily-or-faster) and 0.701 / 0.623 / 0.668 per benchmark
    chk = A.published_check(m)
    assert chk["agrees"] is True
    assert chk["recomputed_daily_or_faster"] == pytest.approx(m.published_overlap["daily_or_faster"], abs=1e-3)
    assert chk["recomputed_by_bench"]["gift"] == pytest.approx(0.701, abs=1e-3)
    assert chk["recomputed_by_bench"]["boom"] == pytest.approx(0.623, abs=1e-3)
    assert chk["recomputed_by_bench"]["time"] == pytest.approx(0.668, abs=1e-3)
    # the report and provenance carry the cross-check
    rep = A.archetype_report(Counter({"A09": 1}), Counter({"A09": 1}), m, steering=0.0, n_sources=1, n_series=1)
    assert rep["forge_published"]["agrees"] is True and rep["model_fitted"] == "2026-11-03"
    assert "forge published daily-or-faster 0.811" in A.render_report(rep)


def test_list_form_needs_names_of_the_same_length():
    doc = json.loads(REAL_MODEL.read_text())
    with pytest.raises(A.ArchetypeModelError, match="12 entries for 11 names"):
        A.parse_archetype_model({**doc, "names": doc["names"][:-1]})
    with pytest.raises(A.ArchetypeModelError, match="carries no names"):
        A.parse_archetype_model({k: v for k, v in doc.items() if k != "names"})
    # a by-bench list of the wrong length is an error too, not a silent drop
    bad = {**doc, "ref_share_by_bench": {**doc["ref_share_by_bench"], "gift": doc["ref_share_by_bench"]["gift"][:3]}}
    with pytest.raises(A.ArchetypeModelError, match=r"ref_share_by_bench\[gift\] has 3 entries"):
        A.parse_archetype_model(bad)


def test_overlap_is_sum_of_mins_over_tagged_series_only():
    ref = {"A01": 0.1, "A04": 0.3, "A09": 0.6}
    assert A.overlap(Counter({"A01": 10, "A04": 30, "A09": 60}), ref) == pytest.approx(1.0)
    assert A.overlap(Counter({"A01": 50, "A04": 50}), ref) == pytest.approx(0.1 + 0.3)
    # unknown is left out of the denominator and reported separately
    c = Counter({"A01": 50, "A04": 50, A.UNKNOWN: 900})
    assert A.overlap(c, ref) == pytest.approx(0.4)
    assert A.unknown_fraction(c) == pytest.approx(0.9)
    assert A.overlap(Counter({A.UNKNOWN: 5}), ref) == 0.0


def test_steering_targets_blend_reference_and_natural_and_keep_unknown():
    natural = Counter({"A01": 40, "A04": 40, "A09": 10, A.UNKNOWN: 10})
    t0 = A.steering_targets(natural, MODEL, 0.0)
    assert t0["A01"] == pytest.approx(0.4) and t0[A.UNKNOWN] == pytest.approx(0.1)
    t1 = A.steering_targets(natural, MODEL, 1.0)
    assert t1["A09"] == pytest.approx(0.6 * 0.9) and t1[A.UNKNOWN] == pytest.approx(0.1)
    th = A.steering_targets(natural, MODEL, 0.5)
    assert th["A09"] == pytest.approx((0.5 * 0.6 + 0.5 * (10 / 90)) * 0.9)
    assert sum(th.values()) == pytest.approx(1.0)


def test_load_model_absent_is_none_and_bad_json_raises(tmp_path):
    assert A.load_archetype_model(tmp_path / "archetypes.json") is None
    (tmp_path / "archetypes.json").write_text("{not json")
    with pytest.raises(A.ArchetypeModelError):
        A.load_archetype_model(tmp_path / "archetypes.json")
    (tmp_path / "archetypes.json").write_text(json.dumps({
        "model_version": "m1", "ref_share_daily_or_faster": {"A01": 1}}))
    m = A.load_archetype_model(tmp_path / "archetypes.json")
    assert m.version == "m1" and len(m.sha256) == 64


# ── the picker ───────────────────────────────────────────────────────────────


def test_steering_zero_or_no_model_selects_exactly_the_unsteered_pool():
    items = _catalog()
    plain, _ = collect_records([_ListSource(items)], CTX, _cfg(max_series_per_domain=20), fetch=None)
    zero, _ = collect_records([_ListSource(items)], CTX, _cfg(max_series_per_domain=20),
                              fetch=None, archetypes=MODEL)
    nomodel, _ = collect_records([_ListSource(items)], CTX,
                                 _cfg(max_series_per_domain=20, archetype_steering=1.0), fetch=None)
    ids = [r.series_id for r in plain]
    assert [r.series_id for r in zero] == ids and [r.series_id for r in nomodel] == ids


def test_steering_moves_the_mix_toward_the_reference_within_the_caps():
    items = _catalog()
    cfg0 = _cfg(max_series_per_domain=20)
    cfg1 = _cfg(max_series_per_domain=20, archetype_steering=1.0)
    before, _ = collect_records([_ListSource(items)], CTX, cfg0, fetch=None, archetypes=MODEL)
    after, _ = collect_records([_ListSource(items)], CTX, cfg1, fetch=None, archetypes=MODEL)
    assert len(before) == 20 and len(after) == 20            # the cap is the cap
    m0, m1 = _mix(before), _mix(after)
    # A09 is scarce (3 clusters) and wanted (60 %): steering serves every A09
    # cluster in the first pass — one series each, the rotation rule holds —
    # while A01 (plentiful, barely wanted) gets the leftovers
    assert m1["A09"] == 3 and m1["A09"] >= m0["A09"]
    assert m1["A01"] < m0["A01"]
    assert A.overlap(m1, MODEL.ref_share) > A.overlap(m0, MODEL.ref_share)
    # the untagged keep their natural share (never traded away for the mix)
    assert m1[A.UNKNOWN] >= 1
    # one series per cluster per pass still holds: 30 clusters > 20 slots ⇒ no doubles
    assert max(Counter(r.metadata["source"] for r in after).values()) == 1


def test_half_steering_sits_between_zero_and_one():
    items = _catalog()
    mixes = {}
    for s in (0.0, 0.5, 1.0):
        recs, _ = collect_records([_ListSource(items)], CTX,
                                  _cfg(max_series_per_domain=20, archetype_steering=s),
                                  fetch=None, archetypes=MODEL)
        mixes[s] = A.overlap(_mix(recs), MODEL.ref_share)
    # overlap is not strictly monotone in S on a tiny capped catalog (the
    # tagged denominator moves with the unknown picks); the direction is
    assert mixes[0.0] < mixes[1.0]
    assert mixes[0.0] - 0.05 <= mixes[0.5] <= mixes[1.0] + 1e-9


def test_steering_respects_the_cell_cap_and_is_deterministic():
    items = _catalog() + [_series(f"d{k}", 900 + k, "A09", freq="D") for k in range(4)]
    cfg = _cfg(max_series_per_domain_freq=5, archetype_steering=1.0)
    a, drops = collect_records([_ListSource(items)], CTX, cfg, fetch=None, archetypes=MODEL)
    b, _ = collect_records([_ListSource(list(reversed(items)))], CTX, cfg, fetch=None, archetypes=MODEL)
    assert [r.series_id for r in a] == [r.series_id for r in b]
    per_cell = Counter((r.domain, r.metadata["freq"]) for r in a)
    assert max(per_cell.values()) <= 5 and drops["domain_freq_cap"] > 0


def test_provenance_records_model_steering_catalog_hash_and_tags_used(tmp_path):
    cfg = _cfg(max_series_per_domain=20, archetype_steering=1.0)
    build_pool([_ListSource(_catalog())], tmp_path / "p", CTX, cfg, fetch=None,
               archetypes=MODEL, catalog_sha256="c" * 64)
    prov = json.loads((tmp_path / "p" / "provenance.json").read_text())
    assert prov["builder_version"] == BUILDER_VERSION == "cascade-pool/2"
    arch = prov["archetypes"]
    assert arch["model_version"] == "arch-2026-11-01" and arch["model_sha256"] == "abc"
    assert arch["steering"] == 1.0 and arch["catalog_sha256"] == "c" * 64
    assert arch["by_archetype"]["A09"] == 3 and arch["unknown_series"] >= 1
    assert all(v in ("A01", "A04", "A09") for v in arch["tags_used"].values())
    assert "unk_0" not in arch["tags_used"]                     # untagged: no tag recorded
    assert arch["report"]["overlap_daily_or_faster"] > 0 and set(arch["report"]["overlap_by_bench"]) == {"gift", "boom", "time"}
    meta = json.loads((tmp_path / "p" / "metadata.json").read_text())
    assert {m.get("stat_archetype") for m in meta.values()} >= {"A01", "A04", "A09", None}


def test_provenance_without_a_model_still_carries_the_block(tmp_path):
    build_pool([_ListSource(_catalog())], tmp_path / "p", CTX, _cfg(max_series_per_domain=20), fetch=None)
    arch = json.loads((tmp_path / "p" / "provenance.json").read_text())["archetypes"]
    assert arch["model_version"] is None and arch["steering"] == 0.0 and arch["n_clusters"] == 20


def test_cluster_floor_refuses_a_narrow_pool(tmp_path):
    with pytest.raises(ClusterFloorError, match="min_clusters"):
        build_pool([_ListSource(_catalog())], tmp_path / "p", CTX,
                   _cfg(max_series_per_domain=20, min_clusters=25), fetch=None)
    build_pool([_ListSource(_catalog())], tmp_path / "q", CTX,
               _cfg(max_series_per_domain=20, min_clusters=20), fetch=None)


def test_steering_sweep_reports_before_and_after_from_the_same_candidates():
    reps = steering_sweep([_ListSource(_catalog())], CTX, _cfg(max_series_per_domain=20),
                          archetypes=MODEL, steerings=(0.0, 0.5, 1.0), fetch=None)
    assert [r["steering"] for r in reps] == [0.0, 0.5, 1.0]
    r0, r1 = reps[0], reps[-1]
    assert r0["supply_by_archetype"] == r1["supply_by_archetype"] == {"A01": 24, "A04": 24, "A09": 6, A.UNKNOWN: 6}
    assert r1["overlap_daily_or_faster"] > r0["overlap_daily_or_faster"]
    assert r1["n_series"] == 20 and r1["n_sources"] == 20
    assert set(r1["overlap_by_bench"]) == {"gift", "boom", "time"}
    assert "transport" in r1["per_domain"] and len(r1["series_ids"]) == 20
    # supply-limited archetypes are named (A09 wants 60 % of ~17 tagged ≈ 10 > 6 exist)
    assert "A09" in r1["supply_limited"]
    text = A.render_report(r1)
    assert "overlap daily-or-faster" in text and "supply-limited: A09" in text


def test_cli_flags_plumb_into_the_config_and_reject_bad_values(tmp_path, monkeypatch):
    import argparse

    from cascade.pool import cli

    p = argparse.ArgumentParser()
    cli._add_build_args(p)
    args = p.parse_args(["--archetype-steering", "0.5", "--min-clusters", "3", "--selection", "balanced"])
    assert args.archetype_steering == 0.5 and args.min_clusters == 3
    captured = {}

    def fake_build_pool(sources, out_dir, ctx, cfg, **kw):
        captured["cfg"] = cfg
        captured["kw"] = kw
        return "built"

    monkeypatch.setattr(cli, "build_pool", fake_build_pool)
    monkeypatch.setattr(cli, "get_sources", lambda names: [_ListSource([])])
    model_path = tmp_path / "archetypes.json"
    model_path.write_text(json.dumps({"model_version": "m9", "ref_share_daily_or_faster": {"A01": 1}}))
    args = p.parse_args(["--archetype-steering", "0.5", "--selection", "balanced",
                         "--archetypes", str(model_path), "--sources", "list"])
    args.out = tmp_path / "out"
    chain_cfg = type("C", (), {"eval": type("E", (), {"context_length": 512, "horizon": 16})()})()
    assert cli._build(args, chain_cfg, out_dir=args.out, overwrite=False) == "built"
    assert captured["cfg"].archetype_steering == 0.5 and captured["kw"]["archetypes"].version == "m9"
    # steering without a model, or outside [0, 1], or without balanced selection: refused
    args = p.parse_args(["--archetype-steering", "0.5", "--selection", "balanced", "--sources", "list"])
    args.out = tmp_path / "out2"
    with pytest.raises(ValueError, match="archetypes.json"):
        cli._build(args, chain_cfg, out_dir=args.out, overwrite=False)
    args = p.parse_args(["--archetype-steering", "1.5", "--sources", "list"])
    args.out = tmp_path / "out3"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cli._build(args, chain_cfg, out_dir=args.out, overwrite=False)
