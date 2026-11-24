"""R13 / R12: elastic lane D eval hosts on Lium only (README.md)."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from .helpers import REPO, T0, FakeClock, FakeLium, config, mod

H200 = [("H200", {"id": f"ex-{i}", "price_per_hour": 3.0 + i}) for i in range(5)]


def _ledger(m, tmp_path):
    return m.SpendLedger(tmp_path / "state" / "spend.jsonl")


def test_eval_role_is_lium_only_no_shadeform_gpu_path():
    m = mod()
    src = inspect.getsource(m.eval_up)
    assert "Shadeform" not in src.replace("never Shadeform", "")
    assert m.EvalConfig().provider == "lium"


def test_qualified_list_from_scoring_envs(tmp_path):
    m = mod()
    envs = tmp_path / "scoring_envs.toml"
    envs.write_text(
        '[[env]]\nname = "cu124-torch241"\nstatus = "qualified"\n'
        'gpu_models = ["NVIDIA H200"]\n'
        '[[env]]\nname = "sm120-torch212"\nstatus = "candidate"\n'
        'gpu_models = ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]\n')
    cfg = replace(m.EvalConfig(), scoring_envs=str(envs))
    keep, refused = m.qualified_candidates(cfg)
    assert [c.lium_gpu_type for c in keep] == ["H200"]
    assert refused == [{"gpu_type": "Edition", "reason": "not_qualified"}]
    # the authoritative list without H200 refuses every candidate
    envs.write_text('[[env]]\nname = "x"\nstatus = "qualified"\ngpu_models = ["NVIDIA A100"]\n')
    assert m.qualified_candidates(cfg)[0] == []


def test_fallback_list_only_without_scoring_envs(tmp_path):
    m = mod()
    cfg = replace(m.EvalConfig(), scoring_envs=str(tmp_path / "absent.toml"))
    models, src = m.qualified_gpu_models(cfg)
    assert src == "fallback_qualified_gpu_models" and models == {"NVIDIA H200"}


@pytest.mark.parametrize("entries,cap,maxh,want", [
    (0, 2, 4, 0), (1, 2, 4, 1), (2, 2, 4, 1), (3, 2, 4, 2), (5, 2, 4, 3), (40, 2, 4, 4)])
def test_hosts_needed(entries, cap, maxh, want):
    m = mod()
    cfg = replace(m.EvalConfig(), per_host_entries=cap, max_hosts=maxh)
    assert m.hosts_needed(entries, cfg) == want


def test_up_rents_n_hosts_cheapest_first_under_cap(tmp_path):
    m = mod()
    cfg = config(tmp_path, max_price_per_hour=5.5)
    lium = FakeLium(executors=H200)
    out = m.eval_up(cfg, lium, entries=5, cohort="b100", dry_run=False, clock=FakeClock())
    assert out["hosts"] == 3
    assert [ex for _n, ex, _s in lium.launched] == ["ex-0", "ex-1", "ex-2"]
    assert all(i["price_per_hour"] <= 5.5 for i in out["instances"])
    assert [i["shard"] for i in out["instances"]] == [0, 1, 2]
    rows = _ledger(m, tmp_path).rows()
    assert [r["event"] for r in rows] == ["open"] * 3


def test_dry_run_rents_nothing(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    lium = FakeLium(executors=H200)
    out = m.eval_up(cfg, lium, entries=3, cohort="b1", dry_run=True, clock=FakeClock())
    assert out["dry_run"] and out["hosts"] == 2 and not lium.launched
    assert out["offers"][0] == {"gpu_type": "H200", "price_per_hour": 3.0}


def test_over_cap_offers_are_never_rented(tmp_path):
    m = mod()
    cfg = config(tmp_path, max_price_per_hour=2.0, retry_window_minutes=0)
    lium = FakeLium(executors=H200)
    with pytest.raises(m.ElasticError) as ei:
        m.eval_up(cfg, lium, entries=1, cohort="b1", dry_run=False, clock=FakeClock())
    assert ei.value.code == "no_capacity" and not lium.launched


def test_no_lium_gpu_retries_lium_then_alerts(tmp_path):
    m = mod()
    cfg = config(tmp_path, retry_window_minutes=10, retry_interval_seconds=120)
    lium = FakeLium(executors=[])
    clock = FakeClock()
    with pytest.raises(m.ElasticError) as ei:
        m.eval_up(cfg, lium, entries=1, cohort="b1", dry_run=False, clock=clock)
    assert ei.value.code == "no_capacity"
    assert lium.calls >= 4 and sum(clock.slept) <= 600
    assert "lane_d_eval_no_lium_gpu" in (tmp_path / "state" / "alerts.jsonl").read_text()


def test_not_qualified_refuses(tmp_path):
    m = mod()
    cfg = config(tmp_path, fallback_qualified_gpu_models=())
    with pytest.raises(m.ElasticError) as ei:
        m.eval_up(cfg, FakeLium(executors=H200), entries=1, cohort="b", dry_run=False,
                  clock=FakeClock())
    assert ei.value.code == "not_qualified"


def test_monthly_cap_limits_host_count(tmp_path):
    m = mod()
    cfg = config(tmp_path, monthly_cap_usd=20.0, max_price_per_hour=5.0)
    out = m.eval_up(cfg, FakeLium(executors=H200), entries=8, cohort="b", dry_run=True,
                    clock=FakeClock())
    # 4 hosts wanted; offers $3/$4/$5 under the cap; 3 h at $5 = $15/host: $20 buys 1
    assert 1 <= out["hosts"] < out["hosts_wanted"]
    assert "lane_d_eval_hosts_capped" in (tmp_path / "state" / "alerts.jsonl").read_text()


def test_lemon_is_recorded_and_every_rented_host_deleted(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    lium = FakeLium(executors=H200, ready=False)
    with pytest.raises(m.ElasticError):
        m.eval_up(cfg, lium, entries=4, cohort="b", dry_run=False, clock=FakeClock())
    assert lium.faults and lium.faults[0][1] == "lane_d_eval_boot_timeout"
    assert not lium.live
    rows = _ledger(m, tmp_path).rows()
    assert sorted(r["event"] for r in rows) == ["close", "open"]


def test_same_hosts_for_a_cohort_and_down_waits_for_it(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    lium = FakeLium(executors=H200)
    first = m.eval_up(cfg, lium, entries=3, cohort="b7", dry_run=False, clock=FakeClock())
    again = m.eval_up(cfg, lium, entries=3, cohort="b7", dry_run=False, clock=FakeClock())
    assert again["reused"] and again["instances"] == first["instances"]
    assert len(lium.launched) == 2
    with pytest.raises(m.ElasticError) as ei:
        m.eval_up(cfg, lium, entries=3, cohort="b8", dry_run=False, clock=FakeClock())
    assert ei.value.code == "cohort_open"
    with pytest.raises(m.ElasticError) as ei:
        m.down(cfg, m.EVAL_HOST, lium, clock=FakeClock())
    assert ei.value.code == "cohort_open" and len(lium.live) == 2
    out = m.down(cfg, m.EVAL_HOST, lium, cohort_done=True, clock=FakeClock())
    assert len(out["deleted"]) == 2 and not lium.live
    assert m.load_state(cfg, m.EVAL_HOST)["cohort"] is None


def test_eval_bootstrap_uses_the_r12_kit(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    inst = {"id": "p0", "ip": "198.51.100.1", "port": 40001, "user": "root",
            "profile": "scripts/evalhost/profiles/elastic_h200.toml"}
    steps = m.eval_bootstrap_steps(cfg, inst, release_dir=str(REPO))
    names = [n for n, _ in steps]
    assert names == ["keyscan", "pin_host_key", "smi_capture", "on_rent_check", "venv_sync"]
    for _n, argv in steps:
        j = " ".join(argv)
        assert "accept-new" not in j and "UserKnownHostsFile=/dev/null" not in j
    assert (REPO / inst["profile"]).is_file()


# ── multi-host sharding (owner 2026-11-24) ─────────────────────────────────


BH = "0x" + "ab" * 32


def test_assignment_is_deterministic_and_anchors_every_host():
    m = mod()
    ids = [f"e{i}" for i in range(7)]
    a = m.assign_shards(BH, ids, 3, king="KING", house="HOUSE")
    b = m.assign_shards(BH, list(reversed(ids)), 3, king="KING", house="HOUSE")
    assert a == b
    assert sorted(e for s in a for e in s["entries"]) == sorted(ids)
    assert all(s["anchors"] == ["KING", "HOUSE"] for s in a)
    assert [len(s["entries"]) for s in a] == [3, 2, 2]
    c = m.assign_shards("0x" + "cd" * 32, ids, 3, king="KING", house="HOUSE")
    assert c != a                                    # the block hash drives it
    assert len(m.assign_shards(BH, ["e0"], 3, king="KING")) == 1   # never an idle shard


def _score(model, window):
    """A deterministic fake forecaster: identical on every host."""
    seed = sum(map(ord, f"{model}|{window}"))
    rng = np.random.default_rng(seed)
    base = 1.0 if model == "KING" else (0.9 if model.startswith("good") else 1.05)
    return (float(base * (1 + 0.05 * rng.random())), float(base * (1 + 0.05 * rng.random())))


def _host_run(models, windows):
    return {mdl: {w: _score(mdl, w) for w in windows} for mdl in models}


def _verdict(king_rows, member_rows, windows):
    from types import SimpleNamespace

    from cascade.validator.lane_d import judge_cohort

    cfg = SimpleNamespace(alpha=0.05, bootstrap_B=200, d_margin=0.0)
    arr = lambda rows: (np.array([rows[w][0] for w in windows]),  # noqa: E731
                        np.array([rows[w][1] for w in windows]))
    names = sorted(member_rows)
    res = judge_cohort(arr(king_rows), [arr(member_rows[n]) for n in names],
                       clusters=list(range(len(windows))), cfg=cfg, seed="s")
    return dict(zip(names, res, strict=True))


def test_three_identical_hosts_give_the_one_host_verdict():
    m = mod()
    windows = [f"w{i}" for i in range(40)]
    entries = ["good-a", "good-b", "bad-c", "bad-d", "good-e"]
    one = _host_run(["KING", "HOUSE", *entries], windows)
    v1 = _verdict(one["KING"], {e: one[e] for e in [*entries, "HOUSE"]}, windows)

    shards = m.assign_shards(BH, entries, 3, king="KING", house="HOUSE")
    runs = [{"host": f"h{s['shard']}", "rows": _host_run(s["anchors"] + s["entries"], windows)}
            for s in shards]
    merged = m.merge_shard_scores(runs, anchors=["KING", "HOUSE"])
    assert set(merged["anchor_by_host"]["KING"]) == {"h0", "h1", "h2"}
    assert set(merged["host_of"]) == set(entries)
    members = {**merged["members"], "HOUSE": merged["anchors"]["HOUSE"]}
    v3 = _verdict(merged["anchors"]["KING"], members, windows)
    assert v3 == v1


def test_merge_refuses_a_king_that_differs_between_hosts():
    m = mod()
    windows = ["w0", "w1"]
    r0 = _host_run(["KING", "e1"], windows)
    r1 = _host_run(["KING", "e2"], windows)
    r1["KING"]["w0"] = (9.0, 9.0)
    with pytest.raises(m.ElasticError) as ei:
        m.merge_shard_scores([{"host": "h0", "rows": r0}, {"host": "h1", "rows": r1}],
                             anchors=["KING"])
    assert ei.value.code == "host_mismatch"
    with pytest.raises(m.ElasticError):
        m.merge_shard_scores([{"host": "h0", "rows": r0}, {"host": "h1", "rows": r0}],
                             anchors=["KING"])          # e1 on two hosts


def test_shard_plan_maps_shards_to_live_hosts(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    lium = FakeLium(executors=H200)
    m.eval_up(cfg, lium, entries=4, cohort="b9", dry_run=False, clock=FakeClock(T0))
    plan = m.eval_shard_plan(cfg, block_hash=BH, entry_ids=["e1", "e2", "e3", "e4"],
                             king="KING", house="HOUSE")
    assert plan["cohort"] == "b9" and len(plan["shards"]) == 2
    assert {s["instance"] for s in plan["shards"]} == {n for n, _e, _s in lium.launched}
