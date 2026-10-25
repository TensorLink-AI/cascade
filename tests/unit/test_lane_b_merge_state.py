"""DEC-CA-0050 "state at merge" + merge guards: what an era init carries
after a lane-B merge (F1), expert-row exclusivity until the init consumes a
merge (F2), the recorded ``exclude`` (F3), the affinity>0 selection rule and
the consensus-gated non-finite / overlap refusals."""

from __future__ import annotations

import dataclasses
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file  # noqa: E402

from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.trainer.ckpt_digest import save_tensors_hashed  # noqa: E402
from cascade.trainer.expert_lane import (  # noqa: E402
    _moe_names,
    apply_merges,
    check_finite_rows,
    merge_row_overlap,
    select_routed_experts,
    settle_leg,
)
from cascade.trainer.moe_model import MergeError  # noqa: E402
from cascade.trainer.moe_trainer import POWER_EMA_FILE, build_moe_optimizer  # noqa: E402
from cascade.trainer.toto2_trainer import (  # noqa: E402
    OPTIM_STATE_FILE,
    STABLE_WEIGHTS_FILE,
    Toto2Trainer,
)

from .test_lane_b_driver import Ops, _kings  # noqa: E402
from .test_moe_trainer import _C, _train  # noqa: E402

# a lineage checkpoint as the mainnet preset writes it: scored power EMA +
# raw weights_stable + optimizer (wsd) + the EMA shadows
LC = dataclasses.replace(_C(), copula_loss_cap=0.0, power_ema_eval=0)
SEL = {0: [1, 2, 3]}


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merge-state")
    _train(tmp / "king", LC, budget=600)
    for f in ("weights.safetensors", STABLE_WEIGHTS_FILE, OPTIM_STATE_FILE, POWER_EMA_FILE):
        assert (tmp / "king" / f).is_file(), f
    _train(tmp / "leg", LC, budget=600, warm=tmp / "king", sel=SEL, seed=3)
    settle_leg(tmp / "king", tmp / "king", tmp / "leg", SEL, tmp / "rebased")
    return tmp


def _driver(tmp, lb_extra=None):
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
                       "min_experts": 2, "max_experts": 4, "bootstrap_B": 300,
                       "min_target_windows": 20, "min_guard_windows": 20, **(lb_extra or {})})
    ops = Ops(tmp, {"energy": 0.9})
    ops.cfg = SimpleNamespace(lane_b=lb)
    from cascade.trainer.lane_b_driver import LaneBDriver

    return LaneBDriver(ops.cfg, ops, tmp / "lane_b.json"), ops


def _merged(lineage, tmp_path):
    drv, ops = _driver(tmp_path)
    king = ops._ptr(lineage / "king")
    reb = ops._ptr(lineage / "rebased")
    m = {"rebased_pointer": reb, "king_pointer": king,
         "selection": {str(k): v for k, v in SEL.items()}, "round_id": "r1"}
    drv.st.pending_merges = [m]
    ptr, merges = drv.init_for(2, king, tmp_path / "w")
    assert merges == [m] and ptr != king
    return ops.fetch(ptr), m


def test_f1_merged_init_carries_the_delta_into_every_lineage_file(lineage, tmp_path):
    out, m = _merged(lineage, tmp_path)
    names = _moe_names(lineage / "king")
    king = load_file(str(lineage / "king" / "weights.safetensors"))
    reb = load_file(str(lineage / "rebased" / "weights.safetensors"))
    # the scored file: exactly apply_merges' bytes (what validators rebuild)
    want = apply_merges(dict(king), [m], lambda p: reb if p == m["rebased_pointer"] else king,
                        names)
    got = load_file(str(out / "weights.safetensors"))
    assert want.keys() == got.keys() and all(torch.equal(want[k], got[k]) for k in want)
    rows = torch.tensor(SEL[0])
    keep = torch.tensor([e for e in range(king[names[0][0]].shape[0]) if e not in SEL[0]])
    # weights_stable: the same delta on the same rows, nothing else moved
    st0 = load_file(str(lineage / "king" / STABLE_WEIGHTS_FILE))
    st1 = load_file(str(out / STABLE_WEIGHTS_FILE))
    for n in names[0]:
        delta = reb[n][rows].float() - king[n][rows].float()
        assert delta.abs().sum() > 0
        assert torch.equal(st1[n][rows], (st0[n][rows].float() + delta).to(st0[n].dtype))
        assert torch.equal(st1[n][keep], st0[n][keep])
    assert all(torch.equal(st0[k], st1[k]) for k in st0 if k not in names[0])
    # optimizer: merged rows' moments zero, every other row / tensor untouched
    o0 = load_file(str(lineage / "king" / OPTIM_STATE_FILE))
    o1 = load_file(str(out / OPTIM_STATE_FILE))
    assert o0.keys() == o1.keys()
    for n in names[0]:
        for st in ("momentum", "row_v"):
            k = f"muon.{n}.{st}"
            assert o0[k][rows].abs().sum() > 0          # there was state to reset
            assert not o1[k][rows].any()
            assert torch.equal(o1[k][keep], o0[k][keep])
    assert all(torch.equal(o0[k], o1[k]) for k in o0
               if not any(k.startswith(f"muon.{n}.") for n in names[0]))
    # power EMA: every shadow carries the delta on the merged rows
    p0 = load_file(str(lineage / "king" / POWER_EMA_FILE))
    p1 = load_file(str(out / POWER_EMA_FILE))
    pres = [k[: -len(".__t__")] for k in p0 if k.endswith(".__t__")]
    assert len(pres) == 2
    for pre in pres:
        assert torch.equal(p0[f"{pre}.__t__"], p1[f"{pre}.__t__"])
        for n in names[0]:
            delta = reb[n][rows].float() - king[n][rows].float()
            k = f"{pre}.{n}"
            assert torch.equal(p1[k][rows], (p0[k][rows].float() + delta).to(p0[k].dtype))
            assert torch.equal(p1[k][keep], p0[k][keep])


def test_f1_lane_a_warm_start_from_the_merged_init(lineage, tmp_path, monkeypatch):
    """One lane-A step from the merged init: it trains from the merged raw
    weights (weights_stable, not the scored EMA) and resumes an optimizer whose
    moments on the merged rows are zero."""
    out, _ = _merged(lineage, tmp_path)
    names = _moe_names(lineage / "king")
    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel

    model = CascadeMoEModel(CascadeMoEConfig.from_contract(LC))
    opt = build_moe_optimizer(model, LC)
    tr = Toto2Trainer(device="cpu")
    assert tr._load_optimizer_state(out, opt, model)
    params = dict(model.named_parameters())
    for n in names[0]:
        buf = opt._bufs[params[n]]                    # (E, d_out, d_in) NorMuon view
        assert not buf[SEL[0]].any()
        assert buf[[0, 4]].abs().sum() > 0
    import safetensors.torch as st

    seen, real = [], st.load_file
    monkeypatch.setattr(st, "load_file", lambda p, *a, **k: (seen.append(str(p)),
                                                             real(p, *a, **k))[1])
    res = _train(tmp_path / "next", LC, budget=1, warm=out)
    assert res.metrics["optim_state_resumed"] is True and res.metrics["steps"] == 1
    assert str(out / STABLE_WEIGHTS_FILE) in seen
    assert str(out / "weights.safetensors") not in seen


def test_f2_init_for_defers_an_overlapping_merge(lineage, tmp_path):
    drv, ops = _driver(tmp_path)
    king = ops._ptr(lineage / "king")
    reb = ops._ptr(lineage / "rebased")
    a = {"rebased_pointer": reb, "king_pointer": king, "selection": {"0": [1, 2, 3]},
         "round_id": "r1"}
    b = {**a, "selection": {"0": [3, 5]}, "round_id": "r2"}
    drv.st.pending_merges = [a, b]
    _ptr, merges = drv.init_for(2, king, tmp_path / "w", era_start=0)
    assert merges == [a] and drv.st.pending_merges == [b]     # b waits for the next era


def test_f2_rows_stay_exclusive_until_the_init_consumes_the_merge(tmp_path):
    from cascade.trainer.expert_lane import LaneBQueue, LaneBSubmission

    q = LaneBQueue()
    sub = LaneBSubmission("alice", 7, "a/b@sha256:" + "a" * 64, "energy", 5)
    lb = parse_lane_b({"from_block": 1, "domains": ["energy"]})
    q.submit(sub, lb)
    with pytest.raises(ValueError):
        q.start(sub, {0: [2, 3]}, "k", 1, held={0: {3}})
    leg = q.start(sub, {0: [2, 4]}, "k", 1, held={0: {3}})
    assert leg.selection == {0: [2, 4]}


def test_f2_driver_plans_around_finished_legs_and_pending_merges(tmp_path):
    _train(tmp_path / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    drv, ops = _driver(tmp_path)
    prev, king = _kings(tmp_path, ops)
    E = load_file(str(tmp_path / "king" / "weights.safetensors"))[
        "blocks.1.ffn.w_in"].shape[0]
    held_merge, held_fin = list(range(0, E // 2)), [E // 2]
    drv.st.pending_merges = [{"rebased_pointer": king, "king_pointer": king,
                              "selection": {"0": held_merge}}]
    drv.st.finished = [{"leg": {"selection": {"0": held_fin}, "sub": {"hotkey": "z"}},
                        "leg_pointer": king}]
    drv.add_lineage_tokens(1000)
    ops.dispatch_leg = lambda leg, era, block, done: None      # keep the leg open
    drv.tick(1, 0, prev)
    (leg,) = drv.tick(2, 1, king)
    assert not set(leg.selection[0]) & set(held_merge + held_fin)
    assert set(held_merge + held_fin) <= set(leg.exclude[0])


def test_f2_strict_rebuild_refuses_overlap_and_non_finite():
    names = [("blocks.1.ffn.w_in", "blocks.1.ffn.w_out")]
    a = {"selection": {"0": [1, 2]}}
    b = {"selection": {"0": [2, 7]}}
    assert merge_row_overlap([a, b]) == (0, 2)
    assert merge_row_overlap([a, {"selection": {"0": [3]}}]) is None
    k = {n: torch.zeros(8, 2, 2) for n in names[0]}
    c = {n: t.clone() for n, t in k.items()}
    c[names[0][0]][1] = 1.0
    c2 = {n: t.clone() for n, t in k.items()}
    c2[names[0][0]][3] = 2.0
    store = {"k": k, "c": c, "c2": c2}
    ma = {"king_pointer": "k", "rebased_pointer": "c", "selection": {"0": [1, 2]}}
    mb = {**ma, "rebased_pointer": "c2", "selection": {"0": [2, 3]}}
    apply_merges(dict(k), [ma, mb], store.get, names)            # pre-guard: accepted
    with pytest.raises(MergeError, match="overlap"):
        apply_merges(dict(k), [ma, mb], store.get, names, strict=True)
    c[names[0][1]][2, 0, 0] = float("nan")
    apply_merges(dict(k), [ma], store.get, names)
    with pytest.raises(MergeError, match="non-finite"):
        apply_merges(dict(k), [ma], store.get, names, strict=True)
    with pytest.raises(MergeError):
        check_finite_rows(c, {0: [2]}, names)
    check_finite_rows(c, {0: [1]}, names)                          # other rows are fine


def test_merge_guard_config_and_load_check():
    lb = parse_lane_b({"from_block": 1001, "domains": ["energy"]})
    assert lb.merge_guard_from_block == 1000                       # "v2" default
    assert lb.merge_guard_at_era(1000) and not lb.merge_guard_at_era(999)
    assert lb.merge_guard_at_settlement(1001) and not lb.merge_guard_at_settlement(1000)
    off = parse_lane_b({"from_block": 1001, "domains": ["energy"], "merge_guard_from_block": 0})
    assert not off.merge_guard_at_era(10**9)
    assert parse_lane_b({"domains": ["energy"]}).merge_guard_from_block == 0
    with pytest.raises(ValueError):
        parse_lane_b({"domains": ["energy"], "merge_guard_from_block": "soon"})
    with pytest.raises(ValueError):
        parse_lane_b({"domains": ["energy"], "leg_funding": "both"})
    import tomllib
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    raw = tomllib.loads(Path("chain.testnet.toml").read_text())
    assert load_chain_config(raw=raw).lane_b.merge_guard_from_block == 8280000
    raw["lane_b"]["merge_guard_from_block"] = 8280001              # not an era boundary
    with pytest.raises(ValueError, match="era boundary"):
        load_chain_config(raw=raw)
    raw["lane_b"]["merge_guard_from_block"] = 8189400              # before [v2] from_block
    with pytest.raises(ValueError, match="era boundary"):
        load_chain_config(raw=raw)
    main = load_chain_config("chain.toml").lane_b
    assert main.leg_funding == "miner" and main.merge_guard_from_block == 0   # = v2 (unset)


def test_validator_era_init_refuses_overlapping_merges_under_the_guard(cfg, tmp_path):
    from cascade.shared.manifest import TrainedEntry, TrainingManifest
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    from .test_lane_b_multi_target import GEN, _ptr

    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"]})
    v = dataclasses.replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    r = ValidatorRunner(cfg=dataclasses.replace(cfg, lane_b=lb, validator=v),
                        state=genesis("K", 0), verify_signatures=False)
    a = {"rebased_pointer": _ptr("r"), "king_pointer": _ptr("k"), "selection": {"0": [1, 2]}}
    b = {"rebased_pointer": _ptr("s"), "king_pointer": _ptr("k"), "selection": {"0": [2, 9]}}
    m = TrainingManifest(
        round_id="2", created_block=2000, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", GEN, _ptr("king"), "3" * 64, 1999)],
        warm_start_ckpt=_ptr("merged"), warm_start_base=_ptr("member"),
        warm_start_merges=(a, b))
    assert "overlap on MoE layer 0 expert row 2" in r._verify_merged_init(m, strict=True)
    # pre-guard the overlap is not a reason (it fails later, unrecorded)
    assert "not a lane-B win" in r._verify_merged_init(m)


def test_validator_judge_refuses_non_finite_rows_only_under_the_guard(tmp_path):
    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest
    from cascade.shared.manifest import format_trained_pointer
    from cascade.validator.expert_lane import judge_entry

    from .test_validator_expert_lane import _scores, _ws

    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp_path / "king", c, budget=600)
    shutil.copytree(tmp_path / "king", tmp_path / "cand")
    w = load_file(str(tmp_path / "cand" / "weights.safetensors"))
    w["blocks.1.ffn.w_in"][2, 0, 0] = float("inf")
    save_tensors_hashed(w, tmp_path / "cand" / "weights.safetensors")
    kp = format_trained_pointer("cascade/king@sha256:" + "1" * 64)
    cp = format_trained_pointer("cascade/cand@sha256:" + "2" * 64)
    dirs = {kp: tmp_path / "king", cp: tmp_path / "cand"}
    e = ExpertEntry(miner_hotkey="a", miner_uid=1, gen_ref="a/b@sha256:" + "a" * 64,
                    domain="energy", selection={0: [1, 2]}, exclude={},
                    base_king_pointer=kp, rebased_pointer=cp, corpus_digest="d",
                    train_block=1, commit_block=1)
    man = ExpertManifest(round_id="1", created_block=1, king_pointer=kp, lineage_tokens=0,
                         entries=(e,))
    rows = _scores({"energy": 0.9})
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "bootstrap_B": 300,
                       "min_target_windows": 20, "min_guard_windows": 20})
    kw = dict(manifest=man, king_pointer=kp, king_scores=_ws(rows, False),
              evaluate=lambda p: _ws(rows, True), fetch_dir=dirs.get, cfg=lb, seed="1")
    assert judge_entry(e, strict=True, **kw).stage == "merge"
    assert judge_entry(e, **kw).stage == "verdict"


def test_selection_never_pads_with_zero_affinity_experts():
    aff = np.zeros((2, 16), dtype=np.int64)
    aff[0, [3, 5]] = [10, 4]
    aff[1, :12] = 1
    sel = select_routed_experts(aff, p=0.2, min_n=8, max_n=38)
    assert sel[0] == [3, 5]                       # 2 routed experts, never padded to 8
    assert len(sel[1]) == 8 and set(sel[1]) <= set(range(12))
    sel = select_routed_experts(aff, p=0.2, min_n=8, max_n=38, exclude={0: {3}})
    assert sel[0] == [5]
    from cascade.trainer.moe_model import select_domain_experts

    # the digest-pinned rule itself is unchanged (it still pads)
    assert len(select_domain_experts(torch.as_tensor(aff), p=0.2, min_n=8, max_n=38)[0]) == 8


def test_f3_manifest_records_the_legs_exclusions(tmp_path):
    _train(tmp_path / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    drv, ops = _driver(tmp_path)
    prev, king = _kings(tmp_path, ops)
    drv.st.pending_merges = [{"rebased_pointer": king, "king_pointer": king,
                              "selection": {"0": [0]}}]
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    drv.tick(2, 1, king)
    drv.st.pending_merges = []
    drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
               work_dir=tmp_path / "w")
    (entry,) = ops.published[0].entries
    assert entry.exclude == {0: [0]}


def test_audit_replays_the_overlap_guard():
    from cascade.audit.checks import check_lane_b

    lb = parse_lane_b({"from_block": 11, "domains": ["energy"]})
    merges = [{"rebased_pointer": "a", "king_pointer": "k", "selection": {"0": [1, 2]}},
              {"rebased_pointer": "b", "king_pointer": "k", "selection": {"0": [2]}}]
    rec = SimpleNamespace(epoch_start_block=20, lane_b=[], lane_b_pay={},
                          manifest={"warm_start_merges": merges})
    cfg = SimpleNamespace(lane_b=lb)
    res = check_lane_b(rec, cfg)
    assert res.status == "FAIL" and "overlap" in res.detail
    off = SimpleNamespace(lane_b=dataclasses.replace(lb, merge_guard_from_block=0))
    assert check_lane_b(rec, off).status != "FAIL"
