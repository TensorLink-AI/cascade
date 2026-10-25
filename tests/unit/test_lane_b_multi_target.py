"""DEC-CA-0059: multi-target lane-B legs (up to ``max_domains_per_leg``
declared targets: domains and horizon slots), judged per target at alpha/k
with the guards run once, paid per slot, merged once."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.expert_lane import (
    ExpertLaneParams,
    judge_expert_leg,
    judge_expert_targets,
)
from cascade.shared.config import load_chain_config, parse_lane_b
from cascade.shared.expert_manifest import (
    TARGET_FIELD_MAX,
    ExpertEntry,
    ExpertManifest,
    format_expert_commit,
    format_expert_targets,
    leg_label,
    load_expert_manifest,
    parse_expert_commit,
    parse_expert_targets,
)
from cascade.shared.lane_b import (
    DomainWin,
    LaneBState,
    combined_weights,
    lane_b_payouts,
    lane_b_slots,
    merge_identities,
)
from cascade.shared.manifest import format_trained_pointer
from cascade.trainer.expert_lane import (
    LaneBQueue,
    LaneBSubmission,
    union_selection,
)
from cascade.validator import expert_lane as vel

ROOT = Path(__file__).resolve().parents[2]
GEN = "alice/metro-gen@sha256:" + "a" * 64
P = ExpertLaneParams(B=400, min_target_windows=20, min_guard_windows=20)


def _ptr(tag: str) -> str:
    return format_trained_pointer(
        f"cascade/ckpt-{tag}@sha256:" + hashlib.sha256(tag.encode()).hexdigest())


def _data(effect: dict, n_per=60, doms=("energy", "web", "sales"), horizons=None, seed=0):
    """Paired per-window arrays; ``effect[d]`` / ``effect["horizon:h"]``
    scales the challenger's losses on those windows (< 1 = better)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in doms:
        for i in range(n_per):
            h = None if horizons is None else horizons[i % len(horizons)]
            rows.append((d, h, i))
    n = len(rows)
    kq = rng.lognormal(0, 0.3, (n, 9))
    km = rng.lognormal(0, 0.3, n)
    f = np.array([effect.get(d, 1.0) * effect.get(f"horizon:{h}", 1.0) for d, h, _ in rows])
    return dict(domains=[d for d, _, _ in rows], horizons=[h for _, h, _ in rows],
                clusters=[f"{d}-feed{i % 12}" for d, _, i in rows],
                king_qloss=kq, king_abs_target=np.full(n, 50.0), king_mase=km,
                chal_qloss=kq * f[:, None], chal_abs_target=np.full(n, 50.0),
                chal_mase=km * f)


def _j(v) -> str:
    return json.dumps(asdict(v), sort_keys=True, default=str)   # NaN-safe equality


def _arrays(d):
    return {k: d[k] for k in ("domains", "clusters", "king_qloss", "king_abs_target",
                              "king_mase", "chal_qloss", "chal_abs_target", "chal_mase")}


# ── 1. single-domain legs stay bit-identical ─────────────────────────────────


def test_single_domain_commit_and_manifest_bytes_unchanged():
    lb = load_chain_config(ROOT / "chain.toml").lane_b
    for d in lb.domains:
        assert format_expert_targets((d,), GEN, lb.targets) == format_expert_commit(d, GEN)
        assert parse_expert_targets(format_expert_commit(d, GEN), lb.targets) == ((d,), GEN)
    # a multi-target payload is invisible to the pre-0059 parser
    multi = format_expert_targets(("energy", "sales"), GEN, lb.targets)
    assert parse_expert_commit(multi) is None
    # the canonical body of a single-target entry is the pre-0059 body
    e = ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref=GEN, domain="energy",
                    selection={1: [3, 2]}, exclude={}, base_king_pointer=_ptr("k"),
                    rebased_pointer=_ptr("r"), corpus_digest="c", train_block=5,
                    commit_block=4)
    m = ExpertManifest(round_id="9", created_block=10, king_pointer=_ptr("k"),
                       lineage_tokens=1, entries=(e,))
    old = asdict(m)
    old.pop("signature")
    for x in old["entries"]:
        x.pop("targets")
        x["selection"] = {str(k): sorted(v) for k, v in sorted(x["selection"].items())}
        x["exclude"] = {}
    assert m.canonical_body() == json.dumps(old, sort_keys=True,
                                            separators=(",", ":")).encode()
    assert load_expert_manifest(m.to_json()).digest() == m.digest()


def test_single_target_verdict_equals_judge_expert_leg():
    for effect in ({"energy": 0.9}, {"energy": 0.9, "web": 1.08}, {"energy": 1.0}):
        d = _data(effect)
        old = judge_expert_leg(target_domain="energy", params=P, seed="s", **_arrays(d))
        new = judge_expert_targets(targets=("energy",), params=P, seed="s", **_arrays(d))
        assert new.alpha == P.alpha
        assert _j(new.verdicts["energy"]) == _j(old)
    # inconclusive: no guards in either
    d = _data({"energy": 0.9}, n_per=10)
    old = judge_expert_leg(target_domain="energy", params=P, seed="s", **_arrays(d))
    new = judge_expert_targets(targets=("energy",), params=P, seed="s", **_arrays(d))
    assert _j(new.verdicts["energy"]) == _j(old) and old.inconclusive == "windows"


def test_single_target_receipt_row_unchanged():
    d = _data({"energy": 0.9})
    old = judge_expert_leg(target_domain="energy", params=P, seed="s", **_arrays(d))
    e = ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref=GEN, domain="energy",
                    selection={0: [1]}, exclude={}, base_king_pointer=_ptr("k"),
                    rebased_pointer=_ptr("r"), corpus_digest="c", train_block=5,
                    commit_block=4)
    j = vel.ExpertJudgement(e, old.passed, "verdict", old.reason, old)
    assert vel.judgements_record([j]) == [{
        "hotkey": "alice", "domain": "energy", "stage": "verdict", "passed": old.passed,
        "reason": old.reason, "target_lcb": old.target_lcb, "guard_lcbs": old.guard_lcbs,
        "public_lcbs": old.public_lcbs}]


def test_testnet_has_no_horizon_targets_and_mainnet_has_ten_slots():
    tn = load_chain_config(ROOT / "chain.testnet.toml").lane_b
    assert tn.horizons == () and tn.targets == tn.domains and lane_b_slots(tn) == tn.domains
    mn = load_chain_config(ROOT / "chain.toml").lane_b
    assert mn.horizon_target_names == ("horizon:64", "horizon:256", "horizon:720")
    assert len(mn.targets) == 10 and mn.max_domains_per_leg == 3
    off = parse_lane_b({"domains": ["energy"], "horizon_targets": False}, horizons=(64,))
    assert off.targets == ("energy",)
    # testnet weight vector: the slot count is the domain count (share / 7)
    st = LaneBState()
    st.record(DomainWin(domain="energy", hotkey="a", uid=3, round_id="1", merge_block=100, target_lcb=0.03, tier=0.02, full_eras=5))
    vec = combined_weights([1], 8, king_decay=1.0, burn_uid=0, state=st, block=100,
                           era_blocks=10, cfg=dataclasses.replace(tn, from_block=1))
    assert vec[3] == pytest.approx(tn.share / 7)
    vec = combined_weights([1], 8, king_decay=1.0, burn_uid=0, state=st, block=100,
                           era_blocks=10, cfg=dataclasses.replace(mn, from_block=1))
    assert vec[3] == pytest.approx(mn.share / 10)


def test_domain_names_cannot_collide_with_horizon_codes():
    with pytest.raises(ValueError, match="horizon code"):
        parse_lane_b({"domains": ["energy", "h64"]})
    with pytest.raises(ValueError, match="max_domains_per_leg"):
        parse_lane_b({"domains": ["energy"], "max_domains_per_leg": 0})


# ── 2. the commitment encoding ───────────────────────────────────────────────


def test_three_target_commit_fits_the_single_target_bound():
    lb = load_chain_config(ROOT / "chain.toml").lane_b
    long_ref = "a-very-long-namespace/" + "g" * 40 + "@sha256:" + "f" * 64
    longest_single = max(len(format_expert_commit(d, long_ref)) for d in lb.domains)
    worst = ""
    for k in (2, 3):
        for combo in itertools.combinations(lb.targets, k):
            p = format_expert_targets(combo, long_ref, lb.targets, max_targets=3)
            field_ = p.split(":")[2]
            assert len(field_) <= TARGET_FIELD_MAX, p
            # a multi-target payload is never longer than the longest
            # single-domain payload the grammar already allows (32 chars)
            assert len(p) <= len(long_ref) + len("metro-v1:expert::hippius:") + 32
            assert parse_expert_targets(p, lb.targets, max_targets=3) == (combo, long_ref)
            worst = max(worst, p, key=len)
    assert len(worst) <= longest_single + (TARGET_FIELD_MAX - len("web_cloudops"))
    # the 33-char full-name combination falls back to unique prefixes
    p = format_expert_targets(("healthcare", "transport", "web_cloudops"), GEN, lb.targets)
    assert p.startswith("metro-v1:expert:hea+tra+web:hippius:")
    assert format_expert_targets(("energy", "horizon:720"), GEN, lb.targets).startswith(
        "metro-v1:expert:energy+h720:hippius:")
    # four targets, unknown / ambiguous / repeated codes are refused
    with pytest.raises(ValueError):
        format_expert_targets(lb.targets[:4], GEN, lb.targets, max_targets=3)
    assert parse_expert_targets(f"metro-v1:expert:energy+sales+nature+web_cloudops:hippius:"
                                f"{GEN}", lb.targets, max_targets=3) is None
    assert parse_expert_targets(f"metro-v1:expert:e+sales:hippius:{GEN}", lb.targets) is None
    assert parse_expert_targets(f"metro-v1:expert:en+sales:hippius:{GEN}", lb.targets) is None
    assert parse_expert_targets(f"metro-v1:expert:sales+sal:hippius:{GEN}", lb.targets) is None
    assert parse_expert_targets(f"metro-v1:expert:h99+sales:hippius:{GEN}", lb.targets) is None
    # testnet: horizon codes do not resolve
    tn = load_chain_config(ROOT / "chain.testnet.toml").lane_b
    assert parse_expert_targets(f"metro-v1:expert:h64:hippius:{GEN}", tn.targets) is None


def test_sealed_commit_size_at_maximum_length():
    """The timelock path seals ``compact(len) + UTF-8`` (DEC-CA-0051): the
    ciphertext grows byte-for-byte with the plaintext, so a 3-target payload
    seals no larger than a single-domain payload of the same length."""
    bc = pytest.importorskip("bittensor_core")
    lb = load_chain_config(ROOT / "chain.toml").lane_b
    ref = "ns/" + "g" * 60 + "@sha256:" + "f" * 64
    multi = format_expert_targets(("econ_fin", "healthcare", "web_cloudops"), ref, lb.targets)
    single = format_expert_commit("web_cloudops", ref)
    sm = len(bytes(bc.get_encrypted_commitment(multi, 360, 12.0)[0]))
    ss = len(bytes(bc.get_encrypted_commitment(single, 360, 12.0)[0]))
    assert sm - ss == len(multi) - len(single) <= TARGET_FIELD_MAX - len("web_cloudops")
    # the commitments pallet bounds a TimelockEncrypted field at 1024 bytes
    # (subtensor MAX_TIMELOCK_COMMITMENT_SIZE_BYTES; not in this repo)
    assert sm <= 1024


# ── 3. joint union selection and row exclusivity ─────────────────────────────


def test_joint_union_selection_and_row_exclusivity():
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web", "sales"]})
    q = LaneBQueue()
    multi = LaneBSubmission("alice", 7, GEN, "energy", 5, targets=["energy", "web"])
    single = LaneBSubmission("bob", 8, GEN, "web", 6)
    other = LaneBSubmission("carol", 9, GEN, "sales", 7)
    for s in (multi, single, other):
        assert q.submit(s, lb)
    with pytest.raises(ValueError, match="max_domains_per_leg"):
        q.submit(LaneBSubmission("x", 1, GEN, "energy", 1,
                                 targets=["energy", "web", "sales"]),
                 dataclasses.replace(lb, max_domains_per_leg=2))
    # the multi leg holds capacity on BOTH its targets: bob (web) waits
    assert [s.hotkey for s in q.next_dispatchable(lb)] == ["alice", "carol"]
    sel = union_selection([{0: [1, 2], 1: [5]}, {0: [2, 3], 1: [6]}])
    assert sel == {0: [1, 2, 3], 1: [5, 6]}
    leg = q.start(multi, sel, _ptr("king"), 0)
    assert leg.sub.declared == ("energy", "web")
    with pytest.raises(ValueError, match="overlaps"):
        q.start(other, {0: [3]}, _ptr("king"), 0)          # a row of the union
    q.start(other, {0: [4]}, _ptr("king"), 0)
    assert q.next_dispatchable(lb) == []                    # web still held by alice
    q.finish(leg)
    assert [s.hotkey for s in q.next_dispatchable(lb)] == ["bob"]
    assert LaneBQueue.from_json(q.to_json()).to_json() == q.to_json()
    assert leg_label(("energy",)) == "energy" and leg_label(("energy", "horizon:64")) == \
        "energy.h64"


def test_validator_rederives_the_union_selection():
    calls = []

    def recompute(king, target, exclude):
        calls.append(target)
        return {"energy": {0: [1, 2]}, "web": {0: [2, 3]}}[target]

    e = ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref=GEN, domain="energy",
                    selection={0: [1, 2, 3]}, exclude={}, base_king_pointer=_ptr("k"),
                    rebased_pointer=_ptr("r"), corpus_digest="c", train_block=5,
                    commit_block=4, targets=("energy", "web"))
    m = ExpertManifest(round_id="9", created_block=10, king_pointer=_ptr("k"),
                       lineage_tokens=1, entries=(e,))
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"]})

    def _boom(p):
        raise RuntimeError("fetched")          # selection passed ⇒ reaches the fetch

    with pytest.raises(RuntimeError, match="fetched"):
        vel.judge_entry(e, manifest=m, king_pointer=_ptr("k"), king_scores=[],
                        evaluate=None, fetch_dir=_boom, cfg=lb, seed="s",
                        recompute_selection=recompute)
    assert calls == ["energy", "web"]
    bad = dataclasses.replace(e, selection={0: [1, 2]})
    j = vel.judge_entry(bad, manifest=m, king_pointer=_ptr("k"), king_scores=[],
                        evaluate=None, fetch_dir=_boom, cfg=lb, seed="s",
                        recompute_selection=recompute)
    assert j.stage == "selection" and not j.passed
    cap = dataclasses.replace(lb, max_domains_per_leg=1)
    j = vel.judge_entry(e, manifest=m, king_pointer=_ptr("k"), king_scores=[],
                        evaluate=None, fetch_dir=_boom, cfg=cap, seed="s")
    assert j.stage == "targets" and not j.passed


# ── 4/5. per-target alpha/k; guards once ─────────────────────────────────────


def test_per_target_alpha_over_k():
    d = _data({"energy": 0.93, "web": 0.93})
    two = judge_expert_targets(targets=("energy", "web"), params=P, seed="s", **_arrays(d))
    assert two.alpha == pytest.approx(P.alpha / 2)
    half = dataclasses.replace(P, alpha=P.alpha / 2)
    for t in ("energy", "web"):
        alone = judge_expert_leg(target_domain=t, params=half, seed="s",
                                 **{**_arrays(d), "domains": [x if x == t else
                                                              ("other" if x in ("energy", "web")
                                                               else x) for x in d["domains"]]})
        assert two.verdicts[t].target_lcb == pytest.approx(alone.target_lcb)
        assert two.verdicts[t].target_lcb < judge_expert_leg(
            target_domain=t, params=P, seed="s", **_arrays(d)).target_lcb
    # declared targets are never guards; the undeclared one is, once
    assert set(two.verdicts["energy"].guard_lcbs) == {"sales"}
    assert two.verdicts["energy"].guard_lcbs == two.verdicts["web"].guard_lcbs


def test_guard_failure_blocks_every_target_and_gain_failure_is_just_unpaid():
    d = _data({"energy": 0.9, "web": 0.9, "sales": 1.10})
    tv = judge_expert_targets(targets=("energy", "web"), params=P, seed="s", **_arrays(d))
    assert tv.guard_reason.startswith("regression on domain sales")
    assert tv.won == [] and all(v.reason == tv.guard_reason for v in tv.verdicts.values())
    d = _data({"energy": 0.9, "web": 1.0})
    tv = judge_expert_targets(targets=("energy", "web"), params=P, seed="s", **_arrays(d))
    assert tv.won == ["energy"] and tv.guard_reason == ""
    assert tv.verdicts["web"].reason.startswith("no gain")


def test_horizon_target_judged_on_its_windows_with_other_horizons_as_guards():
    hz = (64, 256)
    hts = ("horizon:64", "horizon:256")
    d = _data({"horizon:64": 0.9}, horizons=hz)
    tv = judge_expert_targets(targets=("horizon:64",), params=P, seed="s",
                              horizons=d["horizons"], horizon_targets=hts, **_arrays(d))
    v = tv.verdicts["horizon:64"]
    assert v.target_n == sum(1 for h in d["horizons"] if h == 64)
    # every domain + the undeclared horizon guard the leg
    assert set(v.guard_lcbs) == {"energy", "web", "sales", "horizon:256"}
    assert v.passed, v.reason
    d = _data({"horizon:64": 0.9, "horizon:256": 1.12}, horizons=hz)
    tv = judge_expert_targets(targets=("horizon:64",), params=P, seed="s",
                              horizons=d["horizons"], horizon_targets=hts, **_arrays(d))
    assert not tv.won
    assert "horizon:256" in tv.guard_reason or "domain" in tv.guard_reason
    assert vel.window_horizon("h64-s12") == 64 and vel.window_horizon("s12") is None


# ── 6/7. two won slots, one merge; independent decay ─────────────────────────


def _multi_judgement(effect, hotkey="alice", commit_block=4, tag="r"):
    d = _data(effect)
    e = ExpertEntry(miner_hotkey=hotkey, miner_uid=7, gen_ref=GEN, domain="energy",
                    selection={0: [1, 2, 3]}, exclude={}, base_king_pointer=_ptr("k"),
                    rebased_pointer=_ptr(tag), corpus_digest="c", train_block=5,
                    commit_block=commit_block, targets=("energy", "web"))
    tv = judge_expert_targets(targets=e.declared, params=P, seed="s", **_arrays(d))
    v = tv.verdicts["energy"]
    return vel.ExpertJudgement(e, bool(tv.won), "verdict", "x", v, king_pointer=_ptr("k"),
                               targets=tv)


def test_two_won_slots_record_two_wins_and_one_merge():
    j = _multi_judgement({"energy": 0.9, "web": 0.9})
    assert j.passed and j.targets.won == ["energy", "web"]
    st = LaneBState()
    wins = vel.apply_judgements(st, [j], round_id="r1", settle_block=1000)
    assert [w.domain for w in wins] == ["energy", "web"]
    assert len(merge_identities(wins)) == 1                 # one checkpoint, one delta
    assert st.has_merge(_ptr("r"), _ptr("k"), {"0": [1, 2, 3]})
    # re-applying the same round (catch-up / re-judge) does not stack
    again = LaneBState.from_json(st.to_json())
    vel.apply_judgements(again, [j], round_id="r1", settle_block=1000)
    assert again.wins == st.wins and again.history == st.history
    # the driver merges each WINNING leg once
    from cascade.trainer.lane_b_driver import _winning_legs

    assert _winning_legs([j]) == {_ptr("r")}
    # a better single-target leg on web takes that slot only
    from cascade.eval.expert_lane import judge_expert_leg as jl

    d = _data({"web": 0.8})
    sv = jl(target_domain="web", params=P, seed="s", **_arrays(d))
    se = dataclasses.replace(j.entry, miner_hotkey="bob", domain="web", targets=(),
                             rebased_pointer=_ptr("b"))
    sj = vel.ExpertJudgement(se, sv.passed, "verdict", sv.reason, sv, king_pointer=_ptr("k"))
    st2 = LaneBState()
    wins = vel.apply_judgements(st2, [j, sj], round_id="r1", settle_block=1000)
    assert {w.domain: w.hotkey for w in wins} == {"energy": "alice", "web": "bob"}
    assert _winning_legs([j, sj]) == {_ptr("r"), _ptr("b")}


def test_won_slots_decay_independently():
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"],
                       "full_pay_tiers": [[0.01, 1]],
                       "half_life_rounds": 1.0, "floor": 0.1, "win_ttl_eras": 0})
    st = LaneBState()
    j = _multi_judgement({"energy": 0.9, "web": 0.9})
    vel.apply_judgements(st, [j], round_id="r1", settle_block=100, tiers=lb.full_pay_tiers)
    pays = lane_b_payouts(st, block=100 + 2 * 10, era_blocks=10, cfg=lb, n_uids=16)
    assert [(p["domain"], p["age"]) for p in pays] == [("energy", 2), ("web", 2)]
    # a newer win on web restarts ONLY web's clock
    st.record(DomainWin(domain="web", hotkey="bob", uid=8, round_id="r2", merge_block=120,
                        rebased_pointer=_ptr("b"), king_pointer=_ptr("k2"),
                        selection={"0": [9]}, target_lcb=0.03, tier=0.01, full_eras=1))
    pays = {p["domain"]: p for p in lane_b_payouts(st, block=130, era_blocks=10, cfg=lb,
                                                   n_uids=16)}
    assert pays["energy"]["age"] == 3 and pays["web"]["age"] == 1
    assert pays["energy"]["multiplier"] < pays["web"]["multiplier"]
    assert pays["energy"]["uid"] == 7 and pays["web"]["uid"] == 8


# ── merge-once at the era init (validator) and catch-up ─────────────────────


def test_era_init_refuses_a_merge_declared_twice(cfg, tmp_path):
    from cascade.shared.manifest import TrainedEntry, TrainingManifest
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"]})
    v = dataclasses.replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    r = ValidatorRunner(cfg=dataclasses.replace(cfg, lane_b=lb, validator=v),
                        state=genesis("K", 0), verify_signatures=False)
    st = r._lane_b()
    vel.apply_judgements(st, [_multi_judgement({"energy": 0.9, "web": 0.9})],
                         round_id="r1", settle_block=1000)
    merge = {"rebased_pointer": _ptr("r"), "king_pointer": _ptr("k"),
             "selection": {"0": [1, 2, 3]}, "round_id": "r1"}
    m = TrainingManifest(
        round_id="2", created_block=2000, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", GEN, _ptr("king"), "3" * 64, 1999)],
        warm_start_ckpt=_ptr("merged"), warm_start_base=_ptr("member"),
        warm_start_merges=(merge, dict(merge)))
    assert "more than once" in r._verify_merged_init(m)
    fetched = []

    def _fetch(entry):
        fetched.append(entry.trained_pointer)
        raise OSError("stop here")

    r._fetch_checkpoint_dir = _fetch
    from cascade.shared.hippius import StorageError

    with pytest.raises(StorageError):          # listed once: recorded, then rebuilt
        r._verify_merged_init(dataclasses.replace(m, warm_start_merges=(merge,)))
    assert fetched


def test_catch_up_fold_of_a_multi_target_settlement_converges(cfg, tmp_path, monkeypatch):
    """A validator that missed the settlement folds it by catch-up to the
    same two wins (one merge identity); folding it again is a no-op."""
    from tests.unit import test_validator_lane_b_convergence as conv

    ms, store = conv._chain(cfg)
    # settlement 1's leg declares energy + web and wins both
    from cascade.shared.expert_manifest import expert_manifest_key

    em = load_expert_manifest(store.objs[expert_manifest_key(ms[1].round_id)])
    e = dataclasses.replace(em.entries[0], targets=("energy", "web"))
    em = dataclasses.replace(em, entries=(e,))
    store.objs[expert_manifest_key(ms[1].round_id)] = em.to_json()
    ms[1] = dataclasses.replace(ms[1], expert_manifest_sha256=em.digest())
    from cascade.shared.hippius import manifest_round_key
    from cascade.shared.manifest import dump_manifest

    store.objs[manifest_round_key(ms[1].round_id)] = dump_manifest(ms[1])

    def judge_manifest(em, **kw):
        out = []
        for e in em.entries:
            if e.domain not in kw["cfg"].domains:
                continue
            if e.targets:
                d = _data({"energy": 0.9, "web": 0.9})
                tv = judge_expert_targets(targets=e.declared, params=P, seed="s",
                                          **_arrays(d))
                out.append(vel.ExpertJudgement(e, bool(tv.won), "verdict", "x",
                                               tv.verdicts[e.domain],
                                               king_pointer=kw["king_pointer"], targets=tv))
            else:
                out.append(vel.ExpertJudgement(
                    e, e.gen_ref == conv.GEN, "verdict", "stub",
                    SimpleNamespace(target_lcb=0.01 * e.miner_uid, guard_lcbs={},
                                    public_lcbs={}), king_pointer=kw["king_pointer"]))
        return out

    monkeypatch.setattr(vel, "judge_manifest", judge_manifest)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    live = conv._runner(cfg, tmp_path / "a", store)
    for m in ms:
        conv._handle(live, m)
    late = conv._runner(cfg, tmp_path / "b", store)
    conv._handle(late, ms[-1])                       # catches up 0..N-2 first
    a, b = live._lane_b(), late._lane_b()
    assert a.history == b.history and a.wins == b.wins
    rid = ms[1].round_id
    from_leg = [w for w in a.history if w.round_id == rid]
    assert sorted(w.domain for w in from_leg) == ["energy", "web"]
    assert len(merge_identities(from_leg)) == 1
    hist = list(b.history)
    late._lane_b_fold_on_demand(rid)                 # already applied: no-op
    conv._handle(late, ms[-1])                       # re-judged round: no stacking
    assert late._lane_b().history == hist


# ── horizon selection from masked patches ────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_king(tmp_path_factory):
    pytest.importorskip("torch")
    from .test_moe_trainer import _C, _train

    tmp = tmp_path_factory.mktemp("hz")
    _train(tmp / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    return tmp / "king"


def test_horizon_selection_reads_the_masked_forecast_patches(tiny_king):
    import torch

    from cascade.trainer.expert_lane import (
        horizon_affinity_batches,
        horizon_decode_window,
        load_king,
        probe_horizon_routing,
        probe_target_routing,
    )

    lb = parse_lane_b({"domains": ["energy"], "min_experts": 2, "max_experts": 4},
                      horizons=(8, 64))
    model = load_king(tiny_king)
    c = model.cfg
    rng = np.random.default_rng(0)
    ctx = [np.cumsum(rng.normal(size=40)) for _ in range(6)]
    for h in (8, 64):
        cp, block = horizon_decode_window(h, patch_size=c.patch_size,
                                          context_patches=c.context_length // c.patch_size,
                                          max_patches=c.max_patches)
        assert block == min(-(-h // c.patch_size), c.max_patches - 2)
        assert cp + block <= c.max_patches
        batches = list(horizon_affinity_batches(
            ctx, horizon=h, patch_size=c.patch_size,
            context_patches=c.context_length // c.patch_size, max_patches=c.max_patches))
        for patches, mask, b in batches:
            assert b == block
            assert torch.all(mask[:, -block:] == 1) and torch.all(mask[:, :-block] == 0)
            assert torch.all(patches[:, -block:] == 0)      # no target value is an input
        probe = probe_horizon_routing(tiny_king, ctx, lb, h)
        assert probe.tokens == len(ctx) * block             # masked tokens only
        # the counted routing is exactly the masked tail of the forward
        patches, mask, b = batches[0]
        model(patches, mask)
        n, P_ = patches.shape[:2]
        tail = model.moe_layers()[0].last_topk.view(n, P_, -1)[:, P_ - b:, :].reshape(-1, c.top_k)
        assert np.array_equal(probe.topk[0][: tail.shape[0]], tail.numpy().astype(np.uint8))
        assert probe_target_routing(tiny_king, ctx, lb, f"horizon:{h}").affinity.tolist() == \
            probe.affinity.tolist()
        sel = probe.select(lb)
        assert all(2 <= len(v) <= 4 for v in sel.values())


def test_horizon_jaccard_script_smoke(tiny_king, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lane_b_horizon_jaccard", ROOT / "scripts" / "lane_b_horizon_jaccard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(tiny_king), "--synthetic", "4", "--length", "40",
                     "--horizons", "8", "16", "32", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert set(res["jaccard"]) == {"8/16", "8/32", "16/32"}
    assert all(0.0 <= x <= 1.0 for per in res["jaccard"].values() for x in per)


# ── trainer: a multi-target leg end to end (one merge, rollback-safe) ────────


def test_driver_multi_target_leg_wins_two_slots_and_merges_once(tmp_path):
    pytest.importorskip("torch")
    import torch
    from safetensors.torch import load_file

    from cascade.trainer.expert_lane import _moe_names, apply_merges

    from . import test_lane_b_driver as tld
    from .test_moe_trainer import _train

    _train(tmp_path / "king", tld.C, budget=600)
    drv, ops = tld._driver(tmp_path, {"energy": 0.9, "web": 0.9})
    payload = format_expert_targets(("energy", "web"), tld.GEN, drv.cfg.lane_b.targets)
    ops.commitments = lambda: [SimpleNamespace(hotkey="alice", uid=7, commit_block=5,
                                               payload=payload)]
    prev, king = tld._kings(tmp_path, ops)
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    legs = drv.tick(2, 1, king)
    assert [leg.sub.declared for leg in legs] == [("energy", "web")]
    js = drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
                    work_dir=tmp_path / "w")
    e = ops.published[0].entries[0]
    assert e.targets == ("energy", "web") and e.domain == "energy"
    assert js[0].targets.won == ["energy", "web"], js[0].reason
    st = LaneBState()
    assert [w.domain for w in vel.apply_judgements(st, js, round_id="r1",
                                                   settle_block=20)] == ["energy", "web"]
    assert len(drv.st.pending_merges) == 1                  # two slots, ONE merge
    # rollback: the era-init upload fails — the single merge stays pending
    real_upload = ops.upload

    def _fail_upload(d, repo):
        raise OSError("hub 503")

    ops.upload = _fail_upload
    with pytest.raises(OSError):
        drv.init_for(2, king, tmp_path / "w")
    assert len(drv.st.pending_merges) == 1 and "2" not in drv.st.merged_inits
    ops.upload = real_upload
    merged, merges = drv.init_for(2, king, tmp_path / "w")
    assert len(merges) == 1 and drv.st.pending_merges == []
    names = _moe_names(ops.fetch(king))
    load = lambda p: load_file(str(ops.fetch(p) / "weights.safetensors"))  # noqa: E731
    once = apply_merges(load(king), merges, load, names)
    twice = apply_merges(load(king), merges * 2, load, names)
    got = load(merged)
    assert all(torch.equal(once[k], got[k]) for k in once)
    assert any(not torch.equal(twice[k], got[k]) for k in twice)
    # a re-called settle for the round is a no-op (no second pending merge)
    assert drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
                      work_dir=tmp_path / "w") == []
    assert drv.st.pending_merges == []


# ── audit replay + tamper ────────────────────────────────────────────────────


def _audit_world(cfg):
    from cascade.shared.era import era_length_blocks

    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web", "sales"],
                       "min_target_windows": 20, "min_guard_windows": 20,
                       "bootstrap_B": 400}, horizons=())
    c = dataclasses.replace(cfg, lane_b=lb)
    block = 10 * era_length_blocks(c.round, 10**6)
    good = _multi_judgement({"energy": 0.9, "web": 0.9})
    rows = vel.judgements_record([good])
    eb = era_length_blocks(c.round, block)
    from cascade.eval.expert_lane import full_pay

    lcb = {x["domain"]: x["target_lcb"] for x in rows}
    pay = {"block": block, "era_blocks": eb, "wins": [
        {"domain": d, "hotkey": "alice", "uid": 7, "merge_block": block, "age": 0,
         "multiplier": 1.0, **full_pay(lcb[d], lb.full_pay_tiers).record()}
        for d in ("energy", "web")]}
    merge = {"rebased_pointer": _ptr("r"), "king_pointer": _ptr("k"),
             "selection": {"0": [1, 2, 3]}, "round_id": "r1"}
    rec = SimpleNamespace(epoch_start_block=block, lane_b=rows, lane_b_pay=pay,
                          manifest={"warm_start_merges": [merge]})
    return c, rec


def test_audit_rederives_multi_target_verdicts_and_fails_tampering(cfg):
    from cascade.audit.checks import FAIL, PASS, _lane_b_paid_wins, check_lane_b

    c, rec = _audit_world(cfg)
    r = check_lane_b(rec, c)
    assert r.status == PASS, r.detail
    assert len(_lane_b_paid_wins(rec, c, 16)[0]) == 2

    def _tamper(fn):
        t = SimpleNamespace(**{**rec.__dict__, "lane_b": [dict(x) for x in rec.lane_b],
                               "lane_b_pay": json.loads(json.dumps(rec.lane_b_pay)),
                               "manifest": json.loads(json.dumps(rec.manifest))})
        fn(t)
        return check_lane_b(t, c)

    cases = {
        "alpha": lambda t: t.lane_b[0].update(alpha=c.lane_b.alpha),          # not alpha/k
        "flip": lambda t: t.lane_b[1].update(passed=False),                   # re-derived True
        "gain": lambda t: t.lane_b[0].update(target_gain=-0.01),              # passed, no gain
        "guards-once": lambda t: t.lane_b[1].update(
            guard_lcbs={k: v + 0.001 for k, v in t.lane_b[1]["guard_lcbs"].items()}),
        "guard-blocks": lambda t: [x.update(guard_lcbs={**x["guard_lcbs"], "sales": -0.5},
                                            guard_reason="regression on domain sales")
                                   for x in t.lane_b],
        "declared-guard": lambda t: [x.update(guard_lcbs={**x["guard_lcbs"], "web": 0.0})
                                     for x in t.lane_b],
        "coverage": lambda t: t.lane_b.pop(),
        "cap": lambda t: [x.update(targets=["energy", "web", "sales", "nature"])
                          for x in t.lane_b],
        "merge-twice": lambda t: t.manifest["warm_start_merges"].append(
            dict(t.manifest["warm_start_merges"][0])),
        "slot-thief": lambda t: t.lane_b_pay["wins"][1].update(hotkey="mallory"),
        "stale-slot": lambda t: t.lane_b_pay["wins"][1].update(merge_block=0),
        "unknown-target": lambda t: t.lane_b[0].update(domain="crypto"),
        # gain-tiered full-pay (owner 2026-10-20): per target, from its own row
        "forged-tier": lambda t: t.lane_b_pay["wins"][1].update(full_eras=99),
        "foreign-lcb": lambda t: t.lane_b_pay["wins"][1].update(
            gain_lcb=t.lane_b_pay["wins"][1]["gain_lcb"] + 0.5),
    }
    for name, fn in cases.items():
        res = _tamper(fn)
        assert res.status == FAIL, (name, res.detail)
    # a guard failure that blocks both targets (honest rows) is consistent
    ok = _tamper(lambda t: [x.update(guard_lcbs={**x["guard_lcbs"], "sales": -0.5},
                                     guard_reason="regression on domain sales",
                                     passed=False, reason="regression on domain sales")
                            for x in t.lane_b] and t.lane_b_pay.update(wins=[]))
    assert ok.status == PASS, ok.detail


def test_audit_pays_horizon_slots_at_share_over_ten(cfg):
    from cascade.audit.checks import PASS, _lane_b_paid_wins, check_lane_b
    from cascade.shared.era import era_length_blocks

    lb = dataclasses.replace(load_chain_config(ROOT / "chain.toml").lane_b, from_block=1)
    c = dataclasses.replace(cfg, lane_b=lb)
    block = 10 * era_length_blocks(c.round, 10**6)
    pay = {"block": block, "era_blocks": era_length_blocks(c.round, block), "wins": [
        {"domain": "horizon:720", "hotkey": "h", "uid": 5, "merge_block": block - 1,
         "age": 0, "multiplier": 1.0, "gain_lcb": 0.03, "tier": 0.02, "full_eras": 5,
         "below_floor": False}]}
    rec = SimpleNamespace(epoch_start_block=block, lane_b=[], lane_b_pay=pay, manifest={})
    wins, why = _lane_b_paid_wins(rec, c, 16)
    assert why is None and [w.domain for w in wins] == ["horizon:720"]
    assert check_lane_b(rec, c).status == PASS
    from cascade.shared.emission import lane_parts

    _a, b = lane_parts([1], 16, king_decay=1.0, burn_uid=0, lane_b_cfg=lb, wins=wins)
    assert b[5] == pytest.approx(lb.share / 10)
