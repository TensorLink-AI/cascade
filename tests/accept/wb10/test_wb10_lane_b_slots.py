"""WB10 (plan rev6.22, owner 2026-11-24): the lane B slot rule.

* **Pay:** a held slot pays its FULL slot share every settlement (no decay,
  no TTL) until a newer merge takes it; empty slots burn. The decay / TTL
  keys (`full_pay_tiers`, `half_life_rounds`, `floor`, `win_ttl_eras`) are
  retired (`RETIRED_KEYS["lane_b"]`), so an older file still loads. There is
  no activation key (lane B is not rolled out).
* **Testing:** every judged leg is tested on EVERY slot
  (`judge_expert_slots`). Declared targets (1..`max_domains_per_leg` = 4)
  form one max-T family at `gain_margin`; every other slot forms a second
  max-T family at `undeclared_gain_margin` (0.02, overlay-settable). A slot
  clears on CRPS OR MASE (family LCB >= bar, positive point gain) with the
  other metric non-inferior and every guard holding; each cleared slot is
  claimable, so one miner may hold every slot.

Config checks use the real loader on the shipped files; payouts and judging
use the producer's own code on synthetic numpy scores. Nothing from PATHS is
imported at module level."""

from __future__ import annotations

import numpy as np
import pytest

from .helpers import SHIPPED, load_ok, overrides_refusal, raw_with, refused


def _spec(fn):
    """A missing behaviour fails as an AssertionError (never an import,
    attribute, key or value error), so BASE fails for the right reason."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001 - every other failure is the spec failing
            raise AssertionError(f"{type(e).__name__}: {e}") from e

    return run

RETIRED = ("full_pay_tiers", "half_life_rounds", "floor", "win_ttl_eras")
ERA = 100


# ── config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SHIPPED)
@_spec
def test_wb10_shipped_files_carry_no_retired_payout_key(name):
    import tomllib
    from pathlib import Path

    raw = tomllib.loads((Path(__file__).resolve().parents[3] / name).read_text())
    assert not set(RETIRED) & set(raw["lane_b"]), "retired [lane_b] payout keys still shipped"
    lb = load_ok(raw_with(name)).lane_b
    assert lb.undeclared_gain_margin == 0.02 and lb.max_domains_per_leg == 4
    for k in (*RETIRED, "indefinite_pay_from_block"):
        assert not hasattr(lb, k), f"LaneBConfig still has {k!r}"


@pytest.mark.parametrize("name", SHIPPED)
@_spec
def test_wb10_retired_payout_keys_load_and_are_ignored(name):
    from cascade.shared.config import RETIRED_KEYS

    assert set(RETIRED) <= set(RETIRED_KEYS["lane_b"])
    old = {"full_pay_tiers": [[0.05, 10], [0.02, 5], [0.01, 2]], "half_life_rounds": 1.0,
           "floor": 0.1, "win_ttl_eras": 28}
    assert load_ok(raw_with(name, old)).lane_b == load_ok(raw_with(name)).lane_b


@pytest.mark.parametrize("value", [0.005, 1.0, float("nan"), -0.1])
@_spec
def test_wb10_undeclared_gain_margin_refuses_bad_values(value):
    assert refused(raw_with("chain.toml", {"undeclared_gain_margin": value}))


@_spec
def test_wb10_undeclared_gain_margin_parses_and_is_overlay_settable():
    lb = load_ok(raw_with("chain.toml", {"undeclared_gain_margin": 0.03})).lane_b
    assert lb.undeclared_gain_margin == 0.03
    assert lb.verdict_params().undeclared_gain_margin == 0.03
    assert overrides_refusal({"lane_b": {"undeclared_gain_margin": 0.03}}) is None


# ── payouts ──────────────────────────────────────────────────────────────────


def _lb(**kw):
    from cascade.shared.config import parse_lane_b

    return parse_lane_b({"from_block": 1, "domains": ["energy", "web", "sales", "nature"],
                         "share": 0.5, **kw})


def _win(domain, hotkey, uid, merge_block, **kw):
    from cascade.shared.lane_b import DomainWin

    return DomainWin(domain=domain, hotkey=hotkey, uid=uid, round_id=str(merge_block),
                     merge_block=merge_block, target_lcb=0.03, **kw)


def _state(*wins):
    from cascade.shared.lane_b import LaneBState

    st = LaneBState()
    for w in wins:
        st.record(w)
    return st


@_spec
def test_wb10_held_slot_pays_full_forever():
    from cascade.shared.lane_b import lane_b_payouts

    st = _state(_win("energy", "alice", 9, 10 * ERA))
    for block in (10 * ERA, 38 * ERA, 10_000 * ERA):
        pay = lane_b_payouts(st, block=block, era_blocks=ERA, cfg=_lb(), n_uids=16)
        assert [(p["domain"], p["uid"], p["multiplier"], p["since_era"]) for p in pay] == [
            ("energy", 9, 1.0, 10)], f"at block {block} the holder must be paid in full"
        assert pay[0]["age"] == (block - 10 * ERA) // ERA
        assert set(pay[0]) == {"domain", "hotkey", "uid", "merge_block", "age", "since_era",
                               "multiplier", "gain_lcb"}


@_spec
def test_wb10_one_miner_holds_every_slot_and_empty_slots_burn():
    from cascade.shared.lane_b import combined_weights

    lb = _lb()
    st = _state(*(_win(d, "alice", 9, 10 * ERA) for d in ("energy", "web", "sales")))
    vec = combined_weights([3], 16, king_decay=1.0, burn_uid=0, state=st,
                           block=900 * ERA, era_blocks=ERA, cfg=lb)
    slot = 0.5 / 4
    assert vec[9] == pytest.approx(3 * slot)
    assert vec[0] == pytest.approx(slot), "the one empty slot must burn"
    st.record(_win("nature", "alice", 9, 20 * ERA))
    vec = combined_weights([3], 16, king_decay=1.0, burn_uid=0, state=st,
                           block=900 * ERA, era_blocks=ERA, cfg=lb)
    assert vec[9] == pytest.approx(0.5) and vec[0] == pytest.approx(0.0)


@_spec
def test_wb10_holder_keeps_slot_until_a_newer_accepted_merge():
    from cascade.shared.lane_b import LaneBState, lane_b_payouts

    st = LaneBState()
    m = {"rebased_pointer": "a@1", "king_pointer": "k@1", "selection": {"0": [1]},
         "round_id": "r1"}
    st.accept(m, [_win("energy", "alice", 9, 0)], 10 * ERA)
    newer = _win("energy", "bob", 4, 0, qualified_block=150 * ERA)
    st.qualify(newer)                     # qualified, not accepted: the slot holds
    pay = lane_b_payouts(st, block=160 * ERA, era_blocks=ERA, cfg=_lb(), n_uids=16)
    assert [p["hotkey"] for p in pay] == ["alice"]
    st.accept({**m, "rebased_pointer": "old@1"}, [_win("energy", "carol", 5, 0)], 5 * ERA)
    assert st.wins["energy"].hotkey == "alice", "an older accepted merge never displaces"
    st.accept({**m, "rebased_pointer": "b@1", "round_id": "r2"}, [newer], 170 * ERA)
    pay = lane_b_payouts(st, block=171 * ERA, era_blocks=ERA, cfg=_lb(), n_uids=16)
    assert [(p["hotkey"], p["since_era"]) for p in pay] == [("bob", 170)]


# ── every-slot judging, two max-T families ──────────────────────────────────


DOMS = ("energy", "web", "sales", "nature")


def _scores(effects, *, n_per=150, n_cl=10, seed=0, mase_effects=None):
    """King and candidate scores per window; ``effects[d]`` scales the
    candidate's quantile loss on domain ``d`` (CRPS), ``mase_effects`` its
    MASE (default: the same)."""
    rng = np.random.default_rng(seed)
    dom, cl = [], []
    for d in DOMS:
        for i in range(n_per):
            dom.append(d)
            cl.append(f"{d}-{i % n_cl}")
    n = len(dom)
    kq = rng.uniform(0.5, 1.5, size=(n, 3))
    ka = rng.uniform(5.0, 10.0, size=n)
    km = rng.uniform(0.8, 1.2, size=n)
    # per-window noise so the bootstrap is not degenerate
    cq = kq * np.exp(rng.normal(0.0, 0.01, size=(n, 1)))
    cm = km * np.exp(rng.normal(0.0, 0.01, size=n))
    me = effects if mase_effects is None else mase_effects
    for d in DOMS:
        m = np.array([x == d for x in dom])
        cq[m] *= effects.get(d, 1.0)
        cm[m] *= me.get(d, 1.0)
    return dict(domains=dom, clusters=cl, king_qloss=kq, king_abs_target=ka, king_mase=km,
                chal_qloss=cq, chal_abs_target=ka, chal_mase=cm)


def _judge(declared, effects, **kw):
    from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_slots

    p = ExpertLaneParams(B=1000, gain_margin=0.01, undeclared_gain_margin=0.02,
                         min_target_windows=100, min_target_clusters=5)
    return judge_expert_slots(slots=DOMS, declared=declared, params=p, seed="wb10",
                              **_scores(effects, **kw))


@_spec
def test_wb10_every_slot_is_tested_and_each_cleared_slot_is_claimable():
    tv = _judge(["energy"], {"energy": 0.90, "web": 0.95, "sales": 0.985})
    assert tv.targets == DOMS and tv.declared == ("energy",)
    v = tv.verdicts
    assert v["energy"].passed and v["energy"].declared and v["energy"].bar == 0.01
    assert v["web"].passed and not v["web"].declared and v["web"].bar == 0.02
    # sales improves 1.5%: above the declared bar, below the undeclared one
    assert not v["sales"].passed and v["sales"].metric_lcbs["crps"] < 0.02
    assert not v["nature"].passed
    assert tv.won == ["energy", "web"]
    assert set(tv.maxt_crits) == {"declared", "undeclared"}


@_spec
def test_wb10_declared_target_clears_at_the_lower_bar():
    tv = _judge(["sales"], {"sales": 0.985})
    assert tv.verdicts["sales"].passed and tv.verdicts["sales"].bar == 0.01


@_spec
def test_wb10_declared_family_is_independent_of_the_undeclared_slots():
    """Testing every other slot never makes a declared target harder: the
    declared family's critical value and LCBs are the same whatever the
    undeclared slots do."""
    a = _judge(["energy"], {"energy": 0.97})
    b = _judge(["energy"], {"energy": 0.97, "web": 0.80, "sales": 1.0, "nature": 0.9})
    assert a.maxt_crits["declared"] == b.maxt_crits["declared"]
    assert a.verdicts["energy"].metric_lcbs == b.verdicts["energy"].metric_lcbs


@_spec
def test_wb10_crps_or_mase_with_the_other_non_inferior():
    # CRPS improves 5%, MASE flat: clears on CRPS
    tv = _judge(["energy"], {"energy": 0.95}, mase_effects={"energy": 1.0})
    assert tv.verdicts["energy"].passed and tv.verdicts["energy"].gain_metric == "crps"
    # CRPS improves 5% but MASE is 5% worse (inferior): refused
    tv = _judge(["energy"], {"energy": 0.95}, mase_effects={"energy": 1.05})
    assert not tv.verdicts["energy"].passed


@_spec
def test_wb10_a_guard_failure_blocks_every_slot():
    tv = _judge(["energy"], {"energy": 0.90, "web": 0.90, "nature": 1.10})
    assert tv.guard_reason and tv.won == []


@_spec
def test_wb10_inconclusive_slot_is_never_claimed():
    from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_slots

    p = ExpertLaneParams(B=500, min_target_windows=100, min_target_clusters=5)
    tv = judge_expert_slots(slots=DOMS + ("suite:gift",), declared=["energy"], params=p,
                            seed="x", **_scores({"energy": 0.9}))
    assert tv.verdicts["suite:gift"].inconclusive == "items"
    assert not tv.verdicts["suite:gift"].passed
