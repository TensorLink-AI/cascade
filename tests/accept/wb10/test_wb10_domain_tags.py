"""WB10 (owner 2026-11-25): lane B domain tags — RECORDS AND DISPLAY ONLY.

Each judged lane B generator is tagged with the domain evals it ACTUALLY
improved, read from the validator's signed verdict rows (never re-evaluated):

* **improved** = the slot's gain test of ``judge_expert_slots`` cleared: a
  conclusive slot where, for CRPS or MASE, the family LCB is ``>= gain_bar``
  with a positive point gain and the other metric non-inferior. A leg guard
  failure does not un-improve a slot (it is ``improved``, ``claimed: false``).
* per slot: status, declared, bar, LCB, ``margin = target_lcb - gain_bar``.
* the tags live in unsigned documents only (``receipts/index.json``
  ``lane_b_domains``, ``status/v2.json`` ``lane_b.domain_tags``); the signed
  receipt body, its digest and the weight vector are unchanged with and
  without them; every reader tolerates their absence.

Verdicts come from the producer's own judge on synthetic numpy scores and its
own row writer. Nothing from PATHS is imported at module level."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np


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


DOMS = ("energy", "web", "sales", "nature")
TAU = 0.01
HK = "5Alice" + "a" * 42
LEG = "metro-v1:trained:hippius:cascade/leg@sha256:" + "b" * 64


def _scores(effects, *, n_per=150, n_cl=10, seed=0, mase_effects=None):
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
    cq = kq * np.exp(rng.normal(0.0, 0.01, size=(n, 1)))
    cm = km * np.exp(rng.normal(0.0, 0.01, size=n))
    me = effects if mase_effects is None else mase_effects
    for d in DOMS:
        m = np.array([x == d for x in dom])
        cq[m] *= effects.get(d, 1.0)
        cm[m] *= me.get(d, 1.0)
    return dict(domains=dom, clusters=cl, king_qloss=kq, king_abs_target=ka, king_mase=km,
                chal_qloss=cq, chal_abs_target=ka, chal_mase=cm)


def _rows(declared, effects, **kw):
    """The receipt's signed lane B rows for one judged leg, via the real
    judge and the real row writer; also returns the TargetsVerdict."""
    from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_slots
    from cascade.validator.expert_lane import ExpertJudgement, judgements_record

    p = ExpertLaneParams(B=1000, gain_margin=0.01, undeclared_gain_margin=0.02,
                         tau_domain=TAU, min_target_windows=100, min_target_clusters=5)
    tv = judge_expert_slots(slots=DOMS, declared=declared, params=p, seed="wb10-tags",
                            **_scores(effects, **kw))
    entry = SimpleNamespace(miner_hotkey=HK, miner_uid=7, domain=declared[0],
                            declared=tuple(declared), rebased_pointer=LEG)
    j = ExpertJudgement(entry=entry, passed=bool(tv.won), stage="verdict", reason="",
                        verdict=tv.verdicts[declared[0]], targets=tv)
    return judgements_record([j]), tv


def _cleared(v) -> bool:
    """The judge's own gain rule, recomputed from the verdict internals."""
    ms = ("crps", "mase")
    return not v.inconclusive and any(
        v.metric_lcbs[m] >= v.bar and v.metric_gains[m] > 0.0
        and all(v.metric_ni_lcbs[o] >= -TAU for o in ms if o != m) for m in ms)


@_spec
def test_wb10_tags_improved_not_improved_and_untargeted_domains():
    from cascade.shared.lane_b_domains import lane_b_domain_tags

    # declared: energy (improves 10%) and sales (flat); web is NOT targeted
    # and improves 5% (clears the stricter undeclared bar); nature flat
    rows, tv = _rows(["energy", "sales"], {"energy": 0.90, "web": 0.95})
    before = json.dumps(rows, sort_keys=True)
    tags = lane_b_domain_tags(rows)
    assert json.dumps(rows, sort_keys=True) == before, "tagging mutated the signed rows"
    assert len(tags) == 1
    g = tags[0]
    assert g["hotkey"] == HK and g["leg"] == LEG and g["targets"] == ["energy", "sales"]
    assert g["improved"] == ["energy", "web"]
    assert g["not_improved"] == ["sales", "nature"]
    assert g["inconclusive"] == [] and g["not_evaluated"] == []
    d = g["domains"]
    assert d["energy"]["declared"] is True and d["energy"]["claimed"] is True
    assert d["web"]["declared"] is False and d["web"]["gain_bar"] == 0.02
    assert d["web"]["status"] == "improved", "an untargeted domain the leg improved is tagged"
    assert d["sales"]["declared"] is True and d["sales"]["status"] == "not_improved"
    for s in DOMS:
        v = tv.verdicts[s]
        # same rule as the judge, and margin = the verdict's LCB over its bar
        assert (d[s]["status"] == "improved") == _cleared(v), s
        assert abs(d[s]["margin"] - round(v.target_lcb - v.bar, 6)) < 1e-9, s
    assert d["energy"]["margin"] > 0 and d["sales"]["margin"] < 0


@_spec
def test_wb10_guard_blocked_slot_is_improved_but_unclaimed():
    from cascade.shared.lane_b_domains import lane_b_domain_tags

    rows, tv = _rows(["energy"], {"energy": 0.90, "nature": 1.10})
    assert tv.guard_reason and tv.won == []
    d = lane_b_domain_tags(rows)[0]["domains"]
    assert d["energy"]["status"] == "improved" and d["energy"]["claimed"] is False
    assert d["energy"]["metric"] in ("crps", "mase")
    assert d["nature"]["status"] == "not_improved"


@_spec
def test_wb10_inferior_other_metric_is_not_improved():
    from cascade.shared.lane_b_domains import lane_b_domain_tags

    # CRPS 5% better but MASE 5% worse: the judge refuses the gain test
    rows, tv = _rows(["energy"], {"energy": 0.95}, mase_effects={"energy": 1.05})
    assert tv.verdicts["energy"].metric_lcbs["crps"] >= 0.01
    assert lane_b_domain_tags(rows)[0]["domains"]["energy"]["status"] == "not_improved"


@_spec
def test_wb10_inconclusive_and_unjudged_rows():
    from cascade.shared.lane_b_domains import lane_b_domain_tags

    rows = [
        {"hotkey": "b", "domain": "energy", "stage": "verdict", "passed": False,
         "reason": "inconclusive: 3 target clusters < 5", "target_lcb": None,
         "targets": ["energy", "nature"], "leg": "L", "declared": True, "gain_bar": 0.01,
         "metric_lcbs": {}, "metric_gains": {}, "inconclusive": "clusters"},
        {"hotkey": "c", "domain": "nature", "stage": "stale", "passed": False,
         "reason": "base too old", "target_lcb": None, "guard_lcbs": {}, "public_lcbs": {}},
        {"error": "bad_signature"},
    ]
    tags = {g["hotkey"]: g for g in lane_b_domain_tags(rows)}
    assert tags["b"]["inconclusive"] == ["energy"] and tags["b"]["improved"] == []
    assert tags["c"]["not_evaluated"] == ["nature"]
    assert lane_b_domain_tags(None) == [] and lane_b_domain_tags([]) == []


def _receipt(rows, weights=(0.5, 0.25, 0.25)):
    from cascade.shared.receipt import RoundReceipt

    return RoundReceipt(round_id="r-1", status="rejected", epoch_start_block=7200,
                        epoch_block_hash="0x" + "1" * 64, base_seed=1, generation_seed=2,
                        training_seed=3, manifest={}, reward_uids=(0,), weights=weights,
                        reject_reason="fixture",
                        lane_b=None if rows is None else tuple(rows),
                        lane_b_pay={"wins": [{"domain": "energy", "uid": 1,
                                              "multiplier": 1.0}]})


@_spec
def test_wb10_tags_stay_out_of_the_signed_receipt_digest_and_weights(monkeypatch):
    import cascade.shared.lane_b_domains as tags_mod
    from cascade.shared.v2_status import v2_index_fields

    rows, _ = _rows(["energy"], {"energy": 0.90})
    r = _receipt(rows)
    body = r.canonical_body()
    digest = hashlib.sha256(body).hexdigest()
    cols = v2_index_fields(r)
    assert cols["lane_b_domains"][0]["improved"] == ["energy"]
    assert b"lane_b_domains" not in body and b"domain_tags" not in body
    assert hashlib.sha256(r.canonical_body()).hexdigest() == digest
    assert tuple(r.weights) == (0.5, 0.25, 0.25)
    # without the field every other column (verdicts, pay, emission split)
    # is identical
    monkeypatch.setattr(tags_mod, "lane_b_domain_tags", lambda _rows: [])
    plain = v2_index_fields(r)
    assert "lane_b_domains" not in plain
    assert {k: v for k, v in cols.items() if k != "lane_b_domains"} == plain
    assert hashlib.sha256(r.canonical_body()).hexdigest() == digest
    # a receipt with no lane B rows carries no tags column
    assert "lane_b_domains" not in v2_index_fields(_receipt(None))


@_spec
def test_wb10_status_doc_and_readers_tolerate_absence():
    from cascade.miner.dashboard import lane_lines
    from cascade.miner.live_status import lane_b_domain_info
    from cascade.shared.config import load_chain_config
    from cascade.shared.lane_b_domains import lane_b_domain_tags
    from cascade.shared.v2_status import build_v2_status

    from .helpers import REPO

    cfg = load_chain_config(REPO / "chain.toml")
    rows, _ = _rows(["energy"], {"energy": 0.90, "web": 0.95})
    tags = lane_b_domain_tags(rows)
    plain = build_v2_status(cfg, block=100, as_of="t")
    tagged = build_v2_status(cfg, block=100, as_of="t", lane_b_domains=tags)
    assert "domain_tags" not in plain["lane_b"]
    assert tagged["lane_b"]["domain_tags"] == tags
    tagged["lane_b"].pop("domain_tags")
    assert tagged == plain, "the tags change nothing else in status/v2.json"
    # dashboard: an older index row (no column) renders exactly as before
    entry = {"lane_b_verdicts": [{"hotkey": HK, "domain": "energy", "passed": True}],
             "lane_b_pay": {"wins": []}}
    old = lane_lines(entry)
    assert not any("lane B domains" in s for s in old)
    new = lane_lines({**entry, "lane_b_domains": tags})
    shown = [s for s in new if "lane B domains" in s]
    assert len(shown) == 1 and "improved energy" in shown[0] and "web*" in shown[0]
    assert [s for s in new if "lane B domains" not in s] == old
    # cascade status: lines from lanes.b.domain_tags, none when absent
    snap = {"lanes": {"b": {"domain_tags": tags}}}
    assert lane_b_domain_info(snap, HK) and lane_b_domain_info(snap, "other") == []
    assert lane_b_domain_info({"lanes": {"b": {}}}) == []
