"""DEC-CA-0056 validator lane C, MarketCast-only (owner 2026-10-20): a
signed entry on a non-forward-only target is rejected at the ``target``
stage (the generic pool_c verdict is retired), read failures HOLD the
settlement, the fold + pre-round snapshot, and the audit (three-lane vector
rebuild + tamper fail; records refused before activation). The MarketCast
verdict itself: tests/unit/test_marketcast_validator.py."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config import parse_lane_b
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.hippius import StorageError, manifest_round_key
from cascade.shared.lane_c import TargetWin
from cascade.shared.lane_c_manifest import (
    LaneCEntry,
    LaneCManifest,
    dump_lane_c_manifest,
    format_finetune_commit,
    lane_c_manifest_key,
    lane_c_text_digest,
)
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    format_trained_pointer,
)
from cascade.validator.lane_c import LaneCTransient, best_per_target
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

KING = "5King"
EB = 7200
BLOCK = 20 * EB
REF = {hk: f"{hk}/recipe@sha256:" + hashlib.sha256(hk.encode()).hexdigest()
       for hk in ("alice", "bob", "carol", "dave")}
UIDS = {KING: 3, "alice": 7, "bob": 8, "carol": 9, "dave": 10}


def _ptr(tag: str) -> str:
    return format_trained_pointer(f"cascade/ckpt-{tag}@sha256:"
                                  + hashlib.sha256(tag.encode()).hexdigest())


KING_PTR = _ptr("king")


class _Store:
    def __init__(self):
        self.objs: dict[str, str] = {}
        self.fail: set[str] = set()

    def get_text(self, key):
        if key in self.fail:
            raise StorageError(f"503 on {key}")
        return self.objs[key]


M, V = "mkt:returns", "mkt:vol"


def _entry(hk, target=M, ptr=None):
    return LaneCEntry(miner_hotkey=hk, miner_uid=UIDS[hk], target=target, recipe_ref=REF[hk],
                      trained_pointer=ptr or _ptr(hk), train_block=BLOCK - 100)


def _history(entries, *, block=BLOCK - 500):
    return [SimpleNamespace(hotkey=e.miner_hotkey, commit_block=block,
                            payload=format_finetune_commit(e.target, e.recipe_ref,
                                           mkt_suites=("returns", "vol")))
            for e in entries]


def _settle(store, rid, entries, *, base=KING_PTR, prev="", block=BLOCK, bind=True):
    cm = LaneCManifest(round_id=rid, era_index=1, created_block=block + 10, base_pointer=base,
                       base_round_id=rid, pool_block=block, entries=tuple(entries))
    text = dump_lane_c_manifest(cm)
    store.objs[lane_c_manifest_key(rid)] = text
    m = TrainingManifest(
        round_id=rid, created_block=block + 10, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry(KING, 3, "king", "k/g@sha256:" + "3" * 64, KING_PTR, "4" * 64,
                              block - 1)],
        prev_round_id=prev, lane_c_digest=lane_c_text_digest(text) if bind else "")
    store.objs[manifest_round_key(rid)] = dump_manifest(m)
    return m


def _runner(cfg, tmp_path, store, *, history=None, mkt_from=0):
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "share": 0.5})
    lc = replace(parse_lane_c({"targets": [M, V]}), from_block=EB)
    v = replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    mc = replace(cfg.marketcast, from_block=int(mkt_from))
    r = ValidatorRunner(cfg=replace(cfg, lane_b=lb, lane_c=lc, validator=v, marketcast=mc),
                        state=genesis(KING, 3), verify_signatures=False)
    r._lane_c_store = store
    r.commitment_history_fn = (lambda: history) if history is not None else None
    return r


def _judge(r, m):
    r._lane_c_block = r._epoch_start_block(m)
    rows = r._lane_c_commit(r._lane_c_prepare(m))
    return {row.get("hotkey"): row for row in rows}


def test_non_forward_only_target_is_refused_everywhere(cfg, tmp_path):
    """A retired DEC-CA-0056 generic target cannot be an entry at all, so a
    signed lane-C manifest carrying one is malformed (a verdict on the
    lane-C part; lane A proceeds) — never judged on a pool."""
    with pytest.raises(ValueError, match="forward-only eval"):
        replace(_entry("alice"), target="domain:energy")
    store = _Store()
    m = _settle(store, "1001", [_entry("alice")])
    key = lane_c_manifest_key("1001")
    store.objs[key] = store.objs[key].replace('"mkt:returns"', '"domain:energy"')
    m = replace(m, lane_c_digest=lane_c_text_digest(store.objs[key]))
    r = _runner(cfg, tmp_path, store, history=[])
    rows = _judge(r, m)
    assert "lane_c_manifest_malformed" in rows[None]["error"]
    assert "forward-only eval" in rows[None]["error"]
    assert r._lane_c().slots == {} and r._lane_c().applied == {"1001": BLOCK}


def test_read_failures_are_transient_never_verdicts(cfg, tmp_path):
    store = _Store()
    m = _settle(store, "1001", [_entry("alice")])
    r = _runner(cfg, tmp_path / "a", store, history=[])
    store.fail.add(lane_c_manifest_key("1001"))
    with pytest.raises(LaneCTransient):
        r._lane_c_prepare(m)
    assert r._lane_c().applied == {}


def test_tampered_binding_is_a_verdict_and_unbound_round_is_empty(cfg, tmp_path):
    store = _Store()
    es = [_entry("alice")]
    m = _settle(store, "1001", es)
    store.objs[lane_c_manifest_key("1001")] += " "
    rows = _judge(_runner(cfg, tmp_path / "a", store, history=_history(es)), m)
    assert rows[None]["error"] == "lane_c_digest_mismatch"
    m2 = _settle(store, "1002", es, bind=False)
    r = _runner(cfg, tmp_path / "b", store, history=_history(es))
    assert _judge(r, m2) == {} and r._lane_c().applied == {"1002": BLOCK}


def test_inactive_at_the_block_reads_nothing(cfg, tmp_path):
    store = _Store()
    m = _settle(store, "1001", [_entry("alice")], block=EB)     # the settlement AT from_block
    r = _runner(cfg, tmp_path, store, history=[])
    assert r._lane_c_prepare(m) is None


def test_handle_manifest_holds_on_a_lane_c_read_failure(cfg, tmp_path):
    store = _Store()
    es = [_entry("alice")]
    m = _settle(store, "1001", es)
    r = _runner(cfg, tmp_path, store, history=_history(es))
    r.check_manifest = lambda _m: None
    r.check_pool_pin = lambda *a, **k: None
    r._lane_b_prepare = lambda _m: None
    store.fail.add(lane_c_manifest_key("1001"))
    before = r.state
    assert r._handle_manifest(object(), object(), m, "d" * 64) is False
    assert r.state == before and "1001|lane_c" in r._read_holds
    assert r._lane_c().applied == {}


def _win(hk="alice", target=M, block=BLOCK):
    return TargetWin(target=target, hotkey=hk, uid=UIDS[hk], round_id="1001", win_block=block,
                     gain_lcb=0.03, tier=0.02, full_eras=5)


@pytest.mark.parametrize("recovery", ["fresh", "torn", "lagging"])
def test_recovery_keeps_expired_holder_and_old_report_state(cfg, tmp_path, recovery):
    from cascade.shared.lane_c import LaneCState
    from cascade.validator.lane_c import _LaneCRound
    from cascade.validator.marketcast import MktPlan

    store = _Store()
    live = _runner(cfg, tmp_path / "live", store, history=[])
    recovered = _runner(cfg, tmp_path / "recovered", store, history=[])
    for r in (live, recovered):
        r.cfg = replace(r.cfg, lane_c=replace(r.cfg.lane_c, replay_settlements=2,
                                             win_ttl_eras=1))
    old = replace(_win(block=2 * EB), base_pointer=KING_PTR, gain_lcb=0.10)
    new = replace(old, hotkey="bob", uid=UIDS["bob"], round_id="1007", win_block=8 * EB,
                  gain_lcb=0.03)
    seen = {"a" * 64: {"hotkey": "alice", "commit_block": EB}}
    backlog = [{"round_id": "1001", "candidates": [], "age": 0}]
    cutoff = 5 * EB
    applied_configs = []

    def wire(r):
        base = r.cfg
        newer = replace(base, lane_c=replace(base.lane_c, gain_margin=0.05))
        r.netconfig = SimpleNamespace(enabled=True,
            config_at=lambda cfg, block: newer if block >= cutoff else base)

        def judge(m, state):
            block = r._epoch_start_block(m)
            applied_configs.append((m.round_id, r.cfg.lane_c.gain_margin))
            plan = MktPlan(offered=[old] if m.round_id == "1001" else (
                [new] if m.round_id == "1007" else []),
                backlog=backlog if m.round_id == "1001" else list(state.mkt_backlog),
                seen=seen if m.round_id == "1001" else dict(state.mkt_seen))
            return _LaneCRound(m.round_id, block, [], [], mkt=plan)

        r._lane_c_judge_round = judge
        return base, newer

    wire(live)
    base, newer = wire(recovered)
    prev = ""
    for i in range(1, 8):
        m = _settle(store, str(1000 + i), [], prev=prev, block=(i + 1) * EB)
        with live._settlement_cfg(live._epoch_start_block(m)):
            live._lane_c_commit(live._lane_c_prepare(m))
        if i == 1 and recovery == "lagging":
            recovered._lane_c_state = LaneCState.from_json(live._lane_c().to_json())
        prev = m.round_id
    if recovery == "torn":
        path = recovered._lane_c_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"slots":')
    applied_configs.clear()
    with recovered._settlement_cfg(8 * EB):
        recovered._lane_c_commit(recovered._lane_c_prepare(m))
        assert recovered.cfg is newer
    actual = recovered._lane_c()
    assert actual.to_json() == live._lane_c().to_json()
    assert actual.slots[M] == old       # expired, still defends against a lower LCB
    assert actual.mkt_seen == seen and actual.mkt_backlog == backlog
    assert ("1002", base.lane_c.gain_margin) in applied_configs
    assert ("1006", 0.05) in applied_configs


def test_recovery_holds_on_cyclic_history(cfg, tmp_path):
    store = _Store()
    _settle(store, "1001", [], prev="1002")
    _settle(store, "1002", [], prev="1001")
    r = _runner(cfg, tmp_path, store, history=[])
    with pytest.raises(LaneCTransient, match="cyclic"):
        r._lane_c_walk_back("1002")
    assert r._lane_c().applied == {}


def test_pre_round_snapshot_carries_lane_c(cfg, tmp_path):
    store = _Store()
    m = _settle(store, "1001", [_entry("alice")])
    r = _runner(cfg, tmp_path, store, history=[])
    snap = r._snapshot("1001")
    assert "lane_c" in snap
    _judge(r, m)
    r._lane_c().record(_win())
    assert r._lane_c().applied and r._lane_c().slots
    r.state = replace(r.state, pre_round=snap)
    assert r._restore_pre_round("1001")
    st = r._lane_c()
    assert st.slots == {} and st.applied == {} and st.pending == []
    # an inert lane C keeps the snapshot's shape
    off = ValidatorRunner(cfg=cfg, state=genesis(KING, 3))
    assert "lane_c" not in off._snapshot("x")


def test_best_per_target_tie_breaks_by_train_block_then_hotkey():
    from cascade.validator.lane_c import LaneCJudgement

    a = LaneCJudgement(replace(_entry("alice"), train_block=50), True, "verdict", "win", 0.02, 0.05)
    b = LaneCJudgement(replace(_entry("bob"), train_block=40), True, "verdict", "win", 0.02, 0.05)
    c = LaneCJudgement(replace(_entry("carol"), train_block=40), True, "verdict", "win", 0.02, 0.05)
    assert best_per_target([a, c, b])[M].entry.miner_hotkey == "bob"


# ── audit ────────────────────────────────────────────────────────────────────


class _Client:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return UIDS.get(hk)

    def set_weight_vector(self, vec):
        pass

    def set_equal_share_weights(self, *a, **k):
        pass


def _receipt(cfg, tmp_path, *, mkt_from=EB):
    """A settlement's receipt with alice holding the mkt:returns slot (a
    MarketCast win folded earlier) — paid while MarketCast is active."""
    store = _Store()
    b2 = BLOCK + EB
    r = _runner(cfg, tmp_path, store, history=[], mkt_from=mkt_from)
    r._lane_c().record(_win(block=BLOCK))
    m2 = _settle(store, "1002", [], block=b2, bind=False)
    r._lane_b_block = b2
    rows = r._lane_c_commit(r._lane_c_prepare(m2))
    r._lane_c_block = b2
    vec = r._apply_weights(_Client(b2), "1002", [3], boundary=b2)
    kw = {**r._lane_b_receipt_kw(None), **r._lane_c_receipt_kw(rows)}
    rc = SimpleNamespace(status="scored", weights=vec, reward_uids=(3,), verdict=None,
                         epoch_start_block=b2, emission=None, king_check=None,
                         round_id="1002", **kw)
    return r, rc


def test_audit_rebuilds_the_three_lane_vector_and_flags_tampering(cfg, tmp_path):
    from cascade.audit.checks import check_weights

    r, rc = _receipt(cfg, tmp_path)
    assert rc.weights[7] == pytest.approx(0.125)                 # alice holds a lane-C slot
    assert check_weights(rc, r.cfg).status == "PASS"

    def _with(**kw):
        return SimpleNamespace(**{**rc.__dict__, **kw})

    bad = list(rc.weights)
    bad[7] += 0.05
    bad[0] -= 0.05
    assert check_weights(_with(weights=tuple(bad)), r.cfg).status == "FAIL"
    pay = dict(rc.lane_c_pay)
    pay["wins"] = [{**pay["wins"][0], "multiplier": 2.0}]
    assert check_weights(_with(lane_c_pay=pay), r.cfg).status == "FAIL"
    pay = {**rc.lane_c_pay, "share": 0.5}
    assert check_weights(_with(lane_c_pay=pay), r.cfg).status == "FAIL"
    assert check_weights(_with(lane_c_pay=None), r.cfg).status == "FAIL"


def test_audit_rebuilds_a_burned_lane_c_share_with_marketcast_off(cfg, tmp_path):
    from cascade.audit.checks import check_weights

    r, rc = _receipt(cfg, tmp_path, mkt_from=0)
    assert rc.lane_c_pay["slots"] == 0 and rc.lane_c_pay["wins"] == []
    assert rc.weights[7] == 0.0 and sum(rc.weights) == pytest.approx(1.0)
    assert check_weights(rc, r.cfg).status == "PASS"


def test_audit_refuses_lane_c_records_before_activation(cfg, tmp_path):
    from cascade.audit.checks import check_lane_c

    r, rc = _receipt(cfg, tmp_path)
    early = SimpleNamespace(**{**rc.__dict__, "epoch_start_block": EB})
    assert check_lane_c(early, r.cfg).status == "FAIL"
    clean = SimpleNamespace(**{**early.__dict__, "lane_c": None, "lane_c_pay": None})
    assert check_lane_c(clean, r.cfg).status == "SKIP"
