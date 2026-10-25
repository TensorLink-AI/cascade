"""DEC-CA-0055 on the trainer: king-only promotion on a passing king check,
the init bench entry never a candidate, the king leg on the king's own key
with the operator fallback, the king's vault TTL, and the small carry-overs
(hash-link rollout gate, the non-burning leg-failure default, the intake's
not_before skip)."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.funding.queue import OPERATOR_FAULT_CLASS
from cascade.shared.bench_report import BenchEntry, BenchReport
from cascade.shared.era import era_length_blocks, min_effective_era, v2_first_warm_era
from cascade.shared.manifest import BenchScores, TrainedEntry
from cascade.trainer import rolling as R
from cascade.trainer.faults import king_key_failed
from cascade.trainer.promotion import TrainerPromotion
from cascade.trainer.runner_promotion import _PromotionMixin
from cascade.trainer.runner_rolling import _KingKeyFailed, _RollingMixin
from tests.unit.test_rolling import (
    REF,
    Clock,
    FakeClient,
    _armed,
    _commit,
    _join,
    _ptr,
    _sched,
)

KING_PTR = _ptr("KKING-12")


def _scores(x=0.5):
    return BenchScores(x, x, x, x, x, x)


def _v2(cfg):
    armed = _armed(cfg)
    L = era_length_blocks(armed.round, armed.round.rolling_from_block)
    v2_block = armed.round.rolling_from_block + 2 * L
    return replace(armed, v2=replace(armed.v2, from_block=v2_block, random_init_eras=3,
                                     king_ladder=True,
                                     reign_warm_start_blocks=3 * L)), v2_block, L


# ── the engine: king-only promotion ─────────────────────────────────────────


def test_promote_king_fires_one_member_once(tmp_path):
    eng = TrainerPromotion(reign_threshold=1.0, k_max=3, quality_epsilon=0.1,
                           state_path=tmp_path / "s.json", pointer_path=tmp_path / "p.json")
    eng.king_hotkey, eng.reign_start_block = "KING", 0
    rec = eng.promote_king(check_round="r9", king_pointer=KING_PTR, size="moe", score=0.4,
                           epoch_block=9000, round_id="r10", effective_era=17)
    assert rec is not None and rec.generation == 1 and rec.effective_era == 17
    assert [m.checkpoint_id for m in rec.members] == [KING_PTR]
    assert rec.members[0].source_round == "r9" and rec.fired_block == 9000
    assert eng.reign_start_block == 9000                        # the clock restarts
    assert eng.promote_king(check_round="r9", king_pointer=KING_PTR, size="moe", score=0.4,
                            epoch_block=9900, round_id="r11", effective_era=18) is None
    again = TrainerPromotion.load(reign_threshold=1.0, k_max=3, quality_epsilon=0.1,
                                  state_path=tmp_path / "s.json",
                                  pointer_path=tmp_path / "p.json")
    assert again.last_king_check == "r9" and again.generation == 1


def test_the_init_entry_is_never_a_candidate(tmp_path):
    eng = TrainerPromotion(reign_threshold=1.0, k_max=3, quality_epsilon=0.1)
    eng.king_hotkey = "KING"
    entries = (BenchEntry(role="king", size="m", miner_hotkey="KING", miner_uid=0,
                          trained_pointer=KING_PTR, scores=_scores()),)
    try:
        entries = (*entries, BenchEntry(role="init", size="m", miner_hotkey="", miner_uid=-1,
                                        trained_pointer="", scores=_scores(0.9)))
    except ValueError:      # a build whose roles do not know "init" yet
        entries = (*entries, SimpleNamespace(role="init", size="m", miner_hotkey="",
                                             miner_uid=-1, trained_pointer="",
                                             scores=_scores(0.9)))
    report = SimpleNamespace(round_id="r1", created_block=100, entries=entries)
    assert eng.record_bench(SimpleNamespace(round_id="r1", warm_start_ckpt=""), report) == 1
    assert [c.checkpoint_id for c in eng.candidates] == [KING_PTR]


# ── the runner: reading the king check off the receipt trail ────────────────


class _Store:
    def __init__(self, texts):
        self.texts = texts

    def get_text(self, key):
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


class _Runner(_PromotionMixin):
    def __init__(self, cfg, tmp_path, check):
        self.cfg = cfg
        self.promotion = TrainerPromotion(
            reign_threshold=1.0, k_max=3, quality_epsilon=0.1,
            state_path=tmp_path / "s.json", pointer_path=tmp_path / "p.json")
        self._check = check
        self._last_receipt_king = "KING"
        self._receipt_problem = ""
        self.published = []
        report = BenchReport(round_id="r-check", created_block=1, entries=(
            BenchEntry(role="king", size="cascade-moe-28m", miner_hotkey="KING", miner_uid=0,
                       trained_pointer=KING_PTR, scores=_scores(0.4)),))
        from cascade.shared.bench_report import bench_report_key, dump_bench_report

        self._store = _Store({bench_report_key("r-check"): dump_bench_report(report)})

    def manifest_store(self):
        return self._store

    def _receipt_king(self):
        return "KING"

    def _receipt_king_check(self):
        return self._check

    def _publish_promotion_record(self, record):
        self.published.append(record)

    wallet = None


def test_a_passing_check_fires_a_king_only_promotion_at_the_checks_boundary(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg)
    boundary = v2_block + 4 * L + 900
    check = {"result": "pass", "round_id": "r-check", "king_pointer": KING_PTR, "reasons": [],
             "block": boundary}
    r = _Runner(v2cfg, tmp_path, (check, boundary, "r-check"))
    # The trainer reads the receipt a boundary later, within the same era.
    later = boundary + 900
    r._promotion_boundary_step("KING", later, "r-later", effective_era=99)
    assert len(r.published) == 1
    rec = r.published[0]
    assert [m.checkpoint_id for m in rec.members] == [KING_PTR]
    assert rec.members[0].size == "cascade-moe-28m"
    assert rec.fired_block == later  # never backdate when the record actually fired
    assert rec.effective_era == max(min_effective_era(v2cfg.round, boundary),
                                    v2_first_warm_era(v2cfg))
    # the same check never fires twice
    r._promotion_boundary_step("KING", later + 900, "r-later2", effective_era=99)
    assert len(r.published) == 1 and r.promotion.generation == 1


@pytest.mark.parametrize("result", ["fail", "deferred"])
def test_a_failed_or_deferred_check_fires_nothing(cfg, tmp_path, result):
    v2cfg, v2_block, L = _v2(cfg)
    boundary = v2_block + 4 * L + 900
    check = {"result": result, "round_id": "r-check", "king_pointer": KING_PTR,
             "reasons": ["six-mean"]}
    r = _Runner(v2cfg, tmp_path, (check, boundary, "r-check"))
    # a ripe trainer clock + candidates never fire in the ladder lineage
    r.promotion.king_hotkey, r.promotion.reign_start_block = "KING", v2_block
    r._promotion_boundary_step("KING", boundary + 900, "r-x", effective_era=99)
    assert r.published == [] and r.promotion.generation == 0
    # a judged (failed) check restarts the reign clock at its boundary; deferred does not
    want = boundary if result == "fail" else v2_block
    assert r.promotion.reign_start_block == want


def test_pre_v2_promotion_path_is_unchanged(cfg, tmp_path, monkeypatch):
    armed = _armed(cfg)
    r = _Runner(armed, tmp_path, None)
    called = {}
    monkeypatch.setattr(r.promotion, "maybe_promote",
                        lambda **kw: called.setdefault("kw", kw))
    monkeypatch.setattr(r, "_v2_king_check_step",
                        lambda *a: pytest.fail("king check outside v2"))
    r._promotion_boundary_step("KING", armed.round.rolling_from_block + 900, "r1",
                               effective_era=12)
    assert "kw" in called


# ── the king's own key ───────────────────────────────────────────────────────


def test_king_key_failure_classification():
    assert king_key_failed(("no vaulted key", True, "auth", False))
    assert king_key_failed(("pod identity", True, "tamper", False))
    assert king_key_failed(("lium up said: Insufficient balance", False, "infra", True))
    assert king_key_failed(None, RuntimeError("HTTP 402 payment required"))
    assert not king_key_failed(("sold out", False, "no_capacity", False))
    assert not king_key_failed(("ssh dropped", False, "infra", True))
    assert not king_key_failed(("pod ab402cd stalled", False, "infra", True))


class _KingRunner(_RollingMixin):
    """Just enough runner for _rolling_train_king's two paths."""

    def __init__(self, cfg, tmp_path, *, own):
        self.cfg = cfg
        self.work_root = tmp_path
        self.trainer_spec = "x"
        self._own = own
        self._final_role_hosts = {}
        self._funded_king_lock = threading.Lock()
        self._rolling_king_host_era = None
        self._funded_king_host = None
        self._funded_wait_abort = threading.Event()
        self.persisted = []
        self.operator_dispatches = 0

    def _rolling_train_king_own_key(self, gen, era, block, **kw):
        if isinstance(self._own, Exception):
            raise self._own
        return self._own

    def _dispatcher(self):
        runner = self

        class D:
            def dispatch(self, host, **kw):
                runner.operator_dispatches += 1
                return TrainedEntry("KING", 0, "king", REF["KING"], KING_PTR, "d", kw["block"])
        return D()

    def _rent_king_host(self, rid):
        return SimpleNamespace(name="op")

    def _stage_king_vault(self, host, gen):
        return host

    def _refuse_diverged_king(self, entry, contract):
        pass

    def _persist_completed_leg(self, entry, **kw):
        self.persisted.append(kw.get("funded_by", ""))


def _era(v2_block):
    return R.EraState(index=12, start_block=v2_block, seed_block=v2_block - 3600,
                      base_seed=7, epoch_start=12)


def test_king_leg_runs_on_the_kings_key_else_the_operator_trains_it(cfg, tmp_path):
    v2cfg, v2_block, _ = _v2(cfg)
    v2cfg = replace(v2cfg, round=replace(v2cfg.round, funded_pods="rent",
                                         payer_vault_dir=str(tmp_path / "vault")))
    gen = SimpleNamespace(hotkey="KING", uid=0, ref=REF["KING"])
    own = TrainedEntry("KING", 0, "challenger", REF["KING"], _ptr("own"), "d", v2_block)
    r = _KingRunner(v2cfg, tmp_path, own=own)
    assert r._rolling_train_king(gen, _era(v2_block), v2_block, end_wall=0.0) is own
    assert r.persisted == ["king"] and r.operator_dispatches == 0
    r = _KingRunner(v2cfg, tmp_path, own=_KingKeyFailed("revoked"))
    entry = r._rolling_train_king(gen, _era(v2_block), v2_block, end_wall=0.0)
    assert entry.trained_pointer == KING_PTR
    assert r.persisted == ["operator"] and r.operator_dispatches == 1
    # a market/infra failure is not the key's: it re-raises (retried on the key)
    r = _KingRunner(v2cfg, tmp_path, own=RuntimeError("sold out"))
    with pytest.raises(RuntimeError):
        r._rolling_train_king(gen, _era(v2_block), v2_block, end_wall=0.0)
    assert r.operator_dispatches == 0


def test_pre_v2_king_leg_never_touches_the_kings_key(cfg, tmp_path):
    armed = replace(_armed(cfg), round=replace(_armed(cfg).round, funded_pods="rent",
                                               payer_vault_dir=str(tmp_path / "v")))
    r = _KingRunner(armed, tmp_path, own=AssertionError("own key used pre-v2"))
    gen = SimpleNamespace(hotkey="KING", uid=0, ref=REF["KING"])
    era = R.EraState(index=10, start_block=armed.round.rolling_from_block,
                     seed_block=0, base_seed=7)
    r._rolling_train_king(gen, era, era.start_block, end_wall=0.0)
    assert r.persisted == [""] and r.operator_dispatches == 1


def test_the_kings_vault_ttl_is_restamped(cfg, tmp_path):
    from cascade.trainer.runner_funded_pods import _FundedPodMixin

    class Runner(_RollingMixin, _FundedPodMixin):
        pass

    r = Runner()
    r.work_root = tmp_path
    r.cfg = replace(cfg, round=replace(cfg.round, funded_pods="rent",
                                       payer_vault_dir=str(tmp_path / "vault"),
                                       funded_entry_ttl_hours=36))
    vault = r._payer_vault()
    vault.insert("KING", "lium-key")
    path = tmp_path / "vault" / "KING.json"
    rec = json.loads(path.read_text())
    import time

    rec["stored_at"] = time.time() - 30 * 3600            # 30h old of a 36h TTL
    path.write_text(json.dumps(rec))
    assert r._rolling_refresh_king_key("KING")
    assert time.time() - json.loads(path.read_text())["stored_at"] < 60
    assert not r._rolling_refresh_king_key("NOBODY")


# ── carry-overs ──────────────────────────────────────────────────────────────


def test_hash_link_is_rollout_gated():
    sched = R.RollingScheduler.__new__(R.RollingScheduler)
    sched.cfg = SimpleNamespace(manifest=SimpleNamespace())
    assert not sched._hash_link_active(10_000)                # key absent ⇒ never
    sched.cfg = SimpleNamespace(manifest=SimpleNamespace(hash_link_from_block=0))
    assert not sched._hash_link_active(10_000)
    sched.cfg = SimpleNamespace(manifest=SimpleNamespace(hash_link_from_block=9_000))
    assert not sched._hash_link_active(8_100) and sched._hash_link_active(9_000)


def test_an_unrecorded_leg_failure_never_burns():
    ops = LegOpsNoRunner()
    msg, miner_fault, cls, burn = ops.leg_failure("ALFA")
    assert (miner_fault, cls, burn) == (False, OPERATOR_FAULT_CLASS, False)


class LegOpsNoRunner(R.LegOps):
    def __init__(self):
        self.r = SimpleNamespace(_funded_leg_failures={})


def test_intake_skips_an_entry_backing_off_before_admission(cfg, tmp_path, monkeypatch):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    start = armed.round.rolling_from_block
    ops.commits.append(_commit("ALFA", REF["ALFA"], start - 100))
    q = ops.queue()
    q.add("ALFA", REF["ALFA"], reveal_block=start - 100)
    assert q.mark_in_flight("ALFA", REF["ALFA"], target_boundary=start + 900, era_index=10,
                            started_block=start)
    q.requeue("ALFA", error="storage down", error_class=OPERATOR_FAULT_CLASS,
              burn_attempt=False)
    assert ops.queue().get("ALFA").not_before > clock()
    real_admit = R.admit
    seen = []
    monkeypatch.setattr(R, "admit", lambda *a, **k: seen.append(1) or real_admit(*a, **k))
    sched.tick(client, start + 5)
    _join(sched)
    assert seen == [] and ops.leg_calls == []
    clock.t += 3600                                           # past the backoff
    sched.tick(client, start + 6)
    _join(sched)
    assert seen and [c[0] for c in ops.leg_calls] == ["ALFA"]


def test_funded_by_is_set_only_on_a_build_that_has_the_field():
    @dataclass(frozen=True)
    class WithField:
        role: str
        funded_by: str = ""

    assert R._with_funded_by(WithField("king"), "operator").funded_by == "operator"
    e = TrainedEntry("KING", 0, "king", REF["KING"], KING_PTR, "d", 1)
    assert R._with_funded_by(e, "king") is e or R._with_funded_by(e, "king").funded_by == "king"


def test_lane_b_init_for_reads_only_without_apply_pending(tmp_path):
    from cascade.trainer.lane_b_driver import LaneBDriver, LaneBDriverState

    drv = LaneBDriver.__new__(LaneBDriver)
    drv._lock = threading.RLock()
    drv.st = LaneBDriverState()
    drv.st.pending_merges = [{"leg_pointer": "L", "base": "B", "selection": {}}]
    assert drv.init_for(15, "M", Path(tmp_path), apply_pending=False) == ("M", [])
    assert drv.st.pending_merges                               # untouched
    drv.st.merged_inits["15"] = {"pointer": "MM", "base": "M", "merges": [{"x": 1}]}
    assert drv.init_for(15, "M", Path(tmp_path), apply_pending=False) == ("MM", [{"x": 1}])

