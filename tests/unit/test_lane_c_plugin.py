"""A second lane-C competition slots in through
:mod:`cascade.trainer.lane_c_competitions` and
:mod:`cascade.validator.lane_c_competitions` with no change to the core:
dispatched while live, its entry data and arbiter reports ride the signed
manifest under the generic ``competition`` / ``reports`` fields, and the
validator judges and folds it next to MarketCast. MarketCast's wire bytes
are untouched by the new fields."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config.lane_c import COMPETITION_KINDS, register_competition_kind
from cascade.shared.lane_c import LaneCState, target_full_pay_record
from cascade.shared.lane_c_manifest import (
    LaneCEntry,
    LaneCManifest,
    dump_lane_c_manifest,
    format_finetune_commit,
    load_lane_c_manifest,
    parse_finetune_commit,
)
from cascade.trainer.finetune import LegRecipe
from cascade.trainer.lane_c_competitions import (
    TRAINER_COMPETITIONS,
    TrainerCompetition,
    register_trainer_competition,
    trainer_competition,
)
from cascade.trainer.lane_c_driver import LaneCDriver, RunnerLaneCOps
from cascade.validator.lane_c import LaneCJudgement, LaneCMixin
from cascade.validator.lane_c_competitions import (
    VALIDATOR_COMPETITIONS,
    ValidatorCompetition,
    judging_order,
    register_validator_competition,
)

from .test_lane_c_trainer_driver import FROM, KING, REF, UID, Clock, E, FakeOps, _cfg, _ptr

TOY = "toyc"
T = f"{TOY}:a"
if TOY not in COMPETITION_KINDS:
    register_competition_kind(TOY, eval="toy forward-only eval", forward_only=True,
                              parse_value=lambda v: v)


def _report_key(rid):
    return f"toy_reports/{rid}.json"


class ToyTrainer(TrainerCompetition):
    kind = TOY

    def live(self, cfg, block):
        return int(block) >= FROM

    def read_recipe(self, ops, tree, *, target, submission_time, train_time):
        return LegRecipe(target=target, lr_scale=1.0, steps_fraction=1.0,
                         warmup_fraction=0.0, schedule="cosine")

    def leg_hours(self, cfg, recipe):
        return 2.0

    def train_leg(self, ops, leg, recipe):
        return {"trained_pointer": _ptr("toy-" + leg["hotkey"]), "train_block": leg["started_block"],
                "competition": {"model": "toy-1", "seed": 7}}

    def report_key(self, rid):
        return _report_key(rid)

    def report_timeout(self, cfg):
        return 2


class ToyValidator(ValidatorCompetition):
    """Every toy entry passes its judge phase; the pending one confirms at the
    next settlement — the generic pending → confirm fold does the rest."""

    kind = TOY
    seen: list = []

    def judge(self, runner, manifest, cm, *, rid, block, era_index, st, commitments_fn):
        self.seen.append(rid)
        lc = runner.cfg.lane_c
        js = []
        for p in st.pending:
            if p.target.startswith(TOY + ":") and p.kind == "confirm":
                e = LaneCEntry(**p.entry)
                js.append(LaneCJudgement(entry=e, passed=True, stage="verdict", reason="ok",
                                         gain_lcb=0.05, gain=0.06, phase="confirm",
                                         base_pointer=p.base_pointer,
                                         full_pay=target_full_pay_record(
                                             e.target, 0.05, lc, era_blocks=7200)))
        for e in (cm.entries if cm is not None else ()):
            if e.target.startswith(TOY + ":"):
                js.append(LaneCJudgement(entry=e, passed=True, stage="verdict", reason="ok",
                                         gain_lcb=0.05, gain=0.06, base_pointer=KING))
        return {"n": len(js)}, js


@pytest.fixture(autouse=True)
def _toy_registered():
    register_trainer_competition(ToyTrainer())
    register_validator_competition(ToyValidator())
    yield
    TRAINER_COMPETITIONS.pop(TOY, None)
    VALIDATOR_COMPETITIONS.pop(TOY, None)


class ToyOps(FakeOps):
    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.reports: dict = {}
        self.train = lambda leg, recipe: ToyTrainer().train_leg(self, leg, recipe)

    def read_report(self, key):
        return self.reports.get(key)


def _env(tmp_path):
    base = _cfg()
    cfg = SimpleNamespace(lane_c=replace(base.lane_c, targets=(E, T)), marketcast=base.marketcast)
    ops = ToyOps(tmp_path, Clock())
    return LaneCDriver(cfg, ops, tmp_path / "d.json"), ops


def _fund(ops, hk, block=100):
    ops.q.add_lane_c(hk, T, REF[hk], block)
    ops.commits.append(SimpleNamespace(uid=UID[hk], hotkey=hk, coldkey=None, commit_block=block,
                                       payload=format_finetune_commit(T, REF[hk])))


def test_the_commitment_parses_generically_and_mkt_still_needs_its_suites():
    p = format_finetune_commit(T, REF["ALFA"])
    assert parse_finetune_commit(p) == (T, REF["ALFA"])
    assert parse_finetune_commit(f"metro-v1:ft:mkt:returns:hippius:{REF['ALFA']}") is None
    assert parse_finetune_commit(f"metro-v1:ft:nokind:a:hippius:{REF['ALFA']}") is None
    assert parse_finetune_commit(f"metro-v1:ft:domain:a:hippius:{REF['ALFA']}") is None


def test_a_second_competition_dispatches_trains_and_records_its_reports(tmp_path):
    drv, ops = _env(tmp_path)
    _fund(ops, "ALFA")
    assert drv.tick(FROM - 5, 9, KING, FROM - 3600) == []          # not live yet
    assert [x["target"] for x in drv.tick(FROM + 10, 10, KING, FROM)] == [T]
    drv.settle(round_id="1", block=FROM + 900, era_index=10, base_pointer=KING)
    m1 = load_lane_c_manifest(ops.published[-1][1])
    (e,) = m1.entries
    assert (e.target, e.competition, e.marketcast) == (T, {"model": "toy-1", "seed": 7}, {})
    assert m1.reports == {} and drv.st.awaits == {TOY: [{"rid": "1", "age": 0}]}
    assert drv.st.mkt_await == []      # MarketCast's own path: no MarketCast entry, no wait
    # the main validator's toy report for round 1 lands; round 2 records it
    ops.reports[_report_key("1")] = '{"rows": []}'
    drv.settle(round_id="2", block=FROM + 1800, era_index=10, base_pointer=KING)
    m2 = load_lane_c_manifest(ops.published[-1][1])
    sha = hashlib.sha256(b'{"rows": []}').hexdigest()
    assert m2.reports == {TOY: {"1": sha}} and m2.mkt_reports == {}
    assert drv.st.awaits == {TOY: [{"rid": "2", "age": 0}]}
    # an unpublished report goes absent after report_timeout (2) settlements
    drv.settle(round_id="3", block=FROM + 2700, era_index=10, base_pointer=KING)
    drv.settle(round_id="4", block=FROM + 3600, era_index=10, base_pointer=KING)
    assert load_lane_c_manifest(ops.published[-1][1]).reports == {TOY: {"2": "absent"}}
    # persisted and reloaded
    assert LaneCDriver(drv.cfg, ops, drv.path).st.awaits == drv.st.awaits


def test_the_leg_wall_is_the_competitions_own(tmp_path):
    from cascade.shared.config.lane_c import LaneCCompetition

    drv, ops = _env(tmp_path)
    until = FROM + 7200
    drv.cfg.lane_c = replace(drv.cfg.lane_c,
                             competitions=(LaneCCompetition(TOY, from_block=FROM,
                                                            until_block=until),))
    _fund(ops, "ALFA")
    # 2h (600 blocks) recipe: the [lane_c] 1h default dispatches, the
    # competition's own wall is re-checked before any rent
    assert [x["hotkey"] for x in drv.tick(until - 400, 10, KING, FROM)] == ["ALFA"]
    assert ops.trained == [] and ops.q.get("ALFA").last_error_class == "competition_ended"


def test_runner_ops_route_recipe_and_training_to_the_competition(tmp_path):
    ops = RunnerLaneCOps.__new__(RunnerLaneCOps)
    ops.r = SimpleNamespace(cfg=None, work_root=tmp_path)
    leg = {"hotkey": "ALFA", "target": T, "started_block": 5}
    assert ops.train_leg(leg, None)["competition"] == {"model": "toy-1", "seed": 7}
    assert trainer_competition(E).kind == "mkt" and trainer_competition("nokind:x") is None


def test_marketcast_manifest_bytes_carry_no_new_fields():
    e = LaneCEntry(miner_hotkey="ALFA", miner_uid=1, target=E, recipe_ref=REF["ALFA"],
                   trained_pointer=_ptr("x"), train_block=1, marketcast={"train_time": 1.0})
    m = LaneCManifest(round_id="1", era_index=0, created_block=1, base_pointer=KING,
                      base_round_id="1", pool_block=0, entries=(e,), mkt_reports={"0": "absent"})
    text = dump_lane_c_manifest(m)
    assert '"competition"' not in text and '"reports"' not in text
    assert load_lane_c_manifest(text) == m
    with pytest.raises(ValueError, match="marketcast"):
        replace(e, competition={"x": 1})
    with pytest.raises(ValueError, match="reports kind"):
        replace(m, reports={"mkt": {"0": "absent"}})


# ── the validator half ───────────────────────────────────────────────────────


class _V(LaneCMixin):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mkt_calls = []

    def _mkt_settle(self, manifest, cm, **kw):
        self.mkt_calls.append(kw["rid"])
        return None, []


def _toy_entry(hk):
    return LaneCEntry(miner_hotkey=hk, miner_uid=UID[hk], target=T, recipe_ref=REF[hk],
                      trained_pointer=_ptr("toy-" + hk), train_block=FROM + 10,
                      competition={"model": "toy-1"})


def test_the_validator_judges_and_folds_every_competition(monkeypatch):
    from cascade.shared.config.lane_c import parse_lane_c

    lc = replace(parse_lane_c({}), from_block=FROM, targets=(E, T))
    v = _V(SimpleNamespace(lane_c=lc, marketcast=None, round=None))
    assert [c.kind for c in judging_order()] == ["mkt", TOY]
    st = LaneCState()

    def judge(rid, entries):
        cm = SimpleNamespace(entries=tuple(entries))
        js, plans, rows = [], {}, []
        for comp in judging_order():
            plan, cjs = comp.judge(v, None, cm, rid=rid, block=FROM + 900, era_index=1, st=st,
                                   commitments_fn=None)
            plans[comp.kind] = plan
            js += cjs
        from cascade.validator.lane_c import _LaneCRound

        return _LaneCRound(rid, FROM + 900, rows, js, 1, mkt=plans.pop("mkt"), plans=plans)

    wins = v._lane_c_fold(st, judge("1", [_toy_entry("ALFA")]))
    assert wins == [] and [p.target for p in st.pending] == [T]      # judge phase → pending
    wins = v._lane_c_fold(st, judge("2", []))
    assert [(w.target, w.hotkey) for w in wins] == [(T, "ALFA")]    # confirm → the slot
    assert st.slots[T].hotkey == "ALFA" and v.mkt_calls == ["1", "2"]


def test_state_keeps_a_competitions_own_section():
    st = LaneCState(comp={TOY: {"cursor": 3}})
    assert LaneCState.from_json(st.to_json()).comp == {TOY: {"cursor": 3}}
    assert '"comp"' not in LaneCState().to_json()
