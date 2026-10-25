"""Lane-aware observability: W&B lane tags / run ids per lane (mocked sink),
the trainer's king-check / promotion events, the optional validator
settlement summary, the receipt-index lane columns (emission split with lane
C, the DEC-CA-0055 ladder and king check), the ``status/v2.json`` lane-A and
lane-C holder fields, and the miner dashboard's lane rows.

No real wandb is imported: a fake module is injected into ``sys.modules``.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.miner.dashboard import lane_lines
from cascade.shared.config import WandbConfig
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.v2_status import (
    build_v2_status,
    king_check_fields,
    ladder_fields,
    lane_a_view,
    lane_c_state_from_receipt,
    v2_index_fields,
)
from cascade.trainer.runner_promotion import king_check_event, promotion_event
from cascade.trainer.wandb_sink import (
    lane_of,
    lane_run_fields,
    log_validator_settlement,
    log_wandb_event,
    open_wandb_run,
    settlement_summary,
    wandb_run_id,
)
from cascade.trainer.worker import expert_target_label

RECIPE = {"target": "mkt:returns", "lr_scale": 0.5, "steps_fraction": 0.5,
          "warmup_fraction": 0.1, "schedule": "cosine"}
EMISSION = {"boundary": 14400, "king_since_block": 7200,
            "policy": {"prelaunch_burn": 0.0, "lane_a_cap": None, "king_share": 0.6,
                       "former_share": 0.05, "king_slash": None},
            "ladder": {"prev_bp": 5000, "bp": 6000, "event": "pass",
                       "court_hotkey": "5Former" + "x" * 40, "court_until_block": 50000,
                       "slashed": False, "funded_by": ""}}
CHECK = {"result": "pass", "round_id": "r-77", "block": 14400, "king_hotkey": "hk-king",
         "king_pointer": "metro-v1:trained:hippius:cascade/k@sha256:" + "a" * 64,
         "reasons": [], "lcb": 0.0123, "margin": 0.01}


class _FakeRun:
    def __init__(self):
        self.logged: list[dict] = []
        self.finished = False

    def log(self, record):
        self.logged.append(record)

    def finish(self):
        self.finished = True

    def define_metric(self, *a, **k):
        pass


@pytest.fixture
def fake_wandb(monkeypatch):
    calls: list[dict] = []
    runs: list[_FakeRun] = []

    def init(**kwargs):
        calls.append(kwargs)
        runs.append(_FakeRun())
        return runs[-1]

    mod = types.ModuleType("wandb")
    mod.init = init
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setenv("WANDB_API_KEY", "k")
    return SimpleNamespace(calls=calls, runs=runs)


def _open(lane="A", target="", recipe=None, hotkey="hk1"):
    return open_wandb_run(WandbConfig(enabled=True), round_id="900", role="challenger-moe-28m",
                          hotkey=hotkey, uid=4, size="moe-28m", lane=lane, target=target,
                          recipe=recipe)


# ── W&B lane tags and run ids ────────────────────────────────────────────────


def test_every_run_carries_its_lane_tag_target_and_recipe(fake_wandb):
    assert _open() is not None
    assert _open("B", "energy,horizon:64") is not None
    assert _open("C", "mkt:returns", recipe=RECIPE) is not None
    a, b, c = fake_wandb.calls
    assert "lane:A" in a["tags"] and not any(t.startswith("target:") for t in a["tags"])
    assert a["config"]["lane"] == "A" and a["config"]["target"] == ""
    assert a["name"] == "round-900-challenger-moe-28m"            # lane A unchanged
    assert "lane:B" in b["tags"] and "target:energy,horizon:64" in b["tags"]
    assert b["config"]["target"] == "energy,horizon:64"
    assert "lane:C" in c["tags"] and "target:mkt:returns" in c["tags"]
    for k in ("lr_scale", "steps_fraction", "warmup_fraction", "schedule"):
        assert c["config"][f"recipe_{k}"] == RECIPE[k]
    assert "recipe_lr_scale" not in b["config"]                    # recipe is lane C only
    # the role (job type) stays the receipt-role label for every lane
    assert a["job_type"] == b["job_type"] == c["job_type"] == "challenger-moe-28m"


def test_same_hotkey_same_round_never_shares_a_run_across_lanes(fake_wandb):
    _open()
    _open("B", "energy")
    _open("B", "retail")
    _open("C", "mkt:returns", recipe=RECIPE)
    ids = [c["id"] for c in fake_wandb.calls]
    assert len(set(ids)) == 4
    # deterministic: a retried leg resumes its own run
    assert wandb_run_id("900", "challenger-moe-28m", "hk1", lane="B", target="energy") == ids[1]
    # lane A's id is the pre-lane id (an in-flight run still resumes)
    assert wandb_run_id("900", "challenger-moe-28m", "hk1") == ids[0]
    assert wandb_run_id("900", "challenger-moe-28m", "hk1", lane="A") == ids[0]


def test_lane_helpers():
    assert lane_of() == "A" and lane_of(expert=True) == "B"
    assert lane_of(finetune=True, expert=True) == "C"
    assert lane_run_fields("x")["lane"] == "A"
    rec = SimpleNamespace(**RECIPE)
    assert lane_run_fields("c", "t", rec)["recipe_schedule"] == "cosine"
    assert expert_target_label("energy") == "energy"
    assert expert_target_label("energy.h64") == "energy,horizon:64"
    assert expert_target_label("h720") == "horizon:720"
    assert expert_target_label(None) == ""


def test_runner_labels_an_expert_leg_lane_b(cfg, tmp_path, monkeypatch):
    """``_train_for_entry`` derives lane B from an expert selection and hands
    the caller's target to the W&B open (via ``_train_checkpoint``)."""
    from cascade.trainer import loop as loop_mod
    from cascade.trainer.contract import TrainResult

    seen: list[dict] = []
    runner = loop_mod.TrainerRunner.__new__(loop_mod.TrainerRunner)
    runner.work_root = tmp_path
    monkeypatch.setattr(runner, "_reusable_checkpoint", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_write_train_complete_marker", lambda *a, **k: None,
                        raising=False)

    def _capture(gen, seeds, contract, budget, out_dir, *, log_role, warm_start_dir=None,
                 expert_selection=None, lane_info=None):
        seen.append(dict(lane_info or {}))
        return TrainResult(local_dir=out_dir, param_count=1, train_seconds=1.0,
                           metrics={}), "d", 1, 1

    monkeypatch.setattr(runner, "_train_checkpoint", _capture, raising=False)
    gen = loop_mod.ResolvedGenerator(hotkey="hk", uid=1, ref="a/b@sha256:" + "1" * 64)
    from cascade.trainer.contract import RoundSeeds

    seeds = RoundSeeds.derive(5, cfg.training.primary_size)
    contract = cfg.training.primary_size
    runner._train_for_entry(gen, "challenger", seeds, contract=contract, token_budget=1,
                            repo_suffix="-xenergy-u1", heat=False, warm_start_ref=None,
                            expert_selection={1: [2]}, lane_info={"target": "energy"})
    assert seen == [{"lane": "B", "target": "energy"}]


# ── trainer events + validator summary ───────────────────────────────────────


def test_king_check_and_promotion_events(fake_wandb):
    rec = king_check_event(CHECK, 14400, EMISSION)
    assert rec["result"] == "pass" and rec["lcb"] == 0.0123 and rec["margin"] == 0.01
    assert rec["king_pointer"] == CHECK["king_pointer"] and rec["ladder_king_share"] == 0.6
    assert rec["check_round"] == "r-77" and rec["block"] == 14400
    assert log_wandb_event(WandbConfig(enabled=True), kind="king_check", round_id="r-77",
                           record=rec)
    call = fake_wandb.calls[-1]
    assert call["job_type"] == "king_check" and "event:king_check" in call["tags"]
    assert fake_wandb.runs[-1].logged[0]["result"] == "pass" and fake_wandb.runs[-1].finished
    member = SimpleNamespace(checkpoint_id="ptr-1")
    record = SimpleNamespace(generation=3, fired_round="r-77", fired_block=14400,
                             effective_era=5, king_hotkey="hk-king", members=(member,),
                             signature="sig")
    pe = promotion_event(record)
    assert pe == {"generation": 3, "fired_round": "r-77", "fired_block": 14400,
                  "effective_era": 5, "king_hotkey": "hk-king", "n_members": 1,
                  "members": "ptr-1", "signed": True}
    # same event re-logged ⇒ the same run id (resume, never a duplicate)
    log_wandb_event(WandbConfig(enabled=True), kind="promotion", round_id="r-77", record=pe,
                    key="gen3")
    log_wandb_event(WandbConfig(enabled=True), kind="promotion", round_id="r-77", record=pe,
                    key="gen3")
    assert fake_wandb.calls[-1]["id"] == fake_wandb.calls[-2]["id"]


def test_events_are_gated_and_never_raise(fake_wandb, monkeypatch):
    assert not log_wandb_event(WandbConfig(enabled=False), kind="k", round_id="1", record={})
    assert not log_wandb_event(None, kind="k", round_id="1", record={})

    def boom(**kw):
        raise RuntimeError("wandb down")

    monkeypatch.setattr(sys.modules["wandb"], "init", boom)
    assert not log_wandb_event(WandbConfig(enabled=True), kind="k", round_id="1", record={})


def test_trainer_logs_each_king_check_once(fake_wandb):
    from cascade.trainer.runner_promotion import _PromotionMixin

    class _R(_PromotionMixin):
        cfg = SimpleNamespace(wandb=WandbConfig(enabled=True))

    r = _R()
    r.__dict__["_king_check_emission"] = EMISSION
    r._wandb_king_check_event(CHECK, 14400)
    r._wandb_king_check_event(CHECK, 14400)
    assert len(fake_wandb.calls) == 1
    assert fake_wandb.calls[0]["config"]["ladder_king_share"] == 0.6


ROW = {"round_id": "r-80", "status": "scored", "epoch_start_block": 14400, "lcb": -0.01,
       "margin": 0.02, "challenger_wins_round": False, "dethroned": False,
       "inconclusive": False, "post_round_king_hotkey": "hk-king", "post_round_king_uid": 3,
       "lane_b_verdicts": [{"hotkey": "b1", "domain": "energy", "passed": True},
                           {"hotkey": "b2", "domain": "retail", "passed": False}],
       "lane_b_pay": {"wins": [{"domain": "energy", "uid": 9, "multiplier": 1.0}]},
       "lane_c_verdicts": [{"hotkey": "c1", "target": "horizon:64", "passed": True,
                            "outcome": "pending"}],
       "lane_c_pay": {"share": 0.25, "slots": 2,
                      "wins": [{"target": "horizon:64", "uid": 7, "multiplier": 0.5}]},
       "emission_split": {"lane_a": 0.5, "lane_b": 0.125, "lane_c": 0.0625, "burn": 0.3125},
       "ladder": ladder_fields(EMISSION), "king_check": king_check_fields(CHECK)}


def test_validator_settlement_summary_is_off_by_default(fake_wandb):
    assert WandbConfig().validator_summary is False
    # the trainer mirror's switch does not turn the validator summary on
    assert not log_validator_settlement(WandbConfig(enabled=True), ROW)
    assert fake_wandb.calls == []
    assert log_validator_settlement(WandbConfig(validator_summary=True), ROW,
                                    validator_hotkey="5Val")
    call = fake_wandb.calls[0]
    assert call["job_type"] == "settlement" and "round:r-80" in call["tags"]
    s = call["config"]
    assert s["lane_b_judged"] == 2 and s["lane_b_passed"] == 1 and s["lane_b_paid_slots"] == 1
    assert s["lane_c_judged"] == 1 and s["lane_c_paid_slots"] == 1
    assert s["emission_lane_c"] == 0.0625 and s["emission_burn"] == 0.3125
    assert s["ladder_king_share"] == 0.6 and s["king_check_result"] == "pass"
    assert s["lane_b_payouts"] == {"energy": {"uid": 9, "multiplier": 1.0, "tier": None,
                                             "full_eras": None}}
    assert s["validator_hotkey"] == "5Val"


def test_settlement_summary_of_a_pre_v2_row():
    s = settlement_summary({"round_id": "1", "status": "scored", "lcb": 0.1})
    assert s["lane_b_judged"] == 0 and "emission_burn" not in s and "ladder_king_share" not in s


def test_loader_reads_validator_summary(tmp_path):
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    src = (Path(__file__).resolve().parents[2] / "chain.toml").read_text()
    assert load_chain_config(None).wandb.validator_summary is False
    patched = src.replace("[wandb]\n", "[wandb]\nvalidator_summary = true\n", 1)
    p = tmp_path / "chain.toml"
    p.write_text(patched)
    assert load_chain_config(p).wandb.validator_summary is True


# ── receipt index columns ────────────────────────────────────────────────────


def _receipt(**kw):
    base = dict(weights=(), reward_uids=(), lane_b=None, lane_b_pay=None, lane_c=None,
                lane_c_pay=None, emission=None, king_check=None)
    return SimpleNamespace(**{**base, **kw})


def test_index_split_counts_lane_c_pay_without_lane_b_pay():
    w = (0.5, 0.0, 0.25, 0.25)
    c_pay = {"share": 0.5, "slots": 2, "wins": [{"target": "horizon:64", "uid": 2,
                                                  "multiplier": 1.0}]}
    cols = v2_index_fields(_receipt(weights=w, reward_uids=(0,), lane_c_pay=c_pay))
    assert cols["emission_split"] == {"lane_a": 0.5, "lane_b": 0.0, "lane_c": 0.25,
                                      "burn": 0.25}
    assert "lane_b_pay" not in cols


def test_index_ladder_and_king_check_columns():
    assert v2_index_fields(_receipt()) == {}                    # a pre-v2 row is unchanged
    cols = v2_index_fields(_receipt(emission=EMISSION, king_check=CHECK))
    lad = cols["ladder"]
    assert lad["king_share"] == 0.6 and lad["king_share_bp"] == 6000 and lad["event"] == "pass"
    assert lad["court_hotkey"].startswith("5Former") and lad["court_until_block"] == 50000
    assert lad["slashed"] is False and lad["former_share"] == 0.05
    kc = cols["king_check"]
    assert kc["result"] == "pass" and kc["lcb"] == 0.0123 and kc["margin"] == 0.01
    # a ladder that has not moved since the crowning reads the policy's share
    no_bp = {**EMISSION, "ladder": {**EMISSION["ladder"], "bp": None}}
    assert ladder_fields(no_bp)["king_share"] == 0.6


def test_index_multi_target_and_lane_c_phase_columns():
    rows = ({"hotkey": "m", "domain": "energy", "stage": "verdict", "passed": True,
             "reason": "merge", "target_lcb": 0.02, "targets": ["energy", "horizon:64"]},
            {"hotkey": "s", "domain": "retail", "stage": "verdict", "passed": False,
             "reason": "x", "target_lcb": 0.0})
    c_rows = ({"hotkey": "c", "uid": 4, "target": "horizon:64", "stage": "verdict",
               "passed": True, "reason": "pass", "gain_lcb": 0.02, "gain": 0.03,
               "phase": "judge", "outcome": "pending"},)
    cols = v2_index_fields(_receipt(lane_b=rows, lane_c=c_rows))
    assert cols["lane_b_verdicts"][0]["targets"] == ["energy", "horizon:64"]
    assert "targets" not in cols["lane_b_verdicts"][1]
    v = cols["lane_c_verdicts"][0]
    assert v["uid"] == 4 and v["phase"] == "judge" and v["outcome"] == "pending"


def test_summarize_receipt_carries_the_lane_a_columns():
    from cascade.shared.receipt import summarize_receipt

    from .receipt_fixture import make_scored_receipt

    rc, _k, _c = make_scored_receipt()
    row = summarize_receipt(replace(rc, emission=EMISSION, king_check=CHECK))
    assert row["ladder"]["king_share"] == 0.6 and row["king_check"]["result"] == "pass"
    assert "ladder" not in summarize_receipt(rc)


# ── status/v2.json ───────────────────────────────────────────────────────────

RECEIPT_BODY = {
    "round_id": "r-90", "epoch_start_block": 21600, "emission": EMISSION, "king_check": CHECK,
    "lane_c": [{"hotkey": "alice", "uid": 7, "target": "mkt:vol", "outcome": "confirmed",
                "passed": True, "gain_lcb": 0.031, "origin_round": "r-89", "phase": "confirm",
                "trained_pointer": "ptr-alice"},
               {"hotkey": "bob", "uid": 8, "target": "mkt:returns", "outcome": "pending",
                "passed": True, "gain_lcb": 0.02, "phase": "judge"}],
    "lane_c_pay": {"block": 21600, "era_blocks": 7200, "share": 0.25, "slots": 9,
                   "wins": [{"target": "mkt:vol", "hotkey": "alice", "uid": 7,
                             "win_block": 14400, "age": 1, "multiplier": 1.0}]},
}


def test_lane_c_holders_from_the_latest_receipt(cfg):
    st = lane_c_state_from_receipt(RECEIPT_BODY)
    assert st["source"] == "receipt"
    h = st["slots"]["mkt:vol"]
    assert h["hotkey"] == "alice" and h["round_id"] == "r-89" and h["gain_lcb"] == 0.031
    assert st["pending"][0]["hotkey"] == "bob" and st["pending"][0]["kind"] == "confirm"
    assert lane_c_state_from_receipt({"round_id": "x"}) is None
    lc = replace(parse_lane_c({}), from_block=7200)
    doc = build_v2_status(replace(cfg, lane_c=lc), block=22000, as_of="t", lane_c_state=st,
                          lane_a=lane_a_view(RECEIPT_BODY))
    per = doc["lane_c"]["per_target"]
    assert per["mkt:vol"]["holder"]["hotkey"] == "alice"
    assert per["mkt:vol"]["holder"]["multiplier"] == 1.0
    assert per["mkt:returns"]["holder"] is None
    assert per["mkt:returns"]["pending"][0]["hotkey"] == "bob"
    assert doc["lane_c"]["holders_source"] == "receipt"
    assert doc["lane_a"]["ladder"]["king_share"] == 0.6
    assert doc["lane_a"]["king_check"]["result"] == "pass"


def test_trainer_publisher_reads_holders_and_ladder_from_receipts(cfg, monkeypatch):
    from cascade.shared import hippius, v2_status

    lc = replace(parse_lane_c({}), from_block=7200)
    anchor = "5Anchor"
    texts = {hippius.receipt_latest_key(anchor): json.dumps(RECEIPT_BODY)}
    store = SimpleNamespace(get_text=lambda key: texts[key])
    c = replace(cfg, lane_c=lc, manifest=replace(cfg.manifest, validator_hotkey=anchor))
    sched = SimpleNamespace(cfg=c, lane_b=None, lane_c=None, state=SimpleNamespace(current=None),
                            r=SimpleNamespace(manifest_store=lambda: store))
    published = []
    monkeypatch.setattr(v2_status, "publish_v2_status", lambda s, doc: published.append(doc))
    assert v2_status.publish_trainer_v2_status(sched, 22000, min_interval_s=0)
    doc = published[0]
    assert doc["lane_c"]["per_target"]["mkt:vol"]["holder"]["hotkey"] == "alice"
    assert doc["lane_a"]["ladder"]["court_until_block"] == 50000


def test_lane_b_view_lists_horizon_slots(cfg):
    doc = build_v2_status(cfg, block=100, as_of="t")
    lb = doc["lane_b"]
    assert lb["targets"][: len(lb["domains"])] == lb["domains"]
    assert all(t.startswith("horizon:") for t in lb["horizon_targets"])
    assert lb["payout"]["slots"] == len(lb["targets"])
    assert set(lb["per_domain"]) == set(lb["targets"])


# ── miner dashboard ──────────────────────────────────────────────────────────


def test_dashboard_lane_rows():
    lines = lane_lines(ROW)
    text = "\n".join(lines)
    assert "lane A" in text and "king ladder share 60% (pass)" in text
    assert "court 5Forme" in text and "until block 50,000" in text and "no key slash" in text
    assert "king check pass (LCB 0.0123 vs margin 0.0100)" in text
    assert "lane B          2 judged · 1 passed · paid: energy→uid 9 (100%)" in text
    assert "lane C          1 judged · 1 passed · paid: horizon:64→uid 7 (50%)" in text
    assert "emission        lane A 50% · lane B 12% · lane C 6% · burn 31%" in text
    assert lane_lines({"round_id": "1", "status": "scored"}) == []
    assert lane_lines(None) == []


def test_dashboard_frame_shows_the_lanes():
    from cascade.miner.dashboard import LiveFeed, compose_frame, round_status
    from cascade.shared.config import RoundConfig

    rc = RoundConfig()
    st = round_status(14400 + 10, rc)
    feed = LiveFeed(SimpleNamespace(poll_commitments=lambda **k: []))
    feed.index_doc = {"rounds": [{**ROW, "epoch_start_block": st.epoch_start}]}
    frame = compose_frame(st, "test", rc, feed)
    assert "king ladder share 60%" in frame and "lane C" in frame
