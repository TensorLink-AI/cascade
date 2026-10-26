"""DEC-CA-0058 satellite: a clean receipt chain sets the derived vector; every
tamper class burns; a silent main validator is held then burned; a clean
receipt recovers; an overlay-stamped receipt replays under its overlay.

The chain is mocked (a fake client); storage is an in-memory dict store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from cascade.eval.koth import evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.shared.chain import decayed_share_vector, seed_from_block_hash
from cascade.shared.hippius import (
    ObjectNotFound,
    manifest_round_key,
    receipt_latest_key,
    receipt_round_key,
)
from cascade.shared.manifest import TrainedEntry, TrainingManifest, dump_manifest, sign_manifest
from cascade.shared.receipt import (
    EntryScores,
    EvalContext,
    VerdictRecord,
    WindowScoreRecord,
    build_receipt,
    dump_receipt,
    sign_receipt,
)
from cascade.shared.settlement_state import state_digest
from cascade.trainer.contract import RoundSeeds
from cascade.validator import satellite as sat_mod
from cascade.validator import state as state_mod

from .receipt_fixture import CKPT_CHAL, CKPT_KING, GEN_REF_CHAL, GEN_REF_KING, make_scores

bt = pytest.importorskip("bittensor.sp_core")

TRAINER_KP = bt.Keypair.create_from_uri("//Trainer")
MAIN_KP = bt.Keypair.create_from_uri("//Main")
OTHER_KP = bt.Keypair.create_from_uri("//Mallory")
HOTKEYS = ["king_hk", "chal_hk", "x2_hk", "x3_hk"]
GRID = 3600
BOUNDARIES = [6 * GRID, 7 * GRID, 8 * GRID, 9 * GRID]
POST = {"king": "chal_hk", "king_since_block": None, "tenure_rounds": 0,
        "former_kings": ["king_hk"],
        "ladder": {"bp": None, "court_hotkey": None, "court_until_block": 0, "slashed": False}}


# ── fixtures ─────────────────────────────────────────────────────────────────


def _keyed(cfg):
    """The test chain.toml (conftest's neutralised pins) re-parsed from ONE raw
    table with the trainer / main-validator keys set, so a config an overlay
    rebuilds from raw matches the parsed one."""
    from cascade.shared.config import parse_chain_config

    raw = json.loads(json.dumps(cfg.raw))
    sets = {
        "manifest": {"trainer_hotkey": TRAINER_KP.ss58_address, "validator_hotkey": ""},
        "netconfig": {"owner_hotkey": MAIN_KP.ss58_address},
        "training": {"expected_gpu": "", "train_image_digest": ""},
        "generator": {"corpus_target_points": 0},
        "v2": {"launch_at_utc": "", "commit_open_at_utc": "", "commit_from_block": 0},
        "round": {"commit_floor_block": 0, "dedup_mode": "off", "epoch_blocks_prev": 0,
                  "epoch_activation_block": 0},
        "scoring": {"win_margin_start_prev": 0.0, "margin_activation_block": 0},
    }
    for table, kv in sets.items():
        raw.setdefault(table, {}).update(kv)
    return parse_chain_config(raw)


@pytest.fixture
def scfg(cfg):
    return _keyed(cfg)


@pytest.mark.parametrize("head", [30000, 100000])
def test_dated_launch_replays_historical_signed_receipts_with_chain_time(scfg, tmp_path, head):
    from cascade.shared.calendar_launch import launch_timestamp
    from cascade.shared.config import parse_chain_config
    from tests.unit.test_calendar_launch import _chain

    store = DictStore()
    receipt, _ = make_settlement(scfg, store, 0)
    raw = json.loads(json.dumps(scfg.raw))
    raw["v2"]["launch_at_utc"] = "2026-10-30T12:00:00Z"
    raw["v2"]["commit_open_at_utc"] = "2026-10-30T09:00:00Z"
    dated = parse_chain_config(raw)
    runner = _runner(dated, store, tmp_path)
    chain = _chain(launch_timestamp(dated.v2.launch_at_utc), head)
    runner.sync(client=chain)
    assert runner.state.verified_round == receipt.round_id
    assert not runner.state.failing


class DictStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def get_text(self, key: str) -> str:
        if key not in self.objects:
            raise ObjectNotFound(f"missing: {key}")
        return self.objects[key]

    def put_text(self, key: str, text: str, **_kw) -> None:
        self.objects[key] = text


class FakeClient:
    def __init__(self, block: int, hotkeys=None) -> None:
        self.block = block
        self.hotkeys = list(hotkeys or HOTKEYS)
        self.pushed: list[list[float]] = []

    def current_block(self) -> int:
        return self.block

    def registered_hotkeys(self) -> list[str]:
        return list(self.hotkeys)

    def n_uids(self) -> int:
        return len(self.hotkeys)

    def set_weight_vector(self, vec) -> bool:
        self.pushed.append(list(vec))
        return True

    def hotkey_ss58(self) -> str:
        return "sat_hk"


@pytest.fixture(autouse=True)
def _no_anon_s3(monkeypatch):
    """The audit's funded-roster read goes through anonymous S3; keep it offline."""
    import cascade.audit.main as audit_main

    def _offline(cfg, key):
        raise RuntimeError("offline")
    monkeypatch.setattr(audit_main, "_fetch_text", _offline)


def make_settlement(judge_cfg, store: DictStore, i: int, *, prev: dict | None = None,
                    chal_scale: float = 0.6, sign_manifest_with=TRAINER_KP,
                    netconfig: tuple[int, str] = (0, ""), tamper=None, publish=True,
                    sign_with=MAIN_KP):
    """Build, sign and publish settlement ``i`` of the chain. ``prev`` is the
    previous settlement's ``{"round_id", "manifest_sha256", "state_post"}``.
    ``tamper(receipt) -> receipt`` edits the receipt before the main signs."""
    boundary = BOUNDARIES[i]
    block_hash = "0x" + f"{i + 1:02x}" * 32
    base_seed = seed_from_block_hash(block_hash)
    size = judge_cfg.training.arch_preset
    from cascade.shared.manifest import contract_digest

    manifest = TrainingManifest(
        round_id=str(base_seed), created_block=boundary + 50,
        contract_digest=contract_digest(judge_cfg.training),
        base_arch_digest=judge_cfg.training.base_arch_digest,
        eval_dataset=judge_cfg.eval.eval_dataset,
        entries=[
            TrainedEntry("king_hk", 0, "king", GEN_REF_KING, CKPT_KING, "3" * 64,
                         boundary + 50, size=size),
            TrainedEntry("chal_hk", 1, "challenger", GEN_REF_CHAL, CKPT_CHAL, "4" * 64,
                         boundary + 50, size=size),
        ],
        prev_round_id=(prev or {}).get("round_id", ""),
        prev_manifest_sha256=(prev or {}).get("manifest_sha256", ""),
    )
    manifest = sign_manifest(manifest, sign_manifest_with)
    mtext = dump_manifest(manifest)
    params = judge_cfg.koth_params(block=boundary)
    king_scores = make_scores(1.0, i)
    chal_scores = [WindowScore(s.series_id, s.mase * chal_scale, s.qloss_per_q * chal_scale,
                               s.abs_target) for s in king_scores]
    result = evaluate_round(king_scores, chal_scores, params, seed=base_seed,
                            king_tenure_rounds=0)
    transition = state_mod.apply_round(
        state_mod.genesis("king_hk", 0), challenger_hotkey="chal_hk", challenger_uid=1,
        result=result, dethrone_cp=params.dethrone_cp, keep_former_kings=1)
    reward_uids = (1, 0) if transition.dethroned else (0,)
    weights = tuple(decayed_share_vector(list(reward_uids), len(HOTKEYS),
                                         decay=judge_cfg.scoring.king_decay,
                                         burn_uid=judge_cfg.scoring.burn_uid))
    post = dict(POST, king=transition.state.king_hotkey,
                former_kings=list(transition.state.former_kings))
    receipt = build_receipt(
        round_id=manifest.round_id, status="scored", epoch_start_block=boundary,
        epoch_block_hash=block_hash, base_seed=base_seed,
        seeds=RoundSeeds.derive(base_seed, judge_cfg.training), manifest=manifest,
        validator_hotkey=MAIN_KP.ss58_address,
        eval_context=EvalContext(pool_ref="p", pool_digest="sha256:" + "e" * 64,
                                 window_ids=tuple(s.series_id for s in king_scores),
                                 n_windows=len(king_scores),
                                 num_samples=judge_cfg.eval.num_samples),
        entry_scores=(
            EntryScores("king", size, "king_hk", 0,
                        tuple(WindowScoreRecord.from_score(s) for s in king_scores)),
            EntryScores("challenger", size, "chal_hk", 1,
                        tuple(WindowScoreRecord.from_score(s) for s in chal_scores)),
        ),
        verdict=VerdictRecord.from_round(result, transition, params=params,
                                         bootstrap_seed=base_seed, king_tenure_rounds=0),
        reward_uids=reward_uids, weights=weights,
        netconfig_seq=netconfig[0], netconfig_digest=netconfig[1],
        manifest_sha256=hashlib.sha256(mtext.encode("utf-8")).hexdigest(),
        state_prior_sha256=state_digest((prev or {}).get("state_post") or POST),
        state_post=post,
    )
    if tamper is not None:
        receipt = tamper(receipt)
    receipt = sign_receipt(receipt, sign_with)
    if publish:
        rtext = dump_receipt(receipt)
        store.put_text(manifest_round_key(manifest.round_id), mtext)
        store.put_text(receipt_round_key(manifest.round_id, MAIN_KP.ss58_address), rtext)
        store.put_text(receipt_latest_key(MAIN_KP.ss58_address), rtext)
    link = {"round_id": manifest.round_id, "manifest_sha256": receipt.manifest_sha256,
            "state_post": post}
    return receipt, link


def _runner(scfg, store, tmp_path):
    return sat_mod.build_satellite(scfg, state_dir=tmp_path, reader=store, hotkey="sat_hk")


def _burn(scfg):
    return decayed_share_vector([], len(HOTKEYS), decay=scfg.scoring.king_decay,
                                burn_uid=scfg.scoring.burn_uid)


# ── clean replay ─────────────────────────────────────────────────────────────


def test_clean_chain_sets_the_derived_vector(scfg, tmp_path):
    from cascade.audit.checks import derive_weights

    store = DictStore()
    r0, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    want, _ = derive_weights(r0, scfg)
    assert client.pushed[-1] == want
    assert doc["mode"] == sat_mod.FOLLOWING and doc["last_verified_round"] == r0.round_id
    # two more settlements land between polls: the walk replays both, oldest first
    r1, l1 = make_settlement(scfg, store, 1, prev=l0)
    r2, _ = make_settlement(scfg, store, 2, prev=l1)
    client.block = BOUNDARIES[2] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "" and doc["last_verified_round"] == r2.round_id
    assert client.pushed[-1] == derive_weights(r2, scfg)[0]
    # the status file exists
    assert (tmp_path / "status" / "satellite-sat_hk.json").is_file()


def test_state_survives_a_restart(scfg, tmp_path):
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    _runner(scfg, store, tmp_path).poll(FakeClient(BOUNDARIES[0] + 300))
    r1, _ = make_settlement(scfg, store, 1, prev=l0)
    runner = _runner(scfg, store, tmp_path)          # restarted: anchor from disk
    doc = runner.poll(FakeClient(BOUNDARIES[1] + 300))
    assert doc["last_verified_round"] == r1.round_id and doc["last_failing_check"] == ""


# ── every tamper class burns ─────────────────────────────────────────────────


def _scale_chal(rc):
    rows = list(rc.entry_scores)
    ch = rows[1]
    rows[1] = replace(ch, scores=tuple(replace(s, mase=s.mase * 2.0) for s in ch.scores))
    return replace(rc, entry_scores=tuple(rows))


def _shift_weights(rc):
    w = list(rc.weights)
    w[2] += w[1]
    w[1] = 0.0
    return replace(rc, weights=tuple(w))


def _bad_digest(rc):
    return replace(rc, manifest_sha256="0" * 64)


def _bad_prior(rc):
    return replace(rc, state_prior_sha256="f" * 64)


def _claim_bench_pass(rc):
    return replace(rc, verdict=replace(rc.verdict, bench_gate_passed=True))


@pytest.mark.parametrize("kwargs, check", [
    ({"sign_with": OTHER_KP}, "main-signature"),
    ({"sign_manifest_with": OTHER_KP}, "manifest-signature"),
    ({"tamper": _bad_digest}, "manifest-digest"),
    ({"tamper": _bad_prior}, "continuity"),
    ({"tamper": _scale_chal}, "verdict"),
    ({"tamper": _shift_weights}, "weights"),
    ({"tamper": _claim_bench_pass}, "bench-gate"),
])
def test_tamper_burns(scfg, tmp_path, kwargs, check):
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)
    good = client.pushed[-1]
    assert good != _burn(scfg)
    make_settlement(scfg, store, 1, prev=l0, **kwargs)
    client.block = BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["mode"] == sat_mod.BURNING
    assert doc["last_failing_check"] == check, doc
    assert client.pushed[-1] == _burn(scfg)
    # stays burning, and the failing check is re-reported every push
    client.block += 400
    doc = runner.poll(client)
    assert doc["last_failing_check"] == check and client.pushed[-1] == _burn(scfg)


def test_reward_uid_not_registered_burns(scfg, tmp_path):
    store = DictStore()
    make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    hotkeys = list(HOTKEYS)
    hotkeys[1] = "newcomer_hk"           # uid 1 recycled to a hotkey the receipt does not pay
    client = FakeClient(BOUNDARIES[0] + 300, hotkeys)
    doc = runner.poll(client)
    assert doc["mode"] == sat_mod.BURNING and doc["last_failing_check"] == "reward-uid"
    assert client.pushed[-1] == _burn(scfg)
    # re-registered: the verified vector is set again
    client.hotkeys = list(HOTKEYS)
    client.block += 400
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "" and client.pushed[-1] != _burn(scfg)


def test_lane_payout_tamper_burns(scfg, tmp_path):
    """A lane-B payout record whose multiplier does not re-derive."""
    from cascade.eval.expert_lane import LaneBWin, reward_multiplier
    from cascade.shared.emission import lane_parts
    from cascade.shared.era import era_length_blocks

    lb_cfg = replace(scfg, lane_b=replace(scfg.lane_b, from_block=BOUNDARIES[0] - GRID))
    domain = lb_cfg.lane_b.domains[0]

    def with_lane_b(mult_bump: float):
        def tamper(rc):
            b = int(rc.epoch_start_block)
            eb = int(era_length_blocks(lb_cfg.round, b))
            merge = BOUNDARIES[0] - GRID          # merged at an earlier settlement
            age = max(0, (b - merge) // eb)
            mult = reward_multiplier(age, full_eras=5,
                                     half_life_rounds=lb_cfg.lane_b.half_life_rounds,
                                     floor=lb_cfg.lane_b.floor)
            a, bb = lane_parts(list(rc.reward_uids), len(rc.weights),
                               king_decay=lb_cfg.scoring.king_decay,
                               burn_uid=lb_cfg.scoring.burn_uid, lane_b_cfg=lb_cfg.lane_b,
                               wins=[LaneBWin(uid=2, domain=domain, multiplier=mult)])
            pay = {"block": b, "era_blocks": eb,
                   "wins": [{"domain": domain, "hotkey": "x2_hk", "uid": 2, "merge_block": merge,
                             "age": age, "multiplier": mult * mult_bump,
                             "gain_lcb": 0.03, "tier": 0.02, "full_eras": 5,
                             "below_floor": False}]}
            post = dict(rc.state_post, lane_b={domain: ["x2_hk", 2, merge, "r0", 0.03, 0.02, 5]})
            return replace(rc, lane_b=(), lane_b_pay=pay, state_post=post,
                           weights=tuple(x + y for x, y in zip(a, bb, strict=True)))
        return tamper

    store = DictStore()
    _, l0 = make_settlement(lb_cfg, store, 0, tamper=with_lane_b(1.0))
    l0["state_post"] = json.loads(store.get_text(
        receipt_round_key(l0["round_id"], MAIN_KP.ss58_address)))["state_post"]
    runner = _runner(lb_cfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    assert client.pushed[-1][2] > 0                       # the lane-B slot is paid
    make_settlement(lb_cfg, store, 1, prev=l0, tamper=with_lane_b(0.5))
    client.block = BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "weights" and client.pushed[-1] == _burn(lb_cfg)


def test_broken_chain_link_fails_continuity(scfg, tmp_path):
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)
    # settlement 1 is never published; settlement 2 names it as its predecessor
    _, l1 = make_settlement(scfg, store, 1, prev=l0, publish=False)
    make_settlement(scfg, store, 2, prev=l1)
    client.block = BOUNDARIES[2] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "continuity" and client.pushed[-1] == _burn(scfg)


# ── silence, recovery ────────────────────────────────────────────────────────


def test_main_silent_holds_then_burns(scfg, tmp_path):
    store = DictStore()
    make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)
    good = client.pushed[-1]
    stale = runner.sat.stale_after_blocks
    assert stale == 3600
    # past two grids but inside stale_after_blocks: re-assert the verified vector
    client.block = BOUNDARIES[0] + stale - 10
    n = len(client.pushed)
    doc = runner.poll(client)
    assert len(client.pushed) == n + 1 and client.pushed[-1] == good
    assert doc["mode"] in (sat_mod.HOLDING, sat_mod.FOLLOWING)
    # past stale_after_blocks: burn
    client.block = BOUNDARIES[0] + stale + 200
    doc = runner.poll(client)
    assert doc["mode"] == sat_mod.BURNING and client.pushed[-1] == _burn(scfg)
    assert doc["last_failing_check"] == ""


def test_recovery_after_a_clean_receipt(scfg, tmp_path):
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)
    _, l1 = make_settlement(scfg, store, 1, prev=l0, tamper=_shift_weights)
    client.block = BOUNDARIES[1] + 300
    assert runner.poll(client)["last_failing_check"] == "weights"
    r2, _ = make_settlement(scfg, store, 2, prev=l1)
    client.block = BOUNDARIES[2] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "" and doc["last_verified_round"] == r2.round_id
    assert client.pushed[-1] != _burn(scfg)


def test_unreadable_bench_report_holds_not_burns(scfg, tmp_path, monkeypatch):
    """A check that cannot run is retried, never a verdict."""
    import cascade.audit.main as audit_main

    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)
    good = client.pushed[-1]
    make_settlement(scfg, store, 1, prev=l0, tamper=_claim_bench_pass)
    monkeypatch.setattr(audit_main, "fetch_bench_inputs", lambda *a, **k: None)
    client.block = BOUNDARIES[0] + 3500      # inside stale_after_blocks of round 0
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "" and doc["last_verified_round"] != ""
    assert runner.state.anchor["round_id"] == l0["round_id"]     # not advanced
    assert client.pushed[-1] == good


# ── DEC-CA-0057 overlay ──────────────────────────────────────────────────────


def test_overlay_stamped_receipt_replays_under_the_overlay(scfg, tmp_path):
    from cascade.shared.netconfig import (
        body_digest,
        build_config,
        next_overlay,
        seq_key,
        sign_overlay,
    )

    body = next_overlay({"scoring": {"bootstrap_alpha": 0.1}}, BOUNDARIES[0], None, network={"netuid": 91, "chain": "bittensor:finney"})
    store = DictStore()
    store.put_text(seq_key(1), json.dumps(sign_overlay(body, MAIN_KP)))
    judged = build_config(scfg, body)
    assert judged.koth_params(block=BOUNDARIES[0]) != scfg.koth_params(block=BOUNDARIES[0])
    _, l0 = make_settlement(judged, store, 0, netconfig=(1, body_digest(body)))
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    assert client.pushed[-1] != _burn(scfg)
    # the same receipt WITHOUT the stamp is judged under the base config: the
    # recorded params no longer match
    make_settlement(judged, store, 1, prev=l0)
    client.block = BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "koth-params"


def test_overlay_digest_mismatch_fails(scfg, tmp_path):
    from cascade.shared.netconfig import build_config, next_overlay, seq_key, sign_overlay

    body = next_overlay({"scoring": {"bootstrap_alpha": 0.1}}, BOUNDARIES[0], None, network={"netuid": 91, "chain": "bittensor:finney"})
    store = DictStore()
    store.put_text(seq_key(1), json.dumps(sign_overlay(body, MAIN_KP)))
    make_settlement(build_config(scfg, body), store, 0, netconfig=(1, "ab" * 32))
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "netconfig" and client.pushed[-1] == _burn(scfg)


# ── config ───────────────────────────────────────────────────────────────────


def test_satellite_table_parses(scfg):
    from cascade.shared.config.satellite import parse_satellite, satellite_config

    assert satellite_config(scfg).stale_after_blocks == 3600
    assert parse_satellite(None).stale_after_blocks == 3600
    with pytest.raises(ValueError):
        parse_satellite({"stale_after_blocks": 0})
    with pytest.raises(ValueError):
        parse_satellite({"bogus": 1})


# ── bench gate replayed from the TRAINER-SIGNED report ──────────────────────


def _publish_bench_report(store: DictStore, round_id: str) -> None:
    """A trainer-signed report whose rows make the challenger significantly
    worse on BOOM: the gate BLOCKS."""
    import numpy as np

    from cascade.shared.bench_report import (
        BenchEntry,
        BenchReport,
        bench_report_key,
        bench_rows_key,
        dump_bench_report,
        dump_bench_rows,
        sign_bench_report,
    )
    from cascade.shared.manifest import BenchScores

    def rows(boom: float) -> dict:
        rng = np.random.default_rng(0)
        out = {}
        for suite in ("gift", "boom", "time"):
            k = boom if suite == "boom" else 1.0
            out[suite] = {"status": "ok", "rows": [
                {"item": f"{suite}/{i}", "crps": float(c * k), "mase": float(m * k)}
                for i, (c, m) in enumerate(rng.uniform(0.5, 1.5, size=(40, 2)))]}
        return out

    six = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps",
           "time_mase")
    entries = []
    for role, hk, uid, ptr, boom, score in (("king", "king_hk", 0, CKPT_KING, 1.0, 1.0),
                                            ("challenger", "chal_hk", 1, CKPT_CHAL, 1.2, 0.9)):
        text = dump_bench_rows(rows(boom))
        key = bench_rows_key(ptr)
        store.put_text(key, text)
        entries.append(BenchEntry(role=role, size="", miner_hotkey=hk, miner_uid=uid,
                                  trained_pointer=ptr,
                                  scores=BenchScores(**dict.fromkeys(six, score)),
                                  rows_key=key,
                                  rows_sha256=hashlib.sha256(text.encode()).hexdigest()))
    rep = sign_bench_report(BenchReport(round_id=round_id, created_block=1,
                                        entries=tuple(entries)), TRAINER_KP)
    store.put_text(bench_report_key(round_id), dump_bench_report(rep))


@pytest.mark.parametrize("recorded, check", [(False, ""), (True, "bench-gate")])
def test_bench_gate_replayed_from_signed_rows(scfg, tmp_path, recorded, check):
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    runner.poll(client)

    def stamp(rc):
        return replace(rc, verdict=replace(rc.verdict, bench_gate_passed=recorded))
    r1, _ = make_settlement(scfg, store, 1, prev=l0, tamper=stamp)
    _publish_bench_report(store, r1.round_id)
    client.block = BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["last_failing_check"] == check, doc
    assert (client.pushed[-1] == _burn(scfg)) == bool(check)
