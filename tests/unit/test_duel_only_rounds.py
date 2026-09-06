"""Duel-only rounds (``[round] duel_from_block`` / ``duel_field_cap``).

From the gate block the heat is skipped: the screened field seats into the
duel in reveal order — as many as the fleet's lanes can finish inside the
epoch (or an explicit cap) — the overflow waits with its submission intact,
and the provisioner rents no heat fleet sized to fit the field. Inert at 0 —
every pre-gate path is byte-identical to the heat → final pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from cascade.shared.chain import Commitment
from cascade.shared.hippius import HubRef, HubUpload, StorageError
from cascade.trainer import loop as loop_mod
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner

REF = {k: f"{k}/gen-{k}@sha256:" + k * 64 for k in "abcd"}
REF_OUT = "cascade/ckpt-out@sha256:" + "e" * 64


# ── the GPU/registry boundaries, faked (mirrors test_trainer_round) ──────────


class _FakeStream:
    n_series = 3
    total_points = 192

    def __init__(self, digest="corpusdigest"):
        self.digest = digest

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def series(self):
        for _ in range(3):
            yield np.ones((1, 64))


class _FakeBaseTrainer:
    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
        for _ in stream:
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000, train_seconds=1.0,
                           metrics={"final_loss": 0.1})


def _patch_train_boundaries(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(
        loop_mod, "open_round_stream",
        lambda mode, gen_dir, *a, **k: _FakeStream(digest=f"digest-{gen_dir}"))
    monkeypatch.setattr(
        loop_mod, "upload_dir_to_hub_or_hf",
        lambda local_dir, repo, hub=None, *, hf_repo=None, hf_token=None:
            HubUpload(ref=HubRef.parse(REF_OUT), size_bytes=1))


def _commit(uid, hotkey, ref, block):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


def _field():
    # Reveal order c(6) < b(7) < d(8): UID order would seat b before c.
    return [_commit(0, "a", REF["a"], 5), _commit(1, "b", REF["b"], 7),
            _commit(2, "c", REF["c"], 6), _commit(3, "d", REF["d"], 8)]


def _screen_must_not_run(*a, **k):
    raise AssertionError("the heat screen ran on a duel-only round")


@pytest.fixture()
def duel_cfg(cfg):
    """Gate at block 1000, no explicit cap: a local final (1 lane) on a 12h
    grid with 3h legs fits 3 waves ⇒ king + 2 seats. Legacy screen mechanics
    (single finalist, inert tie cap) below the gate."""
    assert cfg.training.target_train_hours == 3.0
    return replace(cfg, round=replace(cfg.round, max_finalists=1, finalists=1,
                                      duel_from_block=1000, duel_field_cap=0,
                                      # flat 3600 grid (12h): block 5000 ⇒ boundary 3600
                                      epoch_blocks_prev=0, epoch_activation_block=0))


def _runner(cfg, tmp_path, monkeypatch, **kw):
    _patch_train_boundaries(monkeypatch)
    return TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                         use_sandbox=False, **kw)


def _burned(cfg, tmp_path) -> set[str]:
    p = tmp_path / cfg.round.submissions_db_path
    return set(json.loads(p.read_text())) if p.exists() else set()


# ── config round-trip (every knob needs the loader parse) ────────────────────


def test_duel_knobs_parse_validate_and_gate(tmp_path):
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    # The shipped toml may carry the armed keys; strip them to pin the defaults.
    bare = re.sub(r"^duel_(from_block|field_cap)\s*=.*$", "",
                  DEFAULT_CHAIN_TOML.read_text(), flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(bare)
    cfg = load_chain_config(p)
    assert cfg.round.duel_from_block == 0 and cfg.round.duel_field_cap == 0
    assert not cfg.round.duel_only(10**9)          # unset gate: never

    p.write_text(bare.replace(
        "\n[round]\n", "\n[round]\nduel_from_block = 5000\nduel_field_cap = 4\n", 1))
    cfg = load_chain_config(p)
    assert (cfg.round.duel_from_block, cfg.round.duel_field_cap) == (5000, 4)
    assert cfg.round.duel_only(5000) and cfg.round.duel_only(9000)
    assert not cfg.round.duel_only(4999) and not cfg.round.duel_only(None)

    p.write_text(bare.replace("\n[round]\n", "\n[round]\nduel_field_cap = -1\n", 1))
    with pytest.raises(ValueError, match="duel_field_cap"):
        load_chain_config(p)

    # An armed shipped toml gates on an epoch boundary, in step with the
    # scored horizon ladder (one flip, one announced block).
    shipped = load_chain_config(DEFAULT_CHAIN_TOML)
    if shipped.round.duel_from_block:
        assert shipped.round.duel_from_block % shipped.round.epoch_blocks == 0
        assert shipped.eval.scored_from_block in (0, shipped.round.duel_from_block)


def test_seats_follow_the_lanes_and_the_epoch():
    from cascade.shared.config import RoundConfig, duel_waves_that_fit

    assert duel_waves_that_fit(12.0, 3.0) == 3        # (12 − 1.5) // 3
    assert duel_waves_that_fit(3.0, 3.0) == 1         # never below one leg
    assert duel_waves_that_fit(12.0, 0.0) == 1
    # short grids scale the overhead (12.5%) instead of paying the 12h figure
    assert duel_waves_that_fit(0.5, 0.05) == 8        # (0.5 − 0.0625) // 0.05
    assert duel_waves_that_fit(3.0, 0.25) == 10       # (3 − 0.375) // 0.25
    auto = RoundConfig(duel_field_cap=0)
    assert auto.duel_seats(lanes=4, epoch_hours=12.0, leg_hours=3.0) == 11   # 4×3 − king
    assert auto.duel_seats(lanes=1, epoch_hours=12.0, leg_hours=3.0) == 2
    assert auto.duel_seats(lanes=0, epoch_hours=3.0, leg_hours=3.0) == 1     # floor: one seat
    capped = RoundConfig(duel_field_cap=5)
    assert capped.duel_seats(lanes=4, epoch_hours=12.0, leg_hours=3.0) == 5


# ── the round: seats, overflow, burn, marker ─────────────────────────────────


def test_duel_only_round_seats_by_reveal_order_and_holds_the_overflow(
        duel_cfg, tmp_path, monkeypatch):
    runner = _runner(duel_cfg, tmp_path, monkeypatch, screen_fn=_screen_must_not_run)
    commits = _field()
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=5000)

    chal = {e.miner_hotkey: e for e in manifest.entries_for_role("challenger")}
    assert set(chal) == {"c", "b"}                 # the two earliest reveals
    assert chal["c"].duel_rank == 0 and chal["b"].duel_rank == 1
    assert manifest.heat is None                   # no screen ran
    marker = json.loads((tmp_path / "1" / "heat_complete.json").read_text())
    assert marker == {"round_id": "1", "screened": 3, "finalists": ["c", "b"]}
    assert _burned(duel_cfg, tmp_path) == {"c", "b"}   # d keeps its submission

    # Next round: d seats without re-committing.
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=2, block=5100)
    assert [e.miner_hotkey for e in manifest.entries_for_role("challenger")] == ["d"]
    assert _burned(duel_cfg, tmp_path) == {"b", "c", "d"}


def test_pre_gate_round_keeps_the_heat_screen(duel_cfg, tmp_path, monkeypatch):
    calls: list[str] = []

    def screen(ckpt_dir, gen, base_seed, block=None):
        calls.append(gen.hotkey)
        return {"b": 0.9, "c": 0.2, "d": 0.5}[gen.hotkey]

    runner = _runner(duel_cfg, tmp_path, monkeypatch, screen_fn=screen)
    # block 10 ⇒ epoch boundary 0 < the gate at 1000
    manifest = runner.run_round(_field(), king_hotkey="a", base_seed=1, block=10)
    assert sorted(calls) == ["b", "c", "d"]
    assert [e.miner_hotkey for e in manifest.entries_for_role("challenger")] == ["c"]
    assert manifest.heat is not None
    assert _burned(duel_cfg, tmp_path) == {"b", "c", "d"}   # legacy: everyone screened burns


def test_duel_only_field_within_the_seats_seats_everyone(duel_cfg, tmp_path, monkeypatch):
    runner = _runner(duel_cfg, tmp_path, monkeypatch, screen_fn=_screen_must_not_run)
    commits = _field()[:3]                          # a (king), b, c
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=5000)
    assert sorted(e.miner_hotkey for e in manifest.entries_for_role("challenger")) == ["b", "c"]
    assert _burned(duel_cfg, tmp_path) == {"b", "c"}


def test_explicit_cap_overrides_the_fleet_derived_seats(duel_cfg, tmp_path, monkeypatch):
    one = replace(duel_cfg, round=replace(duel_cfg.round, duel_field_cap=1))
    runner = _runner(one, tmp_path, monkeypatch, screen_fn=_screen_must_not_run)
    manifest = runner.run_round(_field(), king_hotkey="a", base_seed=1, block=5000)
    assert [e.miner_hotkey for e in manifest.entries_for_role("challenger")] == ["c"]
    assert _burned(one, tmp_path) == {"c"}


def test_duel_only_retry_after_settle_reuses_the_seated_field(duel_cfg, tmp_path, monkeypatch):
    """The r47 shape on a duel-only round: the settle burned the seated pair
    and wrote the marker, then the final died. The retry re-seats exactly
    them — the waiting entrant is neither pulled in nor burned."""
    runner = _runner(duel_cfg, tmp_path, monkeypatch, screen_fn=_screen_must_not_run)
    commits = _field()
    real_train_final = runner._train_final

    def _boom(*a, **k):
        raise RuntimeError("final pod died")

    monkeypatch.setattr(runner, "_train_final", _boom)
    with pytest.raises(RuntimeError, match="final pod died"):
        runner.run_round(commits, king_hotkey="a", base_seed=1, block=5000)
    assert _burned(duel_cfg, tmp_path) == {"c", "b"}

    monkeypatch.setattr(runner, "_train_final", real_train_final)
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=5000)
    assert sorted(e.miner_hotkey for e in manifest.entries_for_role("challenger")) == ["b", "c"]
    assert _burned(duel_cfg, tmp_path) == {"c", "b"}


# ── the public standings name the waiting entrants ───────────────────────────


class _Store:
    def __init__(self):
        self.objects: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="", acl=None):
        self.objects[key] = text

    def get_text(self, key):
        if key not in self.objects:
            raise StorageError(key)
        return self.objects[key]


def test_duel_only_standings_name_the_waiting_entrants(duel_cfg, tmp_path, monkeypatch):
    store = _Store()
    runner = _runner(duel_cfg, tmp_path, monkeypatch, publish_stage_status=True,
                     screen_fn=_screen_must_not_run)
    runner._manifest_store = store
    runner.run_round(_field(), king_hotkey="a", base_seed=1, block=5000)

    doc = json.loads(store.objects["status/heat.json"])
    assert doc["round_id"] == "1" and doc["screened"] == 2
    assert "duel-only" in doc["no_screen_reason"]
    assert "1 wait" in doc["no_screen_reason"] and "cap" not in doc["no_screen_reason"]
    # the whole field is listed: seated in seat order, then the waiting entrant
    assert [(e["hotkey"], e["status"]) for e in doc["entrants"]] == [
        ("c", "seated"), ("b", "seated"), ("d", "waiting")]
    assert doc["duel_only"] is True


# ── provisioner plan: no heat fleet, final sized off the seats ───────────────


def test_plan_payload_marks_duel_only_rounds(duel_cfg, tmp_path):
    from cascade.trainer.main import _plan_payload

    class _Client:
        def __init__(self, block):
            self.block = block

        def current_block(self):
            return self.block

        def poll_commitments(self, include_history=False):
            return _field()

        def highest_incentive_hotkey(self):
            return "a"

    eb = duel_cfg.round.epoch_blocks
    payload = _plan_payload(duel_cfg, _Client(eb + 100), tmp_path)
    assert payload["next_boundary_block"] == 2 * eb        # past the gate at 1000
    assert payload["duel_only"] is True
    assert payload["finalists"] == 3                        # everyone: the fleet sizes to fit
    assert payload["max_finalists"] == 0
    assert payload["screened_challengers"] == 3

    capped = replace(duel_cfg, round=replace(duel_cfg.round, duel_field_cap=2))
    assert _plan_payload(capped, _Client(eb + 100), tmp_path)["finalists"] == 2

    legacy = replace(duel_cfg, round=replace(duel_cfg.round, duel_from_block=10**9))
    payload = _plan_payload(legacy, _Client(eb + 100), tmp_path)
    assert payload["duel_only"] is False
    assert payload["finalists"] == legacy.round.finalists
    assert payload["max_finalists"] == legacy.round.max_finalists


def test_size_fleet_no_heat_rents_no_heat_pods_and_queues_the_final():
    from cascade.provision.policy import StagePolicy, size_fleet
    from tests.unit.test_provision_loop import _policy

    pol = _policy()                                          # final: 2× L40S pods, 2 GPU each
    # 24 entrants + king = 25 legs; a 12h epoch fits 3 legs per lane ⇒ 9 lanes
    plan = size_fleet(24, 24, 1.0, 12.0, 3.0, pol, no_heat=True)
    assert (plan.heat.pods, plan.heat.slots) == (0, 0)
    assert plan.final.slots == 9
    assert plan.final.pods == 2                              # clamped by max_pods: legs queue
    roomy = _policy(final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=8,
                                      providers=("lium",), max_price_hr=3.0))
    assert size_fleet(24, 24, 1.0, 12.0, 3.0, roomy, no_heat=True).final.pods == 5
    assert size_fleet(24, 7, 1.0, 12.0, 3.0, pol).heat.pods > 0   # legacy path untouched
    assert size_fleet(0, 0, 1.0, 12.0, 3.0, pol, no_heat=True).final.slots == 1


def test_provisioner_rents_the_final_at_the_margin_for_a_duel_only_plan(tmp_path):
    """With no heat fleet there is nothing to defer the final behind: the
    margin trigger rents it directly, sized to fit king + field inside the
    epoch, even under the JIT (``final_rent_on = "heat_complete"``) policy."""
    from cascade.provision.state import load_state
    from tests.unit.test_provision_loop import PLAN, FakeProvider, _policy, cycle, make_loop

    prov = FakeProvider("lium")
    plan = dict(PLAN, eligible_challengers=12, screened_challengers=12,
                finalists=3, max_finalists=0, duel_only=True)
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, plan=plan,
                        policy=_policy(max_spend_per_round=100.0),
                        final_rent_on="heat_complete")
    cycle(loop)
    # 3h epoch, 0.25h legs ⇒ 6 legs per lane: 4 legs need one 2-GPU pod.
    assert prov.launched == ["cascade-900-final-0"]
    assert not any("heat" in name for name in prov.launched)
    assert load_state(tmp_path / "state.json").final_pending is False
