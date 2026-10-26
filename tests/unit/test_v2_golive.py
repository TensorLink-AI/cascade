"""DEC-CA-0050 go-live: v2 lane A and the later lane-B launch.

`[v2] from_block` (0 = off, the mainnet placeholder) flips lane A to the
Cascade-MOE-28M contract for every era starting at or after it (a new lineage
from random init). Lane B has its own era-start launch block; its later
launch arms the era-denominated lane-B payout with its TTL, and the
receipt `v2_block` stamp the audit replays. Everything before the block is
byte-identical to the pre-v2 release."""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.config import V2Config, load_chain_config, parse_lane_b
from cascade.shared.era import (
    era_index_of,
    settlement_contract_block,
    v2_first_era,
    v2_generation_usable,
    v2_settlement_active,
)
from cascade.shared.manifest import contract_digest, contract_payload

ROOT = Path(__file__).resolve().parents[2]


def _text(name="chain.toml"):
    return (ROOT / name).read_text()


# ── config: independent era-aligned launches ────────────────────────────────


def test_mainnet_ships_inert_with_the_v2_contract_scheduled():
    cfg = load_chain_config(ROOT / "chain.toml")
    # DEC-CA-0054 rides the go-live: the bench gate enforces from the v2 block
    # DEC-CA-0055's launch economics ship configured but inert until the block
    # (the pre-launch burn waits for its on-chain vote: no start block yet)
    assert cfg.v2 == V2Config(from_block=0, launch_at_utc="2026-10-30T12:00:00Z",
                              commit_open_at_utc="2026-10-30T09:00:00Z",
                              prelaunch_burn_readiness="r2",
                              bench_gate_mode="enforce", random_init_eras=14,
                              reign_warm_start_blocks=21600, prelaunch_burn=0.95,
                              carryover_king_share=0.05, king_ladder=True,
                              win_margin_start=0.10, win_margin_end=0.01,
                              margin_warmup_blocks=18000, king_check_margin=0.2,
                              # DEC-CA-0063: armed with the v2 go-live ("v2" = 0 now)
                              bench_source="validator", bench_source_from_block=0,
                              # DEC-CA-0063 item 9: settlement cap / per-sweep allowance
                              bench_budget_hours=6.0, bench_sweep_hours=2.0)
    assert not cfg.v2.active(10**9)
    assert cfg.lane_b.from_block == 0 and not cfg.lane_b.active(10**9)
    v2c = cfg.training.v2_contract
    assert v2c is not None and v2c.base_arch == "cascade_moe"
    assert (v2c.router_lr_scale, v2c.route_bias_rate_scale) == (0.1, 0.25)
    # inert: every block resolves to the toto2 contract, same object
    assert cfg.training.at_block(10**9) is cfg.training
    assert cfg.lane_b.domains == ("econ_fin", "energy", "healthcare", "nature", "sales",
                                  "transport", "web_cloudops")
    assert (cfg.lane_b.full_pay_tiers, cfg.lane_b.half_life_rounds, cfg.lane_b.floor,
            cfg.lane_b.win_ttl_eras) == (((0.05, 10), (0.02, 5), (0.01, 2)), 1.0, 0.1, 28)
    assert cfg.lane_b.replay_settlements == 29 * cfg.round.era_settlements


def test_testnet_is_armed_on_its_era_grid():
    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    n = cfg.v2.from_block
    assert n > 0 and n >= cfg.round.rolling_from_block
    era_len = cfg.round.epoch_blocks * cfg.round.era_settlements
    assert n % era_len == 0
    assert cfg.lane_b.from_block == n + 1          # explicit testnet launch preserved
    t = cfg.training
    assert t.at_block(n - 1) is t                  # every pre-v2 era: toto2, same object
    assert t.at_block(n).base_arch == "cascade_moe"
    assert contract_digest(t.at_block(n)) == contract_digest(t.v2_contract)
    assert contract_digest(t.at_block(n - 1)) == contract_digest(t) != contract_digest(
        t.at_block(n))


def test_v2_block_must_be_an_era_boundary(tmp_path):
    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    era_len = cfg.round.epoch_blocks * cfg.round.era_settlements
    text = _text("chain.testnet.toml")
    good = cfg.v2.from_block + era_len
    for bad in (cfg.v2.from_block + cfg.round.epoch_blocks, cfg.v2.from_block + 1):
        p = tmp_path / "bad.toml"
        p.write_text(text.replace(f"from_block = {cfg.v2.from_block}", f"from_block = {bad}"))
        with pytest.raises(ValueError, match="era boundary"):
            load_chain_config(p)
    p = tmp_path / "good.toml"
    p.write_text(text.replace(f"from_block = {cfg.v2.from_block}", f"from_block = {good}"))
    assert load_chain_config(p).v2.from_block == good


def test_v2_needs_its_contract_and_the_same_seed_salt(tmp_path):
    text = _text("chain.testnet.toml")
    n = load_chain_config(ROOT / "chain.testnet.toml").v2.from_block
    i = text.index("[v2.training]")
    j = text.index("[lane_b]")
    p = tmp_path / "a.toml"
    p.write_text(text[:i] + text[j:])
    with pytest.raises(ValueError, match=r"\[v2.training\]"):
        load_chain_config(p)
    p.write_text(text[:i] + text[i:j].replace("train_seed_salt    = 1337",
                                               "train_seed_salt    = 7") + text[j:])
    with pytest.raises(ValueError, match="train_seed_salt"):
        load_chain_config(p)
    assert n > 0


def test_lane_b_can_stay_off_or_launch_after_lane_a():
    raw = tomllib.loads(_text("chain.testnet.toml"))
    n = raw["v2"]["from_block"]
    era = raw["round"]["epoch_blocks"] * raw["round"]["era_settlements"]
    raw["lane_b"].pop("from_block")
    assert load_chain_config(raw=raw).lane_b.from_block == 0    # absent defaults to off
    raw["lane_b"]["from_block"] = 0
    assert not load_chain_config(raw=raw).lane_b.active(n + 100 * era)
    start = n + 2 * era
    raw["lane_b"]["from_block"] = start
    cfg = load_chain_config(raw=raw)
    assert cfg.v2.active(n) and not cfg.lane_b.active(n + era)
    assert not cfg.lane_b.active(start)            # launch-boundary settlement is prior era
    assert cfg.lane_b.active(start + cfg.round.epoch_blocks)


@pytest.mark.parametrize("start", [-1, 1.5, True, "v2", 8189400, 8190001])
def test_lane_b_launch_rejects_invalid_blocks(start):
    raw = tomllib.loads(_text("chain.testnet.toml"))
    raw["lane_b"]["from_block"] = start
    with pytest.raises(ValueError, match=r"\[lane_b\] from_block"):
        load_chain_config(raw=raw)


def test_lane_b_cannot_launch_without_v2():
    raw = tomllib.loads(_text("chain.toml"))
    raw["lane_b"]["from_block"] = 14400
    with pytest.raises(ValueError, match=r"needs \[v2\] from_block"):
        load_chain_config(raw=raw)


def test_settlement_semantics_at_the_rollover():
    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    n, eb = cfg.v2.from_block, cfg.round.epoch_blocks
    # the settlement AT the v2 block belongs to the last pre-v2 era
    assert not v2_settlement_active(cfg, n) and not cfg.lane_b.active(n)
    assert settlement_contract_block(cfg.round, n) == n - eb * cfg.round.era_settlements
    # the first settlement of the first v2 era
    assert v2_settlement_active(cfg, n + eb) and cfg.lane_b.active(n + eb)
    assert settlement_contract_block(cfg.round, n + eb) == n
    assert v2_first_era(cfg) == era_index_of(cfg.round, n)


def test_lineage_reset_rule():
    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    n = cfg.v2.from_block
    first = v2_first_era(cfg)
    era_len = cfg.round.epoch_blocks * cfg.round.era_settlements
    # pre-v2 eras: every generation counts
    assert v2_generation_usable(cfg, n - era_len, 0)
    # v2 eras: only a generation fired at/after the v2 block (eff >= first + 2)
    assert not v2_generation_usable(cfg, n, 0)
    assert not v2_generation_usable(cfg, n + 5 * era_len, first + 1)
    assert v2_generation_usable(cfg, n + 5 * era_len, first + 2)
    # a config without [v2] never masks
    assert v2_generation_usable(SimpleNamespace(round=cfg.round), n, 0)


# ── trainer: the ledger, the promotion engine, the leg flag ──────────────────


def test_rolling_ledger_masks_toto2_generations_in_v2_eras():
    from cascade.trainer import rolling as R

    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    first = v2_first_era(cfg)
    era_len = cfg.round.epoch_blocks * cfg.round.era_settlements
    gens = {"3": {"members": [["toto-a", "toto2-4m"]], "effective_era": first - 4},
            "4": {"members": [["toto-b", "toto2-4m"]], "effective_era": first + 1},
            "5": {"members": [["moe-a", "cascade-moe-28m"]], "effective_era": first + 3}}

    def init(idx):
        start = idx * era_len
        return R.era_init(gens, idx, lambda eff: v2_generation_usable(cfg, start, eff))

    assert init(first - 1) == (3, 0, "toto-a", "toto2-4m")      # pre-v2: unchanged
    assert init(first) == (0, 0, "", "")                          # first v2 era: random
    assert init(first + 1) == (0, 0, "", "")                      # toto2 gen 4 masked
    assert init(first + 2) == (0, 0, "", "")
    assert init(first + 3) == (5, 0, "moe-a", "cascade-moe-28m")  # the MoE lineage
    assert R.era_init(gens, first + 1) == (4, 0, "toto-b", "toto2-4m")   # no predicate


def test_promotion_engine_starts_the_v2_lineage(tmp_path):
    from cascade.shared.promotion import PromotedMember
    from cascade.trainer.promotion import Candidate, TrainerPromotion

    eng = TrainerPromotion(reign_threshold=1, k_max=2, quality_epsilon=0.05,
                           state_path=tmp_path / "p.json")
    eng.generation = 4
    eng.members = (PromotedMember("toto-a", "toto2-4m", "r1", 0.5),)
    eng.candidates = (Candidate("toto-c", "toto2-4m", "hk", "king", "r2", 1, 0.4),)
    assert eng.start_lineage(9000) and not eng.start_lineage(9000)
    assert eng.members == () and eng.candidates == () and eng.generation == 4
    again = TrainerPromotion.load(reign_threshold=1, k_max=2, quality_epsilon=0.05,
                                  state_path=tmp_path / "p.json",
                                  pointer_path=tmp_path / "ptr.json")
    assert again.lineage_block == 9000 and again.members == ()
    entry = SimpleNamespace(trained_pointer="moe-k", size="cascade-moe-28m",
                            miner_hotkey="hk", role="king",
                            scores=SimpleNamespace(gifteval_crps=1, gifteval_mase=1,
                                                   boom_crps=1, boom_mase=1,
                                                   time_crps=1, time_mase=1))
    report = SimpleNamespace(round_id="r9", created_block=9100, entries=(entry,))
    # a v2 round trained from random init seeds the new lineage …
    v2_round = SimpleNamespace(round_id="r9", warm_start_ckpt="",
                               era={"start_block": 9000})
    assert eng.record_bench(v2_round, report) == 1
    # … a pre-v2 round never does (even a random-init one)
    old = SimpleNamespace(round_id="r8", warm_start_ckpt="", era={"start_block": 8400})
    report2 = SimpleNamespace(round_id="r8", created_block=9000, entries=(
        replace_ns(entry, trained_pointer="toto-z"),))
    assert eng.record_bench(old, report2) == 0


def replace_ns(ns, **kw):
    return SimpleNamespace(**{**vars(ns), **kw})


def test_worker_argv_carries_router_slow_only_when_true():
    from cascade.trainer.remote import RemoteHost, worker_argv

    host = RemoteHost(name="h", host="127.0.0.1", port=22, user="root", key_path="k",
                      remote_python="python", workdir="/w")
    base = dict(gen_ref="r@sha256:" + "0" * 64, uid=1, hotkey="hk", role="king",
                base_seed=1, block=5, trainer_spec="m:C")
    assert worker_argv(host, **base) == worker_argv(host, **base, router_slow=False)
    assert "--router-slow" not in worker_argv(host, **base)
    assert "--router-slow" in worker_argv(host, **base, router_slow=True)


def test_worker_parses_router_slow():
    from cascade.trainer.worker import _build_parser

    need = ["--gen-ref", "r", "--uid", "1", "--hotkey", "h", "--role", "king",
            "--base-seed", "1", "--block", "1", "--trainer", "m:C"]
    assert _build_parser().parse_args(need).router_slow is False
    assert _build_parser().parse_args([*need, "--router-slow"]).router_slow is True


def test_era_router_slow_follows_the_latch_at_the_seed_block():
    from cascade.trainer.loop import TrainerRunner

    cfg = load_chain_config(ROOT / "chain.testnet.toml")
    n = cfg.v2.from_block
    era_len = cfg.round.epoch_blocks * cfg.round.era_settlements
    era = SimpleNamespace(start_block=n + 2 * era_len, seed_block=n + era_len)

    def slow(opened):
        r = SimpleNamespace(cfg=cfg, lane_b_driver=SimpleNamespace(
            st=SimpleNamespace(gate_opened_block=opened)))
        return TrainerRunner._era_router_slow(r, era)

    assert not slow(0)                         # gate closed: full-rate router
    assert slow(n + era_len)                   # opened by the era's seed block
    assert not slow(n + era_len + 1)           # opened later: next era
    flat = replace(cfg, training=replace(cfg.training, v2_contract=replace(
        cfg.training.v2_contract, router_lr_scale=1.0, route_bias_rate_scale=1.0)))
    r = SimpleNamespace(cfg=flat, lane_b_driver=SimpleNamespace(
        st=SimpleNamespace(gate_opened_block=1)))
    assert not TrainerRunner._era_router_slow(r, era)    # knobs off ⇒ no flag


# ── slow router numerics ─────────────────────────────────────────────────────

torch = pytest.importorskip("torch")


def _tiny():
    from cascade.trainer.moe_model import CascadeMoEModel
    from tests.unit.test_moe_model import TINY

    torch.manual_seed(0)
    return CascadeMoEModel(TINY)


def _contract(**kw):
    base = dict(optimizer="normuon_adamw", base_lr=0.01, weight_decay=0.0,
                d_model=32, umup_base_d_model=32, adamw_lr_scale=0.2,
                router_lr_scale=0.1, route_bias_rate_scale=0.25, router_slow=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_router_lr_scaled_only_when_flagged():
    from cascade.trainer.moe_trainer import build_moe_optimizer

    m = _tiny()
    off = build_moe_optimizer(m, _contract())
    assert len(off.adamw.param_groups) == 1
    assert off.adamw.param_groups[0]["lr_scale"] == pytest.approx(0.2)
    on = build_moe_optimizer(m, _contract(router_slow=True))
    groups = on.adamw.param_groups
    assert len(groups) == 2
    router_ids = {id(p) for n, p in m.named_parameters() if n.endswith("router.weight")}
    assert router_ids and {id(p) for p in groups[1]["params"]} == router_ids
    assert not router_ids & {id(p) for p in groups[0]["params"]}
    assert groups[0]["lr_scale"] == pytest.approx(0.2)
    assert groups[1]["lr_scale"] == pytest.approx(0.2 * 0.1)
    # the trainer's LR loop multiplies every group by its scale
    for g in on.param_groups:
        g["lr"] = 1.0 * g.get("lr_scale", 1.0)
    assert groups[1]["lr"] == pytest.approx(0.02)


def test_bias_step_scaled_only_when_flagged():
    from cascade.trainer.moe_trainer import route_bias_scale, update_route_bias

    m = _tiny()
    layer = next(mod for mod in m.modules() if hasattr(mod, "route_bias"))
    load = torch.arange(layer.route_bias.numel(), dtype=torch.float32)
    layer.route_bias.zero_()
    update_route_bias(layer, load, route_bias_scale(_contract()))
    full = layer.route_bias.clone()
    layer.route_bias.zero_()
    layer.update_bias(load)
    assert torch.equal(full, layer.route_bias)           # scale 1 = the model's own step
    layer.route_bias.zero_()
    update_route_bias(layer, load, route_bias_scale(_contract(router_slow=True)))
    assert torch.allclose(layer.route_bias, full * 0.25)


def test_toto2_trainer_refuses_router_knobs():
    from cascade.trainer.moe_trainer import refuse_moe_knobs

    with pytest.raises(ValueError, match="router_lr_scale"):
        refuse_moe_knobs(SimpleNamespace(base_arch="toto2", router_lr_scale=0.1))


def test_router_knobs_are_drop_when_default_and_flag_never_hashed():
    cfg = load_chain_config(ROOT / "chain.toml")
    t = cfg.training
    payload = contract_payload(t)
    for k in ("router_lr_scale", "route_bias_rate_scale", "router_slow",
              "v2_from_block", "v2_contract"):
        assert k not in payload
    assert contract_digest(replace(t, router_slow=True)) == contract_digest(t)
    v2 = t.v2_contract
    assert contract_payload(v2)["router_lr_scale"] == 0.1


# ── lane-B queue dedup ───────────────────────────────────────────────────────


def test_lane_b_queue_admits_a_commitment_once():
    from cascade.trainer.expert_lane import MAX_LEG_ATTEMPTS, LaneBQueue, LaneBSubmission

    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"]})
    q = LaneBQueue()
    sub = LaneBSubmission(hotkey="a", uid=1, gen_ref="a/g@sha256:" + "1" * 64,
                          domain="energy", commit_block=100)
    assert q.submit(sub, lb) and not q.submit(sub, lb)              # pending
    leg = q.start(sub, {1: [0]}, "king", 0)
    assert not q.submit(replace(sub), lb)                           # in flight
    q.finish(leg)
    assert not q.submit(replace(sub), lb)                           # finished: never again
    back = LaneBQueue.from_json(q.to_json())
    assert not back.submit(replace(sub), lb)                        # survives a restart
    # a re-commit at a NEW block is a new submission
    assert back.submit(replace(sub, commit_block=200), lb)
    # a failed leg frees its domain and may retry, up to the cap
    q2 = LaneBQueue()
    for i in range(MAX_LEG_ATTEMPTS):
        assert q2.submit(replace(sub), lb), i
        q2.finish(q2.start(q2.pending[0], {1: [0]}, "king", 0), ok=False)
    assert not q2.submit(replace(sub), lb)


# ── miner CLI: expert commitments ────────────────────────────────────────────


def test_expert_domain_validation():
    from cascade.miner.cli import expert_domain_error

    cfg = load_chain_config(ROOT / "chain.toml")
    assert expert_domain_error(cfg, "energy") is None
    assert "unknown lane-B domain" in expert_domain_error(cfg, "crypto")
    bare = SimpleNamespace(lane_b=SimpleNamespace(domains=()))
    assert "no [lane_b] domains" in expert_domain_error(bare, "energy")


def test_deploy_commits_an_expert_entry(monkeypatch, capsys):
    from cascade.miner import cli
    from cascade.shared import chain as chain_mod
    from cascade.shared.expert_manifest import parse_expert_commit

    commits = []

    class _Client:
        def current_block(self):
            return 1000

        def commit_submission(self, payload, *, blocks_until_reveal):
            commits.append(payload)

    monkeypatch.setattr(chain_mod.ChainClient, "from_config",
                        classmethod(lambda cls, *a, **k: _Client()))
    monkeypatch.setattr(cli, "_live_cfg", lambda cfg, client: cfg)
    ref = "me/gen@sha256:" + "a" * 64
    args = argparse.Namespace(
        chain_toml=None, network="test", wallet_name="w", wallet_hotkey="h",
        wallet_path=None, blocks_until_reveal=1, reveal_now=False, next_epoch=False,
        skip_verify=True, hub_repo=None, hub_namespace=None, hf_repo=None, ref=ref,
        repo_dir=Path("."), expert_domain="energy")
    assert cli._cmd_deploy(args) == 0
    assert parse_expert_commit(commits[-1]) == ("energy", ref)
    args.expert_domain = "crypto"
    assert cli._cmd_deploy(args) == 2 and len(commits) == 1
