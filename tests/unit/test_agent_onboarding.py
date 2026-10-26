"""Agent preparation must be bounded, reproducible and honest about missing evidence."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.miner import cli
from cascade.miner.discovery_check import CAPABILITIES, check_snapshot
from cascade.miner.iterate import cache_context, settings_key
from cascade.miner.live_status import LiveStateUnavailable, status_snapshot
from cascade.miner.preflight import assess
from cascade.miner.training_context import contract_from_snapshot, live_context
from cascade.miner.verify import verify_repo
from cascade.shared.manifest import contract_digest, contract_payload

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts" / "example_generator"


def snapshot(cfg):
    now = datetime.now(UTC)
    contract = cfg.training.v2_contract.primary_size
    docs = {"chain": {"as_of": now.isoformat(), "current_block": 510, "era_blocks": 100, "epoch_blocks": 20},
            "competition": {"as_of": now.isoformat(), "warm_start": {
                "mode": "checkpoint", "pointer": "metro-v1:trained:hippius:cascade/init@sha256:" + "a" * 64,
                "era_start_block": 500}, "lane_a": {"active": True}, "lane_b": {"active": False},
                "lane_c": {"active": False}, "policy": {"generator": asdict(cfg.generator),
                                                          "generation": asdict(cfg.generation)},
                "admission": {"lane_a_spent_hotkeys": []},
                "training_contract": {"schema": 1, "era_start_block": 500,
                                      "body": contract_payload(contract), "digest": contract_digest(contract)},
                "agent_protocol": {"schema": 1, "capabilities": list(CAPABILITIES),
                                   "submission_status": {"schema": 1}}},
            "lane_a_queue": {"as_of": now.isoformat(), "entries": []}}
    return status_snapshot(docs, now=now, max_age=180)


def run(capsys, args):
    rc = cli.main([*args, "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["exit_code"] == rc
    return rc, result


def test_quick_verifier_runs_real_determinism_without_full_corpus(cfg):
    report = verify_repo(EXAMPLE, cfg, quick=True)
    assert report.ok and report.corpus_digest and report.scope == "quick"
    assert not report.runtime_skipped
    assert "run full verify" in report.render()


def test_quick_verifier_catches_unseeded_rng(cfg, tmp_path):
    d = tmp_path / "generator"
    shutil.copytree(EXAMPLE, d)
    path = d / "generator.py"
    path.write_text(path.read_text().replace("np.random.default_rng(self._seed)", "np.random.default_rng()"))
    result = verify_repo(d, cfg, quick=True)
    assert not result.ok and "non-deterministic" in str(result.to_dict())


def test_quick_verifier_timeout_is_a_structured_failure(cfg, monkeypatch):
    def timeout(*a, **kw):
        assert kw["timeout"] == 30
        raise subprocess.TimeoutExpired("worker", 30)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = verify_repo(EXAMPLE, cfg, quick=True)
    assert result.to_dict()["failures"][0]["reason"] == "runtime_timeout"


@pytest.mark.parametrize("damage", ["hash", "era", "unknown_field", "schema"])
def test_live_contract_cannot_silently_use_wrong_or_unrepresentable_rules(cfg, damage):
    data = snapshot(cfg)
    contract = data["training_contract"]
    if damage == "hash":
        contract["body"]["base_lr"] *= 2
    elif damage == "era":
        contract["era_start_block"] -= 100
    elif damage == "unknown_field":
        contract["body"]["unknown_training_semantics"] = True
        contract["digest"] = contract_digest(contract["body"])
    else:
        contract["schema"] = 2
    with pytest.raises(LiveStateUnavailable):
        contract_from_snapshot(data)


def test_live_context_pins_moe_contract_and_checkpoint_together(cfg, monkeypatch):
    data = snapshot(cfg)
    monkeypatch.setattr("cascade.miner.live_status.fetch_snapshot", lambda cfg: data)
    effective, pointer, _ = live_context(cfg)
    assert effective.screen_contract().base_arch == "cascade_pyramid"
    assert effective.eval.context_length == effective.training.context_length
    assert contract_digest(effective.training) == data["training_contract"]["digest"]
    assert pointer == data["warm_start"]["pointer"]


def test_cache_invalidates_for_contract_and_same_size_data_changes(cfg, tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    series = pool / "series.npy"
    np.save(series, np.ones(64))
    args = SimpleNamespace(pool_dir=pool, pool_ref="", train_hours=None, n_windows=None,
                           device="cpu", warm_start=None, _cache_context=cache_context(cfg, None))
    original = settings_key(args, 0)
    np.save(series, np.zeros(64))
    assert original != settings_key(args, 0)
    changed = replace(cfg, training=replace(cfg.training, base_lr=cfg.training.base_lr * 2))
    assert cache_context(cfg, None) != cache_context(changed, None)
    # JSON history and in-memory settings must compare identically.
    assert json.loads(json.dumps(original)) == original


def test_local_checkpoint_change_invalidates_cache(cfg, tmp_path):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"first")
    first = cache_context(cfg, tmp_path)
    weights.write_bytes(b"other")
    assert first != cache_context(cfg, tmp_path)


def test_missing_or_cross_lane_busy_queue_cannot_mean_idle(cfg):
    data = snapshot(cfg)
    data["lanes"]["b"] = {"active": True}
    checks = assess(data, lane="a", targets=(), hotkey="H", max_price=1., max_spend=5.)
    assert next(c for c in checks if c["check"] == "hotkey_idle")["state"] == "unknown"
    data["queues"]["b"] = {"queue": {"entries": [{"hotkey": "H", "status": "in_flight"}]}}
    checks = assess(data, lane="a", targets=(), hotkey="H", max_price=1., max_spend=5.)
    assert next(c for c in checks if c["check"] == "hotkey_idle")["state"] == "fail"


@pytest.mark.parametrize("state,end,expected", [
    ("pending", 0, "fail"), ("ended", 400, "fail"),
    ("winding_down", 400, "fail"), ("active", 500, "fail"),
    ("active", 600, "pass"), ("active", 0, "pass"),
])
def test_lane_c_preflight_checks_competition_admission(cfg, state, end, expected):
    data = snapshot(cfg)
    data["chain"] = {"current_block": 500}
    data["lanes"]["c"] = {"active": True, "targets": ["mkt:returns"],
                           "competitions": [{"kind": "mkt", "state": state,
                                             "until_block": end}]}
    checks = assess(data, lane="c", targets=("mkt:returns",), hotkey=None,
                    max_price=1., max_spend=5.)
    assert next(c for c in checks if c["check"] == "competition_active")["state"] == expected


def test_preflight_never_reads_wallet_or_sends_a_request(cfg, monkeypatch, capsys):
    data = snapshot(cfg)
    monkeypatch.setattr("cascade.miner.preflight.fetch_snapshot", lambda *a, **kw: data)
    monkeypatch.setattr("cascade.miner.preflight.identity_check", lambda *a: {"registered": True})
    monkeypatch.setattr("cascade.shared.chain.open_wallet", lambda *a: pytest.fail("wallet opened"))
    rc, result = run(capsys, ["preflight", str(EXAMPLE), "--hotkey", "H", "--max-price", "1", "--max-spend", "5"])
    assert rc == 0, result
    assert result["data"]["ready_to_propose"] and not result["data"]["authorized_to_spend"]
    assert not result["data"]["submitted"]


def test_preflight_missing_live_feed_is_retryable(cfg, monkeypatch, capsys):
    data = snapshot(cfg)
    data["fresh"] = False
    monkeypatch.setattr("cascade.miner.preflight.fetch_snapshot", lambda *a, **kw: data)
    rc, result = run(capsys, ["preflight", str(EXAMPLE)])
    assert rc == 3 and result["error"]["code"] == "preflight_unavailable"


def test_lane_c_preflight_checks_its_optional_generator(cfg, monkeypatch, capsys, tmp_path):
    cfg = replace(cfg, generator=replace(cfg.generator, min_channels=8))
    monkeypatch.setattr("cascade.shared.config.load_chain_config", lambda *_: cfg)
    monkeypatch.setattr("cascade.miner.preflight.fetch_snapshot", lambda *a, **k: snapshot(cfg))
    monkeypatch.setattr("cascade.miner.lane_c.recipe_error", lambda *_: None)
    monkeypatch.setattr("cascade.miner.lane_c.has_local_generator", lambda *_: True)
    shutil.copytree(EXAMPLE, tmp_path / "entry")
    (tmp_path / "entry/config.json").write_text(json.dumps({"channels": 7}))
    rc, result = run(capsys, ["preflight", str(tmp_path / "entry"), "--lane", "c", "--targets", "mkt:returns"])
    check = next(c for c in result["data"]["checks"] if c["check"] == "local_entry")
    assert rc == 1 and check["state"] == "fail"
    assert check["detail"]["scope"] == "recipe_and_generator"
    assert not check["detail"]["generator"]["ok"]
    assert not result["data"]["authorized_to_spend"]
    assert not result["data"]["ready_to_propose"]


def test_preflight_rejects_revealed_ref_from_the_wrong_lane(cfg, monkeypatch, capsys):
    data = snapshot(cfg)
    monkeypatch.setattr("cascade.miner.preflight.fetch_snapshot", lambda *a, **kw: data)
    monkeypatch.setattr("cascade.miner.preflight.identity_check", lambda *a:
                        {"registered": True, "ref_matches": True, "lane": "b", "targets": ["energy"]})
    rc, result = run(capsys, ["preflight", str(EXAMPLE), "--hotkey", "H", "--ref", "same-ref",
                              "--max-price", "1", "--max-spend", "5"])
    assert rc == 1
    assert next(c for c in result["data"]["checks"] if c["check"] == "revealed_lane_and_targets")["state"] == "fail"


def test_preflight_reads_actual_commitment_payload_without_wallet(cfg, monkeypatch):
    from cascade.interface.validation import format_commit
    from cascade.miner.preflight import identity_check

    ref = "vault/direct@sha256:" + "a" * 64
    client = SimpleNamespace(poll_commitments=lambda: [SimpleNamespace(
        hotkey="H", commit_block=100, payload=format_commit(ref))],
        registered_hotkeys=lambda: ["H"], pending_commitment=lambda hotkey: None)
    monkeypatch.setattr("cascade.shared.chain.ChainClient.from_config", lambda *a, **kw: client)
    monkeypatch.setattr("cascade.shared.chain.open_wallet", lambda *a, **kw: pytest.fail("wallet opened"))
    result = identity_check(cfg, "H", "finney", ref)
    assert result["registered"] and result["ref_matches"] and result["lane"] == "a"


def test_lane_c_upload_returns_an_immutable_ref_without_committing(cfg, monkeypatch, tmp_path, capsys):
    from cascade.shared.hippius import HubRef, HubUpload

    monkeypatch.setattr("cascade.miner.lane_c.recipe_error", lambda *a: None)
    monkeypatch.setattr("cascade.shared.hippius.upload_dir_to_hub", lambda *a:
                        HubUpload(HubRef("miner/recipe", "sha256:" + "a" * 64), 123))
    rc, result = run(capsys, ["lane-c", "upload", str(tmp_path), "--target", "mkt:returns", "--hub-repo", "miner/recipe"])
    assert rc == 0 and result["data"]["ref"] == "miner/recipe@sha256:" + "a" * 64
    assert not result["data"]["committed"] and not result["data"]["funded"]


def test_snapshot_command_limits_public_download_to_recipe_assets(cfg, monkeypatch, tmp_path, capsys):
    from cascade.miner.starter import marketcast_starter

    recipe = marketcast_starter(cfg.marketcast, "mkt:returns")
    (tmp_path / "marketcast.json").write_text(json.dumps(recipe))
    calls = []

    def build(out, mc, parsed, **kw):
        calls.append(mc.assets_map)
        return {"public": True}

    monkeypatch.setattr("cascade.trainer.marketcast_leg.build_training_snapshot", build)
    rc, result = run(capsys, ["lane-c", "snapshot", str(tmp_path), "--out", str(tmp_path / "train")])
    assert rc == 0 and not result["data"]["gpu_rented"]
    assert {k: list(v) for k, v in calls[0].items()} == recipe["data"]["assets"]


@pytest.mark.parametrize("lane,target", [("a", None), ("b", "energy"), ("c", "mkt:returns")])
def test_starters_work_offline_and_never_overwrite(cfg, monkeypatch, tmp_path, capsys, lane, target):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network access"))
    path = tmp_path / "entry"
    args = ["init", str(path), "--lane", lane] + (["--target", target] if target else [])
    rc, result = run(capsys, args)
    assert rc == 0, result
    assert (path / ("marketcast.json" if lane == "c" else "generator.py")).exists()
    rc, result = run(capsys, args)
    assert rc == 2 and result["error"]["code"] == "destination_exists"


def test_release_check_requires_current_contract_and_active_queues(cfg):
    data = snapshot(cfg)
    assert all(check_snapshot(data).values())
    data["training_contract"]["body"]["context_length"] += 32
    assert not check_snapshot(data)["effective_contract"]
    data["lanes"]["b"] = {"active": True}
    assert not check_snapshot(data)["lane_b_queue"]


def test_release_check_rejects_old_publisher_without_submission_tracking(cfg):
    data = snapshot(cfg)
    data["agent_protocol"].pop("submission_status")
    assert not check_snapshot(data)["submission_tracking"]


def test_real_publishers_supply_matching_clock_contract_and_live_queue(cfg):
    from cascade.shared.chain_status import build_chain_status
    from cascade.shared.era import era_start_block
    from cascade.shared.v2_status import publish_trainer_v2_status
    from tests.unit.test_v2_status import _Store

    block = 100
    store = _Store()
    now = datetime.now(UTC)
    current = SimpleNamespace(index=0, start_block=era_start_block(cfg.round, block),
                              warm_start_ckpt="", warm_start_size="", generation=0, member_index=0,
                              king=lambda: None)
    sched = SimpleNamespace(cfg=cfg, lane_b=None, state=SimpleNamespace(current=current),
                            r=SimpleNamespace(manifest_store=lambda: store),
                            ops=SimpleNamespace(burned=lambda: {"spent"}, queue=lambda:
                                SimpleNamespace(public_view=lambda: {"entries": []})))
    assert publish_trainer_v2_status(sched, block, clock=lambda: 0.)
    competition = store.writes[-1][1]
    clock = build_chain_status(cfg, current_block=block, commitments=[], as_of=now.isoformat())
    data = status_snapshot({"chain": clock, "competition": competition}, now=now, max_age=180)
    assert data["ready_for_training"]
    assert contract_from_snapshot(data).base_arch == cfg.training.base_arch
    assert data["freshness"]["lane_a_queue"]["fresh"]
    assert data["queues"]["a"]["entries"] == []
    assert data["admission"]["lane_a_spent_hotkeys"] == ["spent"]


def test_discovery_check_detects_html_instead_of_guide(cfg, monkeypatch, capsys):
    monkeypatch.setattr("cascade.miner.discovery_check.fetch_snapshot", lambda *a, **kw: snapshot(cfg))
    monkeypatch.setattr("cascade.miner.discovery_check.read_guide", lambda *a: "<!doctype html><h1>Not found</h1>")
    rc, result = run(capsys, ["discovery-check"])
    assert rc == 3 and result["data"]["remaining"] == ["agent_guide"]


def test_lane_b_local_test_trains_union_and_runs_acceptance(cfg, tmp_path, monkeypatch):
    from cascade.eval.scoring import WindowScore
    from cascade.miner.lane_tests import test_lane_b as experiment
    from cascade.trainer.contract import TrainResult

    cfg = replace(cfg, lane_b=replace(cfg.lane_b, require_public=(), min_target_windows=3,
                                      min_target_clusters=2, min_guard_windows=3, bootstrap_B=100))
    base = tmp_path / "base"
    base.mkdir()
    calls = []
    monkeypatch.setattr("cascade.validator.pool.window_source_from_dir", lambda *a, **kw: object())
    monkeypatch.setattr("cascade.validator.expert_lane.affinity_contexts", lambda *a, **kw: [np.ones(64)])
    monkeypatch.setattr("cascade.validator.expert_lane.verdict_windows", lambda *a, **kw: [object()] * 20)
    monkeypatch.setattr("cascade.trainer.expert_lane.probe_target_routing", lambda *a:
                        SimpleNamespace(select=lambda cfg: {0: [1, 3]}))

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def series(self): return iter([np.ones(64)])

    monkeypatch.setattr("cascade.trainer.stream.open_round_stream", lambda *a, **kw: Stream())

    def train(self, stream, contract, **kw):
        calls.append(kw)
        return TrainResult(local_dir=kw["out_dir"], param_count=1, train_seconds=1., metrics={})

    monkeypatch.setattr("cascade.trainer.toto2_trainer.Toto2Trainer.train", train)

    def scores(path, windows, **kw):
        loss = 1. if path == base else .8
        return [WindowScore(series_id=f"s{i}", source=f"feed{i}", domain="energy", mase=loss,
                            qloss_per_q=np.full(9, loss), abs_target=2.) for i in range(20)]

    monkeypatch.setattr("cascade.validator.evaluator.evaluate_checkpoint", scores)
    args = SimpleNamespace(repo_dir=EXAMPLE, base=str(base), at_block=None, targets="energy",
                           pool_dir=tmp_path, work_dir=tmp_path / "work", device="cpu", seed=7,
                           train_hours=.00001, public_project=None)
    result = experiment(args, cfg)
    assert calls[0]["expert_selection"] == {0: [1, 3]}
    assert calls[0]["warm_start_dir"] == base
    assert result["qualification"]["verdicts"]["energy"]["passed"]
    assert result["accepted_targets"] == ["energy"]
    assert result["directional_only"] and not result["live_reward_guaranteed"]


def test_lane_c_local_test_trains_and_scores_forward_data(cfg, tmp_path, monkeypatch):
    import hashlib

    import torch

    from cascade.miner.lane_tests import test_lane_c as experiment
    from cascade.pool.marketcast import write_snapshot
    from tests.unit.test_marketcast_trainer import MC, T0, _base_dir, _recipe, _Toy

    model = _Toy()
    base = _base_dir(tmp_path, model)
    digest = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
    mc = replace(MC, base_sha256=digest, eval_context_length=128, min_dollar_volume_24h=0.)
    cfg = replace(cfg, marketcast=mc)
    entry = tmp_path / "recipe"
    entry.mkdir()
    recipe = _recipe()
    recipe["optimizer"]["steps"] = 2
    (entry / "marketcast.json").write_text(json.dumps(recipe))
    rng = np.random.default_rng(4)
    for name, t0 in (("train", T0), ("eval", T0 + 86400 * 4)):
        bars = {}
        for asset in ("BTC", "ETH"):
            close = np.exp(np.cumsum(rng.normal(0, .001, 2500))) * 100
            bars[("crypto", asset)] = {"open": close, "close": close, "high": close * 1.01,
                                        "low": close * .99, "volume": np.ones(2500) * 1e6}
        write_snapshot(tmp_path / name, t0=t0, minutes=2500, bars=bars,
                       sources={"crypto": "local"}, anchor_block=0, anchor_time=float(t0))

    class Forecaster:
        def __init__(self):
            self.inner = SimpleNamespace(model=model)

        def forecast_quantiles(self, context, steps, **kw):
            with torch.no_grad():
                return model(np.asarray(context)[None], steps)[0].numpy()

    monkeypatch.setattr("cascade.eval.timestral_adapter.TimestralQuantileForecaster.from_dir", lambda *a, **kw: Forecaster())
    monkeypatch.setattr("cascade.eval.timestral_adapter.forward_block",
                        lambda inner, contexts, steps, **kw: inner.model(contexts, steps))
    args = SimpleNamespace(repo_dir=entry, target="mkt:returns", train_snapshot=tmp_path / "train",
                           eval_snapshot=tmp_path / "eval", base_dir=base, device="cpu", seed=4,
                           train_hours=.01, work_dir=tmp_path / "work")
    result = experiment(args, cfg)
    assert result["training"]["steps"] == 2
    assert result["windows"] > 0 and result["confirmation"] == "not_tested"
    assert result["train_digest"] != result["eval_digest"]
    assert not result["live_reward_guaranteed"]
    args.eval_snapshot = args.train_snapshot
    with pytest.raises(ValueError, match="held-out snapshot"):
        experiment(args, cfg)
