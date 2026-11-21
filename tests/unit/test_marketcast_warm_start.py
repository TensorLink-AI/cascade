"""Warm-starting a MarketCast leg from an earlier checkpoint (owner
2026-11-18): the recipe names ``init``, the trainer accepts only a
checkpoint IT produced from a leg dispatched before the submission (so a
chain of warm-starts never reads the embargo gap and nothing trained
off-platform gets in), and the pod loads exactly those weights."""

from __future__ import annotations

import json

import pytest

from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.manifest import format_trained_pointer
from cascade.shared.marketcast_recipe import parse_marketcast_recipe
from cascade.trainer.finetune import RecipeError
from cascade.trainer.marketcast_runner import (
    MktLegRecipe,
    load_mkt_lineage,
    record_mkt_lineage,
    resolve_mkt_init,
)

PTR = format_trained_pointer("cascade/ckpt-mkt@sha256:" + "a" * 64)
MC = parse_marketcast({"suites": ["returns"], "frequencies": {"returns": ["1h"]},
                       "horizons": {"returns": ["1h"]}, "embargo_days": 1})


def _raw(**over):
    r = {"target": "mkt:returns",
         "data": {"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
                  "frequencies": ["1h"], "start": "2026-09-01", "end": "2026-09-03"},
         "context_length": 64, "horizon_mix": {"1h": 1.0}, "covariates": [],
         "augmentations": [], "objective": {"suite_weights": {"returns": 1.0}},
         "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 2,
                       "batch": 2, "weight_decay": 0.0, "clip": 1.0},
         "trainable": {"scope": "full"}, "synthetic_fraction": 0.0}
    r.update(over)
    return r


def test_recipe_init_is_optional_and_a_trained_pointer():
    assert parse_marketcast_recipe(_raw(), MC).init == ""
    assert parse_marketcast_recipe(_raw(init=PTR), MC).init == PTR
    assert parse_marketcast_recipe(_raw(leg_hours=2.0), MC).leg_hours == 2.0   # still optional
    with pytest.raises(ValueError, match="trained pointer"):
        parse_marketcast_recipe(_raw(init="hf://someone/else"), MC)


def test_only_an_earlier_checkpoint_this_trainer_produced(tmp_path):
    assert load_mkt_lineage(tmp_path) == {}
    record_mkt_lineage(tmp_path, PTR, train_time=100.0, checkpoint_sha256="c" * 64)
    lin = load_mkt_lineage(tmp_path)
    assert lin[PTR]["checkpoint_sha256"] == "c" * 64
    assert resolve_mkt_init(PTR, lin, submission_time=200.0) == (PTR, "c" * 64)
    assert resolve_mkt_init("", lin, submission_time=200.0) == ("", "")
    with pytest.raises(RecipeError, match="this trainer produced"):
        resolve_mkt_init(format_trained_pointer("x/y@sha256:" + "b" * 64), lin,
                         submission_time=200.0)
    with pytest.raises(RecipeError, match="after this submission"):
        resolve_mkt_init(PTR, lin, submission_time=100.0)


def test_the_leg_wrap_carries_the_resolved_init():
    r = MktLegRecipe("mkt:returns", json.dumps(_raw(init=PTR)), False, train_time=5.0,
                     init_pointer=PTR, init_sha256="c" * 64)
    w = json.loads(r.to_json())
    assert (w["init_pointer"], w["init_sha256"]) == (PTR, "c" * 64)
    assert "init_pointer" not in json.loads(MktLegRecipe("mkt:returns", "{}", False).to_json())


def test_the_pod_loads_exactly_the_resolved_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from cascade.eval.timestral_adapter import WEIGHTS_FILE, sha256_file
    from cascade.trainer.marketcast_leg import load_init_weights

    base = torch.nn.Linear(3, 2)
    warm = torch.nn.Linear(3, 2)
    (tmp_path / "base").mkdir()
    (tmp_path / "init").mkdir()
    save_file({k: v.contiguous() for k, v in base.state_dict().items()},
              str(tmp_path / "base" / WEIGHTS_FILE))
    save_file({k: v.contiguous() for k, v in warm.state_dict().items()},
              str(tmp_path / "init" / WEIGHTS_FILE))
    sha = sha256_file(tmp_path / "init" / WEIGHTS_FILE)
    with pytest.raises(RuntimeError, match="resolved"):
        load_init_weights(base, tmp_path / "init", base_dir=tmp_path / "base", sha256="0" * 64)
    load_init_weights(base, tmp_path / "init", base_dir=tmp_path / "base", sha256=sha)
    assert torch.equal(base.weight, warm.weight) and torch.equal(base.bias, warm.bias)
    save_file({"other": torch.zeros(1)}, str(tmp_path / "init" / WEIGHTS_FILE))
    with pytest.raises(RuntimeError, match="architecture"):
        load_init_weights(base, tmp_path / "init", base_dir=tmp_path / "base",
                          sha256=sha256_file(tmp_path / "init" / WEIGHTS_FILE))


def test_the_audit_re_derives_the_lineage_from_signed_manifests(monkeypatch):
    from types import SimpleNamespace

    import cascade.shared.lane_c_manifest as LM
    from cascade.audit.checks import _mkt_init_lineage

    def _m(ptr, t, target="mkt:market"):
        return SimpleNamespace(entries=[SimpleNamespace(trained_pointer=ptr, target=target,
                                                        marketcast={"train_time": t})])

    texts = {"r1": "a", "r2": "b"}
    parsed = {"a": _m(PTR, 100.0), "b": _m("other", 50.0)}
    monkeypatch.setattr(LM, "load_lane_c_manifest", lambda t: parsed[t])
    monkeypatch.setattr(LM, "verify_lane_c_manifest", lambda m, hk: True)
    cfg = SimpleNamespace(manifest=SimpleNamespace(trainer_hotkey="trainer"))
    assert _mkt_init_lineage(PTR, 200.0, cfg, texts, {}) == ""
    assert "not before" in _mkt_init_lineage(PTR, 100.0, cfg, texts, {})
    assert "no trainer-signed" in _mkt_init_lineage("ghost", 200.0, cfg, texts, {})
    assert _mkt_init_lineage(PTR, 200.0, cfg, {}, {}) is None              # nothing to settle it
    monkeypatch.setattr(LM, "verify_lane_c_manifest", lambda m, hk: False)
    assert _mkt_init_lineage(PTR, 200.0, cfg, texts, {}) is None           # unsigned ⇒ unusable


def test_concurrent_legs_never_drop_a_lineage_entry(tmp_path):
    import threading

    ptrs = [format_trained_pointer(f"cascade/ckpt-{i}@sha256:" + f"{i:064x}") for i in range(24)]
    ts = [threading.Thread(target=record_mkt_lineage, args=(tmp_path, p),
                           kwargs={"train_time": float(i), "checkpoint_sha256": "c" * 64})
          for i, p in enumerate(ptrs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert set(load_mkt_lineage(tmp_path)) == set(ptrs)
    (tmp_path / "marketcast_lineage.json").write_text("{torn", encoding="utf-8")
    assert len(load_mkt_lineage(tmp_path)) == 23        # the .bak (one write behind) restores it
