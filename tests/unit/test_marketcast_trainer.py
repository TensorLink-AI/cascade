"""MarketCast (DEC-CA-0064) trainer half + the Timestral wrapper's fine-tune
hook: augmentations, the sampler, LoRA / trainable scopes on a
Timestral-shaped toy model, the fine-tune loop, export + the harvest check,
the worker role and the lane-C driver's MarketCast legs. The real
``timestral`` library is not installed in CI: its calls are stubbed, and the
one test that needs it skips."""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval import timestral_adapter as TA
from cascade.pool import marketcast as P
from cascade.shared.config.marketcast import TIMESTRAL_KNOTS, parse_marketcast
from cascade.shared.marketcast_recipe import parse_marketcast_recipe
from cascade.trainer import marketcast_leg as L

torch = pytest.importorskip("torch")
nn = torch.nn

T0 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
K = len(TIMESTRAL_KNOTS)


# ── a Timestral-shaped toy model (the module paths select_trainable uses) ────


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wQKV = nn.Linear(d, 3 * d, bias=False)
        self.wO = nn.Linear(3 * d, d, bias=False)

    def forward(self, x):
        return self.wO(self.wQKV(x))


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attns = nn.ModuleList([_Attn(d)])
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d, bias=False), nn.SiLU(),
                                 nn.Linear(2 * d, d, bias=False))

    def forward(self, x):
        return x + self.attns[0](x) + self.mlp(x)


class _Toy(nn.Module):
    def __init__(self, d=8, n_layers=3):
        super().__init__()
        self.representation = nn.Module()
        self.representation.patch_proj = nn.Linear(8, d)
        self.representation.forecast_query = nn.Parameter(torch.zeros(d))
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([_Layer(d) for _ in range(n_layers)])
        self.transformer.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K)

    def forward(self, ctx, steps):
        x = torch.nan_to_num(torch.as_tensor(ctx, dtype=torch.float32)[:, -8:])
        h = self.representation.patch_proj(x) + self.representation.forecast_query
        for layer in self.transformer.layers:
            h = layer(h)
        q = self.head(self.transformer.out_norm(h))
        q = torch.cumsum(torch.nn.functional.softplus(q), -1) - 10.0
        return q[:, None, :].expand(-1, int(steps), -1)


def _fwd(model):
    return lambda ctx, steps, past, known: model(ctx, steps)


def _base_dir(tmp_path, model) -> Path:
    from safetensors.torch import save_file

    d = tmp_path / "base"
    d.mkdir()
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
              str(d / TA.WEIGHTS_FILE))
    (d / TA.CONFIG_FILE).write_text(json.dumps({"patch_size": 32}), encoding="utf-8")
    return d


# ── the fine-tune hook ───────────────────────────────────────────────────────


def test_trainable_scopes():
    m = _Toy()
    n_all = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in TA.select_trainable(m, "full")) == n_all
    fe = TA.select_trainable(_Toy(), "frozen_embeddings")
    assert sum(p.numel() for p in fe) == n_all - 8 * 8 - 8 - 8
    m = _Toy()
    last = TA.select_trainable(m, "last_n_blocks", n_blocks=1)
    want = (sum(p.numel() for p in m.transformer.layers[-1].parameters())
            + sum(p.numel() for p in m.transformer.out_norm.parameters())
            + sum(p.numel() for p in m.head.parameters()))
    assert sum(p.numel() for p in last) == want
    with pytest.raises(ValueError):
        TA.select_trainable(_Toy(), "last_n_blocks", n_blocks=4)
    with pytest.raises(ValueError):
        TA.select_trainable(_Toy(), "adapter")


def test_lora_starts_at_the_base_and_merges_back():
    torch.manual_seed(0)
    m = _Toy()
    keys = set(m.state_dict())
    x = np.random.default_rng(0).normal(size=(4, 16))
    before = m(x, 2).detach().clone()
    params = TA.select_trainable(m, "lora", lora_rank=2)
    assert params and all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    torch.testing.assert_close(m(x, 2), before)            # B = 0: exactly the base
    with torch.no_grad():
        for p in params:
            p.add_(0.1)
    after = m(x, 2).detach().clone()
    assert not torch.allclose(after, before)
    assert TA.merge_lora(m) == 3 * 4                      # wQKV, wO + 2 MLP linears per layer
    assert set(m.state_dict()) == keys                    # the base's exact key set
    torch.testing.assert_close(m(x, 2), after, rtol=1e-5, atol=1e-5)


def test_export_checkpoint_and_harvest_check(tmp_path):
    torch.manual_seed(1)
    base = _Toy()
    bdir = _base_dir(tmp_path, base)
    ft = _Toy()
    ft.load_state_dict(base.state_dict())
    TA.select_trainable(ft, "lora", lora_rank=2)
    mc = parse_marketcast({})
    meta = {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256, "target": "mkt:returns",
            "covariates": ["funding", "calendar"], "data_digest": "f" * 64}
    sha = TA.export_checkpoint(ft, tmp_path / "ft", base_dir=bdir, meta=meta)
    assert sha == hashlib.sha256((tmp_path / "ft" / TA.WEIGHTS_FILE).read_bytes()).hexdigest()
    assert TA.same_structure(bdir / TA.WEIGHTS_FILE, tmp_path / "ft" / TA.WEIGHTS_FILE) == []
    assert TA.safetensors_all_finite(tmp_path / "ft" / TA.WEIGHTS_FILE)
    assert TA.read_meta(tmp_path / "ft") == meta
    from cascade.trainer.marketcast_runner import verify_harvested_mkt_checkpoint

    runner = SimpleNamespace(cfg=SimpleNamespace(marketcast=mc))
    rec = SimpleNamespace(marketcast={"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
                                      "checkpoint_sha256": sha},
                          tensor_digests={TA.WEIGHTS_FILE: sha})
    kw = dict(base_weights=bdir / TA.WEIGHTS_FILE, base_config={"patch_size": 32})
    assert verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", rec, **kw) == ""
    bad = SimpleNamespace(marketcast={**rec.marketcast, "checkpoint_sha256": "0" * 64},
                          tensor_digests=rec.tensor_digests)
    assert "sha256" in verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", bad, **kw)
    other = SimpleNamespace(marketcast={**rec.marketcast, "base_sha256": "1" * 64},
                            tensor_digests=rec.tensor_digests)
    assert "pinned" in verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", other, **kw)
    assert "config" in verify_harvested_mkt_checkpoint(
        runner, tmp_path / "ft", rec, base_weights=bdir / TA.WEIGHTS_FILE,
        base_config={"patch_size": 16})
    # review 2026-10-19: marketcast_meta.json is validated at harvest (a bad
    # covariate name / a missing field / a foreign base is the miner's fault)
    meta_p = tmp_path / "ft" / TA.META_FILE
    for bad_meta, why in (({**meta, "covariates": ["open_interest"]}, "not allowed"),
                          ({**meta, "covariates": ["funding", "funding"]}, "repeat"),
                          ({k: v for k, v in meta.items() if k != "data_digest"}, "lacks"),
                          ({**meta, "base_sha256": "0" * 64}, "pinned")):
        meta_p.write_text(json.dumps(bad_meta))
        assert why in verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", rec, **kw)
    meta_p.write_text("x" * 70_000)
    assert "unreadable" in verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", rec, **kw)
    meta_p.write_text(json.dumps({**meta, "pad": "x" * 70_000}))
    assert "bytes" in verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", rec, **kw)
    meta_p.write_text(json.dumps(meta))
    assert verify_harvested_mkt_checkpoint(runner, tmp_path / "ft", rec, **kw) == ""
    # a different architecture
    from safetensors.torch import save_file

    small = _Toy(n_layers=2)
    save_file({k: v.contiguous() for k, v in small.state_dict().items()},
              str(tmp_path / "small.safetensors"))
    assert TA.same_structure(bdir / TA.WEIGHTS_FILE, tmp_path / "small.safetensors")
    # non-finite weights
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    sd["head.bias"][0] = float("nan")
    save_file(sd, str(tmp_path / "nan.safetensors"))
    assert not TA.safetensors_all_finite(tmp_path / "nan.safetensors")


def test_wrapper_over_a_stub_library_forecaster():
    class _Fc:
        quantile_levels = TIMESTRAL_KNOTS

        def __init__(self):
            self.calls = []

        def forecast(self, ctx, h, **kw):
            self.calls.append((np.asarray(ctx).shape, h, kw))
            q = np.tile(np.asarray(TIMESTRAL_KNOTS), (ctx.shape[0], 1, h, 1))
            return SimpleNamespace(quantiles=torch.as_tensor(q))

    inner = _Fc()
    w = TA.TimestralQuantileForecaster(inner)
    assert isinstance(w, TA.QuantileForecaster)
    q = w.forecast_quantiles(np.arange(10.0), 3)
    assert q.shape == (3, K)
    shape, h, kw = inner.calls[0]
    assert shape == (1, 10) and h == 3
    assert kw["symmetric_averaging"] and kw["clip_nonnegative"]
    # batched, ragged contexts are NaN-left-padded; output mapped to levels
    out = TA.batched_forecasts(w, [np.arange(5.0), np.arange(8.0)], 2, levels=(0.1, 0.5, 0.9))
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0], [0.1, 0.5, 0.9])
    assert inner.calls[-1][0] == (2, 8)


def test_real_library_wrapper_if_installed():
    pytest.importorskip("timestral")
    lib = TA.load_timestral()
    assert hasattr(lib, "TimestralForecaster")


@pytest.mark.parametrize("horizon", [4, 11, 12])
@pytest.mark.parametrize("covariates", [False, True])
def test_training_decode_matches_real_inference_and_has_gradients(horizon, covariates):
    pytest.importorskip("timestral")
    from timestral.config import TimestralConfig
    from timestral.forecaster import TimestralForecaster
    from timestral.model import TimestralModel

    torch.manual_seed(0)
    config = TimestralConfig(patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
        block_schedule=("time", "time", "variate"), quantile_knots=(0.1, 0.5, 0.9),
        block_size=4, context_length=64)
    fc = TimestralForecaster(TimestralModel(config))
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(2, 64)).cumsum(-1)
    contexts[0, :7] = np.nan
    kw = {}
    if covariates:
        past = rng.normal(size=(2, 1, 64))
        past[0, :1, :10] = np.nan
        known = rng.normal(size=(2, 1, 64 + horizon))
        kw = {"past_covariates": past, "known_covariates": known}
    trained = TA.forward_block(fc, contexts, horizon, **kw)
    evaluated = fc.forecast(contexts, horizon, symmetric_averaging=False,
                            clip_nonnegative=False, **kw).quantiles[:, 0]
    torch.testing.assert_close(trained, evaluated)
    trained.square().mean().backward()
    grads = [p.grad for p in fc.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


# ── augmentations / sampler / loop ───────────────────────────────────────────


def test_augmentations_are_seeded_and_bounded():
    rng = np.random.default_rng(0)
    c, y = np.linspace(1, 2, 200), np.array([2.0, 2.1])
    augs = (("scale", 1.0, ()), ("jitter", 1.0, (("sigma", 0.05),)),
            ("crop", 1.0, (("min_frac", 0.5),)), ("time_warp", 1.0, ()),
            ("mixup", 1.0, ()))
    a1 = L.augment(c, y, augs, np.random.default_rng(3), (c * 2, y * 2))
    a2 = L.augment(c, y, augs, np.random.default_rng(3), (c * 2, y * 2))
    np.testing.assert_array_equal(a1[0], a2[0])
    np.testing.assert_array_equal(a1[1], a2[1])
    assert L.MIN_CONTEXT <= a1[0].size <= c.size
    same = L.augment(c, y, (("scale", 0.0, ()),), rng)
    np.testing.assert_array_equal(same[0], c)
    with pytest.raises(ValueError):
        L.augment(c, y, (("flip", 1.0, ()),), rng)


def test_lr_schedules():
    kw = dict(total=100, lr=1.0, warmup_fraction=0.1)
    assert L.lr_at(0, schedule="cosine", **kw) == pytest.approx(0.1)
    assert L.lr_at(10, schedule="cosine", **kw) == pytest.approx(1.0)
    assert L.lr_at(99, schedule="cosine", **kw) < 0.01
    assert L.lr_at(55, schedule="linear", **kw) == pytest.approx(0.5)
    assert L.lr_at(80, schedule="constant", **kw) == 1.0


MC = parse_marketcast({
    "suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"]},
    "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
    "horizons": {"returns": ["1m", "5m"], "vol": ["5m"]}, "max_steps": 1000,
})


def _recipe(**over):
    r = {"target": "mkt:returns",
         "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                  "frequencies": ["1m", "5m"], "start": "2025-01-01", "end": "2025-01-02"},
         "context_length": 128, "horizon_mix": {"1m": 1.0, "5m": 1.0},
         "covariates": ["volume", "calendar"],
         "augmentations": [{"name": "jitter", "p": 0.5, "params": {"sigma": 0.01}}],
         "objective": {"suite_weights": {"returns": 1.0, "vol": 0.5}, "coverage_penalty": 0.1},
         "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 6,
                       "batch": 4, "weight_decay": 0.0, "clip": 1.0, "ema_decay": 0.9},
         "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}
    r.update(over)
    return r


def _train_snap(tmp_path, minutes=3000):
    rng = np.random.default_rng(5)
    bars = {}
    for a in ("BTC", "ETH"):
        c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
        bars[("crypto", a)] = {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
                               "volume": rng.uniform(1, 2, minutes)}
    P.write_snapshot(tmp_path / "train", t0=T0, minutes=minutes, bars=bars,
                     sources={"crypto": "hyperliquid"}, anchor_block=0, anchor_time=float(T0))
    return P.load_snapshot(tmp_path / "train", verify_signature=False)


def test_sampler_batches(tmp_path):
    snap = _train_snap(tmp_path)
    rc = parse_marketcast_recipe(_recipe(), MC)
    s = L.MktSampler(rc, MC, snap, seed=7)
    b = s.batch(4)
    assert b.contexts.shape[0] == 4 and b.targets.shape == (4, b.steps)
    assert b.past.shape == (4, 1, b.contexts.shape[1])
    assert b.known.shape == (4, 4, b.contexts.shape[1] + b.steps)
    s2 = L.MktSampler(rc, MC, snap, seed=7)
    np.testing.assert_array_equal(np.nan_to_num(s2.batch(4).contexts), np.nan_to_num(b.contexts))
    # synthetic mix
    gen = iter([np.arange(500.0)[None] for _ in range(1000)])
    rs = parse_marketcast_recipe(_recipe(synthetic_fraction=0.5), MC, has_generator=True)
    bs = L.MktSampler(rs, MC, snap, seed=1, synthetic=gen).batch(16)
    assert 0 < bs.synthetic < 16
    with pytest.raises(L.MktLegError):
        L.MktSampler(rs, MC, snap, seed=1)                        # no generator series
    with pytest.raises(L.MktLegError):
        L.MktSampler(rc, MC, _train_snap(tmp_path / "x", minutes=40), seed=1)


def test_finetune_loop_trains_only_the_adapters(tmp_path):
    torch.manual_seed(0)
    snap = _train_snap(tmp_path)
    rc = parse_marketcast_recipe(_recipe(), MC)
    m = _Toy()
    base_head = m.head.weight.detach().clone()
    t = iter(range(10**6))
    res = L.finetune(m, _fwd(m), L.MktSampler(rc, MC, snap, seed=1), rc,
                     levels=TIMESTRAL_KNOTS, max_seconds=1e9, clock=lambda: float(next(t)))
    assert res.steps == 6 and np.isfinite(res.final_loss) and res.real_samples == 24
    torch.testing.assert_close(m.head.weight, base_head)        # frozen under lora
    # the wall budget stops the loop
    m2 = _Toy()
    t2 = iter(range(0, 10**6, 100))
    r2 = L.finetune(m2, _fwd(m2), L.MktSampler(rc, MC, snap, seed=1), rc,
                    levels=TIMESTRAL_KNOTS, max_seconds=250, clock=lambda: float(next(t2)))
    assert r2.steps < 6


def test_quantile_loss_masks_nan_and_weights_knots():
    pred = torch.zeros((1, 2, 3))
    y = np.array([[1.0, np.nan]])
    ctx = np.array([[0.0, 1.0, 2.0]])
    lv = (0.1, 0.5, 0.9)
    loss = L.quantile_loss(pred, y, ctx, lv, (1.0, 1.0, 1.0))
    assert float(loss) == pytest.approx((0.1 + 0.5 + 0.9) / 3)
    only_med = L.quantile_loss(pred, y, ctx, lv, (0.0, 1.0, 0.0))
    assert float(only_med) == pytest.approx(0.5)


def test_run_mkt_leg_and_worker(tmp_path, monkeypatch):
    torch.manual_seed(2)
    base = _Toy()
    bdir = _base_dir(tmp_path, base)
    snap = _train_snap(tmp_path)
    rc = parse_marketcast_recipe(_recipe(), MC)
    inner = SimpleNamespace(model=base)
    monkeypatch.setattr(TA, "forward_block",
                        lambda fc, ctx, steps, **kw: fc.model(ctx, steps))
    out = L.run_mkt_leg(recipe=rc, mc=MC, base_dir=bdir, base_fc_inner=inner, train_snap=snap,
                        out_dir=tmp_path / "out", seed=3, max_seconds=1e9)
    assert out["steps"] == 6 and out["data_digest"] == snap.digest
    assert out["checkpoint_sha256"] == TA.sha256_file(tmp_path / "out" / TA.WEIGHTS_FILE)
    meta = TA.read_meta(tmp_path / "out")
    assert meta["covariates"] == ["volume", "calendar"] and meta["base_sha256"] == MC.base_sha256

    # the worker role (stubbed feeds + base); MarketCast armed in its config
    from dataclasses import replace

    from cascade.shared.config import load_chain_config
    from cascade.trainer import worker as W

    cfg = load_chain_config("chain.toml")
    cfg = replace(cfg, marketcast=replace(MC, from_block=7200))
    T1 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())

    def post(url, body):
        if body["type"] == "fundingHistory":
            return []
        s, e = body["req"]["startTime"], body["req"]["endTime"]
        rng = np.random.default_rng(abs(hash(body["req"]["coin"])) % 1000)
        n = min(3000, (e - s) // 60000 + 1)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
        return [{"t": s + 60000 * i, "o": str(x), "h": str(x * 1.001), "l": str(x * 0.999),
                 "c": str(x), "v": "1.5"} for i, x in enumerate(c) if s + 60000 * i < (T1 + 3000 * 60) * 1000]

    args = SimpleNamespace(local_only=True, warm_start_ref=MC.base_ref,
                           recipe_json=json.dumps({"marketcast": _recipe(),
                                                   "has_generator": False}),
                           work_root=tmp_path / "w", hotkey="5Miner", uid=4, block=123,
                           base_seed=9, gen_ref="m/r@sha256:" + "a" * 64)
    log = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None,
                          exception=lambda *a, **k: None)
    rc_code = W.run_mkt_worker(args, cfg, log, post=post, get=lambda u: [],
                               base_loader=lambda: (SimpleNamespace(inner=inner), bdir))
    assert rc_code == 0
    # rc 3: the miner's recipe
    bad = SimpleNamespace(**{**args.__dict__, "recipe_json": json.dumps(
        {"marketcast": _recipe(target="mkt:range")})})
    assert W.run_mkt_worker(bad, cfg, log, post=post, get=lambda u: [],
                            base_loader=lambda: (inner, bdir)) == 3
    # rc 2: not armed / wrong base / not local-only
    assert W.run_mkt_worker(args, replace(cfg, marketcast=MC), log) == 2
    assert W.run_mkt_worker(SimpleNamespace(**{**args.__dict__, "warm_start_ref": "x/y"}),
                            cfg, log) == 2
    assert W.run_mkt_worker(SimpleNamespace(**{**args.__dict__, "local_only": False}),
                            cfg, log) == 2
    # synthetic leg: the generator series come from the injected (sandboxed) stream
    syn = SimpleNamespace(**{**args.__dict__, "recipe_json": json.dumps(
        {"marketcast": _recipe(synthetic_fraction=0.3), "has_generator": True}),
        "work_root": tmp_path / "w2"})

    @contextlib.contextmanager
    def _stream():
        yield iter([np.arange(400.0)[None] for _ in range(10**4)])

    assert W.run_mkt_worker(syn, cfg, log, post=post, get=lambda u: [], synthetic_fn=_stream,
                            base_loader=lambda: (inner, bdir)) == 0


@pytest.mark.parametrize("outage", [False, True])
def test_worker_distinguishes_feed_outage_from_empty_training_range(tmp_path, outage):
    import logging
    from dataclasses import replace

    from cascade.shared.config import load_chain_config
    from cascade.trainer.faults import classify_funded_worker_failure
    from cascade.trainer.worker import run_mkt_worker

    cfg = replace(load_chain_config("chain.toml"), marketcast=replace(MC, from_block=7200))
    args = SimpleNamespace(local_only=True, warm_start_ref=MC.base_ref,
        recipe_json=json.dumps({"marketcast": _recipe(), "has_generator": False}),
        work_root=tmp_path, hotkey="miner", uid=1, block=123, base_seed=9, gen_ref="g")

    def feed(*args, **kw):
        if outage:
            raise TimeoutError("market data unavailable")
        return []

    rc = run_mkt_worker(args, cfg, logging.getLogger(__name__), post=feed, get=feed,
                        base_loader=lambda: (SimpleNamespace(model=None), tmp_path))
    assert rc == (1 if outage else 3)
    miner_fault, _, _ = classify_funded_worker_failure(rc, "", stalled_before=False)
    assert miner_fault is not outage


def test_coarse_training_feed_failure_is_infrastructure(tmp_path, monkeypatch):
    recipe = parse_marketcast_recipe(_recipe(), MC)
    monkeypatch.setattr(P, "needs_coarse", lambda mc: True)
    monkeypatch.setattr(P, "fetch_class_bars", lambda *a, **kw: ({"close": np.ones(128)}, {}))

    def outage(*args, **kw):
        raise TimeoutError("coarse feed timed out")

    monkeypatch.setattr(P, "fetch_coarse_bars", outage)
    with pytest.raises(L.MktDataUnavailable, match="coarse feed unavailable"):
        L.build_training_snapshot(tmp_path, MC, recipe, post=None, get=None)


def test_worker_argv_and_receipt_role():
    from cascade.trainer.remote import receipt_role, receipt_to_local, worker_argv

    host = SimpleNamespace(remote_python="python", chain_toml="")
    argv = worker_argv(host, gen_ref="g", uid=1, hotkey="h", role="mkt_finetune", base_seed=1,
                       block=2, trainer_spec="m:C", warm_start_ref="r@" + "a" * 40,
                       local_only=True, finetune_recipe='{"marketcast": {}}')
    assert argv[argv.index("--role") + 1] == "mkt_finetune" and "--recipe-json" in argv
    with pytest.raises(ValueError):
        worker_argv(host, gen_ref="g", uid=1, hotkey="h", role="mkt_finetune", base_seed=1,
                    block=2, trainer_spec="m:C", warm_start_ref="r", finetune_recipe="{}")
    assert receipt_role("mkt_finetune") == "challenger"
    loc = receipt_to_local({"miner_hotkey": "h", "miner_uid": 1, "role": "challenger",
                            "gen_ref": "g", "corpus_digest": "d", "train_block": 3,
                            "local_checkpoint_dir": "/x", "marketcast": {"base_ref": "r"}})
    assert loc.marketcast == {"base_ref": "r"}
    assert receipt_to_local({"miner_hotkey": "h", "miner_uid": 1, "role": "challenger",
                             "gen_ref": "g", "corpus_digest": "d", "train_block": 3,
                             "local_checkpoint_dir": "/x"}).marketcast == {}


def test_read_mkt_recipe(tmp_path):
    from dataclasses import replace

    from cascade.shared.config import load_chain_config
    from cascade.trainer.finetune import RecipeError
    from cascade.trainer.marketcast_runner import read_mkt_recipe

    cfg = replace(load_chain_config("chain.toml"), marketcast=MC)
    d = tmp_path / "tree"
    d.mkdir()
    with pytest.raises(RecipeError, match="marketcast.json"):
        read_mkt_recipe(d, cfg, target="mkt:returns")
    (d / "marketcast.json").write_text(json.dumps(_recipe()), encoding="utf-8")
    r = read_mkt_recipe(d, cfg, target="mkt:returns")
    assert not r.has_generator and json.loads(r.to_json())["has_generator"] is False
    (d / "marketcast.json").write_text(json.dumps(_recipe(synthetic_fraction=0.2)))
    with pytest.raises(RecipeError, match="generator"):
        read_mkt_recipe(d, cfg, target="mkt:returns")
    (d / "generator.py").write_text("def generate(seed): yield []\n")
    assert read_mkt_recipe(d, cfg, target="mkt:returns").has_generator
    with pytest.raises(RecipeError, match="committed target"):
        read_mkt_recipe(d, cfg, target="mkt:vol")


# ── the lane-C driver's MarketCast legs ─────────────────────────────────────


def test_driver_dispatches_mkt_legs_only_while_active_and_never_expires_them(tmp_path):
    from dataclasses import replace

    from cascade.funding.queue import LaneCQueue
    from cascade.shared.chain import Commitment
    from cascade.shared.config.lane_c import LaneCConfig
    from cascade.shared.lane_c_manifest import format_finetune_commit, load_lane_c_manifest
    from cascade.shared.manifest import format_trained_pointer
    from cascade.trainer.lane_c_driver import LaneCDriver
    from cascade.trainer.marketcast_runner import MktLegRecipe

    FROM = 9000
    king = format_trained_pointer("cascade/ckpt@sha256:" + "a" * 64)
    king2 = format_trained_pointer("cascade/ckpt@sha256:" + "b" * 64)
    ft = format_trained_pointer("cascade/ckpt@sha256:" + "c" * 64)
    ref = "alfa/recipe@sha256:" + "1" * 64
    mc = replace(MC, from_block=FROM + 7200)
    cfg = SimpleNamespace(lane_c=LaneCConfig(from_block=FROM, targets=("mkt:returns",),
                                             legs_per_target=1, max_base_age_eras=1),
                          marketcast=mc)
    clock = lambda: 1_000_000.0  # noqa: E731
    mkt = {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
           "checkpoint_sha256": "d" * 64, "recipe_digest": "e" * 64, "data_digest": "f" * 64}
    published = []

    class Ops:
        q = LaneCQueue(tmp_path / "q.json", clock=clock)
        commits = [Commitment(uid=1, hotkey="ALFA", coldkey=None, commit_block=100,
                              payload=format_finetune_commit("mkt:returns", ref,
                                                             mkt_suites=("returns", "vol")))]

        def commitments(self):
            return self.commits

        def queue(self):
            return self.q

        def lane_a_busy(self, hk):
            return False

        def fetch_recipe(self, r, target, **kw):
            # review M12: the embargo is checked against the commit time on the
            # dispatch clock; the pod is clipped to the dispatch time
            assert kw["train_time"] == 1_234_567.0
            assert kw["submission_time"] == 1_234_567.0 - (FROM + 7300 - 100) * 12
            return MktLegRecipe(target=target, payload="{}", has_generator=False,
                                train_time=kw["train_time"])

        def train_leg(self, leg, recipe):
            assert isinstance(recipe, MktLegRecipe)
            return {"trained_pointer": ft, "corpus_digest": "c" * 64,
                    "train_block": leg["started_block"], "marketcast": mkt}

        def bench_rows(self, p):
            raise AssertionError("no public guard on an mkt target")

        def sign(self, m):
            return replace(m, signature="ab" * 32)

        def publish(self, key, text):
            published.append((key, text))

        def spawn(self, fn, name):
            fn()

        def now(self):
            return 1_234_567.0

    ops = Ops()
    ops.q.add_lane_c("ALFA", "mkt:returns", ref, 100)
    drv = LaneCDriver(cfg, ops, tmp_path / "drv.json")
    # lane C live but MarketCast not yet: the entry waits (not failed)
    assert drv.tick(FROM + 10, 10, king, FROM) == []
    assert [e.hotkey for e in ops.q.queued()] == ["ALFA"]
    started = drv.tick(FROM + 7300, 11, king, FROM + 7200)
    assert [leg["target"] for leg in started] == ["mkt:returns"]
    # five eras later, on a NEW king: no expiry, no base_moved drop
    digest = drv.settle(round_id="r1", block=FROM + 5 * 7200, era_index=16, base_pointer=king2)
    assert digest
    m = load_lane_c_manifest(published[-1][1])
    # review 2026-10-19: the trainer stamps the dispatch time (the forward-only
    # clock) into the signed record, next to train_block
    assert [(e.target, e.marketcast) for e in m.entries] == [
        ("mkt:returns", {**mkt, "train_time": 1_234_567.0})]


def test_lane_c_entry_and_commit_grammar_for_mkt():
    from cascade.shared.lane_c_manifest import (
        LaneCEntry,
        LaneCManifest,
        dump_lane_c_manifest,
        format_finetune_commit,
        load_lane_c_manifest,
        parse_finetune_commit,
    )
    from cascade.shared.manifest import format_trained_pointer

    ref = "alfa/recipe@sha256:" + "1" * 64
    payload = f"metro-v1:ft:mkt:returns:hippius:{ref}"
    assert parse_finetune_commit(payload) is None                    # inert: refused
    assert parse_finetune_commit(payload, mkt_suites=("returns",)) == ("mkt:returns", ref)
    assert parse_finetune_commit(payload, mkt_suites=("vol",)) is None
    with pytest.raises(ValueError):
        format_finetune_commit("mkt:returns", ref)
    ptr = format_trained_pointer("cascade/ckpt@sha256:" + "c" * 64)
    e = LaneCEntry("hk", 1, "mkt:returns", ref, ptr, 5, marketcast={"base_sha256": "x"})
    plain = LaneCEntry("hk", 1, "mkt:vol", ref, ptr, 5)
    with pytest.raises(ValueError):
        LaneCEntry("hk", 1, "mkt:Bad Suite", ref, ptr, 5)
    with pytest.raises(ValueError, match="forward-only"):      # MarketCast-only lane C
        LaneCEntry("hk", 1, "domain:energy", ref, ptr, 5)
    m = LaneCManifest("r", 1, 2, ptr, "r", 0, (e, plain))
    back = load_lane_c_manifest(dump_lane_c_manifest(m))
    assert back.entries[0].marketcast == {"base_sha256": "x"}
    assert "marketcast" not in json.loads(m.canonical_body())["entries"][1]   # byte-identical


def test_driver_records_report_arbiters_and_requeues_infra_drops(tmp_path):
    """Review fixes 2026-10-19 (B1/B2): while MarketCast is active every
    lane-C manifest records each awaited settlement's report — the sha256 of
    the exact text the trainer read, or "absent" after report_timeout
    settlements — publishing a manifest even with no legs; an entry the main
    validator dropped past max_infra_deferrals is requeued uncharged. Inert:
    nothing recorded, the driver state is byte-identical."""
    from dataclasses import replace

    from cascade.funding.queue import LaneCQueue
    from cascade.shared.config.lane_c import LaneCConfig
    from cascade.shared.lane_c_manifest import load_lane_c_manifest
    from cascade.shared.manifest import format_trained_pointer
    from cascade.shared.marketcast_report import (
        INFRA_DEFERRED,
        MktEntryRows,
        MktReport,
        mkt_text_digest,
    )
    from cascade.trainer.lane_c_driver import LaneCDriver

    FROM = 9000
    king = format_trained_pointer("cascade/ckpt@sha256:" + "a" * 64)
    ft = format_trained_pointer("cascade/ckpt@sha256:" + "c" * 64)
    ref = "alfa/recipe@sha256:" + "1" * 64
    mc = replace(MC, from_block=FROM, report_timeout_settlements=2)
    cfg = SimpleNamespace(lane_c=LaneCConfig(from_block=FROM, targets=("mkt:returns",),
                                             legs_per_target=1), marketcast=mc)
    published, reports = [], {}

    class Ops:
        q = LaneCQueue(tmp_path / "q.json", clock=lambda: 1_000_000.0)

        def queue(self):
            return self.q

        def sign(self, m):
            return replace(m, signature="ab" * 32)

        def publish(self, key, text):
            published.append((key, text))

        def read_mkt_report(self, rid):
            return reports.get(rid)

    drv = LaneCDriver(cfg, Ops(), tmp_path / "drv.json")
    assert drv.settle(round_id="r1", block=FROM + 7200, era_index=1, base_pointer=king) == ""
    assert drv.st.mkt_await == [{"rid": "r1", "age": 0}] and not published
    rep = MktReport("r1", 1, {}, (MktEntryRows("ALFA", ft, "mkt:returns", "judge", 1.0),))
    reports["r1"] = rep.to_json()
    d2 = drv.settle(round_id="r2", block=FROM + 2 * 7200, era_index=1, base_pointer=king)
    m2 = load_lane_c_manifest(published[-1][1])
    assert d2 and m2.entries == () and m2.mkt_reports == {"r1": mkt_text_digest(reports["r1"])}
    assert [a["rid"] for a in drv.st.mkt_await] == ["r2"]
    drv.settle(round_id="r3", block=FROM + 3 * 7200, era_index=1, base_pointer=king)
    m4 = drv.settle(round_id="r4", block=FROM + 4 * 7200, era_index=1, base_pointer=king)
    assert m4 and load_lane_c_manifest(published[-1][1]).mkt_reports == {"r2": "absent"}
    # idempotent per round: the same bytes, nothing re-derived
    n = len(published)
    assert drv.settle(round_id="r4", block=FROM + 4 * 7200, era_index=1, base_pointer=king) == m4
    assert len(published) == n
    # an infra drop in a report requeues the done entry, uncharged
    Ops.q.add_lane_c("ALFA", "mkt:returns", ref, 100)
    Ops.q.mark_in_flight("ALFA", Ops.q.get("ALFA").ref, target_boundary=0, era_index=1,
                         started_block=1)
    Ops.q.mark_done("ALFA")
    drop = MktEntryRows("ALFA", ft, "mkt:returns", "judge", 1.0, status=INFRA_DEFERRED,
                        infra_reason="compute failed 2x: OOM — past max_infra_deferrals (9 > 8)")
    reports["r3"] = MktReport("r3", 1, {}, (drop,)).to_json()
    drv.settle(round_id="r5", block=FROM + 5 * 7200, era_index=1, base_pointer=king)
    e = Ops.q.get("ALFA")
    assert e.status == "queued" and e.attempts == 0 and e.last_error_class == "operator"
    # inert: nothing read or recorded; the persisted state carries no new key
    cfg2 = SimpleNamespace(lane_c=cfg.lane_c, marketcast=replace(MC, from_block=0))
    published.clear()
    drv2 = LaneCDriver(cfg2, Ops(), tmp_path / "drv2.json")
    assert drv2.settle(round_id="r1", block=FROM + 7200, era_index=1, base_pointer=king) == ""
    assert "mkt_await" not in json.loads((tmp_path / "drv2.json").read_text())
    assert not published


def test_marketcast_continuation_preserves_optimizer_sampler_and_ema(tmp_path):
    from cascade.trainer.continuation import Continuation, SnapshotStore
    snap = _train_snap(tmp_path)
    rc = parse_marketcast_recipe(_recipe(), MC)
    def model():
        torch.manual_seed(123)
        return _Toy()
    baseline = model()
    expected = L.finetune(baseline, _fwd(baseline), L.MktSampler(rc, MC, snap, seed=1), rc,
                          levels=TIMESTRAL_KNOTS, max_seconds=1e9)
    store = SnapshotStore(tmp_path / 'partial', {'recipe': 'fixture', 'data': snap.digest})
    ctx = Continuation(store, interval=0)
    save = ctx.save
    def interrupt(state):
        save(state)
        if state['step'] == 3:
            raise KeyboardInterrupt('fixture pod loss')
    ctx.save = interrupt
    lost = model()
    with pytest.raises(KeyboardInterrupt):
        L.finetune(lost, _fwd(lost), L.MktSampler(rc, MC, snap, seed=1), rc,
                   levels=TIMESTRAL_KNOTS, max_seconds=1e9, continuation=ctx)
    recovered = model()
    actual = L.finetune(recovered, _fwd(recovered), L.MktSampler(rc, MC, snap, seed=1), rc,
                         levels=TIMESTRAL_KNOTS, max_seconds=1e9,
                         continuation=Continuation(store, interval=0))
    assert actual.steps == expected.steps
    assert actual.real_samples == expected.real_samples
    assert actual.final_loss == expected.final_loss
    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, recovered.state_dict()[key]), key
