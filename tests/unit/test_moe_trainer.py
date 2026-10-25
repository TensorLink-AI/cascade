"""DEC-CA-0050 P1: the cascade_moe training path end to end on a tiny shape —
dispatch from Toto2Trainer, batched NorMuon, copula loss, micro-batches,
loss-free balancing, power EMAs, warm-start continuity, checkpoint guard +
wrapper, and a lane-B expert leg that merges cleanly."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.eval.checkpoint_guard import (  # noqa: E402
    CheckpointTampered,
    verify_checkpoint,
    verify_checkpoint_code,
)
from cascade.trainer.contract import compute_base_arch_digest, model_source_file  # noqa: E402
from cascade.trainer.moe_model import merge_expert_slices, moe_param_names  # noqa: E402
from cascade.trainer.moe_trainer import (  # noqa: E402
    POWER_EMA_FILE,
    BatchedNorMuonAdamW,
    PowerEMA,
    anneal_lr,
    gamma_for_sigma_rel,
    mup_lr_factor,
    polar_express_batched,
    refuse_moe_knobs,
    sigma_rel_of_gamma,
)
from cascade.trainer.toto2_trainer import Toto2Trainer  # noqa: E402


@dataclasses.dataclass
class _C:
    base_arch: str = "cascade_moe"
    arch_preset: str = "cascade-moe-tiny"
    d_model: int = 32
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int = 16
    patch_size: int = 4
    mlp_expansion: int = 2
    d_ff: int = 40
    num_quantiles: int = 9
    masking: str = "cpm"
    cpm_c_max: int = 2
    cpm_p_max: float = 0.4
    input_transform: str = "arcsinh_causal"
    context_length: int = 32
    horizon: int = 8
    batch_size: int = 4
    optimizer: str = "normuon_adamw"
    base_lr: float = 0.0025
    weight_decay: float = 2e-8
    lr_schedule: str = "wsd"
    umup_base_d_model: int = 32
    max_train_seconds: int = 600
    warmup_tokens: int = 64
    muon_momentum: float = 0.96
    muon_row_beta2: float = 0.999
    grad_clip: float = 7.0
    adamw_beta1: float = 0.91
    adamw_beta2: float = 0.972
    adamw_lr_scale: float = 0.2
    warm_lr_scale: float = 1.0
    budget_denomination: str = "points"
    batch_denomination: str = "series"
    anneal_fraction: float = 0.0
    ema_decay: float = 0.0
    micro_batches: int = 2
    mup_lr_exponent: float = 1.42
    power_ema_sigma_rels: tuple = (0.05, 0.10)
    power_ema_eval: int = -1
    power_ema_scope: str = "lineage"
    anneal_shape: str = "cosine"
    copula_loss_cap: float = 0.1
    copula_samples: int = 4
    copula_patches: int = 2


def _stream(n=48, C=2, L=32, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.cumsum(rng.normal(size=(C, L)), axis=-1)


def _train(tmp, contract, budget=2000, warm=None, sel=None, seed=7):
    tr = Toto2Trainer(device="cpu")
    return tr.train(_stream(), contract, training_seed=seed, token_budget=budget,
                    out_dir=tmp, warm_start_dir=warm, expert_selection=sel)


def test_digest_dispatch_keeps_toto2_source():
    assert model_source_file(SimpleNamespace(base_arch="toto2")) == "toto2_model.py"
    assert model_source_file(SimpleNamespace(base_arch="cascade_moe")) == "moe_model.py"
    c = _C()
    assert compute_base_arch_digest(c) != compute_base_arch_digest(
        dataclasses.replace(c, base_arch="toto2"))


def test_power_ema_gamma_round_trip():
    for s in (0.05, 0.10):
        assert sigma_rel_of_gamma(gamma_for_sigma_rel(s)) == pytest.approx(s, rel=1e-6)


def test_anneal_shapes():
    assert anneal_lr(0, 10, 1.0, "1-sqrt") == 1.0
    assert anneal_lr(10, 10, 1.0, "1-sqrt") == 0.0
    assert anneal_lr(5, 10, 1.0, "linear") == 0.5
    assert anneal_lr(25, 100, 1.0, "1-sqrt") == pytest.approx(0.5)


def test_mup_factor():
    assert mup_lr_factor(_C()) == 1.0
    assert mup_lr_factor(dataclasses.replace(_C(), d_model=64)) == pytest.approx(0.5 ** 1.42)
    assert mup_lr_factor(dataclasses.replace(_C(), mup_lr_exponent=0.0, d_model=64)) == 1.0


def test_batched_polar_matches_per_matrix():
    from cascade.trainer.toto2_trainer import _polar_express

    g = torch.randn(3, 8, 5)
    b = polar_express_batched(g)
    for i in range(3):
        torch.testing.assert_close(b[i], _polar_express(g[i]), rtol=1e-4, atol=1e-5)
    z = torch.zeros(2, 4, 6)
    assert torch.equal(polar_express_batched(z), z)


def test_toto2_refuses_moe_knobs():
    with pytest.raises(ValueError, match="cascade_moe recipe knobs"):
        refuse_moe_knobs(SimpleNamespace(base_arch="toto2", micro_batches=2))
    refuse_moe_knobs(SimpleNamespace(base_arch="toto2"))


def test_end_to_end_train_checkpoint_and_wrapper(tmp_path):
    c = _C()
    res = _train(tmp_path / "a", c)
    m = res.metrics
    assert m["base_arch"] == "cascade_moe" and m["steps"] > 0
    assert m["active_param_count"] < m["param_count"]
    assert np.isfinite(m["final_loss"]) and np.isfinite(m["copula_es"])
    d = tmp_path / "a"
    for f in ("weights.safetensors", "config.json", "model.py", "forecast_wrapper.py",
              "optimizer.safetensors", POWER_EMA_FILE):
        assert (d / f).is_file(), f
    assert verify_checkpoint_code(d) == "cascade_moe"
    rep = verify_checkpoint(d, c)
    assert rep.n_params == m["param_count"]
    with pytest.raises(CheckpointTampered, match="contract pins toto2"):
        verify_checkpoint(d, dataclasses.replace(c, base_arch="toto2"))
    # routing bias moved off zero (loss-free balancing ran)
    from safetensors.torch import load_file

    sd = load_file(str(d / "weights.safetensors"))
    assert float(sd["blocks.1.ffn.route_bias"].abs().sum()) > 0
    # the shipped wrapper loads and forecasts
    spec = importlib.util.spec_from_file_location("moe_wrap", d / "forecast_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    w = mod.Wrapper(d)
    out = w.forecast(np.cumsum(np.random.default_rng(1).normal(size=40)), 8, 5)
    assert out.shape == (1, 5, 8) and np.isfinite(out).all()
    assert w.forecast_quantiles_batch([np.arange(20.0)], 8).shape == (1, 8, 9)


def test_training_is_deterministic(tmp_path):
    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp_path / "x", c, budget=600)
    _train(tmp_path / "y", c, budget=600)
    from safetensors.torch import load_file

    a = load_file(str(tmp_path / "x" / "weights.safetensors"))
    b = load_file(str(tmp_path / "y" / "weights.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_warm_start_resumes_power_ema_and_optimizer(tmp_path):
    c = _C()
    _train(tmp_path / "g0", c, budget=600)
    from safetensors.torch import load_file

    t0 = int(load_file(str(tmp_path / "g0" / POWER_EMA_FILE))["ema0.05.__t__"].item())
    res = _train(tmp_path / "g1", c, budget=600, warm=tmp_path / "g0")
    assert res.metrics["optim_state_resumed"] is True
    t1 = int(load_file(str(tmp_path / "g1" / POWER_EMA_FILE))["ema0.05.__t__"].item())
    assert t1 == t0 + res.metrics["steps"]


def test_ema_eval_ships_ema_with_live_bias(tmp_path):
    c = dataclasses.replace(_C(), power_ema_eval=0)
    _train(tmp_path / "e", c, budget=600)
    from safetensors.torch import load_file

    scored = load_file(str(tmp_path / "e" / "weights.safetensors"))
    raw = load_file(str(tmp_path / "e" / "weights_stable.safetensors"))
    assert torch.equal(scored["blocks.1.ffn.route_bias"], raw["blocks.1.ffn.route_bias"])
    assert not torch.equal(scored["head.weight"], raw["head.weight"])


def test_expert_leg_changes_only_selected_rows_and_merges(tmp_path):
    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp_path / "king", c, budget=600)
    sel = {0: [3, 7, 11]}
    res = _train(tmp_path / "leg", c, budget=600, warm=tmp_path / "king", sel=sel, seed=9)
    assert res.metrics["expert_selection"] == {"0": [3, 7, 11]}
    assert res.metrics["expert_training_backend"] == "compact_rows"
    assert res.metrics["expert_trainable_param_count"] == 3 * 2 * c.d_model * 92
    from safetensors.torch import load_file

    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel

    king = load_file(str(tmp_path / "king" / "weights.safetensors"))
    leg = load_file(str(tmp_path / "leg" / "weights.safetensors"))
    names = moe_param_names(CascadeMoEModel(CascadeMoEConfig.from_contract(c)))
    merged = merge_expert_slices(king, leg, sel, names)   # raises if anything leaked
    w_in = names[0][0]
    assert not torch.equal(merged[w_in][[3, 7, 11]], king[w_in][[3, 7, 11]])
    assert torch.equal(leg["blocks.1.ffn.route_bias"], king["blocks.1.ffn.route_bias"])


@pytest.mark.parametrize("anneal", [0.0, 0.25])
def test_compact_expert_leg_determinism_and_anneal(tmp_path, anneal):
    """Compact leaves must work with copula autograd, accumulation and the
    optional cooldown, and produce identical artifacts on repeat runs."""
    from safetensors.torch import load_file

    c = dataclasses.replace(_C(), anneal_fraction=anneal)
    _train(tmp_path / "king", c, budget=400)
    for name in ("x", "y"):
        _train(tmp_path / name, c, budget=400, warm=tmp_path / "king",
               sel={0: [3, 7, 11]}, seed=9)
    a = load_file(str(tmp_path / "x" / "weights.safetensors"))
    b = load_file(str(tmp_path / "y" / "weights.safetensors"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not (tmp_path / "x" / "optimizer.safetensors").exists()


def test_expert_selection_refused_on_toto2():
    with pytest.raises(ValueError, match="requires base_arch='cascade_moe'"):
        Toto2Trainer(device="cpu").train(iter(()), SimpleNamespace(base_arch="toto2"),
                                         training_seed=1, token_budget=1, out_dir=Path("."),
                                         expert_selection={0: [1]})


def test_batched_normuon_state_round_trip():
    p3 = torch.nn.Parameter(torch.randn(3, 6, 4))
    p2 = torch.nn.Parameter(torch.randn(5, 6))
    opt = BatchedNorMuonAdamW([p3, p2], [], lr=0.01, weight_decay=0.0)
    p3.grad, p2.grad = torch.randn_like(p3), torch.randn_like(p2)
    opt.step()
    names = {id(p3): "e", id(p2): "m"}
    flat = opt.state_tensors(names)
    opt2 = BatchedNorMuonAdamW([p3, p2], [], lr=0.01, weight_decay=0.0)
    assert opt2.load_state_tensors(flat, names) == 0
    assert torch.equal(opt2._row_v[p3], opt._row_v[p3])
    _ = PowerEMA  # imported for API presence


def test_preset_digest_matches_source():
    """deploy/presets/cascade-moe-28m.toml pins base_arch_digest; an edit to
    moe_model.py must re-pin it (the preset is what operators paste)."""
    import tomllib

    root = Path(__file__).resolve().parents[2]
    t = tomllib.loads((root / "deploy/presets/cascade-moe-28m.toml").read_text())["training"]
    c = SimpleNamespace(**t)
    assert compute_base_arch_digest(c) == t["base_arch_digest"]
    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel, param_counts

    with torch.device("meta"):
        m = CascadeMoEModel(CascadeMoEConfig.from_contract(c))
    counts = param_counts(m)
    assert counts["total"] == pytest.approx(191e6, rel=0.01)


def test_expert_leg_under_power_ema_preset_settles_bit_exact(tmp_path):
    """Audit finding 6: under the MoE preset (power EMA scored, raw endpoint
    in weights_stable) an expert leg must start from the king's SCORED
    weights and leave every frozen tensor bit-identical — else
    merge_expert_slices refuses EVERY leg. Owner 2026-10-19: the leg scores
    a power EMA of its own training, frozen rows restored."""
    from safetensors.torch import load_file

    from cascade.trainer.expert_lane import settle_leg
    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.moe_trainer import POWER_EMA_FILE

    c = dataclasses.replace(_C(), copula_loss_cap=0.0, power_ema_sigma_rels=(0.05, 0.10),
                            power_ema_eval=0)
    _train(tmp_path / "king", c, budget=600)
    king = load_file(str(tmp_path / "king" / "weights.safetensors"))
    stable = load_file(str(tmp_path / "king" / "weights_stable.safetensors"))
    assert not torch.equal(king["head.weight"], stable["head.weight"])   # EMA ≠ raw
    sel = {0: [2, 5], 1: [0]}
    res = _train(tmp_path / "leg", c, budget=600, warm=tmp_path / "king", sel=sel, seed=9)
    assert res.metrics["optim_state_resumed"] is False                   # fresh optimizer
    leg_dir = tmp_path / "leg"
    assert not (leg_dir / POWER_EMA_FILE).exists()                       # no lineage files
    assert not (leg_dir / "weights_stable.safetensors").exists()
    leg = load_file(str(leg_dir / "weights.safetensors"))
    names = moe_param_names(CascadeMoEModel(CascadeMoEConfig.from_contract(c)))
    owned = {n for pair in names for n in pair}
    for k in king:
        if k not in owned:
            assert torch.equal(king[k], leg[k]), k
    for li, (w_in, w_out) in enumerate(names):
        rows = sel.get(li, [])
        out = [r for r in range(king[w_in].shape[0]) if r not in rows]
        for n in (w_in, w_out):
            assert torch.equal(king[n][out], leg[n][out]), n
            if rows:
                assert not torch.equal(king[n][rows], leg[n][rows]), n
    # the settlement rebase (validators run the same byte check) succeeds
    settle_leg(tmp_path / "king", tmp_path / "king", leg_dir, sel, tmp_path / "rebased")
    merged = load_file(str(tmp_path / "rebased" / "weights.safetensors"))
    assert torch.equal(merged["head.weight"], king["head.weight"])


def test_restore_frozen_copies_back_everything_unselected():
    from cascade.trainer.moe_trainer import restore_frozen

    base = {"a": torch.zeros(3), "e_in": torch.zeros(4, 2, 2), "e_out": torch.zeros(4, 2, 2)}
    live = {k: v + 1 for k, v in base.items()}
    out = restore_frozen(live, base, {0: [1, 3]}, [("e_in", "e_out")])
    assert torch.equal(out["a"], base["a"])
    assert torch.equal(out["e_in"][[0, 2]], base["e_in"][[0, 2]])
    assert torch.equal(out["e_in"][[1, 3]], live["e_in"][[1, 3]])
    with pytest.raises(ValueError):
        restore_frozen({"a": live["a"]}, base, {}, [])


def test_expert_leg_scores_its_own_power_ema(tmp_path):
    """The trained rows ship the leg's EMA (not the raw endpoint, not the
    king's lineage average); a raw-export run of the same leg differs."""
    from safetensors.torch import load_file

    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel

    c = dataclasses.replace(_C(), copula_loss_cap=0.0, power_ema_eval=0)
    _train(tmp_path / "king", c, budget=600)
    sel = {0: [2, 5]}
    _train(tmp_path / "ema", c, budget=600, warm=tmp_path / "king", sel=sel, seed=9)
    raw_c = dataclasses.replace(c, power_ema_eval=-1)
    _train(tmp_path / "raw", raw_c, budget=600, warm=tmp_path / "king", sel=sel, seed=9)
    names = moe_param_names(CascadeMoEModel(CascadeMoEConfig.from_contract(c)))
    w_in = names[0][0]
    ema = load_file(str(tmp_path / "ema" / "weights.safetensors"))[w_in]
    raw = load_file(str(tmp_path / "raw" / "weights.safetensors"))[w_in]
    king = load_file(str(tmp_path / "king" / "weights.safetensors"))[w_in]
    assert not torch.equal(ema[[2, 5]], raw[[2, 5]])           # averaged, not the endpoint
    assert torch.equal(ema[[0, 1, 3]], king[[0, 1, 3]])         # frozen rows exact


def test_leg_scope_restarts_the_power_ema(tmp_path):
    from safetensors.torch import load_file

    c = dataclasses.replace(_C(), power_ema_scope="leg")
    _train(tmp_path / "g0", c, budget=600)
    res = _train(tmp_path / "g1", c, budget=600, warm=tmp_path / "g0")
    t1 = int(load_file(str(tmp_path / "g1" / POWER_EMA_FILE))["ema0.05.__t__"].item())
    assert t1 == res.metrics["steps"]                     # this leg only
    assert res.metrics["optim_state_resumed"] is True     # the lineage itself continues


def test_power_ema_scope_is_digest_inert_at_its_default():
    from cascade.shared.manifest import contract_digest

    c = _C()
    assert contract_digest(c) == contract_digest(dataclasses.replace(c, power_ema_scope="lineage"))
    assert contract_digest(c) != contract_digest(dataclasses.replace(c, power_ema_scope="leg"))



@pytest.mark.parametrize('expert,anneal', [(False, False), (True, False), (False, True)])
def test_continuation_matches_uninterrupted_optimizer_ema_router_and_weights(tmp_path, expert, anneal):
    from safetensors.torch import load_file

    from cascade.trainer.continuation import Continuation, SnapshotStore

    contract = dataclasses.replace(_C(), anneal_fraction=0.2 if anneal else 0)
    warm = tmp_path / 'base' if expert else None
    if warm is not None:
        _train(warm, contract, budget=512)
    selection = {0: [0, 1], 1: [2]} if expert else None
    trainer = Toto2Trainer(device='cpu')
    kwargs = dict(contract=contract, training_seed=7, token_budget=2000,
                  warm_start_dir=warm, expert_selection=selection)
    baseline = trainer.train(_stream(), out_dir=tmp_path / 'baseline', **kwargs)
    store = SnapshotStore(tmp_path / 'partial', {'base': 'fixture', 'selection': selection})
    checkpoint = Continuation(store, interval=0)
    original = checkpoint.save
    def crash(state):
        original(state)
        if state['step'] == 3:
            raise KeyboardInterrupt('simulated pod loss after durable optimizer boundary')
    checkpoint.save = crash
    with pytest.raises(KeyboardInterrupt):
        trainer.train(_stream(), out_dir=tmp_path / 'lost', continuation=checkpoint, **kwargs)
    recovered = trainer.train(_stream(), out_dir=tmp_path / 'recovered',
                              continuation=Continuation(store, interval=0), **kwargs)
    assert recovered.metrics['steps'] == baseline.metrics['steps']
    assert recovered.metrics['tokens_seen'] == baseline.metrics['tokens_seen']
    for path in (tmp_path / 'baseline').glob('*.safetensors'):
        expected, actual = load_file(str(path)), load_file(str(tmp_path / 'recovered' / path.name))
        assert expected.keys() == actual.keys()
        for name in expected:
            assert torch.equal(expected[name], actual[name]), (path.name, name)


def test_continuation_refuses_a_changed_input_prefix(tmp_path):
    from cascade.trainer.continuation import Continuation, SnapshotStore
    store = SnapshotStore(tmp_path / 'partial', {'job': 'fixture'})
    trainer = Toto2Trainer(device='cpu')
    contract = _C()
    checkpoint = Continuation(store, interval=0)
    original = checkpoint.save
    def crash(state):
        original(state)
        raise KeyboardInterrupt()
    checkpoint.save = crash
    with pytest.raises(KeyboardInterrupt):
        trainer.train(_stream(), contract, training_seed=7, token_budget=2000,
                      out_dir=tmp_path / 'lost', continuation=checkpoint)
    with pytest.raises(ValueError, match='input replay differs'):
        trainer.train(_stream(seed=99), contract, training_seed=7, token_budget=2000,
                      out_dir=tmp_path / 'changed', continuation=Continuation(store))


def test_completed_continuation_does_not_consume_an_extra_input_batch(tmp_path):
    from cascade.trainer.continuation import Continuation, SnapshotStore
    trainer = Toto2Trainer(device='cpu')
    contract = _C()
    consumed = []
    def stream():
        for array in _stream():
            consumed.append(1)
            yield array
    store = SnapshotStore(tmp_path / 'partial', {'job': 'fixture'})
    trainer.train(stream(), contract, training_seed=7, token_budget=512,
                  out_dir=tmp_path / 'first', continuation=Continuation(store, interval=0))
    original = len(consumed)
    consumed.clear()
    trainer.train(stream(), contract, training_seed=7, token_budget=512,
                  out_dir=tmp_path / 'recovered', continuation=Continuation(store, interval=0))
    assert len(consumed) == original
