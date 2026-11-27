"""W-BMASK observed_mask/2 (owner 2026-11-27): in the cascade_pyramid model's
attention, a patch that is ENTIRELY left padding (before the first real
point) is never a KEY. Each padding token keeps itself. Gaps, CPM-style
masked spans and the horizon are untouched. No leading padding ⇒ the model
runs exactly as before (outputs and grads byte-equal, fp32 and bf16).

The rule is applied by the bench sidecar (observed.py) to the LOADED model's
attention module instances, so it reaches every checkpoint through its own
byte-pinned model.py, including a pre-groups one (fixture: model.py at
4f100495). Tests without the ``torch`` mark never import torch; the torch
tests run in CI only.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import OBSERVED, REPO, SIDECAR, load_observed, pyramid_contract, series

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

FIXTURES = Path(__file__).with_name("fixtures")
OLD_MODEL = FIXTURES / "pyramid_model_4f100495.py.txt"     # pre-groups: forward(h)
RELEASE_MODEL = REPO / "cascade" / "trainer" / "pyramid_model.py"
RELEASE_WRAPPER = REPO / "cascade" / "trainer" / "pyramid_wrapper.py"
PS = 32


def _obs():
    obs = load_observed()
    for name in ("LeadingPadKeys", "leading_pad_points", "leading_pad_patches",
                 "attention_source_sha256", "KNOWN_ATTENTION_SHA256", "time_key_mask"):
        assert hasattr(obs, name), f"observed.py has no {name} (leading-pad key rule)"
    return obs


# ── non-torch: the method stamp, the padding mask, the version check ────────


def test_method_is_bumped_and_v1_stays_reproducible():
    obs = load_observed()
    assert obs.METHOD == "observed_mask/2"
    assert set(obs.METHODS) == {"observed_mask/2", "observed_mask/1", "impute/0"}
    from cascade.eval import benchmarks as B

    assert B.BENCH_METHOD == "observed_mask/2"
    assert set(B.BENCH_METHODS) == set(obs.METHODS)
    assert B.bench_method_arg("observed_mask/1") == ["--bench-method", "observed_mask/1"]
    cli = (SIDECAR / "cli.py").read_text()
    assert '"observed_mask/2", "observed_mask/1", "impute/0"' in cli


def test_padding_mask_is_leading_padding_only_never_gaps_or_missing():
    obs = _obs()
    L = 512
    h = series(100, seed=1)
    h[:5] = np.nan                    # missing real points at the start: NOT padding
    h[50:60] = np.nan                 # a gap: NOT padding
    pad = obs.leading_pad_points([h, series(L + 9, seed=2)], L)
    assert pad.shape == (2, L)
    assert pad[0, :L - 100].all() and not pad[0, L - 100:].any()
    assert not pad[1].any()
    patches = obs.leading_pad_patches(pad, PS)
    # 412 padded points: 12 whole patches; patch 12 holds 28 pad + 4 real points
    assert patches[0].tolist() == [True] * 12 + [False] * 4
    assert not patches[1].any()
    # only LEADING whole patches count: a padding-marked patch after real data is a gap
    marked = np.zeros((1, 4 * PS), dtype=bool)
    marked[0, :PS] = True
    marked[0, 2 * PS:3 * PS] = True
    assert obs.leading_pad_patches(marked, PS).tolist() == [[True, False, False, False]]


def test_every_released_attention_source_is_known_and_others_refused(tmp_path):
    obs = _obs()
    old = obs.attention_source_sha256(OLD_MODEL.read_text())
    new = obs.attention_source_sha256(RELEASE_MODEL.read_text())
    assert old in obs.KNOWN_ATTENTION_SHA256 and new in obs.KNOWN_ATTENTION_SHA256
    assert old != new

    from types import SimpleNamespace

    edited = RELEASE_MODEL.read_text().replace(
        "        B, T, _ = h.shape\n", "        B, T, _ = h.shape  # edited\n", 1)
    assert obs.attention_source_sha256(edited) not in obs.KNOWN_ATTENTION_SHA256
    (tmp_path / "model.py").write_text(edited)

    class Model:
        def forward(self, patches, mask=None):
            return patches

    m = SimpleNamespace(__name__="cascade_pyramid_checkpoint", __file__=str(tmp_path / "model.py"),
                        causal_standardize=lambda *a: a)
    w = SimpleNamespace(m=m, model=Model(), cfg=object(), device="cpu", quantile_levels=(0.5,))
    with pytest.raises(TypeError, match="unknown _GatedAttention source"):
        obs.adapt(w, "observed_mask/2")
    assert obs.adapt(w, "observed_mask/1").pad_hook is None      # v1: no hook needed


def test_the_release_model_and_wrapper_files_are_untouched_by_the_rule():
    # the rule lives in the sidecar: pyramid_model.py's bytes are folded into
    # base_arch_digest and pinned per checkpoint (checkpoint_guard)
    assert "pad_mask" not in RELEASE_MODEL.read_text()
    assert "pad_mask" not in RELEASE_WRAPPER.read_text()
    assert "LeadingPadKeys.install(inner.m, inner.model)" in OBSERVED.read_text()


# ── torch helpers ────────────────────────────────────────────────────────────


def _load_module(path: Path, name: str):
    import importlib.machinery
    import importlib.util
    import sys

    # SourceFileLoader: the old fixture is stored as .py.txt (never linted/collected)
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dense_cfg(pm, context: int = 512):
    """A small all-dense pyramid (no routed experts, so float noise can
    never flip a top-k pick in the approximate comparisons)."""
    return pm.CascadeMoEConfig(d_model=64, num_layers=4, num_heads=4, head_dim=16, patch_size=PS,
                               context_length=context, variate_blocks=(2,), moe_blocks=(),
                               experts_per_layer=(), dense_ffn=128, shared_hidden=64,
                               conv_channels=16)


def _dense_model(pm, seed: int = 5):
    import torch

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = pm.CascadeMoEModel(_dense_cfg(pm)).eval()
        # random-init head weights are ~1e-3; widen them so differences show
        torch.nn.init.normal_(model.head.weight, std=0.5)
    return model


def _inputs(b, c, real, horizon, lead=0, seed=3):
    """``(patches, mask, pad)``: ``lead`` all-padding patches, ``real``
    observed patches, ``horizon`` masked patches."""
    import torch

    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(b, c, real, PS, generator=g)
    p = lead + real + horizon
    patches = torch.zeros(b, c, p, PS)
    patches[:, :, lead:lead + real] = vals
    mask = torch.ones(b, c, p, PS, dtype=torch.bool)
    mask[:, :, lead:lead + real] = False
    pad = torch.zeros(b, c, p, dtype=torch.bool)
    pad[:, :, :lead] = True
    return patches, mask, pad


# ── torch: the rule on the model ─────────────────────────────────────────────


@torch_mark
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_no_leading_padding_is_bit_identical_outputs_and_grads(precision):
    """The proof that training (never pads) and every numerics digest are
    unaffected: hooked, with no / all-False / gap-only padding, the release
    model's outputs and every parameter grad are byte-equal to the unhooked
    model's, fp32 and bf16 autocast (the real 2m MoE size, train mode)."""
    import torch

    obs = _obs()
    from cascade.trainer import pyramid_model as pm

    with torch.random.fork_rng():
        torch.manual_seed(11)
        model = pm.CascadeMoEModel(pm.CascadeMoEConfig.from_contract(pyramid_contract())).train()
    patches, mask, _ = _inputs(2, 2, real=6, horizon=2, seed=4)
    gap = torch.zeros(2, 2, 8, dtype=torch.bool)
    gap[:, :, 3] = True                      # a padding-marked patch after real data

    def run(hook=None, pad=None):
        model.zero_grad(set_to_none=True)
        amp = (torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16"
               else contextlib.nullcontext())
        scope = hook.active(pad) if hook is not None else contextlib.nullcontext()
        with amp, scope:
            out = model(patches, mask=mask, with_parameters=True)
        (out.float().square().mean() + model.aux_loss).backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return out.detach().clone(), grads

    base_out, base_grads = run()
    assert base_grads
    hook = obs.LeadingPadKeys.install(pm, model)
    for out, grads in (run(), run(hook, torch.zeros(2, 2, 8, dtype=torch.bool)), run(hook, gap)):
        assert torch.equal(out, base_out)
        assert grads.keys() == base_grads.keys()
        for name, g in grads.items():
            assert torch.equal(g, base_grads[name]), name


@torch_mark
def test_padding_keys_get_self_only_and_real_queries_skip_them():
    import torch

    obs = _obs()
    pad = torch.tensor([[True, True, False, False, False]])
    m = obs.time_key_mask(pad)[0, 0]
    assert m.shape == (5, 5)
    assert m[0].tolist() == [True, False, False, False, False]
    assert m[1].tolist() == [False, True, False, False, False]      # self kept, never empty
    assert m[4].tolist() == [False, False, True, True, True]        # causal, no pad keys
    assert bool(m.any(-1).all())
    v = obs.variate_key_mask(torch.tensor([[True, False, False]]))[0, 0]
    # a padding QUERY may read real keys (harmless); no query reads a padding key but itself
    assert v.tolist() == [[True, True, True], [False, True, True], [False, True, True]]


# The reference for "the padding has no influence": the same window with ONE
# leading padding patch. The pinned stem's causal dilated conv reaches 24
# points back (kernel 7, dilations 1 and 4; < one patch), so the first real
# patch's embedding sees padding either way; with the rule, the real tokens'
# outputs are then independent of HOW MUCH padding precedes them (xPos is
# relative, the scaler reads observed points only).


@torch_mark
@pytest.mark.parametrize("channels", [1, 2])
def test_leading_pad_patches_are_never_keys(channels):
    import torch

    obs = _obs()
    from cascade.trainer import pyramid_model as pm

    model = _dense_model(pm)
    real, hz = 6, 2
    many_p, many_m, many_pad = _inputs(2, channels, real, hz, lead=5)
    one_p, one_m, one_pad = _inputs(2, channels, real, hz, lead=1)
    with torch.no_grad():
        leak_many = model(many_p, mask=many_m, with_parameters=True)[:, :, 5:]
        leak_one = model(one_p, mask=one_m, with_parameters=True)[:, :, 1:]
        hook = obs.LeadingPadKeys.install(pm, model)
        with hook.active(many_pad):
            got_many = model(many_p, mask=many_m, with_parameters=True)
        with hook.active(one_pad):
            got_one = model(one_p, mask=one_m, with_parameters=True)
    assert torch.isfinite(got_many).all() and torch.isfinite(got_one).all()  # pad rows too
    torch.testing.assert_close(got_many[:, :, 5:], got_one[:, :, 1:], rtol=1e-3, atol=1e-3)
    # control: without the rule every padding patch is a key, so the amount leaks
    assert float((leak_many - leak_one).abs().max()) > 1e-2


@torch_mark
def test_gaps_and_horizon_stay_keys():
    """An interior masked patch (a gap / CPM span) and the horizon are NOT
    padding: marking the gap as padding is ignored (not leading), and the
    window's outputs with the rule still match the one-padding-patch window,
    gap included."""
    import torch

    obs = _obs()
    from cascade.trainer import pyramid_model as pm

    model = _dense_model(pm)
    real, hz = 8, 2

    def window(lead):
        patches, mask, pad = _inputs(1, 1, real, hz, lead=lead)
        patches[:, :, lead + 3] = 0
        mask[:, :, lead + 3] = True                                 # a masked interior patch
        return patches, mask, pad

    p3, m3, pad3 = window(3)
    p1, m1, pad1 = window(1)
    hook = obs.LeadingPadKeys.install(pm, model)
    with torch.no_grad():
        with hook.active(pad1):
            ref = model(p1, mask=m1, with_parameters=True)[:, :, 1:]
        with hook.active(pad3):
            got = model(p3, mask=m3, with_parameters=True)
        wrong = pad3.clone()
        wrong[:, :, 3 + 3] = True                                   # the gap: not leading
        wrong[:, :, -1] = True                                      # a horizon patch: same
        with hook.active(wrong):
            same = model(p3, mask=m3, with_parameters=True)
    torch.testing.assert_close(got[:, :, 3:], ref, rtol=1e-3, atol=1e-3)
    assert torch.equal(same, got)


# ── torch: through the sidecar adapter, old and current pinned model.py ──────


def _checkpoint(root: Path, model_src: Path, context: int) -> Path:
    """A dense checkpoint dir whose model.py is ``model_src``'s bytes and
    whose forecast_wrapper.py is the release's pinned wrapper. The weights
    do not depend on ``context`` (positional buffers are non-persistent)."""
    from safetensors.torch import save_file

    pm = _load_module(model_src, f"w_bmask_lp_{model_src.stem.replace('.', '_')}")
    model = _dense_model(pm)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_bytes(model_src.read_bytes())
    (root / "forecast_wrapper.py").write_bytes(RELEASE_WRAPPER.read_bytes())
    (root / "config.json").write_text(json.dumps({"moe": _dense_cfg(pm, context).to_dict()}))
    save_file({k: v.detach().contiguous().clone() for k, v in model.state_dict().items()},
              str(root / "weights.safetensors"))
    return root


@torch_mark
@pytest.mark.parametrize("model_src", [OLD_MODEL, RELEASE_MODEL], ids=["4f100495", "release"])
def test_sidecar_adapter_applies_the_rule_to_a_pinned_checkpoint(tmp_path, model_src):
    """The bench sidecar's adapter (observed_mask/2) on a checkpoint whose
    pinned model.py predates the rule: a 192-point history at context 512
    (10 padding patches) forecasts what it does at context 224 (1 padding
    patch). Under observed_mask/1 the padding leaks (control)."""
    from .helpers import load_pinned_wrapper

    obs = _obs()
    horizon = 40
    h = series(6 * PS, seed=12)
    wide = load_pinned_wrapper(_checkpoint(tmp_path / "wide", model_src, 512))
    narrow = load_pinned_wrapper(_checkpoint(tmp_path / "narrow", model_src, 7 * PS))
    v1 = [obs.adapt(w, "observed_mask/1").forecast_quantiles_batch([h], horizon)
          for w in (wide, narrow)]
    wide2, narrow2 = obs.adapt(wide, "observed_mask/2"), obs.adapt(narrow, "observed_mask/2")
    assert wide2.pad_hook is not None and not wide2.native_pad
    v2 = [w.forecast_quantiles_batch([h], horizon) for w in (wide2, narrow2)]
    assert np.isfinite(v2[0]).all()
    np.testing.assert_allclose(v2[0], v2[1], rtol=1e-3, atol=1e-3)
    assert float(np.abs(v1[0] - v1[1]).max()) > 1e-2            # v1: the padding leaked
    # the pinned file is never edited; a full-length history is untouched
    assert (tmp_path / "wide" / "model.py").read_bytes() == model_src.read_bytes()
    full = series(512, seed=13)
    np.testing.assert_array_equal(wide2.forecast_quantiles_batch([full], horizon),
                                  wide.forecast_quantiles_batch([full], horizon))
    s = wide2.forecast(h, horizon, 4)
    assert s.shape == (1, 4, horizon) and np.isfinite(s).all()
