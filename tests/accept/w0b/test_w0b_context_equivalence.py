"""W0b: context equivalence of the lane D scorer (plan rev6 §3.0 W0b row,
§1.1, B7). Contract: README.md.

Tests without the ``torch`` mark never import torch. Tests marked ``torch``
build tiny 2m pyramid checkpoints with the repo's trainer code; CI runs them
(``uv sync --all-extras``), the production box runs only ``-m 'not torch'``.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    ATOL,
    HEAD_LEVELS,
    HORIZON,
    REPO,
    SCORING_LEVELS,
    TRAINED_CONTEXT,
    W0B_CONTEXTS,
    long_histories,
    pyramid_contract,
    series,
    tiny_checkpoint,
)

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)


# ── non-torch: the variant list ──────────────────────────────────────────────


def test_w0b_iterates_exactly_the_scorer_contexts():
    from cascade.eval.lane_d_scorer import SCORER_CONTEXTS

    assert tuple(SCORER_CONTEXTS) == W0B_CONTEXTS, (
        "W0b's strict-load sweep must cover every context the scorer can build a model at")
    assert all(c % 32 == 0 and c >= TRAINED_CONTEXT for c in SCORER_CONTEXTS)


def test_lane_d_contexts_are_a_subset_of_the_scorer_contexts():
    from cascade.eval.lane_d_scorer import SCORER_CONTEXTS
    from cascade.shared.config.lane_d import LANE_D_CONTEXTS

    raw = tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))
    armed = raw["lane_d"]["contexts"]
    assert armed and set(armed) <= set(SCORER_CONTEXTS), (armed, SCORER_CONTEXTS)
    assert set(LANE_D_CONTEXTS) <= set(SCORER_CONTEXTS), (LANE_D_CONTEXTS, SCORER_CONTEXTS)


def test_scorer_reads_the_nine_level_grid_out_of_the_head():
    from cascade.eval.lane_d_scorer import SCORING_LEVELS as LEVELS
    from cascade.eval.lane_d_scorer import level_indices

    assert tuple(LEVELS) == SCORING_LEVELS
    idx = level_indices(HEAD_LEVELS)
    assert [HEAD_LEVELS[i] for i in idx] == pytest.approx(list(SCORING_LEVELS), abs=1e-12)


# ── torch: fixtures ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _checkpoints(root: str) -> dict[str, Path]:
    """An F32 checkpoint (+3 median shift, like a lane A king) and a BF16 one
    (a lane D entry), built once per session under ``root``."""
    contract = pyramid_contract()
    base = Path(root)
    return {
        "f32": tiny_checkpoint(base / "f32", contract, seed=21, median_shift=3.0, bf16=False),
        "bf16": tiny_checkpoint(base / "bf16", contract, seed=22, median_shift=0.0, bf16=True),
    }


@pytest.fixture
def ckpts(tmp_path_factory):
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield _checkpoints(str(tmp_path_factory.getbasetemp() / "w0b_ckpts"))
    finally:
        torch.set_num_threads(threads)


def _lane_d(ckpt: Path, histories, context: int) -> np.ndarray:
    from cascade.eval.lane_d_scorer import forecast_quantiles

    return np.asarray(forecast_quantiles(ckpt, histories, context=context, horizon=HORIZON,
                                         contexts=W0B_CONTEXTS))


def _lane_a(ckpt: Path):
    """Lane A's quantile-head path under ``[v2] quantile_head_after``: the
    checkpoint's pinned ``forecast_wrapper.py`` (``pyramid_wrapper``), its
    ``forecast_quantiles_batch``, the 9 grid levels picked by exact match."""
    from cascade.validator.evaluator import load_quantile_forecaster

    return load_quantile_forecaster(ckpt, trust_checkpoint_code=True)


# ── torch: quantile-head equality at 4096 ────────────────────────────────────


@torch_mark
@pytest.mark.parametrize("kind", ["f32", "bf16"])
def test_checkpoint_is_the_pinned_4096_format(ckpts, kind):
    d = ckpts[kind]
    assert json.loads((d / "config.json").read_text())["moe"]["context_length"] == TRAINED_CONTEXT
    assert (d / "forecast_wrapper.py").read_bytes() == (
        REPO / "cascade" / "trainer" / "pyramid_wrapper.py").read_bytes()
    assert (d / "model.py").read_bytes() == (
        REPO / "cascade" / "trainer" / "pyramid_model.py").read_bytes()


@torch_mark
@pytest.mark.parametrize("kind", ["f32", "bf16"])
def test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_per_window(ckpts, kind):
    """Lane A decodes one window per forward through the pinned wrapper; the
    lane D scorer at 4096 must give the same 9-level quantiles (≤ ATOL)."""
    hists = long_histories()
    assert all(h.size >= TRAINED_CONTEXT for h in hists)
    q_d = _lane_d(ckpts[kind], hists, TRAINED_CONTEXT)
    assert q_d.shape == (len(hists), HORIZON, len(SCORING_LEVELS)) and np.isfinite(q_d).all()
    qfn = _lane_a(ckpts[kind])
    q_a = np.stack([np.asarray(qfn(h, HORIZON))[0] for h in hists])
    assert q_a.shape == q_d.shape
    diff = float(np.max(np.abs(q_d - q_a)))
    assert diff <= ATOL, f"lane D at 4096 vs pyramid_wrapper quantile head: max |diff| {diff}"


@torch_mark
@pytest.mark.parametrize("kind", ["f32", "bf16"])
def test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_batched(ckpts, kind):
    """The same windows through the wrapper in one batched forward
    (DEC-CA-0068 ``quantile_batch``) equal the lane D scorer too."""
    hists = long_histories()
    q_d = _lane_d(ckpts[kind], hists, TRAINED_CONTEXT)
    q_a = np.asarray(_lane_a(ckpts[kind]).batch(hists, HORIZON))
    assert q_a.shape == q_d.shape == (len(hists), HORIZON, len(SCORING_LEVELS))
    diff = float(np.max(np.abs(q_d - q_a)))
    assert diff <= ATOL, f"lane D at 4096 vs batched pyramid_wrapper: max |diff| {diff}"


@torch_mark
def test_tolerance_discriminates_the_wrapper_edge_pad(ckpts):
    """Control: on a 320-point window the wrapper edge-pads to 4096 and the
    lane D scorer does not (§1.1 no-pad), so they must differ by far more
    than ATOL. This shows ATOL is tight enough to catch a real divergence."""
    short = series(320, seed=7)
    q_d = _lane_d(ckpts["f32"], [short], TRAINED_CONTEXT)
    q_a = np.asarray(_lane_a(ckpts["f32"])(short, HORIZON))
    diff = float(np.max(np.abs(q_d - q_a)))
    assert diff > 100 * ATOL, f"edge-padded wrapper vs no-pad scorer differ only by {diff}"


# ── torch: strict load at every context variant ──────────────────────────────


class _LoadSpy:
    """Records every ``CascadeMoEModel.load_state_dict`` call the scorer
    makes (observation only: the original method runs unchanged)."""

    def __init__(self, monkeypatch):
        from cascade.trainer import pyramid_model as pm

        self.calls: list[dict] = []
        original = pm.CascadeMoEModel.load_state_dict
        spy = self

        def load_state_dict(model, state_dict, strict=True, assign=False):
            result = original(model, state_dict, strict=strict, assign=assign)
            spy.calls.append({
                "strict": strict,
                "missing": list(result.missing_keys),
                "unexpected": list(result.unexpected_keys),
                "context_length": int(model.cfg.context_length),
                "cfg": json.loads(json.dumps(model.cfg.to_dict(), sort_keys=True)),
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            })
            return result

        monkeypatch.setattr(pm.CascadeMoEModel, "load_state_dict", load_state_dict)


@torch_mark
@pytest.mark.parametrize("kind", ["f32", "bf16"])
def test_4096_checkpoint_loads_strict_at_every_context_variant(ckpts, kind, monkeypatch):
    import torch
    from safetensors.torch import load_file

    d = ckpts[kind]
    on_disk = load_file(str(d / "weights.safetensors"))
    saved_cfg = json.loads((d / "config.json").read_text())["moe"]
    spy = _LoadSpy(monkeypatch)
    hist = [series(TRAINED_CONTEXT, seed=5)]
    for c in W0B_CONTEXTS:
        before = len(spy.calls)
        q = _lane_d(d, hist, c)
        assert q.shape == (1, HORIZON, len(SCORING_LEVELS)) and np.isfinite(q).all(), c
        assert len(spy.calls) == before + 1, f"context {c}: the scorer did not load the weights once"
        call = spy.calls[-1]
        assert call["strict"] is True, f"context {c}: loaded with strict={call['strict']!r}"
        assert call["missing"] == [] and call["unexpected"] == [], (c, call["missing"],
                                                                   call["unexpected"])
        assert call["context_length"] == c, f"context {c}: model built at {call['context_length']}"
        assert {k: v for k, v in call["cfg"].items() if k != "context_length"} == {
            k: v for k, v in saved_cfg.items() if k != "context_length"}, (
            f"context {c}: only context_length may differ from the checkpoint's config.json")
        state = call["state"]
        assert set(state) == set(on_disk), f"context {c}: state dict keys changed"
        for name, tensor in on_disk.items():
            want = tensor.to(torch.float32) if tensor.is_floating_point() else tensor
            assert state[name].dtype == want.dtype, (c, name, state[name].dtype)
            assert torch.equal(state[name], want), f"context {c}: tensor {name} changed at load"


def _edited_copy(src: Path, dst: Path, *, add: str | None = None, drop: bool = False) -> Path:
    import torch
    from safetensors.torch import load_file, save_file

    d = Path(shutil.copytree(src, dst))
    state = load_file(str(d / "weights.safetensors"))
    if add is not None:
        state[add] = torch.zeros(4, dtype=torch.float32)
    if drop:
        state.pop(sorted(state)[0])
    save_file(state, str(d / "weights.safetensors"))
    return d


@torch_mark
@pytest.mark.parametrize("context", [TRAINED_CONTEXT, W0B_CONTEXTS[-1]])
def test_checkpoint_with_an_extra_key_fails_strict(ckpts, tmp_path, context):
    d = _edited_copy(ckpts["bf16"], tmp_path / "extra", add="w0b_extra.weight")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _lane_d(d, [series(TRAINED_CONTEXT, seed=6)], context)


@torch_mark
@pytest.mark.parametrize("context", [TRAINED_CONTEXT, W0B_CONTEXTS[-1]])
def test_checkpoint_with_a_missing_key_fails_strict(ckpts, tmp_path, context):
    d = _edited_copy(ckpts["bf16"], tmp_path / "missing", drop=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        _lane_d(d, [series(TRAINED_CONTEXT, seed=6)], context)
