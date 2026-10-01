"""Tests for scripts/soup_rounds.py — the offline cross-round model-soup harness.

Only the pure parts are tested (weight arithmetic, the checkpoint-dir round
trip, the greedy selection rule): they are numpy + safetensors, torch-free.
The scorer is injected, so the greedy rule is exercised with a hand-built
objective where the answer is known.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")

_SPEC = importlib.util.spec_from_file_location(
    "soup_rounds",
    Path(__file__).resolve().parents[2] / "scripts" / "soup_rounds.py",
)
soup = importlib.util.module_from_spec(_SPEC)
sys.modules["soup_rounds"] = soup
_SPEC.loader.exec_module(soup)


def _state(scale: float, *, dtype=np.float32, flag: int = 7) -> dict:
    return {
        "w": (np.arange(6, dtype=np.float64).reshape(2, 3) * scale).astype(dtype),
        "b": np.full((3,), scale, dtype=dtype),
        "steps": np.array([flag], dtype=np.int64),
    }


def test_uniform_average_is_the_mean_and_keeps_dtype():
    out = soup.average_states([_state(1.0), _state(3.0)])
    np.testing.assert_allclose(out["w"], _state(2.0)["w"])
    np.testing.assert_allclose(out["b"], np.full((3,), 2.0))
    assert out["w"].dtype == np.float32
    assert out["steps"].dtype == np.int64 and out["steps"][0] == 7


def test_weighted_average_normalises_weights():
    out = soup.average_states([_state(0.0), _state(4.0)], weights=[3.0, 1.0])
    np.testing.assert_allclose(out["b"], np.full((3,), 1.0))


def test_interpolate_endpoints_and_midpoint():
    a, b = _state(0.0), _state(2.0)
    np.testing.assert_allclose(soup.interpolate(a, b, 0.0)["b"], a["b"])
    np.testing.assert_allclose(soup.interpolate(a, b, 1.0)["b"], b["b"])
    np.testing.assert_allclose(soup.interpolate(a, b, 0.5)["b"], np.full((3,), 1.0))
    with pytest.raises(ValueError):
        soup.interpolate(a, b, 1.5)


def test_bf16_like_half_precision_averages_in_float64():
    a = _state(1.0, dtype=np.float16)
    b = _state(1.0 + 2**-9, dtype=np.float16)  # below fp16 resolution at 1.0
    out = soup.average_states([a, b])
    assert out["b"].dtype == np.float16
    assert np.isfinite(out["w"]).all()


def test_mismatched_tensor_sets_and_shapes_are_refused():
    a = _state(1.0)
    b = _state(1.0)
    del b["b"]
    with pytest.raises(ValueError, match="different tensor set"):
        soup.average_states([a, b])
    c = _state(1.0)
    c["w"] = c["w"].reshape(3, 2)
    with pytest.raises(ValueError, match="shape"):
        soup.average_states([a, c])
    d = _state(1.0, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        soup.average_states([a, d])


def test_differing_non_float_tensor_is_refused():
    with pytest.raises(ValueError, match="non-float"):
        soup.average_states([_state(1.0, flag=1), _state(1.0, flag=2)])


def test_bad_weights_are_refused():
    with pytest.raises(ValueError):
        soup.average_states([_state(1.0), _state(2.0)], weights=[1.0])
    with pytest.raises(ValueError):
        soup.average_states([_state(1.0), _state(2.0)], weights=[-1.0, 2.0])
    with pytest.raises(ValueError):
        soup.average_states([_state(1.0), _state(2.0)], weights=[0.0, 0.0])


def _write_ckpt(d: Path, state: dict, arch: dict | None = None) -> Path:
    from safetensors.numpy import save_file

    d.mkdir(parents=True)
    save_file(state, str(d / soup.WEIGHTS_FILE))
    (d / "config.json").write_text(json.dumps(
        {"arch": "toto2-4m", "toto2": arch or {"d_model": 8}, "quantile_levels": [0.5],
         "input_transform": "arcsinh_causal"}))
    (d / "model.py").write_text("# model\n")
    (d / "forecast_wrapper.py").write_text("# wrapper\n")
    return d


def test_write_soup_round_trips_and_copies_side_files(tmp_path):
    a = _write_ckpt(tmp_path / "a", _state(1.0))
    _write_ckpt(tmp_path / "b", _state(3.0))
    states = [soup.load_weights(a), soup.load_weights(tmp_path / "b")]
    digest = soup.write_soup(a, tmp_path / "soup", soup.average_states(states))
    assert len(digest) == 64
    back = soup.load_weights(tmp_path / "soup")
    np.testing.assert_allclose(back["b"], np.full((3,), 2.0))
    for name in soup.SIDE_FILES:
        assert (tmp_path / "soup" / name).read_text() == (a / name).read_text()
    assert not list((tmp_path / "soup").glob(".weights.safetensors.tmp-*"))


def test_check_same_arch_refuses_cross_size_soups(tmp_path):
    a = _write_ckpt(tmp_path / "a", _state(1.0), arch={"d_model": 8})
    b = _write_ckpt(tmp_path / "b", _state(1.0), arch={"d_model": 16})
    soup.check_same_arch([a, a])
    with pytest.raises(ValueError, match="different architecture"):
        soup.check_same_arch([a, b])


def test_greedy_soup_keeps_only_improving_members():
    # Objective: distance of the "b" vector's mean from 2.0 (lower is better).
    # Singles: x=1.0 (d=1), y=3.0 (d=1), z=10 (d=8). Best-first order ties
    # x/y by insertion; soup(x, y) hits 2.0 exactly (kept), adding z moves
    # away (rejected).
    states = {"x": _state(1.0), "y": _state(3.0), "z": _state(10.0)}

    def score(s):
        return abs(float(s["b"].mean()) - 2.0)

    singles = {k: score(v) for k, v in states.items()}
    g = soup.greedy_soup(singles, states, score)
    assert g.members == ["x", "y"]
    assert g.score == pytest.approx(0.0)
    kept = [t["kept"] for t in g.trace]
    assert kept == [True, True, False]
    assert g.trace[-1]["candidate"] == "z"


def test_greedy_soup_never_worse_than_best_single_and_honours_min_gain():
    states = {"x": _state(1.0), "y": _state(1.1)}

    def score(s):
        return float(s["b"].mean())  # lower is better: x alone is optimal

    singles = {k: score(v) for k, v in states.items()}
    g = soup.greedy_soup(singles, states, score)
    assert g.members == ["x"] and g.score == singles["x"]

    # A tiny improvement is rejected under a min_gain larger than it.
    states2 = {"x": _state(1.0), "y": _state(0.99)}
    singles2 = {k: score(v) for k, v in states2.items()}
    g2 = soup.greedy_soup(singles2, states2, score, min_gain=0.1)
    assert g2.members == ["y"]


def test_greedy_soup_requires_scores():
    with pytest.raises(ValueError):
        soup.greedy_soup({}, {}, lambda s: 0.0)
