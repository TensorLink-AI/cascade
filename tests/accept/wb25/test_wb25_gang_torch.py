"""WB25 on real tiny pyramid legs (marked ``torch``: CI only, never the 7 GB
box). A 2-rank gang runs on CPU with the gloo backend (README "The CPU
seam"): ``cascade.trainer.pyramid_dp.gang_size`` is patched to return 2, and
the trainer, as on main, looks it up on the module when the leg starts.

Lane B: the 2-rank expert leg trains through the Gang, all-reduces and
shards only the selected rows, leaves the router, routing bias and every
frozen tensor byte-identical, and equals the 1-rank leg within a tight
tolerance on the selected rows. Lane A: legs are byte-for-byte main's
(oracle: main's two PATHS files, ``main_snapshot/``). README.md has the
contract.
"""

from __future__ import annotations

import importlib
import types
from collections import Counter

import pytest

from . import helpers as h

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

MERGE_MODULES = ["cascade.trainer.moe_model", "cascade.trainer.pyramid_model"]


def _one_thread():
    import torch

    torch.set_num_threads(1)


def _leg(path, contract, **kw):
    _one_thread()
    return h.train(path, contract, **kw)


def _two_rank(path, contract, **kw):
    """One leg with a forced 2-rank gang, recording on rank 0: the Gang's
    world, every all_reduce (shape, dtype, numel) and every
    assign_muon_owners call. A failure of the leg itself (main's CUDA-only
    Gang on a CPU runner) is reported as an assertion."""
    import torch.distributed as dist

    from cascade.trainer import pyramid_dp, pyramid_optimizer

    rec = types.SimpleNamespace(worlds=[], reduces=[], owners=[])
    real_gang = pyramid_dp.Gang

    class SpyGang(real_gang):
        def __init__(self, world, *args, **kwargs):
            rec.worlds.append(int(world))
            super().__init__(world, *args, **kwargs)

    real_reduce = dist.all_reduce

    def all_reduce(t, *args, **kwargs):
        rec.reduces.append((tuple(t.shape), t.dtype, t.numel()))
        return real_reduce(t, *args, **kwargs)

    real_owners = pyramid_optimizer.assign_muon_owners

    def assign_muon_owners(params, world):
        rec.owners.append(([tuple(p.shape) for p in params], int(world)))
        return real_owners(params, world)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pyramid_dp, "Gang", SpyGang)
        mp.setattr(pyramid_dp, "gang_size", lambda *a, **k: 2)
        mp.setattr(dist, "all_reduce", all_reduce)
        mp.setattr(pyramid_optimizer, "assign_muon_owners", assign_muon_owners)
        _one_thread()
        try:
            leg = h.train(path, contract, **kw)
        except Exception as e:  # noqa: BLE001 — reported as the assertion it is
            raise AssertionError(f"2-rank CPU leg failed: {type(e).__name__}: {e}") from e
        finally:
            _one_thread()
    assert rec.worlds == [2], f"the leg did not train on a 2-rank Gang (Gang worlds: {rec.worlds})"
    return leg, rec


# ── shared legs (module scope: each 2-rank leg spawns a helper process) ─────


@pytest.fixture(scope="module")
def contract():
    pytest.importorskip("torch")
    return h.tiny_contract()


@pytest.fixture(scope="module")
def king(tmp_path_factory, contract):
    return _leg(tmp_path_factory.mktemp("king"), contract).local_dir


@pytest.fixture(scope="module")
def selection(king):
    _one_thread()
    return h.selection_of(king)


@pytest.fixture(scope="module")
def one_rank(tmp_path_factory, contract, king, selection):
    return _leg(tmp_path_factory.mktemp("one"), contract, warm_start_dir=king,
                expert_selection=selection)


_CACHE: dict = {}


@pytest.fixture(scope="module")
def two_rank_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("two")


@pytest.fixture
def two_rank(two_rank_dir, contract, king, selection):
    """The shared 2-rank expert leg, built on first use INSIDE the test body
    (a callable, not a value), so on main its assertion is a test failure,
    not a fixture error. Cached only once it succeeds."""
    def get():
        if "leg" not in _CACHE:
            out = two_rank_dir / f"try{len(list(two_rank_dir.iterdir()))}"
            _CACHE["leg"] = _two_rank(out, contract, warm_start_dir=king,
                                      expert_selection=selection)
        return _CACHE["leg"]
    return get


# ── lane B on the gang ──────────────────────────────────────────────────────


def test_slice_leg_trains_on_the_gang(two_rank, one_rank):
    leg, _ = two_rank()
    assert leg.metrics.get("expert_training_backend") == "compact_rows"
    assert leg.metrics.get(h.GPU_N) == 2, f"{h.GPU_N}={leg.metrics.get(h.GPU_N)!r}"
    assert leg.metrics["steps"] == one_rank.metrics["steps"] == 3
    assert leg.metrics["expert_trainable_param_count"] == \
        one_rank.metrics["expert_trainable_param_count"]


def test_one_rank_slice_leg_records_gpu_n_1(one_rank):
    assert one_rank.metrics.get(h.GPU_N) == 1, f"{h.GPU_N}={one_rank.metrics.get(h.GPU_N)!r}"


def test_gang_all_reduces_only_selected_rows(two_rank, king, selection):
    """Every float all_reduce on rank 0, summed over the leg, is at least one
    and at most two copies of the selected leaves per step (the marginal and
    copula gradients) plus the two losses: never a full expert tensor, a
    router or any other frozen weight."""
    import torch

    leg, rec = two_rank()
    leaves = sum(int(torch.Size(s).numel())
                 for s in h.leaf_shapes(h.weights(king), selection, h.moe_names(king)))
    floats = sum(n for _, dtype, n in rec.reduces if dtype.is_floating_point)
    steps = leg.metrics["steps"]
    assert floats >= steps * leaves, (floats, steps, leaves)
    assert floats <= steps * (2 * leaves + 2), (floats, steps, leaves)
    full = {tuple(t.shape) for t in h.weights(king).values()}
    small = {s for s in full if torch.Size(s).numel() <= 2}
    reduced = {shape for shape, dtype, _ in rec.reduces if dtype.is_floating_point}
    assert not (reduced & (full - small)) - set(h.leaf_shapes(h.weights(king), selection,
                                                              h.moe_names(king))), \
        f"a frozen tensor's shape was all-reduced: {sorted(reduced & full)}"


def test_optimizer_sharded_over_selected_rows(two_rank, king, selection):
    """Rank 0 shards the muon step with assign_muon_owners over exactly the
    compact selected-row leaves, for world 2."""
    _, rec = two_rank()
    want = Counter(h.leaf_shapes(h.weights(king), selection, h.moe_names(king)))
    assert any(world == 2 and Counter(shapes) == want for shapes, world in rec.owners), \
        f"no assign_muon_owners(<selected leaves>, 2) call: {rec.owners}"


def test_router_and_bias_byte_identical_after_two_rank_leg(two_rank, king):
    import torch

    leg, _ = two_rank()
    ks, ls = h.weights(king), h.weights(leg.local_dir)
    keys = h.router_keys(ks)
    assert keys
    moved = [k for k in keys if not torch.equal(ks[k], ls[k])]
    assert not moved, moved


def test_two_rank_leg_matches_one_rank_leg(two_rank, one_rank, king, selection):
    """Selected rows: equal within RTOL/ATOL (float summation order only).
    Every other tensor and every unselected row: exactly equal, and equal to
    the king."""
    import torch

    leg, _ = two_rank()
    assert set(h.file_bytes(leg.local_dir)) == set(h.file_bytes(one_rank.local_dir))
    ks, one, two = h.weights(king), h.weights(one_rank.local_dir), h.weights(leg.local_dir)
    assert one.keys() == two.keys() == ks.keys()
    names = h.moe_names(king)
    owned = {n: li for li, pair in enumerate(names) for n in pair}
    moved = False
    for k in ks:
        if k not in owned:
            assert torch.equal(two[k], one[k]) and torch.equal(two[k], ks[k]), k
            continue
        rows = sorted(selection.get(owned[k], []))
        outside = torch.ones(ks[k].shape[0], dtype=torch.bool)
        outside[rows] = False
        assert torch.equal(two[k][outside], one[k][outside]), k
        assert torch.equal(two[k][outside], ks[k][outside]), k
        if rows:
            torch.testing.assert_close(two[k][rows], one[k][rows], rtol=h.RTOL, atol=h.ATOL)
            moved |= not torch.equal(two[k][rows], ks[k][rows])
    assert moved, "no selected row trained"
    assert leg.metrics["final_loss"] == pytest.approx(one_rank.metrics["final_loss"], rel=1e-5)


@pytest.mark.parametrize("module", MERGE_MODULES)
def test_merge_accepts_a_multi_gpu_leg(two_rank, king, selection, module):
    leg, _ = two_rank()
    merge = importlib.import_module(module).merge_expert_slices
    merge(h.weights(king), h.weights(leg.local_dir), selection, h.moe_names(king))


def test_two_rank_leg_is_deterministic(tmp_path, two_rank, contract, king, selection):
    leg, _ = two_rank()
    again, _ = _two_rank(tmp_path / "again", contract, warm_start_dir=king,
                         expert_selection=selection)
    assert h.file_bytes(again.local_dir) == h.file_bytes(leg.local_dir)
    assert h.numeric_metrics(again.metrics) == h.numeric_metrics(leg.metrics)


def test_gang_size_honours_cascade_train_gpus(monkeypatch):
    import torch

    from cascade.trainer.pyramid_dp import GANG_ENV, gang_size

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.delenv(GANG_ENV, raising=False)
    assert gang_size("cuda") == 8
    monkeypatch.setenv(GANG_ENV, "3")
    assert gang_size("cuda") == 3
    assert gang_size("cpu") == 1


# ── lane A: byte-for-byte main ──────────────────────────────────────────────


@pytest.fixture
def oracle():
    mod = h.load_main_oracle()
    yield mod
    h.unload_main_oracle()


def _direct(module, path, contract, **kw):
    """``module.train_pyramid`` with the arguments Toto2Trainer.train passes."""
    _one_thread()
    return module.train_pyramid(h.trainer(), iter(h.data(84)), contract, training_seed=1,
                                token_budget=contract.train_tokens, out_dir=path,
                                logger=None, continuation=None, **kw)


def _assert_same_leg(live, main, *, extra=()):
    assert h.file_bytes(live.local_dir) == h.file_bytes(main.local_dir)
    got = {k: v for k, v in h.numeric_metrics(live.metrics).items() if k not in extra}
    assert got == h.numeric_metrics(main.metrics)


def test_lane_a_random_init_leg_is_main_byte_for_byte(tmp_path, contract, oracle):
    from cascade.trainer import pyramid_trainer

    _assert_same_leg(_direct(pyramid_trainer, tmp_path / "live", contract),
                     _direct(oracle, tmp_path / "main", contract))


def test_lane_a_warm_start_leg_is_main_byte_for_byte(tmp_path, contract, king, oracle):
    from cascade.trainer import pyramid_trainer

    _assert_same_leg(_direct(pyramid_trainer, tmp_path / "live", contract, warm_start_dir=king),
                     _direct(oracle, tmp_path / "main", contract, warm_start_dir=king))


def test_one_rank_slice_leg_is_main_byte_for_byte(tmp_path, contract, king, selection, oracle):
    """A 1-GPU lane B leg is main's leg; only the gpu_n metric is new."""
    from cascade.trainer import pyramid_trainer

    kw = {"warm_start_dir": king, "expert_selection": selection}
    _assert_same_leg(_direct(pyramid_trainer, tmp_path / "live", contract, **kw),
                     _direct(oracle, tmp_path / "main", contract, **kw), extra=(h.GPU_N,))


def test_lane_a_metrics_gain_no_gpu_n(tmp_path, contract):
    leg = _leg(tmp_path / "a", contract)
    assert h.GPU_N not in leg.metrics


def test_lane_a_two_rank_leg_keeps_the_gang_path(tmp_path, contract):
    """Lane A on the same 2-rank CPU gang: the whole model is all-reduced
    (unchanged path) and the result equals the 1-rank leg within tolerance."""
    import torch

    one = _leg(tmp_path / "one", contract)
    two, rec = _two_rank(tmp_path / "two", contract)
    a, b = h.weights(one.local_dir), h.weights(two.local_dir)
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(b[k], a[k], rtol=h.RTOL, atol=h.ATOL)
    params = sum(t.numel() for t in a.values() if t.dtype.is_floating_point)
    floats = sum(n for _, dtype, n in rec.reduces if dtype.is_floating_point)
    assert floats > params, (floats, params)      # lane A reduces the full model
    assert h.GPU_N not in two.metrics
