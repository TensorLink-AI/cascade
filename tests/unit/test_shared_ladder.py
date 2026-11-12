"""DEC-CA-0068 shared-series ladder, DECIDED ON CHAIN through the DEC-CA-0045 machinery.

* ``[eval] shared_ladder_windows / shared_ladder_from_block`` parse, round-trip,
  default off, and refuse a typed block without the count / the horizon ladder.
* The draw: ONE even-by-domain pick, the SAME series at every rung, equal rung
  sizes by construction, NaN-padded targets where a series is too short, every
  eligible series when the pool is short of the count; deterministic in seed;
  the per-rung ladder and :func:`horizon_draw` stay byte-identical (the pick
  was factored out, not changed).
* The scorer masks NaN-padded target steps (pinball + MASE) and is unchanged on
  finite targets.
* ``verdict_windows`` branches on the gate; the feature name hashes the rule;
  lock-in writes the next era start; a typed block is the override; one-way.
* The validator signals the segment beside the older ones (the mainnet note
  stays under 128 bytes with every segment locked), locks in, persists,
  restores and stamps receipts; the audit replays a stamped receipt under it.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cascade.shared.activation as A
from cascade.eval.crps import pinball_components
from cascade.eval.mase import mase
from cascade.shared.config import (
    DEFAULT_CHAIN_TOML,
    check_shared_ladder,
    load_chain_config,
    shared_ladder_active,
    shared_ladder_configured,
)
from cascade.validator.windows import (
    build_padded_windows,
    horizon_draw,
    ladder_windows,
    scored_ladder,
    shared_ladder,
    shared_ladder_windows,
    shared_ladder_windows_for_round,
)

REPO = Path(__file__).resolve().parents[2]
PRIMARY = "rolling-era-king"
B0 = 9046800 + 3600
HORIZONS = (64, 256, 720)


@pytest.fixture
def cfg():
    """The shipped chain.toml with the other on-chain votes disarmed, so each
    test arms exactly what it needs."""
    c = load_chain_config(REPO / "chain.toml")
    return replace(c, scoring=replace(c.scoring, forfeit_hotkeys=(), forfeit_from_block=0,
                                      forfeit_successor_hotkey="", margin_v2_from_block=0,
                                      win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                                      margin_warmup_blocks_v2=0),
                   v2=replace(c.v2, prelaunch_burn=0.0, launch_at_utc="", from_block=0),
                   eval=replace(c.eval, scored_horizons=HORIZONS, scored_from_block=B0,
                                shared_ladder_windows=40, shared_ladder_from_block=0))


def _typed(cfg):
    """Rolling eras already armed (the deployed shape): 3600 -> 900 grid at B0."""
    return replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                      epoch_blocks_prev=3600, epoch_activation_block=B0),
                   scoring=replace(cfg.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                                   cohort_maxt_increment_from_block=B0))


def _v(hk, stake):
    return A.ValidatorStake(hotkey=hk, stake=stake, permit=True, last_update=0)


def _fleet(*stakes):
    return [_v(f"v{i}", s) for i, s in enumerate(stakes, 1)]


class FakeChain:
    def __init__(self, validators, signals, *, block, hotkey="v1"):
        self.validators = list(validators)
        self.signals = dict(signals)
        self.block = block
        self.hotkey = hotkey
        self.written: list[str] = []

    def current_block(self):
        return self.block

    def validator_stakes(self, block=None):
        return list(self.validators)

    def read_plain_commitments(self, block=None):
        return dict(self.signals)

    def set_plain_commitment(self, payload):
        self.written.append(payload)
        self.signals[self.hotkey] = payload

    def hotkey_ss58(self):
        return self.hotkey


# ── config ─────────────────────────────────────────────────────────


def test_shipped_values_parse_and_default_off(tmp_path):
    shipped = load_chain_config(DEFAULT_CHAIN_TOML)
    assert shipped.eval.shared_ladder_windows == 4000          # the owner's count
    assert shipped.eval.shared_ladder_from_block == 0          # decided on chain
    assert shipped.eval.scored_horizons == HORIZONS
    assert shared_ladder_configured(shipped.eval)
    assert not shared_ladder_active(shipped.eval, 10_000_000)  # no block yet
    # The vote's name is pinned: a changed count or ladder is a FRESH vote.
    assert A.shared_ladder_feature_name(4000, HORIZONS) == "lad-d0b94833"
    assert A.shared_ladder_feature(shipped).name == "lad-d0b94833"

    src = re.sub(r"^shared_ladder_(windows|from_block)\s*=.*$", "",
                 DEFAULT_CHAIN_TOML.read_text(), flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(src)
    c = load_chain_config(p)
    assert c.eval.shared_ladder_windows == 0 and c.eval.shared_ladder_from_block == 0
    assert not shared_ladder_configured(c.eval)
    assert A.shared_ladder_feature(c) is None                  # nothing to vote on

    p.write_text(src.replace("\n[eval]\n", "\n[eval]\nshared_ladder_windows = 2000\n"
                             "shared_ladder_from_block = 9500400\n", 1))
    c = load_chain_config(p)
    assert c.eval.shared_ladder_windows == 2000
    assert c.eval.shared_ladder_from_block == 9500400
    assert A.shared_ladder_feature(c).typed_block == 9500400   # the owner override
    assert not A.shared_ladder_feature(c).decided_on_chain


@pytest.mark.parametrize("args", [
    (-1, 0, HORIZONS, 100),            # negative count
    (100, -5, HORIZONS, 100),          # negative block
    (0, 2000, HORIZONS, 100),          # block without a count
    (100, 2000, (), 100),              # block without a ladder
    (100, 2000, HORIZONS, 0),          # ladder gate unset
    (100, 50, HORIZONS, 100),          # before the ladder's own gate
])
def test_load_check_refuses_bad_shapes(args):
    with pytest.raises(ValueError, match="shared_ladder"):
        check_shared_ladder(*args)


def test_load_check_accepts_the_valid_shapes():
    check_shared_ladder(0, 0, (), 0)
    check_shared_ladder(4000, 0, HORIZONS, 100)
    check_shared_ladder(4000, 100, HORIZONS, 100)
    check_shared_ladder(4000, 7200, HORIZONS, 100)


def test_gate_needs_the_ladder_and_the_block():
    e = SimpleNamespace(scored_horizons=HORIZONS, scored_from_block=1000,
                        shared_ladder_windows=40, shared_ladder_from_block=5000)
    assert shared_ladder(e, 5000) == 40
    assert shared_ladder(e, 9000) == 40
    assert shared_ladder(e, 4999) == 0           # before the gate: per-rung draw
    assert shared_ladder(e, None) == 0
    assert shared_ladder(replace_ns(e, shared_ladder_from_block=0), 9000) == 0
    assert shared_ladder(replace_ns(e, shared_ladder_windows=0), 9000) == 0
    assert shared_ladder(replace_ns(e, scored_horizons=()), 9000) == 0
    assert scored_ladder(replace_ns(e, scored_horizons=()), 9000) == ()


def replace_ns(ns, **kw):
    d = dict(vars(ns))
    d.update(kw)
    return SimpleNamespace(**d)


# ── the draw ───────────────────────────────────────────────────────


def _series_set():
    """Four domains: long, mid, SHORT (320 steps — eligible at 64 and 256, not
    at 720 without padding) and a scarce long one."""
    rng = np.random.default_rng(7)
    series, metadata = [], []
    for dom, n, length in (("a", 30, 1500), ("b", 30, 900), ("c", 20, 320), ("d", 10, 2000)):
        for _ in range(n):
            series.append(rng.normal(size=length).cumsum())
            metadata.append({"domain": dom})
    return series, metadata


def _by_horizon(windows):
    out = {}
    for w in windows:
        h, sid = w.series_id.split("-", 1)
        out.setdefault(int(h[1:]), []).append(sid)
    return out


def test_shared_draw_scores_the_same_series_at_every_rung():
    series, metadata = _series_set()
    windows = shared_ladder_windows(series, metadata, horizons=HORIZONS, n_windows=40,
                                    context_length=4096, seed=3)
    by = _by_horizon(windows)
    assert set(by) == set(HORIZONS)
    assert by[64] == by[256] == by[720] and len(by[64]) == 40
    assert len(windows) == 40 * len(HORIZONS)
    for w in windows:
        h = int(w.series_id.split("-", 1)[0][1:])
        assert w.target.shape == (1, h)
        assert w.history.shape[-1] >= 64
    # deterministic in seed; a different seed draws differently
    again = shared_ladder_windows(series, metadata, horizons=HORIZONS, n_windows=40,
                                  context_length=4096, seed=3)
    assert [w.series_id for w in windows] == [w.series_id for w in again]
    other = shared_ladder_windows(series, metadata, horizons=HORIZONS, n_windows=40,
                                  context_length=4096, seed=4)
    assert [w.series_id for w in windows] != [w.series_id for w in other]
    # even by domain: the scarce domain (10) at capacity, the rest split evenly
    doms = {}
    for sid in by[64]:
        doms[metadata[int(sid[1:])]["domain"]] = doms.get(metadata[int(sid[1:])]["domain"], 0) + 1
    assert doms["d"] == 10 and sorted(doms[k] for k in "abc") == [10, 10, 10]


def test_short_series_are_nan_padded_only_at_the_rungs_they_cannot_fill():
    series, metadata = _series_set()
    windows = shared_ladder_windows(series, metadata, horizons=HORIZONS, n_windows=90,
                                    context_length=4096, seed=5)
    short = {f"s{i}" for i, m in enumerate(metadata) if m["domain"] == "c"}
    for w in windows:
        h, sid = w.series_id.split("-", 1)
        h = int(h[1:])
        padded = not np.isfinite(w.target).all()
        if sid in short and h == 720:
            # 320 steps: 64 of history, 256 observed target steps, 464 NaN
            assert padded
            assert w.history.shape == (1, 64)
            assert int(np.isfinite(w.target).sum()) == 256
            assert np.isfinite(w.target[0, :256]).all() and np.isnan(w.target[0, 256:]).all()
            # the observed tail IS the series' tail
            src = series[int(sid[1:])]
            assert np.array_equal(w.target[0, :256], src[64:])
            assert np.array_equal(w.history[0], src[:64])
        else:
            assert not padded, (w.series_id,)
    assert any(not np.isfinite(w.target).all() for w in windows)


def test_padded_cut_matches_the_plain_cut_for_long_series():
    from cascade.validator.windows import build_windows_from_series

    series, metadata = _series_set()
    long = [s for s, m in zip(series, metadata, strict=True) if m["domain"] != "c"]
    for h in HORIZONS:
        a = build_windows_from_series(long, context_length=4096, horizon=h)
        b = build_padded_windows(long, context_length=4096, horizon=h)
        assert len(a) == len(b) == len(long)
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x.history, y.history) and np.array_equal(x.target, y.target)
    # a series with no observed target step is skipped
    assert build_padded_windows([np.arange(64.0)], context_length=4096, horizon=720) == []
    assert len(build_padded_windows([np.arange(65.0)], context_length=4096, horizon=720)) == 1


def test_pool_short_of_the_count_draws_every_eligible_series(caplog):
    series, metadata = _series_set()
    with caplog.at_level("WARNING", logger="cascade.validator.windows"):
        windows = shared_ladder_windows(series, metadata, horizons=(64, 720), n_windows=4000,
                                        context_length=4096, seed=1)
    by = _by_horizon(windows)
    assert len(by[64]) == len(by[720]) == len(series)           # all 90, at both rungs
    assert any("every eligible series drawn" in r.message for r in caplog.records)


def test_shared_draw_raises_loudly_instead_of_falling_back():
    series, metadata = _series_set()
    with pytest.raises(ValueError, match="non-empty"):
        shared_ladder_windows(series, metadata, horizons=(), n_windows=8,
                              context_length=4096, seed=1)
    with pytest.raises(ValueError, match="positive"):
        shared_ladder_windows(series, metadata, horizons=(64,), n_windows=0,
                              context_length=4096, seed=1)
    with pytest.raises(ValueError, match="no eligible"):
        shared_ladder_windows([np.arange(10.0)], [{"domain": "x"}], horizons=(64,),
                              n_windows=8, context_length=4096, seed=1)


def test_per_rung_ladder_and_horizon_draw_are_unchanged():
    """The pick was factored out of ``horizon_draw``; these pins come from the
    pre-factoring code on the same inputs."""
    series, metadata = _series_set()
    ids = [w.series_id for w in ladder_windows(series, metadata, horizons=HORIZONS,
                                               n_windows=60, context_length=4096, seed=3)]
    by = _by_horizon(ids_to_windows(ids))
    assert len(by[64]) == len(by[256]) == len(by[720]) == 20
    # the 720 rung never holds a short ("c") series under the per-rung rule
    assert not any(metadata[int(s[1:])]["domain"] == "c" for s in by[720])
    one = horizon_draw(series, metadata, horizon=64, n_windows=12, context_length=4096, seed=9)
    assert [w.series_id for w in one] == [
        w.series_id for w in horizon_draw(series, metadata, horizon=64, n_windows=12,
                                          context_length=4096, seed=9)]
    assert len(one) == 12 and all(w.series_id.startswith("h64-s") for w in one)


def ids_to_windows(ids):
    return [SimpleNamespace(series_id=i) for i in ids]


# ── the scorer masks padded steps ──────────────────────────────────


def test_pinball_and_mase_mask_nan_padded_steps_and_keep_finite_paths():
    rng = np.random.default_rng(11)
    q = np.linspace(0.1, 0.9, 9)
    H, k = 720, 256
    obs = rng.normal(size=H)
    padded = obs.copy()
    padded[k:] = np.nan
    qp = np.sort(rng.normal(size=(9, 1, H)), axis=0)
    full_l, full_a = pinball_components(qp, obs[None, :], q)
    pad_l, pad_a = pinball_components(qp, padded[None, :], q)
    cut_l, cut_a = pinball_components(qp[:, :, :k], obs[None, :k], q)
    assert np.allclose(pad_l, cut_l) and np.allclose(pad_a, cut_a)
    assert np.isfinite(pad_l).all() and np.isfinite(pad_a).all()
    assert not np.allclose(pad_l, full_l)                       # the mask does something
    # finite targets: the unchanged path, bit-exact with the plain arithmetic
    diff = obs[None, None, :] - qp
    expect = np.maximum(q[:, None, None] * diff, (q[:, None, None] - 1.0) * diff).sum(-1).T
    assert np.array_equal(full_l, expect) and np.array_equal(full_a, np.abs(obs).sum()[None])

    point, hist = rng.normal(size=H), rng.normal(size=64)
    assert mase(point, padded, hist, 1) == pytest.approx(mase(point[:k], obs[:k], hist, 1))
    assert mase(point, obs, hist, 1) == float(np.mean(np.abs(obs - point)) /
                                            max(float(np.mean(np.abs(np.diff(hist)))), 1e-9))
    with pytest.raises(ValueError, match="finite"):
        mase(point, np.full(H, np.nan), hist, 1)


def test_window_scorer_scores_a_padded_window_on_its_observed_steps():
    from cascade.eval.scoring import (
        score_forecaster_on_windows,
        score_joint_quantile_forecaster_on_windows,
    )
    from cascade.eval.window import EvalWindow

    rng = np.random.default_rng(3)
    q = tuple(np.linspace(0.1, 0.9, 9))
    hist = rng.normal(size=64).cumsum()
    tail = rng.normal(size=256).cumsum()
    target = np.full(720, np.nan)
    target[:256] = tail
    padded = EvalWindow("h720-s1", hist, target)
    cut = EvalWindow("h256-s1", hist, tail)

    def quantiles(history, horizon):
        base = np.linspace(-1, 1, 9)[None, None, :]
        return np.broadcast_to(base, (1, horizon, 9)).copy()

    def samples(history, horizon, n):
        return np.broadcast_to(np.linspace(-1, 1, n)[None, :, None], (1, n, horizon)).copy()

    for scorer in (lambda w: score_joint_quantile_forecaster_on_windows(quantiles, [w], q),
                   lambda w: score_forecaster_on_windows(samples, [w], 9, q)):
        a, b = scorer(padded)[0], scorer(cut)[0]
        assert np.isfinite(a.qloss_per_q).all() and np.isfinite(a.mase)
        assert np.allclose(a.qloss_per_q, b.qloss_per_q)
        assert a.abs_target == pytest.approx(b.abs_target)
        assert a.mase == pytest.approx(b.mase)


# ── verdict_windows branches on the gate ───────────────────────────


def _pool_dir(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    series, metadata = _series_set()
    md = {}
    for i, (s, m) in enumerate(zip(series, metadata, strict=True)):
        np.save(d / f"s{i}.npy", s)
        md[f"s{i}"] = m
    (d / "metadata.json").write_text(json.dumps(md))
    return d


def test_verdict_windows_branches_on_the_gate(tmp_path, cfg):
    from cascade.validator.expert_lane import verdict_windows
    from cascade.validator.pool import window_source_from_dir

    src = window_source_from_dir(_pool_dir(tmp_path), cfg, label="test")
    c = replace(cfg, eval=replace(cfg.eval, scored_horizons=(64, 720), scored_from_block=1000,
                                  n_windows=16, shared_ladder_windows=12,
                                  shared_ladder_from_block=5000))
    pre = verdict_windows(c, src, 12345, block=4999)          # the per-rung ladder
    by = _by_horizon(pre)
    assert len(by[64]) == len(by[720]) == 8 and by[64] != by[720]
    post = verdict_windows(c, src, 12345, block=5000)         # the shared draw
    by = _by_horizon(post)
    assert len(by[64]) == len(by[720]) == 12 and by[64] == by[720]
    assert [w.series_id for w in post] == [
        w.series_id for w in shared_ladder_windows_for_round(
            src, horizons=(64, 720), n_windows=12, context_length=4096,
            round_seed=12345, block=5000)]
    # the budget (n_windows / the v2 budget) no longer sizes the ladder
    bigger = replace(c, eval=replace(c.eval, n_windows=4000))
    assert len(verdict_windows(bigger, src, 12345, block=5000)) == 24


def test_shared_windows_for_round_errors_without_a_series_dir():
    with pytest.raises(ValueError, match="series directory"):
        shared_ladder_windows_for_round(object(), horizons=(64,), n_windows=8,
                                        context_length=4096, round_seed=1, block=None)


# ── the feature ────────────────────────────────────────────────────


def test_feature_name_hashes_the_rule(cfg):
    a = A.shared_ladder_feature_name(40, HORIZONS)
    assert a.startswith("lad-") and len(a) == 12
    assert A.shared_ladder_feature_name(40, HORIZONS) == a
    assert A.shared_ladder_feature_name(41, HORIZONS) != a
    assert A.shared_ladder_feature_name(40, (64, 256)) != a
    assert A.shared_ladder_feature_name(0, HORIZONS) == ""
    assert A.shared_ladder_feature_name(40, ()) == ""
    spec = A.shared_ladder_feature(cfg)
    assert spec is not None and spec.decided_on_chain and spec.name == a and spec.align == "era"
    assert A.shared_ladder_feature(replace(cfg, activation=replace(cfg.activation, feature=""))) is None
    assert A.shared_ladder_feature(replace(cfg, eval=replace(cfg.eval, shared_ladder_windows=0))) is None
    typed = replace(cfg, eval=replace(cfg.eval, shared_ladder_from_block=B0 + 3600))
    assert A.shared_ladder_feature(typed).typed_block == B0 + 3600
    assert not A.shared_ladder_feature(typed).decided_on_chain


def test_apply_writes_an_era_start_is_one_way_and_respects_the_typed_override(cfg):
    c = _typed(cfg)
    c1 = A.apply_shared_ladder_activation(c, B0 + 3600)
    assert c1.eval.shared_ladder_from_block == B0 + 3600
    assert c1.activation.resolved_shared_ladder_block == B0 + 3600
    assert A.configured_shared_ladder_block(c1) == 0 and A.resolved_shared_ladder_block(c1) == B0 + 3600
    assert shared_ladder_active(c1.eval, B0 + 3600) and not shared_ladder_active(c1.eval, B0 + 2700)
    assert A.apply_shared_ladder_activation(c1, B0 + 3600) is c1                  # idempotent
    with pytest.raises(ValueError, match="one-way"):
        A.apply_shared_ladder_activation(c1, B0 + 7200)
    with pytest.raises(ValueError, match="era start"):
        A.apply_shared_ladder_activation(c, B0 + 3600 + 17)
    with pytest.raises(ValueError, match="era start"):            # a settlement, not an era
        A.apply_shared_ladder_activation(c, B0 + 3600 + 900)
    typed = replace(c, eval=replace(c.eval, shared_ladder_from_block=B0 + 900))
    assert A.apply_shared_ladder_activation(typed, B0 + 3600) is typed          # the override wins
    assert A.configured_shared_ladder_block(typed) == B0 + 900
    off = replace(c, eval=replace(c.eval, shared_ladder_windows=0))
    assert A.apply_shared_ladder_activation(off, B0 + 3600).eval.shared_ladder_from_block == 0


def test_resolver_locks_in_and_the_draw_applies_from_the_next_era_start(cfg):
    c = _typed(cfg)
    spec = A.shared_ladder_feature(c)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3600 + 5)
    res = A.resolve_activation(c, chain, now_block=B0 + 3600 + 5, record=A.ActivationRecord(),
                               feature=spec)
    assert res.record.locked and res.record.feature == spec.name
    # a lock-in ON an era start takes the FOLLOWING one: the crossing era
    # finishes on the per-rung draw
    assert res.record.lock_block == B0 + 3600 and res.record.activation_block == B0 + 7200
    armed = A.apply_shared_ladder_activation(c, res.record.activation_block)
    for settlement in (B0 + 3600, B0 + 4500, B0 + 5400, B0 + 6300):
        assert shared_ladder(armed.eval, settlement) == 0
    assert shared_ladder(armed.eval, B0 + 7200) == 40
    mid = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 5400 + 5)
    res_mid = A.resolve_activation(c, mid, now_block=B0 + 5400 + 5, record=A.ActivationRecord(),
                                   feature=spec)
    assert res_mid.record.lock_block == B0 + 5400 and res_mid.record.activation_block == B0 + 7200
    chain2 = FakeChain(_fleet(40, 60), {"v1": note}, block=B0 + 3600 + 5)
    assert not A.resolve_activation(c, chain2, now_block=B0 + 3600 + 5,
                                    record=A.ActivationRecord(), feature=spec).record.locked
    assert A.record_for(c, A.ActivationRecord(feature="lad-deadbeef", lock_block=1,
                                              activation_block=2), spec) == A.ActivationRecord()


def test_note_carries_the_segment_and_stays_under_the_budget_with_every_mainnet_segment_locked():
    shipped = load_chain_config(DEFAULT_CHAIN_TOML)
    assert shipped.activation.enabled
    lad = A.shared_ladder_feature(shipped)
    v2b = A.image_decoupled_feature(shipped)
    plb = A.prelaunch_burn_feature(shipped)
    assert lad is not None and v2b is not None and plb is not None   # the live note's segments
    # unlocked: the fresh release's note
    note = A.own_signal_payload(shipped, A.ActivationRecord())
    assert note is not None and lad.name in note.split(":")
    assert len(note.encode()) <= A.NOTE_MAX_BYTES
    # every segment locked at 7-digit blocks (any lock-in before block
    # 10,000,000 ≈ January 2027): still one note, nothing trimmed
    big = 9_999_999
    locked = {f: A.ActivationRecord(feature=f.name, lock_block=big, activation_block=big,
                                    source="tally") for f in (plb, v2b, lad)}
    note = A.own_signal_payload(shipped, A.ActivationRecord(), None, None, locked[plb], None, None,
                                locked[v2b], shared_ladder_record=locked[lad])
    assert lad.name in note.split(":") and len(note.encode()) <= A.NOTE_MAX_BYTES
    assert A.note_trimmed_segments(shipped, A.ActivationRecord(), None, None, locked[plb], None,
                                   None, locked[v2b], locked[lad]) == []
    # and the v2b startup guard is untouched by the new segment
    assert A.v2b_note_problem(shipped, A.ActivationRecord()) == ""
    # trimming order: the NEWEST (ours) goes first, never the older votes
    cramped = A.own_signal_payload(shipped, A.ActivationRecord(), None, None, locked[plb],
                                   A.ReadySignal("cfg-ffffffff", lock_block=big, activation_block=big),
                                   None, locked[v2b], shared_ladder_record=locked[lad])
    assert v2b.name in cramped.split(":") and plb.name in cramped.split(":")


def test_summary_reports_the_block(cfg):
    c = _typed(cfg)
    rec = A.ActivationRecord(feature=A.shared_ladder_feature(c).name, lock_block=B0 + 3600,
                             activation_block=B0 + 7200, source="tally")
    s = A.summary(c, A.ActivationRecord(), None, shared_ladder_record=rec)
    assert s["shared_ladder"]["activation_block"] == B0 + 7200
    assert s["shared_ladder"]["shared_ladder_windows"] == 40
    assert s["shared_ladder"]["scored_horizons"] == list(HORIZONS)
    assert "shared_ladder" not in A.summary(
        replace(c, eval=replace(c.eval, shared_ladder_windows=0)), A.ActivationRecord(), None)


# ── validator runner ───────────────────────────────────────────────


def test_validator_runner_decides_on_chain_persists_restores_and_stamps(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    c = _typed(cfg)
    spec = A.shared_ladder_feature(c)
    runner = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    assert chain.written[-1] == (f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:"
                                 f"{B0 + 3600}:{B0 + 7200}")
    assert runner._shared_ladder.locked
    assert runner.cfg.eval.shared_ladder_from_block == B0 + 7200
    assert runner.shared_ladder_block == B0 + 7200                  # stamped on receipts
    assert runner.activation_block == 0                              # the typed rollover records nothing
    assert shared_ladder(runner.cfg.eval, B0 + 6300) == 0
    assert shared_ladder(runner.cfg.eval, B0 + 7200) == 40
    saved = A.ActivationStore(tmp_path / "activation_shared_ladder_state.json").load()
    assert saved.activation_block == B0 + 7200
    again = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = SimpleNamespace(current_block=lambda: (_ for _ in ()).throw(RuntimeError("down")),
                           hotkey_ss58=lambda: "v1")
    again._activation_startup(dead)
    assert again.cfg.eval.shared_ladder_from_block == B0 + 7200


def test_validator_runner_is_silent_on_a_typed_block(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    c = _typed(cfg)
    c = replace(c, eval=replace(c.eval, shared_ladder_from_block=B0 + 3600))
    runner = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1)
    runner._activation_startup(chain)
    assert chain.written == []
    assert runner.shared_ladder_block == 0                           # typed ⇒ nothing stamped
    assert shared_ladder(runner.cfg.eval, B0 + 3600) == 40


# ── receipts + audit ───────────────────────────────────────────────


def test_receipt_field_is_drop_when_default_and_round_trips():
    from cascade.shared import receipt as R

    base = R.RoundReceipt(round_id="1", status="rejected", epoch_start_block=B0,
                          epoch_block_hash="0x", base_seed=1, generation_seed=1, training_seed=1,
                          manifest={}, participants=(), validator_hotkey="v1", reject_reason="x")
    assert b"shared_ladder_block" not in base.canonical_body()      # old signatures stay valid
    stamped = replace(base, shared_ladder_block=B0 + 7200)
    assert b'"shared_ladder_block":' + str(B0 + 7200).encode() in stamped.canonical_body()
    assert R.load_receipt(R.dump_receipt(stamped)).shared_ladder_block == B0 + 7200
    assert R.load_receipt(R.dump_receipt(base)).shared_ladder_block == 0
    assert R.summarize_receipt(stamped)["shared_ladder_block"] == B0 + 7200


def test_audit_replays_a_stamped_receipt_under_the_shared_draw(cfg):
    from cascade.audit.checks import check_shared_ladder as audit_check

    c = _typed(cfg)
    receipt = SimpleNamespace(shared_ladder_block=B0 + 7200, epoch_start_block=B0 + 7200)
    replayed = A.apply_receipt_activation(c, receipt)
    assert replayed.eval.shared_ladder_from_block == B0 + 7200
    assert shared_ladder(replayed.eval, B0 + 7200) == 40
    assert audit_check(receipt, replayed, None).status == "WARN"    # agreement unverified
    assert audit_check(SimpleNamespace(shared_ladder_block=0), c, None).status == "PASS"
    # a block the config cannot carry fails
    wrong = SimpleNamespace(shared_ladder_block=B0 + 7200 + 17)
    assert audit_check(wrong, c, None).status == "FAIL"
    # agreement in the notes passes
    spec = A.shared_ladder_feature(c)
    note = A.format_signals(A.ReadySignal(PRIMARY),
                            A.ReadySignal(spec.name, lock_block=B0 + 3600,
                                          activation_block=B0 + 7200))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 7300)
    assert audit_check(receipt, replayed, chain).status == "PASS"
    # a typed block in the loaded config wins over the receipt
    typed = replace(c, eval=replace(c.eval, shared_ladder_from_block=B0 + 3600))
    assert A.apply_receipt_activation(typed, receipt) is typed
    assert audit_check(receipt, typed, None).status == "FAIL"


def test_netconfig_effective_raw_writes_the_resolved_block(cfg):
    from cascade.shared.netconfig import effective_raw

    c = A.apply_shared_ladder_activation(_typed(cfg), B0 + 3600)
    assert effective_raw(c)["eval"]["shared_ladder_from_block"] == B0 + 3600
    assert "shared_ladder_from_block" not in effective_raw(_typed(cfg)).get("eval", {}) or \
        effective_raw(_typed(cfg))["eval"]["shared_ladder_from_block"] == 0
