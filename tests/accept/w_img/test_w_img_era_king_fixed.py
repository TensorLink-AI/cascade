"""#934 (W-IMG trainer side): the era king pointer is FIXED for its seed epoch.

Testnet 2026-11-25: a mid-era trainer restart onto a new worker image made the
trainer drop the era's king entry (its digest no longer matched the new pin)
and RETRAIN the king under a new checkpoint pointer. Every validator then
rejected every later settlement of the epoch (``era_king_pointer_mismatch``)
and the lane B/C rows of those rounds were dropped with them.

The rule these tests pin:

* a king entry the validators already hold (published in a settlement, or
  carried into a later era of its seed epoch) is never dropped or retrained;
* an image-only repin keeps proving it with the image it was dispatched under
  (DEC-CA-0067), so it keeps publishing with the SAME pointer;
* a contract change that cannot prove it holds the settlement with a loud
  error instead of retraining; it takes effect at the next seed epoch;
* the normal rollover (one king leg per seed epoch, carried) is unchanged.

No torch: the trainer's fakes from ``tests/unit/test_rolling.py``.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from .helpers import IMG_X, IMG_Y, base_cfg


def _repin(cfg, image, **change):
    """``cfg`` with the v2 contract's worker pin (and any numerics ``change``)."""
    t = replace(cfg.training, train_image_digest=image, **change)
    v2c = getattr(t, "v2_contract", None)
    if v2c is not None:
        t = replace(t, v2_contract=replace(v2c, train_image_digest=image, **change))
    return replace(cfg, training=t)


def _setup(tmp_path, *, decoupled=True):
    from tests.unit.test_dec55_trainer_epochs import EpochOps, _v2
    from tests.unit.test_rolling import Clock, FakeClient, _sched

    v2cfg, v2_block, L = _v2(base_cfg(tmp_path / "cfg"))
    v2cfg = replace(v2cfg, v2=replace(
        v2cfg.v2, image_decoupled_from_block=v2_block if decoupled else 0,
        prior_train_image_digests=()))
    v2cfg = _repin(v2cfg, IMG_X)
    assert v2cfg.training.at_block(v2_block).train_image_digest == IMG_X
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    return sched, ops, clock, FakeClient(), v2cfg, v2_block, L


def _first_settlement(sched, ops, clock, client, v2_block):
    """Train the epoch's king and publish the era's first settlement."""
    from tests.unit.test_dec55_trainer_epochs import _join
    from tests.unit.test_rolling import EB, REF, _advance, _commit

    # one challenger, so the boundary has something to settle
    ops.commits.append(_commit("ALFA", REF["ALFA"], v2_block - 100))
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=v2_block - 100)
    sched.tick(client, v2_block + 5)
    _join(sched)
    _advance(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + EB + 3)
    _join(sched)
    king = sched.state.current.king()
    assert king is not None
    assert [m for m in ops.manifests if m.entry_for_role("king") is not None], \
        "fixture: the era's first settlement carries the king"
    return king, v2_block + EB + 3


def _restart(cfg, tmp_path, ops, clock, *, challenger="", block=0):
    """A trainer restart onto ``cfg`` (reloads era_state.json); optionally a
    fresh challenger ``challenger`` revealed at ``block`` so the next boundary
    has something to settle."""
    from tests.unit.test_rolling import REF, _commit, _sched

    sched, _ = _sched(cfg, tmp_path, clock, ops=ops)
    if challenger:
        ops.commits.append(_commit(challenger, REF[challenger], block))
        ops.queue().add(challenger, REF[challenger], reveal_block=block)
    return sched


def _pub(era) -> str:
    """The king pointer the trainer recorded as published (#934 field)."""
    return str(getattr(era, "king_pointer_published", None) or "")


def _img(era) -> str:
    """The image the era king leg was dispatched under (#934 field)."""
    return str(getattr(era, "king_image", None) or "")


def _walk(clock, ops, sched, client, last, n):
    """Tick ``n`` settlement intervals, a few ticks per interval."""
    from tests.unit.test_dec55_trainer_epochs import _join
    from tests.unit.test_rolling import EB, _advance

    for _ in range(n * 3):
        _advance(clock, ops, sched, client, from_block=last, to_block=last + EB // 3)
        _join(sched)
        last += EB // 3
    return last


def test_published_king_records_its_pointer_and_dispatch_image(tmp_path):
    sched, ops, clock, client, cfg, v2_block, _ = _setup(tmp_path)
    king, _ = _first_settlement(sched, ops, clock, client, v2_block)
    cur = sched.state.current
    assert _pub(cur) == king.trained_pointer
    assert _img(cur) == IMG_X
    # persisted: a restart reads both back
    again = _restart(cfg, tmp_path, ops, clock)
    assert _pub(again.state.current) == king.trained_pointer
    assert _img(again.state.current) == IMG_X


def test_restart_mid_epoch_with_a_new_image_keeps_the_era_king_pointer(tmp_path):

    sched, ops, clock, client, cfg, v2_block, _ = _setup(tmp_path)
    king, last = _first_settlement(sched, ops, clock, client, v2_block)
    kings, published = len(ops.king_calls), len(ops.manifests)
    # the operator repins the worker image and restarts the trainer mid-epoch
    # WITHOUT listing the old pin in [v2] prior_train_image_digests
    new = _repin(cfg, IMG_Y)
    ops.cfg = new
    sched = _restart(new, tmp_path, ops, clock, challenger="BRAV", block=last)
    _walk(clock, ops, sched, client, last, 4)
    assert len(ops.king_calls) == kings                          # never retrained
    assert sched.state.current.king().trained_pointer == king.trained_pointer
    later = [m.entry_for_role("king") for m in ops.manifests[published:]]
    later = [k for k in later if k is not None]
    assert later, "settlements keep publishing after the repin"
    for k in later:                                              # the SAME pointer
        assert k.trained_pointer == king.trained_pointer
        assert k.train_image_digest == IMG_X                     # proven with its own pin


def test_unprovable_contract_change_mid_epoch_holds_and_never_retrains(tmp_path, caplog):

    sched, ops, clock, client, cfg, v2_block, _ = _setup(tmp_path, decoupled=False)
    king, last = _first_settlement(sched, ops, clock, client, v2_block)
    kings, published = len(ops.king_calls), len(ops.manifests)
    new = _repin(cfg, IMG_Y)          # full-digest rule: the old king cannot prove
    ops.cfg = new
    sched = _restart(new, tmp_path, ops, clock, challenger="BRAV", block=last)
    with caplog.at_level(logging.ERROR, logger="cascade.trainer.rolling"):
        _walk(clock, ops, sched, client, last, 4)
    assert len(ops.king_calls) == kings                          # never retrained
    cur = sched.state.current
    assert cur.king() is not None and cur.king().trained_pointer == king.trained_pointer
    assert not [m for m in ops.manifests[published:]
                if (m.entry_for_role("king") or king).trained_pointer != king.trained_pointer]
    assert len(ops.manifests) == published                       # held, not a new pointer
    assert "CONTRACT CHANGED MID SEED EPOCH" in caplog.text
    assert sched.state.last_settled_boundary >= last             # the boundary clock moves


def test_contract_change_takes_effect_at_the_next_seed_epoch(tmp_path):
    from tests.unit.test_dec55_trainer_epochs import _join, _tick_era
    from tests.unit.test_rolling import _ptr

    sched, ops, clock, client, cfg, v2_block, L = _setup(tmp_path, decoupled=False)
    king, last = _first_settlement(sched, ops, clock, client, v2_block)
    first = sched.state.current.index
    new = _repin(cfg, IMG_Y)
    ops.cfg = new
    sched = _restart(new, tmp_path, ops, clock)
    # a king-only promotion opens a new seed epoch at first + 3
    ops.gens = (1, [(_ptr("KKING-first"), "")], first + 3)
    kings = len(ops.king_calls)
    for k in (1, 2):                  # the rest of the epoch: held, no retrain
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
        # only the NEXT epoch's leg may start (its pre-train window)
        assert all(c[1] == first + 3 for c in ops.king_calls[kings:])
        assert sched.state.current.king().trained_pointer == king.trained_pointer
    last = _tick_era(clock, ops, sched, client, v2_block + 3 * L, L, last=last)
    _join(sched)
    cur = sched.state.current
    assert cur.index == first + 3 and cur.epoch_start == first + 3
    assert len(ops.king_calls) == kings + 1                      # one leg, new epoch
    assert cur.king().trained_pointer != king.trained_pointer
    assert _img(cur) == IMG_Y                               # under the new pin


def test_normal_rollover_carries_the_king_and_its_published_pointer(tmp_path):
    from tests.unit.test_dec55_trainer_epochs import _tick_era

    sched, ops, clock, client, cfg, v2_block, L = _setup(tmp_path)
    king, last = _first_settlement(sched, ops, clock, client, v2_block)
    first = sched.state.current.index
    kings = len(ops.king_calls)
    for k in (1, 2):
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
        cur = sched.state.current
        assert cur.index == first + k
        assert cur.king().trained_pointer == king.trained_pointer
        assert _pub(cur) == king.trained_pointer
        assert _img(cur) == IMG_X
    assert len(ops.king_calls) == kings


def test_unpublished_first_era_king_still_retrains_on_a_repin(tmp_path):
    """Before any validator holds the pointer (the epoch's first era, nothing
    published yet) a stale king is still dropped and retrained — the
    rc7→rc8 behaviour (``test_legs_from_before_an_image_repin_…``)."""
    from cascade.trainer.rolling import RollingScheduler

    sched, ops, clock, client, cfg, v2_block, _ = _setup(tmp_path)
    sched.tick(client, v2_block + 5)
    from tests.unit.test_dec55_trainer_epochs import _join

    _join(sched)
    cur = sched.state.current
    assert cur.king() is not None and not _pub(cur)
    assert not RollingScheduler._in_epoch_tail(cur)
    fixed = getattr(sched, "_era_king_fixed", None)
    assert fixed is not None, "the #934 fixed-pointer predicate exists"
    assert not fixed(cur, cur.king())
