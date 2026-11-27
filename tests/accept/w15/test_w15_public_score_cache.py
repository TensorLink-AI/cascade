"""W15 B6 for the public half (owner 2026-11-27): the eval host's public
scores are cached like the live forecasts. Before, ``ScoreCache`` covered the
live forecasts only, so every retry of a settlement re-read and re-scored all
10,000 public windows for every digest.

Key: the same (host identity, code) namespace as the live rows, plus
``(digest, context, horizon)`` and the public draw (draws, block hash,
max_channels). A host change, another boundary or another draw misses. The
live-forecast ``retain`` never drops the public replies; ``retain_public``
keeps only the current settlement's.

W8c's public rig + W15's host hello: numpy only, no pod, no torch.
"""

from __future__ import annotations

from dataclasses import replace

from tests.accept.w7a import helpers as w7a
from tests.accept.w8c import helpers as w8c
from tests.accept.w8c import test_w8c_public as w8cp

from . import test_w15_score_cache as w15


def _rig(tmp_path, cache):
    rig, indexes = w8cp._rig(tmp_path)
    rig.cfg = replace(rig.cfg, lane_d=w7a.lane_d_cfg(public_weight=0.5,
                                                     window_ratio_floor=0.5))
    rig.eval_host.hello_doc = w15._host_hello()
    rig.source = rig.build(public_indexes=indexes, scoring_envs=w15._envs(), score_cache=cache)
    return rig


def _sc(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    c = ScoreCache(tmp_path / "score_cache")
    assert hasattr(c, "get_public") and hasattr(c, "put_public"), (
        "2026-11-27: the score cache has no public half (every retry redoes the public pass)")
    return c


def test_a_retry_on_the_same_host_reuses_the_public_scores(tmp_path):
    cache = _sc(tmp_path)
    rig = _rig(tmp_path, cache)
    first = rig.source(w8c.B1)
    n = len(rig.eval_host.public_calls)
    assert n >= 2, "the cohort is scored on the public half"
    again = rig.source(w8c.B1)
    assert len(rig.eval_host.public_calls) == n, "same host, same draw: no public re-score"
    assert again.public_scores == first.public_scores
    assert again.public_indexes == first.public_indexes


def test_a_host_change_rescores_the_public_half(tmp_path):
    cache = _sc(tmp_path)
    rig = _rig(tmp_path, cache)
    rig.source(w8c.B1)
    n = len(rig.eval_host.public_calls)
    rig.eval_host.hello_doc = w15._host_hello(uuid="GPU-2")
    rig.source(w8c.B1)
    assert len(rig.eval_host.public_calls) == 2 * n, "B6: a host change re-scores the cohort"
    assert len(list((tmp_path / "score_cache").iterdir())) == 1


def test_live_retain_keeps_public_replies_and_retain_public_prunes(tmp_path):
    c = _sc(tmp_path)
    from cascade.validator.lane_d_score_cache import public_key

    ns = w15._ns()
    reply = {"scores": {"rmisc": [{"window_id": "w", "family": "f", "crps": 1.0,
                                   "mase": 2.0}]}, "dropped": {"rmisc": []}}
    k1 = public_key([{"source": "rmisc"}], block_hash="0x1", max_channels=4)
    k2 = public_key([{"source": "rmisc"}], block_hash="0x2", max_channels=4)
    assert k1 != k2 and k1 != public_key([{"source": "rmisc"}], block_hash="0x1",
                                         max_channels=8)
    c.put_public(ns, w15.DG, context=4096, horizon=w15.H, key=k1, reply=reply)
    c.put_public(ns, w15.DG, context=4096, horizon=w15.H, key=k2, reply=reply)
    c.retain(ns, [])
    assert c.get_public(ns, w15.DG, context=4096, horizon=w15.H, key=k1) == reply, \
        "the live-forecast retain never drops public replies"
    c.retain_public(ns, [(w15.DG, 4096, w15.H, k2)])
    assert c.get_public(ns, w15.DG, context=4096, horizon=w15.H, key=k1) is None
    assert c.get_public(ns, w15.DG, context=4096, horizon=w15.H, key=k2) == reply
    assert c.get_public(ns, w15.DG, context=8192, horizon=w15.H, key=k2) is None
    c.put_public(ns, w15.DG, context=1, horizon=w15.H, key=k2, reply={"scores": []})
    assert c.get_public(ns, w15.DG, context=1, horizon=w15.H, key=k2) is None, \
        "a malformed reply is never cached"
