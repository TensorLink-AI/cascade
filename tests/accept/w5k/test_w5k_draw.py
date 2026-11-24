"""W5k (plan rev6.18 A1, A2, A4; F3): the per-settlement public draw, the
reading of the drawn series and their per-window scores. Contract:
README.md. The draw rule is re-stated here from the spec (never read back
from the producer) and compared with `cascade.validator.lane_d_public`.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect

import numpy as np
import pytest

from .f3_fixture import fx

GIFT, RMISC = "gift_eval_pretrain", "rmisc"
H = 64


def _pub():
    return importlib.import_module("cascade.validator.lane_d_public")


def _H(seed: str, kind: str, o: int, r: int) -> int:
    return int(hashlib.sha256(f"{seed}:{kind}:{o}:{r}".encode()).hexdigest(), 16)


def spec_quota(caps: dict, total: int) -> dict:
    """The W5j even split, re-stated: smallest capacity first (ties by key),
    scarce cells whole, the surplus re-split."""
    quota, rem, want = {k: 0 for k in caps}, dict(sorted(caps.items())), int(total)
    while rem and want > 0:
        share = max(1, want // len(rem))
        key = min(rem, key=lambda k: (rem[k], k))
        cap = rem.pop(key)
        quota[key] = min(cap, share if rem else want)
        want -= quota[key]
    return quota


def spec_draw(rows, *, source, block_hash, n, h=H, m=128, multi):
    seed = f"{block_hash}:lane_d_public:{source}"
    slots = {r["ordinal"]: max(0, (r["length"] - m) // h) for r in rows}
    cells = {}
    for r in rows:
        if slots[r["ordinal"]] >= 1:
            cells.setdefault(f"{r['family']}|{r['freq']}", []).append(r)
    caps = {k: (sum(slots[r["ordinal"]] for r in v) if multi else len(v)) for k, v in cells.items()}
    quota = spec_quota(caps, n)
    cuts = {}
    for k in sorted(cells):
        left, rnd = quota[k], 1
        while left > 0:
            pool = sorted((r for r in cells[k] if len(cuts.get(r["ordinal"], [])) == rnd - 1
                           and slots[r["ordinal"]] >= rnd),
                          key=lambda r: (_H(seed, "pick", r["ordinal"], rnd), r["ordinal"]))
            if not pool:
                break
            for r in pool[:left]:
                o, L = r["ordinal"], r["length"]
                if not multi:
                    c = m + _H(seed, "cut", o, 1) % (L - h - m + 1)
                else:
                    kk = slots[o]
                    off = _H(seed, "offset", o, 0) % (L - m - kk * h + 1)
                    free = [m + off + j * h for j in range(kk)
                            if m + off + j * h not in cuts.get(o, [])]
                    c = free[_H(seed, "slot", o, rnd) % len(free)]
                cuts.setdefault(o, []).append(c)
                left -= 1
            if not multi:
                break
            rnd += 1
    by = {r["ordinal"]: r for r in rows}
    ids = sorted(f"{source}:{by[o]['series_id']}:{c}:h{h}" for o, cs in cuts.items() for c in cs)
    return caps, quota, ids


def _rows(b):
    from cascade.eval.lane_d_public_index import iter_index

    return list(iter_index(b["index"]))


def _draw(b, source, block_hash=None, n=40):
    return _pub().draw_public(b["index"], source=source,
                              boundary_block_hash=block_hash or fx.BLOCK_HASH,
                              horizon=H, n_windows=n)


@pytest.mark.parametrize("source", [GIFT, RMISC])
def test_the_draw_is_the_spec_rule_and_bound_to_the_block_hash(tmp_path, source):
    pub = _pub()
    b = fx.build(tmp_path, source)
    d = _draw(b, source)
    caps, quota, ids = spec_draw(_rows(b), source=source, block_hash=fx.BLOCK_HASH, n=40,
                                 multi=source == RMISC)
    assert d["capacity"] == caps and d["quota"] == quota
    assert [w["window_id"] for w in d["windows"]] == ids
    assert d["seed"] == f"{fx.BLOCK_HASH}:lane_d_public:{source}"
    assert d == _draw(b, source)
    other = _draw(b, source, fx.OTHER_BLOCK_HASH)
    assert [w["window_id"] for w in other["windows"]] != ids
    assert pub.PUBLIC_MIN_HISTORY == d["min_history"] == 128 and d["horizon"] == H


def test_budget_is_5000_windows_per_source(tmp_path):
    pub = _pub()
    assert pub.PUBLIC_WINDOWS_PER_SOURCE == 5000
    assert inspect.signature(pub.draw_public).parameters["n_windows"].default == 5000
    b = fx.build(tmp_path, GIFT)
    d = pub.draw_public(b["index"], source=GIFT, boundary_block_hash=fx.BLOCK_HASH, horizon=H)
    assert d["n_windows"] == 5000
    assert len(d["windows"]) == sum(d["capacity"].values())          # all of a small source
    assert len(_draw(b, GIFT, n=10)["windows"]) == 10


def test_equal_share_per_family_and_frequency_reuses_the_live_rule(tmp_path):
    from cascade.validator.lane_d_draw import split_quota

    b = fx.build(tmp_path, GIFT)
    d = _draw(b, GIFT, n=20)
    assert d["quota"] == split_quota(d["capacity"], 20)
    # cmip6_1850 + cmip6_1855 are one family; scarce cells are taken whole
    assert "cmip6|6H" in d["capacity"] and d["capacity"]["cmip6|6H"] == 12
    assert not any(k.startswith("cmip6_") for k in d["capacity"])
    assert d["quota"]["scarce_hourly|H"] == 2
    assert d["quota"] == spec_quota(d["capacity"], 20)
    count = {}
    for w in d["windows"]:
        count[f"{w['family']}|{w['freq']}"] = count.get(f"{w['family']}|{w['freq']}", 0) + 1
    assert count == {k: v for k, v in d["quota"].items() if v}


def test_one_window_per_series_with_the_cut_inside_the_series(tmp_path):
    b = fx.build(tmp_path, GIFT)
    d = _draw(b, GIFT)
    sids = [w["series_id"] for w in d["windows"]]
    assert len(sids) == len(set(sids))
    for w in d["windows"]:
        assert 128 <= w["cut"] <= w["length"] - H
        assert w["window_id"] == f"{GIFT}:{w['series_id']}:{w['cut']}:h{H}"
    assert not any(w["dataset"] == "too_short" for w in d["windows"])
    assert len({w["cut"] for w in d["windows"]}) > 3                  # cuts vary


def test_small_rmisc_folders_give_several_non_overlapping_windows_per_series(tmp_path):
    b = fx.build(tmp_path, RMISC)
    d = _draw(b, RMISC)
    few = [w for w in d["windows"] if w["dataset"] == "FewLong"]
    assert len(few) == d["quota"]["FewLong|h"] > 2
    by = {}
    for w in few:
        by.setdefault(w["series_id"], []).append(w["cut"])
    assert set(by) == {"FewLong/s0", "FewLong/s1"}
    for cuts in by.values():
        cuts = sorted(cuts)
        assert all(b2 - a >= H for a, b2 in zip(cuts, cuts[1:], strict=False))
        assert len(cuts) >= 2


def test_reads_only_the_drawn_series(tmp_path, monkeypatch):
    pub = _pub()
    b = fx.build(tmp_path, GIFT)
    d = _draw(b, GIFT, n=6)
    seen = []
    orig = pub.read_series

    def spy(root, row, **kw):
        seen.append(row["series_id"])
        return orig(root, row, **kw)

    monkeypatch.setattr(pub, "read_series", spy)
    ws, dropped = pub.read_public_windows(d, index_dir=b["index"], data_root=b["root"],
                                          block_hash=fx.BLOCK_HASH, max_channels=32)
    assert sorted(seen) == sorted({w["series_id"] for w in d["windows"]})
    assert len(seen) == 6 < len(_rows(b))
    assert [w["window_id"] for w in ws] == [w["window_id"] for w in d["windows"]] and not dropped
    w0 = ws[0]
    src = next(x for x in d["windows"] if x["window_id"] == w0["window_id"])
    full = fx.series_values(src["dataset"], int(src["series_id"].rsplit("#", 1)[1]),
                            src["length"]).astype(np.float32).astype(np.float64)[0]
    assert np.array_equal(w0["history"], full[:src["cut"]])
    assert np.array_equal(w0["target"], full[src["cut"]:src["cut"] + H])


def _spec_pick(block_hash, series_id, candidates, cap):
    cand = sorted(candidates)
    if len(cand) <= cap:
        return cand
    seed = hashlib.sha256(f"{block_hash}{series_id}lane_d_channels".encode()).hexdigest()
    return sorted(sorted(cand, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode())
                         .hexdigest())[:cap])


def test_multichannel_windows_follow_the_f2_channel_rule(tmp_path):
    pub = _pub()
    b = fx.build(tmp_path, RMISC)
    d = _draw(b, RMISC)
    ws, _ = pub.read_public_windows(d, index_dir=b["index"], data_root=b["root"],
                                    block_hash=fx.BLOCK_HASH, max_channels=32)
    wide = [w for w in ws if w["series_id"].startswith("WideSensors/")]
    assert wide
    for w in wide:
        empty = [5, 6] if w["series_id"] == "WideSensors/w1" else []
        want = _spec_pick(fx.BLOCK_HASH, w["series_id"],
                          [c for c in range(40) if c not in empty], 32)
        assert w["channels"]["channels"] == want and w["channels"]["empty_channels"] == empty
        assert w["history"].shape[0] == 32 and w["target"].shape == (32, H)
    few = next(w for w in ws if w["series_id"] == "FewLong/s0")
    assert few["history"].shape[0] == 2 and few["channels"]["n_channels"] == 2


def test_unscorable_windows_are_dropped_for_every_model(tmp_path):
    pub = _pub()
    base = {"window_id": "gift_eval_pretrain:x/y#0:200:h64", "series_id": "x/y#0",
            "family": "x", "freq": "H", "cut": 200, "h": H}
    good = np.arange(1.0, 401.0)
    win, why = pub.build_window(good, base, block_hash=fx.BLOCK_HASH, max_channels=32)
    assert why is None and win["history"].shape == (200,) and win["target"].shape == (H,)
    a = good.copy()
    a[210] = np.nan
    assert pub.build_window(a, base, block_hash=fx.BLOCK_HASH,
                            max_channels=32) == (None, "public_nonfinite_target")
    a = good.copy()
    a[150] = np.nan                                       # finite suffix 49 < 128
    assert pub.build_window(a, base, block_hash=fx.BLOCK_HASH,
                            max_channels=32) == (None, "public_short_history")
    a = good.copy()
    a[200:264] = 0.0
    assert pub.build_window(a, base, block_hash=fx.BLOCK_HASH,
                            max_channels=32) == (None, "public_no_scale")
    a = np.full((3, 400), np.nan)
    assert pub.build_window(a, base, block_hash=fx.BLOCK_HASH,
                            max_channels=32) == (None, "public_no_channel")
    assert set(pub.DROP_REASONS) == {"public_nonfinite_target", "public_short_history",
                                     "public_no_channel", "public_no_scale"}


def _quantiles(target, spread):
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
    t = np.asarray(target, dtype=np.float64)
    return t[..., None] + spread * z


def test_public_windows_score_through_the_f2_per_window_api(tmp_path):
    pub = _pub()
    from cascade.eval.lane_d_scorer import score_window

    out = {}
    for source in (GIFT, RMISC):
        b = fx.build(tmp_path / source, source)
        d = _draw(b, source, n=8)
        ws, _ = pub.read_public_windows(d, index_dir=b["index"], data_root=b["root"],
                                        block_hash=fx.BLOCK_HASH, max_channels=32)
        out[source] = ws
    ws = out[GIFT] + out[RMISC]
    fc = {"king": {w["window_id"]: _quantiles(w["target"], 0.5) for w in ws},
          "chall": {w["window_id"]: _quantiles(w["target"] + 0.3, 0.5) for w in ws}}
    s = pub.score_public_windows(ws, fc)
    assert set(s) == {"king", "chall"} and set(s["king"]) == {GIFT, RMISC}
    for dg in s:
        for source in (GIFT, RMISC):
            got = s[dg][source]
            assert [x["window_id"] for x in got] == sorted(w["window_id"] for w in out[source])
            for x in got:
                w = next(y for y in ws if y["window_id"] == x["window_id"])
                c, m = score_window(fc[dg][w["window_id"]], w["target"], w["history"],
                                    w["seasonal_period"])
                assert (x["crps"], x["mase"]) == (c, m) and x["family"] == w["family"]
    assert s["king"][GIFT][0]["crps"] < s["chall"][GIFT][0]["crps"]
    del fc["chall"][ws[0]["window_id"]]
    with pytest.raises(ValueError):
        pub.score_public_windows(ws, fc)


def test_one_draw_serves_the_king_the_house_and_every_challenger():
    pub = _pub()
    params = set(inspect.signature(pub.draw_public).parameters)
    assert params == {"index_dir", "source", "boundary_block_hash", "horizon", "n_windows",
                      "min_history"}
