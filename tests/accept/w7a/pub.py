"""F3 / F4 (plan rev6.18 A1/A2/A3/A6): a lane D settlement that also draws
the public sources, shared by the W5e, W7a and W9 acceptance tests.

It is F2's chain (`mv.build_mv_chain`, univariate here) settled with
``public_indexes`` = the two tiny public mirrors' indexes
(`tests/accept/w6p/fixtures.py`, reached lazily). `settle_pub` asserts that
`settle_lane_d` takes ``public_indexes`` before calling it, so on a main
without F3 every caller fails on an assertion.

F4 (rev6.18 A3): at ``public_weight > 0`` the settlement also takes the eval
host's per-window public scores (``public_scores``). `public_scores` builds
them from the drawn windows, deterministically: per window a base CRPS /
MASE (from the window id), times a per-digest per-source quality factor,
times a small per-(digest, window) jitter, so each model's ratio varies
around its factor. `settle_pub(..., public_weight=0.5, quality=...)` asserts
that `settle_lane_d` takes ``public_scores`` before calling it.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from . import helpers as w7a
from . import mv

GIFT, RMISC = "gift_eval_pretrain", "rmisc"


def _fx():
    try:
        from tests.accept.w6p import fixtures
    except ImportError:
        raise AssertionError("F3: tests/accept/w6p/fixtures.py (the public mirror fixtures) "
                             "is missing") from None
    return fixtures


def public_mirrors(tmp: Path) -> dict:
    fx = _fx()
    return {src: fx.build(Path(tmp) / src, src) for src in (fx.GIFT, fx.RMISC)}


# Index-only public sources (no pyarrow, no data files): the box draws from
# the index alone, and F4's scores come from `public_scores`, so a
# settlement test needs only the two W6p indexes. Families per source:
# GiftEvalPretrain "alpha" (12 series) and the year-split "cmip6_*" folders
# (one family, 10 series); RMISC "beta" (3 long series: several windows per
# series) and "gamma" (6 series).
INDEX_ONLY = {
    "gift_eval_pretrain": [("alpha", "H", 12, 600), ("cmip6_1850", "D", 5, 500),
                           ("cmip6_2000", "D", 5, 520)],
    "rmisc": [("beta", "H", 3, 2400), ("gamma", "15T", 6, 700)],
}


def index_only(tmp: Path) -> dict:
    """``{source: index dir}`` of two small W6p indexes (module docstring)."""
    from cascade.eval.lane_d_public_index import assign_ordinals, family_of, write_index

    out = {}
    for src, sets in INDEX_ONLY.items():
        rows = []
        for ds, freq, n, length in sets:
            for i in range(n):
                rows.append({"series_id": f"{ds}/{i}", "dataset": ds, "family": family_of(ds),
                             "freq": freq, "length": length, "n_channels": 1,
                             "file": f"{ds}/data.arrow", "rows": [[i, i + 1]]})
        files = [{"path": f"{ds}/data.arrow", "size": 1, "sha256": "0" * 64}
                 for ds, *_ in sets]
        d = Path(tmp) / src / "index"
        write_index(d, source=src, rows=assign_ordinals(rows), files=files)
        out[src] = d
    return out


def _unit(*parts) -> float:
    """A deterministic number in [0, 1) from ``parts``."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def public_scores(indexes: dict, block_hash: str, digests, quality: dict | None = None, *,
                  horizon: int = 64, jitter: float = 0.04, dropped: dict | None = None,
                  ) -> dict:
    """The eval host's ``public_scores`` for ``digests`` on the settlement's
    public draw: ``{"scores": {digest: {source: [{window_id, family, crps,
    mase}]}}, "dropped": {source: [...]}}``. ``quality[digest]`` is a factor
    (all sources) or ``{source: factor}`` (default 1.0: the bar's level)."""
    from cascade.validator.lane_d_public import draw_public

    quality = quality or {}
    dropped = dropped or {}
    out: dict[str, dict] = {dg: {} for dg in digests}
    for src in sorted(indexes):
        d = draw_public(indexes[src], source=src, boundary_block_hash=block_hash,
                        horizon=horizon)
        gone = {x["window_id"] for x in dropped.get(src, [])}
        for dg in digests:
            q = quality.get(dg, 1.0)
            f = float(q.get(src, 1.0) if isinstance(q, dict) else q)
            rows = []
            for w in d["windows"]:
                if w["window_id"] in gone:
                    continue
                base = 0.5 + _unit("base", w["window_id"])
                j = 1.0 + jitter * (2.0 * _unit("jit", dg, w["window_id"]) - 1.0)
                rows.append({"window_id": w["window_id"], "family": w["family"],
                             "crps": base * f * j, "mase": (1.0 + base) * f * j})
            out[dg][src] = rows
    return {"scores": out, "dropped": {src: list(dropped.get(src, [])) for src in sorted(indexes)}}


def settle_pub(tmp: Path, *, public_weight: float = 0.5, quality: dict | None = None,
               dropped: dict | None = None, scores_hook=None, mirrors: bool = True,
               window_ratio_floor: float = 0.5) -> dict:
    """One challenger settled at BOUNDARY_1 (it is crowned on W7a's chain
    when it is no worse on the public half) with both public sources drawn.
    ``public_weight > 0`` (F4; the default 0.5) also hands the settlement the
    public scores; ``scores_hook(public_scores)`` may edit them first.
    ``mirrors=False`` uses :func:`index_only` (no pyarrow) instead of the W6p
    mirror fixtures."""
    from cascade.validator.lane_d import settle_lane_d

    params = inspect.signature(settle_lane_d).parameters
    assert "public_indexes" in params, (
        "F3 (rev6.18 A1/A6): settle_lane_d(..., public_indexes=) draws the public sources "
        "and records them in the report's `public` block")
    tmp = Path(tmp)
    chain = mv.build_mv_chain(tmp / "chain", channels={})
    if mirrors:
        mirrors = public_mirrors(tmp / "public")
        indexes = {k: v["index"] for k, v in mirrors.items()}
    else:
        mirrors = None
        indexes = index_only(tmp / "public")
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    uid_of = {e["hotkey"]: w7a.CHALLENGER_UID}
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    kw = w7a.gate_inputs(w7a.BOUNDARY_1, chain["house"]["digest"])
    cfg = w7a.lane_d_cfg(public_weight=public_weight, window_ratio_floor=window_ratio_floor)
    extra = {}
    scores = None
    if public_weight > 0:
        assert "public_scores" in params, (
            "F4 (rev6.18 A3): settle_lane_d(..., public_scores=) takes the eval host's "
            "per-window scores of the public half")
        scores = public_scores(indexes, kw["boundary_block_hash"],
                               sorted(chain["forecasts"][1]), quality, dropped=dropped)
        if scores_hook is not None:
            scores_hook(scores)
        extra["public_scores"] = scores
    s1 = settle_lane_d(cfg=cfg, boundary=w7a.BOUNDARY_1, pool_dir=chain["pools"][1],
                       entries=[dict(e)], house=dict(chain["house"]),
                       forecasts=chain["forecasts"][1], uid_of=dict(uid_of), state_path=state,
                       signer=w7a.fixture_key("signer"), max_channels=chain["max_channels"],
                       public_indexes=indexes, **kw, **extra)
    return {"chain": chain, "mirrors": mirrors, "s1": s1, "block_hash": kw["boundary_block_hash"],
            "indexes": indexes, "cfg": cfg, "entry": e, "scores": scores, "state": state}
