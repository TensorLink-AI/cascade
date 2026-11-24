"""F3 (plan rev6.18 A1/A2/A6): a lane D settlement that also draws the
public sources, shared by the W7a and W9 acceptance tests.

It is F2's chain (`mv.build_mv_chain`, univariate here) settled with
``public_indexes`` = the two tiny public mirrors' indexes
(`tests/accept/w6p/fixtures.py`, reached lazily). `settle_pub` asserts that
`settle_lane_d` takes ``public_indexes`` before calling it, so on a main
without F3 every caller fails on an assertion.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from . import helpers as w7a
from . import mv


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


def settle_pub(tmp: Path) -> dict:
    """One challenger settled at BOUNDARY_1 (it is crowned, as on W7a's
    chain) with both public sources drawn."""
    from cascade.validator.lane_d import settle_lane_d

    assert "public_indexes" in inspect.signature(settle_lane_d).parameters, (
        "F3 (rev6.18 A1/A6): settle_lane_d(..., public_indexes=) draws the public sources "
        "and records them in the report's `public` block")
    tmp = Path(tmp)
    chain = mv.build_mv_chain(tmp / "chain", channels={})
    mirrors = public_mirrors(tmp / "public")
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    uid_of = {e["hotkey"]: w7a.CHALLENGER_UID}
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    kw = w7a.gate_inputs(w7a.BOUNDARY_1, chain["house"]["digest"])
    s1 = settle_lane_d(**w7a._rev614_ts(settle_lane_d, w7a.BOUNDARY_1), cfg=w7a.lane_d_cfg(), boundary=w7a.BOUNDARY_1, pool_dir=chain["pools"][1],
                       entries=[dict(e)], house=dict(chain["house"]),
                       forecasts=chain["forecasts"][1], uid_of=dict(uid_of), state_path=state,
                       signer=w7a.fixture_key("signer"), max_channels=chain["max_channels"],
                       public_indexes={k: v["index"] for k, v in mirrors.items()}, **kw)
    return {"chain": chain, "mirrors": mirrors, "s1": s1, "block_hash": kw["boundary_block_hash"],
            "indexes": {k: v["index"] for k, v in mirrors.items()}}
