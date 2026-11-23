"""W9 (plan rev6.10 (a), rev6.12 (a), W9 row): the audit re-derives the
embargo-aware eligibility from the snapshot, independently of producer code.

For every series of every cadence, `e = min(feeds[feed].embargo_hours,
[lane_d] embargo_hours)` (the `e` pinned in the snapshot's `feeds`, never the
catalog). A forward window is eligible iff, for every target point,
`first_seen + e > b` and `data_ts > b − e`, with `b = t_max + margin` (the
`bound` argument); S0 and the backfill rule use `b − e`. The refusal labels
keep their §1.3 meaning against the feed bound `b − e`. `e = 0` is the rev3
rule (test_w9_snapshot.py). A listed feed whose `embargo_hours` is absent,
not a number or negative fails closed (`unlisted_feed`, W5a row). With
`pool_dirs`, a snapshot whose feed `e` exceeds `[lane_d] embargo_hours`, or
that carries a non-zero `e` on a publish-ahead feed (a point whose data time
is after its first-seen time), FAILs.

README.md "rev6.11 / rev6.12" has the contract. On main `window_refusal`
reads no `embargo_hours` (assertion failures on the e > 0 cases) and
`check_lane_d(..., pool_dirs=...)` doesn't exist (an `assert`).
"""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from . import helpers as h
from .test_w9_snapshot import BOUND, _series

HOUR = 3600
E = 48.0


def _feeds(e=E, **over) -> dict:
    f = {"mirror_lag_hours": 1.0, "forecast_product": False, "embargo_hours": e}
    f.update(over)
    return {"fe": f}


def _refusal(rec, feeds):
    from cascade.audit.lane_d import window_refusal

    got = window_refusal(copy.deepcopy(rec), copy.deepcopy(feeds), BOUND, h.w7a.lane_d_cfg())
    return None if got is None else str(got)


# A target inside the embargo window (b − 48 h, b] (and on past b): data time
# from b − 24 h, first seen 30 min later, an S0 frontier at b − 30 h.
INSIDE = {"data_start": BOUND - 24 * HOUR, "seen_start": BOUND - 24 * HOUR + 1800,
          "feed": "fe", "frontier": ((BOUND - 30 * HOUR, BOUND - 31 * HOUR),)}

E_CASES = [
    ("inside the embargo: eligible", {}, None),
    ("first-seen at b − e: release == b fails", {"seen_start": BOUND - E * HOUR},
     "publish_ahead"),
    ("data time at b − e fails", {"data_start": BOUND - E * HOUR,
                                  "frontier": ((BOUND - 47 * HOUR, BOUND - 50 * HOUR),)},
     "late_mirror"),
    ("both clocks before b − e", {"data_start": BOUND - 50 * HOUR,
                                  "seen_start": BOUND - 50 * HOUR,
                                  "frontier": ((BOUND - 47 * HOUR, BOUND - 52 * HOUR),)},
     "before_margin"),
    ("no S0 record after b − e", {"frontier": ((BOUND - 49 * HOUR, BOUND - 50 * HOUR),)},
     "absent_from_s0"),
    ("S0 at b − e holds the target: backfill",
     {"frontier": ((BOUND - 30 * HOUR, BOUND - 20 * HOUR),)}, "backfill"),
    ("a post-submission target is eligible too",
     {"data_start": BOUND + HOUR, "seen_start": BOUND + 5400,
      "frontier": ((BOUND + 600, BOUND - HOUR),)}, None),
]


@pytest.mark.parametrize("what,over,want", E_CASES, ids=[c[0] for c in E_CASES])
def test_rev610_window_refusal_at_the_feed_bound(what, over, want):
    rec = _series(**{**INSIDE, **over})
    got = _refusal(rec, _feeds())
    assert got == want, f"{what}: {got!r} != {want!r}"


def test_rev610_the_same_target_at_e_zero_is_the_rev3_rule():
    """`e = 0` is the rev3 rule at bound `b`: the embargoed target fails both
    clocks there (data and first-seen before `b`)."""
    assert _refusal(_series(**INSIDE), _feeds(0.0)) == "before_margin"
    assert _refusal(_series(**INSIDE), _feeds(E)) is None


def test_rev610_an_e_above_the_config_is_clamped_to_it():
    """`e = min(feed, [lane_d] embargo_hours)`: a feed stating 60 h admits
    nothing the 48 h config refuses (first seen 50 h before `b`)."""
    rec = _series(**{**INSIDE, "data_start": BOUND - 24 * HOUR,
                     "seen_start": BOUND - 50 * HOUR})
    assert _refusal(rec, _feeds(60.0)) == "publish_ahead"


def test_rev610_a_shorter_feed_embargo_applies():
    """The catalog override can only shorten: at e = 12 h a target from
    b − 24 h fails the data clock."""
    rec = _series(**{**INSIDE, "frontier": ((BOUND - 10 * HOUR, BOUND - 30 * HOUR),)})
    assert _refusal(rec, _feeds(12.0)) == "before_margin"


@pytest.mark.parametrize("value", [None, "48", -1.0, float("nan"), True],
                         ids=["absent", "string", "negative", "nan", "bool"])
def test_rev610_a_listed_feed_without_a_valid_embargo_fails_closed(value):
    feeds = _feeds()
    if value is None:
        del feeds["fe"]["embargo_hours"]
    else:
        feeds["fe"]["embargo_hours"] = value
    assert _refusal(_series(**INSIDE), feeds) == "unlisted_feed"


def test_rev612_no_cadence_term_a_sub_hourly_series_uses_the_feed_e():
    """rev6.12 (a): the embargo applies to every cadence. A 1-min series in
    the embargo window is as eligible as an hourly one (no `freq` term)."""
    rec = _series(**INSIDE)
    rec["freq"] = "1min"
    rec["target_ts"] = [int(BOUND - 24 * HOUR + i * 60) for i in range(len(rec["target_ts"]))]
    rec["first_seen_ts"] = [t + 30 for t in rec["target_ts"]]
    rec["first_target_ts"], rec["end_ts"] = rec["target_ts"][0], rec["target_ts"][-1]
    assert _refusal(rec, _feeds()) is None


# ── check_lane_d(..., pool_dirs=...) on a snapshot's pinned e ──────────────

_CACHE: dict = {}


def _settled_on(tmp, name: str, edit) -> dict:
    """b1 settled on a copy of W7a's snapshot 1 (feeds pinned at 48 h) edited
    by `edit(meta)`; the report is genuine for that snapshot."""
    if name not in _CACHE:
        chain = h.pin_embargo(h.w7a.build_chain(tmp / f"chain_{name}"))
        pool = tmp / f"pool_{name}"
        shutil.copytree(chain["pools"][1], pool)
        meta = json.loads((pool / "metadata.json").read_text())
        edit(meta)
        (pool / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
        e = h.w7a.entry(chain["digests"][0], f"w9-{name}", coldkey=h.w7a.coldkey(f"w9-{name}"))
        cfg = h.w7a.lane_d_cfg()
        path = tmp / f"state_{name}" / "lane_d_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        s = h.w7a.settle(dict(chain, pools={1: pool, 2: chain["pools"][2]}), cfg, path,
                         h.signer(), boundary=h.BOUNDARY_1, snap=1, entries=[e],
                         uid_of={e["hotkey"]: h.CHALLENGER_UID})
        _CACHE[name] = {"pool": pool, "report": s.report, "cfg": cfg}
    return _CACHE[name]


def _over_config(meta):
    for f in meta["feeds"].values():
        f["embargo_hours"] = 60.0


def _publish_ahead(meta):
    """One point of the defect series is first seen BEFORE its data time: its
    feed is publish-ahead (W6fs), so only e = 0 protects it."""
    rec = next(s for s in meta["series"] if s["series_id"] == h.DEFECT_SERIES)
    rec["first_seen_ts"][-1] = int(rec["target_ts"][-1]) - 60


def test_rev610_honest_snapshot_with_its_pinned_e_passes(tmp_path_factory):
    run = _settled_on(tmp_path_factory.mktemp("w9-e-honest"), "honest", lambda meta: None)
    assert run["report"]["cohorts"], "fixture: b1 judges X"
    res = h.audit_snapshots(run["report"], run["cfg"], [run["pool"]])
    assert res.status == "PASS", res.detail


@pytest.mark.parametrize("name,edit", [("over_config", _over_config),
                                       ("publish_ahead", _publish_ahead)])
def test_rev610_a_snapshot_whose_e_the_rules_refuse_fails(tmp_path_factory, name, edit):
    run = _settled_on(tmp_path_factory.mktemp(f"w9-e-{name}"), name, edit)
    assert run["report"]["cohorts"], "fixture: b1 judges X"
    res = h.audit_snapshots(run["report"], run["cfg"], [run["pool"]])
    assert res.status == "FAIL", (
        f"{name}: check_lane_d {res.status} ({res.detail}): the audit must refuse an e above "
        "[lane_d] embargo_hours and a non-zero e on a publish-ahead feed")
