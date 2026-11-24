"""W8b x W13: when a lane D crowning settles, the production step leaves the
``laned_king_publish_request/1`` the king-publish ops job consumes
(``<state dir>/lane_d_publish_requests/<boundary>.json``: the boundary, the
crowned digest and, with W-S1 armed, the era's sentinel record for the new
king). Nothing heavy runs in the step: the weights never move there. The
request is written after the report is published, once (a re-call keeps it),
and a failure to write it never withholds the settlement or pay.

Real ``settle_lane_d`` runs on W7a's synthetic chain through the W8b rig
(contract: README.md); only the external services are fakes.
"""

from __future__ import annotations

import json

from . import helpers as h


def _request(rig, boundary: int):
    p = rig.state_dir / "lane_d_publish_requests" / f"{boundary}.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_a_crowning_leaves_one_publish_request(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=False)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    for b in (h.B1, h.B2, h.B3):
        rig.step(b)
    assert _request(rig, h.B1) is None and _request(rig, h.B3) is None
    req = _request(rig, h.B2)
    assert req == {"schema": "laned_king_publish_request/1", "boundary": h.B2,
                   "digest": rig.fx["x"], "sentinel": None}
    assert rig.published_report(h.B2)["crowned"] == rig.fx["x"]


def test_the_request_carries_the_era_sentinel_record_for_the_king(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=True)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    for b in (h.B1, h.B2):
        rig.step(b)
    req = _request(rig, h.B2)
    report = rig.published_report(h.B2)
    assert req is not None, "the crowning left no publish request"
    assert req["sentinel"]["king_digest"] == rig.fx["x"]
    assert req["sentinel"]["suite"] == report["gate"]["sentinel"] == h.era_sentinel(cfg, h.B2)
    from cascade.lane_d.publish import sentinel_entry

    entry = sentinel_entry(report, req["sentinel"])
    assert entry["suite"] == report["gate"]["sentinel"]


def test_a_re_call_keeps_the_request_and_a_write_failure_never_blocks(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=False)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    rig.step(h.B1)
    rig.step(h.B2)
    path = rig.state_dir / "lane_d_publish_requests" / f"{h.B2}.json"
    assert path.is_file(), "the crowning left no publish request"
    before = path.read_bytes()
    rig.restart()(h.B2)
    assert path.read_bytes() == before

    other = h.Rig(tmp_path / "rig2", cfg, h.fixture(tmp_path / "fx2"))
    (other.state_dir / "lane_d_publish_requests").write_text("not a directory")
    other.step(h.B1)
    s = other.step(h.B2)
    assert s is not None and other.published_report(h.B2)["crowned"] == other.fx["x"]
