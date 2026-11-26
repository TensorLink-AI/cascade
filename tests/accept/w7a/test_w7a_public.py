"""W7a (plan rev6.18 A6; F3): the signed report's ``public`` block. Per
public source: the pinned revision, ``index_sha256``, ``files_sha256``, the
draw (seed on the settlement block hash, capacity, quota) and the drawn
window ids; the schema closes it and the cross-field rules bind it.
Contract: README.md "Public sources (rev6.18 F3)".
"""

from __future__ import annotations

import copy

import pytest

from . import pub

GIFT, RMISC = "gift_eval_pretrain", "rmisc"
REVISIONS = {GIFT: "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874",
             RMISC: "d0cdd935300ca9b3b2d6f23e6085db24cb7560c2"}


def test_the_report_carries_each_public_source_with_its_hashes_and_windows(tmp_path):
    run = pub.settle_pub(tmp_path)
    rep = run["s1"].report
    assert rep["public"]["schema"] == "lane-d-public/1"
    srcs = {s["source"]: s for s in rep["public"]["sources"]}
    assert list(srcs) == [GIFT, RMISC]
    from cascade.validator.lane_d_public import draw_public

    for src, s in srcs.items():
        man = run["mirrors"][src]["manifest"]
        assert s["revision"] == REVISIONS[src] == man["revision"]
        assert s["index_sha256"] == man["index_sha256"]
        assert s["files_sha256"] == man["files_sha256"]
        assert s["seed"] == f"{run['block_hash']}:lane_d_public:{src}"
        assert s["n_windows"] == 5000 and s["horizon"] == 64 and s["min_history"] == 128
        d = draw_public(run["indexes"][src], source=src, boundary_block_hash=run["block_hash"],
                        horizon=64)
        assert [w["window_id"] for w in s["windows"]] == [w["window_id"] for w in d["windows"]]
        assert s["windows"] and s["dropped"] == []
        assert set(s["windows"][0]) == {"window_id", "ordinal", "family", "freq", "cut", "h",
                                        "length"}
    assert rep["draw"]["boundary_block_hash"] == run["block_hash"]


def test_a_report_without_public_indexes_has_no_public_block(tmp_path):
    import inspect

    from cascade.validator.lane_d import settle_lane_d
    from tests.accept.w7a import mv

    assert "public_indexes" in inspect.signature(settle_lane_d).parameters
    run = mv.crowned_mv(tmp_path, channels={})
    assert "public" not in run["s1"].report


def _bad(report: dict, fn) -> dict:
    out = copy.deepcopy(report)
    fn(out["public"]["sources"][0])
    return out


@pytest.mark.parametrize("name", ["seed", "dup", "unsorted", "overlap", "cut", "quota",
                                  "capacity", "dropped", "revision", "extra"])
def test_the_public_block_is_closed_and_bound(tmp_path, name):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    rep = pub.settle_pub(tmp_path)["s1"].report
    validate_report(rep)

    def mutate(s):
        ws = s["windows"]
        if name == "seed":
            s["seed"] = "0x" + "11" * 32 + ":lane_d_public:" + s["source"]
        elif name == "dup":
            ws.insert(1, dict(ws[0]))
        elif name == "unsorted":
            ws[0], ws[1] = ws[1], ws[0]
        elif name == "overlap":
            w = dict(ws[0])
            w["cut"] += 1
            w["window_id"] = w["window_id"].rsplit(":", 2)[0] + f":{w['cut']}:h64"
            ws.append(w)
            ws.sort(key=lambda x: x["window_id"])
        elif name == "cut":
            ws[0]["cut"] = ws[0]["length"]
        elif name == "quota":
            k = next(iter(s["quota"]))
            s["quota"][k] = s["capacity"][k] + 1
        elif name == "capacity":
            k = next(iter(s["quota"]))
            s["quota"][k] = 0
        elif name == "dropped":
            s["dropped"] = [{"window_id": "nope", "reason": "public_no_scale"}]
        elif name == "revision":
            s["revision"] = "main"
        elif name == "extra":
            s["extra"] = 1

    with pytest.raises(ReportSchemaError):
        validate_report(_bad(rep, mutate))
