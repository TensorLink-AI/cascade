"""W7a (plan rev6.18 A3 / A7 / A8 / D24, F4): the report's schema minor.

Every verdict carries its per-half, per-source and per-domain point gains
(`halves`, `sources`, `domains`); the `public` block carries the public rows
and their `rows_sha256`; `params` carries `public_weight` and
`window_ratio_floor`; the A10 `lowfreq` block and the backtest window kind /
bins are gone. Contract: README.md "Schema minor (rev6.18 F4)".
"""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as w7a
from . import pub


def _settled(tmp_path, chal_factor=0.9):
    assert hasattr(pub, "index_only"), "F4: the index-only public fixture is missing"
    probe = pub.settle_pub(tmp_path / "probe", mirrors=False)
    chal = probe["chain"]["digests"][0]
    run = pub.settle_pub(tmp_path / "run", mirrors=False, quality={chal: chal_factor})
    return run, chal


def test_verdicts_carry_per_half_per_source_per_domain_gains(tmp_path):
    run, chal = _settled(tmp_path)
    rep = run["s1"].report
    (v,) = rep["verdicts"]
    assert "lowfreq" not in v
    assert set(v["halves"]) == {"live", "public"}
    for half in v["halves"].values():
        assert set(half) == {"n", "gain"} and half["n"] >= 1
    assert set(v["sources"]) == {"live", "gift_eval_pretrain", "rmisc"}
    assert v["sources"]["live"] == v["halves"]["live"]
    assert sum(v["sources"][s]["n"] for s in ("gift_eval_pretrain", "rmisc")) \
        == v["halves"]["public"]["n"]
    live_dom = {k: d for k, d in v["domains"].items() if k.startswith("live|")}
    pub_dom = {k: d for k, d in v["domains"].items() if not k.startswith("live|")}
    assert sum(d["n"] for d in live_dom.values()) == v["halves"]["live"]["n"]
    assert sum(d["n"] for d in pub_dom.values()) == v["halves"]["public"]["n"]
    # the year-split cmip6_* folders are ONE family (A2)
    assert "gift_eval_pretrain|cmip6" in pub_dom and not any("cmip6_" in k for k in pub_dom)


def test_the_public_block_carries_its_rows_and_their_digest(tmp_path):
    from cascade.validator.lane_d_report import rows_sha256

    run, chal = _settled(tmp_path)
    rep = run["s1"].report
    p = rep["public"]
    assert set(p) >= {"schema", "sources", "rows", "rows_sha256"}
    assert p["rows_sha256"] == rows_sha256(p["rows"])
    kept = sorted(w["window_id"] for s in p["sources"] for w in s["windows"])
    judged = {rep["cohorts"][0]["king_digest"], *rep["cohorts"][0]["members"]}
    assert [(r["window_id"], r["digest"]) for r in p["rows"]] == \
        [(w, d) for w in kept for d in sorted(judged)]
    assert set(p["rows"][0]) == {"window_id", "digest", "crps", "mase"}


def test_params_carry_the_public_weight_and_the_clamp(tmp_path):
    run, _ = _settled(tmp_path)
    params = run["s1"].report["params"]
    assert params["public_weight"] == 0.5 and params["window_ratio_floor"] == 0.5
    assert "lowfreq_freqs" not in params


def test_the_schema_has_no_lowfreq_or_backtest_kind():
    schema = json.loads((w7a.REPO / "cascade/validator/schemas/lane_d_report_1.json")
                        .read_text(encoding="utf-8"))
    defs = schema["$defs"]
    assert "lowfreq" not in defs and "lowfreq_lcbs" not in defs
    assert defs["window_kind"]["enum"] == ["forward"]
    assert defs["window_bin"]["enum"] == ["forward"]
    assert {"halves", "sources", "domains"} <= set(defs["verdict"]["required"])
    assert {"rows", "rows_sha256"} <= set(defs["public"]["required"])


def _refused(report, code=None, field=None):
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as e:
        validate_report(report)
    if code is not None:
        assert e.value.code == code, e.value
    if field is not None:
        assert field in str(e.value), e.value


@pytest.mark.parametrize("how", ["row_value", "row_dropped", "no_public", "neg_public_cleared",
                                 "public_at_w0", "domain_count", "sources_live"])
def test_the_validator_binds_the_new_fields(tmp_path, how):
    from cascade.validator.lane_d_report import validate_report

    run, _ = _settled(tmp_path)
    rep = copy.deepcopy(run["s1"].report)
    validate_report(rep)
    v = rep["verdicts"][0]
    if how == "row_value":
        rep["public"]["rows"][0]["crps"] += 1.0
        return _refused(rep, "rows_sha256_mismatch", "public.rows_sha256")
    if how == "row_dropped":
        from cascade.validator.lane_d_report import rows_sha256

        rep["public"]["rows"] = rep["public"]["rows"][1:]
        rep["public"]["rows_sha256"] = rows_sha256(rep["public"]["rows"])
        return _refused(rep, "bad_value", "public.rows")
    if how == "no_public":
        del rep["public"]
        return _refused(rep, "missing_field", "public")
    if how == "neg_public_cleared":
        v["halves"]["public"]["gain"] = -0.01
        return _refused(rep, "bad_value", "cleared")
    if how == "public_at_w0":
        rep["params"]["public_weight"] = 0.0
        return _refused(rep)
    if how == "domain_count":
        k = next(iter(v["domains"]))
        v["domains"][k]["n"] += 1
        return _refused(rep, "bad_value", "domains")
    if how == "sources_live":
        v["sources"]["live"] = {"n": v["sources"]["live"]["n"], "gain": 0.123}
        return _refused(rep, "bad_value", "sources")
    raise KeyError(how)
