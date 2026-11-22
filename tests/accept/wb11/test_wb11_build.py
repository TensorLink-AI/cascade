"""WB11: the artefact ``build_hashset`` produces (plan §12b.1 LB1, §3.0 WB11).

Schema, suite identities (W3b4 formula), window/stride/normalisation record,
the exact sub-window set (byte-exact against an independent computation), the
normalised set's shape, canonical bytes + sha256, determinism, and what moves
the digest (suite version, data, artefact version). Contract: README.md."""

from __future__ import annotations

import copy
import importlib.util
import re

import pytest

from .helpers import (
    HASH,
    NAN_AT,
    SCHEMA,
    SIG_FIGS,
    STRIDE,
    SUITE_KEYS,
    WINDOW,
    build,
    canonical_bytes,
    expected_exact,
    expected_identities,
    fixture_suites,
    match,
    mod,
    nonconstant_count,
    sha256_pin,
    shuffled,
    window_count,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_artefact_schema_and_recorded_parameters():
    a = build(version=3)
    assert a["schema"] == SCHEMA, a.get("schema")
    assert a["version"] == 3
    assert a["window"] == WINDOW and a["stride"] == STRIDE, (a.get("window"), a.get("stride"))
    assert a["hash"] == HASH
    norm = a["normalisation"]
    assert norm["method"] == "zscore", norm
    assert norm["scope"] == "subwindow", norm
    assert norm["sig_figs"] == SIG_FIGS, norm
    assert set(a["suites"]) == set(SUITE_KEYS)
    m = mod()
    assert m.WINDOW == WINDOW and m.STRIDE == STRIDE and m.SIG_FIGS == SIG_FIGS
    assert m.SCHEMA == SCHEMA


def test_artefact_suite_identities_follow_the_w3b4_formula():
    a = build()
    assert a["suites"] == expected_identities(fixture_suites()), a["suites"]


def test_artefact_suite_identities_reuse_cascade_eval_suite_identity():
    """When W3b4's module is on main the identities are its ``suite_identity``."""
    a = build()
    if importlib.util.find_spec("cascade.eval.suite_identity") is None:
        pytest.skip("cascade.eval.suite_identity (W3b4) is not on main yet; the W3b4 "
                    "formula is pinned by test_artefact_suite_identities_follow_the_w3b4_formula")
    from cascade.eval.suite_identity import suite_identity

    for k, spec in fixture_suites().items():
        entry = {"status": "ok", "data_revision": spec["version"],
                 "rows": [{"item": i, "crps": None, "mase": None} for i in spec["series"]]}
        assert a["suites"][k] == suite_identity(k, entry), k


def test_exact_set_is_sha256_of_float64_sub_windows():
    a = build()
    want = expected_exact(fixture_suites())
    assert set(a["exact"]) == set(want), (
        f"exact set differs: {len(set(a['exact']) - set(want))} extra, "
        f"{len(set(want) - set(a['exact']))} missing")
    assert a["exact"] == want, "exact hash -> [suite, window_id] mapping differs"
    assert all(HEX64.match(h) for h in a["exact"])


def test_exact_set_skips_nan_and_short_series_and_ties_break_to_the_smallest_window():
    suites = fixture_suites()
    a = build(suites)
    ids = [v[1] for v in a["exact"].values()]
    assert not any(i.startswith("fev/short/0@") for i in ids), "a 31-point series has no window"
    gappy = [int(i.rsplit("@", 1)[1]) for i in ids if i.startswith("time/gappy/1@")]
    # offsets 0..NAN_AT cover the NaN at index NAN_AT
    assert sorted(gappy) == list(range(NAN_AT + 1, 48 - WINDOW + 1)), gappy
    # fev/copy/0 duplicates boom's windows: the smallest (suite, item, offset) wins
    assert not any(i.startswith("fev/copy/0@") for i in ids)
    # the flat series' 9 identical windows collapse to offset 0
    flat = [i for i in ids if i.startswith("time/flat/")]
    assert flat == ["time/flat/0@0"], flat


def test_normalised_set_shape():
    a = build()
    suites = fixture_suites()
    assert a["normalised"], "empty normalised set"
    assert all(HEX64.match(h) for h in a["normalised"])
    total = sum(nonconstant_count(v) for s in suites.values() for v in s["series"].values())
    assert len(a["normalised"]) <= total
    assert not set(a["normalised"]) & set(a["exact"]), "normalised hashes collide with exact"
    for suite, wid in a["normalised"].values():
        assert suite in SUITE_KEYS
        item, off = wid.rsplit("@", 1)
        assert item in suites[suite]["series"] and int(off) >= 0, wid
        assert not item.startswith("time/flat/"), "a constant window has no z-score"


def test_every_suite_window_is_in_both_sets():
    """The suite's own series hit on every sub-window, exact and normalised."""
    a = build()
    for suite, spec in fixture_suites().items():
        for item, values in spec["series"].items():
            r = match(a, values)
            assert r["exact_hits"] == window_count(values), (suite, item, r)
            assert r["normalised_hits"] == nonconstant_count(values), (suite, item, r)
            if item == "time/flat/0":
                assert r["normalised_hits"] == 0 and r["exact_hits"] == 40 - WINDOW + 1


def test_artefact_bytes_are_canonical_json_and_sha256_pins_them():
    m = mod()
    a = build()
    data = m.artefact_bytes(a)
    assert data == canonical_bytes(a), "artefact_bytes is not the pinned canonical JSON"
    assert m.artefact_sha256(a) == sha256_pin(data)


def test_build_is_deterministic_in_process():
    m = mod()
    s = fixture_suites()
    a1 = m.build_hashset(s, version=1)
    a2 = m.build_hashset(shuffled(s, "order-1"), version=1)
    a3 = m.build_hashset(shuffled(s, "order-2"), version=1)
    assert m.artefact_bytes(a1) == m.artefact_bytes(a2) == m.artefact_bytes(a3)
    assert s == fixture_suites(), "build_hashset mutated its input"


def test_build_accepts_none_and_nan_as_missing_values():
    m = mod()
    s = fixture_suites()
    t = copy.deepcopy(s)
    t["time"]["series"]["time/gappy/1"][10] = float("nan")
    assert m.artefact_sha256(m.build_hashset(s, version=1)) == \
        m.artefact_sha256(m.build_hashset(t, version=1))


def test_rebuild_from_the_same_suite_identities_reproduces_the_artefact(tmp_path):
    m = mod()
    a = m.build_hashset(fixture_suites(), version=1)
    p = tmp_path / "hashset-v1.json"
    sha = m.write_artefact(a, p)
    assert sha == m.artefact_sha256(a)
    assert p.read_bytes() == m.artefact_bytes(a), "written file is not the canonical bytes"
    loaded = m.load_artefact(p)
    assert loaded == a
    rebuilt = m.build_hashset(shuffled(fixture_suites(), "rebuild"), version=1)
    assert rebuilt["suites"] == loaded["suites"]
    assert m.artefact_sha256(rebuilt) == sha256_pin(p.read_bytes())


def test_a_different_suite_version_gives_a_different_artefact():
    m = mod()
    s = fixture_suites()
    a = m.build_hashset(s, version=1)
    t = copy.deepcopy(s)
    t["gift"]["version"] = "4f2e9c01" + "0" * 32
    b = m.build_hashset(t, version=1)
    assert b["suites"]["gift"]["version"] == t["gift"]["version"]
    assert b["suites"]["gift"]["digest"] != a["suites"]["gift"]["digest"]
    assert b["exact"] == a["exact"], "same data, same exact hashes"
    assert m.artefact_sha256(b) != m.artefact_sha256(a)


def test_different_data_or_items_or_artefact_version_move_the_digest():
    m = mod()
    s = fixture_suites()
    base = m.artefact_sha256(m.build_hashset(s, version=1))
    assert m.artefact_sha256(m.build_hashset(s, version=2)) != base
    t = copy.deepcopy(s)
    t["boom"]["series"]["boom/svc-b/mem"][5] += 1.0
    tb = m.build_hashset(t, version=1)
    assert m.artefact_sha256(tb) != base
    assert tb["suites"] == m.build_hashset(s, version=1)["suites"], "data edit is not an identity edit"
    u = copy.deepcopy(s)
    u["boom"]["series"]["boom/svc-c/disk"] = list(u["boom"]["series"]["boom/svc-b/mem"])
    ub = m.build_hashset(u, version=1)
    assert ub["suites"]["boom"]["n_items"] == 3
    assert m.artefact_sha256(ub) != base


def test_stride_is_recorded_and_the_scan_is_still_every_offset():
    m = mod()
    s = fixture_suites()
    a = m.build_hashset(s, version=1, stride=4)
    assert a["stride"] == 4
    assert a["exact"] == expected_exact(s, stride=4)
    series = s["gift"]["series"]["gift/m4_hourly/0"]
    r = m.match_series(a, series[1:])          # suite offsets 4k land on 4k-1
    assert r["exact_hits"] == len(range(4, 80 - WINDOW + 1, 4)), r


@pytest.mark.parametrize("bad", [
    {"suites": {}},                                                  # nothing to cover
    {"rename": ("gift", "gift-eval")},                               # not a suite key
    {"rename": ("gift", "m4")},
    {"version_of": ("boom", "")},                                    # unpinned data
    {"version_of": ("boom", None)},
    {"version_of": ("boom", 69325)},
])
def test_build_refuses_bad_suites(bad):
    m = mod()
    s = fixture_suites()
    if "suites" in bad:
        s = bad["suites"]
    if "rename" in bad:
        old, new = bad["rename"]
        s[new] = s.pop(old)
    if "version_of" in bad:
        k, v = bad["version_of"]
        s[k]["version"] = v
    with pytest.raises((ValueError, TypeError)):
        m.build_hashset(s, version=1)


@pytest.mark.parametrize("kw", [{"version": 0}, {"version": -1}, {"version": True},
                                {"version": "1"}, {"version": 1, "window": 1},
                                {"version": 1, "stride": 0}])
def test_build_refuses_bad_parameters(kw):
    m = mod()
    with pytest.raises((ValueError, TypeError)):
        m.build_hashset(fixture_suites(), **kw)


def test_artefact_sha256_is_a_pin_the_wb10_loader_accepts():
    """The owner step: the artefact's sha256 goes into [lane_b] contamination_hashset."""
    import tomllib

    from cascade.shared.config import parse_chain_config

    from .helpers import REPO

    sha = mod().artefact_sha256(build())
    raw = tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))
    raw.setdefault("lane_b", {})["contamination_hashset"] = sha
    assert parse_chain_config(raw).lane_b.contamination_hashset == sha
