"""W7d (plan rev6.6, owner 2026-11-23; §1.1 context bullet): the satellite
VERIFIES the report's armed context set from the evidence the report itself
carries (``armed_contexts`` + ``armed_evidence``, W7a), independently of the
producer and of W5d, instead of re-deriving it from ``cfg.contexts``.
Contract: README.md "Armed contexts (rev6.6)".

The armed fields are spliced into an honest chain's settlement 2 report,
which is then re-signed by the validator's own key and its receipt re-issued
(``helpers.doctored``), so the only thing that can disagree is the armed
record. On main the replay ignores the fields: the consistent cases MATCH
and the lies fail on the MISMATCH assertion.
"""

from __future__ import annotations

import pytest

from .helpers import CHAINS, describe, doctored, has_path, replay, replay_module

N = 2                                  # the settlement doctored
MIN_LONG_SERIES = 300                  # the fixture config's (W2 default)
MIN_SOURCES = 20


def armed_fields(armed=(4096, 8192), *, contexts=(4096, 8192, 16384)) -> dict:
    variants = []
    for c in contexts:
        on = c in armed
        variants.append({"context": c, "series": 4633 if on else 12,
                         "sources": 42 if on else 3, "t7_fits": True, "armed": on})
    return {"armed_contexts": [c for c in contexts if c in armed],
            "armed_evidence": {"min_long_series": MIN_LONG_SERIES,
                               "min_sources": MIN_SOURCES,
                               "readiness_sha256": "a1" * 32, "t7_sha256": "b2" * 32,
                               "variants": variants}}


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    return CHAINS["gated"](tmp_path_factory.mktemp("w7d-armed") / "gated")


def _run(c, mutate):
    reports, receipts = doctored(c.pub, N, mutate)
    return replay(c.pub, reports=reports, receipts=receipts)


def _splice(**over):
    def m(rep):
        f = armed_fields(**over)
        rep.update(f)
        return f
    return m


def _lie(c, edit, paths=("armed_contexts", "armed_evidence")):
    def m(rep):
        rep.update(armed_fields())
        edit(rep)
    res = _run(c, m)
    mod = replay_module()
    for r in res[:N - 1]:
        assert r.status == mod.MATCH, describe(res)
    r = res[N - 1]
    assert r.status == mod.MISMATCH, ("an armed-record lie was not caught", describe(res))
    assert any(has_path(r, p) for p in paths), (paths, list(r.mismatches))
    return r


def test_cfg_contexts_match_the_fixture():
    from .helpers import lane_d_cfg

    cfg = lane_d_cfg()
    assert tuple(cfg.contexts) == (4096, 8192, 16384)
    assert int(cfg.min_long_series) == MIN_LONG_SERIES


def test_a_consistent_armed_subset_matches(gated):
    """A strict subset of ``cfg.contexts`` with evidence that supports it is
    MATCH: the satellite verifies the stated set, it does not demand
    ``cfg.contexts``."""
    res = _run(gated, _splice(armed=(4096, 8192)))
    mod = replay_module()
    assert all(r.status == mod.MATCH for r in res), describe(res)


def test_every_variant_armed_matches(gated):
    res = _run(gated, _splice(armed=(4096, 8192, 16384)))
    mod = replay_module()
    assert all(r.status == mod.MATCH for r in res), describe(res)


def test_an_armed_set_its_evidence_does_not_support_is_a_mismatch(gated):
    def e(rep):
        rep["armed_contexts"] = [4096, 8192, 16384]     # 16384 fails gate 1
    _lie(gated, e)


def test_a_flag_lying_about_its_gates_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["variants"][2]["armed"] = True
        rep["armed_contexts"] = [4096, 8192, 16384]
    _lie(gated, e, paths=("armed_evidence", "armed_contexts"))


def test_a_t7_failure_stated_as_armed_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["variants"][1]["t7_fits"] = False   # 8192 still "armed"
    _lie(gated, e)


def test_evidence_floor_off_config_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["min_long_series"] = 10
        for v in rep["armed_evidence"]["variants"]:
            v.update(series=20, sources=42, armed=True)
        rep["armed_contexts"] = [4096, 8192, 16384]
    _lie(gated, e, paths=("armed_evidence.min_long_series", "armed_evidence"))


def test_min_sources_other_than_20_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["min_sources"] = 2
    _lie(gated, e, paths=("armed_evidence.min_sources", "armed_evidence"))


def test_variants_not_covering_params_contexts_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["variants"].pop()
    _lie(gated, e, paths=("armed_evidence.variants", "armed_evidence"))


def test_an_armed_context_outside_params_contexts_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["variants"].append(
            {"context": 32768, "series": 4633, "sources": 42, "t7_fits": True, "armed": True})
        rep["armed_contexts"] = [4096, 8192, 32768]
    _lie(gated, e)


def test_an_empty_armed_set_is_a_mismatch(gated):
    def e(rep):
        for v in rep["armed_evidence"]["variants"]:
            v.update(series=12, sources=3, armed=False)
        rep["armed_contexts"] = []
    _lie(gated, e, paths=("armed_contexts",))


def test_one_armed_field_without_the_other_is_a_mismatch(gated):
    _lie(gated, lambda rep: rep.pop("armed_evidence"), paths=("armed_evidence",))
    _lie(gated, lambda rep: rep.pop("armed_contexts"), paths=("armed_contexts",))


def test_a_malformed_evidence_sha_is_a_mismatch(gated):
    def e(rep):
        rep["armed_evidence"]["readiness_sha256"] = "not-hex"
    _lie(gated, e, paths=("armed_evidence.readiness_sha256", "armed_evidence"))
