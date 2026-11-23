"""W7a (plan rev6.6, owner 2026-11-23; §1.1 context bullet): the armed
context set and its evidence in the signed report. Contract: README.md
"Armed contexts (rev6.6)".

The main validator computes the armed set per settlement (the W5d
``armed_record``: ``{"armed_contexts", "evidence"}``) and passes it to
``settle_lane_d(..., armed=record)``. The report then carries it as the
top-level ``armed_contexts`` and ``armed_evidence``, inside the signed body,
so satellites verify it instead of re-deriving it from ``cfg.contexts``.
The record here is written by hand (no W5d import): W7a checks its
consistency, it does not compute it.

On main ``settle_lane_d`` has no ``armed`` keyword: every settle test fails
on the ``_settle_takes_armed`` assertion, never a ``TypeError``.
"""

from __future__ import annotations

import copy
import inspect
import json

import pytest

from .helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    assert_valid,
    build_chain,
    coldkey,
    entry,
    fixture_key,
    gate_inputs,
    lane_d_cfg,
)

CONTEXTS = (4096, 8192, 16384)
MIN_SOURCES = 20
SHA_R = "a1" * 32
SHA_T = "b2" * 32


def armed_rec(armed=(4096, 8192), *, contexts=CONTEXTS, mls: int = 300) -> dict:
    """A consistent W5d record: armed variants pass both gates, the others
    fail the readiness gate."""
    variants = []
    for c in contexts:
        on = c in armed
        variants.append({"context": c, "series": 4633 if on else 12,
                         "sources": 42 if on else 3, "t7_fits": True, "armed": on})
    return {"armed_contexts": [c for c in contexts if c in armed],
            "evidence": {"min_long_series": mls, "min_sources": MIN_SOURCES,
                         "readiness_sha256": SHA_R, "t7_sha256": SHA_T,
                         "variants": variants}}


def _settle_takes_armed() -> None:
    from cascade.validator.lane_d import settle_lane_d

    p = inspect.signature(settle_lane_d).parameters
    assert "armed" in p, "rev6.6: settle_lane_d(..., armed=None) takes the W5d armed record"
    assert p["armed"].kind is inspect.Parameter.KEYWORD_ONLY and p["armed"].default is None


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return build_chain(tmp_path_factory.mktemp("w7a_armed") / "chain", rich=True)


def _settle(chain, tmp, *, armed, entries=None, cfg=None):
    from cascade.validator.lane_d import settle_lane_d

    _settle_takes_armed()
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    ents = [e] if entries is None else entries
    uid_of = {x["hotkey"]: CHALLENGER_UID + i for i, x in enumerate(ents)}
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if armed is None else {"armed": armed}
    s = settle_lane_d(**_rev614_ts(settle_lane_d, BOUNDARY_1), cfg=cfg or lane_d_cfg(), boundary=BOUNDARY_1, pool_dir=chain["pools"][1],
                      entries=[dict(x) for x in ents], house=dict(chain["house"]),
                      forecasts=chain["forecasts"][1], uid_of=uid_of, state_path=state,
                      signer=fixture_key("signer"),
                      **gate_inputs(BOUNDARY_1, chain["house"]["digest"]), **kw)
    return s, state


def _refusal(exc) -> str:
    return str(getattr(exc, "code", "")) or str(exc)


# ── the report carries it, signed ───────────────────────────────────────────


def test_the_report_carries_the_armed_set_and_its_evidence(chain, tmp_path):
    rec = armed_rec()
    s, _ = _settle(chain, tmp_path, armed=copy.deepcopy(rec))
    r = s.report
    assert r["armed_contexts"] == rec["armed_contexts"] == [4096, 8192]
    assert r["armed_evidence"] == rec["evidence"]
    assert r["params"]["contexts"] == list(CONTEXTS), "params.contexts stays the declarable set"
    assert_valid(r)


def test_the_armed_fields_are_inside_the_signed_body(chain, tmp_path):
    from cascade.validator.lane_d_report import canonical_body, verify_report

    s, _ = _settle(chain, tmp_path, armed=armed_rec())
    r = s.report
    body = json.loads(canonical_body(r))
    assert body["armed_contexts"] == [4096, 8192] and "armed_evidence" in body
    signer = fixture_key("signer").ss58_address
    assert verify_report(r, signer)
    lie = copy.deepcopy(r)
    lie["armed_contexts"] = [4096]
    lie["armed_evidence"]["variants"][1].update(series=12, sources=3, armed=False)
    assert not verify_report(lie, signer), "an edited armed set must break the signature"


def test_without_a_record_the_report_has_no_armed_fields(chain, tmp_path):
    """The legacy call (no ``armed``) is the pre-rev6.6 report, field for field."""
    s, _ = _settle(chain, tmp_path, armed=None)
    assert "armed_contexts" not in s.report and "armed_evidence" not in s.report
    assert_valid(s.report)


def test_an_entry_declaring_an_unarmed_variant_is_registered(chain, tmp_path):
    """16384 is in ``contexts`` but unarmed: the entry is open (it is scored at
    the highest armed variant by the eval host), never ``context_not_armed``."""
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    e["context"] = 16384
    s, _ = _settle(chain, tmp_path, armed=armed_rec((4096, 8192)), entries=[e])
    rec = {x["digest"]: x for x in s.report["entries"]}[e["digest"]]
    assert rec["context"] == 16384
    assert rec["status"] != "closed" and rec["reason"] != "context_not_armed", rec
    assert e["digest"] not in {x["digest"] for x in s.report["skipped"]
                               if x["reason"] == "context_not_armed"}


def test_an_entry_declaring_a_context_outside_contexts_is_closed(chain, tmp_path):
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    e["context"] = 32768
    s, _ = _settle(chain, tmp_path, armed=armed_rec(), entries=[e])
    rec = {x["digest"]: x for x in s.report["entries"]}[e["digest"]]
    assert rec["status"] == "closed" and rec["reason"] == "context_not_armed", rec


# ── a bad record is refused before anything is read or written ──────────────


def _bad_records():
    out = []
    r = armed_rec()
    r["armed_contexts"] = []
    for v in r["evidence"]["variants"]:
        v.update(series=12, sources=3, armed=False)
    out.append(("no_armed_context", r))
    r = armed_rec()
    r["armed_contexts"] = [4096, 8192, 16384]            # 16384 fails gate 1
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["evidence"]["variants"][2]["armed"] = True          # a flag lying about its gates
    r["armed_contexts"] = [4096, 8192, 16384]
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["evidence"]["variants"].pop()                       # does not cover contexts
    out.append(("armed_invalid", r))
    r = armed_rec(mls=150)                                # off the config's floor
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["evidence"]["min_sources"] = 10
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["evidence"]["t7_sha256"] = "xyz"
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["evidence"]["note"] = "extra"
    out.append(("armed_invalid", r))
    r = armed_rec()
    r["armed_contexts"] = [8192, 4096]                    # not ascending
    out.append(("armed_invalid", r))
    out.append(("armed_invalid", {"armed_contexts": [4096]}))
    out.append(("armed_invalid", [4096, 8192]))
    return out


@pytest.mark.parametrize(("code", "rec"), _bad_records())
def test_a_bad_armed_record_is_refused_and_writes_nothing(chain, tmp_path, code, rec):
    from cascade.validator.lane_d import LaneDArmedError

    with pytest.raises(LaneDArmedError) as ei:
        _settle(chain, tmp_path, armed=rec)
    assert isinstance(ei.value, ValueError)
    assert ei.value.code == code, _refusal(ei.value)
    state = tmp_path / "state"
    assert not any(state.rglob("*")), f"a refused settlement wrote {sorted(state.rglob('*'))}"


# ── validate_report: the cross-field rules ───────────────────────────────────


def _armed_report(chain, tmp_path):
    """Called from test bodies (never a fixture): on main the settle
    assertion is a test failure, not a fixture error."""
    s, _ = _settle(chain, tmp_path, armed=armed_rec())
    return s.report


def _refused(report, code: str) -> str:
    from cascade.validator.lane_d_report import ReportSchemaError, validate_report

    with pytest.raises(ReportSchemaError) as ei:
        validate_report(report)
    assert ei.value.code == code, (ei.value.code, ei.value.field, str(ei.value))
    return ei.value.field


def test_validate_accepts_the_armed_report(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    assert_valid(armed_report)


def test_validate_refuses_one_armed_field_without_the_other(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    a = copy.deepcopy(armed_report)
    del a["armed_evidence"]
    assert _refused(a, "missing_field").startswith("armed_evidence")
    b = copy.deepcopy(armed_report)
    del b["armed_contexts"]
    assert _refused(b, "missing_field").startswith("armed_contexts")


def test_validate_refuses_an_armed_set_outside_params_contexts(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    r = copy.deepcopy(armed_report)
    r["armed_contexts"] = [4096, 32768]
    assert _refused(r, "bad_value").startswith("armed_")


def test_validate_refuses_an_armed_set_its_evidence_does_not_support(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    r = copy.deepcopy(armed_report)
    r["armed_contexts"] = [4096, 8192, 16384]
    assert _refused(r, "bad_value").startswith("armed_")
    r = copy.deepcopy(armed_report)
    r["armed_evidence"]["variants"][0]["armed"] = False
    assert _refused(r, "bad_value").startswith("armed_")


def test_validate_refuses_evidence_off_params(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    r = copy.deepcopy(armed_report)
    r["armed_evidence"]["min_long_series"] = 299
    assert _refused(r, "bad_value").startswith("armed_evidence")
    r = copy.deepcopy(armed_report)
    r["armed_evidence"]["variants"] = r["armed_evidence"]["variants"][:2]
    assert _refused(r, "bad_value").startswith("armed_evidence")


def test_validate_refuses_an_empty_or_null_armed_set(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    r = copy.deepcopy(armed_report)
    r["armed_contexts"] = []
    for v in r["armed_evidence"]["variants"]:
        v.update(series=12, sources=3, armed=False)
    assert _refused(r, "bad_value").startswith("armed_contexts")
    r = copy.deepcopy(armed_report)
    r["armed_contexts"] = None
    assert _refused(r, "bad_type").startswith("armed_contexts")


def test_validate_refuses_an_unknown_evidence_field(chain, tmp_path):
    armed_report = _armed_report(chain, tmp_path)
    r = copy.deepcopy(armed_report)
    r["armed_evidence"]["note"] = "x"
    assert _refused(r, "unknown_field").startswith("armed_evidence")
    r = copy.deepcopy(armed_report)
    r["armed_evidence"]["variants"][0]["note"] = "x"
    assert _refused(r, "unknown_field").startswith("armed_evidence")


def test_schema_file_states_the_armed_fields():
    from .helpers import SCHEMA_FILE

    doc = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    props = doc["properties"]
    assert "armed_contexts" in props and "armed_evidence" in props
    assert "armed_contexts" not in doc["required"] and "armed_evidence" not in doc["required"]


# ── rev6.15: no armed variant ≤ the declared context ⇒ skipped, pending ──────

BELOW = "context_below_armed"


def _below_armed(chain, tmp_path):
    """Declared 4096 with only 8192 and 16384 armed (plan rev6.15, owner
    2026-11-23: "Highest ≤ declared, else skip")."""
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    e["context"] = 4096
    s, state = _settle(chain, tmp_path, armed=armed_rec((8192, 16384)), entries=[e])
    return s, state, e["digest"]


def test_context_below_armed_is_a_skip_reason():
    from cascade.validator.lane_d_report import ENTRY_REASONS, SKIP_REASONS

    assert BELOW in tuple(SKIP_REASONS), "rev6.15: the typed skip reason"
    assert BELOW in tuple(ENTRY_REASONS)


def test_an_entry_below_every_armed_variant_is_skipped_and_stays_pending(chain, tmp_path):
    s, state, dg = _below_armed(chain, tmp_path)
    r = s.report
    rec = {x["digest"]: x for x in r["entries"]}[dg]
    assert rec["context"] == 4096, "the declared context is recorded as declared"
    assert (rec["status"], rec["reason"]) == ("pending", BELOW), rec
    assert [x["reason"] for x in r["skipped"] if x["digest"] == dg] == [BELOW], r["skipped"]
    assert dg not in r["judged"], "never judged above its declared context"
    assert dg not in {v["digest"] for v in r["verdicts"]}
    assert r["armed_contexts"] == [8192, 16384]
    st = json.loads(state.read_text(encoding="utf-8"))["entries"][dg]
    assert (st["status"], st["reason"]) == ("pending", BELOW), st
    assert_valid(r)


def test_the_below_armed_skip_is_deterministic(chain, tmp_path):
    a, _, dg = _below_armed(chain, tmp_path / "a")
    b, _, _ = _below_armed(chain, tmp_path / "b")
    pick = ("entries", "skipped", "judged", "verdicts", "armed_contexts")
    assert {k: a.report[k] for k in pick} == {k: b.report[k] for k in pick}


def test_an_entry_with_an_armed_variant_below_it_is_not_skipped_below_armed(chain, tmp_path):
    """Declared 16384 with 4096 and 8192 armed: scored at 8192, never this skip."""
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    e["context"] = 16384
    s, _ = _settle(chain, tmp_path, armed=armed_rec((4096, 8192)), entries=[e])
    rec = {x["digest"]: x for x in s.report["entries"]}[e["digest"]]
    assert rec["reason"] != BELOW
    assert BELOW not in {x["reason"] for x in s.report["skipped"]}


def test_validate_refuses_a_below_armed_skip_the_armed_set_contradicts(chain, tmp_path):
    s, _, dg = _below_armed(chain, tmp_path)
    assert [x["reason"] for x in s.report["skipped"] if x["digest"] == dg] == [BELOW], \
        "the fixture report carries the rev6.15 skip"
    assert_valid(s.report)
    r = copy.deepcopy(s.report)
    for k in ("armed_contexts", "armed_evidence"):
        del r[k]
    _refused(r, "bad_value")                    # no armed set: the skip is unsupported
    r = copy.deepcopy(s.report)
    rec = {x["digest"]: x for x in r["entries"]}[dg]
    rec["context"] = 8192                       # 8192 is armed: it is not below
    _refused(r, "bad_value")


def _rev614_ts(settle, boundary) -> dict:
    """rev6.14 (a): the fixture's on-chain boundary timestamp (far after every
    fixture T + eval_delay_hours, 12 s per block), passed only where
    ``settle_lane_d`` takes it."""
    import inspect

    if "boundary_ts" not in inspect.signature(settle).parameters:
        return {}
    return {"boundary_ts": 2_000_000_000 + 12 * int(boundary)}
