"""W8c follow-up (plan rev6.6, §1.1 context bullet; W5d ``armed_record``,
W7a ``settle_lane_d(..., armed=)``): the source computes the settlement's
armed context set from the pinned readiness (``laned_readiness/1``) and T7
(``laned_t7/1``) document bytes, forecasts each digest at its
``effective_context``, and hands the record to ``settle_lane_d`` through
``LaneDInputs.armed`` so the signed report carries it. A missing or invalid
document, or an empty armed set, fails closed. Contract: README.md "Armed
contexts (rev6.6)".

On a build without the ``readiness`` / ``t7`` keywords every test here fails
on an assertion (``helpers.assert_takes_armed_docs`` or an explicit
``hasattr`` check), never a ``TypeError``.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import helpers as h


def _refused(rig, code: str, boundary: int = h.B1):
    before = rig.files()
    with pytest.raises(Exception) as ei:
        rig.source(boundary)
    h.assert_inputs_error(ei.value, code)
    assert rig.files() == before, f"{code}: a refusal writes nothing"
    assert rig.eval_host.hellos == 0 and rig.eval_host.calls == [], \
        f"{code}: nothing is sent to the eval host"
    return ei.value


def _ctx(rig) -> dict:
    return {c["digest"]: c["context"] for c in rig.eval_host.calls}


# ── the record ──────────────────────────────────────────────────────────────


def test_armed_codes_are_exported():
    m = h.inputs_mod()
    missing = sorted(set(h.ARMED_CODES) - set(m.INPUTS_ERROR_CODES))
    assert not missing, f"rev6.6: INPUTS_ERROR_CODES lacks {missing}"


def test_lane_d_inputs_carries_an_optional_armed_record():
    """W8b's ``LaneDInputs`` gains ``armed`` (default ``None``: W8b's own
    callers and the legacy settle are unchanged)."""
    import dataclasses

    from cascade.validator.lane_d_wiring import LaneDInputs

    names = {f.name: f for f in dataclasses.fields(LaneDInputs)}
    assert "armed" in names, "rev6.6: LaneDInputs.armed carries the W5d armed record"
    assert names["armed"].default is None
    inp = LaneDInputs(pool_dir="p", entries=[], forecasts={}, uid_of={}, house={})
    assert getattr(inp, "armed", "missing") is None


def test_inputs_carry_the_w5d_armed_record(tmp_path):
    rig = h.Rig(tmp_path)
    got = rig.source(h.B1)
    armed = getattr(got, "armed", None)
    assert armed == rig.expected_armed(), \
        "armed is exactly W5d armed_record(cfg contexts, readiness, t7, min_long_series)"
    assert armed["armed_contexts"] == list(h.DEFAULT_ARMED)
    ev = armed["evidence"]
    assert ev["readiness_sha256"] == hashlib.sha256(rig.readiness.data).hexdigest()
    assert ev["t7_sha256"] == hashlib.sha256(rig.t7.data).hexdigest()


def test_build_reads_no_document_and_unarmed_reads_none(tmp_path):
    rig = h.Rig(tmp_path)
    assert rig.readiness.calls == [] and rig.t7.calls == [], "building reads no document"
    cfg = h.w8b.chain_cfg(tmp_path / "u", armed=False)
    out = rig.build(readiness=h.Untouchable("readiness"), t7=h.Untouchable("t7"))
    assert out is not None
    out = h.inputs_mod().build_lane_d_inputs(
        cfg, state_dir=tmp_path / "ns", intake_state_dir=tmp_path / "ni",
        chain_factory=h.Untouchable("chain_factory"), pool=h.Untouchable("pool"),
        eval_host=h.Untouchable("eval_host"), house_pointer=tmp_path / "nh.json",
        house_signer_ss58=h.house_key().ss58_address, bench_host=None,
        expected_identity=h.identity(), readiness=h.Untouchable("readiness"),
        t7=h.Untouchable("t7"))
    assert out is None


def test_documents_are_read_on_the_callers_thread_at_every_settlement(tmp_path):
    """The documents are re-read per settlement: a readiness update between
    boundaries arms 16384 at the next one, no restart."""
    import threading

    rig = h.Rig(tmp_path)
    a = rig.source(h.B1)
    me = threading.get_ident()
    assert rig.readiness.calls == [me] and rig.t7.calls == [me]
    rig.readiness.data = h.readiness_doc(armed=h.CONTEXTS)
    b = rig.source(h.B2)
    assert len(rig.readiness.calls) == 2 and len(rig.t7.calls) == 2
    assert a.armed["armed_contexts"] == [4096, 8192]
    assert b.armed["armed_contexts"] == [4096, 8192, 16384]


def test_same_documents_same_inputs(tmp_path):
    rig = h.Rig(tmp_path)
    a = rig.source(h.B1)
    b = rig.build()(h.B1)
    assert a.armed == b.armed
    assert json.dumps(a.armed, sort_keys=True) == json.dumps(rig.expected_armed(), sort_keys=True)


# ── effective contexts ──────────────────────────────────────────────────────


@pytest.mark.parametrize("declared,armed,want_entry,want_house", [
    (4096, (4096, 8192), 4096, 4096),       # armed: its own context
    (16384, (4096, 8192), 8192, 4096),      # unarmed: the highest armed below it
    (8192, (4096, 16384), 4096, 4096),
    (4096, (8192, 16384), 8192, 8192),      # nothing armed below: the lowest armed
    (16384, (16384,), 16384, 16384),
])
def test_each_digest_is_forecast_at_its_effective_context(tmp_path, declared, armed,
                                                          want_entry, want_house):
    fx = h.w8b.fixture(tmp_path)
    rig = h.Rig(tmp_path, fx=fx, entries=[dict(fx["entry"], context=declared)])
    rig.readiness.data = h.readiness_doc(armed=armed)
    got = rig.source(h.B1)
    assert got.armed["armed_contexts"] == list(armed)
    ctx = _ctx(rig)
    assert ctx[fx["x"]] == want_entry, \
        f"declared {declared}, armed {armed}: the entry runs at {want_entry}"
    assert ctx[fx["house"]["digest"]] == want_house, \
        f"armed {armed}: the house (trained at 4096) runs at {want_house}"
    assert [e["context"] for e in got.entries] == [declared], \
        "the entry keeps its declared context (settle_lane_d records it)"


def test_open_entries_and_king_d_follow_the_armed_set(tmp_path):
    """After B1 X is open (its state record has its declared context); at B2
    with only 8192 armed both X and the house run at 8192."""
    rig = h.Rig(tmp_path)
    rig.step()(h.B1)
    rig.readiness.data = h.readiness_doc(armed=(8192,))
    rig.eval_host.calls.clear()
    rig.source(h.B2)
    ctx = _ctx(rig)
    assert ctx[rig.fx["x"]] == 8192 and ctx[rig.fx["house"]["digest"]] == 8192


def test_eval_host_is_told_the_armed_set_before_any_forecast(tmp_path):
    rig = h.Rig(tmp_path)
    rig.source(h.B1)
    assert rig.eval_host.armed_calls == [(list(h.DEFAULT_ARMED), 0)], \
        "set_armed(armed contexts) once per settlement, before the first forecast"


def test_ssh_eval_host_sends_the_armed_set_as_the_request_contexts(tmp_path):
    m = h.inputs_mod()
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "10.0.0.9"\nstage = "final"\n',
                     encoding="utf-8")
    from cascade.eval.lane_d_forecasts import REPLY_SCHEMA, decode_request

    sent = []
    n, hz = 2, 4

    def run(argv, *, input, capture_output, timeout, check):
        if "hello" in argv[-1].split():
            out = json.dumps(h.hello())
        else:
            sent.append(decode_request(input)[0])
            out = json.dumps({"schema": REPLY_SCHEMA, "ok": True,
                              "quantiles": np.zeros((n, hz, 9)).tolist()})
        return SimpleNamespace(returncode=0, stdout=out.encode(), stderr=b"")

    host = m.SSHEvalHost(hosts, cache_dir="/c", budget_bytes=10, contexts=h.CONTEXTS, run=run)
    assert hasattr(host, "set_armed"), "rev6.6: SSHEvalHost.set_armed(contexts)"
    host.hello()
    wins = [{"window_id": f"w{i}", "history": np.ones(64)} for i in range(n)]
    host.forecast("d" * 64, context=4096, horizon=hz, windows=wins)
    host.set_armed([4096, 8192])
    host.forecast("d" * 64, context=8192, horizon=hz, windows=wins)
    assert sent[0]["contexts"] == list(h.CONTEXTS), "before set_armed: the declarable contexts"
    assert sent[1]["contexts"] == [4096, 8192], "after set_armed: the armed set"


# ── fail closed ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which,code", [("readiness", "readiness_unavailable"),
                                        ("t7", "t7_unavailable")])
def test_absent_document_fails_closed(tmp_path, which, code):
    rig = h.Rig(tmp_path)
    getattr(rig, which).fail = True
    _refused(rig, code)


@pytest.mark.parametrize("which,data", [
    ("readiness", b"{not json"),
    ("readiness", json.dumps({"schema": "other/1", "rungs": []}).encode()),
    ("readiness", json.dumps({"schema": "laned_readiness/1",
                              "rungs": [{"context": 4096, "series": -1, "sources": 3}]}).encode()),
    ("t7", b"[]"),
    ("t7", h.t7_doc(lie=8192)),
])
def test_invalid_document_fails_closed(tmp_path, which, data):
    rig = h.Rig(tmp_path)
    getattr(rig, which).data = data
    _refused(rig, "armed_invalid")


@pytest.mark.parametrize("readiness,t7", [
    (h.readiness_doc(armed=()), h.t7_doc()),          # no rung has the long series
    (h.readiness_doc(), h.t7_doc(fits=())),            # no variant fits its budget
    (h.readiness_doc(armed=(16384,)), h.t7_doc(fits=(4096,))),   # the gates never meet
])
def test_no_armed_context_fails_closed(tmp_path, readiness, t7):
    rig = h.Rig(tmp_path)
    rig.readiness.data, rig.t7.data = readiness, t7
    _refused(rig, "no_armed_context")


def test_step_reports_an_armed_refusal_as_inputs_unavailable(tmp_path):
    rig = h.Rig(tmp_path)
    store = h.w8b.Store()
    step = rig.step(store=store)
    rig.t7.fail = True
    with pytest.raises(Exception) as ei:
        step(h.B1)
    h.w8b.assert_wiring_error(ei.value, "inputs_unavailable")
    h.assert_inputs_error(ei.value.__cause__, "t7_unavailable")
    assert store.puts == [] and not rig.state_path.exists()
    rig.t7.fail = False
    assert step(h.B1) is not None


# ── the production collaborators ────────────────────────────────────────────


def test_inputs_file_names_both_documents():
    m = h.inputs_mod()
    for k in ("readiness", "t7"):
        assert k in m.INPUTS_FILE_KEYS, f"rev6.6: the inputs TOML key {k!r}"
        assert k in m._REQUIRED_KEYS, f"{k!r} is required (fail closed)"


@pytest.mark.parametrize("drop", ["readiness", "t7"])
def test_inputs_file_without_a_document_is_refused(tmp_path, drop):
    m = h.inputs_mod()
    assert drop in m.INPUTS_FILE_KEYS, f"rev6.6: the inputs TOML key {drop!r}"
    keys = {"intake_state_dir": "/i", "pool_pins": "/p.json", "house_pointer": "/h.json",
            "house_signer_ss58": "5x", "eval_hosts": "/e.toml", "eval_cache_dir": "/c",
            "readiness": "/r.json", "t7": "/t.json"}
    keys.pop(drop)
    lines = [f'{k} = "{v}"' for k, v in keys.items()] + ["eval_cache_budget_bytes = 10"]
    p = tmp_path / "inputs.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        m.load_inputs_services(SimpleNamespace(), p)
    except m.LaneDInputsConfigError as e:
        assert drop in str(e), f"the refusal names the missing key {drop!r}: {e}"
    else:
        raise AssertionError(f"an inputs file without {drop!r} must be refused")


def test_document_file_reads_its_bytes_at_every_call(tmp_path):
    m = h.inputs_mod()
    assert hasattr(m, "DocumentFile"), "rev6.6: DocumentFile(path) is the production source"
    p = tmp_path / "readiness.json"
    doc = m.DocumentFile(p)
    with pytest.raises(OSError):
        doc()
    p.write_bytes(h.readiness_doc())
    assert doc() == h.readiness_doc()
    p.write_bytes(h.readiness_doc(armed=h.CONTEXTS))
    assert doc() == h.readiness_doc(armed=h.CONTEXTS), "re-read at every call"
