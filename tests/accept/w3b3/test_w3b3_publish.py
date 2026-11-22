"""W3b3: the sentinel record is published under its deterministic key,
idempotently; no new king means no sweep and no publish; sweep errors and
store failures are recorded or returned, never raised (plan §1.3b, §16 rule
4, T7b). README.md states the contract."""

from __future__ import annotations

import math

import pytest

from . import helpers as h

ROLES = ("king", "previous_king", "house")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return h.build_chain(tmp_path_factory.mktemp("w3b3-publish-chain"))


def _report(chain):
    return chain[h.BOUNDARY_4][0]


def _host(report, **kw):
    r = h.roles_of(report)
    return h.host_for(r["king"], r["previous_king"], r["house"], **kw)


def test_the_record_is_published_once_under_its_key(chain):
    report = _report(chain)
    store = h.MemoryStore()
    res = h.run(report, era_hash_=h.hash_drawing("gift"), bench=h.session(_host(report)),
                store=store)
    assert res["published"] is True
    assert store.puts == [(res["key"], res["text"])]


def test_republishing_the_same_record_writes_nothing(chain):
    report = _report(chain)
    store = h.MemoryStore()
    eh = h.hash_drawing("gift")
    first = h.run(report, era_hash_=eh, bench=h.session(_host(report)), store=store)
    again = h.run(report, era_hash_=eh, bench=h.session(_host(report)), store=store)
    assert again["published"] is True
    assert again["text"] == first["text"]
    assert len(store.puts) == 1
    assert store.objects == {first["key"]: first["text"]}


def test_a_published_ok_record_is_never_replaced(chain):
    """A re-sweep in a later session (different bytes) leaves the first ok
    record in place."""
    report = _report(chain)
    store = h.MemoryStore()
    eh = h.hash_drawing("gift")
    first = h.run(report, era_hash_=eh, bench=h.session(_host(report)), store=store)
    later = h.run(report, era_hash_=eh,
                  bench=h.session(_host(report, king_factor=1.4), f"{h.SESSION}/later"),
                  store=store)
    assert later["published"] is True
    assert len(store.puts) == 1
    assert store.objects[first["key"]] == first["text"]


def test_no_new_king_means_no_sweep_and_no_publish(chain):
    for boundary in (h.BOUNDARY_1, h.BOUNDARY_3):
        report = chain[boundary][0]
        assert report["crowned"] is None
        host = h.host_for(h.Y["digest"], h.K1["digest"], h.H0["digest"])
        store = h.MemoryStore()
        res = h.run(report, era_hash_=h.hash_drawing("boom"), bench=h.session(host),
                    store=store)
        assert res is None
        assert host.calls == []
        assert store.puts == [] and store.objects == {}


def test_an_unarmed_settlement_report_publishes_nothing():
    host = h.host_for(h.Y["digest"], h.K1["digest"], h.H0["digest"])
    store = h.MemoryStore()
    assert h.run(None, era_hash_=h.hash_drawing("boom"), bench=h.session(host),
                 store=store) is None
    assert host.calls == [] and store.puts == []


def test_a_host_error_publishes_an_error_record_and_never_raises(chain):
    report = _report(chain)
    host = h.RecordingHost({}, raises=RuntimeError("bench pod unreachable"))
    store = h.MemoryStore()
    res = h.run(report, era_hash_=h.hash_drawing("time"), bench=h.session(host), store=store)
    rec = res["record"]
    assert rec["status"] == "error"
    assert isinstance(rec["reason"], str) and rec["reason"]
    assert rec["suite"] == "time" and rec["king_digest"] == h.Y["digest"]
    for role, dg in h.roles_of(report).items():
        m = rec["models"][role]
        assert m["digest"] == dg and m["status"] == "error"
        assert m["crps"] is None and m["mase"] is None
    assert res["published"] is True
    assert store.objects[res["key"]] == h.canonical(rec)
    assert h.strict_loads(store.objects[res["key"]]) == rec


def test_an_error_record_is_replaced_by_a_later_good_sweep(chain):
    """The typed retry: the same key, re-swept in a new session, publishes
    the numbers over the error record."""
    report = _report(chain)
    store = h.MemoryStore()
    eh = h.hash_drawing("time")
    bad = h.run(report, era_hash_=eh,
                bench=h.session(h.RecordingHost({}, raises=OSError("connection reset"))),
                store=store)
    good = h.run(report, era_hash_=eh, bench=h.session(_host(report), f"{h.SESSION}/retry"),
                 store=store)
    assert bad["key"] == good["key"]
    assert good["record"]["status"] == "ok"
    assert good["published"] is True
    assert store.objects[good["key"]] == good["text"]


@pytest.mark.parametrize("fault", ["nan", "inf", "none", "missing_metric", "partial",
                                   "status_error", "no_rows", "absent"])
def test_a_faulty_model_is_recorded_as_a_fault(chain, fault):
    report = _report(chain)
    r = h.roles_of(report)
    host = _host(report)
    good = h.rows("fev", r["king"], factor=0.95)
    if fault == "absent":
        del host.table[r["king"]]["fev"]
    elif fault in ("nan", "inf", "none", "missing_metric"):
        bad = [dict(x) for x in good]
        if fault == "missing_metric":
            del bad[3]["mase"]
        else:
            bad[3]["crps"] = {"nan": math.nan, "inf": math.inf, "none": None}[fault]
        host.table[r["king"]]["fev"] = bad
    elif fault == "partial":
        host.table[r["king"]]["fev"] = {"status": "ok", "rows": good, "partial": True}
    elif fault == "status_error":
        host.table[r["king"]]["fev"] = {"status": "error", "rows": [], "error": "oom"}
    else:
        host.table[r["king"]]["fev"] = {"status": "ok", "rows": []}
    store = h.MemoryStore()
    res = h.run(report, era_hash_=h.hash_drawing("fev"), bench=h.session(host), store=store)
    rec = res["record"]
    assert rec["status"] == "fault"
    assert isinstance(rec["reason"], str) and rec["reason"]
    k = rec["models"]["king"]
    assert k["status"] == "fault" and k["crps"] is None and k["mase"] is None
    for role in ("previous_king", "house"):
        assert rec["models"][role]["status"] == "ok"
        assert isinstance(rec["models"][role]["crps"], float)
    assert h.strict_loads(store.objects[res["key"]]) == rec


def test_a_store_failure_never_raises(chain):
    from cascade.shared.hippius import StorageError

    report = _report(chain)
    store = h.MemoryStore(fail_put=StorageError("s3_put_failed: 503"))
    res = h.run(report, era_hash_=h.hash_drawing("gift"), bench=h.session(_host(report)),
                store=store)
    assert res["published"] is False
    assert res["record"]["status"] == "ok"
    assert res["text"] == h.canonical(res["record"])
    ok = h.MemoryStore()
    again = h.run(report, era_hash_=h.hash_drawing("gift"), bench=h.session(_host(report)),
                  store=ok)
    assert again["published"] is True and ok.objects[again["key"]] == res["text"]


def test_a_sentinel_equal_to_the_gate_suite_is_refused(chain):
    report = _report(chain)
    host = _host(report)
    store = h.MemoryStore()
    with pytest.raises(ValueError):
        h.run(report, era_hash_=h.hash_drawing("boom"), bench=h.session(host), store=store,
              gate_suite="boom")
    assert host.calls == [] and store.puts == []


@pytest.mark.parametrize("bad", ["", "deadbe", "0x1234", 7])
def test_a_malformed_era_hash_is_refused(chain, bad):
    report = _report(chain)
    host = _host(report)
    store = h.MemoryStore()
    with pytest.raises((ValueError, TypeError)):
        h.run(report, era_hash_=bad, bench=h.session(host), store=store, gate_suite="boom")
    assert host.calls == [] and store.puts == []


def test_empty_sentinel_suites_are_refused(chain):
    report = _report(chain)
    host = _host(report)
    with pytest.raises(ValueError):
        h.run(report, era_hash_=h.hash_drawing("boom"), bench=h.session(host),
              store=h.MemoryStore(), suites=(), gate_suite="boom")
    assert host.calls == []
