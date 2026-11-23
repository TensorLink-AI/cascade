"""W3b6a, box side: ``SSHBenchHost`` is the ``SweepSession`` sweep. It sends
the bench host digests, suites, contexts and presigned grants (never weights),
runs the pod command over SSH on the first final/any host of a hosts file
re-read at every call, and refuses anything but a well-formed reply. Every
failure raises, which ``settle_lane_d`` turns into ``bench_hold``. Plus the
optional ``bench_hosts`` key of the ``--lane-d-inputs`` TOML. Contract:
README.md."""

from __future__ import annotations

import json
import shlex

import pytest

from . import helpers as h

A, B = "a" * 64, "b" * 64


def _host(tmp_path, run, **kw):
    kw.setdefault("cache_dir", "/root/lane_d_cache")
    kw.setdefault("budget_bytes", 10 ** 11)
    host = h.host_mod().SSHBenchHost(h.hosts_file(tmp_path), run=run, **kw)
    host.set_contexts({A: 4096, B: 16384})
    return host


def test_round_trip(tmp_path):
    rows = {A: h.rows_entry(1), B: h.rows_entry(2)}
    run = h.Run(h.reply(rows))
    got = _host(tmp_path, run, grants=lambda d: h.grant(d) if d == A else None)((A, B), h.SUITES)
    assert got == rows
    (call,) = run.calls
    remote = call["argv"][-1]
    assert "cascade.eval.lane_d_bench_sweep" in remote and " sweep" in remote
    assert "10.0.0.7" in " ".join(call["argv"]), "the hosts file's host"
    req = run.request()
    assert set(req) == {"schema", "digests", "suites", "contexts", "grants"}, \
        "only names, contexts and grants leave the box: no weights, no histories"
    assert req["schema"] == h.REQUEST_SCHEMA
    assert req["digests"] == [A, B] and req["suites"] == list(h.SUITES)
    assert req["contexts"] == {A: 4096, B: 16384}
    assert req["grants"] == {A: h.grant(A), B: None}
    toks = shlex.split(remote.split("&&")[-1])
    assert "--cache-dir" in toks and "--budget-bytes" in toks


def test_last_reply_line_is_read(tmp_path):
    run = h.Run(h.reply({A: h.rows_entry(1)}))
    assert set(_host(tmp_path, run)((A,), h.SUITES)) == {A}


def test_hosts_file_is_reread_and_missing_raises(tmp_path):
    run = h.Run(h.reply({A: h.rows_entry(1)}))
    host = _host(tmp_path, run)
    host((A,), h.SUITES)
    (tmp_path / "bench_hosts.toml").unlink()
    with pytest.raises(h.host_mod().LaneDBenchHostError):
        host((A,), h.SUITES)
    assert len(run.calls) == 1, "no host: nothing is run"
    h.hosts_file(tmp_path, stage="init")
    with pytest.raises(h.host_mod().LaneDBenchHostError):
        host((A,), h.SUITES)
    assert len(run.calls) == 1, "an init-only lane is never a bench host"


def test_digest_without_a_context_raises_before_ssh(tmp_path):
    run = h.Run(h.reply({}))
    host = _host(tmp_path, run)
    with pytest.raises(h.host_mod().LaneDBenchHostError):
        host(("c" * 64,), h.SUITES)
    assert run.calls == []


@pytest.mark.parametrize("bad", ["exit", "not_json", "schema", "rows_type", "extra_digest",
                                 "entry_type", "suite_type"])
def test_hostile_or_broken_reply_raises(tmp_path, bad):
    good = h.reply({A: h.rows_entry(1)})
    if bad == "exit":
        run = h.Run(good, returncode=1)
    elif bad == "not_json":
        run = h.Run(raw=b"{nope\n")
    else:
        r = json.loads(json.dumps(good))
        if bad == "schema":
            r["schema"] = "other/1"
        elif bad == "rows_type":
            r["rows"] = []
        elif bad == "extra_digest":
            r["rows"]["d" * 64] = h.rows_entry(9)
        elif bad == "entry_type":
            r["rows"][A] = "ok"
        else:
            r["rows"][A] = {"gift": ["not", "a", "mapping"]}
        run = h.Run(r)
    host = _host(tmp_path, run)
    with pytest.raises(h.host_mod().LaneDBenchHostError):
        host((A,), h.SUITES)
    assert len(run.calls) == 1


def test_a_raising_host_is_a_bench_hold_in_the_session(tmp_path):
    """Through W3b2's SweepSession: the sweep raises, the session raises, and
    settle_lane_d's gate turns that into bench_hold (lane_d._run_gate)."""
    from cascade.validator.lane_d_bench import SweepSession

    host = _host(tmp_path, h.Run(h.reply({})))
    (tmp_path / "bench_hosts.toml").unlink()
    s = SweepSession("lane-d-bench/1", host)
    with pytest.raises(h.host_mod().LaneDBenchHostError):
        s.paired(A, B, B, suites=h.SUITES)


def test_session_records_come_from_the_reply(tmp_path):
    from cascade.validator.lane_d_bench import SweepSession

    rows = {A: h.rows_entry(1), B: h.rows_entry(2)}
    s = SweepSession("lane-d-bench/7", _host(tmp_path, h.Run(h.reply(rows))))
    cand, king, house = s.paired(A, B, B, suites=h.SUITES)
    assert cand["suites"] == rows[A] and king["suites"] == rows[B] and house["suites"] == rows[B]


# ── the inputs TOML ─────────────────────────────────────────────────────────


def _inputs_toml(tmp_path, **extra) -> object:
    keys = {"intake_state_dir": "/i", "pool_pins": "/p.json", "house_pointer": "/h.json",
            "house_signer_ss58": "5x", "eval_hosts": "/e.toml", "eval_cache_dir": "/c",
            "readiness": "/r.json", "t7": "/t.json", **extra}
    lines = [f'{k} = "{v}"' for k, v in keys.items()] + ["eval_cache_budget_bytes = 10"]
    p = tmp_path / "inputs.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _cfg(tmp_path):
    from tests.accept.w8b import helpers as w8b

    return w8b.chain_cfg(tmp_path / "v")


def test_inputs_file_bench_hosts_key(tmp_path):
    m = __import__("cascade.validator.lane_d_inputs", fromlist=["x"])
    assert "bench_hosts" in m.INPUTS_FILE_KEYS, "W3b6a: the optional bench_hosts key"
    assert "bench_hosts" not in m._REQUIRED_KEYS, "absent bench_hosts keeps today's behaviour"
    hosts = h.hosts_file(tmp_path)
    got = m.load_inputs_services(_cfg(tmp_path), _inputs_toml(tmp_path, bench_hosts=str(hosts)))
    assert isinstance(got["bench_host"], h.host_mod().SSHBenchHost)


def test_inputs_file_without_bench_hosts_has_no_bench_host(tmp_path):
    m = __import__("cascade.validator.lane_d_inputs", fromlist=["x"])
    assert "bench_hosts" in m.INPUTS_FILE_KEYS, "W3b6a: the optional bench_hosts key"
    got = m.load_inputs_services(_cfg(tmp_path), _inputs_toml(tmp_path))
    assert got["bench_host"] is None


def test_source_hands_the_bench_host_each_digests_context(tmp_path):
    """The W8c source calls ``bench_host.set_contexts({digest: effective
    context})`` at every settlement, the same map it forecasts at."""
    from tests.accept.w8c import helpers as w8c

    class Bench:
        def __init__(self):
            self.contexts = []

        def set_contexts(self, mapping):
            self.contexts.append(dict(mapping))

        def __call__(self, digests, suites):
            raise AssertionError("not swept by the source")

    bench = Bench()
    rig = w8c.Rig(tmp_path)
    src = rig.build(bench_host=bench)
    src(w8c.B1)
    forecast_at = {c["digest"]: c["context"] for c in rig.eval_host.calls}
    assert bench.contexts == [forecast_at], \
        "set_contexts once per settlement with every needed digest's effective context"
