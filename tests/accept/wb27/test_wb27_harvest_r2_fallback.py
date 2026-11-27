"""WB27 incident fix (2026-11-27, owner: "should we have that safeguard?"): a
finished, paid lane B leg (8xH200 in Australia, ~14 GB checkpoint) was being
harvested tar-over-ssh at ~0.5 MB/s. The watchdog would kill the stream at
3600 s, the timeout was not retryable, the R2 fallback ran only on local-disk
faults, the leg was classed ``infra`` (burn) and the ``finally`` tore the pod
down — the finished run lost. And after a trainer restart the lane B
re-attach refused the operator-billed leg ("routed to operator capacity").

Pinned here:

* a watchdog breach (stall / trickle / early slow probe) raises
  ``HarvestStreamSlow`` with ``keep_pod`` while the pod still answers;
* ``guarded_harvest`` switches such a breach straight to the pod → R2
  transport; if that fails (or R2 is absent) the leg fails as an unburned
  operator fault that KEEPS the pod;
* ``[round] harvest_ssh_probe_seconds`` (the early switch) parses and
  round-trips through the loader;
* the re-attach of a leg already on a ledgered operator pod is allowed.

Torch-free: every PATHS module is imported inside a test body with torch
blocked."""

from __future__ import annotations

from types import SimpleNamespace

from tests.accept.wb27.helpers import CHAIN_TOML, torch_blocked


def _outcome(fn):
    """``("ok", result)`` or ``("raised", exc)`` — so a base without the fix
    fails these tests by ASSERTION, not by an unexpected exception."""
    try:
        return "ok", fn()
    except Exception as e:  # noqa: BLE001
        return "raised", e


def _runner(tmp_path, **round_kw):
    with torch_blocked():
        from cascade.shared.config.rounds import RoundConfig
    rnd = RoundConfig(min_free_disk_gb=0.0, expected_checkpoint_gb=0.0, **round_kw)
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    r.sleeps = []
    r._harvest_retry_sleep = r.sleeps.append
    return r


def _slow(msg="checkpoint harvest from p:/w/ck timed out: stream stalled", reachable=True):
    with torch_blocked():
        from cascade.trainer import remote
    cls = getattr(remote, "HarvestStreamSlow", None)
    if cls is None:     # base: the watchdog raised a plain RemoteDispatchError
        return remote.RemoteDispatchError(msg)
    return cls(msg, pod_reachable=reachable)


# ── remote: a watchdog breach of a reachable pod keeps it ───────────────────


def _serve(monkeypatch, remote, script: str, reachable=True):
    monkeypatch.setattr(remote, "build_ssh_argv", lambda h, c: ["sh", "-c", script])
    probes = []
    monkeypatch.setattr(remote, "remote_dir_reachable",
                        lambda h, d: (probes.append(d), reachable)[1])
    return probes


def test_stalled_stream_raises_slow_and_keeps_a_reachable_pod(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import harvest_guard, remote
    probes = _serve(monkeypatch, remote, "printf x; sleep 30")
    kind, e = _outcome(lambda: remote.harvest_remote_dir(
        remote.RemoteHost(name="p", host="h"), "/w/ck", tmp_path / "out",
        stall_timeout=0.4))
    assert kind == "raised" and isinstance(e, remote.RemoteDispatchError)
    assert getattr(e, "harvest_slow", False) is True
    assert getattr(e, "keep_pod", False) is True and probes == ["/w/ck"]
    assert harvest_guard.keeps_pod(e)
    assert not list(tmp_path.glob("out.partial-*"))


def test_slow_stream_on_an_unreachable_pod_is_not_kept(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import remote
    _serve(monkeypatch, remote, "printf x; sleep 30", reachable=False)
    kind, e = _outcome(lambda: remote.harvest_remote_dir(
        remote.RemoteHost(name="p", host="h"), "/w/ck", tmp_path / "out",
        stall_timeout=0.4))
    assert kind == "raised" and getattr(e, "harvest_slow", False) is True
    assert getattr(e, "pod_reachable", None) is False and not getattr(e, "keep_pod", False)


def test_early_slow_probe_abandons_a_trickle_well_before_the_wall(tmp_path, monkeypatch):
    """A stream still moving (never a stall) but far below 1 MB/s is dropped
    at ``slow_probe_seconds`` — not trickled to the 1 h wall."""
    with torch_blocked():
        from cascade.trainer import remote
    _serve(monkeypatch, remote, "while true; do printf x; sleep 0.05; done")
    import time

    t0 = time.monotonic()
    kind, e = _outcome(lambda: remote.harvest_remote_dir(
        remote.RemoteHost(name="p", host="h"), "/w/ck", tmp_path / "out",
        stall_timeout=30, timeout=60, slow_probe_seconds=0.5))
    took = time.monotonic() - t0
    assert kind == "raised" and getattr(e, "keep_pod", False) is True
    assert "slow stream" in str(e) and took < 20


# ── guarded_harvest: slow → R2 at once, else keep the pod unburned ──────────


def test_slow_stream_falls_back_to_r2_without_retrying_ssh(tmp_path):
    with torch_blocked():
        from cascade.trainer import harvest_guard
    r = _runner(tmp_path, harvest_disk_retries=3, harvest_disk_retry_seconds=5)
    calls = []

    def attempt():
        calls.append(1)
        raise _slow()

    kind, out = _outcome(lambda: harvest_guard.guarded_harvest(
        r, tmp_path / "d", attempt, label="hkA", r2_fallback=lambda: "via-r2"))
    assert (kind, out) == ("ok", "via-r2")
    assert calls == [1] and r.sleeps == []          # no ssh re-pull over the slow link


def test_r2_failure_after_a_slow_stream_keeps_the_pod_and_burns_nothing(tmp_path):
    with torch_blocked():
        from cascade.trainer import harvest_guard
        from cascade.trainer.faults import _OperatorFault, classify_leg_exception

    r = _runner(tmp_path, harvest_disk_retries=1)

    def attempt():
        raise _slow()

    def r2_down():
        raise RuntimeError("R2 presigned PUT failed: 503")

    for fallback in (r2_down, None):
        kind, e = _outcome(lambda fb=fallback: harvest_guard.guarded_harvest(
            r, tmp_path / "d", attempt, label="hkA", r2_fallback=fb))
        assert kind == "raised"
        assert harvest_guard.keeps_pod(e) and isinstance(e, _OperatorFault), repr(e)
        gen = SimpleNamespace(hotkey="hkA", ref="r@1")
        verdict = classify_leg_exception(r, gen, SimpleNamespace(base_seed=7), e, rc=None,
                                         text=str(e), stalled_before=False, shards_leg=False)
        assert verdict == (False, "operator", False)   # never a miner burn
    # held for the priced keep, capped at harvest_keep_max_seconds
    assert harvest_guard.keep_budget_seconds(r, {"trained_seconds": 12 * 3600}) == \
        r.cfg.round.harvest_keep_max_seconds


# ── the knob: loader parse + round trip ─────────────────────────────────────


def test_harvest_ssh_probe_seconds_parses_and_round_trips(tmp_path):
    with torch_blocked():
        from cascade.shared.config import load_chain_config
        from cascade.shared.config.money import MONEY_KNOBS
        from cascade.shared.config.rounds import RoundConfig

    assert MONEY_KNOBS.get("harvest_ssh_probe_seconds", (None,))[0] == 600
    assert getattr(RoundConfig(), "harvest_ssh_probe_seconds", None) == 600
    assert getattr(load_chain_config(CHAIN_TOML).round, "harvest_ssh_probe_seconds",
                   None) == 600
    src = CHAIN_TOML.read_text(encoding="utf-8")
    anchor = "funded_operator_fallback = true\n"
    assert src.count(anchor) == 1
    for value, want in (("120", 120), ("0", 0)):
        p = tmp_path / f"chain-{value}.toml"
        p.write_text(src.replace(anchor, anchor + f"harvest_ssh_probe_seconds = {value}\n"),
                     encoding="utf-8")
        assert getattr(load_chain_config(p).round, "harvest_ssh_probe_seconds", None) == want
    bad = tmp_path / "chain-bad.toml"
    bad.write_text(src.replace(anchor, anchor + "harvest_ssh_probe_seconds = -5\n"),
                   encoding="utf-8")
    kind, _ = _outcome(lambda: load_chain_config(bad))
    assert kind == "raised"


def test_probe_is_passed_to_ssh_only_when_r2_can_take_over(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import harvest_guard, loop
    seen = {}

    def ssh_attempt(self, host, receipt, contract, seeds, suffix, *, release=None, **kw):
        seen["probe"] = kw.get("slow_probe_seconds", 0.0)
        return "entry"

    monkeypatch.setattr(loop.TrainerRunner, "_harvest_funded_checkpoint_ssh", ssh_attempt)
    monkeypatch.setattr(loop, "harvest_dir", lambda *a, **k: tmp_path / "d")
    monkeypatch.setattr(harvest_guard, "_room", lambda *a, **k: 0)
    r = _runner(tmp_path, harvest_ssh_probe_seconds=300) if hasattr(
        _runner(tmp_path).cfg.round, "harvest_ssh_probe_seconds") else _runner(tmp_path)
    r.cfg.storage = SimpleNamespace(backup_s3_endpoint="")
    receipt = SimpleNamespace(marketcast=None, miner_hotkey="hkA" * 4, job_id="j1",
                              checkpoint_dir="ck")
    host = SimpleNamespace(name="p", allocation_id="pod-1")
    seeds = SimpleNamespace(base_seed=7)
    for available, want in ((True, 300.0), (False, 0.0)):
        monkeypatch.setattr(loop, "_r2_fallback_available", lambda _r, a=available: a)
        seen.clear()
        assert loop.TrainerRunner._harvest_funded_checkpoint(
            r, host, receipt, SimpleNamespace(arch_preset="x"), seeds, "") == "entry"
        assert seen["probe"] == want, (available, seen)


# ── re-attach of a finished operator-billed leg ─────────────────────────────


def test_reattach_of_a_ledgered_operator_leg_is_not_refused(tmp_path):
    with torch_blocked():
        from cascade.trainer import runner_funded_legs as rfl
        from cascade.trainer.faults import _FundedOperatorFallback, _FundedOperatorRent
    decide = getattr(rfl, "_reattach_operator_leg", None)
    assert decide is not None, "no re-attach path for operator-billed legs"
    pod = SimpleNamespace(instance_id="op-leg-777-hkA-0")
    runner = SimpleNamespace(_operator_leg_pod_prefix=lambda rid, hk: f"op-leg-{rid}-{hk}",
                             _load_funded_ledger=lambda: [pod])
    gen = SimpleNamespace(hotkey="hkA")
    seeds = SimpleNamespace(base_seed=777)
    rent = _FundedOperatorRent("hkA")
    assert decide(runner, rent, gen, seeds, role="expert", bench=False) is True
    # never a NEW operator rent / pool dispatch on a re-attach
    assert decide(runner, _FundedOperatorFallback("hkA"), gen, seeds, role="expert",
                  bench=False) is False
    assert decide(runner, rent, gen, seeds, role="challenger", bench=True) is False
    empty = SimpleNamespace(_operator_leg_pod_prefix=runner._operator_leg_pod_prefix,
                            _load_funded_ledger=lambda: [])
    assert decide(empty, rent, gen, seeds, role="expert", bench=False) is False


def test_reattach_never_rents_a_fresh_operator_pod():
    """Under ``reattach_only`` the operator rent path adopts the ledgered pod
    or refuses — the source carries the guard before any rent / lemon."""
    import inspect

    with torch_blocked():
        from cascade.trainer.runner_operator_pods import _OperatorPodMixin
    src = inspect.getsource(_OperatorPodMixin._rent_operator_leg_host)
    guard = src.find("if attach_only():")
    assert guard != -1, "no attach-only guard in the operator rent path"
    assert guard < src.find("_lemon(\"adopted operator leg pod unusable\")")
    assert src.count("if attach_only():") >= 2
    assert src.rfind("if attach_only():") < src.find("measured_wall_refusal(rnd)")
