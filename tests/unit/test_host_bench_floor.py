"""Host-bench floor (``[round] funded_host_bench_floor``): a freshly rented pod
slower than the round SKU's floor is released and the leg rents again on the
bad-pod budget; the JIT king gets the same gate on a fresh rent; a probe that
cannot be measured never rejects; the ssh probe parses the pod's own facts."""
import subprocess
from types import SimpleNamespace

import pytest

from cascade.trainer import remote as remote_mod
from cascade.trainer.remote import RemoteHost, probe_host_bench
from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

FLOOR = (("RTX4090", 450e6),)


def _host(**kw) -> RemoteHost:
    base = dict(name="h", host="10.0.0.1", port=22, user="root", key_path="/k",
                remote_python="/root/cascade/.venv/bin/python", workdir="/root/cascade",
                cuda_device="1")
    base.update(kw)
    return RemoteHost(**base)


# ── the ssh probe ───────────────────────────────────────────────────────────

def test_probe_host_bench_runs_the_pods_own_bench_on_the_lane(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 0, stdout="cuda banner\nHOSTBENCH {\"host_bench_tokens_per_s\": 393688672.7, "
                            "\"host_bench_device\": \"cuda\"}\n", stderr="")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake_run)
    tps, err = probe_host_bench(_host())
    assert (tps, err) == (393688672.7, "")
    cmd = seen["argv"][-1]
    assert "cd /root/cascade" in cmd and "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "cascade.trainer.host_probe import host_bench" in cmd


@pytest.mark.parametrize("outcome, needle", [
    ("rc", "failed rc=2"),
    ("timeout", "timed out"),
    ("silent", "printed no result"),
    ("garbage", "malformed"),
])
def test_probe_host_bench_reports_failures_instead_of_raising(monkeypatch, outcome, needle):
    def fake_run(argv, **kw):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, kw.get("timeout", 1))
        if outcome == "rc":
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="boom")
        if outcome == "silent":
            return subprocess.CompletedProcess(argv, 0, stdout="nothing here\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="HOSTBENCH {\"nope\": 1}\n", stderr="")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake_run)
    tps, err = probe_host_bench(_host())
    assert tps is None and needle in err


# ── the verdict ─────────────────────────────────────────────────────────────

def test_floor_off_never_probes(tmp_path, monkeypatch):
    runner = _runner(tmp_path)                                  # no floor at all
    monkeypatch.setattr(remote_mod, "probe_host_bench",
                        lambda host, **kw: pytest.fail("probed with the floor off"))
    assert runner._host_bench_below_floor(_host(), "RTX4090", "x") == ""
    other = _runner(tmp_path, funded_host_bench_floor=FLOOR)   # armed for another SKU
    assert other._host_bench_below_floor(_host(), "A6000", "x") == ""


@pytest.mark.parametrize("measured, expect_reject", [
    (393688672.7, True),        # the 2026-09-15 king host
    (450e6, False),             # exactly the floor passes
    (646202582.7, False),
])
def test_floor_rejects_only_a_measured_slow_host(tmp_path, monkeypatch, measured, expect_reject):
    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    monkeypatch.setattr(remote_mod, "probe_host_bench", lambda host, **kw: (measured, ""))
    why = runner._host_bench_below_floor(_host(), "rtx4090", "funded pod p-0")
    if expect_reject:
        assert why.startswith("slow host:") and "393,688,673" in why and "450,000,000" in why
    else:
        assert why == ""


def test_unmeasurable_probe_lets_the_pod_through(tmp_path, monkeypatch):
    from cascade.trainer import loop as loop_mod

    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    monkeypatch.setattr(remote_mod, "probe_host_bench",
                        lambda host, **kw: (None, "host bench probe timed out after 120s"))
    # Capture on the module logger itself: caplog's root handler misses it
    # when an earlier test in the session disables propagation.
    warned = []
    monkeypatch.setattr(loop_mod.log, "warning",
                        lambda msg, *a, **kw: warned.append(msg % a if a else msg))
    assert runner._host_bench_below_floor(_host(), "RTX4090", "funded pod p-0") == ""
    assert any("not enforced" in w and "timed out" in w for w in warned)


def test_funded_pod_too_slow_pins_the_rented_pods_identity(tmp_path, monkeypatch):
    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    seen = {}

    def fake_probe(host, **kw):
        seen["host"] = host
        return 100e6, ""

    monkeypatch.setattr(remote_mod, "probe_host_bench", fake_probe)
    result = _rent_ok()
    profile = SimpleNamespace(user="root", key_path="/k", remote_python="/p",
                              workdir="/root/cascade", ssh_options=("StrictHostKeyChecking=yes",))
    why = runner._funded_pod_too_slow(result, profile)
    assert why.startswith("slow host:")
    h = seen["host"]
    assert (h.host, h.port) == (result.address.ip, result.address.ssh_port)
    assert h.pinned_host_key == result.host_key and h.cuda_device == "0"


# ── the rent loop ───────────────────────────────────────────────────────────

def test_slow_funded_pod_is_released_and_the_leg_rents_again(tmp_path, monkeypatch):
    import cascade.provision.core as core_mod
    from cascade.provision import funded as funded_mod

    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    _vault(tmp_path, "hkA")
    rents, torn, quarantined = [], [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    monkeypatch.setattr(core_mod, "record_host_quarantine",
                        lambda ip, why: quarantined.append(ip))
    runner._funded_pod_code_mismatch = lambda result, profile: ""
    verdicts = ["slow host: calibration bench 393,688,673 tokens/s < floor 450,000,000 for RTX4090", ""]
    runner._funded_pod_too_slow = lambda result, profile: verdicts.pop(0)
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == 2                          # first pod rejected, second served
    assert torn == [pod.instance_id]                # the slow pod was released
    assert host.host == "10.9.9.9"
    assert "hkA" not in runner._funded_leg_failures  # nothing recorded against the miner
    assert quarantined == []                        # slowness is per card: no host quarantine


def test_an_adopted_pod_skips_the_throughput_gate(tmp_path, monkeypatch):
    # 2026-09-24 12:59: the pod adopted on restart was mid-leg; the calibration
    # bench read 483M on a busy RTX4090 (floor 500M) and the trainer tore the
    # 3 h leg down. An adopted pod passed the gate when it was rented.
    from dataclasses import replace

    from cascade.provision import funded as funded_mod

    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    _vault(tmp_path, "hkA")
    rents, torn, probed = [], [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: rents.append(1) or replace(_rent_ok(), adopted=True))
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    runner._funded_pod_code_mismatch = lambda result, profile: ""

    def too_slow(result, profile):
        probed.append(1)
        return "slow host: calibration bench 483,426,297 tokens/s < floor 500,000,000 for RTX4090"

    runner._funded_pod_too_slow = too_slow
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert rents == [1] and probed == [] and torn == []   # never benched, never released
    assert host.host == "10.9.9.9"


def test_repeated_slow_pods_settle_infra_without_a_burn(tmp_path, monkeypatch):
    from cascade.provision import funded as funded_mod
    from cascade.trainer.loop import TrainerRunner, _FundedLegSkip

    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    _vault(tmp_path, "hkA")
    rents, torn = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    runner._funded_pod_code_mismatch = lambda result, profile: ""
    runner._funded_pod_too_slow = lambda result, profile: "slow host: 1 tokens/s < floor 2 for RTX4090"
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == len(torn) == TrainerRunner.FUNDED_MAX_STALE_PODS
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "infra", False)
    assert "slow host" in msg


def test_stale_image_still_wins_over_the_floor(tmp_path, monkeypatch):
    """A stale pod is rejected for its image (host quarantined) and never
    benched — the fingerprint verdict stays the first word."""
    from cascade.provision import funded as funded_mod

    runner = _runner(tmp_path, funded_host_bench_floor=FLOOR)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded", lambda pods, vault, **kw: [])
    verdicts = ["stale worker image: x != y", ""]
    runner._funded_pod_code_mismatch = lambda result, profile: verdicts.pop(0)
    probed = []
    runner._funded_pod_too_slow = lambda result, profile: probed.append(1) or ""
    runner._rent_funded_host("777", _challenger("hkA"))
    assert probed == [1]                            # only the pinned second pod was benched


# ── the JIT king ────────────────────────────────────────────────────────────

def test_king_fresh_rent_is_gated_by_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    import cascade.provision.core as core_mod
    from cascade.provision import funded as funded_mod

    r = _runner(tmp_path, funded_king_rent=True, funded_host_bench_floor=FLOOR)
    r._funded_round_sku = "RTX4090"
    r._rent_wait_now = lambda: 0.0                  # far before the latest safe start ⇒ retry
    r._rent_wait_sleep = lambda s: None
    verdicts = ["slow host: calibration bench 393,688,673 tokens/s < floor 450,000,000 for RTX4090", ""]
    r._host_bench_below_floor = lambda host, sku, label: verdicts.pop(0)
    launched, terminated, lemons = [], [], []
    monkeypatch.setattr(funded_mod, "quarantine_lemon_host",
                        lambda provider, pod_id, why: lemons.append((pod_id, why)))
    monkeypatch.setattr(funded_mod, "terminate_verified",
                        lambda provider, pod_id: terminated.append(pod_id) or True)

    class _Prov:
        name = "lium"
        def capacity(self, sku, *, gpus=1, exclude_ids=()):
            return 1
        def launch(self, spec):
            launched.append(spec.sku)
            return [f"{spec.name_prefix}-0"]
        def wait_ready(self, pod_id, *, timeout):
            return True
        def get_ip(self, pod_id):
            return SimpleNamespace(ip="9.9.9.9", ssh_port=41000)
        def machine_of(self, pod_id):
            return f"exec-{len(launched)}"
        def terminate(self, pod_id):
            pass

    orig = core_mod.LiumProvider
    core_mod.LiumProvider = _Prov
    try:
        host = r._rent_king_host("42")
    finally:
        core_mod.LiumProvider = orig
    assert launched == ["RTX4090", "RTX4090"]        # slow pod released, rented again
    assert terminated == ["cascade-n91-42-funded-king-0"]
    assert lemons and "slow host" in lemons[0][1]
    assert (host.host, host.port) == ("9.9.9.9", 41000)
    assert verdicts == []
