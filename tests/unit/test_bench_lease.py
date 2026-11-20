"""The elastic bench host's lease (cascade.validator.bench_lease), stressed.

Every test drives the real :class:`BenchLease` against a scripted provider,
a scripted SSH and a hand-moved clock: slow pods, no capacity, lemons, old
drivers, a bootstrap that fails, a pod that dies, a validator that restarts
mid-rent. What must never happen: a second pod for one need, a pod left
billing after a failure, a pod torn down under a running sweep, or a wait
that outlives its bound.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.provision.core import PodAddress
from cascade.validator.bench_lease import (
    MAX_HOST_LOSSES,
    PHASE_IDLE,
    PHASE_READY,
    PHASE_RENTING,
    BenchLease,
    BenchLeaseConfig,
    BenchLeaseConfigError,
    LeaseState,
    build_bench_lease,
    check_bench_lease_config,
    fingerprint,
    hosts_toml,
    load_bench_lease_config,
)

IMAGE = "ghcr.io/x/cascade@sha256:" + "a" * 64
PROBE_OK = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 580.173.02\n" \
           '["2.12.1+cu130", "13.0"]\n'
HW_OK = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition|driver 580|torch 2.12.1+cu130 cuda 13.0"


def _offer(i, *, vram=96, cuda="13.0", price=1.6):
    return {"id": f"ex{i}", "vram_gb": vram, "max_cuda_version": cuda,
            "price_per_hour": price, "gpu_count": 1}


class FakeProvider:
    """Scripted Lium: offers, launches, readiness, termination — all logged."""

    def __init__(self, offers=None, *, ready=True, launch_error=None, ip="10.0.0.9"):
        self.offers = offers if offers is not None else [("Edition", _offer(1))]
        self.ready = ready                     # bool, or a list consumed per wait_ready
        self.launch_error = launch_error
        self.ip = ip
        self.launched: list = []
        self.terminated: list[str] = []
        self.live: set[str] = set()
        self.terminate_error = None

    def list_offers(self, skus, *, gpus=1, exclude_ids=()):
        return [(s, e) for s, e in self.offers if s in skus]

    def launch(self, spec):
        if self.launch_error is not None:
            raise self.launch_error
        self.launched.append(spec)
        self.live.add(spec.name_prefix)
        return [spec.name_prefix]

    def wait_ready(self, pod, *, timeout):
        if isinstance(self.ready, list):
            return self.ready.pop(0) if self.ready else False
        return self.ready

    def get_ip(self, pod):
        return PodAddress(self.ip, 2222)

    def sku_of(self, pod):
        return "Edition"

    def terminate(self, pod):
        if self.terminate_error is not None:
            raise self.terminate_error
        self.terminated.append(pod)
        self.live.discard(pod)


class FakeSSH:
    """Scripted SSH: per-command-kind return codes / stdout."""

    def __init__(self, *, boot_rc=0, probe=PROBE_OK, reachable=True, busy=False):
        self.boot_rc = boot_rc
        self.probe = probe
        self.reachable = reachable
        self.busy = busy
        self.calls: list[tuple[str, str]] = []
        self.inputs: list[str | None] = []

    def __call__(self, ip, port, command, timeout, input=None):
        self.calls.append((ip, command))
        self.inputs.append(input)

        def done(rc, out=""):
            return subprocess.CompletedProcess([command], rc, out, "")

        if command == "true":
            return done(0 if self.reachable else 255)
        if command.startswith("pgrep"):
            return done(0 if self.busy else 1)
        if command.startswith("nvidia-smi"):
            return done(0, self.probe)
        if isinstance(self.boot_rc, Exception):
            raise self.boot_rc
        return done(self.boot_rc)


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += minutes * 60.0


def _cfg(tmp_path, **kw) -> BenchLeaseConfig:
    pub = tmp_path / "id.pub"
    pub.write_text("ssh-ed25519 AAAA test\n")
    script = tmp_path / "boot.sh"
    script.write_text("echo bootstrap\n")
    base = BenchLeaseConfig(hosts_path=tmp_path / "eval-hosts.toml",
                            state_path=tmp_path / "lease.json", image=IMAGE,
                            ssh_pubkey_path=pub, gpu_types=("RTX5090", "Edition"),
                            bootstrap_script=str(script))
    return replace(base, **kw)


def _lease(tmp_path, provider=None, ssh=None, clock=None, **kw):
    provider = provider or FakeProvider()
    ssh = ssh or FakeSSH()
    clock = clock or Clock()
    cfg = _cfg(tmp_path, **kw)
    lease = BenchLease(cfg=cfg, provider=provider, ssh=ssh, clock=clock,
                       spawn=lambda fn: fn())
    return lease, provider, ssh, clock


# ── the happy path ───────────────────────────────────────────────────────────


def test_first_need_rents_bootstraps_probes_and_publishes(tmp_path):
    lease, prov, ssh, _ = _lease(tmp_path)
    assert lease.need() is False             # the caller holds on the first ask
    assert lease.ready() and lease.hw() == HW_OK
    assert len(prov.launched) == 1
    spec = prov.launched[0]
    assert spec.image == IMAGE and spec.count == 1 and spec.gpus_per_pod == 1
    assert spec.ttl_seconds == pytest.approx(12 * 3600)
    # the local bootstrap script is piped to the pod (the image predates it)
    assert "echo bootstrap\n" in ssh.inputs
    text = lease.cfg.hosts_path.read_text()
    assert 'host = "10.0.0.9"' in text and "port = 2222" in text and 'stage = "any"' in text
    assert lease.need() is True              # next poll: go
    assert json.loads(lease.cfg.state_path.read_text())["phase"] == PHASE_READY


def test_hosts_file_parses_with_the_validators_own_loader(tmp_path):
    from cascade.trainer.remote import load_hosts

    cfg = _cfg(tmp_path, key_path="~/.ssh/k")
    p = tmp_path / "h.toml"
    p.write_text(hosts_toml(cfg, name="cascade-benchlease-1", ip="1.2.3.4", port=40022))
    (h,) = load_hosts(p)
    assert (h.host, h.port, h.stage, h.workdir) == ("1.2.3.4", 40022, "any", "/root/cascade")


def test_cheapest_fitting_offer_wins_and_unfit_executors_are_excluded(tmp_path):
    offers = [("RTX5090", _offer(1, vram=32, price=0.7)),
              ("RTX5090", _offer(2, vram=32, cuda="12.8", price=0.5)),   # old driver
              ("Edition", _offer(3, price=1.5)),
              ("Edition", _offer(4, price=9.0))]                        # over the cap
    lease, prov, _, _ = _lease(tmp_path, FakeProvider(offers))
    lease.need()
    spec = prov.launched[0]
    assert spec.sku == "RTX5090"
    assert "ex2" in spec.exclude_ids and "ex1" not in spec.exclude_ids


# ── slow / failing rents: hold, back off, never leak a pod ──────────────────


def test_no_capacity_backs_off_and_expires_only_after_the_bound(tmp_path):
    lease, prov, _, clock = _lease(tmp_path, FakeProvider(offers=[]), max_wait_hours=2.0)
    assert lease.need() is False and not lease.expired()
    assert prov.launched == [] and "no RTX5090/Edition offer" in lease.state.last_error
    for _ in range(11):                       # 110 min of polls
        clock.advance(10)
        lease.tick()
        assert not lease.expired()
    clock.advance(10)                          # 120 min
    assert lease.expired()
    assert lease.state.attempts >= 12 and prov.launched == []


def test_retry_waits_for_the_backoff(tmp_path):
    lease, prov, _, clock = _lease(tmp_path, FakeProvider(offers=[]), retry_minutes=10)
    lease.need()
    n = lease.state.attempts
    clock.advance(3)
    lease.tick()
    lease.need()
    assert lease.state.attempts == n          # inside the backoff: no new attempt
    clock.advance(8)
    lease.tick()
    assert lease.state.attempts == n + 1


def test_lemon_that_never_boots_is_terminated_then_a_retry_succeeds(tmp_path):
    prov = FakeProvider(ready=[False, True])
    lease, prov, _, clock = _lease(tmp_path, prov)
    lease.need()
    assert not lease.ready() and prov.terminated == [prov.launched[0].name_prefix]
    clock.advance(11)
    lease.tick()
    assert lease.ready() and len(prov.launched) == 2
    assert prov.live == {prov.launched[1].name_prefix}     # exactly one pod billing


def test_failed_bootstrap_terminates_the_pod(tmp_path):
    lease, prov, _, _ = _lease(tmp_path, ssh=FakeSSH(boot_rc=5))
    lease.need()
    assert not lease.ready() and prov.live == set()
    assert "bootstrap rc=5" in lease.state.last_error
    assert not lease.cfg.hosts_path.exists()      # never published


def test_bootstrap_timeout_terminates_the_pod(tmp_path):
    ssh = FakeSSH(boot_rc=subprocess.TimeoutExpired("ssh", 3600))
    lease, prov, _, _ = _lease(tmp_path, ssh=ssh)
    lease.need()
    assert not lease.ready() and prov.live == set()
    assert "TimeoutExpired" in lease.state.last_error


def test_old_driver_on_the_booted_pod_is_rejected(tmp_path):
    ssh = FakeSSH(probe="NVIDIA GeForce RTX 5090, 570.86.10\n[\"2.12.1+cu130\", \"13.0\"]\n")
    lease, prov, _, _ = _lease(tmp_path, ssh=ssh)
    lease.need()
    assert not lease.ready() and prov.live == set()
    assert "driver 570 < 580" in lease.state.last_error


def test_unusable_probe_is_a_failed_rent(tmp_path):
    lease, prov, _, _ = _lease(tmp_path, ssh=FakeSSH(probe="garbage"))
    lease.need()
    assert not lease.ready() and prov.live == set()


def test_launch_error_backs_off_without_terminating_anything_live(tmp_path):
    prov = FakeProvider(launch_error=RuntimeError("402 insufficient balance"))
    lease, prov, _, _ = _lease(tmp_path, prov)
    lease.need()
    assert lease.state.phase == PHASE_IDLE and "402" in lease.state.last_error
    assert prov.live == set()


def test_a_failing_terminate_never_raises(tmp_path):
    prov = FakeProvider(ready=False)
    prov.terminate_error = RuntimeError("lium down")
    lease, prov, _, _ = _lease(tmp_path, prov)
    lease.need()                                # must not raise
    assert lease.state.phase == PHASE_IDLE


def test_a_need_while_a_rent_is_in_flight_never_double_rents(tmp_path):
    started = []
    cfg = _cfg(tmp_path)
    prov = FakeProvider()
    lease = BenchLease(cfg=cfg, provider=prov, ssh=FakeSSH(), clock=Clock(),
                       spawn=lambda fn: started.append(fn))   # worker not run yet
    for _ in range(5):
        assert lease.need() is False
        lease.tick()
    assert len(started) == 1 and lease.state.phase == PHASE_RENTING
    started[0]()
    assert lease.ready() and len(prov.launched) == 1


# ── losing a host ────────────────────────────────────────────────────────────


def test_lost_host_is_terminated_and_replaced_with_one_fresh_wait(tmp_path):
    lease, prov, _, clock = _lease(tmp_path)
    lease.need()
    first = prov.launched[0].name_prefix
    clock.advance(60)
    lease.host_lost("ssh: connection refused")
    assert first in prov.terminated
    assert lease.ready() and len(prov.launched) == 2 and prov.live == {
        prov.launched[1].name_prefix}
    assert lease.state.losses == 1


def test_repeated_losses_stop_granting_fresh_waits(tmp_path):
    lease, prov, _, clock = _lease(tmp_path, FakeProvider(ready=[True] + [False] * 50))
    lease.need()
    lease.host_lost("gone")               # loss 1: fresh wait, replacement fails to boot
    assert not lease.expired()
    lease.state = replace(lease.state, phase=PHASE_READY, pod_id="p", ip="x", port=1,
                          hw=HW_OK)
    lease.host_lost("gone again")          # loss 2: the wait starts already spent
    assert lease.state.losses == MAX_HOST_LOSSES and lease.expired()


def test_a_good_sweep_resets_the_loss_budget(tmp_path):
    lease, _, _, _ = _lease(tmp_path)
    lease.need()
    lease.host_lost("x")
    lease.need()
    lease.sweep_ok()
    assert lease.state.losses == 0


# ── release ──────────────────────────────────────────────────────────────────


def test_idle_pod_is_released_after_the_idle_window(tmp_path):
    lease, prov, _, clock = _lease(tmp_path, idle_release_minutes=20)
    lease.need()
    lease.need()
    clock.advance(19)
    lease.tick()
    assert lease.ready()
    clock.advance(2)
    lease.tick()
    assert lease.state.phase == PHASE_IDLE and prov.live == set()
    assert "[[host]]" not in lease.cfg.hosts_path.read_text()


def test_pod_running_a_sweep_is_never_released(tmp_path):
    ssh = FakeSSH(busy=True)
    lease, prov, _, clock = _lease(tmp_path, ssh=ssh, idle_release_minutes=20)
    lease.need()
    lease.need()
    clock.advance(30)
    lease.tick()
    assert lease.ready() and prov.terminated == []
    ssh.busy = False
    clock.advance(21)
    lease.tick()
    assert prov.live == set()


def test_tick_without_a_need_never_rents(tmp_path):
    lease, prov, _, clock = _lease(tmp_path)
    for _ in range(10):
        clock.advance(30)
        lease.tick()
    assert prov.launched == []


def test_an_expired_need_ends_and_stops_retrying(tmp_path):
    lease, prov, _, clock = _lease(tmp_path, FakeProvider(offers=[]), max_wait_hours=1.0)
    lease.need()
    clock.advance(61)
    assert lease.expired()
    lease.end_need()
    n = lease.state.attempts
    for _ in range(6):
        clock.advance(20)
        lease.tick()
    assert lease.state.attempts == n and not lease.expired()


# ── restarts ─────────────────────────────────────────────────────────────────


def test_restart_mid_rent_terminates_the_orphan(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_path.write_text(LeaseState(phase=PHASE_RENTING, pod_id="cascade-benchlease-7",
                                         requested_at=1.0).to_json())
    prov = FakeProvider()
    lease = BenchLease(cfg=cfg, provider=prov, ssh=FakeSSH(), clock=Clock(),
                       spawn=lambda fn: fn())
    assert prov.terminated == ["cascade-benchlease-7"] and lease.state.phase == PHASE_IDLE
    assert lease.state.requested_at == 1.0          # the wait keeps its original start


def test_restart_resumes_a_live_ready_pod(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_path.write_text(LeaseState(phase=PHASE_READY, pod_id="p1", ip="9.9.9.9",
                                         port=22, hw=HW_OK).to_json())
    prov = FakeProvider()
    lease = BenchLease(cfg=cfg, provider=prov, ssh=FakeSSH(), clock=Clock(),
                       spawn=lambda fn: fn())
    assert lease.ready() and prov.terminated == []
    assert 'host = "9.9.9.9"' in cfg.hosts_path.read_text()


def test_restart_drops_a_dead_ready_pod(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_path.write_text(LeaseState(phase=PHASE_READY, pod_id="p1", ip="9.9.9.9",
                                         port=22, hw=HW_OK).to_json())
    prov = FakeProvider()
    lease = BenchLease(cfg=cfg, provider=prov, ssh=FakeSSH(reachable=False), clock=Clock(),
                       spawn=lambda fn: fn())
    assert lease.state.phase == PHASE_IDLE and prov.terminated == ["p1"]
    assert "[[host]]" not in cfg.hosts_path.read_text()


def test_corrupt_state_starts_idle(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_path.write_text("{not json")
    lease = BenchLease(cfg=cfg, provider=FakeProvider(), ssh=FakeSSH(), clock=Clock(),
                       spawn=lambda fn: fn())
    assert lease.state.phase == PHASE_IDLE


# ── fingerprint ──────────────────────────────────────────────────────────────


def test_fingerprint_is_model_driver_major_and_torch_build():
    assert fingerprint(PROBE_OK) == HW_OK
    point = PROBE_OK.replace("580.173.02", "580.95.05")
    assert fingerprint(point) == HW_OK                       # point release: same kernels
    assert fingerprint(PROBE_OK.replace("580.173.02", "590.1")) != HW_OK
    assert fingerprint(PROBE_OK.replace("Workstation Edition", "Server Edition")) != HW_OK
    with pytest.raises(ValueError):
        fingerprint("only one line\n")


# ── config ───────────────────────────────────────────────────────────────────


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "bench_lease.toml"
    p.write_text(body)
    return p


BASE_TOML = f'''
[bench_lease]
hosts_path = "/root/cascade/bench-eval-hosts.toml"
state_path = "/root/cascade/bench_lease_state.json"
image = "{IMAGE}"
ssh_pubkey_path = "~/.ssh/id_ed25519.pub"
gpu_types = ["RTX5090", "Edition"]
max_price_per_hour = 1.8
max_wait_hours = 2.0
'''


def test_config_round_trip(tmp_path):
    cfg = load_bench_lease_config(_write(tmp_path, BASE_TOML + "idle_release_minutes = 15\n"
                                         "min_driver = 575\n"))
    assert cfg.gpu_types == ("RTX5090", "Edition")
    assert cfg.max_price_per_hour == 1.8 and cfg.idle_release_minutes == 15.0
    assert cfg.min_driver == 575 and cfg.hosts_path == Path(
        "/root/cascade/bench-eval-hosts.toml")


@pytest.mark.parametrize("extra,match", [
    ('typo_key = 1\n', "unknown keys"),
    ('max_wait_hours = 0\n', "must be > 0"),
    ('max_wait_hours = 18\n', "absent deadline"),
    ('ttl_hours = 3\n', "ttl_hours"),
])
def test_config_refusals(tmp_path, extra, match):
    body = BASE_TOML.replace("max_wait_hours = 2.0\n", "") + extra
    if "max_wait_hours" not in extra:
        body += "max_wait_hours = 2.0\n"
    with pytest.raises(BenchLeaseConfigError, match=match):
        load_bench_lease_config(_write(tmp_path, body))


def test_config_refuses_an_unpinned_image(tmp_path):
    with pytest.raises(BenchLeaseConfigError, match="digest-pinned"):
        load_bench_lease_config(_write(tmp_path, BASE_TOML.replace(IMAGE, "ghcr.io/x:latest")))


def test_wait_bound_tracks_the_deployed_report_timeout(tmp_path):
    cfg = _cfg(tmp_path, max_wait_hours=2.0)
    # mainnet: 8 settlements × 3 h = 24 h; 2 + 2×2 + 3 = 9 h fits
    check_bench_lease_config(cfg, epoch_blocks=900, block_seconds=12.0,
                             report_timeout_settlements=8, sweep_hours=2.0)
    # a 2-settlement timeout (6 h) cannot hold a 2 h wait + two sweeps + slack
    with pytest.raises(BenchLeaseConfigError, match="absent deadline"):
        check_bench_lease_config(cfg, epoch_blocks=900, block_seconds=12.0,
                                 report_timeout_settlements=2, sweep_hours=2.0)


def test_build_refuses_a_non_main_validator_and_a_mismatched_hosts_file(tmp_path, cfg):
    p = _write(tmp_path, BASE_TOML)
    with pytest.raises(BenchLeaseConfigError, match="MAIN validator only"):
        build_bench_lease(p, cfg, eval_hosts="/root/cascade/bench-eval-hosts.toml",
                          is_main=False, provider=FakeProvider())
    with pytest.raises(BenchLeaseConfigError, match="same file as --eval-hosts"):
        build_bench_lease(p, cfg, eval_hosts="/root/cascade/targon-eval.toml",
                          is_main=True, provider=FakeProvider())
