"""Miner-cost protections (2026-10-31 incident).

v2 funded challenger legs trained on their miners' own Lium keys at 0.13 steps/s
against a wall budget that assumed 2.86 (``funded_sku_wall_seconds`` "8xH200"
was never measured on the pinned image): every leg would be wall-capped at
~10 % by ``max_train_seconds``; four pods billed ~3 h each before anyone
noticed. A trainer stop also left ``cgen-*`` CPU VMs billing, and a key went
'Invalid API key' under a live pod. Five guards:

1. a progress watchdog per funded leg ("THROUGHPUT ABORT", operator fault);
2. a measured-wall gate (``funded_sku_wall_measured_image``);
3. no doomed starts (projected end past the target settlement);
4. a provider-side TTL on every rent (Lium ``--ttl``, Shadeform auto_delete);
5. a teardown-key check ("PAYER KEY CANNOT STOP POD").
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.provision.funded as funded_mod
import cascade.trainer.remote as remote_mod
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.trainer.cost_guards import (
    PAYER_KEY_CANNOT_STOP,
    THROUGHPUT_ABORT,
    ProgressWatchdog,
    check_payer_key,
    doomed_leg_reason,
    generation_ttl_seconds,
    measured_wall_refusal,
    payer_key_listing_error,
    provider_ttl_seconds,
)
from cascade.trainer.loop import _FundedLegSkip
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    LegAborted,
    RemoteDispatcher,
    RemoteHost,
    parse_progress,
    run_detached,
)
from tests.unit.test_funded_pod_wiring import (
    _challenger,
    _leg_runner,
    _rent_ok,
    _runner,
    _vault,
)

ROOT = Path(__file__).resolve().parents[2]
DIGEST_A = "sha256:" + "a" * 64
DIGEST_C = "sha256:" + "c" * 64          # the _runner fixture's funded_pod_image digest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)


def _with_round(text: str, extra: str) -> str:
    return re.sub(r"(?m)^\[round\]$", "[round]\n" + extra, text, count=1)


# ── config: every knob is parsed by the loader (round-trip) ──────────────────

def test_knobs_default_and_round_trip_through_the_loader(tmp_path):
    base = load_chain_config(ROOT / "chain.toml").round
    assert base.funded_sku_wall_measured_image == ""          # gate off by default
    assert base.progress_watchdog_grace_seconds == 1200
    assert base.progress_watchdog_tolerance == 1.5
    assert base.provider_ttl_margin_seconds == 3600
    assert base.payer_key_check_seconds == 1800
    text = _with_round((ROOT / "chain.toml").read_text(), "\n".join([
        f'funded_sku_wall_measured_image = "ghcr.io/x/worker@{DIGEST_A}"',
        "progress_watchdog_grace_seconds = 600",
        "progress_watchdog_tolerance = 2.0",
        "provider_ttl_margin_seconds = 0",
        "payer_key_check_seconds = 300"]))
    p = tmp_path / "chain.toml"
    p.write_text(text)
    rnd = load_chain_config(p).round
    assert rnd.funded_sku_wall_measured_image == DIGEST_A    # normalised to the digest
    assert rnd.progress_watchdog_grace_seconds == 600
    assert rnd.progress_watchdog_tolerance == 2.0
    assert rnd.provider_ttl_margin_seconds == 0
    assert rnd.payer_key_check_seconds == 300


@pytest.mark.parametrize("line", [
    'funded_sku_wall_measured_image = "latest"',
    "progress_watchdog_grace_seconds = -1",
    "progress_watchdog_tolerance = 0.5",
    "provider_ttl_margin_seconds = true",
    'payer_key_check_seconds = "x"',
])
def test_bad_knob_values_fail_the_load(tmp_path, line):
    p = tmp_path / "chain.toml"
    p.write_text(_with_round((ROOT / "chain.toml").read_text(), line))
    with pytest.raises(ValueError):
        load_chain_config(p)


# ── 1. progress watchdog ─────────────────────────────────────────────────────

class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _watch(clock, **kw):
    kw.setdefault("label", "funded leg hkA")
    kw.setdefault("hotkey", "hkA")
    kw.setdefault("pod_name", "cascade-n91-777-funded-hka-0")
    kw.setdefault("training_steps", 20000)
    kw.setdefault("max_train_seconds", 8333)
    kw.setdefault("end_wall", None)
    kw.setdefault("grace_seconds", 1200)
    return ProgressWatchdog(clock=clock, **kw)


def test_watchdog_aborts_the_incident_rate_with_one_clear_error(caplog):
    # 2026-10-31: 0.13 steps/s against a budget planned at 2.86.
    clock = _Clock()
    w = _watch(clock)
    assert w({"step": 1}) is None                    # first sighting: training started
    clock.t = 600
    assert w({"step": 1 + int(0.13 * 600)}) is None  # inside the grace window
    clock.t = 1200
    with caplog.at_level(logging.ERROR, logger="cascade.trainer"):
        why = w({"step": 1 + int(0.13 * 1200)})
    assert why and why.startswith(THROUGHPUT_ABORT)
    assert "hotkey=hkA" in why and "rate=0.130 steps/s" in why
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and THROUGHPUT_ABORT in errors[0]
    assert "42.74 h" in errors[0]                    # 20000 / 0.13 s projected, in hours
    assert w({"step": 999}) == why                   # the verdict sticks


def test_watchdog_leaves_a_leg_on_budget_alone():
    clock = _Clock()
    w = _watch(clock)
    w({"step": 1})
    clock.t = 1200
    assert w({"step": 1 + int(2.86 * 1200)}) is None     # 20000 / 2.86 ≈ 6993 s < 8333 s


def test_watchdog_tolerance_absorbs_a_modest_overrun():
    clock = _Clock()
    w = _watch(clock, tolerance=1.5)
    w({"step": 0})
    clock.t = 1200
    rate = 20000 / (8333 * 1.4)                      # 1.4x over the cap: tolerated
    assert w({"step": int(rate * 1200)}) is None


def test_watchdog_aborts_a_leg_that_misses_its_target_settlement():
    clock = _Clock()
    # fast enough for the contract cap, too slow for the settlement 1.5 h out
    w = _watch(clock, end_wall=1200 + 1.5 * 3600, tolerance=1.0)
    w({"step": 0})
    clock.t = 1200
    why = w({"step": int(2.6 * 1200)})   # 20000/2.6 ≈ 7692 s < cap; ~6492 s left > 5400 s
    assert why and "target settlement" in why


def test_watchdog_reads_a_stall_as_rate_zero():
    clock = _Clock()
    w = _watch(clock)
    w({"step": 50})
    clock.t = 1300
    assert (w({"step": 50}) or "").startswith(THROUGHPUT_ABORT)


@pytest.mark.parametrize("kw,record", [
    ({}, None),                                       # an image without the progress file
    ({}, {"tokens": 5}),                              # malformed record
    ({"grace_seconds": 0}, {"step": 1}),              # off
    ({"training_steps": 0}, {"step": 1}),             # no step budget to project against
])
def test_watchdog_is_inert_without_usable_progress(kw, record):
    clock = _Clock()
    w = _watch(clock, **kw)
    for t in (0, 5000, 10000):
        clock.t = t
        assert w(record) is None


def test_watchdog_runs_the_periodic_key_check_without_aborting():
    clock = _Clock()
    calls = []
    w = _watch(clock, grace_seconds=0, key_check=lambda: calls.append(clock.t),
               key_check_seconds=1800)
    for t in (100, 1900, 2000, 3800):
        clock.t = t
        assert w(None) is None
    assert calls == [1900, 3800]


def test_watchdog_survives_a_failing_key_check():
    clock = _Clock()

    def boom():
        raise RuntimeError("lium down")

    w = _watch(clock, grace_seconds=0, key_check=boom, key_check_seconds=1)
    clock.t = 5
    assert w(None) is None


# ── the detached poll carries the worker's progress record ───────────────────

def test_poll_command_prints_the_progress_record_before_the_token(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "pid").write_text(str(__import__("os").getpid()))
    host = RemoteHost(name="x", host="h", workdir=str(tmp_path))
    cmd = remote_mod._detached_poll_command(host, str(rd))
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    assert out.splitlines() == ["RUNNING"]                       # no file: unchanged
    (rd / "progress.json").write_text('{"step": 50,\n "tokens": 7}')
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    lines = out.splitlines()
    assert lines[-1] == "RUNNING"
    assert parse_progress(lines) == {"step": 50, "tokens": 7}


def test_progress_file_sink_keeps_the_latest_step_only(tmp_path):
    from cascade.trainer.progress_file import progress_file_sink

    assert progress_file_sink(env={}) is None
    path = tmp_path / "progress.json"
    sink = progress_file_sink(env={"CASCADE_PROGRESS_FILE": str(path)})
    sink.emit({"event": "host", "gpu": "H200"})
    assert not path.exists()
    sink.emit({"event": "step", "step": 1, "tokens": 10, "loss": 1.0})
    sink.emit({"event": "step", "step": 50, "tokens": 500, "loss": 0.5})
    rec = json.loads(path.read_text())
    assert (rec["step"], rec["tokens"]) == (50, 500)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _Scripted:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, argv, timeout, stdin=None):
        self.calls.append(argv[-1])
        step = self.steps.pop(0) if self.steps else _proc(stdout="RUNNING\n")
        return step


def test_run_detached_kills_and_raises_when_the_watchdog_aborts():
    seen = []
    runner = _Scripted([
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout='PROGRESS:{"step":1}\nRUNNING\n'),
        _proc(stdout='PROGRESS:{"step":2}\nRUNNING\n'),
        _proc(),                                                   # the kill
    ])

    def on_progress(rec):
        seen.append(rec)
        return "THROUGHPUT ABORT: test" if len(seen) == 2 else None

    with pytest.raises(LegAborted, match="THROUGHPUT ABORT: test"):
        run_detached(RemoteHost(name="p", host="h", workdir="/w"), "true", None, "/w/rd",
                     timeout=3600, poll_seconds=1, grace_seconds=60, runner=runner,
                     on_progress=on_progress)
    assert seen == [{"step": 1}, {"step": 2}]
    assert "kill -KILL" in runner.calls[-1]                       # worker killed on the pod


def test_detached_dispatch_names_the_progress_file_and_wires_the_watchdog():
    runner = _Scripted([
        _proc(stdout=""),                                          # find: no prior run
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
        _proc(stdout='PROGRESS:{"step":1}\nRUNNING\n'),
        _proc(),
    ])
    disp = RemoteDispatcher(trainer_spec="x:y", detached=True, poll_seconds=1, _runner=runner,
                            on_progress=lambda rec: "THROUGHPUT ABORT: x" if rec else None)
    with pytest.raises(LegAborted):
        disp.dispatch(RemoteHost(name="p", host="h", workdir="/w"), gen_ref="r@sha256:" + "b" * 64,
                      uid=1, hotkey="hkA", role="challenger", base_seed=7, block=1)
    launch = runner.calls[1]
    assert "CASCADE_PROGRESS_FILE=/w/_train_work/_dispatch/challenger-hkA-7-" in launch
    assert "/progress.json" in launch


def test_a_watchdog_abort_is_an_operator_fault_and_the_pod_is_torn_down(tmp_path, monkeypatch):
    class _AbortDisp:
        def dispatch(self, host, **kw):
            raise LegAborted("remote challenger on funded-hka: THROUGHPUT ABORT: hotkey=hkA "
                             "rate=0.130 steps/s")

    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=None)
    from cascade.trainer.faults import _OperatorFault

    with pytest.raises(_OperatorFault):
        runner._run_funded_leg(_AbortDisp(), _challenger("hkA"), seeds, 100, contract, "",
                               warm_start_ref=None)
    assert torn == ["cascade-n91-777-funded-hka-0"]               # verified teardown ran
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "operator", False)  # never the miner's attempt
    assert THROUGHPUT_ABORT in msg


def test_leg_watchdog_attaches_only_to_a_detached_remote_dispatcher(tmp_path):
    from cascade.trainer.runner_funded_legs import _with_leg_watchdog

    runner = _runner(tmp_path)
    runner.__dict__["_leg_local_obj"] = SimpleNamespace(end_wall=5000.0)
    pod = SimpleNamespace(instance_id="pod-0", payer_hotkey="hkA")
    contract = SimpleNamespace(training_steps=20000, max_train_seconds=8333)
    attached = RemoteDispatcher(trainer_spec="x:y", detached=False)
    assert _with_leg_watchdog(runner, attached, _challenger("hkA"), pod, contract) is attached
    other = object()
    assert _with_leg_watchdog(runner, other, _challenger("hkA"), pod, contract) is other
    det = RemoteDispatcher(trainer_spec="x:y", detached=True)
    watched = _with_leg_watchdog(runner, det, _challenger("hkA"), pod, contract)
    w = watched.on_progress
    assert isinstance(w, ProgressWatchdog)
    assert (w.training_steps, w.max_train_seconds, w.end_wall, w.grace) == (20000, 8333.0,
                                                                           5000.0, 1200.0)
    assert w.key_check is not None and w.key_check_seconds == 1800.0


# ── 2. measured-wall gate ────────────────────────────────────────────────────

def test_measured_wall_gate_is_off_by_default_and_passes_a_match():
    img = "ghcr.io/x/worker@" + DIGEST_C
    assert measured_wall_refusal(RoundConfig(funded_pod_image=img)) == ""
    assert measured_wall_refusal(RoundConfig(funded_pod_image=img,
                                             funded_sku_wall_measured_image=DIGEST_C)) == ""
    why = measured_wall_refusal(RoundConfig(funded_pod_image=img,
                                            funded_sku_wall_measured_image=DIGEST_A))
    assert "no measured wall for the pinned image" in why


def test_unmeasured_image_refuses_the_rent_unburned(tmp_path, monkeypatch, caplog):
    runner = _runner(tmp_path, funded_sku_wall_measured_image=DIGEST_A)
    _vault(tmp_path, "hkA")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(kw))
    with caplog.at_level(logging.ERROR), pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert rents == []                                             # nothing rented
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "operator", False)
    assert any("no measured wall" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "funded_pods.json").exists() or json.loads(
        (tmp_path / "funded_pods.json").read_text()) == []         # no write-ahead row left


def test_measured_image_rents_as_before(tmp_path, monkeypatch):
    runner = _runner(tmp_path, funded_sku_wall_measured_image=DIGEST_C)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert pod.payer_hotkey == "hkA"


def test_a_ledgered_allocation_is_adopted_past_the_gate(tmp_path, monkeypatch):
    # A restart mid-leg: the pod of THIS leg is on the ledger — it is
    # re-attached (adopted), never stranded by a rent-time guard.
    from tests.unit.test_funded_pod_wiring import _pod

    runner = _runner(tmp_path, funded_sku_wall_measured_image=DIGEST_A)
    _vault(tmp_path, "hkA")
    runner._ledger_add(_pod("hkA"))
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert pod.instance_id == "cascade-n91-777-funded-hka-0"


# ── 3. doomed legs ───────────────────────────────────────────────────────────

def _leg_target(runner, *, now, end_wall, cap=3600):
    local = threading.local()
    local.end_wall = end_wall
    local.funded_contract = SimpleNamespace(max_train_seconds=cap)
    runner.__dict__["_leg_local_obj"] = local
    runner._leg_local = local
    runner._rent_wait_now = lambda: now


def test_doomed_leg_reason_compares_projected_end_with_the_target(tmp_path):
    runner = _runner(tmp_path)
    assert doomed_leg_reason(runner, ("RTX4090",)) == ""            # no target: never doomed
    _leg_target(runner, now=1000.0, end_wall=1000.0 + 3600 + 900 + 60)
    assert doomed_leg_reason(runner, ("RTX4090",)) == ""            # fits with a minute left
    _leg_target(runner, now=1000.0, end_wall=1000.0 + 3600)
    why = doomed_leg_reason(runner, ("RTX4090",))
    assert why.startswith("doomed leg") and "+0.25 h past" in why


def test_a_doomed_leg_is_not_rented(tmp_path, monkeypatch, caplog):
    import time

    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    now = time.time()
    _leg_target(runner, now=now, end_wall=now + 1800)                # 1 h wall, 30 min left
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: rents.append(kw) or _rent_ok())
    monkeypatch.setattr(runner, "_yield_to_king", lambda describe: True, raising=False)
    with caplog.at_level(logging.ERROR), pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert rents == []
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)
    assert msg.startswith("doomed leg")
    ledger = json.loads((tmp_path / "funded_pods.json").read_text())
    assert ledger == []                                            # write-ahead row dropped


# ── 4. provider-side TTL ─────────────────────────────────────────────────────

def test_lium_launch_carries_a_ttl_rounded_up_to_minutes(monkeypatch):
    from cascade.provision.core import LaunchSpec, LiumProvider, lium_ttl_arg

    assert lium_ttl_arg(0) == "" and lium_ttl_arg(61) == "2m" and lium_ttl_arg(3600) == "60m"
    spawned = []
    p = LiumProvider(api_key="k", _spawn=spawned.append, _template=lambda payload: "tpl-1")
    monkeypatch.setattr(p, "_list_executors",
                        lambda sku, gpus=1: [{"id": "ex1", "price_per_hour": 1.0}])
    p.launch(LaunchSpec(sku="H200", count=1, image="img@" + DIGEST_A, ssh_pubkey="ssh-ed25519 K",
                        name_prefix="cascade-n91-1-funded-x", ttl_seconds=7261))
    argv = spawned[0]
    assert argv[argv.index("--ttl") + 1] == "122m"
    spawned.clear()
    p.launch(LaunchSpec(sku="H200", count=1, image="img@" + DIGEST_A, ssh_pubkey="ssh-ed25519 K",
                        name_prefix="cascade-n91-1-funded-y"))
    assert "--ttl" not in spawned[0]                                 # 0 = no TTL


def test_shadeform_create_carries_a_delete_date():
    from cascade.provision.core import LaunchSpec, ShadeformProvider, shadeform_delete_at

    assert shadeform_delete_at(0) == ""
    assert shadeform_delete_at(3600, now=0) == "1970-01-01T01:00:00Z"
    posted = []

    class _P(ShadeformProvider):
        def _offer(self, sku, *, gpus=1, exclude_regions=()):
            return {"cloud": "c", "region": "r", "shade_instance_type": "t",
                    "price_usd_hr": 1.0, "sku": sku}

        def _post(self, path, body=None):
            posted.append(body)
            return {"id": "iid-1"}

    _P(api_key="k").launch(LaunchSpec(sku="H100", count=1, image="img@" + DIGEST_A,
                                      ssh_pubkey="K", ttl_seconds=600))
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                        posted[0]["auto_delete"]["date_threshold"])


def test_cgen_vm_create_carries_a_delete_date(monkeypatch):
    from cascade.provision.cpu import CpuShape, ShadeformCpuProvider

    posted = []
    p = ShadeformCpuProvider(api_key="k", ssh_key_id="key-1")
    monkeypatch.setattr(p, "_post", lambda path, body=None: posted.append(body) or {"id": "vm1"})
    shape = CpuShape(vcpus=32, memory_gb=64, price_per_hour=0.5, provider="shadeform",
                     type_id="t", region="r", cloud="c")
    assert p.launch(shape, name="cgen-x", ssh_pubkey="K", ttl_seconds=9000) == "vm1"
    assert "date_threshold" in posted[0]["auto_delete"]
    p.launch(shape, name="cgen-y", ssh_pubkey="K")
    assert "auto_delete" not in posted[1]


def test_generation_stage_passes_the_ttl_only_to_launches_that_take_it():
    from cascade.generation.stage import _ttl_kw

    cfg = SimpleNamespace(round=RoundConfig(provider_ttl_margin_seconds=1800))
    assert generation_ttl_seconds(cfg, 5400) == 5400 + 3600 + 1800

    class _WithTtl:
        def launch(self, shape, *, name, ssh_pubkey, image="", ttl_seconds=0.0):
            pass

    class _Without:
        def launch(self, shape, *, name, ssh_pubkey, image=""):
            pass

    assert _ttl_kw(_WithTtl(), cfg, 5400) == {"ttl_seconds": 10800.0}
    assert _ttl_kw(_Without(), cfg, 5400) == {}
    off = SimpleNamespace(round=RoundConfig(provider_ttl_margin_seconds=0))
    assert _ttl_kw(_WithTtl(), off, 5400) == {}


def test_leg_ttl_covers_the_cap_boot_bench_and_margin(tmp_path):
    runner = _runner(tmp_path, funded_bench_margin_seconds=600)
    runner.cfg.eval = SimpleNamespace(bench_hold_max_hours=2.0)
    runner._keep_funded_pod_for_bench = lambda: False
    contract = SimpleNamespace(max_train_seconds=8333)
    assert provider_ttl_seconds(runner, contract=contract) == 8333 + 900 + 600 + 3600
    runner._keep_funded_pod_for_bench = lambda: True
    assert provider_ttl_seconds(runner, contract=contract) == 8333 + 900 + 600 + 7200 + 3600
    off = _runner(tmp_path, provider_ttl_margin_seconds=0)
    assert provider_ttl_seconds(off, contract=contract) == 0.0


def test_funded_rent_hands_the_ttl_to_the_launch_spec():
    launched = []

    class _Prov:
        name = "lium"

        def live_pod_address(self, pod_id):
            return None

        def list_tagged(self, prefix):
            return []

        def launch(self, spec):
            launched.append(spec)
            raise funded_mod.ProvisionError("sold out: no_capacity")

        def terminate(self, pod_id):
            pass

    funded_mod.rent_funded_pod(round_id="1", hotkey="hkA", api_key="sk", sku="H200",
                               image="img@" + DIGEST_A, ssh_pubkey="K", netuid=91,
                               provider_factory=lambda key: _Prov(), ttl_seconds=4321.0)
    assert launched and launched[0].ttl_seconds == 4321.0


# ── 5. teardown-key check ────────────────────────────────────────────────────

def test_a_key_that_cannot_list_pods_rents_nothing_and_says_so(caplog):
    launched = []

    class _Prov:
        name = "lium"

        def live_pod_address(self, pod_id):
            return None

        def list_tagged(self, prefix):
            raise funded_mod.ProvisionError("`lium ps` failed (rc=1): Invalid API key sk_bad")

        def launch(self, spec):
            launched.append(spec)
            return ["x"]

    with caplog.at_level(logging.ERROR, logger="cascade.provision.funded"):
        res = funded_mod.rent_funded_pod(round_id="1", hotkey="hkA", api_key="sk_bad",
                                         sku="H200", image="img@" + DIGEST_A, ssh_pubkey="K",
                                         netuid=91, provider_factory=lambda key: _Prov())
    assert not res.ok and launched == []
    assert res.error_class == "auth" and not res.burn_attempt
    lines = [r.getMessage() for r in caplog.records if PAYER_KEY_CANNOT_STOP in r.getMessage()]
    assert lines and "hotkey=hkA" in lines[0] and "cascade-n91-1-funded-hka-0" in lines[0]
    assert "sk_bad" not in lines[0]                                 # key scrubbed


def test_payer_key_listing_error_scrubs_and_reports_the_provider():
    def lister(provider, key, prefix):
        raise RuntimeError(f"Invalid API key {key}")

    err = payer_key_listing_error("sk_live", "pod", lister=lister)
    assert err.startswith("lium:") and "sk_live" not in err
    assert payer_key_listing_error("sk_live", "pod", lister=lambda *a: []) == ""
    assert payer_key_listing_error("", "pod", lister=lambda *a: []) == "no payer key"


def test_mid_leg_key_check_logs_the_alert_for_a_dead_key(tmp_path, caplog):
    from tests.unit.test_funded_pod_wiring import _pod

    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA", key="sk_dead")

    def lister(provider, key, prefix):
        assert (provider, key, prefix) == ("lium", "sk_dead", "cascade-n91-777-funded-hka-0")
        raise RuntimeError("Invalid API key")

    runner.__dict__["_payer_key_lister"] = lister
    with caplog.at_level(logging.ERROR, logger="cascade.trainer"):
        err = check_payer_key(runner, hotkey="hkA", pod_name="cascade-n91-777-funded-hka-0",
                              pod=_pod("hkA"))
    assert "Invalid API key" in err
    assert any(PAYER_KEY_CANNOT_STOP in r.getMessage() and "hotkey=hkA" in r.getMessage()
               for r in caplog.records)
    runner.__dict__["_payer_key_lister"] = lambda *a: []
    assert check_payer_key(runner, hotkey="hkA", pod_name="p", pod=_pod("hkA")) == ""


def test_detached_stderr_mark_is_unchanged():
    # The progress line rides the POLL, never the fetch the receipt parser reads.
    assert DETACHED_STDERR_MARK == "__CASCADE_DETACHED_STDERR__"
