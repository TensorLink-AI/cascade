"""Health gate + hosts publication — a pod proves itself before joining the fleet.

Every check is a pure predicate over an injected ``run_ssh(argv)`` boundary
(canned CompletedProcess-like results; no network, no GPU), and the rendered
hosts.toml must round-trip through the trainer's real ``load_hosts``."""

from __future__ import annotations

import tomllib
from types import SimpleNamespace

from cascade.provision import DEFAULT_FORWARD_ENV, PodAddress, render_hosts_toml
from cascade.provision.health import EXPECTED_TORCH, HealthGate
from cascade.provision.hostsfile import clear_hosts, write_hosts

# ── fake ssh boundary ────────────────────────────────────────────────────────


def _proc(stdout="", rc=0, stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


GOOD_L40S = {
    "echo": _proc("cascade-health-ok\n"),
    "nvidia-smi": _proc("NVIDIA L40S\nNVIDIA L40S\n"),
    "runtime": _proc("3.11 2.4.1+cu124\n"),
    "worker": _proc(""),
    "printenv": _proc("sha256:" + "a" * 64 + "\n"),
    "df": _proc("Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                "/dev/vda1 524288000 104857600 419430400 20% /\n"),   # 400 GB free
}


def _run_ssh(overrides=None, calls=None):
    """A canned run_ssh: routes each remote argv to its scripted result."""
    table = {**GOOD_L40S, **(overrides or {})}

    def run(argv):
        if calls is not None:
            calls.append(list(argv))
        if argv[0] == "echo":
            return table["echo"]
        if argv[0] == "nvidia-smi":
            return table["nvidia-smi"]
        if argv[0] == "printenv":
            return table["printenv"]
        if argv[0] == "df":
            return table["df"]
        if "torch.__version__" in argv[-1]:
            return table["runtime"]
        if "import cascade.trainer.worker" in argv[-1]:
            return table["worker"]
        raise AssertionError(f"unexpected remote argv: {argv}")

    return run


def _gate(**kw):
    kw.setdefault("sku", "NVIDIA L40S")
    kw.setdefault("gpus", 2)
    kw.setdefault("image_digest", "reg.example/worker@sha256:" + "a" * 64)
    return HealthGate(**kw)


# ── the gate: all seven checks ───────────────────────────────────────────────


def test_healthy_pod_passes_all_checks():
    report = _gate(hippius_probe=lambda: True).check(_run_ssh())
    assert report.ok and report.failures == ()
    assert [c.name for c in report.checks] == [
        "ssh_echo", "gpu_sku", "runtime_pin", "worker_import",
        "image_digest", "hippius", "disk",
    ]


def test_ssh_echo_failure():
    report = _gate().check(_run_ssh({"echo": _proc(rc=255, stderr="auth denied")}))
    assert not report.ok
    assert report.failures[0].name == "ssh_echo"


def test_gpu_sku_is_exact_l40_is_not_l40s():
    # The classic marketplace bait: an L40 sold on an L40S listing.
    report = _gate().check(_run_ssh({"nvidia-smi": _proc("NVIDIA L40\nNVIDIA L40\n")}))
    failed = {c.name for c in report.failures}
    assert "gpu_sku" in failed


def test_gpu_sku_every_line_must_match():
    # 7 good GPUs + 1 wrong one is a broken pod, not a 7/8 pod.
    out = "\n".join(["NVIDIA L40S"] * 7 + ["NVIDIA L40"]) + "\n"
    report = _gate(gpus=8).check(_run_ssh({"nvidia-smi": _proc(out)}))
    assert "gpu_sku" in {c.name for c in report.failures}


def test_gpu_count_must_cover_the_pod_shape():
    # An "8x cluster" exposing 4 GPUs can't serve 8 hosts.toml slots.
    report = _gate(gpus=8).check(_run_ssh())        # canned pod shows 2 GPUs
    assert "gpu_sku" in {c.name for c in report.failures}


def test_runtime_pin_rejects_torch_drift():
    report = _gate().check(_run_ssh({"runtime": _proc("3.11 2.12.1+cu130\n")}))
    fail = next(c for c in report.failures if c.name == "runtime_pin")
    assert EXPECTED_TORCH in fail.detail            # detail names the pinned runtime


def test_runtime_pin_rejects_python_drift():
    report = _gate().check(_run_ssh({"runtime": _proc("3.12 2.4.1+cu124\n")}))
    assert "runtime_pin" in {c.name for c in report.failures}


def test_runtime_pin_is_configurable_for_a_repin():
    gate = _gate(expected_python="3.12", expected_torch="2.5.0+cu124")
    report = gate.check(_run_ssh({"runtime": _proc("3.12 2.5.0+cu124\n")}))
    assert "runtime_pin" not in {c.name for c in report.failures}


def test_worker_import_failure():
    report = _gate().check(
        _run_ssh({"worker": _proc(rc=1, stderr="ModuleNotFoundError: cascade")}))
    assert "worker_import" in {c.name for c in report.failures}


def test_image_digest_matches_pin_across_ref_forms():
    # Pod env may carry the bare digest or the full ref — both normalise.
    for env_val in ("sha256:" + "a" * 64, "reg.example/worker@sha256:" + "a" * 64):
        report = _gate().check(_run_ssh({"printenv": _proc(env_val + "\n")}))
        assert "image_digest" not in {c.name for c in report.failures}


def test_image_digest_mismatch_and_unset_fail_when_pinned():
    wrong = _gate().check(_run_ssh({"printenv": _proc("sha256:" + "b" * 64)}))
    assert "image_digest" in {c.name for c in wrong.failures}
    unset = _gate().check(_run_ssh({"printenv": _proc(rc=1)}))
    assert "image_digest" in {c.name for c in unset.failures}


def test_image_digest_skipped_when_unpinned():
    # Mirrors assert_train_image: an empty pin means no check.
    report = _gate(image_digest="").check(_run_ssh({"printenv": _proc(rc=1)}))
    assert "image_digest" not in {c.name for c in report.failures}


def test_hippius_probe_injected():
    assert not _gate(hippius_probe=lambda: False).check(_run_ssh()).ok
    assert _gate(hippius_probe=lambda: True).check(_run_ssh()).ok
    assert _gate(hippius_probe=None).check(_run_ssh()).ok      # unconfigured ⇒ skipped


def test_disk_headroom_gate():
    thin = _proc("Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                 "/dev/vda1 524288000 519045120 5242880 99% /\n")   # 5 GB free
    report = _gate(min_disk_gb=20.0).check(_run_ssh({"df": thin}))
    fail = next(c for c in report.failures if c.name == "disk")
    assert "5.0 GB" in fail.detail
    assert _gate(min_disk_gb=4.0).check(_run_ssh({"df": thin})).ok


def test_transport_exception_fails_the_check_not_the_gate():
    def exploding(argv):
        raise TimeoutError("ssh hung")

    report = _gate().check(exploding)
    assert not report.ok
    assert all(not c.ok for c in report.checks if c.name != "hippius")
    assert "hippius" not in {c.name for c in report.failures}  # orchestrator-side, no ssh


def test_report_summary_names_each_failure():
    report = _gate().check(_run_ssh({"nvidia-smi": _proc("NVIDIA L40\nNVIDIA L40\n")}))
    assert "gpu_sku=FAIL" in report.summary() and "ssh_echo=ok" in report.summary()


# ── per-GPU fan-out + round trip through the trainer's own loader ────────────


def _render(addrs, *, prefix, stage, gpus):
    return render_hosts_toml(
        addrs, key_path="~/.ssh/cascade_ed25519", forward_env=DEFAULT_FORWARD_ENV,
        name_prefix=prefix, stage=stage, gpus_per_pod=gpus,
    )


def test_multi_gpu_pod_fans_out_one_entry_per_gpu():
    data = tomllib.loads(_render([PodAddress("10.0.0.1", 40001)],
                                 prefix="cascade-final", stage="final", gpus=2))
    hosts = data["host"]
    assert [h["name"] for h in hosts] == ["cascade-final-0-g0", "cascade-final-0-g1"]
    assert [h["cuda_device"] for h in hosts] == ["0", "1"]
    # Same physical box behind every entry — that's the expected_gpu win.
    assert all(h["host"] == "10.0.0.1" and h["port"] == 40001 for h in hosts)


def test_single_gpu_keeps_legacy_names():
    data = tomllib.loads(_render([PodAddress("10.0.0.1")], prefix="cascade-pod",
                                 stage="any", gpus=1))
    assert [h["name"] for h in data["host"]] == ["cascade-pod-0"]
    assert data["host"][0]["cuda_device"] == "0"


def test_round_trip_through_trainer_load_hosts(tmp_path):
    # THE contract test: a concatenated heat + final render must parse via the
    # trainer's real loader with stages and cuda_device honoured, exactly as
    # TrainerRunner._hosts_for will consume it.
    from cascade.trainer.remote import load_hosts

    heat = _render([PodAddress("10.0.0.1", 22), PodAddress("10.0.0.2", 40060)],
                   prefix="cascade-900-heat", stage="heat", gpus=4)
    final = _render([PodAddress("10.0.0.3", 40001)],
                    prefix="cascade-900-final", stage="final", gpus=2)
    path = tmp_path / "hosts.toml"
    write_hosts(path, heat + final)

    hosts = load_hosts(path)
    assert len(hosts) == 2 * 4 + 2
    heat_hosts = [h for h in hosts if h.stage == "heat"]
    final_hosts = [h for h in hosts if h.stage == "final"]
    assert len(heat_hosts) == 8 and len(final_hosts) == 2
    assert [h.cuda_device for h in heat_hosts] == ["0", "1", "2", "3"] * 2
    assert [h.name for h in final_hosts] == ["cascade-900-final-0-g0",
                                             "cascade-900-final-0-g1"]
    assert [h.cuda_device for h in final_hosts] == ["0", "1"]
    assert all(h.host == "10.0.0.3" for h in final_hosts)


# ── hosts publication (atomic write / clear) ─────────────────────────────────


def test_write_hosts_is_atomic_and_creates_parent(tmp_path):
    path = tmp_path / "run" / "hosts.toml"
    write_hosts(path, _render([PodAddress("10.0.0.1")], prefix="p", stage="heat", gpus=1))
    assert path.is_file()
    assert not path.with_suffix(".toml.tmp").exists()          # tmp renamed away
    assert tomllib.loads(path.read_text(encoding="utf-8"))["host"]


def test_clear_hosts_means_local_fallback(tmp_path):
    # An empty hosts file is the trainer's "no fleet" signal: load_hosts raises
    # and _reload_remote_hosts falls back to local training — round never lost.
    import pytest

    from cascade.trainer.remote import RemoteDispatchError, load_hosts

    path = tmp_path / "hosts.toml"
    write_hosts(path, _render([PodAddress("10.0.0.1")], prefix="p", stage="heat", gpus=1))
    clear_hosts(path)
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {}  # valid, empty TOML
    with pytest.raises(RemoteDispatchError):
        load_hosts(path)


def test_bootstrap_waits_for_ssh_auth(monkeypatch, tmp_path):
    """Incident 2026-07-15: key injection lags sshd by ~30-60s on marketplace
    pods; bootstrap fired at TCP-ready, hit Permission denied, and a healthy
    eval pod was terminated at t+27s. The bootstrap must poll a no-op ssh
    until auth lands BEFORE running the script — and give up cleanly (script
    never run) when auth never arrives."""
    from types import SimpleNamespace

    from cascade.provision import main as pm

    script = tmp_path / "boot.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    render = pm.RenderSettings(image="", ssh_pubkey="pk", key_path="~/.ssh/k")
    addr = SimpleNamespace(ip="10.0.0.1", ssh_port=22)

    calls = {"auth": 0, "script": 0}

    def fake_run(argv, **kw):
        if argv[0] == "ssh":
            calls["auth"] += 1                       # deny twice, then let auth in
            return SimpleNamespace(returncode=255 if calls["auth"] <= 2 else 0,
                                   stdout="", stderr="Permission denied")
        calls["script"] += 1
        assert calls["auth"] >= 3, "script must not run before auth lands"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pm.subprocess, "run", fake_run)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    boot = pm.make_bootstrap(script, render, timeout_s=60, pod_user="root")
    assert boot(addr, "eval") is True
    assert calls["script"] == 1

    # auth never lands → bootstrap gives up without running the script
    calls.update(auth=0, script=0)

    def always_denied(argv, **kw):
        if argv[0] == "ssh":
            return SimpleNamespace(returncode=255, stdout="", stderr="denied")
        calls["script"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pm.subprocess, "run", always_denied)
    boot = pm.make_bootstrap(script, render, timeout_s=60, pod_user="root",
                             auth_wait_s=0.0)
    assert boot(addr, "eval") is False
    assert calls["script"] == 0


def test_bootstrap_probes_with_the_provider_profile_user(monkeypatch, tmp_path):
    """The bug that burned three healthy shadeform 4xA6000s: make_bootstrap
    captured render BEFORE profiles were attached (frozen dataclass ⇒
    replace() makes a new object), so the auth probe ran as root@ against
    VMs that only accept shadeform@. The probe must use the provider
    profile's user."""
    from types import SimpleNamespace

    from cascade.provision import main as pm
    from cascade.provision.loop import PodProfile

    script = tmp_path / "boot.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    render = pm.RenderSettings(image="", ssh_pubkey="pk", key_path="~/.ssh/k",
                               profiles={"shadeform": PodProfile(user="shadeform")})
    users = []

    def fake_run(argv, **kw):
        if argv[0] == "ssh":
            users.append(next(a for a in argv if "@" in a).split("@")[0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pm.subprocess, "run", fake_run)
    boot = pm.make_bootstrap(script, render, timeout_s=60, pod_user="root")
    assert boot(SimpleNamespace(ip="10.0.0.1", ssh_port=22), "heat", "shadeform") is True
    assert users == ["shadeform"]                       # NOT root


def test_bootstrap_fast_fails_dead_port_but_waits_out_auth_lag(monkeypatch, tmp_path):
    """A pod whose sshd port is REFUSED (lemon) must fail fast so the
    replacement rents while the market is warm — waiting the full 900s on a
    dead lium 2x pool dried it up (2026-07-15). But 'Permission denied'
    (key-injection lag) must still wait the full window."""
    from types import SimpleNamespace

    from cascade.provision import main as pm

    script = tmp_path / "boot.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    render = pm.RenderSettings(image="", ssh_pubkey="pk", key_path="~/.ssh/k")
    addr = SimpleNamespace(ip="10.0.0.1", ssh_port=22)

    clock = {"t": 0.0}
    import time as _t
    monkeypatch.setattr(_t, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(_t, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))

    def refused(argv, **kw):
        if argv[0] == "ssh":
            return SimpleNamespace(returncode=255, stdout="",
                                   stderr="ssh: connect to host 10.0.0.1 port 22: Connection refused")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pm.subprocess, "run", refused)
    boot = pm.make_bootstrap(script, render, timeout_s=60, pod_user="root", auth_wait_s=900.0)
    assert boot(addr, "heat") is False
    assert clock["t"] <= 200                              # dead-port fast-fail, NOT 900

    # permission-denied waits the full window
    clock["t"] = 0.0

    def denied(argv, **kw):
        if argv[0] == "ssh":
            return SimpleNamespace(returncode=255, stdout="",
                                   stderr="root@10.0.0.1: Permission denied (publickey).")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pm.subprocess, "run", denied)
    boot = pm.make_bootstrap(script, render, timeout_s=60, pod_user="root", auth_wait_s=900.0)
    assert boot(addr, "heat") is False
    assert clock["t"] >= 900                              # full auth-lag wait


# ── per-pod attestation must not mutate the frozen, stage-cached gate ─────────


def test_make_health_check_per_pod_attestation_does_not_mutate_frozen_gate(monkeypatch):
    """Regression for the boot/health crash of 2026-07-15 (pod b913a43b):
    ``make_health_check`` caches one HealthGate per (stage, provider, sku, gpus)
    and stamps each pod's provider-attested image digest onto it. HealthGate is
    ``@dataclass(frozen=True)``, so assigning ``attested_digest`` on the cached
    gate raised ``cannot assign to field 'attested_digest'`` and failed EVERY
    pod's boot — counting as a boot failure that burned the once-only
    replacement slot. The per-pod copy must go through ``dataclasses.replace``.

    Here the pod's own env is unreadable (``printenv`` empty, ``/proc/1/environ``
    unreadable — the sshd-as-PID-1 case), so only the provider attestation can
    satisfy the image_digest pin: a matching ``attested_digest`` must PASS
    without raising, and a non-matching one on the SAME cache key must FAIL —
    proving each call gets its own gate rather than mutating a shared one."""
    from cascade.provision.loop import RenderSettings
    from cascade.provision.main import make_health_check
    from cascade.provision.policy import ProvisionPolicy, StagePolicy

    pin = "reg.example/worker@sha256:" + "a" * 64
    df_ok = ("Filesystem 1024-blocks Used Available Capacity Mounted on\n"
             "/dev/vda1 524288000 104857600 419430400 20% /\n")

    def fake_run_ssh(ssh_argv, timeout=120):
        cmd = ssh_argv[-1]                                 # build_ssh_argv puts it last
        if cmd.startswith("echo "):
            return SimpleNamespace(returncode=0, stdout="cascade-health-ok\n", stderr="")
        if cmd.startswith("nvidia-smi"):
            return SimpleNamespace(returncode=0, stdout="NVIDIA L40S\nNVIDIA L40S\n", stderr="")
        if "torch.__version__" in cmd:
            return SimpleNamespace(returncode=0, stdout="3.11 2.4.1+cu124\n", stderr="")
        if "import cascade.trainer.worker" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd.startswith("printenv"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")       # env stripped
        if cmd.startswith("cat "):
            return SimpleNamespace(returncode=1, stdout="", stderr="")       # /proc/1 unreadable
        if cmd.startswith("df"):
            return SimpleNamespace(returncode=0, stdout=df_ok, stderr="")
        raise AssertionError(f"unexpected remote command: {cmd!r}")

    import cascade.trainer.remote as remote_mod
    monkeypatch.setattr(remote_mod, "run_ssh", fake_run_ssh)

    stage = StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=2,
                        providers=("lium",), max_price_hr=3.0)
    policy = ProvisionPolicy(heat=stage, final=stage, trigger_margin_blocks=25,
                             max_spend_per_round=25.0)
    render = RenderSettings(image=pin, ssh_pubkey="ssh-ed25519 AAAA orch",
                            key_path="~/.ssh/cascade_ed25519")
    check = make_health_check(policy, render, image_digest=pin, min_disk_gb=20.0,
                              hippius_probe=lambda: True)
    addr = PodAddress("10.0.0.5", 22)

    # Provider attestation matches the pin → passes, and (the bug) does NOT raise.
    ok = check(addr, "final", "lium", sku="NVIDIA L40S", gpus=2, attested_digest=pin)
    assert ok.ok, ok.summary()

    # Same cache key, a pod with NO usable attestation → image_digest fails. If the
    # first call had mutated the cached gate, this would still carry the old pin.
    bad = check(addr, "final", "lium", sku="NVIDIA L40S", gpus=2, attested_digest="")
    assert "image_digest" in {c.name for c in bad.failures}

    # And the matching attestation still passes afterwards: the cache survived intact.
    again = check(addr, "final", "lium", sku="NVIDIA L40S", gpus=2, attested_digest=pin)
    assert again.ok, again.summary()
