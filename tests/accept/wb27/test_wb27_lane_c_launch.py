"""Lane C (MarketCast) launch readiness, owner bug sweep 2026-11-27. No torch.

1. The provisioner's pod health gate pinned EVERY stage to the fleet release
   (``[training] train_image_digest`` / ``[round] worker_code_fingerprint``).
   MarketCast pods boot ``[marketcast] pod_image`` (its own digest and
   fingerprint), so every mkt pod failed the gate, was replaced once,
   terminated and retried every 600 s; mkt_hosts.toml was never written.
   The ``mkt`` stage now gates on the ``[marketcast]`` pins (each falling back
   to the fleet pin only when empty); the eval stage is unchanged.
2. The pre-rent payer-balance check projected every leg at
   ``max([round] payer_projected_rate_per_hour, chosen rate)``: a MarketCast
   1xRTX3090 leg capped at $0.5/h with $5 on Lium was refused for wanting
   $15+ (PAYER_BALANCE_CLASS) and requeued until its TTL. A MarketCast leg is
   now projected at the cap its rent applies, never the lane A knob; lane A/B
   is unchanged.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

from . import helpers as h

ROOT = Path(__file__).resolve().parents[3]

FLEET_DIGEST = "sha256:" + "a" * 64
FLEET_FP = "f" * 64
MKT_DIGEST = "sha256:" + "b" * 64
MKT_FP = "c" * 64
MKT_IMAGE = "reg.example/cascade-mkt:mkt-rc2@" + MKT_DIGEST


def _cfg(*, pod_image=MKT_IMAGE, fp=MKT_FP):
    return SimpleNamespace(
        training=SimpleNamespace(train_image_digest=FLEET_DIGEST),
        round=SimpleNamespace(worker_code_fingerprint=FLEET_FP,
                              funded_pod_image="reg.example/w@" + FLEET_DIGEST),
        marketcast=SimpleNamespace(pod_image=pod_image, worker_code_fingerprint=fp))


def _provision_main():
    with h.torch_blocked():
        m = importlib.import_module("cascade.provision.main")
    assert hasattr(m, "stage_gate_pins"), \
        "the provisioner has no per-stage health-gate pins (mkt gated on the fleet release)"
    return m


# ── 1. the mkt health gate ───────────────────────────────────────────────────

def test_stage_gate_pins_take_the_marketcast_release():
    m = _provision_main()
    assert m.stage_gate_pins(_cfg()) == {"mkt": (MKT_DIGEST, MKT_FP)}
    # each field falls back to the fleet pin only when the [marketcast] one is empty
    assert m.stage_gate_pins(_cfg(fp="")) == {"mkt": (MKT_DIGEST, FLEET_FP)}
    # no [marketcast] pod_image: mkt pods run the fleet image, no override
    assert m.stage_gate_pins(_cfg(pod_image="", fp="")) == {}


def _fake_pod(monkeypatch, *, digest: str, fp: str):
    """A healthy pod whose launch env carries ``digest`` and whose cascade
    tree hashes to ``fp``."""
    df_ok = ("Filesystem 1024-blocks Used Available Capacity Mounted on\n"
             "/dev/vda1 524288000 104857600 419430400 20% /\n")

    def fake_run_ssh(ssh_argv, timeout=120):
        cmd = ssh_argv[-1]
        if cmd.startswith("echo "):
            return SimpleNamespace(returncode=0, stdout="cascade-health-ok\n", stderr="")
        if cmd.startswith("nvidia-smi"):
            return SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 4090\n", stderr="")
        if "torch.__version__" in cmd:
            return SimpleNamespace(returncode=0, stdout="3.11 2.4.1+cu124\n", stderr="")
        if "import cascade.trainer.worker" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd.startswith("printenv"):
            return SimpleNamespace(returncode=0, stdout=digest + "\n", stderr="")
        if cmd.startswith("df"):
            return SimpleNamespace(returncode=0, stdout=df_ok, stderr="")
        if cmd.startswith(("grep ", "test ")):
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if cmd.startswith(("ls ", "cat ")):
            return SimpleNamespace(returncode=2, stdout="", stderr="")
        raise AssertionError(f"unexpected remote command: {cmd!r}")

    import cascade.provision.codeprint as codeprint
    import cascade.trainer.remote as remote_mod

    monkeypatch.setattr(remote_mod, "run_ssh", fake_run_ssh)
    monkeypatch.setattr(codeprint, "remote_code_fingerprint",
                        lambda run_ssh, workdir="": (fp, 12, ""))


def _check(m, *, stage_pins):
    from cascade.provision import PodAddress
    from cascade.provision.loop import RenderSettings
    from cascade.provision.policy import ProvisionPolicy, StagePolicy

    sp = StagePolicy(sku="NVIDIA GeForce RTX 4090", gpus_per_pod=1, max_pods=1,
                     providers=("lium",), max_price_hr=1.0)
    policy = ProvisionPolicy(eval=sp, mkt=sp, max_spend_per_round=25.0)
    render = RenderSettings(image=MKT_IMAGE, ssh_pubkey="ssh-ed25519 AAAA orch",
                            key_path="~/.ssh/cascade_ed25519")
    check = m.make_health_check(policy, render, image_digest=FLEET_DIGEST, min_disk_gb=20.0,
                                hippius_probe=lambda: True, code_fingerprint=FLEET_FP,
                                stage_pins=stage_pins)
    return lambda stage: check(PodAddress("10.0.0.7", 22), stage, "lium",
                               sku="NVIDIA GeForce RTX 4090", gpus=1)


def _failed(report) -> set[str]:
    return {c.name for c in report.failures}


def test_a_mkt_pod_passes_on_the_marketcast_pins(monkeypatch):
    m = _provision_main()
    _fake_pod(monkeypatch, digest=MKT_DIGEST, fp=MKT_FP)
    rep = _check(m, stage_pins=m.stage_gate_pins(_cfg()))("mkt")
    assert rep.ok, rep.summary()


def test_a_mkt_pod_fails_on_the_fleet_pins(monkeypatch):
    m = _provision_main()
    _fake_pod(monkeypatch, digest=MKT_DIGEST, fp=MKT_FP)
    # no override (base behaviour): the mkt release is a wrong image for the fleet pins
    rep = _check(m, stage_pins={})("mkt")
    assert {"image_digest", "code_fingerprint"} <= _failed(rep), rep.summary()
    # and a fleet-image pod on the mkt stage fails the mkt pins
    _fake_pod(monkeypatch, digest=FLEET_DIGEST, fp=FLEET_FP)
    rep = _check(m, stage_pins=m.stage_gate_pins(_cfg()))("mkt")
    assert {"image_digest", "code_fingerprint"} <= _failed(rep), rep.summary()


def test_a_fleet_pod_is_unchanged(monkeypatch):
    m = _provision_main()
    pins = m.stage_gate_pins(_cfg())
    _fake_pod(monkeypatch, digest=FLEET_DIGEST, fp=FLEET_FP)
    assert _check(m, stage_pins=pins)("eval").ok
    _fake_pod(monkeypatch, digest=MKT_DIGEST, fp=MKT_FP)
    assert {"image_digest", "code_fingerprint"} <= _failed(_check(m, stage_pins=pins)("eval"))


def test_the_service_wires_the_stage_pins_into_its_gate():
    src = (ROOT / "cascade/provision/main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wired = [kw for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "make_health_check"
             for kw in node.keywords if kw.arg == "stage_pins"]
    assert wired, "the provisioner's make_health_check call passes no stage_pins"
    assert any(isinstance(kw.value, ast.Call)
               and getattr(kw.value.func, "id", "") == "stage_gate_pins" for kw in wired), \
        "stage_pins is not stage_gate_pins(cfg)"


# ── 2. the MarketCast payer-balance projection ───────────────────────────────

RND = SimpleNamespace(payer_projected_rate_per_hour=15.0, funded_max_price_per_hour=40.0)


def _rate():
    with h.torch_blocked():
        m = importlib.import_module("cascade.trainer.runner_funded_pods")
    assert hasattr(m, "payer_projection_rate"), \
        "the payer-balance check has no MarketCast-aware projection rate"
    return m.payer_projection_rate


def _choice(gpu="", cap=0.0):
    from cascade.funding.gpu_choice import GpuChoice

    return GpuChoice(gpu=gpu, max_price_per_hour=cap)


def test_a_cheap_marketcast_leg_is_projected_at_the_miners_cap():
    rate = _rate()
    r = rate(RND, _choice("RTX3090", 0.5), marketcast=True)
    assert r == 0.5
    # the reported case: a 1 h leg, $5 on Lium — affordable at the miner's cap
    assert r * 1.0 <= 5.0


def test_a_marketcast_leg_without_a_cap_is_projected_at_its_rent_ceiling():
    rate = _rate()
    # no type / no cap: the operator ceiling its rent applies over
    # [marketcast] funded_pod_skus, never the lane A knob
    assert rate(RND, _choice(), marketcast=True) == 40.0
    assert rate(RND, _choice("RTX4090", 0.0), marketcast=True) == 40.0
    ceiling_only = SimpleNamespace(payer_projected_rate_per_hour=15.0,
                                   funded_max_price_per_hour=2.0)
    assert rate(ceiling_only, _choice(), marketcast=True) == 2.0


def test_lane_a_and_b_projection_is_unchanged():
    rate = _rate()
    assert rate(RND, _choice("RTX3090", 0.5)) == 15.0               # knob floor
    assert rate(RND, _choice("8xH200", 32.0)) == 32.0               # chosen cap above it
    assert rate(RND, _choice("8xH100", 0.0)) == 40.0                # uncapped: ceiling
    assert rate(RND, _choice()) == 15.0                             # no choice: the knob


def test_the_rent_path_projects_through_it():
    src = (ROOT / "cascade/trainer/runner_funded_pods.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "payer_balance_refusal"]
    assert calls, "no payer_balance_refusal call in the rent path"
    for call in calls:
        rate_kw = next(kw for kw in call.keywords if kw.arg == "rate")
        assert isinstance(rate_kw.value, ast.Name), \
            "the payer-balance rate is computed inline, not via payer_projection_rate"
    assert "rate = payer_projection_rate(" in src
