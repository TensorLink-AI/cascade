"""Per-payer rentals: key plumbing, ledger attribution, classified failures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import cascade.provision.core as core
from cascade.funding.vault import PayerKeyVault
from cascade.provision.core import LaunchSpec, LiumProvider, PodAddress, ProvisionError
from cascade.provision.funded import (
    FUNDED_STAGE,
    funded_pod_name,
    payer_pod_pattern,
    reconcile_funded,
    rent_funded_pod,
    teardown_funded,
)
from cascade.provision.loop import is_provisioner_pod_name
from cascade.provision.state import PodInstance, RoundState, load_state, save_state

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


# ── LiumProvider per-key env plumbing ────────────────────────────────────────


def test_lium_env_override_only_when_key_set(monkeypatch):
    captured = {}

    def fake_run(argv, timeout=120.0, env=None):
        captured["env"] = env
        import subprocess
        return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")

    monkeypatch.setattr(core, "_run_cli", fake_run)
    monkeypatch.setenv("LIUM_API_KEY", "sk-operator")

    LiumProvider(bin="lium")._list_pods()
    assert captured["env"] is None                       # operator path: inherit

    LiumProvider(bin="lium", api_key="sk-miner")._list_pods()
    assert captured["env"]["LIUM_API_KEY"] == "sk-miner"  # payer path: override
    import os
    assert os.environ["LIUM_API_KEY"] == "sk-operator"    # never mutated


def test_lium_error_scrubs_key_and_repr_hides_it():
    def fail_run(argv):
        import subprocess
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="denied for sk-miner-123")

    p = LiumProvider(bin="lium", api_key="sk-miner-123", _run=fail_run)
    with pytest.raises(ProvisionError) as e:
        p._list_pods()
    assert "sk-miner-123" not in str(e.value)
    assert "sk-miner-123" not in repr(p)


# ── ledger attribution ───────────────────────────────────────────────────────


def test_pod_instance_payer_roundtrip(tmp_path):
    state = RoundState(round_id="123", instances=(
        PodInstance("lium", "cascade-123-funded-abc", FUNDED_STAGE,
                    "2026-08-28T00:00:00Z", sku="L40S", gpus=1, payer_hotkey=HK),
    ))
    save_state(tmp_path / "state.json", state)
    loaded = load_state(tmp_path / "state.json")
    assert loaded.instances[0].payer_hotkey == HK
    # Pre-funding ledgers (no payer field) still load, defaulting to operator.
    old = {"round_id": "1", "instances": [{
        "provider": "lium", "instance_id": "x", "stage": "final",
        "rented_at_iso": "2026-01-01T00:00:00Z"}]}
    import json
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert load_state(tmp_path / "old.json").instances[0].payer_hotkey == ""


def test_funded_pod_name_is_deployment_scoped_and_off_the_provisioner_scheme():
    name = funded_pod_name("8901234", HK, 91)
    assert name.startswith("cascade-n91-8901234-funded-")
    # Deliberately OUTSIDE the provisioner's reaper scheme: funded pods (and
    # the trainer's JIT king) are trainer-ledgered — a provisioner that
    # matched them would reap the live king mid-round (review 2026-09-02).
    assert not is_provisioner_pod_name(name)
    assert not is_provisioner_pod_name(f"{name}-0")
    with pytest.raises(ProvisionError):
        funded_pod_name("1", "!!!", 91)


# ── rent_funded_pod ──────────────────────────────────────────────────────────


@dataclass
class FakeProvider:
    name: str = "lium"
    fail_launch: str = ""            # raise this message from launch
    ready: bool = True
    rm_noop: bool = False            # terminate "succeeds" but the pod stays live
    rm_raises: str = ""              # terminate raises (e.g. a hung-CLI timeout)
    launched: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    tagged: list = field(default_factory=list)

    def launch(self, spec: LaunchSpec) -> list[str]:
        if self.fail_launch:
            raise ProvisionError(self.fail_launch)
        name = f"{spec.name_prefix}-0"
        self.launched.append((name, spec))
        return [name]

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        return self.ready

    def get_ip(self, pod_id: str) -> PodAddress | None:
        return PodAddress(ip="10.0.0.9", ssh_port=2222)

    def terminate(self, pod_id: str) -> None:
        if self.rm_raises:
            raise RuntimeError(self.rm_raises)
        self.terminated.append(pod_id)
        if not self.rm_noop and pod_id in self.tagged:
            self.tagged.remove(pod_id)

    def list_tagged(self, prefix: str) -> list[str]:
        return [t for t in self.tagged if t.startswith(prefix)]


def _rent(provider, **kw):
    return rent_funded_pod(
        round_id="777", hotkey=HK, api_key="sk-miner", sku="L40S",
        image="img@sha256:" + "0" * 64, ssh_pubkey="ssh-ed25519 AAA",
        netuid=91, provider_factory=lambda key: provider, **kw,
    )


def test_rent_success_ledgers_payer_and_stage():
    provider = FakeProvider()
    res = _rent(provider)
    assert res.ok and res.error_class == ""
    assert res.pod.stage == FUNDED_STAGE
    assert res.pod.payer_hotkey == HK
    assert res.address.ssh_port == 2222
    _, spec = provider.launched[0]
    assert spec.count == 1 and spec.name_prefix == funded_pod_name("777", HK, 91)


@pytest.mark.parametrize("msg,cls,burns", [
    ("401 unauthorized for this api key", "auth", False),
    ("429 too many requests", "rate_limited", False),
    ("lium: only 0 × 1xL40S available, need 1", "no_capacity", False),
    ("executor exploded mid-boot", "infra", True),
])
def test_rent_failure_classes_and_burn(msg, cls, burns):
    res = _rent(FakeProvider(fail_launch=msg))
    assert not res.ok
    assert (res.error_class, res.burn_attempt) == (cls, burns)


def test_rent_not_ready_tears_down_on_payer_key():
    provider = FakeProvider(ready=False)
    res = _rent(provider, ready_timeout=1.0)
    assert not res.ok and res.error_class == "infra"
    assert provider.terminated == [f"{funded_pod_name('777', HK, 91)}-0"]
    assert res.leaked_pod == ""                        # confirmed gone


def test_rent_cleanup_records_unconfirmed_teardown():
    # A revoked key mid-launch: terminate "succeeds" but the pod stays listed
    # — the result must carry the leak, never drop it on the floor.
    pod = f"{funded_pod_name('777', HK, 91)}-0"
    provider = FakeProvider(ready=False, rm_noop=True)
    provider.launch_hook = None

    def launch(spec):
        provider.tagged.append(pod)
        provider.launched.append((pod, spec))
        return [pod]

    provider.launch = launch
    res = _rent(provider, ready_timeout=1.0)
    assert not res.ok
    assert res.leaked_pod == pod


def test_rent_never_leaks_key_in_error():
    res = _rent(FakeProvider(fail_launch="denied: key sk-miner rejected"))
    assert "sk-miner" not in res.error


# ── teardown / reconcile on the payer's account ──────────────────────────────


def _vault_with_key():
    v = PayerKeyVault(dir=None)
    v.insert(HK, "sk-miner")
    return v


POD_ID = f"{funded_pod_name('777', HK, 91)}-0"     # what launch() actually names it


def _funded_instance(instance_id=POD_ID):
    return PodInstance("lium", instance_id, FUNDED_STAGE,
                       "2026-08-28T00:00:00Z", payer_hotkey=HK)


def test_teardown_funded_uses_payer_key_and_verifies():
    provider = FakeProvider(tagged=[POD_ID])
    left = teardown_funded([_funded_instance()], _vault_with_key(),
                           provider_factory=lambda key: provider)
    assert left == [] and provider.terminated == [POD_ID]


def test_teardown_funded_reports_keyless_pods_instead_of_swallowing():
    provider = FakeProvider()
    inst = _funded_instance()
    left = teardown_funded([inst], PayerKeyVault(dir=None),
                           provider_factory=lambda key: provider)
    assert left == [inst] and provider.terminated == []


def test_teardown_detects_pod_still_live_after_terminate():
    # LiumProvider.terminate treats a failed `lium rm` as already-terminated
    # (operator-fleet idempotency); on a REVOKED miner key that turns a 401
    # into silence. The verification re-list is what catches it.
    provider = FakeProvider(tagged=[POD_ID], rm_noop=True)
    inst = _funded_instance()
    left = teardown_funded([inst], _vault_with_key(),
                           provider_factory=lambda key: provider)
    assert left == [inst]


def test_teardown_continues_past_a_crashing_terminate():
    hk2 = "5FakeHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    vault = _vault_with_key()
    vault.insert(hk2, "sk-miner-2")
    bad = _funded_instance()
    good = PodInstance("lium", f"{funded_pod_name('777', hk2, 91)}-0", FUNDED_STAGE,
                       "2026-08-28T00:00:00Z", payer_hotkey=hk2)
    providers = {
        "sk-miner": FakeProvider(rm_raises="lium CLI hung (TimeoutExpired)"),
        "sk-miner-2": FakeProvider(tagged=[good.instance_id]),
    }
    left = teardown_funded([bad, good], vault,
                           provider_factory=lambda key: providers[key])
    assert left == [bad]                       # reported, not swallowed…
    assert providers["sk-miner-2"].terminated == [good.instance_id]  # …and the rest ran


def test_reconcile_funded_scoped_to_this_payers_funded_pods():
    owned = _funded_instance(POD_ID)
    orphan = f"{funded_pod_name('777', HK, 91)}-r1"
    provider = FakeProvider(tagged=[
        POD_ID,                                # owned → keep
        orphan,                                # this payer's funded orphan → kill
        "cascade-777-heat-0",                  # miner's OWN cascade deployment → keep
        "cascade-777-funded-otherpayer1-0",    # someone else's slug → keep
        "cascade-workerpad",                   # hand-rented lookalike → keep
    ])
    killed = reconcile_funded([owned], _vault_with_key(), netuid=91,
                              provider_factory=lambda key: provider)
    assert killed == [orphan]
    assert provider.terminated == killed


def test_reconcile_verifies_termination_not_just_the_rm_call():
    # A revoked key: terminate "succeeds" (rm swallowed) but the pod stays
    # live. reconcile must NOT report it killed — that would mask a still-
    # billing orphan every sweep (review 2026-08-29).
    owned = _funded_instance(POD_ID)
    orphan = f"{funded_pod_name('777', HK, 91)}-r1"
    provider = FakeProvider(tagged=[POD_ID, orphan], rm_noop=True)
    killed = reconcile_funded([owned], _vault_with_key(), netuid=91,
                              provider_factory=lambda key: provider)
    assert killed == []                        # nothing CONFIRMED gone
    assert orphan in provider.terminated       # it was attempted, just unconfirmed


def test_payer_pod_pattern_matches_only_this_payer():
    pat = payer_pod_pattern(HK, 91)
    assert pat.match(POD_ID)
    assert pat.match(f"{funded_pod_name('12345', HK, 91)}")
    assert not pat.match("cascade-777-heat-0")
    assert not pat.match("cascade-777-funded-otherpayer1-0")
