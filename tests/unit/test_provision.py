"""Ephemeral GPU-pod provisioner — pure logic (selection order, hosts.toml
templating, provider-response parsing) and the launch/teardown control flow,
all without touching a real cloud API, CLI, or SSH.

The only untested surface is the Provider adapter I/O (the `lium` CLI shell-out
and Shadeform HTTP), mirroring how test_remote.py leaves `_run_ssh` untested."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from deploy.provision import (
    DEFAULT_FORWARD_ENV,
    LaunchSpec,
    LiumProvider,
    PodAddress,
    ProvisionError,
    RenderOpts,
    build_providers,
    lium_pod_address,
    lium_pod_ready,
    parse_lium_executors,
    parse_lium_pods,
    parse_ssh_host,
    parse_ssh_port,
    pick_shadeform_offer,
    provision_and_run,
    render_hosts_toml,
    select_provider,
    shadeform_create_body,
    shadeform_pod_address,
    validate_digest_pinned,
)

IMG = "reg.example/cascade-worker@sha256:" + "a" * 64


def _spec(count=2, **kw):
    kw.setdefault("sku", "L40S")
    kw.setdefault("image", IMG)
    kw.setdefault("ssh_pubkey", "ssh-ed25519 AAAAkey orchestrator")
    return LaunchSpec(count=count, **kw)


def _render_opts(**kw):
    kw.setdefault("key_path", "~/.ssh/lium_cascade_ed25519")
    kw.setdefault("forward_env", DEFAULT_FORWARD_ENV)
    return RenderOpts(**kw)


# ── digest pin ───────────────────────────────────────────────────────────────


def test_validate_digest_pinned_accepts_digest():
    validate_digest_pinned(IMG)  # no raise


@pytest.mark.parametrize("bad", ["reg/worker:latest", "reg/worker", "reg/worker:v1.2"])
def test_validate_digest_pinned_rejects_tags(bad):
    with pytest.raises(ProvisionError):
        validate_digest_pinned(bad)


# ── provider selection order + fallback ──────────────────────────────────────


class _FakeProvider:
    """Records lifecycle calls; never touches the network."""

    def __init__(self, name, *, available=True, ready=True, ip="203.0.113.5",
                 ready_raises=False, avail_raises=None):
        self.name = name
        self._available = available
        self._ready = ready
        self._ip = ip
        self._ready_raises = ready_raises
        self._avail_raises = avail_raises
        self.launched: list[str] = []
        self.terminated: list[str] = []

    def available(self, sku, count):
        if self._avail_raises:
            raise self._avail_raises
        return self._available

    def launch(self, spec):
        self.launched = [f"{spec.name_prefix}-{i}" for i in range(spec.count)]
        return list(self.launched)

    def wait_ready(self, pod_id, *, timeout):
        if self._ready_raises:
            raise ProvisionError("pod exploded")
        return self._ready

    def get_ip(self, pod_id):
        return PodAddress(self._ip, 22)

    def terminate(self, pod_id):
        self.terminated.append(pod_id)


def test_select_provider_respects_priority_order():
    lium = _FakeProvider("lium", available=True)
    shade = _FakeProvider("shadeform", available=True)
    assert select_provider([lium, shade], "L40S", 2) is lium


def test_select_provider_falls_through_to_next_on_no_capacity():
    lium = _FakeProvider("lium", available=False)
    shade = _FakeProvider("shadeform", available=True)
    assert select_provider([lium, shade], "L40S", 2) is shade


def test_select_provider_returns_none_when_all_empty():
    lium = _FakeProvider("lium", available=False)
    shade = _FakeProvider("shadeform", available=False)
    assert select_provider([lium, shade], "L40S", 2) is None


def test_select_provider_skips_provider_that_errors_but_uses_next():
    broken = _FakeProvider("lium", avail_raises=ValueError("network down"))
    shade = _FakeProvider("shadeform", available=True)
    assert select_provider([broken, shade], "L40S", 2) is shade


def test_select_provider_propagates_provision_error():
    broken = _FakeProvider("lium", avail_raises=ProvisionError("bad config"))
    with pytest.raises(ProvisionError):
        select_provider([broken, _FakeProvider("shadeform")], "L40S", 2)


def test_build_providers_rejects_unknown_name():
    with pytest.raises(ProvisionError):
        build_providers(["lium", "nope"])


def test_build_providers_instantiates_in_order():
    provs = build_providers(["shadeform", "lium"])
    assert [p.name for p in provs] == ["shadeform", "lium"]


class _RecordingCli:
    """Captures the argv `lium` would be called with (no subprocess)."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(argv)
        import types
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_lium_terminate_uses_positional_target_without_yes_flag():
    # `lium rm` has no --yes flag; passing one would error and we'd leak the pod.
    cli = _RecordingCli()
    LiumProvider(bin="lium", _run=cli).terminate("cascade-pod-0")
    assert cli.calls == [["lium", "rm", "cascade-pod-0"]]


def test_lium_launch_injects_ssh_pubkey_env_and_port():
    spawned: list[list[str]] = []

    def _run(argv):
        import types
        # `ls` returns executors; other calls are irrelevant here
        out = '[{"id": "exec-1"}, {"id": "exec-2"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv))
    names = prov.launch(_spec(count=2))
    assert names == ["cascade-pod-0", "cascade-pod-1"]
    up = spawned[0]
    assert up[:3] == ["lium", "up", "exec-1"]
    assert "--image" in up and IMG in up
    assert "-e" in up and f"SSH_PUBKEY={_spec().ssh_pubkey}" in up
    assert up[up.index("--internal-ports") + 1] == "22"


# ── hosts.toml templating ────────────────────────────────────────────────────


def test_render_hosts_toml_matches_schema():
    toml = render_hosts_toml(
        [PodAddress("10.0.0.1", 22), PodAddress("10.0.0.2", 40060)],
        key_path="~/.ssh/lium_cascade_ed25519",
        forward_env=DEFAULT_FORWARD_ENV,
        remote_python="/root/cascade/.venv/bin/python",
        workdir="/root/cascade",
        name_prefix="cascade-pod",
        provider="lium",
    )
    data = tomllib.loads(toml)
    hosts = data["host"]
    assert [h["name"] for h in hosts] == ["cascade-pod-0", "cascade-pod-1"]
    assert hosts[0]["host"] == "10.0.0.1" and hosts[0]["port"] == 22
    assert hosts[1]["host"] == "10.0.0.2" and hosts[1]["port"] == 40060
    for h in hosts:
        assert h["user"] == "root"
        assert h["key_path"] == "~/.ssh/lium_cascade_ed25519"
        assert h["remote_python"] == "/root/cascade/.venv/bin/python"
        assert h["workdir"] == "/root/cascade"
        assert h["cuda_device"] == "0"
        assert h["forward_env"] == list(DEFAULT_FORWARD_ENV)
        assert "StrictHostKeyChecking=accept-new" in h["ssh_options"]


def test_render_hosts_toml_chain_toml_optional():
    without = tomllib.loads(render_hosts_toml(
        [PodAddress("10.0.0.1")], key_path="k", forward_env=()))
    assert "chain_toml" not in without["host"][0]
    with_ct = tomllib.loads(render_hosts_toml(
        [PodAddress("10.0.0.1")], key_path="k", forward_env=(),
        chain_toml="/root/cascade/chain.testnet.toml"))
    assert with_ct["host"][0]["chain_toml"] == "/root/cascade/chain.testnet.toml"


def test_render_hosts_toml_stage_default_any_omits_line():
    # "any" is the schema default (RemoteHost.stage) — don't emit a redundant line.
    data = tomllib.loads(render_hosts_toml(
        [PodAddress("10.0.0.1")], key_path="k", forward_env=()))
    assert "stage" not in data["host"][0]


def test_render_hosts_toml_stage_tagged_pods(tmp_path):
    # A heat/final fleet is a homogeneous batch: every pod carries the stage tag,
    # and it must parse back through the trainer's own hosts loader.
    from cascade.trainer.remote import load_hosts

    toml = render_hosts_toml(
        [PodAddress("10.0.0.1", 22), PodAddress("10.0.0.2", 40060)],
        key_path="k", forward_env=(), name_prefix="cascade-heat", stage="heat")
    data = tomllib.loads(toml)
    assert all(h["stage"] == "heat" for h in data["host"])

    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text(toml, encoding="utf-8")
    hosts = load_hosts(hosts_path)
    assert [h.name for h in hosts] == ["cascade-heat-0", "cascade-heat-1"]
    assert all(h.stage == "heat" for h in hosts)


def test_render_hosts_toml_rejects_empty():
    with pytest.raises(ProvisionError):
        render_hosts_toml([], key_path="k", forward_env=())


# ── lium response parsing ────────────────────────────────────────────────────


def test_parse_lium_executors_empty_is_no_capacity():
    assert parse_lium_executors("") == []
    assert parse_lium_executors("[]") == []


def test_parse_lium_executors_returns_list():
    execs = parse_lium_executors('[{"id": "e1", "gpu_type": "L40S"}]')
    assert execs[0]["id"] == "e1"


def test_parse_lium_executors_rejects_non_array():
    with pytest.raises(ProvisionError):
        parse_lium_executors('{"id": "e1"}')


def test_parse_ssh_port_and_host():
    assert parse_ssh_port("ssh root@1.2.3.4 -p 40060") == 40060
    assert parse_ssh_port("ssh root@1.2.3.4") == 22            # default
    assert parse_ssh_host("ssh root@1.2.3.4 -p 40060") == "1.2.3.4"


def test_lium_pod_ready_requires_running_and_ssh():
    assert lium_pod_ready({"status": "RUNNING", "ssh_cmd": "ssh x@y -p 22"})
    assert not lium_pod_ready({"status": "PENDING", "ssh_cmd": "ssh x@y"})
    assert not lium_pod_ready({"status": "RUNNING", "ssh_cmd": ""})


def test_lium_pod_address_from_ssh_cmd():
    addr = lium_pod_address({"ip": "203.0.113.9", "ssh_cmd": "ssh root@203.0.113.9 -p 40060"})
    assert addr == PodAddress("203.0.113.9", 40060)


def test_lium_pod_address_falls_back_to_ports_map():
    addr = lium_pod_address({"ip": "203.0.113.9", "ssh_cmd": "", "ports": {"22": 33001}})
    assert addr == PodAddress("203.0.113.9", 33001)


def test_lium_pod_address_none_without_ip():
    assert lium_pod_address({"ssh_cmd": ""}) is None


def test_parse_lium_pods_empty():
    assert parse_lium_pods("") == []


# ── shadeform response parsing / body building ───────────────────────────────


def _types(*, gpu="L40S", available=True, price=120, cloud="datacrunch", region="fin-01"):
    return {
        "instance_types": [{
            "cloud": cloud,
            "shade_instance_type": "L40S.1x",
            "configuration": {"gpu_type": gpu},
            "hourly_price": price,
            "availability": [{"region": region, "available": available}],
        }]
    }


def test_pick_shadeform_offer_selects_available():
    offer = pick_shadeform_offer(_types(), "L40S")
    assert offer == {"cloud": "datacrunch", "region": "fin-01", "shade_instance_type": "L40S.1x"}


def test_pick_shadeform_offer_none_when_unavailable():
    assert pick_shadeform_offer(_types(available=False), "L40S") is None


def test_pick_shadeform_offer_filters_by_sku():
    assert pick_shadeform_offer(_types(gpu="H100"), "L40S") is None


def test_pick_shadeform_offer_prefers_cheapest():
    cheap = _types(price=90, cloud="cheapcloud", region="us-1")["instance_types"][0]
    dear = _types(price=200, cloud="dearcloud", region="eu-1")["instance_types"][0]
    offer = pick_shadeform_offer({"instance_types": [dear, cheap]}, "L40S")
    assert offer["cloud"] == "cheapcloud"


def test_shadeform_create_body_injects_only_ssh_pubkey_and_port():
    body = shadeform_create_body(
        _spec(count=1), {"cloud": "c", "region": "r", "shade_instance_type": "L40S.1x"},
        name="cascade-pod-0")
    assert body["cloud"] == "c" and body["region"] == "r"
    assert body["shade_instance_type"] == "L40S.1x" and body["shade_cloud"] is True
    docker = body["launch_configuration"]["docker_configuration"]
    assert docker["image"] == IMG
    # Only SSH_PUBKEY is seeded — Hippius creds are NEVER placed on the pod.
    assert docker["envs"] == [{"name": "SSH_PUBKEY", "value": "ssh-ed25519 AAAAkey orchestrator"}]
    assert all("HIPPIUS" not in e["name"] for e in docker["envs"])
    assert docker["port_mappings"] == [{"host_port": 22, "container_port": 22}]


def test_shadeform_pod_address_reads_ip():
    assert shadeform_pod_address({"ip": "198.51.100.7", "status": "active"}) == \
        PodAddress("198.51.100.7", 22)
    assert shadeform_pod_address({"status": "pending"}) is None


# ── launch + GUARANTEED teardown control flow ────────────────────────────────


def _run(provider, *, hosts_path, run_trainer=False, ssh_ok=True, trainer_rc=0,
         store=None, removed=None, trainer_calls=None):
    """Drive provision_and_run with caller-owned observable containers.

    store/removed/trainer_calls are populated in place, so they remain
    inspectable even when provision_and_run raises (teardown-path tests).
    """
    store = {} if store is None else store
    removed = [] if removed is None else removed
    trainer_calls = [] if trainer_calls is None else trainer_calls

    def _trainer(argv):
        trainer_calls.append(list(argv))
        return trainer_rc

    provision_and_run(
        provider, _spec(count=2),
        hosts_path=hosts_path,
        render_opts=_render_opts(),
        run_trainer=run_trainer,
        ssh_probe=lambda ip, port: ssh_ok,
        trainer_runner=_trainer,
        write_text=lambda p, t: store.__setitem__(p, t),
        remove_file=lambda p: removed.append(p),
    )
    return store, removed, trainer_calls


def test_handoff_keeps_pods_and_writes_hosts(tmp_path):
    prov = _FakeProvider("lium")
    hp = tmp_path / "hosts.toml"
    store, _removed, _calls = _run(prov, hosts_path=hp, run_trainer=False)
    assert prov.terminated == []                 # left running for manual use
    assert hp in store                            # hosts.toml written
    data = tomllib.loads(store[hp])
    assert len(data["host"]) == 2


def test_run_trainer_tears_down_after_success(tmp_path):
    prov = _FakeProvider("lium")
    hp = tmp_path / "hosts.toml"
    _store, _removed, calls = _run(prov, hosts_path=hp, run_trainer=True)
    assert calls and calls[0][:2] == ["cascade-trainer", "--remote-hosts"]
    assert prov.terminated == prov.launched       # torn down after the round
    assert prov.launched                          # (and it did launch)


def test_teardown_on_pod_not_ready(tmp_path):
    prov = _FakeProvider("lium", ready=False)
    hp = tmp_path / "hosts.toml"
    store: dict = {}
    with pytest.raises(ProvisionError):
        _run(prov, hosts_path=hp, store=store)
    assert prov.terminated == prov.launched       # every launched pod terminated
    assert hp not in store                         # hosts never rendered


def test_teardown_on_ssh_unreachable(tmp_path):
    prov = _FakeProvider("lium")
    hp = tmp_path / "hosts.toml"
    store: dict = {}
    with pytest.raises(ProvisionError):
        _run(prov, hosts_path=hp, ssh_ok=False, store=store)
    assert prov.terminated == prov.launched
    assert hp not in store                         # never got to templating


def test_teardown_on_trainer_failure(tmp_path):
    prov = _FakeProvider("lium")
    hp = tmp_path / "hosts.toml"
    with pytest.raises(ProvisionError):
        _run(prov, hosts_path=hp, run_trainer=True, trainer_rc=1)
    assert prov.terminated == prov.launched       # torn down even when trainer fails


def test_teardown_removes_sidecar_record(tmp_path):
    prov = _FakeProvider("lium", ready=False)
    hp = tmp_path / "hosts.toml"
    removed: list[Path] = []
    with pytest.raises(ProvisionError):
        _run(prov, hosts_path=hp, removed=removed)
    # sidecar was recorded on launch and cleaned up during teardown
    assert removed == [hp.with_suffix(".toml.pods.json")]
