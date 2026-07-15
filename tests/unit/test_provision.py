"""Ephemeral GPU-pod provisioner — pure logic (selection order, hosts.toml
templating, provider-response parsing) and the launch/teardown control flow,
all without touching a real cloud API, CLI, or SSH.

The only untested surface is the Provider adapter I/O (the `lium` CLI shell-out
and Shadeform HTTP), mirroring how test_remote.py leaves `_run_ssh` untested."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from cascade.provision import (
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
from cascade.provision.core import filter_tagged_names, shadeform_offer_price_usd_hr

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

    def available(self, sku, count, *, gpus=1):
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


def test_plan_argv_forwards_the_network():
    """Incident 2026-07-14: the COUNT subprocess defaulted to finney, so a
    testnet provisioner counted MAINNET's netuid and planned eligible=0 for
    three consecutive rental windows. The network must ride along."""
    from pathlib import Path

    from cascade.provision.main import plan_argv

    argv = plan_argv(Path("chain.testnet.toml"), Path("_train_work"), "test")
    assert argv[argv.index("--network") + 1] == "test"
    assert argv[argv.index("--chain-toml") + 1] == "chain.testnet.toml"
    assert "--plan-only" in argv
    # No network given (defaults intended) → flag genuinely absent.
    assert "--network" not in plan_argv(None, Path("w"), None)


def test_launch_injects_image_digest_env_when_pinned():
    """Image-boot pods MUST carry CASCADE_TRAIN_IMAGE_DIGEST: the health gate
    requires it and final workers refuse a runtime without it (live dress
    rehearsal 2026-07-15: pod booted fine but had no digest env — it would
    have failed every health check on mainnet)."""
    import types

    from cascade.provision.core import image_digest_of, shadeform_create_body

    digest = "sha256:" + "4" * 64
    pinned = f"ghcr.io/tensorlink-ai/cascade-worker@{digest}"
    assert image_digest_of(pinned) == digest
    assert image_digest_of("") == ""                       # bootstrap mode
    assert image_digest_of("ubuntu:22.04") == ""           # tag, not a pin

    # lium: -e CASCADE_TRAIN_IMAGE_DIGEST rides along in image mode
    spawned: list[list[str]] = []

    def _run(argv):
        out = '[{"id": "exec-1"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv))
    prov.launch(_spec(count=1, image=pinned))
    up = spawned[0]
    assert f"CASCADE_TRAIN_IMAGE_DIGEST={digest}" in up

    # shadeform docker mode: env list carries the digest too
    body = shadeform_create_body(_spec(count=1, image=pinned),
                                 {"cloud": "c", "region": "r", "shade_instance_type": "t"},
                                 name="cascade-x-0")
    envs = {e["name"]: e["value"] for e in body["launch_configuration"]["docker_configuration"]["envs"]}
    assert envs["CASCADE_TRAIN_IMAGE_DIGEST"] == digest


def test_lium_launch_excludes_lemons_and_remembers_machines():
    """Replacement rents must skip the failed pod's executor (the offer list is
    deterministic, so an unexcluded replacement re-rents the exact lemon —
    observed live on round 5052267627071284702's eval slot)."""
    spawned: list[list[str]] = []

    def _run(argv):
        import types
        out = '[{"id": "exec-1"}, {"id": "exec-2"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv))
    names = prov.launch(_spec(count=1, exclude_ids=("exec-1",)))
    assert spawned[0][:3] == ["lium", "up", "exec-2"]        # lemon skipped
    assert prov.machine_of(names[0]) == "exec-2"             # loop can name the machine
    # Exclusions can exhaust the market: explicit error, never a silent re-rent.
    with pytest.raises(ProvisionError):
        prov.launch(_spec(count=2, exclude_ids=("exec-1",)))


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


def test_lium_wait_ready_fast_fails_when_pod_never_appears():
    """Live 2026-07-14: a failed `lium up` (executor in post-teardown cooldown)
    creates NO pod, and wait_ready burned the full 900s polling for a ghost.
    A pod absent from `lium ps` past the appear window will never arrive —
    fail fast so the replacement path gets the time instead."""
    import types

    from cascade.provision.core import LIUM_APPEAR_TIMEOUT

    def _run(argv):
        return types.SimpleNamespace(returncode=0, stdout="[]", stderr="")

    clock = {"t": 0.0}
    prov = LiumProvider(bin="lium", _run=_run,
                        _sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                        _now=lambda: clock["t"])
    assert prov.wait_ready("cascade-1-eval-0", timeout=900.0) is False
    assert clock["t"] <= LIUM_APPEAR_TIMEOUT + prov.poll_interval   # not 900


def test_lium_wait_ready_keeps_polling_a_pod_that_appeared():
    """A listed-but-booting pod gets the FULL timeout (appear fast-fail must
    only fire for pods that were never listed at all)."""
    import types

    calls = {"n": 0}

    def _run(argv):
        calls["n"] += 1
        pod = {"name": "cascade-1-eval-0", "status": "PENDING", "ssh_cmd": ""}
        if calls["n"] >= 30:                       # becomes ready late (t≈300s)
            pod = {"name": "cascade-1-eval-0", "status": "RUNNING",
                   "ssh_cmd": "ssh root@1.2.3.4 -p 55000"}
        return types.SimpleNamespace(returncode=0, stdout=__import__("json").dumps([pod]),
                                     stderr="")

    clock = {"t": 0.0}
    prov = LiumProvider(bin="lium", _run=_run,
                        _sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                        _now=lambda: clock["t"])
    assert prov.wait_ready("cascade-1-eval-0", timeout=900.0) is True
    assert clock["t"] > 180.0                      # outlived the appear window


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
    # Only SSH_PUBKEY + the image-digest pin are seeded — never any credential.
    names = {e["name"] for e in docker["envs"]}
    assert names <= {"SSH_PUBKEY", "CASCADE_TRAIN_IMAGE_DIGEST"}
    assert {"name": "SSH_PUBKEY", "value": "ssh-ed25519 AAAAkey orchestrator"} in docker["envs"]
    assert all("HIPPIUS" not in e["name"] and "KEY" not in e["name"].replace("PUBKEY", "")
               for e in docker["envs"])
    assert docker["port_mappings"] == [{"host_port": 22, "container_port": 22}]


def test_shadeform_pod_address_reads_ip():
    assert shadeform_pod_address({"ip": "198.51.100.7", "status": "active"}) == \
        PodAddress("198.51.100.7", 22)
    assert shadeform_pod_address({"status": "pending"}) is None


def test_shadeform_offer_price_converts_cents_to_usd():
    # hourly_price is in CENTS; the budget breaker works in USD — a mixup would
    # 100× (or 1/100×) every projection.
    assert shadeform_offer_price_usd_hr(_types(price=120), "L40S") == pytest.approx(1.20)
    assert shadeform_offer_price_usd_hr(_types(available=False), "L40S") is None
    assert shadeform_offer_price_usd_hr(_types(gpu="H100"), "L40S") is None


def test_shadeform_offer_price_picks_cheapest():
    cheap = _types(price=90)["instance_types"][0]
    dear = _types(price=200)["instance_types"][0]
    assert shadeform_offer_price_usd_hr({"instance_types": [dear, cheap]}, "L40S") == \
        pytest.approx(0.90)


# ── tagged-pod listing (the reconcile primitive) ─────────────────────────────


def test_filter_tagged_names_by_prefix():
    pods = [
        {"name": "cascade-900-heat-0", "id": "i-1"},
        {"name": "cascade-900-final-0", "id": "i-2"},
        {"name": "someone-elses-box", "id": "i-3"},
        {"id": "i-4"},                                   # nameless: never ours
    ]
    assert filter_tagged_names(pods, "cascade-", id_key="name") == \
        ["cascade-900-heat-0", "cascade-900-final-0"]
    # Shadeform terminates by opaque id, so the id is the returned handle.
    assert filter_tagged_names(pods, "cascade-", id_key="id") == ["i-1", "i-2"]


def test_lium_list_tagged_uses_ps_names():
    def _run(argv):
        import types
        out = ('[{"name": "cascade-900-heat-0", "status": "RUNNING"},'
               ' {"name": "other", "status": "RUNNING"}]') if "ps" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    assert LiumProvider(bin="lium", _run=_run).list_tagged("cascade-") == \
        ["cascade-900-heat-0"]


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


def test_pick_shadeform_offer_filters_pod_shape():
    """The fleet plan fans one lane per GPU — a 1x machine against an 8-lane
    plan strands lanes, so offers must match configuration.num_gpus exactly."""
    types = {"instance_types": [
        {"configuration": {"gpu_type": "A6000", "num_gpus": 1}, "hourly_price": 50,
         "cloud": "hyperstack", "shade_instance_type": "A6000",
         "availability": [{"region": "r1", "available": True}]},
        {"configuration": {"gpu_type": "A6000", "num_gpus": 2}, "hourly_price": 100,
         "cloud": "hyperstack", "shade_instance_type": "A6000x2",
         "availability": [{"region": "r1", "available": True}]},
    ]}
    offer = pick_shadeform_offer(types, "A6000", gpus=2)
    assert offer is not None and offer["shade_instance_type"] == "A6000x2"
    assert pick_shadeform_offer(types, "A6000", gpus=8) is None  # no such shape


def test_lium_executors_filtered_by_gpu_count(monkeypatch):
    prov = LiumProvider()
    canned = ('[{"id": "e1", "gpu_type": "A6000", "gpu_count": 1},'
              ' {"id": "e8", "gpu_type": "A6000", "gpu_count": 8}]')

    class _P:
        stdout = canned
    monkeypatch.setattr(prov, "_cli", lambda argv: _P())
    assert [e["id"] for e in prov._list_executors("A6000", gpus=8)] == ["e8"]
    assert prov.available("A6000", 1, gpus=8) is True
    assert prov.available("A6000", 2, gpus=8) is False  # only one 8x machine


def test_shadeform_create_body_vm_mode():
    """ssh_key_id ⇒ bare-VM launch (bootstrap_script provisions it); no docker
    config, and the account key is what lets the orchestrator in as 'shadeform'."""
    spec = LaunchSpec(sku="RTX4090", count=1, image="ignored-in-vm-mode",
                      ssh_pubkey="ssh-ed25519 AAAA x", gpus_per_pod=4)
    offer = {"cloud": "excesssupply", "region": "us", "shade_instance_type": "RTX4090x4"}
    body = shadeform_create_body(spec, offer, name="cascade-900-heat-0",
                                 ssh_key_id="key-123")
    assert body["ssh_key_id"] == "key-123"
    assert "launch_configuration" not in body
    docker = shadeform_create_body(spec, offer, name="n")     # default: docker mode
    assert docker["launch_configuration"]["type"] == "docker"
    assert "ssh_key_id" not in docker


def test_build_providers_options():
    provs = build_providers(["shadeform"], {"shadeform": {"ssh_key_id": "key-123"}})
    assert provs[0].ssh_key_id == "key-123"


def test_lium_launch_omits_image_in_bootstrap_mode(monkeypatch):
    """Empty image ⇒ default SSH template; a template NAME as --image 400s."""
    calls = []
    prov = LiumProvider(_spawn=lambda argv: calls.append(argv))
    canned = '[{"id": "e1", "gpu_type": "RTX4090", "gpu_count": 4}]'

    class _P:
        stdout = canned
    monkeypatch.setattr(prov, "_cli", lambda argv: _P())
    prov.launch(LaunchSpec(sku="RTX4090", count=1, image="", ssh_pubkey="k",
                           gpus_per_pod=4, name_prefix="cascade-900-heat"))
    assert "--image" not in calls[0] and "--name" in calls[0]
    prov.launch(LaunchSpec(sku="RTX4090", count=1, image="img@sha256:aa", ssh_pubkey="k",
                           gpus_per_pod=4, name_prefix="cascade-900-heat"))
    assert "--image" in calls[1]


# ── shadeform docker-mode readiness (container_status gating) ────────────────


def _shadeform_with_infos(infos):
    """ShadeformProvider whose /info responses replay from a list (last repeats)."""
    from cascade.provision.core import ShadeformProvider

    clock = {"t": 0.0}
    prov = ShadeformProvider(
        _sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        _now=lambda: clock["t"],
    )
    seq = list(infos)
    prov._get = lambda path, params=None: (seq.pop(0) if len(seq) > 1 else seq[0])
    return prov, clock


def test_shadeform_wait_ready_waits_out_container_download():
    """Live 2026-07-15: the INSTANCE goes "active" while the multi-GB worker
    image is still pulling ("container_status": "downloading"); probing then
    reaches the VM's own sshd → "Permission denied" → every image-boot pod was
    killed as a dud. wait_ready must hold until the container itself runs."""
    prov, clock = _shadeform_with_infos([
        {"status": "pending"},
        {"status": "active", "container_status": "downloading"},
        {"status": "active", "container_status": "downloading"},
        {"status": "active", "container_status": "running"},
    ])
    assert prov.wait_ready("i-1", timeout=900.0) is True
    assert clock["t"] >= 3 * prov.poll_interval          # actually waited


def test_shadeform_wait_ready_vm_mode_unchanged():
    """No container_status field (VM-mode rental): active alone is ready."""
    prov, _ = _shadeform_with_infos([{"status": "active"}])
    assert prov.wait_ready("i-1", timeout=900.0) is True


def test_shadeform_wait_ready_raises_on_container_failure():
    import pytest as _pytest

    from cascade.provision.core import ProvisionError

    prov, _ = _shadeform_with_infos([
        {"status": "active", "container_status": "downloading"},
        {"status": "active", "container_status": "failed"},
    ])
    with _pytest.raises(ProvisionError, match="container entered 'failed'"):
        prov.wait_ready("i-1", timeout=900.0)
