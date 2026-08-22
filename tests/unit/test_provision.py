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
from cascade.provision.core import (
    filter_tagged_names,
    filter_vast_offers,
    pick_runpod_gpu_type,
    runpod_create_body,
    runpod_gpu_price_usd_hr,
    runpod_pod_address,
    shadeform_offer_price_usd_hr,
    vast_create_body,
    vast_offer_price_usd_hr,
    vast_offer_query,
    vast_pod_address,
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
        [("cascade-900-heat-0", "cascade-900-heat-0")]


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


def test_shadeform_pod_address_prefers_echoed_port_mapping():
    """Docker-mode: the container's sshd lives at the mapped host_port from the
    /info echo (host 22 belongs to the VM's own sshd — live 2026-07-15)."""
    from cascade.provision.core import shadeform_pod_address

    info = {"ip": "1.2.3.4", "launch_configuration": {"docker_configuration": {
        "port_mappings": [{"host_port": 2222, "container_port": 22}]}}}
    addr = shadeform_pod_address(info)
    assert (addr.ip, addr.ssh_port) == ("1.2.3.4", 2222)
    # VM-mode (no docker config): caller's port wins, default 22.
    assert shadeform_pod_address({"ip": "1.2.3.4"}).ssh_port == 22


def test_health_image_digest_falls_back_to_pid1_environ():
    """sshd sessions don't inherit the container's launch env — printenv comes
    back empty even though PID 1 carries the digest; /proc/1/environ is the
    authoritative fallback (live 2026-07-15)."""
    import types

    from cascade.provision.health import HealthGate

    pin = "sha256:" + "ab" * 32
    calls = []

    def run_ssh(argv):
        calls.append(argv)
        if argv[:1] == ["printenv"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        # cat /proc/1/environ: NUL-separated launch env, parsed locally —
        # run_ssh flattens argv through a remote shell, so no pipelines here.
        environ = f"PATH=/usr/bin\0CASCADE_TRAIN_IMAGE_DIGEST={pin}\0HOME=/root\0"
        return types.SimpleNamespace(returncode=0, stdout=environ, stderr="")

    gate = HealthGate(sku="A6000", image_digest=pin)
    ok, why = gate._check_image_digest(run_ssh)
    assert ok, why
    assert ["cat", "/proc/1/environ"] in calls


def test_health_image_digest_provider_attestation_fallback():
    """sshd-as-PID-1 images destroy /proc/1/environ (setproctitle), so when
    neither printenv nor environ yields the digest, the provider's own launch
    record (attested_digest) decides — matching pin passes, anything else
    keeps the hard failure (live 2026-07-15)."""
    import types

    from cascade.provision.health import HealthGate

    pin = "sha256:" + "cd" * 32

    def run_ssh(argv):
        if argv[:1] == ["printenv"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        # environ clobbered by setproctitle: garbage, no digest entry
        return types.SimpleNamespace(returncode=0, stdout="-D -e [listener]\0\0\0", stderr="")

    gate = HealthGate(sku="A4000", image_digest=pin, attested_digest=pin)
    ok, why = gate._check_image_digest(run_ssh)
    assert ok and "attested" in why

    gate_bad = HealthGate(sku="A4000", image_digest=pin, attested_digest="sha256:" + "ef" * 32)
    ok, _ = gate_bad._check_image_digest(run_ssh)
    assert not ok

    gate_none = HealthGate(sku="A4000", image_digest=pin)
    ok, _ = gate_none._check_image_digest(run_ssh)
    assert not ok


def test_make_health_check_attested_digest_on_frozen_gate(monkeypatch):
    """Regression (live 2026-07-15): per-pod ``attested_digest`` must not be
    assigned onto the stage-cached HealthGate — it is ``frozen=True`` and the
    mutation raised ``cannot assign to field``, failing every pod's
    boot/health before a single probe ran. The closure must take a per-pod
    copy (``dataclasses.replace``) instead."""
    import types

    import cascade.trainer.remote as remote
    from cascade.provision.health import HealthReport
    from cascade.provision.loop import RenderSettings
    from cascade.provision.main import make_health_check
    from cascade.provision.policy import ProvisionPolicy, StagePolicy

    def fake_run_ssh(argv, timeout):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(remote, "run_ssh", fake_run_ssh)
    policy = ProvisionPolicy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=4, max_pods=1,
                         providers=("shadeform",), max_price_hr=2.4),
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=1,
                          providers=("shadeform",), max_price_hr=2.6),
        trigger_margin_blocks=25, max_spend_per_round=25.0,
    )
    render = RenderSettings(image=IMG, ssh_pubkey="ssh-ed25519 AAAA cascade",
                            key_path="/tmp/k")
    check = make_health_check(policy, render, image_digest="sha256:" + "aa" * 32,
                              min_disk_gb=1.0, hippius_probe=None)
    addr = PodAddress(ip="192.0.2.1", ssh_port=2222)
    # Two pods, different attestations, same cached stage gate: both calls
    # must return a report (probes fail — irrelevant), never raise.
    r1 = check(addr, "heat", "shadeform", sku="NVIDIA RTX A6000", gpus=4,
               attested_digest="sha256:" + "aa" * 32)
    r2 = check(addr, "heat", "shadeform", sku="NVIDIA RTX A6000", gpus=4,
               attested_digest="sha256:" + "bb" * 32)
    assert isinstance(r1, HealthReport) and isinstance(r2, HealthReport)


# ── RunPod adapter (pure parts) ──────────────────────────────────────────────


def _runpod_types(gpu="NVIDIA L40S", secure=True, community=True,
                  secure_price=0.86, community_price=0.69):
    return [{
        "id": gpu, "displayName": gpu.replace("NVIDIA ", ""), "memoryInGb": 48,
        "secureCloud": secure, "communityCloud": community,
        "securePrice": secure_price, "communityPrice": community_price,
    }]


def test_pick_runpod_gpu_type_matches_id_or_display_name():
    assert pick_runpod_gpu_type(_runpod_types(), "NVIDIA L40S")["id"] == "NVIDIA L40S"
    assert pick_runpod_gpu_type(_runpod_types(), "l40s")["id"] == "NVIDIA L40S"
    assert pick_runpod_gpu_type(_runpod_types(), "NVIDIA L40") is None


def test_pick_runpod_gpu_type_secure_tier_is_not_satisfied_by_community():
    """The whole reason to pick runpod for the FINAL is Secure Cloud; a type
    sold only on the Community host marketplace must fall through, not
    silently downgrade the tier the duel runs on."""
    only_community = _runpod_types(secure=False)
    assert pick_runpod_gpu_type(only_community, "NVIDIA L40S", secure=True) is None
    assert pick_runpod_gpu_type(only_community, "NVIDIA L40S", secure=False) is not None


def test_pick_runpod_gpu_type_normalizes_graphql_and_rest_shapes():
    rest = _runpod_types()
    assert pick_runpod_gpu_type({"gpuTypes": rest}, "NVIDIA L40S") is not None
    assert pick_runpod_gpu_type({"data": {"gpuTypes": rest}}, "NVIDIA L40S") is not None
    assert pick_runpod_gpu_type({}, "NVIDIA L40S") is None


def test_runpod_price_is_per_pod_not_per_gpu():
    """RunPod quotes per GPU-hour; within_budget bills per POD-hour. Failing to
    multiply under-projects the round breaker by the pod shape."""
    assert runpod_gpu_price_usd_hr(_runpod_types(), "NVIDIA L40S", gpus=1) == \
        pytest.approx(0.86)
    assert runpod_gpu_price_usd_hr(_runpod_types(), "NVIDIA L40S", gpus=4) == \
        pytest.approx(3.44)
    assert runpod_gpu_price_usd_hr(_runpod_types(), "NVIDIA L40S", gpus=2,
                                   secure=False) == pytest.approx(1.38)
    assert runpod_gpu_price_usd_hr(_runpod_types(), "NVIDIA H100") is None


def test_runpod_create_body_seeds_only_ssh_and_digest():
    spec = _spec(count=1, gpus_per_pod=2)
    body = runpod_create_body(spec, "NVIDIA L40S", name="cascade-900-final-0")
    assert body["name"] == "cascade-900-final-0"
    assert body["imageName"] == IMG and body["gpuTypeIds"] == ["NVIDIA L40S"]
    assert body["gpuCount"] == 2 and body["cloudType"] == "SECURE"
    assert body["ports"] == ["22/tcp"] and body["supportPublicIp"] is True
    # A reclaimed spot pod mid-duel costs the round more than the discount.
    assert body["interruptible"] is False
    assert set(body["env"]) == {"SSH_PUBKEY", "PUBLIC_KEY", "CASCADE_TRAIN_IMAGE_DIGEST"}
    assert body["env"]["SSH_PUBKEY"] == body["env"]["PUBLIC_KEY"]
    assert body["env"]["CASCADE_TRAIN_IMAGE_DIGEST"] == "sha256:" + "a" * 64
    assert not any("HIPPIUS" in k for k in body["env"])


def test_runpod_create_body_omits_digest_env_when_unpinned():
    spec = LaunchSpec(sku="NVIDIA L40S", count=1, image="reg.example/worker:latest",
                      ssh_pubkey="ssh-ed25519 AAAA x")
    body = runpod_create_body(spec, "NVIDIA L40S", name="n")
    assert "CASCADE_TRAIN_IMAGE_DIGEST" not in body["env"]


def test_runpod_pod_address_reads_the_nat_mapping_not_port_22():
    """sshd answers on the NATted public port; probing 22 reaches nothing."""
    info = {"publicIp": "198.51.100.9", "portMappings": {"22": 40123}}
    assert runpod_pod_address(info) == PodAddress("198.51.100.9", 40123)


def test_runpod_pod_address_falls_back_to_runtime_ports():
    info = {"publicIp": "198.51.100.9", "runtime": {"ports": [
        {"privatePort": 8888, "publicPort": 41000, "ip": "198.51.100.9"},
        {"privatePort": 22, "publicPort": 40999, "ip": "198.51.100.9"},
    ]}}
    assert runpod_pod_address(info) == PodAddress("198.51.100.9", 40999)


def test_runpod_pod_address_none_without_ip():
    assert runpod_pod_address({"portMappings": {"22": 40123}}) is None
    assert runpod_pod_address({}) is None


def test_runpod_wait_ready_requires_the_port_map_not_just_running(monkeypatch):
    """Regression shape of the 2026-07-15 shadeform lesson: RUNNING while the
    image is still pulling has no mapping, and probing then reads as a dead
    pod. Readiness must mean 'address resolvable'."""
    from cascade.provision.core import RunPodProvider

    states = [
        {"desiredStatus": "RUNNING"},                                  # no map yet
        {"desiredStatus": "RUNNING", "portMappings": {"22": 40123},
         "publicIp": "198.51.100.9"},
    ]
    clock = {"t": 0.0}
    prov = RunPodProvider(_sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                          _now=lambda: clock["t"])
    monkeypatch.setattr(prov, "_get", lambda p, params=None: states.pop(0))
    assert prov.wait_ready("pod-1", timeout=60) is True
    assert states == []          # it kept polling past the first RUNNING


def test_runpod_wait_ready_raises_on_terminal_status(monkeypatch):
    from cascade.provision.core import RunPodProvider

    clock = {"t": 0.0}
    prov = RunPodProvider(_sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                          _now=lambda: clock["t"])
    monkeypatch.setattr(prov, "_get", lambda p, params=None: {"desiredStatus": "FAILED"})
    with pytest.raises(ProvisionError, match="FAILED"):
        prov.wait_ready("pod-1", timeout=60)


def test_runpod_list_tagged_returns_ids(monkeypatch):
    from cascade.provision.core import RunPodProvider

    prov = RunPodProvider()
    rows = [{"name": "cascade-900-heat-0", "id": "p1"},
            {"name": "someone-else", "id": "p2"}]
    monkeypatch.setattr(prov, "_get", lambda p, params=None: rows)
    assert prov.list_tagged("cascade-") == [("cascade-900-heat-0", "p1")]
    monkeypatch.setattr(prov, "_get", lambda p, params=None: {"pods": rows})
    assert prov.list_tagged("cascade-") == [("cascade-900-heat-0", "p1")]


# ── Vast.ai adapter (pure parts) ─────────────────────────────────────────────


def _vast_offer(oid=1, machine=100, price=0.30, gpus=4, name="RTX 4090",
                reliability=0.995, cores=32.0, verified=True, rentable=True):
    return {"id": oid, "machine_id": machine, "dph_total": price, "num_gpus": gpus,
            "gpu_name": name, "reliability2": reliability,
            "cpu_cores_effective": cores, "verified": verified,
            "rentable": rentable, "rented": False, "geolocation": "US"}


def _vast(*offers):
    return {"offers": list(offers)}


def test_filter_vast_offers_applies_quality_floors():
    """DEC-CA-0010 as procurement: the heat ranks runs ACROSS pods, so an
    unvetted or CPU-thin box turns host variance into rank variance."""
    good = _vast_offer(oid=1, machine=1)
    unverified = _vast_offer(oid=2, machine=2, verified=False)
    flaky = _vast_offer(oid=3, machine=3, reliability=0.80)
    cpu_thin = _vast_offer(oid=4, machine=4, cores=8.0)      # 2 cores/GPU at 4×
    got = filter_vast_offers(_vast(good, unverified, flaky, cpu_thin), "RTX 4090", gpus=4)
    assert [o["id"] for o in got] == [1]


def test_filter_vast_offers_rejects_wrong_shape_and_unrentable():
    assert filter_vast_offers(_vast(_vast_offer(gpus=2)), "RTX 4090", gpus=4) == []
    assert filter_vast_offers(_vast(_vast_offer(rentable=False)), "RTX 4090", gpus=4) == []
    assert filter_vast_offers(_vast(_vast_offer(name="RTX 3090")), "RTX 4090", gpus=4) == []


def test_filter_vast_offers_one_per_machine_cheapest_first():
    """Two lanes of one physical box are co-tenants: a '2-pod' fleet that lands
    twice on the same machine bills double and shares a memory bus."""
    a = _vast_offer(oid=1, machine=7, price=0.40)
    b = _vast_offer(oid=2, machine=7, price=0.30)            # same box, cheaper
    c = _vast_offer(oid=3, machine=8, price=0.35)
    got = filter_vast_offers(_vast(a, b, c), "RTX 4090", gpus=4)
    assert [o["id"] for o in got] == [2, 3]                  # 0.30 then 0.35


def test_filter_vast_offers_honours_machine_exclusions():
    """The replacement path must not re-rent the lemon it just tore down —
    offer lists are deterministic, so without this it would."""
    got = filter_vast_offers(
        _vast(_vast_offer(oid=1, machine=7), _vast_offer(oid=2, machine=8)),
        "RTX 4090", gpus=4, exclude_machines=("7",))
    assert [o["id"] for o in got] == [2]


def test_vast_offer_price_prices_the_qualifying_set_only():
    """A cheap offer the floors reject must not set the breaker's projection."""
    cheap_junk = _vast_offer(oid=1, machine=1, price=0.10, verified=False)
    good = _vast_offer(oid=2, machine=2, price=0.32)
    assert vast_offer_price_usd_hr(_vast(cheap_junk, good), "RTX 4090", gpus=4) == \
        pytest.approx(0.32)
    assert vast_offer_price_usd_hr(_vast(cheap_junk), "RTX 4090", gpus=4) is None


def test_vast_offer_query_carries_the_floors_server_side():
    q = vast_offer_query("RTX 4090", gpus=4, min_reliability=0.97)
    assert q["gpu_name"] == {"eq": "RTX 4090"} and q["num_gpus"] == {"eq": 4}
    assert q["reliability2"] == {"gte": 0.97} and q["verified"] == {"eq": True}
    assert q["order"] == [["dph_total", "asc"]] and q["type"] == "on-demand"
    assert "verified" not in vast_offer_query("RTX 4090", verified_only=False)


def test_vast_create_body_seeds_only_ssh_and_digest():
    body = vast_create_body(_spec(count=1, gpus_per_pod=4), label="cascade-900-heat-0")
    assert body["image"] == IMG and body["label"] == "cascade-900-heat-0"
    assert body["runtype"] == "ssh" and body["direct"] is True
    assert body["env"]["-p 22:22"] == "1"
    assert body["env"]["SSH_PUBKEY"] == "ssh-ed25519 AAAAkey orchestrator"
    assert body["env"]["CASCADE_TRAIN_IMAGE_DIGEST"] == "sha256:" + "a" * 64
    assert not any("HIPPIUS" in k for k in body["env"])


def test_vast_pod_address_prefers_direct_mapping_then_proxy():
    direct = {"public_ipaddr": "198.51.100.4",
              "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40010"}]},
              "ssh_host": "ssh5.vast.ai", "ssh_port": 12345}
    assert vast_pod_address(direct) == PodAddress("198.51.100.4", 40010)
    proxy = {"ssh_host": "ssh5.vast.ai", "ssh_port": 12345}
    assert vast_pod_address(proxy) == PodAddress("ssh5.vast.ai", 12345)
    assert vast_pod_address({}) is None


def test_vast_launch_rents_distinct_machines_and_remembers_them(monkeypatch):
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    offers = _vast(_vast_offer(oid=11, machine=71, price=0.30),
                   _vast_offer(oid=12, machine=72, price=0.31))
    monkeypatch.setattr(prov, "_bundles", lambda sku, gpus: offers)
    calls = []

    def _req(method, path, *, params=None, body=None):
        calls.append((method, path, body))
        return {"success": True, "new_contract": 900 + len(calls)}

    monkeypatch.setattr(prov, "_request", _req)
    ids = prov.launch(_spec(count=2, sku="RTX 4090", gpus_per_pod=4,
                            name_prefix="cascade-900-heat"))
    assert ids == ["901", "902"]
    assert [c[1] for c in calls] == ["/asks/11/", "/asks/12/"]
    assert [c[2]["label"] for c in calls] == ["cascade-900-heat-0", "cascade-900-heat-1"]
    # machine_of feeds the loop's lemon exclusion — vast CAN support it.
    assert prov.machine_of("901") == "71" and prov.machine_of("902") == "72"


def test_vast_launch_refuses_a_short_fleet(monkeypatch):
    """Never split or under-deliver a stage: a 2-pod plan with one qualifying
    offer must fall through to the next rung, not rent half a fleet."""
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    monkeypatch.setattr(prov, "_bundles", lambda sku, gpus: _vast(_vast_offer()))
    with pytest.raises(ProvisionError, match="only 1 qualifying"):
        prov.launch(_spec(count=2, sku="RTX 4090", gpus_per_pod=4))


def test_vast_available_counts_distinct_qualifying_boxes(monkeypatch):
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    monkeypatch.setattr(prov, "_bundles", lambda sku, gpus: _vast(
        _vast_offer(oid=1, machine=1), _vast_offer(oid=2, machine=1),   # same box
        _vast_offer(oid=3, machine=2)))
    assert prov.available("RTX 4090", 2, gpus=4) is True
    assert prov.available("RTX 4090", 3, gpus=4) is False


def test_vast_list_tagged_maps_label_to_the_shared_primitive(monkeypatch):
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    monkeypatch.setattr(prov, "_get", lambda p, params=None: {"instances": [
        {"label": "cascade-900-heat-0", "id": 901},
        {"label": "someone-else", "id": 902},
        {"id": 903},
    ]})
    assert prov.list_tagged("cascade-") == [("cascade-900-heat-0", "901")]


# ── budget breaker: per-pod pricing across adapters ──────────────────────────


def test_shadeform_offer_price_scans_only_the_rented_shape():
    """Pricing a 4× rung off the 1× listing under-projects the round breaker
    4× — it approves a spend it will never bill against."""
    one = {"configuration": {"gpu_type": "RTX4090", "num_gpus": 1},
           "hourly_price": 70, "cloud": "c", "shade_instance_type": "RTX4090",
           "availability": [{"region": "r", "available": True}]}
    four = {"configuration": {"gpu_type": "RTX4090", "num_gpus": 4},
            "hourly_price": 260, "cloud": "c", "shade_instance_type": "RTX4090x4",
            "availability": [{"region": "r", "available": True}]}
    types = {"instance_types": [one, four]}
    assert shadeform_offer_price_usd_hr(types, "RTX4090", gpus=4) == pytest.approx(2.60)
    assert shadeform_offer_price_usd_hr(types, "RTX4090", gpus=1) == pytest.approx(0.70)
    assert shadeform_offer_price_usd_hr(types, "RTX4090") == pytest.approx(0.70)


def test_loop_offer_price_passes_the_candidate_shape():
    from cascade.provision.loop import ProvisionerLoop

    seen = {}

    class _P:
        name = "runpod"

        def offer_price(self, sku, *, gpus=1):
            seen["args"] = (sku, gpus)
            return 0.86 * gpus

    assert ProvisionerLoop._offer_price(_P(), "NVIDIA L40S", 2) == pytest.approx(1.72)
    assert seen["args"] == ("NVIDIA L40S", 2)


def test_loop_offer_price_tolerates_an_adapter_without_the_shape_kwarg():
    from cascade.provision.loop import ProvisionerLoop

    class _Old:
        name = "legacy"

        def offer_price(self, sku):
            return 1.25

    assert ProvisionerLoop._offer_price(_Old(), "L40S", 4) == pytest.approx(1.25)


# ── provider registry + options ──────────────────────────────────────────────


def test_build_providers_knows_the_new_adapters():
    provs = build_providers(["runpod", "vast"])
    assert [p.name for p in provs] == ["runpod", "vast"]


def test_build_providers_passes_adapter_options():
    provs = build_providers(
        ["runpod", "vast"],
        {"runpod": {"cloud_type": "COMMUNITY"},
         "vast": {"min_cpu_cores_per_gpu": 8.0, "verified_only": False}})
    assert provs[0].cloud_type == "COMMUNITY" and provs[0]._secure is False
    assert provs[1].min_cpu_cores_per_gpu == 8.0 and provs[1].verified_only is False


def test_validate_provider_opts_rejects_typos():
    """A silently-ignored `min_reliabilty` would rent the heat off unvetted
    machines while the config claims otherwise."""
    from cascade.provision.main import _validate_provider_opts

    _validate_provider_opts({"vast": {"min_reliability": 0.99}})       # no raise
    with pytest.raises(ProvisionError, match="min_reliabilty"):
        _validate_provider_opts({"vast": {"min_reliabilty": 0.99}})


def test_sku_matching_folds_marketplace_spelling_not_distinct_devices():
    """`providers` is stage-level, so one candidate's market_sku is offered to
    every adapter: shadeform says RTX4090, vast says 'RTX 4090'. Folding
    spacing makes the rung portable; it must NOT merge L40 into L40S."""
    offers = _vast(_vast_offer(name="RTX 4090"))
    assert len(filter_vast_offers(offers, "RTX4090", gpus=4)) == 1
    assert len(filter_vast_offers(offers, "rtx 4090", gpus=4)) == 1
    assert filter_vast_offers(_vast(_vast_offer(name="RTX 4090D")), "RTX4090", gpus=4) == []

    types = _runpod_types(gpu="NVIDIA L40S")
    assert pick_runpod_gpu_type(types, "NVIDIA L40S") is not None
    assert pick_runpod_gpu_type(types, "L40 S") is not None      # displayName "L40S"
    assert pick_runpod_gpu_type(types, "NVIDIA L40") is None     # different silicon


# ── orphan reaper: name-vs-handle (the reaper was a no-op on id adapters) ────


def test_filter_tagged_pods_keeps_the_name_alongside_the_handle():
    """Regression: the reaper decides WHAT is ours by matching the pod NAME
    against `cascade-<round>-<stage>`, then kills by the provider's HANDLE.
    Collapsing the two to just the handle made `is_provisioner_pod_name` see a
    uuid, judge it "not ours", and skip it — so every shadeform orphan billed
    until someone noticed by hand."""
    from cascade.provision.core import filter_tagged_pods
    from cascade.provision.loop import is_provisioner_pod_name

    pods = [{"name": "cascade-900-heat-0", "id": "2f1c-uuid"},
            {"name": "cascade-worker", "id": "hand-rented"},   # operator's own
            {"name": "someone-else", "id": "nope"}]
    tagged = filter_tagged_pods(pods, "cascade-", id_key="id")
    assert tagged == [("cascade-900-heat-0", "2f1c-uuid"),
                      ("cascade-worker", "hand-rented")]
    reapable = [h for n, h in tagged if is_provisioner_pod_name(n)]
    assert reapable == ["2f1c-uuid"]          # was [] before the fix
    # And the operator's hand-rented box is still untouchable (2026-07-13).
    assert "hand-rented" not in reapable


def test_filter_tagged_pods_drops_nameless_rows():
    """The tag is the only claim of ownership; an unnamed pod is never ours."""
    from cascade.provision.core import filter_tagged_pods

    assert filter_tagged_pods([{"id": "x"}], "cascade-", id_key="id") == []


def test_reaper_terminates_id_addressed_orphans(tmp_path):
    """End-to-end through the loop: an id-addressed adapter's orphan must
    actually get terminated, not silently filtered out."""
    from cascade.provision.loop import ProvisionerLoop

    killed = []

    class _IdAddressed:
        name = "shadeform-like"

        def list_tagged(self, prefix):
            return [("cascade-900-heat-0", "uuid-1"),      # ours, unowned
                    ("cascade-worker", "uuid-2"),          # operator's
                    ("someone-else", "uuid-3")]

        def terminate(self, pod_id):
            killed.append(pod_id)

    loop = object.__new__(ProvisionerLoop)
    loop.providers = {"shadeform-like": _IdAddressed()}
    loop._state = None
    loop._rent_inflight = False
    loop._eval_inflight = False
    loop.dry_run = False
    loop._reconcile_orphans()
    assert killed == ["uuid-1"]


def test_reaper_still_accepts_a_legacy_string_lister(tmp_path):
    """A third-party adapter that predates the pair contract (name-is-handle,
    the lium shape) must keep working rather than crash the sweep."""
    from cascade.provision.loop import ProvisionerLoop

    killed = []

    class _Legacy:
        name = "lium-like"

        def list_tagged(self, prefix):
            return ["cascade-900-final-0", "cascade-worker"]

        def terminate(self, pod_id):
            killed.append(pod_id)

    loop = object.__new__(ProvisionerLoop)
    loop.providers = {"lium-like": _Legacy()}
    loop._state = None
    loop._rent_inflight = False
    loop._eval_inflight = False
    loop.dry_run = False
    loop._reconcile_orphans()
    assert killed == ["cascade-900-final-0"]


# ── batch launches are atomic ────────────────────────────────────────────────


def test_runpod_partial_launch_unwinds_what_it_rented(monkeypatch):
    """_rent_stage writes the ledger only AFTER launch returns, so a pod
    rented before a mid-batch failure is money with no record. Unwind at the
    source instead of waiting a poll cycle for the reaper."""
    from cascade.provision.core import RunPodProvider

    prov = RunPodProvider()
    monkeypatch.setattr(prov, "_gpu_types", lambda: _runpod_types())
    created, killed = [], []

    def _post(path, body=None):
        if len(created) >= 1:
            raise ProvisionError("runpod 503")
        created.append(body["name"])
        return {"id": f"pod-{len(created)}"}

    monkeypatch.setattr(prov, "_post", _post)
    monkeypatch.setattr(prov, "terminate", killed.append)
    with pytest.raises(ProvisionError, match="503"):
        prov.launch(_spec(count=3, sku="NVIDIA L40S", gpus_per_pod=2))
    assert killed == ["pod-1"]        # the one that was rented, torn back down


def test_vast_partial_launch_unwinds_what_it_rented(monkeypatch):
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    monkeypatch.setattr(prov, "_bundles", lambda sku, gpus: _vast(
        _vast_offer(oid=11, machine=71), _vast_offer(oid=12, machine=72)))
    killed = []
    calls = []

    def _req(method, path, *, params=None, body=None):
        calls.append(path)
        if len(calls) >= 2:
            return {"success": False, "error": "no longer available"}
        return {"success": True, "new_contract": 901}

    monkeypatch.setattr(prov, "_request", _req)
    monkeypatch.setattr(prov, "terminate", killed.append)
    with pytest.raises(ProvisionError, match="failed"):
        prov.launch(_spec(count=2, sku="RTX 4090", gpus_per_pod=4))
    assert killed == ["901"]


def test_vast_launch_rejects_a_non_dict_response(monkeypatch):
    """A marketplace that answers with a list/None must not read as success."""
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    monkeypatch.setattr(prov, "_bundles", lambda sku, gpus: _vast(_vast_offer()))
    monkeypatch.setattr(prov, "_request", lambda *a, **k: [])
    with pytest.raises(ProvisionError, match="failed"):
        prov.launch(_spec(count=1, sku="RTX 4090", gpus_per_pod=4))


# ── REST transport: retry only where it cannot double-rent ───────────────────


class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = {} if payload is None else payload
        self.content = b"{}" if payload is not None or status == 200 else b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _wired(prov, responses):
    """Attach a fake session returning `responses` in order; record calls."""
    calls = []

    class _S:
        def request(self, method, url, params=None, json=None, timeout=None):
            calls.append((method, url))
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    prov._session = _S()
    prov._sleep = lambda _s: None
    return calls


def test_rest_get_retries_transient_status_then_succeeds():
    from cascade.provision.core import RunPodProvider

    prov = RunPodProvider()
    calls = _wired(prov, [_FakeResp(503), _FakeResp(429),
                          _FakeResp(200, {"ok": True})])
    assert prov._get("/pods") == {"ok": True}
    assert len(calls) == 3


def test_rest_get_gives_up_after_the_attempt_budget():
    from cascade.provision.core import REST_RETRY_ATTEMPTS, RunPodProvider

    prov = RunPodProvider()
    calls = _wired(prov, [_FakeResp(503)] * REST_RETRY_ATTEMPTS)
    with pytest.raises(RuntimeError, match="503"):
        prov._get("/pods")
    assert len(calls) == REST_RETRY_ATTEMPTS


def test_rest_create_is_never_retried():
    """A create that failed may still have been ACCEPTED upstream; retrying it
    double-rents a fleet the ledger never learns about — the exact leak
    _unwind_partial_launch exists to close."""
    from cascade.provision.core import RunPodProvider

    prov = RunPodProvider()
    calls = _wired(prov, [_FakeResp(503), _FakeResp(200, {"id": "pod-1"})])
    with pytest.raises(RuntimeError, match="503"):
        prov._post("/pods", {"name": "x"})
    assert len(calls) == 1          # one shot only


def test_rest_delete_is_retried_because_terminate_is_idempotent():
    from cascade.provision.core import VastProvider

    prov = VastProvider()
    calls = _wired(prov, [_FakeResp(502), _FakeResp(200, {})])
    prov._request("DELETE", "/instances/1/")
    assert len(calls) == 2


def test_rest_session_is_built_once_under_concurrency(monkeypatch):
    """Two rent workers (boundary + manifest-triggered eval) can race the lazy
    init; without the lock they build two sessions and leak one."""
    import sys
    import threading
    import types as _types

    from cascade.provision.core import RunPodProvider

    # `requests` lives behind the [deploy] extra; stub it so this exercises the
    # lock rather than skipping wherever the extra is absent.
    fake = _types.ModuleType("requests")
    fake.Session = lambda: _types.SimpleNamespace(headers={}, request=None)
    monkeypatch.setitem(sys.modules, "requests", fake)
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    prov = RunPodProvider()
    seen, barrier = [], threading.Barrier(8)

    def _go():
        barrier.wait()
        seen.append(id(prov._http()))

    threads = [threading.Thread(target=_go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(seen)) == 1


def test_shadeform_keeps_its_own_auth_header_on_the_shared_mixin():
    """Folding shadeform onto _RestMixin must not silently turn X-API-KEY into
    a bearer token — that would 401 every call on the live account."""
    from cascade.provision.core import ShadeformProvider

    assert ShadeformProvider()._auth_headers("secret") == {"X-API-KEY": "secret"}


# ── one contract, every adapter ──────────────────────────────────────────────


ADAPTERS = ["lium", "shadeform", "runpod", "vast"]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_adapter_satisfies_the_provider_protocol(adapter):
    prov = build_providers([adapter])[0]
    for verb in ("available", "launch", "wait_ready", "get_ip", "terminate"):
        assert callable(getattr(prov, verb, None)), f"{adapter} missing {verb}"
    assert prov.name == adapter


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_adapter_lists_tagged_pods_as_name_handle_pairs(adapter):
    """The reaper consumes one shape across adapters; a bare handle here is
    what disabled it on shadeform."""
    import inspect

    prov = build_providers([adapter])[0]
    src = inspect.getsource(type(prov).list_tagged)
    assert "filter_tagged_pods" in src, f"{adapter} must use the shared primitive"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_adapter_prices_per_pod_hour(adapter):
    """offer_price must accept the pod shape: marketplaces that quote per
    INSTANCE need it to scan the right shape, and ones that quote per GPU need
    it to multiply. Either way, missing it under-projects the round breaker."""
    import inspect

    prov = build_providers([adapter])[0]
    fn = getattr(prov, "offer_price", None)
    if fn is None:
        pytest.skip(f"{adapter} exposes no price probe")
    assert "gpus" in inspect.signature(fn).parameters, \
        f"{adapter}.offer_price must take the pod shape"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_adapter_terminate_is_idempotent(adapter, monkeypatch):
    """Terminating an already-gone pod is success, never an exception — the
    teardown sweep retries and must not be derailed by a 404."""
    prov = build_providers([adapter])[0]

    def _boom(*a, **k):
        raise RuntimeError("404 not found")

    for attr in ("_request", "_cli", "_post"):
        if hasattr(prov, attr):
            monkeypatch.setattr(prov, attr, _boom)
    prov.terminate("already-gone")      # no raise


# ── --check-providers preflight ──────────────────────────────────────────────


def _policy_for_preflight(**kw):
    from cascade.provision.policy import ProvisionPolicy, SkuCandidate, StagePolicy

    heat = StagePolicy(
        sku="NVIDIA GeForce RTX 4090", market_sku="RTX4090", gpus_per_pod=4,
        max_pods=2, providers=("vast",), max_price_hr=2.60,
        candidates=(SkuCandidate(sku="NVIDIA RTX A6000", market_sku="A6000",
                                 gpus_per_pod=4, max_price_hr=2.40),))
    final = StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=1,
                        providers=("runpod",), max_price_hr=2.60)
    return ProvisionPolicy(heat=heat, final=final, trigger_margin_blocks=45,
                           max_spend_per_round=120.0, **kw)


class _PreflightProv:
    def __init__(self, name, *, have=True, price=1.0, rows=None, raises=None):
        self.name, self._have, self._price = name, have, price
        self._rows = [("cascade-900-heat-0", "h1")] if rows is None else rows
        self._raises = raises

    def available(self, sku, count, *, gpus=1):
        if self._raises:
            raise self._raises
        return self._have

    def offer_price(self, sku, *, gpus=1):
        return self._price

    def list_tagged(self, prefix):
        return self._rows


def test_check_providers_reports_capacity_and_price_without_renting():
    from cascade.provision.main import check_providers

    provs = {"vast": _PreflightProv("vast", price=1.10),
             "runpod": _PreflightProv("runpod", price=2.40)}
    ok, lines = check_providers(_policy_for_preflight(), provs)
    assert ok
    body = "\n".join(lines)
    assert "heat[0] 4×NVIDIA GeForce RTX 4090 (RTX4090) on vast" in body
    assert "heat[1] 4×NVIDIA RTX A6000 (A6000) on vast" in body
    assert "final[0] 2×NVIDIA L40S (NVIDIA L40S) on runpod" in body
    assert "$2.40/pod-hr (cap $2.60)" in body
    # No launch/terminate on the fakes at all: a preflight that rents is not a
    # preflight. (_PreflightProv has neither method — this would AttributeError.)


def test_check_providers_flags_a_rung_priced_over_its_cap():
    from cascade.provision.main import check_providers

    provs = {"vast": _PreflightProv("vast", price=9.99),
             "runpod": _PreflightProv("runpod", price=1.0)}
    ok, lines = check_providers(_policy_for_preflight(), provs)
    assert ok                       # over-cap is a finding, not a fault
    assert any("OVER cap" in line for line in lines)


def test_check_providers_fails_on_a_raising_adapter():
    from cascade.provision.main import check_providers

    provs = {"vast": _PreflightProv("vast", raises=ProvisionError("401 bad key")),
             "runpod": _PreflightProv("runpod")}
    ok, lines = check_providers(_policy_for_preflight(), provs)
    assert not ok
    assert any("401 bad key" in line for line in lines)


def test_check_providers_catches_the_reaper_shape_regression():
    """The whole point: an adapter returning bare handles must be caught HERE,
    with credentials in hand, not discovered as an unreaped bill later."""
    from cascade.provision.main import check_providers

    provs = {"vast": _PreflightProv("vast", rows=["uuid-only"]),
             "runpod": _PreflightProv("runpod")}
    ok, lines = check_providers(_policy_for_preflight(), provs)
    assert not ok
    assert any("must yield (name, handle) pairs" in line for line in lines)


def test_check_providers_warns_when_orphans_can_never_be_reaped():
    from cascade.provision.main import check_providers

    class _NoLister:
        name = "bare"

        def available(self, sku, count, *, gpus=1):
            return True

    provs = {"vast": _NoLister(), "runpod": _PreflightProv("runpod")}
    provs["vast"].name = "vast"
    ok, lines = check_providers(_policy_for_preflight(), provs)
    assert any("orphans are NEVER reaped" in line for line in lines)


def test_check_providers_skips_an_unmanaged_stage():
    from cascade.provision.main import check_providers
    from cascade.provision.policy import StagePolicy

    policy = _policy_for_preflight()
    policy = type(policy)(
        heat=policy.heat,
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=2, max_pods=0,
                          providers=("runpod",), max_price_hr=2.6),
        trigger_margin_blocks=45, max_spend_per_round=120.0)
    ok, lines = check_providers(policy, {"vast": _PreflightProv("vast"),
                                         "runpod": _PreflightProv("runpod")})
    assert ok
    assert any("final: unmanaged" in line for line in lines)


def test_rest_config_faults_are_not_retried(monkeypatch):
    """A missing API key is configuration, not weather. Retrying it burns the
    full backoff ladder on every probe of every cycle and buries the real cause
    under transport warnings (seen live running --check-providers)."""
    import sys

    from cascade.provision.core import RunPodProvider

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "requests", sys.modules.get("requests") or type(sys)("requests"))
    slept = []
    prov = RunPodProvider(_sleep=slept.append)
    with pytest.raises(ProvisionError, match="RUNPOD_API_KEY"):
        prov._get("/pods")
    assert slept == []          # straight out, no backoff ladder


def test_preflight_end_to_end_through_the_real_adapters():
    """Real adapter classes, recorded marketplace payloads, no credentials:
    exercises session → _request → response parsing → price units → the
    reaper's listing shape, i.e. every assumption a live run would test."""
    from cascade.provision.core import RunPodProvider, VastProvider
    from cascade.provision.main import check_providers

    runpod = RunPodProvider()
    vast = VastProvider()

    def _runpod_session():
        class _S:
            def request(self, method, url, params=None, json=None, timeout=None):
                if url.endswith("/gputypes"):
                    return _FakeResp(200, _runpod_types(gpu="NVIDIA L40S",
                                                        secure_price=1.10))
                if url.endswith("/pods"):
                    return _FakeResp(200, [{"name": "cascade-900-final-0",
                                            "id": "pod-xyz"}])
                raise AssertionError(f"unexpected {method} {url}")
        return _S()

    def _vast_session():
        class _S:
            def request(self, method, url, params=None, json=None, timeout=None):
                if "/bundles/" in url:
                    return _FakeResp(200, _vast(
                        _vast_offer(oid=1, machine=1, price=1.05, name="RTX 4090"),
                        _vast_offer(oid=2, machine=2, price=0.90, name="RTX 4090",
                                    verified=False)))       # junk: must not price
                if "/instances/" in url:
                    return _FakeResp(200, {"instances": []})
                raise AssertionError(f"unexpected {method} {url}")
        return _S()

    runpod._session, vast._session = _runpod_session(), _vast_session()
    ok, lines = check_providers(_policy_for_preflight(),
                                {"runpod": runpod, "vast": vast})
    body = "\n".join(lines)
    assert ok, body
    # runpod quotes PER GPU ($1.10) and the final rung is 2×L40S ⇒ $2.20/pod-hr.
    assert "final[0] 2×NVIDIA L40S (NVIDIA L40S) on runpod: capacity, $2.20/pod-hr" in body
    # vast prices the QUALIFYING set only: the cheaper unverified box is ignored.
    assert "heat[0] 4×NVIDIA GeForce RTX 4090 (RTX4090) on vast: capacity, $1.05/pod-hr" in body
    # config's market_sku is "RTX4090"; vast spells it "RTX 4090" — must match.
    assert "no capacity" not in body.split("heat[1]")[0]
    # listing shape is what the reaper can actually consume
    assert "ok   runpod: list_tagged → 1 tagged pod(s)" in body
