"""Hosts-file and health-gate guards for the provisioner.

* H3 — hosts.toml entries this service does not own (operator-appended
  lanes) survive every republish; stale self-published lanes do not; the
  static fragment is re-read per publish.
* I1 — the health gate fails a RECYCLED container (a previous rental's digest
  pin or work dir already on disk) and walks the replacement to another
  facility; the adopted-pod re-gate skips that check.
"""

from __future__ import annotations

import tomllib

import pytest

from cascade.provision.health import HealthGate
from cascade.provision.hostsfile import parse_host_entries, render_host_entries
from cascade.trainer.remote import RemoteDispatchError, load_hosts
from tests.unit.test_provision_health import _proc, _run_ssh
from tests.unit.test_provision_loop import (
    Clock,
    FakeProvider,
    _eval_loop,
    _report,
    _seed_ledger,
    cycle,
    make_loop,
)

# ── I1: fresh_boot gate ──────────────────────────────────────────────────────


def _gate(**kw):
    kw.setdefault("sku", "NVIDIA L40S")
    kw.setdefault("gpus", 2)
    kw.setdefault("image_digest", "sha256:" + "a" * 64)
    return HealthGate(**kw)


def test_fresh_container_passes_fresh_boot():
    report = _gate().check(_run_ssh())
    assert report.ok, report.summary()
    assert [c.name for c in report.checks][-1] == "fresh_boot"


def test_pinned_digest_line_on_disk_means_recycled_container():
    # grep -q exits 0: /etc/environment or ~/.bashrc already carries the pin —
    # only the provisioner's post-gate hook ever writes it, so this pod is a
    # previous rental's container restarted, whatever digest it reports.
    report = _gate().check(_run_ssh({"grep": _proc("", rc=0)}))
    assert not report.ok
    fb = next(c for c in report.checks if c.name == "fresh_boot")
    assert "recycled container" in fb.detail and "pinned" in fb.detail


def test_digest_var_in_etc_environment_alone_is_fresh():
    # Lium materialises the launch env (which carries CASCADE_TRAIN_IMAGE_DIGEST
    # by design) into /etc/environment on a brand-new pod. That is NOT a
    # recycled-container trace: the gate must never grep /etc/environment
    # (2026-09-11 23:16: every fresh Lium pod failed fresh_boot on it).
    base = _run_ssh()

    def run(argv):
        if argv[0] == "grep" and "/etc/environment" in argv:
            return _proc("", rc=0)                        # would match if consulted
        return base(argv)

    report = _gate().check(run)
    assert report.ok, report.summary()


def test_provisioned_sentinel_means_recycled_container():
    # The post-gate hook drops ~/.cascade-provisioned; a "new" rental carrying
    # it is a previous rental's container handed back by the provider.
    report = _gate().check(_run_ssh({"test": _proc("", rc=0)}))
    assert not report.ok
    fb = next(c for c in report.checks if c.name == "fresh_boot")
    assert "recycled container" in fb.detail and ".cascade-provisioned" in fb.detail


def test_leftover_train_work_means_recycled_container():
    report = _gate().check(_run_ssh({"ls": _proc("14174795993398307910\n_bench_ckpts\n")}))
    assert not report.ok
    fb = next(c for c in report.checks if c.name == "fresh_boot")
    assert "recycled container" in fb.detail and "2 entries" in fb.detail


def test_empty_train_work_dir_is_fresh():
    report = _gate().check(_run_ssh({"ls": _proc("")}))
    assert report.ok, report.summary()


def test_fresh_boot_probes_the_pod_users_home_and_workdir():
    calls = []
    _gate(home_dir="/home/shadeform", workdir="/home/shadeform/cascade").check(
        _run_ssh(calls=calls))
    grep = next(a for a in calls if a[0] == "grep")
    sentinel = next(a for a in calls if a[0] == "test")
    ls = next(a for a in calls if a[0] == "ls")
    # Only the hook's own traces are probed — never /etc/environment, which
    # Lium fills from the launch env (a fresh Lium pod carries the digest var).
    assert grep[-2:] == ["^export CASCADE_TRAIN_IMAGE_DIGEST=", "/home/shadeform/.bashrc"]
    assert "/etc/environment" not in grep
    assert sentinel[-1] == "/home/shadeform/.cascade-provisioned"
    assert ls[-1] == "/home/shadeform/cascade/_train_work"


def test_adopted_regate_skips_fresh_boot():
    # A restart re-gates pods THIS service pinned and trained on: the
    # recycled-container evidence is expected there and must not kill them.
    calls = []
    report = _gate(require_fresh_boot=False).check(
        _run_ssh({"grep": _proc("", rc=0), "ls": _proc("work\n")}, calls=calls))
    assert report.ok, report.summary()
    assert not any(a[0] in ("grep", "ls") for a in calls)


def test_fresh_boot_failure_walks_the_replacement_to_another_facility(tmp_path):
    """Same treatment as image_digest: a recycled container is a facility
    property (its container cache), so the replacement rents elsewhere."""

    class RegionAwareProvider(FakeProvider):
        regions = ("kansascity-usa-1", "chicago-usa-2")

        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            self.region_by_pod: dict[str, str] = {}
            self.specs = []

        def launch(self, spec):
            self.specs.append(spec)
            region = next(r for r in self.regions if r not in spec.exclude_regions)
            ids = super().launch(spec)
            for pid in ids:
                self.region_by_pod[pid] = region
            return ids

        def region_of(self, pod_id):
            return self.region_by_pod.get(pod_id)

    prov = RegionAwareProvider("lium")

    def health(addr, stage, provider="", **shape):
        pid = next(p for p, a in prov.live.items() if a.ip == addr.ip)
        return _report(ok=prov.region_by_pod[pid] != "kansascity-usa-1", name="fresh_boot")

    loop, _ = _eval_loop(tmp_path, prov=prov, health=health)
    cycle(loop)
    rspec = next(s for s in prov.specs if "-r0" in s.name_prefix)
    assert rspec.exclude_regions == ("kansascity-usa-1",)
    assert prov.region_by_pod["cascade-111-eval-r0-0"] == "chicago-usa-2"
    assert loop._gate_failures == {}


def test_regate_passes_adopted_flag_to_the_gate(tmp_path):
    seen = []

    def health(addr, stage, provider="", **shape):
        seen.append(shape)
        return _report(ok=True)

    loop, prov = _eval_loop(tmp_path, health=health)
    cycle(loop)
    assert seen and all(not s.get("adopted") for s in seen)     # fresh rental
    seen.clear()
    loop2 = make_loop(tmp_path, providers={"lium": prov}, health=health)
    loop2.run_once()
    t = loop2._adopt_thread
    if t is not None:
        t.join(timeout=30)
    assert seen and all(s.get("adopted") is True for s in seen)  # the re-gate


# ── operator lanes survive republishes ───────────────────────────────────────

OPERATOR_LANE = '''
[[host]]
name          = "cascade-900-final-m0-g0"
host          = "10.99.0.1"
port          = 2222
user          = "root"
key_path      = "~/.ssh/lium_cascade_ed25519"
remote_python = "/root/cascade/.venv/bin/python"
workdir       = "/root/cascade"
cuda_device   = "0"
stage         = "final"
chain_toml    = "/root/cascade/chain.deployed.toml"
forward_env   = ["HIPPIUS_S3_ACCESS_KEY", "HF_TOKEN"]
ssh_options   = ["StrictHostKeyChecking=accept-new"]
'''

STALE_SELF_LANE = '''
[[host]]
name = "cascade-800-final-0-g0"
host = "10.88.0.1"
stage = "final"
'''



LEFTOVER = [("final", "cascade-900-final-0", "NVIDIA L40S", 2)]


def _leftover_fleet(tmp_path, clock):
    """A ledger leftover of the retired per-round fleet, published to hosts.toml."""
    prov = FakeProvider("lium")
    _seed_ledger(tmp_path, prov, LEFTOVER, clock=clock)
    loop = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop._republish_from_ledger()
    return loop, prov


def _reap(loop, clock):
    clock.t += loop.ttl_hours * 3600.0
    cycle(loop)


def test_operator_lanes_survive_a_teardown_republish(tmp_path):
    clock = Clock()
    loop, prov = _leftover_fleet(tmp_path, clock)
    hosts_path = tmp_path / "hosts.toml"
    # The operator appends a hand-rented lane, and a stale lane a previous
    # provisioner run published for a pod that is gone.
    hosts_path.write_text(hosts_path.read_text() + OPERATOR_LANE + STALE_SELF_LANE)
    _reap(loop, clock)
    assert prov.live == {}
    hosts = load_hosts(hosts_path)
    assert [h.name for h in hosts] == ["cascade-900-final-m0-g0"]   # preserved; stale dropped
    lane = hosts[0]
    assert (lane.host, lane.port, lane.cuda_device, lane.stage) == ("10.99.0.1", 2222, "0", "final")
    assert lane.forward_env == ("HIPPIUS_S3_ACCESS_KEY", "HF_TOKEN")
    assert lane.ssh_options == ("StrictHostKeyChecking=accept-new",)
    assert lane.chain_toml == "/root/cascade/chain.deployed.toml"


def test_our_own_terminated_pods_are_never_preserved_as_operator_lanes(tmp_path):
    clock = Clock()
    loop, prov = _leftover_fleet(tmp_path, clock)
    hosts_path = tmp_path / "hosts.toml"
    # Rename our lanes to an operator-looking shape: address ownership must
    # still drop them once the pod is terminated.
    hosts_path.write_text(hosts_path.read_text().replace("cascade-lium-", "cascade-operator-"))
    _reap(loop, clock)
    with pytest.raises(RemoteDispatchError):
        load_hosts(hosts_path)                               # nothing left: cleared


def test_torn_hosts_file_preserves_nothing(tmp_path):
    clock = Clock()
    loop, _ = _leftover_fleet(tmp_path, clock)
    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text(hosts_path.read_text() + OPERATOR_LANE + "\n[[host]\nbroken = ")
    loop._republish_from_ledger()
    names = [h.name for h in load_hosts(hosts_path)]
    assert "cascade-900-final-m0-g0" not in names            # garbage never re-emitted
    assert names and all(n.startswith("cascade-lium-") for n in names)


def test_lane_names_derive_from_pod_identity_only(tmp_path):
    """A republish under a new round id must never rename a lane the trainer
    already holds (its name-keyed pool would dispatch onto a renamed twin)."""
    from dataclasses import replace as dc_replace

    from cascade.provision.loop import lane_pod_name

    loop, _ = _leftover_fleet(tmp_path, Clock())
    first = {h.name: (h.host, h.cuda_device) for h in load_hosts(tmp_path / "hosts.toml")}
    stem = lane_pod_name("lium", "cascade-900-final-0")
    assert sorted(first) == [stem + "-g0", stem + "-g1"]
    loop._state = dc_replace(loop._state, round_id="9010800")
    loop._republish_from_ledger()
    assert {h.name: (h.host, h.cuda_device)
            for h in load_hosts(tmp_path / "hosts.toml")} == first


def test_lane_pod_name_shapes():
    from cascade.provision.loop import _SELF_LANE_RE, lane_pod_name

    assert lane_pod_name("shadeform", "bbc38fd9-b0f9-4156-a9ad-ccb1864e8e8a") == "cascade-shadeform-bbc38fd9"
    lium = lane_pod_name("lium", "cascade-900-heat-0")
    assert lium.startswith("cascade-lium-") and len(lium) == len("cascade-lium-") + 8
    assert lane_pod_name("lium", "cascade-900-heat-0") == lium          # deterministic
    assert lane_pod_name("lium", "cascade-900-heat-1") != lium
    for name in (lium + "-g0", "cascade-shadeform-bbc38fd9-g1", "cascade-800-final-0-g0"):
        assert _SELF_LANE_RE.match(name), name                          # our shapes
    for name in ("cascade-8996400-final-m0-g0", "cascade-final-b", "benchpod-r64-g0"):
        assert not _SELF_LANE_RE.match(name), name                      # operator shapes


def test_static_fragment_is_reread_on_every_publish(tmp_path):
    static_path = tmp_path / "static.toml"
    static_path.write_text('[[host]]\nname = "cascade-final-b"\nhost = "216.81.245.151"\nstage = "final"\n')
    loop = make_loop(tmp_path)
    loop.static_hosts_text = static_path.read_text()
    loop.static_hosts_path = static_path
    loop._republish_from_ledger()
    assert "cascade-final-b" in (tmp_path / "hosts.toml").read_text()
    # Operator edits the fragment: the next publish carries the edit.
    static_path.write_text('[[host]]\nname = "cascade-final-c"\nhost = "216.81.245.152"\nstage = "final"\n')
    loop._republish_from_ledger()
    text = (tmp_path / "hosts.toml").read_text()
    assert "cascade-final-c" in text and "cascade-final-b" not in text
    # A broken edit keeps the last-good fragment instead of publishing garbage.
    static_path.write_text("[[host]\nbroken")
    loop._republish_from_ledger()
    assert "cascade-final-c" in (tmp_path / "hosts.toml").read_text()
    load_hosts(tmp_path / "hosts.toml")


def test_render_host_entries_round_trips_through_tomllib_and_load_hosts(tmp_path):
    entries = parse_host_entries(OPERATOR_LANE)
    text = render_host_entries(entries)
    assert tomllib.loads(text)["host"] == entries
    p = tmp_path / "h.toml"
    p.write_text(text)
    (h,) = load_hosts(p)
    assert h.name == "cascade-900-final-m0-g0" and h.port == 2222
    # Odd values stay valid TOML: quotes, backslashes, unicode.
    weird = [{"name": 'a"b\\c', "host": "10.0.0.1", "note": "ünïcode", "port": 22,
              "flag": True, "ratio": 0.5, "forward_env": []}]
    assert tomllib.loads(render_host_entries(weird))["host"] == weird
