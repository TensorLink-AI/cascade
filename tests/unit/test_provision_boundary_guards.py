"""Round-boundary guards for the provisioner (2026-09-04 incidents).

* H1 — a heat/final pod whose teardown signal is due is NOT terminated while
  training legs are running on it (or while it cannot be reached); the TTL
  stays the only unconditional kill.
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
    FakeStore,
    _report,
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
    ls = next(a for a in calls if a[0] == "ls")
    assert grep[-2:] == ["/etc/environment", "/home/shadeform/.bashrc"]
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
    """Same treatment as image_digest (r51): a recycled container is a facility
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
        recycled = stage == "heat" and prov.region_by_pod[pid] == "kansascity-usa-1"
        return _report(ok=not recycled, name="fresh_boot")

    loop, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    cycle(loop)
    rspec = next(s for s in prov.specs if "-r0" in s.name_prefix)
    assert rspec.exclude_regions == ("kansascity-usa-1",)
    assert prov.region_by_pod["cascade-900-heat-r0-0"] == "chicago-usa-2"
    assert loop._gate_failures == {}


def test_regate_passes_adopted_flag_to_the_gate(tmp_path):
    seen = []

    def health(addr, stage, provider="", **shape):
        seen.append(shape)
        return _report(ok=True)

    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    cycle(loop)
    assert seen and all(not s.get("adopted") for s in seen)     # fresh rentals
    seen.clear()
    loop2, _ = make_loop(tmp_path, providers={"lium": prov}, health=health)
    loop2.run_once()
    t = loop2._adopt_thread
    if t is not None:
        t.join(timeout=30)
    assert seen and all(s.get("adopted") is True for s in seen)  # the re-gate


# ── H1: liveness guard ───────────────────────────────────────────────────────


def _manifest_round(tmp_path, clock, store, loop, prov):
    """Marker lands, then the round manifest publishes: the final is due."""
    d = tmp_path / "work" / "54321"
    d.mkdir(parents=True)
    (d / "heat_complete.json").write_text("{}")
    clock.t += 1800.0
    cycle(loop)
    assert "cascade-900-final-0" in prov.live
    store.texts["manifests/round-54321.json"] = '{"round_id": "54321"}'
    clock.t += 1800.0


def _fleet(tmp_path, clock, store, legs, heat_legs=0):
    """A heat+final fleet whose FINAL pod reports ``legs()`` live legs (the
    heat pod reports ``heat_legs``, idle by default so the marker reaps it)."""
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock, store=store)
    probed = []

    def live_legs(addr, provider=""):
        probed.append((addr.ip, provider))
        pid = next((p for p, a in prov.live.items() if a.ip == addr.ip), "")
        return heat_legs if "heat" in pid else legs()

    loop.live_legs = live_legs
    cycle(loop)
    assert len(prov.live) == 2
    return loop, prov, probed


def test_final_with_live_legs_survives_its_manifest(tmp_path):
    """The 2026-09-04 20:30 class: the OLD round's manifest+bench signals fired
    7 s after the NEW round dispatched onto the same pods."""
    clock, store = Clock(), FakeStore()
    loop, prov, probed = _fleet(tmp_path, clock, store, legs=lambda: 2)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert "cascade-900-final-0" in prov.live                  # deferred, not killed
    assert probed and probed[-1][1] == "lium"
    hosts = load_hosts(tmp_path / "hosts.toml")
    assert [h.stage for h in hosts] == ["final", "final"]     # still published
    # Legs finish → the next sweep tears it down.
    loop.live_legs = lambda addr, provider="": 0
    clock.t += 60.0
    cycle(loop)
    assert prov.live == {}


def test_unreachable_pod_is_not_proof_of_idleness(tmp_path):
    clock, store = Clock(), FakeStore()
    loop, prov, _ = _fleet(tmp_path, clock, store, legs=lambda: None)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert "cascade-900-final-0" in prov.live

    def boom(addr, provider=""):
        raise OSError("ssh: connect refused")

    loop.live_legs = boom
    clock.t += 60.0
    cycle(loop)
    assert "cascade-900-final-0" in prov.live                  # errors defer too


def test_ttl_still_kills_a_pod_with_live_legs(tmp_path):
    clock, store = Clock(), FakeStore()
    loop, prov, _ = _fleet(tmp_path, clock, store, legs=lambda: 3)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert "cascade-900-final-0" in prov.live
    clock.t += loop.ttl_hours * 3600.0 + 1.0                  # TTL: unconditional
    cycle(loop)
    assert prov.live == {}


def test_heat_pod_with_live_legs_survives_the_marker(tmp_path):
    clock, store = Clock(), FakeStore()
    loop, prov, _ = _fleet(tmp_path, clock, store, legs=lambda: 0, heat_legs=1)
    d = tmp_path / "work" / "54321"
    d.mkdir(parents=True)
    (d / "heat_complete.json").write_text("{}")
    clock.t += 1800.0
    cycle(loop)
    assert "cascade-900-heat-0" in prov.live                   # legs still running
    loop.live_legs = lambda addr, provider="": 0
    clock.t += 60.0
    cycle(loop)
    assert "cascade-900-heat-0" not in prov.live
    assert "cascade-900-final-0" in prov.live


def test_no_probe_configured_keeps_the_old_behaviour(tmp_path):
    clock, store = Clock(), FakeStore()
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock, store=store)
    cycle(loop)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert prov.live == {}


# ── H3: operator lanes survive republishes ───────────────────────────────────

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


def _heat_marker(tmp_path, clock, loop):
    d = tmp_path / "work" / "54321"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heat_complete.json").write_text("{}")
    clock.t += 1800.0
    cycle(loop)


def test_operator_lanes_survive_a_teardown_republish(tmp_path):
    clock = Clock()
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    cycle(loop)
    hosts_path = tmp_path / "hosts.toml"
    # The operator appends a hand-rented lane (2026-09-04 flip-round fleet)
    # and a stale lane a previous provisioner run published for a pod that is gone.
    hosts_path.write_text(hosts_path.read_text() + OPERATOR_LANE + STALE_SELF_LANE)

    _heat_marker(tmp_path, clock, loop)                    # heat reaped → republish
    hosts = load_hosts(hosts_path)
    names = [h.name for h in hosts]
    assert "cascade-900-final-m0-g0" in names               # preserved
    assert "cascade-800-final-0-g0" not in names            # stale self-lane dropped
    assert all("heat" not in n for n in names)              # our heat pods gone
    assert len(names) == len(set(names))                    # no duplicates
    lane = next(h for h in hosts if h.name == "cascade-900-final-m0-g0")
    assert (lane.host, lane.port, lane.cuda_device, lane.stage) == ("10.99.0.1", 2222, "0", "final")
    assert lane.forward_env == ("HIPPIUS_S3_ACCESS_KEY", "HF_TOKEN")
    assert lane.ssh_options == ("StrictHostKeyChecking=accept-new",)
    assert lane.chain_toml == "/root/cascade/chain.deployed.toml"


def test_operator_lane_survives_a_full_teardown_clear(tmp_path):
    """'No dynamic pods' must degrade to 'operator fleet', never an empty file."""
    clock, store = Clock(), FakeStore()
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock, store=store)
    cycle(loop)
    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text(hosts_path.read_text() + OPERATOR_LANE)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert prov.live == {}
    assert [h.name for h in load_hosts(hosts_path)] == ["cascade-900-final-m0-g0"]


def test_our_own_terminated_pods_are_never_preserved_as_operator_lanes(tmp_path):
    clock, store = Clock(), FakeStore()
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock, store=store)
    cycle(loop)
    hosts_path = tmp_path / "hosts.toml"
    # Rename our final lanes to an operator-looking shape: address ownership
    # must still drop them once the pod is terminated.
    text = hosts_path.read_text().replace("cascade-900-final-0-g", "cascade-900-final-mX-g")
    hosts_path.write_text(text)
    _manifest_round(tmp_path, clock, store, loop, prov)
    cycle(loop)
    assert prov.live == {}
    with pytest.raises(RemoteDispatchError):
        load_hosts(hosts_path)                               # nothing left: cleared


def test_torn_hosts_file_preserves_nothing(tmp_path):
    clock = Clock()
    prov = FakeProvider("lium")
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    cycle(loop)
    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text(hosts_path.read_text() + OPERATOR_LANE + "\n[[host]\nbroken = ")
    _heat_marker(tmp_path, clock, loop)
    names = [h.name for h in load_hosts(hosts_path)]
    assert "cascade-900-final-m0-g0" not in names            # garbage never re-emitted
    assert names and all(n.startswith("cascade-900-final-0-g") for n in names)


def test_static_fragment_is_reread_on_every_publish(tmp_path):
    clock = Clock()
    prov = FakeProvider("lium")
    static_path = tmp_path / "static.toml"
    static_path.write_text('[[host]]\nname = "cascade-final-b"\nhost = "216.81.245.151"\nstage = "final"\n')
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, clock=clock)
    loop.static_hosts_text = static_path.read_text()
    loop.static_hosts_path = static_path
    cycle(loop)
    assert "cascade-final-b" in (tmp_path / "hosts.toml").read_text()
    # Operator edits the fragment: the next publish carries the edit.
    static_path.write_text('[[host]]\nname = "cascade-final-c"\nhost = "216.81.245.152"\nstage = "final"\n')
    _heat_marker(tmp_path, clock, loop)
    text = (tmp_path / "hosts.toml").read_text()
    assert "cascade-final-c" in text and "cascade-final-b" not in text
    # A broken edit keeps the last-good fragment instead of publishing garbage.
    static_path.write_text("[[host]\nbroken")
    loop._republish_from_ledger()
    text = (tmp_path / "hosts.toml").read_text()
    assert "cascade-final-c" in text
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
