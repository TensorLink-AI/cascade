"""A leg that is already running on its pod is ATTACHED to, never relaunched or re-rented.

Failure class (2026-09-24 12:03, trainer restart mid-era): seven in-flight legs were
re-dispatched from scratch. Each rent created a SECOND pod under the payer's pod name
while the first kept training; the platform lookups resolved the name to whichever
twin listed first, the identity pin saw the booting twin ("pod status is 'PENDING'")
and four legs were burned as tamper; the rest cycled three rents each. Three layers
close it: the dispatcher attaches to a prior run dir on the pod, the funded rent
adopts a live pod of the leg's name, and the Lium name lookup prefers the pod that
is actually RUNNING when a name is doubled.
"""
from __future__ import annotations

import types

from cascade.provision.core import LiumProvider, PodAddress, ProvisionError
from cascade.provision.funded import FUNDED_STAGE, funded_pod_name, rent_funded_pod
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    RECEIPT_SENTINEL,
    RemoteDispatcher,
    RemoteHost,
    find_detached_run,
    run_detached,
)

HK = "5FbwZ4y4VG9f5iXpCSCqGWVERZgq98j9r4e2peYbUtDudy4o"


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _remote_cmd(argv):
    return argv[-1]


class _Scripted:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, argv, timeout, stdin=None):
        self.calls.append((_remote_cmd(argv), stdin))
        step = self.steps.pop(0) if self.steps else _proc(stdout="RUNNING\n")
        if isinstance(step, Exception):
            raise step
        return step


def _host(**kw):
    kw.setdefault("name", "funded-5fbwz4y4vg9f")
    kw.setdefault("host", "10.0.0.1")
    kw.setdefault("port", 2222)
    kw.setdefault("user", "root")
    kw.setdefault("key_path", "/k")
    kw.setdefault("workdir", "/root/cascade")
    kw.setdefault("remote_python", "/root/cascade/.venv/bin/python")
    kw.setdefault("cuda_device", "0")
    return RemoteHost(**kw)


def _receipt_stdout():
    import json

    from cascade.shared.manifest import format_trained_pointer
    entry = {"role": "challenger", "miner_uid": 7, "miner_hotkey": HK, "size": "toto2-4m",
             "trained_pointer": format_trained_pointer("cascade/ckpt-r1-c@sha256:" + "a" * 64),
             "corpus_digest": "d", "gen_ref": "g", "train_block": 1, "bench_scores": None,
             "duel_rank": None, "gpu_name": "", "warm_started": False}
    return RECEIPT_SENTINEL + json.dumps(entry) + "\n"


# ── remote: find + attach ──────────────────────────────────────────────────

def test_find_detached_run_returns_the_live_or_finished_run_dir():
    runner = _Scripted([_proc(stdout="ATTACH:_train_work/_dispatch/challenger-5FbwZ4y4VG9f-1-ab12cd34\n")])
    rd = find_detached_run(_host(), f"challenger-{HK[:12]}-1", runner=runner)
    assert rd == "/root/cascade/_train_work/_dispatch/challenger-5FbwZ4y4VG9f-1-ab12cd34"
    cmd, stdin = runner.calls[0]
    assert stdin == ""                                   # no credentials for a probe
    assert "_train_work/_dispatch/challenger-5FbwZ4y4VG9f-1-*" in cmd
    assert "exit_code" in cmd and "kill -0" in cmd       # alive OR finished both count


def test_find_detached_run_is_none_when_nothing_is_there_or_the_pod_is_unreachable():
    assert find_detached_run(_host(), "king-x-1", runner=_Scripted([_proc(stdout="")])) is None
    assert find_detached_run(_host(), "king-x-1", runner=_Scripted([_proc(rc=255)])) is None
    assert find_detached_run(_host(), "king-x-1", runner=_Scripted([OSError("down")])) is None


def test_run_detached_attach_skips_the_launch_and_polls_the_given_dir():
    runner = _Scripted([
        _proc(stdout="RUNNING\n"),
        _proc(stdout="EXIT:0\n"),
        _proc(stdout="OUT\n" + DETACHED_STDERR_MARK + "\nERR\n"),
    ])
    proc = run_detached(_host(), "true", "K=v\n", "/root/cascade/_train_work/_dispatch/leg-1",
                        timeout=3600, poll_seconds=0, grace_seconds=60, runner=runner,
                        launch=False)
    assert proc.returncode == 0 and proc.stdout.strip() == "OUT"
    assert all(DETACHED_LAUNCH_TOKEN not in cmd for cmd, _ in runner.calls)
    assert all(stdin == "" for _, stdin in runner.calls)   # credentials never sent
    assert "_dispatch/leg-1" in runner.calls[0][0]


def test_dispatcher_attaches_to_a_prior_run_instead_of_relaunching():
    runner = _Scripted([
        _proc(stdout="ATTACH:_train_work/_dispatch/challenger-5FbwZ4y4VG9f-1-ab12cd34\n"),  # find
        _proc(stdout="EXIT:0\n"),                                                        # poll
        _proc(stdout=_receipt_stdout() + DETACHED_STDERR_MARK + "\nlogs\n"),             # fetch
    ])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=0,
                            reattach_grace_seconds=60, _runner=runner)
    entry = disp.dispatch(_host(), gen_ref="g", uid=7, hotkey=HK, role="challenger",
                          base_seed=1, block=1)
    assert entry.miner_hotkey == HK
    assert all(DETACHED_LAUNCH_TOKEN not in cmd for cmd, _ in runner.calls)
    assert all("challenger-5FbwZ4y4VG9f-1-ab12cd34" in cmd for cmd, _ in runner.calls[1:])


def test_dispatcher_launches_when_no_prior_run_exists():
    runner = _Scripted([
        _proc(stdout=""),                                                    # find: nothing
        _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),                          # launch
        _proc(stdout="EXIT:0\n"),
        _proc(stdout=_receipt_stdout() + DETACHED_STDERR_MARK + "\nlogs\n"),
    ])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=0,
                            reattach_grace_seconds=60, _runner=runner)
    disp.dispatch(_host(), gen_ref="g", uid=7, hotkey=HK, role="challenger",
                  base_seed=1, block=1)
    assert DETACHED_LAUNCH_TOKEN in runner.calls[1][0]
    assert "_dispatch/challenger-5FbwZ4y4VG9f-1-" in runner.calls[1][0]


# ── funded rent: adopt the live pod ────────────────────────────────────────

class _Provider:
    name = "lium"

    def __init__(self, live: PodAddress | None, ident_ok: bool = True):
        self.live = live
        self.ident_ok = ident_ok
        self.launched: list = []
        self.terminated: list = []

    def live_pod_address(self, pod_id):
        # the provider names the pod "<prefix>-0"; the bare prefix is nobody
        return self.live if pod_id.endswith("-0") else None

    def launch(self, spec):
        self.launched.append(spec)
        return [f"{spec.name_prefix}-0"]

    def wait_ready(self, pod_id, *, timeout):
        return True

    def get_ip(self, pod_id):
        return PodAddress(ip="10.0.0.9", ssh_port=2222)

    def terminate(self, pod_id):
        self.terminated.append(pod_id)

    def list_tagged(self, prefix):
        return [f"{prefix}-0"] if self.live is not None else []

    def pod_identity(self, pod_id):
        assert pod_id.endswith("-0"), pod_id
        if not self.ident_ok:
            return None
        return {"id": "uid-old", "huid": "h1", "status": "RUNNING",
                "ip": "10.0.0.7", "port": 20300}


def _rent(provider, **kw):
    return rent_funded_pod(round_id="777", hotkey=HK, api_key="sk-miner", sku="H100",
                           image="img@sha256:" + "0" * 64, ssh_pubkey="ssh-ed25519 AAA",
                           netuid=91, provider_factory=lambda key: provider, **kw)


def test_rent_adopts_a_live_pod_of_this_name_instead_of_renting_a_twin():
    provider = _Provider(live=PodAddress(ip="10.0.0.7", ssh_port=20300))
    scanned: list = []

    def scanner(ip, port):
        scanned.append((ip, port))
        return "ssh-ed25519 PINNED"

    res = _rent(provider, host_key_scanner=scanner)
    assert res.ok and res.adopted
    assert provider.launched == []                       # no second pod
    assert res.address.ip == "10.0.0.7" and res.address.ssh_port == 20300
    assert res.pod.instance_id == funded_pod_name("777", HK, 91) + "-0"   # the listed pod, not the prefix
    assert res.pod.stage == FUNDED_STAGE and res.pod.pod_uid == "uid-old"
    assert res.host_key == "ssh-ed25519 PINNED" and scanned == [("10.0.0.7", 20300)]


def test_rent_without_a_live_pod_rents_as_before():
    provider = _Provider(live=None)
    res = _rent(provider)
    assert res.ok and not res.adopted
    assert len(provider.launched) == 1


def test_adoption_failure_is_infra_and_never_touches_the_live_pod():
    provider = _Provider(live=PodAddress(ip="10.0.0.7", ssh_port=20300), ident_ok=False)
    res = _rent(provider)
    assert not res.ok and res.error_class == "infra" and not res.burn_attempt
    assert "could not be adopted" in res.error
    assert provider.launched == [] and provider.terminated == []


def test_adoption_probe_error_falls_back_to_a_rent():
    class _Flaky(_Provider):
        def live_pod_address(self, pod_id):
            raise ProvisionError("lium ps timed out")

    provider = _Flaky(live=None)
    res = _rent(provider)
    assert res.ok and not res.adopted and len(provider.launched) == 1


# ── lium: a doubled name resolves to the pod that is RUNNING ───────────────

def _lium_with(pods):
    prov = LiumProvider.__new__(LiumProvider)
    prov._list_pods = lambda: pods  # type: ignore[method-assign]
    return prov


def test_lium_name_lookup_prefers_the_running_pod_over_its_booting_twin():
    name = "cascade-n91-1-funded-5fbwz4y4vg9f-0"
    booting = {"name": name, "id": "uid-new", "huid": "h2", "status": "PENDING"}
    running = {"name": name, "id": "uid-old", "huid": "h1", "status": "RUNNING",
               "ssh_cmd": "ssh root@10.0.0.7 -p 20300"}
    for order in ([booting, running], [running, booting]):
        prov = _lium_with(order)
        assert prov.pod_identity(name)["id"] == "uid-old"
        assert prov.pod_identity(name)["status"] == "RUNNING"
    # a lone booting pod still resolves (wait_ready keeps polling it)
    assert _lium_with([booting]).pod_identity(name)["id"] == "uid-new"
    assert _lium_with([]).pod_identity(name) is None
