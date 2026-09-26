"""The JIT king pod's SSH host key is pinned like a payer leg's (2026-09-15).

Round 9075600, 21:09 UTC: the king pod came up on a Lium ip:port that an
earlier container had used. The king's RemoteHost carried no pin, so every
ssh ran accept-new against the shared ~/.ssh/known_hosts, whose stale entry
made the very first mkdir fail with "Host key verification failed"; the king
leg died and the pod idled. Payer legs were immune: rent_funded_pod scans the
pod's key at readiness and the leg pins it. The king now does the same on
both the fresh-rent and the adopt path, and a pod whose key cannot be scanned
is a lemon (torn down, rented again elsewhere).
"""
from __future__ import annotations

from types import SimpleNamespace

from cascade.provision.core import ProvisionError
from cascade.trainer.remote import pinned_known_hosts_file, ssh_transport_options
from tests.unit.test_funded_pod_wiring import _runner

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKingPodKey"


def _install(monkeypatch, prov):
    import cascade.provision.core as core_mod
    from cascade.provision import funded as funded_mod

    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    monkeypatch.setattr(core_mod, "LiumProvider", prov)
    torn, lemons = [], []
    monkeypatch.setattr(funded_mod, "quarantine_lemon_host",
                        lambda provider, pod_id, why: lemons.append((pod_id, why)))
    monkeypatch.setattr(funded_mod, "terminate_verified",
                        lambda provider, pod_id: torn.append(pod_id) or True)
    return torn, lemons


class _Prov:
    name = "lium"
    launched: list

    def __init__(self, *a, **kw):
        pass

    def capacity(self, sku, *, gpus=1, exclude_ids=()):
        return 1

    def launch(self, spec):
        type(self).launched.append(spec.sku)
        return [f"{spec.name_prefix}-0"]

    def wait_ready(self, pod_id, *, timeout):
        return True

    def get_ip(self, pod_id):
        return SimpleNamespace(ip="9.9.9.9", ssh_port=41000 + len(type(self).launched))

    def machine_of(self, pod_id):
        return f"exec-{len(type(self).launched)}"

    def terminate(self, pod_id):
        pass


def _fresh_prov():
    class P(_Prov):
        launched = []
    return P


def test_fresh_rent_pins_the_scanned_key_and_the_transport_enforces_it(tmp_path, monkeypatch):
    P = _fresh_prov()
    _install(monkeypatch, P)
    r = _runner(tmp_path, funded_king_rent=True)
    r._funded_round_sku = "RTX4090"
    scanned = []
    r._host_key_scanner = lambda ip, port: scanned.append((ip, port)) or KEY

    host = r._rent_king_host("42")
    # Credential-free like a payer pod: nothing forwarded, harvested by us.
    assert host.isolated and host.forward_env == ()

    assert scanned == [("9.9.9.9", 41001)]
    assert host.pinned_host_key == KEY
    opts = ssh_transport_options(host)
    assert "StrictHostKeyChecking=yes" in opts
    assert f"UserKnownHostsFile={pinned_known_hosts_file(host)}" in opts
    assert pinned_known_hosts_file(host).read_text() == f"[9.9.9.9]:41001 {KEY}\n"
    assert "StrictHostKeyChecking=accept-new" not in opts


def test_unscannable_pod_is_a_lemon_and_the_king_rents_again(tmp_path, monkeypatch):
    P = _fresh_prov()
    torn, lemons = _install(monkeypatch, P)
    r = _runner(tmp_path, funded_king_rent=True)
    r._funded_round_sku = "RTX4090"
    r._rent_wait_now = lambda: 0.0                  # far before the latest safe start
    r._rent_wait_sleep = lambda s: None
    slept = []
    r._scan_retry_sleep = lambda s: slept.append(s)
    ports = []

    def scanner(ip, port):
        ports.append(port)
        if port == 41001:                           # first pod: sshd never answers
            raise RuntimeError("no ssh host key scanned")
        return KEY

    r._host_key_scanner = scanner
    host = r._rent_king_host("42")

    assert P.launched == ["RTX4090", "RTX4090"]     # first pod dropped, rented again
    assert torn == ["cascade-n91-42-funded-king-0"]
    assert lemons and "could not pin its ssh host key" in lemons[0][1]
    assert ports.count(41001) == r.KING_HOST_KEY_SCAN_TRIES   # retried before giving up
    assert len(slept) == r.KING_HOST_KEY_SCAN_TRIES - 1
    assert host.port == 41002 and host.pinned_host_key == KEY


def test_adopted_king_pod_is_pinned_too(tmp_path, monkeypatch):
    P = _fresh_prov()
    _install(monkeypatch, P)
    P.live_pod_address = lambda self, name: SimpleNamespace(ip="7.7.7.7", ssh_port=2222)
    r = _runner(tmp_path, funded_king_rent=True)
    r._funded_round_sku = "RTX4090"
    scanned = []
    r._host_key_scanner = lambda ip, port: scanned.append((ip, port)) or KEY

    host = r._rent_king_host("42")

    assert P.launched == []                         # adopted, never rented
    assert scanned == [("7.7.7.7", 2222)]
    assert (host.host, host.port, host.pinned_host_key) == ("7.7.7.7", 2222, KEY)


def test_pin_helper_retries_then_raises_provision_error(tmp_path):
    r = _runner(tmp_path, funded_king_rent=True)
    calls = []
    r._host_key_scanner = lambda ip, port: calls.append(1) or ""
    r._scan_retry_sleep = lambda s: None
    try:
        r._pin_king_host_key("p-0", SimpleNamespace(ip="1.2.3.4", ssh_port=22))
    except ProvisionError as e:
        assert "could not pin its ssh host key at 1.2.3.4:22" in str(e)
    else:  # pragma: no cover
        raise AssertionError("empty scans must raise")
    assert len(calls) == r.KING_HOST_KEY_SCAN_TRIES
