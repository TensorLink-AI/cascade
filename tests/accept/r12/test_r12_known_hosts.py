"""R12: pinned known_hosts and endpoint resolution (plan §10.2 "pinned
known_hosts per pod"; OPSLOG 2026-11-22: Lium's NAT moved a RUNNING pod's IP
with the host key unchanged — resolve the endpoint through ``lium ps`` before
declaring a pod dead, and re-pin the same key under the new IP; memory
"Lium reused ip:port poisons shared known_hosts").

``evalhost.py pin-host-key`` pins a scanned key and REFUSES a changed one;
``evalhost.py resolve-endpoint`` finds the pod in captured ``lium ps`` JSON
and tells "same pod, re-pin" apart from "different key". Both read captures
only: no ssh, no ssh-keyscan, no lium. Contract: README.md.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from .helpers import (
    EXIT,
    NEW_HOST,
    OLD_HOST,
    PORT,
    Kit,
    as_json,
    fingerprint,
    hashed_line,
    key_blob,
    keyscan_text,
    known_line,
    output,
    pod_entry,
)

K1, K2 = key_blob("pod-key-1"), key_blob("pod-key-2")
OTHER = known_line("192.0.2.44", 2201, key_blob("other-pod"))


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _pin(kit, kh, *, host=NEW_HOST, port=PORT, blob=K1, extra=()):
    tag = hashlib.sha1(blob.encode()).hexdigest()[:12]
    scanned = kit.write(f"scan_{host}_{port}_{tag}.txt", keyscan_text(host, port, blob))
    proc = kit.tool("pin-host-key", "--known-hosts", kh, "--host", host, "--port", port,
                    "--scanned", scanned, *extra)
    assert kit.shim_calls() == [], f"pin-host-key ran an external command: {kit.shim_calls()}"
    return proc, as_json(proc)


# ── pin-host-key ─────────────────────────────────────────────────────────────


def test_first_pin_adds_the_scanned_key_and_keeps_other_entries(kit):
    kh = kit.write("known_hosts_eval", OTHER + "\n")
    proc, out = _pin(kit, kh)
    assert proc.returncode == 0, output(proc)
    assert out["action"] == "added", out
    assert out["fingerprints"] == [fingerprint(K1)], out
    text = kh.read_text()
    assert text.startswith(OTHER + "\n"), "existing entries must be kept byte for byte"
    assert known_line(NEW_HOST, PORT, K1) in text.splitlines(), text


def test_first_pin_creates_a_missing_file(kit):
    kh = kit.tmp / "ssh" / "known_hosts_eval"
    proc, out = _pin(kit, kh)
    assert proc.returncode == 0, output(proc)
    assert out["action"] == "added", out
    assert kh.read_text().splitlines() == [known_line(NEW_HOST, PORT, K1)]


def test_port_22_uses_the_bare_host_pattern(kit):
    kh = kit.write("known_hosts_eval", "")
    proc, out = _pin(kit, kh, port=22)
    assert proc.returncode == 0, output(proc)
    assert f"{NEW_HOST} ssh-ed25519 {K1}" in kh.read_text().splitlines(), kh.read_text()


def test_same_key_again_is_unchanged_and_idempotent(kit):
    kh = kit.write("known_hosts_eval", OTHER + "\n" + known_line(NEW_HOST, PORT, K1) + "\n")
    before = kh.read_bytes()
    for _ in range(2):
        proc, out = _pin(kit, kh)
        assert proc.returncode == 0, output(proc)
        assert out["action"] == "unchanged", out
        assert kh.read_bytes() == before


def test_changed_key_is_refused_and_the_file_untouched(kit):
    kh = kit.write("known_hosts_eval", known_line(NEW_HOST, PORT, K1) + "\n")
    before = kh.read_bytes()
    proc, out = _pin(kit, kh, blob=K2)
    assert proc.returncode == EXIT["host_key_changed"], output(proc)
    assert out["action"] == "refused" and out.get("reason") == "host_key_changed", out
    assert out["fingerprints"] == [fingerprint(K2)], out
    assert out["pinned_fingerprints"] == [fingerprint(K1)], out
    assert kh.read_bytes() == before


def test_hashed_entries_are_recognised(kit):
    kh = kit.write("known_hosts_eval", hashed_line(NEW_HOST, PORT, K1) + "\n")
    before = kh.read_bytes()
    proc, out = _pin(kit, kh)
    assert proc.returncode == 0 and out["action"] == "unchanged", output(proc)
    proc, out = _pin(kit, kh, blob=K2)
    assert proc.returncode == EXIT["host_key_changed"], output(proc)
    assert kh.read_bytes() == before


def test_expected_fingerprint_mismatch_is_refused(kit):
    kh = kit.tmp / "known_hosts_eval"
    proc, out = _pin(kit, kh, extra=("--fingerprint", fingerprint(K2)))
    assert proc.returncode == EXIT["fingerprint_mismatch"], output(proc)
    assert out["action"] == "refused", out
    assert not kh.exists(), "a refused pin must not create the file"
    proc, out = _pin(kit, kh, extra=("--fingerprint", fingerprint(K1)))
    assert proc.returncode == 0 and out["action"] == "added", output(proc)


def test_repin_under_a_new_ip_needs_the_previous_endpoints_key(kit):
    old = known_line(OLD_HOST, PORT, K1)
    kh = kit.write("known_hosts_eval", old + "\n")
    before = kh.read_bytes()
    prev = ("--previous", f"{OLD_HOST}:{PORT}")
    proc, out = _pin(kit, kh, blob=K2, extra=prev)
    assert proc.returncode == EXIT["host_key_changed"], (
        f"a different key at the new IP is not the same pod\n{output(proc)}")
    assert kh.read_bytes() == before
    proc, out = _pin(kit, kh, blob=K1, extra=prev)
    assert proc.returncode == 0, output(proc)
    assert out["action"] == "repinned", out
    lines = kh.read_text().splitlines()
    assert old in lines and known_line(NEW_HOST, PORT, K1) in lines, lines


def test_previous_endpoint_not_pinned_is_refused(kit):
    kh = kit.write("known_hosts_eval", OTHER + "\n")
    before = kh.read_bytes()
    proc, out = _pin(kit, kh, extra=("--previous", f"{OLD_HOST}:{PORT}"))
    assert proc.returncode != 0, output(proc)
    assert kh.read_bytes() == before


def test_dry_run_reports_the_action_and_writes_nothing(kit):
    kh = kit.tmp / "known_hosts_eval"
    proc, out = _pin(kit, kh, extra=("--dry-run",))
    assert proc.returncode == 0, output(proc)
    assert out["action"] == "added" and out.get("dry_run") is True, out
    assert not kh.exists()
    kh.write_text(known_line(NEW_HOST, PORT, K1) + "\n")
    before = kh.read_bytes()
    proc, out = _pin(kit, kh, blob=K2, extra=("--dry-run",))
    assert proc.returncode == EXIT["host_key_changed"], output(proc)
    assert kh.read_bytes() == before


def test_a_scan_without_keys_is_a_usage_error(kit):
    kh = kit.write("known_hosts_eval", "")
    scanned = kit.write("scan_empty.txt", f"# {NEW_HOST}:{PORT} SSH-2.0-OpenSSH_9.6p1\n")
    proc = kit.tool("pin-host-key", "--known-hosts", kh, "--host", NEW_HOST, "--port", PORT,
                    "--scanned", scanned)
    assert proc.returncode == EXIT["usage"], output(proc)
    assert kh.read_text() == ""


# ── resolve-endpoint ─────────────────────────────────────────────────────────


def _resolve(kit, pods, *, pod="valeval-blackwell4", configured=f"{OLD_HOST}:{PORT}",
             kh_text=None, scan=None):
    ps = kit.write("lium_ps.json", json.dumps(pods))
    args = ["resolve-endpoint", "--lium-ps", ps, "--pod", pod, "--configured", configured]
    if kh_text is not None:
        args += ["--known-hosts", kit.write("known_hosts_eval", kh_text)]
    if scan is not None:
        args += ["--scanned", kit.write("scan_new.txt", scan)]
    proc = kit.tool(*args)
    assert kit.shim_calls() == [], f"resolve-endpoint ran an external command: {kit.shim_calls()}"
    return proc, as_json(proc)


def test_resolve_same_endpoint(kit):
    pods = [pod_entry(host=OLD_HOST), pod_entry(huid="calm-lion-1e", name="valpriv-4090",
                                                host="192.0.2.10", port=3301)]
    proc, out = _resolve(kit, pods)
    assert proc.returncode == 0, output(proc)
    assert out["verdict"] == "same_endpoint", out
    assert (out["host"], out["port"]) == (OLD_HOST, PORT), out
    assert out["endpoint_changed"] is False, out


@pytest.mark.parametrize("form", ["ssh_cmd", "host_port"])
@pytest.mark.parametrize("by", ["name", "huid"])
def test_resolve_ip_moved_same_key_is_same_pod_repin(kit, form, by):
    pods = [pod_entry(host=NEW_HOST, form=form)]
    pod = "valeval-blackwell4" if by == "name" else "lunar-wolf-6a"
    proc, out = _resolve(kit, pods, pod=pod, kh_text=known_line(OLD_HOST, PORT, K1) + "\n",
                         scan=keyscan_text(NEW_HOST, PORT, K1))
    assert proc.returncode == EXIT["same_pod_repin"], output(proc)
    assert out["verdict"] == "same_pod_repin", out
    assert (out["host"], out["port"]) == (NEW_HOST, PORT), out
    assert out["endpoint_changed"] is True, out
    assert out["previous"] == f"{OLD_HOST}:{PORT}", out
    assert out["fingerprints"] == [fingerprint(K1)], out


def test_resolve_ip_moved_different_key_is_refused(kit):
    proc, out = _resolve(kit, [pod_entry(host=NEW_HOST)],
                         kh_text=known_line(OLD_HOST, PORT, K1) + "\n",
                         scan=keyscan_text(NEW_HOST, PORT, K2))
    assert proc.returncode == EXIT["endpoint_key_changed"], output(proc)
    assert out["verdict"] == "endpoint_key_changed", out


def test_resolve_same_endpoint_different_key_is_refused(kit):
    proc, out = _resolve(kit, [pod_entry(host=OLD_HOST)],
                         kh_text=known_line(OLD_HOST, PORT, K1) + "\n",
                         scan=keyscan_text(OLD_HOST, PORT, K2))
    assert proc.returncode == EXIT["endpoint_key_changed"], output(proc)
    assert out["verdict"] == "endpoint_key_changed", out


def test_resolve_ip_moved_without_a_scan_is_unverified(kit):
    proc, out = _resolve(kit, [pod_entry(host=NEW_HOST)])
    assert proc.returncode == EXIT["endpoint_changed_unverified"], output(proc)
    assert out["verdict"] == "endpoint_changed_unverified", out
    assert (out["host"], out["port"]) == (NEW_HOST, PORT), out


def test_resolve_pod_not_listed(kit):
    proc, out = _resolve(kit, [pod_entry(huid="calm-lion-1e", name="valpriv-4090")])
    assert proc.returncode == EXIT["not_found"], output(proc)
    assert out["verdict"] == "not_found", out


def test_resolve_pod_not_running(kit):
    proc, out = _resolve(kit, [pod_entry(host=OLD_HOST, status="STOPPED")])
    assert proc.returncode == EXIT["not_running"], output(proc)
    assert out["verdict"] == "not_running", out


def test_resolve_dry_run_prints_lium_ps_and_runs_nothing(kit):
    proc = kit.tool("resolve-endpoint", "--pod", "valeval-blackwell4", "--configured",
                    f"{OLD_HOST}:{PORT}", "--dry-run")
    assert proc.returncode == 0, output(proc)
    out = as_json(proc)
    assert out.get("dry_run") is True, out
    assert any(c.startswith("lium ps") for c in out.get("would_run", [])), out
    assert kit.shim_calls() == [], kit.shim_calls()
