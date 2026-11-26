"""Pod postmortem before teardown (owner 2026-11-26): the worker logs, the
progress file, nvidia-smi and dmesg come off the pod BEFORE ``lium rm``; a
slow or failed capture never blocks the teardown past its timeout; the size
cap holds; old postmortems are pruned. Torch-free, no ssh (fake transports)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import types
from pathlib import Path

from cascade.trainer import pod_postmortem as pm
from cascade.trainer.remote import RemoteHost


def _host():
    return RemoteHost(name="funded-x", host="203.0.113.5", port=2222, workdir="/root/cascade")


def _nonce_of(argv) -> str:
    return re.search(r"@@PM-([0-9a-f]+)@@", argv[-1]).group(1)


def _fake_pod(sections: dict[str, bytes], *, preamble=b""):
    def ssh(argv, timeout, max_bytes):
        n = _nonce_of(argv)
        out = bytearray(preamble)
        for name, body in sections.items():
            out += f"\n@@PM-{n}@@ {name}\n".encode() + body
        return bytes(out)[:max_bytes]
    return ssh


def test_capture_writes_logs_progress_gpu_and_dmesg(tmp_path):
    ssh = _fake_pod({"expert-5Hk-1-ab.stdout": b"step 10\n",
                     "expert-5Hk-1-ab.stderr": b"Traceback: boom\n",
                     "expert-5Hk-1-ab.progress.json": b'{"step": 10}',
                     "nvidia-smi.txt": b"H100\n", "dmesg_tail.txt": b"NVRM: Xid 79\n",
                     "../../etc/evil": b"x"})
    path = pm.capture(_host(), root=tmp_path, pod_name="cascade-91-5hk-0",
                      reason="THROUGHPUT ABORT", ssh=ssh, clock=lambda: 1_700_000_000)
    d = Path(path)
    assert d.parent == tmp_path and d.name == "cascade-91-5hk-0-20231114T221320Z"
    assert (d / "expert-5Hk-1-ab.stderr").read_text().startswith("Traceback: boom")
    assert (d / "expert-5Hk-1-ab.progress.json").exists()
    assert (d / "nvidia-smi.txt").exists() and "Xid" in (d / "dmesg_tail.txt").read_text()
    assert all(p.parent == d for p in d.rglob("*"))          # no path escapes the dir
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["reason"] == "THROUGHPUT ABORT"
    assert meta["host"] == "203.0.113.5:2222"


def test_remote_command_only_tails_small_named_files():
    cmd = pm.remote_command("/root/cascade", run_dir="/root/cascade/_train_work/_dispatch/r1",
                            nonce="ab", per_file_bytes=4096)
    assert "tail -c 4096" in cmd and "nvidia-smi" in cmd and "dmesg" in cmd
    assert "progress.json" in cmd and "/root/cascade/_train_work/_dispatch/r1" in cmd
    for never in ("safetensors", "checkpoint", "tar ", "scp", "rm "):
        assert never not in cmd


def test_remote_script_runs_and_splits_on_a_fake_pod_dir(tmp_path):
    """The pod-side script itself (run by local bash against a stand-in
    workdir): the newest run dir's files are tailed to the cap, one section
    each; the stream splits back into those files."""
    import subprocess

    rd = tmp_path / "_train_work" / "_dispatch" / "expert-5Hk-1-ab"
    rd.mkdir(parents=True)
    (rd / "stdout").write_bytes(b"a" * 10_000)
    (rd / "stderr").write_text("RuntimeError: NCCL timeout\n")
    (rd / "progress.json").write_text('{"step": 3}')
    (rd / "weights.safetensors").write_bytes(b"w" * 5000)       # never copied
    cmd = pm.remote_command(str(tmp_path), run_dir=None, nonce="cafe", per_file_bytes=2048)
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=60).stdout
    got = dict(pm.split_sections(out, "cafe"))
    assert len(got["expert-5Hk-1-ab.stdout"].strip()) == 2048
    assert b"NCCL timeout" in got["expert-5Hk-1-ab.stderr"]
    assert b'"step": 3' in got["expert-5Hk-1-ab.progress.json"]
    assert {"nvidia-smi.txt", "dmesg_tail.txt", "pod_facts.txt"} <= set(got)
    assert not any("safetensors" in k for k in got)


def test_size_cap_holds(tmp_path):
    big = b"y" * 5000
    seen = {}

    def ssh(argv, timeout, max_bytes):
        seen["cap"] = max_bytes
        return _fake_pod({"a.stdout": big, "b.stderr": big})(argv, timeout, max_bytes)

    d = Path(pm.capture(_host(), root=tmp_path, pod_name="p", ssh=ssh, max_bytes=3000))
    assert seen["cap"] == 3000
    total = sum(p.stat().st_size for p in d.iterdir() if p.name != "meta.json")
    assert total <= 3000


def test_stream_capped_kills_a_flooding_child():
    t0 = time.monotonic()
    out = pm.stream_capped(["yes"], timeout=10, max_bytes=100_000)
    assert len(out) == 100_000 and time.monotonic() - t0 < 10


def test_stream_capped_honours_the_deadline():
    t0 = time.monotonic()
    out = pm.stream_capped(["sleep", "30"], timeout=1.0, max_bytes=1000)
    assert out == b"" and time.monotonic() - t0 < 8


def test_slow_capture_never_blocks_teardown_past_the_timeout(tmp_path):
    release = threading.Event()

    def hung(argv, timeout, max_bytes):
        release.wait(30)
        return b"late"

    order = []
    t0 = time.monotonic()
    path = pm.capture(_host(), root=tmp_path, pod_name="p", ssh=hung, timeout=0.5)
    order.append("rm")                       # the teardown proceeds here
    took = time.monotonic() - t0
    release.set()
    assert took < 3 and order == ["rm"]
    assert json.loads((Path(path) / "meta.json").read_text())["status"] == "timeout"


def test_failed_capture_returns_and_records_it(tmp_path, monkeypatch):
    def broken(argv, timeout, max_bytes):
        raise OSError("ssh: connect refused")

    path = pm.capture(_host(), root=tmp_path, pod_name="p", ssh=broken)
    assert "failed" in json.loads((Path(path) / "meta.json").read_text())["status"]
    # the runner wrapper never raises, whatever it is handed
    monkeypatch.setenv(pm.ENV_DIR, str(tmp_path / "pm"))
    monkeypatch.setenv("CASCADE_POD_POSTMORTEM", "1")
    out = pm.before_teardown(types.SimpleNamespace(work_root=tmp_path), None, pod_name="p",
                             reason="x")
    assert json.loads((Path(out) / "meta.json").read_text())["status"] == "no_address"


def test_unwritable_root_never_raises(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    assert pm.capture(_host(), root=f / "sub", pod_name="p", ssh=_fake_pod({})) is None


def test_lazy_host_resolver_runs_inside_the_bound(tmp_path):
    def slow_lookup():
        time.sleep(5)
        return _host()

    t0 = time.monotonic()
    pm.capture(slow_lookup, root=tmp_path, pod_name="lemon", ssh=_fake_pod({}), timeout=0.5)
    assert time.monotonic() - t0 < 3
    d = pm.capture(lambda: None, root=tmp_path, pod_name="lemon2", ssh=_fake_pod({}))
    assert json.loads((Path(d) / "meta.json").read_text())["status"] == "no_address"


def test_prune_drops_postmortems_older_than_14_days(tmp_path):
    now = time.time()
    old = tmp_path / "pod-a-20251101T000000Z"
    new = tmp_path / "pod-b-20261125T000000Z"
    other = tmp_path / "not-a-postmortem"
    for d in (old, new, other):
        d.mkdir()
        (d / "meta.json").write_text("{}")
    os.utime(old, (now - 15 * 86400, now - 15 * 86400))
    os.utime(other, (now - 30 * 86400, now - 30 * 86400))
    assert pm.prune(tmp_path, now=now) == 1
    assert not old.exists() and new.exists() and other.exists()


def test_before_teardown_is_off_inside_test_runs_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setenv(pm.ENV_DIR, str(tmp_path / "pm"))
    monkeypatch.delenv("CASCADE_POD_POSTMORTEM", raising=False)
    assert pm.before_teardown(types.SimpleNamespace(work_root=tmp_path), _host(),
                              pod_name="p", reason="x") is None
    monkeypatch.setenv("CASCADE_POD_POSTMORTEM", "0")
    assert pm.before_teardown(types.SimpleNamespace(work_root=tmp_path), _host(),
                              pod_name="p", reason="x") is None


def test_root_is_beside_the_work_root(tmp_path, monkeypatch):
    monkeypatch.delenv(pm.ENV_DIR, raising=False)
    assert pm.root_for(tmp_path / "_train_work") == tmp_path / "_pod_postmortem"
    monkeypatch.setenv(pm.ENV_DIR, str(tmp_path / "x"))
    assert pm.root_for(tmp_path / "_train_work") == tmp_path / "x"


# ── capture happens BEFORE the pod is removed, and a failure does not stop it ─


def test_funded_cleanup_calls_the_hook_before_terminate(tmp_path, monkeypatch):
    """The real ``rent_funded_pod`` failure path (a not-ready lemon): the hook
    sees the pod before ``terminate_verified``; a raising hook does not stop
    the removal."""
    from cascade.provision import funded

    for raising in (False, True):
        events = []

        class Prov:
            name = "lium"

            def available(self, *a, **k):
                return True

            def list_tagged(self, name):
                return []

            def launch(self, spec):
                return [spec.name_prefix + "-0"]

            def wait_ready(self, pid, timeout):
                return False

            def get_ip(self, pid):
                return None

        def hook(provider, pid, raising=raising, events=events):
            events.append(("postmortem", pid))
            if raising:
                raise RuntimeError("boom")

        monkeypatch.setattr(funded, "terminate_verified",
                            lambda prov, pid, events=events: (events.append(("rm", pid)),
                                                              True)[1])
        monkeypatch.setattr(funded, "quarantine_lemon_host", lambda *a, **k: "")
        monkeypatch.setattr(funded, "record_not_ready_strike", lambda *a, **k: None)
        res = funded.rent_funded_pod(
            round_id="7", hotkey="5Hk", api_key="k", sku="1xH100", image="img@sha256:" + "a" * 64,
            ssh_pubkey="ssh-ed25519 AAA", netuid=91, ready_timeout=1,
            provider_factory=lambda key: Prov(), before_cleanup=hook)
        assert not res.ok
        names = [e[0] for e in events]
        assert names[:2] == ["postmortem", "rm"], events
