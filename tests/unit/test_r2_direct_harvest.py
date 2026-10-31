"""[round] funded_harvest_transport = "r2": pod → R2 → Hippius, never the box disk.

Everything here runs against in-memory fakes: a fake S3/R2 client, a fake
ssh runner that plays the pod (``tail | head | curl`` → the fake's presigned
links), and fake Hippius pushes. No network, no real ssh.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import re
import struct
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

import cascade.provision.funded as funded_mod
import cascade.trainer.r2_transfer as r2
from cascade.shared.config import RoundConfig
from cascade.trainer import runner_common
from cascade.trainer.loop import (
    TrainerRunner,
    _FundedDiverged,
    _FundedLegSkip,
    _FundedTamper,
)
from cascade.trainer.remote import LocalTrainReceipt, RemoteHost
from tests.unit.test_funded_pod_wiring import (
    _challenger,
    _pod,
    _rent_ok,
    _runner,
    _vault,
)

# ── fakes ────────────────────────────────────────────────────────────────────


class FakeS3:
    """Just enough of boto3's S3 client for the r2 transport."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict] = {}       # upload_id → {key, parts: {n: bytes}}
        self.links: dict[str, tuple] = {}        # presigned url → (method, key, upload_id, n)
        self.aborted: list[str] = []
        self.completed: list[str] = []
        self.deleted: list[str] = []
        self.calls: list[str] = []
        self._n = 0

    def generate_presigned_url(self, method, Params, ExpiresIn):  # noqa: N803 — boto3 names
        self._n += 1
        url = f"https://r2.example/{Params['Key']}?sig={self._n}"
        self.links[url] = (method, Params["Key"], Params.get("UploadId"), Params.get("PartNumber"))
        return url

    def create_multipart_upload(self, Bucket, Key):  # noqa: N803
        uid = f"up{len(self.uploads)}"
        self.uploads[uid] = {"key": Key, "parts": {}}
        return {"UploadId": uid}

    def put_link(self, url: str, data: bytes) -> str:
        method, key, uid, n = self.links[url]
        etag = hashlib.md5(data).hexdigest()  # noqa: S324 — S3's ETag shape
        if method == "put_object":
            self.objects[key] = data
        else:
            self.uploads[uid]["parts"][int(n)] = data
        return etag

    def list_parts(self, Bucket, Key, UploadId, PartNumberMarker=0):  # noqa: N803
        parts = self.uploads[UploadId]["parts"]
        return {"Parts": [{"PartNumber": n, "Size": len(b),
                           "ETag": '"' + hashlib.md5(b).hexdigest() + '"'}  # noqa: S324
                          for n, b in sorted(parts.items())], "IsTruncated": False}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):  # noqa: N803
        parts = self.uploads.pop(UploadId)["parts"]
        self.objects[Key] = b"".join(parts[p["PartNumber"]] for p in MultipartUpload["Parts"])
        self.completed.append(Key)

    def abort_multipart_upload(self, Bucket, Key, UploadId):  # noqa: N803
        self.uploads.pop(UploadId, None)
        self.aborted.append(Key)

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        self.calls.append(f"get {Key} {Range or ''}")
        data = self.objects[Key]
        if Range:
            a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            data = data[a:b + 1]
        return {"Body": io.BytesIO(data)}

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        self.deleted += [o["Key"] for o in Delete["Objects"]]
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)


class FakeStore:
    def __init__(self, bucket="ckpt-bucket"):
        self.cfg = SimpleNamespace(bucket=bucket)
        self.s3 = FakeS3()

    def client(self):
        return self.s3


_TAIL = re.compile(r"tail -c \+(\d+) -- (\S+) \| head -c (\d+)")


class FakePod:
    """Plays the pod behind ``run_ssh``: lists files, streams byte ranges into
    the fake R2 links and answers with curl's ``-D -`` header dump."""

    def __init__(self, files: dict[str, bytes], store: FakeStore, events=None, *, corrupt=None):
        self.files = files
        self.store = store
        self.events = events if events is not None else []
        self.corrupt = corrupt or {}          # name → bytes swapped in at upload
        self.commands: list[str] = []
        self.stdins: list[str] = []

    def __call__(self, argv, timeout, stdin_text=None):
        cmd = argv[-1]
        self.commands.append(cmd)
        self.stdins.append(stdin_text or "")
        if "find . -maxdepth 1" in cmd:
            out = "".join(f"{n}\t{len(b)}\n" for n, b in self.files.items())
            return CompletedProcess(argv, 0, out, "")
        m = _TAIL.search(cmd)
        if m:
            off, name, n = int(m.group(1)) - 1, m.group(2), int(m.group(3))
            data = self.corrupt.get(name, self.files[name])[off:off + n]
            etag = self.store.s3.put_link(stdin_text.strip(), data)
            self.events.append(f"put {name}")
            return CompletedProcess(argv, 0, f'HTTP/1.1 200 OK\r\nETag: "{etag}"\r\n\r\n', "")
        raise AssertionError(f"unexpected pod command {cmd!r}")


def _safetensors(n_floats: int) -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [n_floats],
                               "data_offsets": [0, 4 * n_floats]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x80\x3f" * n_floats


def _ckpt_files() -> dict[str, bytes]:
    return {"weights.safetensors": _safetensors(64),
            "optimizer.safetensors": _safetensors(96),
            "config.json": b'{"toto2": {}}',
            "model.py": b"# model\n",
            ".train_complete": b"",
            "notes.bin": b"x" * 10}           # outside the push allow-list: stays on the pod


def _digests(files):
    return {n: hashlib.sha256(b).hexdigest() for n, b in files.items()
            if n.endswith(".safetensors")}


# ── 1. plan / complete ───────────────────────────────────────────────────────


def test_part_layout_covers_the_file_exactly():
    assert r2.part_layout(0, 10) == [(1, 0, 0)]
    assert r2.part_layout(10, 10) == [(1, 0, 10)]
    assert r2.part_layout(25, 10) == [(1, 0, 10), (2, 10, 10), (3, 20, 5)]


def test_multipart_plan_presigns_every_part_and_completes_after_list_parts():
    store = FakeStore()
    plan = r2.plan_file_upload(store, "w.safetensors", "ckpt/1/hk/j/w.safetensors", 25,
                               part_bytes=10)
    assert plan.upload_id and [p.number for p in plan.parts] == [1, 2, 3]
    assert {store.s3.links[p.url][0] for p in plan.parts} == {"upload_part"}
    data = bytes(range(25))
    etags = {p.number: store.s3.put_link(p.url, data[p.offset:p.offset + p.size])
             for p in plan.parts}
    r2.complete_file_upload(store, plan, etags)
    assert store.s3.objects[plan.key] == data and store.s3.completed == [plan.key]
    assert r2.stream_sha256(store, plan.key) == (hashlib.sha256(data).hexdigest(), 25)


def test_single_part_file_uses_one_presigned_put():
    store = FakeStore()
    plan = r2.plan_file_upload(store, "config.json", "k/config.json", 5, part_bytes=10)
    assert plan.upload_id == "" and store.s3.links[plan.parts[0].url][0] == "put_object"
    store.s3.put_link(plan.parts[0].url, b"12345")
    r2.complete_file_upload(store, plan, {})


@pytest.mark.parametrize("damage", ["missing_part", "short_part"])
def test_complete_rejects_wrong_part_count_or_size_and_aborts(damage):
    store = FakeStore()
    plan = r2.plan_file_upload(store, "w.safetensors", "k/w", 25, part_bytes=10)
    data = bytes(25)
    for p in plan.parts:
        if damage == "missing_part" and p.number == 2:
            continue
        chunk = data[p.offset:p.offset + p.size]
        if damage == "short_part" and p.number == 1:
            chunk = chunk[:-1]
        store.s3.put_link(p.url, chunk)
    with pytest.raises(r2.R2TransferError, match="upload incomplete"):
        r2.complete_file_upload(store, plan, {})
    assert store.s3.aborted == ["k/w"] and "k/w" not in store.s3.objects


def test_complete_rejects_an_etag_the_put_never_returned():
    store = FakeStore()
    plan = r2.plan_file_upload(store, "w", "k/w", 20, part_bytes=10)
    for p in plan.parts:
        store.s3.put_link(p.url, bytes(10))
    with pytest.raises(r2.R2TransferError, match="ETag"):
        r2.complete_file_upload(store, plan, {1: "deadbeef"})
    assert store.s3.aborted == ["k/w"]


def test_part_command_reads_the_link_from_stdin_never_argv():
    host = RemoteHost(name="p", host="h", workdir="/root/cascade")
    part = r2.PartPlan(2, 512, 100, "https://secret-link")
    cmd = r2.part_upload_command(host, "_train_work/ck", "weights.safetensors", part)
    assert "secret-link" not in cmd and "read -r url" in cmd
    assert "tail -c +513 -- weights.safetensors | head -c 100" in cmd
    assert "Content-Length: 100" in cmd
    with pytest.raises(ValueError):
        r2.part_upload_command(host, "ck", "../etc/passwd", part)


def test_part_upload_retries_then_returns_the_etag():
    host = RemoteHost(name="p", host="h")
    calls = []

    def flaky(argv, timeout, stdin_text=None):
        calls.append(stdin_text)
        if len(calls) == 1:
            return CompletedProcess(argv, 56, "", "curl: (56) reset")
        return CompletedProcess(argv, 0, 'HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\n'
                                         'ETag: "abc"\r\n', "")

    tag = r2.upload_part_via_ssh(host, "ck", "w", r2.PartPlan(1, 0, 5, "U"),
                                 runner=flaky, sleep=lambda s: None)
    assert tag == "abc" and calls == ["U\n", "U\n"]


# ── the funded leg in r2 mode ────────────────────────────────────────────────


class _Disp:
    def __init__(self, digests):
        self.digests = digests

    def dispatch(self, host, **kw):
        return LocalTrainReceipt(
            miner_hotkey=kw["hotkey"], miner_uid=kw["uid"], role=kw["role"],
            gen_ref=kw["gen_ref"], corpus_digest="cd", train_block=kw["block"],
            checkpoint_dir="_train_work/777/toto2-4m/challenger/checkpoint",
            gpu_name="RTX4090", size=kw["arch_preset"], tensor_digests=dict(self.digests))


def _r2_runner(tmp_path, monkeypatch, *, files=None, corrupt=None, relay=None, **round_kw):
    work = tmp_path / "work"
    work.mkdir()
    round_kw.setdefault("ckpt_relay_host", "relay" if relay is not None else "")
    runner = _runner(work, funded_pod_checkpoint="harvest", funded_harvest_transport="r2",
                     **round_kw)
    _vault(work, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    from cascade.trainer.remote import PodHygiene
    monkeypatch.setattr("cascade.trainer.remote.probe_pod_hygiene",
                        lambda host, ckpt_dir, **kw: PodHygiene())

    def _no_ssh_harvest(*a, **k):
        raise AssertionError("r2 mode must never tar the checkpoint onto the box")

    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", _no_ssh_harvest)
    events: list[str] = []
    runner._teardown_funded_pod = lambda pod: events.append(f"teardown {pod.instance_id}")
    runner._funded_pod_identity_mismatch = lambda pod: None
    runner._hf_ckpt_repo = lambda repo: None
    runner._upload_retry_sleep = lambda s: None
    store = FakeStore()
    runner._r2_store = store
    files = files or _ckpt_files()
    runner._r2_ssh_runner = FakePod(files, store, events, corrupt=corrupt)
    if relay is not None:
        runner.remote_hosts = [*runner.remote_hosts, relay]
    verified = []
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint",
                        lambda d, contract=None, **kw: verified.append((Path(d), contract)))
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite",
                        lambda d: events.append("finite"))
    pushed = []

    def _push(d, repo, hub, hf_repo=None):
        d = Path(d)
        events.append("hippius push")
        pushed.append({p.name: p.read_bytes() for p in d.iterdir()})
        return SimpleNamespace(ref=SimpleNamespace(
            immutable_ref=f"{repo}@sha256:" + "d" * 64), size_bytes=1)

    monkeypatch.setattr(r2, "upload_dir_to_hub_or_hf", _push)
    seeds = SimpleNamespace(base_seed=777)
    contract = SimpleNamespace(arch_preset="toto2-4m")
    return SimpleNamespace(runner=runner, store=store, files=files, events=events,
                           verified=verified, pushed=pushed, seeds=seeds,
                           contract=contract, work=work)


def _leg(t, digests=None):
    return t.runner._run_funded_leg(_Disp(_digests(t.files) if digests is None else digests),
                                    _challenger("hkA"), t.seeds, 100, t.contract, "",
                                    warm_start_ref=None)


def _checkpoint_files_under(root: Path) -> list[Path]:
    names = set(_ckpt_files())
    return [p for p in root.rglob("*") if p.is_file() and p.name in names]


def test_r2_leg_uploads_verifies_tears_down_then_pushes(tmp_path, monkeypatch):
    t = _r2_runner(tmp_path, monkeypatch)
    entry = _leg(t)
    from cascade.shared.manifest import parse_trained_pointer
    assert parse_trained_pointer(entry.trained_pointer) is not None
    # pod torn down BEFORE any Hippius push, exactly once
    teardown = "teardown cascade-n91-777-funded-hka-0"
    assert t.events.count(teardown) == 1
    assert t.events.index(teardown) < t.events.index("hippius push")
    assert max(i for i, e in enumerate(t.events) if e.startswith("put ")) < t.events.index(teardown)
    # what reached Hippius is byte-for-byte the pod's allow-listed files
    assert t.pushed == [{n: b for n, b in t.files.items()
                         if n not in (".train_complete", "notes.bin")}]
    # the contract guard ran on a skeleton OUTSIDE work_root; the push re-ran
    # the (contract-free) code check where the full bytes landed
    (skel, contract), (pushed_dir, none) = t.verified
    assert t.work not in skel.parents and contract is t.contract
    assert none is None and t.work / "_r2_push" in pushed_dir.parents
    # nothing of the checkpoint ever landed under work_root, and R2 is cleaned
    assert _checkpoint_files_under(t.work) == []
    assert t.store.s3.objects == {}
    keys = sorted(t.store.s3.deleted)
    assert all(k.startswith("ckpt/777/hkA/") for k in keys) and len(keys) == 4


def test_r2_digest_mismatch_is_tamper_and_never_pushes(tmp_path, monkeypatch):
    files = _ckpt_files()
    swapped = dict(files)
    swapped["weights.safetensors"] = _safetensors(64)[:-4] + b"\x00\x00\xc0\x7f"   # a NaN planted
    t = _r2_runner(tmp_path, monkeypatch, files=files,
                   corrupt={"weights.safetensors": swapped["weights.safetensors"]})
    with pytest.raises(_FundedTamper, match="weights.safetensors: sha256"):
        _leg(t)
    msg, miner_fault, cls, burn = t.runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (True, "tamper", False)
    assert "hippius push" not in t.events
    assert t.events.count("teardown cascade-n91-777-funded-hka-0") == 1
    assert _checkpoint_files_under(t.work) == []


def test_r2_planted_tensor_file_is_tamper(tmp_path, monkeypatch):
    files = _ckpt_files()
    t = _r2_runner(tmp_path, monkeypatch, files=files)
    digests = _digests(files)
    digests.pop("optimizer.safetensors")
    with pytest.raises(_FundedTamper, match="never wrote"):
        _leg(t, digests)


def test_r2_requeues_the_push_without_disk_or_relay_and_retry_uses_r2_keys(
        tmp_path, monkeypatch):
    # floor 0 (the rent's disk guard is off) and fewer bytes free than the
    # checkpoint: only the local push can lack room
    t = _r2_runner(tmp_path, monkeypatch, min_free_disk_gb=0.0)
    low = SimpleNamespace(total=1 << 40, used=0, free=100)
    monkeypatch.setattr(runner_common, "DISK_USAGE", lambda p: low)
    from cascade.trainer.faults import _OperatorFault
    with pytest.raises(_OperatorFault, match="kept on R2"):
        _leg(t)
    assert "hippius push" not in t.events
    assert t.events.count("teardown cascade-n91-777-funded-hka-0") == 1   # never blocked
    _msg, miner_fault, cls, burn = t.runner._funded_leg_failures["hkA"]
    assert (miner_fault, burn) == (False, False)
    rec = runner_common.read_pending_upload(t.work, 777, "hkA")
    assert rec is not None and "dest" not in rec
    assert sorted(rec["r2"]["files"]) == ["config.json", "model.py", "optimizer.safetensors",
                                          "weights.safetensors"]
    assert _checkpoint_files_under(t.work) == []
    # disk is back: the requeued leg pushes FROM R2 — no rent, no pod, no retrain
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1 << 50, used=0, free=1 << 50))
    pod_cmds = len(t.runner._r2_ssh_runner.commands)
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("rented again")))
    entry = _leg(t)
    assert "cascade/ckpt-r777-challenger-toto2-4m@sha256:" in entry.trained_pointer
    assert len(t.runner._r2_ssh_runner.commands) == pod_cmds        # pod never touched
    assert runner_common.read_pending_upload(t.work, 777, "hkA") is None
    # the local fallback downloaded into work_root/_r2_push and removed it
    assert _checkpoint_files_under(t.work) == []


def test_r2_push_through_the_relay_host(tmp_path, monkeypatch):
    relay = RemoteHost(name="relay", host="10.1.1.1", workdir="/srv/cascade",
                       remote_python="/srv/cascade/.venv/bin/python",
                       forward_env=("HIPPIUS_HUB_TOKEN",))
    t = _r2_runner(tmp_path, monkeypatch, relay=relay)
    monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "hub-token-value")
    pod = t.runner._r2_ssh_runner
    plans = []

    def relay_ssh(argv, timeout, stdin_text=None):
        if "r2_transfer relay" not in argv[-1]:
            return pod(argv, timeout, stdin_text)
        assert argv[-2] == "root@10.1.1.1"
        plan = json.loads(stdin_text)
        plans.append(plan)
        t.events.append("hippius push")

        def fetch(name, meta, dest):
            method, key, *_ = t.store.s3.links[meta["url"]]
            assert method == "get_object"
            r2._copy_hashed(io.BytesIO(t.store.s3.objects[key]), dest, name,
                            meta["size"], meta["sha256"])

        res = r2.relay_push({k: v for k, v in plan.items() if k != "env"},
                            workdir=tmp_path / "relay-disk", fetch=fetch)
        return CompletedProcess(argv, 0, "log line\n" + r2.RESULT_MARK + json.dumps(res) + "\n", "")

    t.runner._r2_ssh_runner = relay_ssh
    entry = _leg(t)
    assert "cascade/ckpt-r777-challenger-toto2-4m@sha256:" in entry.trained_pointer
    (plan,), = [plans]
    assert plan["env"] == {"HIPPIUS_HUB_TOKEN": "hub-token-value"}    # operator VM only
    assert plan["repo"] == "cascade/ckpt-r777-challenger-toto2-4m"
    assert "BACKUP_S3" not in json.dumps(plan)
    # the POD never saw a credential or a download link: only part links
    for text in pod.stdins:
        assert "hub-token-value" not in text
        method = t.store.s3.links[text.strip()][0] if text.strip() else "list"
        assert method in ("upload_part", "put_object", "list")
    teardown = "teardown cascade-n91-777-funded-hka-0"
    assert t.events.index(teardown) < t.events.index("hippius push")
    assert _checkpoint_files_under(t.work) == []
    assert not list((tmp_path / "relay-disk").iterdir())               # relay cleaned up


def test_r2_relay_diverged_verdict_is_the_miners(tmp_path, monkeypatch):
    t = _r2_runner(tmp_path, monkeypatch)
    from cascade.eval.checkpoint_guard import CheckpointDiverged

    def nan(d):
        raise CheckpointDiverged("weights.safetensors: 1 tensor(s) hold non-finite values")

    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite", nan)
    with pytest.raises(_FundedDiverged):
        _leg(t)
    _msg, miner_fault, cls, _burn = t.runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls) == (True, "generator")
    assert _checkpoint_files_under(t.work) == []


def test_ssh_transport_is_still_the_default(tmp_path):
    runner = _runner(tmp_path, funded_pod_checkpoint="harvest")
    assert runner.cfg.round.funded_harvest_transport == "ssh"
    assert not runner_common.r2_transport_on(runner.cfg)


# ── skeleton ─────────────────────────────────────────────────────────────────


def test_skeleton_keeps_headers_and_lengths_but_not_the_bytes(tmp_path):
    from cascade.eval.checkpoint_guard import safetensors_header

    store = FakeStore()
    files = _ckpt_files()
    meta = {}
    for name, data in files.items():
        if not r2.checkpoint_file_allowed(name):
            continue
        store.s3.objects[f"p/{name}"] = data
        meta[name] = {"key": f"p/{name}", "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}
    ck = r2.R2Checkpoint("b", "p/", meta)
    d = r2.materialize_skeleton(store, ck, tmp_path / "skel")
    orig = tmp_path / "orig.safetensors"
    orig.write_bytes(files["weights.safetensors"])
    skel = d / "weights.safetensors"
    assert skel.stat().st_size == orig.stat().st_size
    assert safetensors_header(skel) == safetensors_header(orig)
    assert skel.read_bytes() != orig.read_bytes()             # tensor body is a hole
    assert (d / "config.json").read_bytes() == files["config.json"]
    # only header ranges of the tensor files were read, never their bodies
    tensor_reads = [c for c in store.s3.calls if ".safetensors" in c]
    assert tensor_reads and all("bytes=" in c for c in tensor_reads)


# ── disk guard ───────────────────────────────────────────────────────────────


def test_disk_guard_requeues_a_new_rent_unburned(tmp_path, monkeypatch):
    runner = _runner(tmp_path, min_free_disk_gb=30.0)
    _vault(tmp_path, "hkA")
    rented = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: rented.append(kw) or _rent_ok())
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1 << 40, used=0, free=10 << 30))
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert rented == []
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)
    assert "min_free_disk_gb" in msg


def test_disk_guard_never_stops_an_adopted_live_leg(tmp_path, monkeypatch):
    runner = _runner(tmp_path, min_free_disk_gb=30.0)
    _vault(tmp_path, "hkA")
    runner._ledger_add(_pod("hkA"))                  # this leg's allocation is live
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: _rent_ok())
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1 << 40, used=0, free=1 << 20))
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert host.host == "10.9.9.9"


def test_disk_guard_off_at_zero(tmp_path, monkeypatch):
    runner = _runner(tmp_path, min_free_disk_gb=0.0)
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1, used=1, free=0))
    assert runner_common.disk_guard_refusal(runner) == ""


# ── ENOSPC-safe teardown ─────────────────────────────────────────────────────


class _Lium:
    def __init__(self, pod_id):
        self.live = [pod_id]
        self.terminated = []

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        self.live.remove(pod_id)

    def list_tagged(self, prefix):
        return [p for p in self.live if p.startswith(prefix)]


def _enospc(*a, **k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_teardown_terminates_even_when_every_disk_write_fails(tmp_path, monkeypatch):
    from cascade.funding.providers import PayerCredential, lease_credentials
    from cascade.funding.vault import PayerKeyVault

    vault = PayerKeyVault(dir=tmp_path / "pv")
    vault.insert("hkA", "sk-miner")
    token = lease_credentials(vault, "hkA", PayerCredential("lium", "sk-miner"))
    pod = replace(_pod("hkA"), payer_credential_id=token, pod_uid="uid-1")
    lium = _Lium(pod.instance_id)
    # the disk is full: every vault write (lease TTL re-stamp, lease removal) fails
    monkeypatch.setattr(PayerKeyVault, "_insert_locked", _enospc)
    monkeypatch.setattr(PayerKeyVault, "remove", _enospc)
    monkeypatch.setattr("cascade.funding.providers.resolve_create", _enospc)
    left = funded_mod.teardown_funded([pod], vault, provider_factory=lambda key: lium)
    assert lium.terminated == [pod.instance_id] and left == []


def test_runner_teardown_survives_enospc_on_the_ledger(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    vault = _vault(tmp_path, "hkA")
    runner._payer_vault = lambda: vault
    pod = _pod("hkA")
    runner._ledger_add(pod)
    gone = []
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, v, **kw: gone.extend(p.instance_id for p in pods) or [])
    runner._ledger_remove = _enospc
    runner._teardown_funded_pod(pod)                   # must not raise
    assert gone == [pod.instance_id]


# ── config ───────────────────────────────────────────────────────────────────


def _toml_with_round(extra: str, *, no_backup: bool = False) -> str:
    from cascade.shared.config.loader import DEFAULT_CHAIN_TOML

    text = DEFAULT_CHAIN_TOML.read_text()
    text = text.replace("\n[round]\n", f"\n[round]\n{extra}\n", 1)
    if no_backup:
        text = re.sub(r"(?m)^backup_s3_endpoint\s*=.*$", 'backup_s3_endpoint = ""', text)
    return text


def test_transport_knobs_round_trip_through_the_loader(tmp_path):
    from cascade.shared.config import load_chain_config

    p = tmp_path / "chain.toml"
    p.write_text(_toml_with_round(
        'funded_harvest_transport = "r2"\nckpt_relay_host = "relay-1"\n'
        'min_free_disk_gb = 42.5\nckpt_r2_bucket = "cascade-ckpt"\nckpt_r2_prefix = "harvest"'))
    rnd = load_chain_config(p).round
    assert (rnd.funded_harvest_transport, rnd.ckpt_relay_host, rnd.min_free_disk_gb,
            rnd.ckpt_r2_bucket, rnd.ckpt_r2_prefix) == ("r2", "relay-1", 42.5,
                                                         "cascade-ckpt", "harvest/")


def test_transport_defaults_ship_dark(tmp_path):
    from cascade.shared.config import load_chain_config
    from cascade.shared.config.loader import DEFAULT_CHAIN_TOML

    rnd = load_chain_config(DEFAULT_CHAIN_TOML).round
    assert (rnd.funded_harvest_transport, rnd.ckpt_relay_host, rnd.min_free_disk_gb,
            rnd.ckpt_r2_bucket, rnd.ckpt_r2_prefix) == ("ssh", "", 30.0, "", "ckpt/")
    assert RoundConfig().funded_harvest_transport == "ssh"


@pytest.mark.parametrize("extra,match", [
    ('funded_harvest_transport = "s3"', "funded_harvest_transport"),
    ('funded_harvest_transport = "r2"', "backup_s3_endpoint"),       # endpoint blanked below
    ("min_free_disk_gb = -1", "min_free_disk_gb"),
    ('ckpt_r2_prefix = "../x/"', "ckpt_r2_prefix"),
    ('ckpt_r2_bucket = "Bad_Bucket"', "ckpt_r2_bucket"),
    ("ckpt_relay_host = 3", "ckpt_relay_host"),
])
def test_transport_knobs_fail_loud(tmp_path, extra, match):
    from cascade.shared.config import load_chain_config

    p = tmp_path / "chain.toml"
    p.write_text(_toml_with_round(extra, no_backup=match == "backup_s3_endpoint"))
    with pytest.raises(ValueError, match=match):
        load_chain_config(p)


def test_ckpt_store_reuses_the_backup_endpoint_with_its_own_bucket():
    storage = SimpleNamespace(backup_s3_endpoint="https://acct.r2.cloudflarestorage.com",
                              backup_s3_region="", backup_bucket="cascade-backup",
                              manifest_bucket="cascade-manifests")
    cfg = SimpleNamespace(storage=storage, round=RoundConfig())
    store = r2.ckpt_store(cfg)
    assert store.cfg.bucket == "cascade-backup"
    assert store.cfg.access_key_env == "BACKUP_S3_ACCESS_KEY"
    cfg2 = SimpleNamespace(storage=storage, round=RoundConfig(ckpt_r2_bucket="cascade-ckpt"))
    assert r2.ckpt_store(cfg2).cfg.bucket == "cascade-ckpt"
    with pytest.raises(r2.R2StorageError):
        r2.ckpt_store(SimpleNamespace(storage=SimpleNamespace(), round=RoundConfig()))


def test_runner_binds_the_new_seams():
    # the r2 glue only touches runner attributes TrainerRunner really has
    for name in ("_harvest_funded_checkpoint", "hub", "_hf_ckpt_repo"):
        assert hasattr(TrainerRunner, name)
