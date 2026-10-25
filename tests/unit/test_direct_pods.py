"""SSH pod dispatch and Hippius harvest, driven through the real pod-side CLI."""

import contextlib
import hashlib
import io
import json
import shlex
import shutil
import subprocess
import sys

import pytest

from cascade.direct import pods
from cascade.direct.compute import PodProfile
from cascade.direct.runtime import LocalTorchDispatcher
from cascade.direct.storage import HippiusObjects, PodObjects, publish
from cascade.provision.core import PodAddress
from cascade.provision.direct import ProviderTransient

IMAGE = "ghcr.io/tensorlink/cascade-worker:v2@sha256:" + "b" * 64
PIN = "ssh-ed25519 AAAApinned"


class FakeS3:
    """The boto surface HippiusObjects uses; 404 on a missing key."""

    def __init__(self):
        self.blobs = {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.blobs:
            err = Exception("missing")
            err.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise err
        return {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.blobs[(Bucket, Key)] = bytes(Body)

    def upload_file(self, path, bucket, key, **kwargs):
        with open(path, "rb") as f:
            self.blobs[(bucket, key)] = f.read()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.blobs[(Bucket, Key)])}


class Gang:
    def address(self, key):
        assert key == "rk"
        return PodAddress("10.0.0.7", 40022), PIN


def harness(tmp_path, monkeypatch, *, attest=None):
    pod = tmp_path / "pod"
    pod.mkdir()
    s3 = FakeS3()
    objects = HippiusObjects(s3, "cascade-direct", "objects")
    calls = []

    def runner(argv, timeout, stdin):
        assert "StrictHostKeyChecking=yes" in argv  # every SSH runs against the pin
        assert "root@10.0.0.7" in argv and "40022" in argv
        assert not stdin or "HIPPIUS" not in stdin  # pods never receive credentials
        args = shlex.split(argv[-1].split(" -m cascade.direct.pods ", 1)[1])
        calls.append(args[0])
        if args[0] == "attest" and attest is not None:
            return subprocess.CompletedProcess(argv, 0, json.dumps(attest), "")
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin or ""))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = pods.main(args)
        return subprocess.CompletedProcess(argv, rc, out.getvalue(), "")

    def tar_runner(direction, argv, local, shas):
        store = pod / pods.OBJECTS_DIR
        store.mkdir(exist_ok=True)
        for sha in shas:
            src, dst = (local / sha, store / sha) if direction == "push" else (store / sha, local / sha)
            shutil.copy(src, dst)
        return 0

    profile = PodProfile("H200", 2, "H200", 141, "bfloat16", 1, 3600, IMAGE, "a" * 64)
    dispatcher = pods.PodDispatcher(objects, {"vast": Gang()}, {"h200x2": profile},
                                    remote_python=sys.executable, workdir=str(pod),
                                    staging_dir=tmp_path, runner=runner, tar_runner=tar_runner)
    dispatcher.bind_allocation("job", "pod-0", "h200x2", "rk", provider="vast")
    return dispatcher, objects, s3, pod, calls


def fake_worker(monkeypatch, pod, *, corrupt=False):
    """Stand in for the supervisor: record the spec, write one checkpoint and done."""
    started = []

    def ensure_started(self, spec):
        started.append(spec)
        root = self.directory(spec["job"], spec["fence"])
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        store = PodObjects(spec["storage"]["root"])
        weights = publish(store, b"trained weights")
        if corrupt:
            (store.root / weights["sha256"]).write_bytes(b"substituted by the payer")
        manifest = {"job": spec["job"], "fence": spec["fence"], "consumed": 10,
                    "ranks": {"0": {"weights": weights}}}
        (root / "checkpoints" / "checkpoint-10.json").write_text(json.dumps({"manifest": manifest}))
        (root / "done.json").write_text(json.dumps({"returncode": 0}))

    monkeypatch.setattr(LocalTorchDispatcher, "ensure_started", ensure_started)
    return started


def test_stage_start_poll_and_harvest_into_hippius(tmp_path, monkeypatch):
    dispatcher, objects, s3, pod, calls = harness(tmp_path, monkeypatch)
    started = fake_worker(monkeypatch, pod)
    initial = publish(objects, b"promoted anchor weights")
    spec = {"job": "job", "fence": 1, "world_size": 2, "initial": {"weights": initial},
            "storage": {"kind": "hippius", "bucket": "cascade-direct", "prefix": "objects"}}
    dispatcher.ensure_started(spec)
    # The input reached the pod's content store; the worker reads it by digest.
    assert (pod / pods.OBJECTS_DIR / initial["sha256"]).read_bytes() == b"promoted anchor weights"
    assert started[0]["storage"] == {"kind": "pod", "root": str(pod / pods.OBJECTS_DIR)}
    assert PodObjects(started[0]["storage"]["root"]).get(initial["uri"]) == b"promoted anchor weights"
    dispatcher.ensure_started(spec)  # already staged: nothing pushed twice
    assert calls.count("missing") == 2
    manifests, done = dispatcher.poll("job", 1)
    assert done == {"returncode": 0}
    weights = manifests[0]["ranks"]["0"]["weights"]
    assert weights["uri"] == objects.prefix + weights["sha256"]
    assert objects.get(weights["uri"]) == b"trained weights"
    assert hashlib.sha256(b"trained weights").hexdigest() == weights["sha256"]


def test_substituted_checkpoint_object_is_refused(tmp_path, monkeypatch):
    dispatcher, objects, s3, pod, _ = harness(tmp_path, monkeypatch)
    fake_worker(monkeypatch, pod, corrupt=True)
    dispatcher.ensure_started({"job": "job", "fence": 1, "world_size": 2,
                               "storage": {"kind": "hippius"}})
    with pytest.raises(ValueError, match="digest mismatch"):
        dispatcher.poll("job", 1)
    assert not s3.blobs  # nothing unverified reached Hippius


@pytest.mark.parametrize("gpus, match", [
    ([{"name": "NVIDIA H200", "bytes": 141 * 1024**3}], "topology mismatch"),
    ([{"name": "NVIDIA GeForce RTX 4090", "bytes": 24 * 1024**3}] * 2, "differs from qualified"),
])
def test_attestation_rejects_wrong_gang(tmp_path, monkeypatch, gpus, match):
    spec = {"job": "job", "fence": 1, "world_size": 2, "architecture": {"preset": "x"}}
    from cascade.shared.direct_moe import digest

    reply = {"torch": "2.4.1", "count": 2, "gpus": gpus, "architecture": digest(spec["architecture"])}
    dispatcher, *_ = harness(tmp_path, monkeypatch, attest=reply)
    with pytest.raises(ValueError, match=match):
        dispatcher.attest(spec)


def test_attestation_accepts_the_qualified_gang(tmp_path, monkeypatch):
    spec = {"job": "job", "fence": 1, "world_size": 2, "architecture": {"preset": "x"}}
    from cascade.shared.direct_moe import digest

    reply = {"torch": "2.4.1", "count": 2, "architecture": digest(spec["architecture"]),
             "gpus": [{"name": "NVIDIA H200", "bytes": 141 * 1024**3}] * 2}
    dispatcher, *_ = harness(tmp_path, monkeypatch, attest=reply)
    assert dispatcher.attest(spec) == [0, 1]


def test_unreachable_pod_is_an_infrastructure_retry(tmp_path, monkeypatch):
    dispatcher, *_ = harness(tmp_path, monkeypatch)

    def down(argv, timeout, stdin):
        raise OSError("connection refused")

    dispatcher.run = down
    with pytest.raises(ProviderTransient):  # OSError: the runtime retries, never a science fault
        dispatcher.poll("job", 1)
    dispatcher.stop("job", 1)  # best-effort; the provider terminates the pod


def test_pod_store_resolves_any_scheme_by_digest(tmp_path):
    store = PodObjects(tmp_path)
    ref = publish(store, b"blob")
    assert store.get("hippius://bucket/objects/" + ref["sha256"]) == b"blob"
    with pytest.raises(ValueError, match="sha256"):
        store.get("hippius://bucket/objects/not-a-digest")
