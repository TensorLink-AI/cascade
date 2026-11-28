"""Lane D full cycle on fake infrastructure (W-E2E).

One cycle, every stage real code except the external services:

1. ``submit``   the miner CLI (``cascade lane-d verify`` + ``submit``) against
                the REAL intake (:class:`cascade.lane_d.service.LaneDIntake`)
                served on loopback HTTP; the CLI's multipart PUTs land in the
                real filesystem vault through a loopback grant minter;
2. ``intake``   ``cascade lane-d ready --attest`` (the commit goes to a fake
                chain), the chain reveals, the intake's refresh tick moves the
                entry to ``revealed`` in the queue file the validator reads;
3. ``pool_build`` the pool_d builder's per-build release sync (#994 R13:
                ``elastic.builder_run_steps`` run for real against a local git
                release with an ssh stub) re-syncs a changed release;
4. ``pool_pin`` each snapshot is published (``pool_d_publish.publish_snapshot``
                into a filesystem bucket, the stale-data guard of #976/#997)
                and pinned box-side (``sync_pins`` -> ``pins.json``), and the
                eval host syncs its own copy from the same bucket. The cycle
                checks on its own that every pinned snapshot's data advanced;
5. ``settle``   the validator's production collaborators from an operator
                inputs TOML (``load_inputs_services``: ``PinFilePool``,
                ``SSHEvalHost``, ...) and the production step
                (``build_lane_d_step``) settle two boundaries inside
                :func:`~tests.e2e_lanes.harness.box_guard` and
                :func:`~tests.e2e_lanes.harness.box_budget`, with a tripwire on
                any pinned pool_d file read in the box process. The eval
                host's SSH is a fake ``run`` that executes the eval host's own
                ``python -m cascade.eval.lane_d_forecasts`` ``main`` in a
                forked child, its stdout prefixed with download chatter;
6. ``publish``  the signed report, receipt and ``status/lane_d.json``;
7. ``verify``   a second validator (the in-process reference path, same
                queue and pins) derives the same report body byte for byte,
                and the outside validators' audit (``check_lane_d``, chain
                mode, the pinned snapshots) passes it;
8. ``miner_status`` ``cascade lane-d status`` (the intake queue) and the
                ``cascade status`` snapshot (``live_status.status_snapshot`` of
                the published documents) show the miner its entry and crown.

The judged snapshots are W7a's synthetic chain (``tests/accept/w7a``), made
``pool_d/1``-valid; the eval host's forecaster is W8c's table (no torch). The
real ``build_pool_d`` needs the ``pool-forge`` extra (pandas, parquet), which
the torch-free tier does not install; its own tests are tests/accept/w6b.

Nothing here imports cascade at module level."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.accept.w3 import helpers as w3
from tests.accept.w7a import helpers as w7a
from tests.accept.w8b import helpers as w8b
from tests.accept.w8c import helpers as w8c
from tests.accept.w10 import helpers as w10
from tests.accept.w_offbox import helpers as w_offbox

from .harness import BlockClock, Cycle, box_budget, box_guard

B1, B2 = w8b.B1, w8b.B2
CONTEXT = w7a.CONTEXT
STAGES = ("submit", "intake", "pool_build", "pool_pin", "settle", "publish", "verify",
          "miner_status")
# The box's heap budget for one settlement step (pools never load here). The
# box path peaks ~0.8 MiB locally, ~2.3 MiB on GitHub runners (first-use
# allocations); the reference in-process path over the same pins is > 24 MiB.
BOX_STEP_BUDGET_BYTES = 8 * 1024 * 1024
FIXTURE_HISTORY = 16384
# What real eval hosts print on stdout before their one JSON line (hf_hub /
# pip download progress; #983 was lane B's copy of this).
CHATTER = (b"Downloading model.safetensors: 100%|##########| 12.0M/12.0M\n"
           b"Fetching 4 files: 100%|##########| 4/4 [00:00<00:00]\n")
MINER = "e2e-lane-d-miner"


class BoxPoolRead(AssertionError):
    """The box process read a pinned pool_d file (that is the eval host's job)."""


class CycleFailed(AssertionError):
    """A stage of the cycle did not complete."""


# ── the intake over loopback HTTP, with the vault's multipart PUT surface ─────


def _minter_cls():
    from cascade.lane_d import service

    class LoopbackMinter(service.GrantMinter):
        """Presigned multipart grants on the loopback server; ``finalize``
        assembles the parts into the real filesystem vault (as Hippius'
        CompleteMultipartUpload does)."""

        upload_path = service.UPLOAD_PATH
        part_size = 4096

        def __init__(self, storage):
            self.storage = storage
            self.base = ""
            self.parts: dict[tuple[str, int], bytes] = {}

        def mint(self, prefix, files):
            out = {}
            for name, size in sorted(files.items()):
                reqs = [{"method": "PUT", "url": f"{self.base}/vault/{prefix}{name}?part="
                                                 f"{i // self.part_size + 1}",
                         "offset": i, "length": min(self.part_size, int(size) - i),
                         "headers": {}}
                        for i in range(0, int(size), self.part_size)]
                out[name] = {"requests": reqs}
            return {"files": out}

        def finalize(self, prefix, files):
            for name, size in files.items():
                key = prefix + name
                parts = sorted((p, b) for (k, p), b in self.parts.items() if k == key)
                blob = b"".join(b for _, b in parts)
                if len(blob) != int(size):
                    raise service.IntakeError("upload_incomplete", f"{name}: {len(blob)} of "
                                                                   f"{size} bytes")
                self.storage.put(key, blob)

    return LoopbackMinter


class IntakeServer:
    """The real intake WSGI app (plus the minter's PUT route) on 127.0.0.1."""

    def __init__(self, app, minter):
        self.app, self.minter = app, minter
        self._srv = None
        self._thread = None

    def _wsgi(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if environ["REQUEST_METHOD"] == "PUT" and path.startswith("/vault/"):
            n = int(environ.get("CONTENT_LENGTH") or 0)
            body = environ["wsgi.input"].read(n)
            part = int(environ.get("QUERY_STRING", "part=1").split("=", 1)[1])
            self.minter.parts[(path[len("/vault/"):], part)] = body
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b""]
        return self.app(environ, start_response)

    def __enter__(self):
        from wsgiref.simple_server import WSGIRequestHandler, make_server

        class Quiet(WSGIRequestHandler):
            def log_message(self, *a):
                return None

        self._srv = make_server("127.0.0.1", 0, self._wsgi, handler_class=Quiet)
        self.minter.base = f"http://127.0.0.1:{self._srv.server_port}"
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return self.minter.base

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(5)


# ── the eval host behind the real SSHEvalHost ────────────────────────────────


def pod_hello() -> dict:
    """W8c's code identity plus a scoring environment the release's
    ``deploy/laned/scoring_envs.toml`` qualifies (B7: the validator refuses
    an eval host that reports none)."""
    import hashlib

    from cascade.shared.config.lane_d import load_scoring_envs

    env = next(e for e in load_scoring_envs(None) if e.status == "qualified")
    return {**w8c.hello(), "env": env.name, "gpu_model": env.gpu_models[0],
            "driver": env.driver_min, "torch_build": env.torch_build,
            "fingerprint": hashlib.sha256(f"e2e-pod-{env.name}".encode()).hexdigest()}


class FakePodSSH:
    """``SSHEvalHost(run=...)``: the ``ssh`` argv's remote command is the eval
    host's ``python -m cascade.eval.lane_d_forecasts <args>``; it runs that
    module's real ``main`` (forecaster = W8c's table, host module check off:
    the pod has its GPU stack) in a forked child, stdin = the request, and
    answers its stdout after :data:`CHATTER`."""

    def __init__(self, forecaster, *, chatter: bytes = CHATTER):
        self.forecaster = forecaster
        self.chatter = chatter
        self.calls: list[list[str]] = []

    def _args(self, argv) -> list[str]:
        remote = shlex.split(argv[-1].split("&&")[-1])
        i = remote.index("cascade.eval.lane_d_forecasts")
        return remote[i + 1:]

    def _serve(self, args: list[str], payload: bytes) -> bytes:
        from cascade.eval import lane_d_forecasts, lane_d_settle

        if args[0] == "hello":
            return json.dumps(pod_hello(), sort_keys=True).encode() + b"\n"
        forecaster = self.forecaster
        out = io.StringIO()
        stdin = io.TextIOWrapper(io.BytesIO(payload))
        saved = (lane_d_settle.LocalForecaster, lane_d_forecasts.check_host_modules, os.sys.stdin)
        try:
            lane_d_settle.LocalForecaster = lambda **kw: forecaster
            lane_d_forecasts.check_host_modules = lambda *a, **k: None
            os.sys.stdin = stdin
            with contextlib.redirect_stdout(out):
                rc = lane_d_forecasts.main(args)
        finally:
            (lane_d_settle.LocalForecaster, lane_d_forecasts.check_host_modules,
             os.sys.stdin) = saved
        if rc not in (0, None):
            raise RuntimeError(f"eval host exited {rc}")
        return out.getvalue().encode()

    def __call__(self, argv, *, input=None, capture_output=True, timeout=None, check=False):
        args = self._args(argv)
        self.calls.append(args)
        if args[0] == "hello":
            body = self._serve(args, b"")
        else:
            body = w_offbox.in_child(self._serve, args, bytes(input or b""))
        return subprocess.CompletedProcess(argv, 0, stdout=self.chatter + body, stderr=b"")


# ── pool_d snapshots ─────────────────────────────────────────────────────────


def make_pool_d1(snap_dir: Path) -> Path:
    """W7a's synthetic snapshot as a ``pool_d/1`` build (the fields the
    builder writes that W7a's lighter fixture omits)."""
    from cascade.pool.pool_d_publish import max_first_seen_ts

    p = Path(snap_dir)
    meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    with np.load(p / "hist.npz") as hist:
        for s in meta["series"]:
            s["history_len"] = int(hist[s["series_id"]].shape[-1])
    meta["as_of_ts"] = int(max_first_seen_ts(meta)) + 3600
    meta["history_length"] = int(w7a.HISTORY)
    (p / "metadata.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return p


def pinned_first_seen(pins_path: Path) -> list[tuple[str, int]]:
    """``[(snapshot, newest first-seen point)]`` of every pin, in pin order."""
    from cascade.pool.pool_d_publish import max_first_seen_ts

    doc = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    out = []
    for pin in doc["pins"]:
        meta = json.loads((Path(pins_path).parent / pin["dir"] / "metadata.json")
                          .read_text(encoding="utf-8"))
        out.append((pin["snapshot"], max_first_seen_ts(meta)))
    return out


def assert_pins_advance(pins_path: Path) -> None:
    """The cycle's own check (independent of the publish guard): each pinned
    snapshot carries newer live data than the one before it."""
    rows = pinned_first_seen(pins_path)
    for (a, ta), (b, tb) in zip(rows, rows[1:], strict=False):
        assert tb is not None and ta is not None and tb > ta, (
            f"pinned pool_d snapshot {b} carries no newer data than {a} (newest first-seen "
            f"{tb} <= {ta}): a frozen builder mirror reached the validator")


# ── the builder's per-build release sync (R13) ───────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "e2e", "GIT_AUTHOR_EMAIL": "e2e@x",
                        "GIT_COMMITTER_NAME": "e2e", "GIT_COMMITTER_EMAIL": "e2e@x"})


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def builder_release_check(tmp: Path) -> list[str]:
    """Two builds of the pool_d builder against a local release: the first
    syncs it (no ``.release_sha`` yet), a release commit in between must make
    the second re-sync. Returns the bootstrap log (one line per sync)."""
    from tests.accept.r13 import helpers as r13

    m = r13.mod()
    release = tmp / "release"
    (release / "deploy").mkdir(parents=True)
    _exe(release / "deploy" / "pod_bootstrap.sh",
         '#!/bin/bash\necho "$CASCADE_SRC_ROOT $POD_WORKDIR" >> "$BOOT_LOG"\n')
    (release / "code.py").write_text("v1\n")
    _git(release, "init", "-q")
    _git(release, "add", "-A")
    _git(release, "commit", "-q", "-m", "v1")
    chain = tmp / "chain.toml"
    chain.write_text("[lane_d]\n")
    workdir = tmp / "builder"
    workdir.mkdir()
    fake = tmp / "bin"
    fake.mkdir()
    _exe(fake / "ssh", '#!/bin/bash\nexec bash -c "${@: -1}"\n')
    cfg = r13.config(tmp)
    cfg = replace(cfg, builder=replace(cfg.builder, release_dir=str(release),
                                       chain_toml=str(chain), workdir=str(workdir)))
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "BOOT_LOG": str(tmp / "boot.log")}
    inst = {"id": "sf-e2e", "ip": "203.0.113.5", "port": 22, "user": "shadeform"}

    def build():
        steps = m.builder_run_steps(cfg, inst, now_ts=r13.T0)
        names = [n for n, _ in steps]
        assert names and names[0] == "release_sync", (
            f"a pool_d build does not start with the release sync (steps {names}): the "
            "builder keeps whatever code it bootstrapped with")
        subprocess.run(dict(steps)["release_sync"], check=True, env=env, capture_output=True)

    build()
    (release / "code.py").write_text("v2\n")
    _git(release, "commit", "-q", "-am", "v2")
    build()
    log = tmp / "boot.log"
    boots = log.read_text().splitlines() if log.is_file() else []
    assert len(boots) == 2, (f"the release changed between two builds but the builder synced "
                             f"{len(boots)} time(s): it builds from stale code")
    return boots


# ── the world ────────────────────────────────────────────────────────────────


@dataclass
class World:
    tmp: Path
    monkeypatch: object
    capsys: object
    caplog: object
    fx: dict = field(default_factory=dict)
    cycle: Cycle = field(default_factory=lambda: Cycle("d"))
    clock: BlockClock = field(default_factory=lambda: BlockClock(w7a.READY_BLOCK,
                                                                 w7a.SETTLEMENT))
    ns: SimpleNamespace = field(default_factory=SimpleNamespace)


# Modules a settlement imports on first use: imported once up front so the
# box budget measures the settlement, not first-import allocations.
WARM_MODULES = (
    "cascade.validator.lane_d", "cascade.validator.lane_d_wiring",
    "cascade.validator.lane_d_inputs", "cascade.validator.lane_d_state",
    "cascade.validator.lane_d_status", "cascade.validator.lane_d_house",
    "cascade.validator.offthread_worker", "cascade.eval.lane_d_scorer",
    "cascade.eval.lane_d_handshake", "cascade.eval.lane_d_settle",
    "cascade.eval.lane_d_forecasts", "cascade.eval.bootstrap", "cascade.shared.sentinel",
    "cascade.shared.sentinel_board", "cascade.shared.config.lane_d", "cascade.audit.lane_d",
    "cascade.shared.sr25519", "numpy.ma", "boto3", "botocore.config", "botocore.client",
)


def warm_box_imports() -> None:
    import importlib

    for name in WARM_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:         # an optional extra (the vault's boto3) not installed
            continue
    w8b.signer().sign(b"warm")


def make_world(tmp: Path, monkeypatch, capsys, caplog) -> World:
    from cascade.lane_d.service import LaneDIntake
    from cascade.lane_d.storage import FilesystemStorage
    from cascade.shared.config.lane_d_ops import lane_d_ops_config

    w = World(Path(tmp), monkeypatch, capsys, caplog)
    # 16384-point histories (W7a's default is 256): an in-process pool load is
    # then several MiB, far over BOX_STEP_BUDGET_BYTES; the box path is not.
    monkeypatch.setattr(w7a, "HISTORY", FIXTURE_HISTORY)
    w.fx = w8b.fixture(w.tmp / "fx")
    warm_box_imports()
    w.ns.cfg = w8b.chain_cfg(w.tmp / "v")
    w.ns.key = w10.fixture_key(MINER)
    w.ns.hotkey = w.ns.key.ss58_address
    w.ns.coldkey = w7a.coldkey()
    w.ns.chain = w10.install_chain(monkeypatch, w10.FakeChain(w.ns.key, block=w.clock.block))
    w.ns.reads = w3.make_reads(block=w.clock.block, coldkeys={w.ns.hotkey: w.ns.coldkey})
    w.ns.vault = FilesystemStorage(w.tmp / "vault")
    w.ns.minter = _minter_cls()(w.ns.vault)
    w.ns.intake_dir = w.tmp / "intake"
    w.ns.intake = LaneDIntake(storage=w.ns.vault, reads=w.ns.reads, minter=w.ns.minter,
                              cfg=w.ns.cfg.lane_d, ops=lane_d_ops_config(w.ns.cfg),
                              state_dir=w.ns.intake_dir, now=time.time,
                              per_ip_rpm=10_000, per_hotkey_rpm=10_000)
    w10.cli_env(monkeypatch, w.tmp)
    return w


def _cli(w: World, *argv: str):
    res = w10.run_cli(w.capsys, w.caplog, *argv)
    if res.rc != 0:
        raise CycleFailed(f"`cascade {' '.join(argv[:2])}` exited {res.rc}: {res!r}")
    return res


def _chain_args() -> list[str]:
    return ["--wallet-name", "e2e", "--wallet-hotkey", "lane-d", "--chain-toml",
            str(w10.CHAIN_TOML)]


# ── stages ───────────────────────────────────────────────────────────────────


def stage_submit(w: World) -> None:
    from cascade.miner.lane_d import repo_code

    entry = w.tmp / "entry"
    entry.mkdir()
    files = {**repo_code(),
             "config.json": json.dumps(w3.entry_config(), sort_keys=True).encode(),
             "weights.safetensors": w10.safetensors_bytes(w10.TENSORS, fill=5)}
    for name, blob in files.items():
        (entry / name).write_bytes(blob)
    # torch-free tier: the allowlist + code checks run, the config / weights
    # checks need torch and say so (``torch_unavailable``), never a crash
    res = w10.run_cli(w.capsys, w.caplog, "lane-d", "verify", str(entry), "--chain-toml",
                      str(w10.CHAIN_TOML))
    if res.rc != 0 and res.code != "torch_unavailable":
        raise CycleFailed(f"`cascade lane-d verify` refused the entry: {res!r}")
    with IntakeServer(w.ns.intake, w.ns.minter) as srv:
        res = _cli(w, "lane-d", "submit", srv.url, str(entry), "--context", str(CONTEXT),
                   *_chain_args())
    w.ns.digest = res.data["digest"]
    assert res.data["uploaded_bytes"] == sum(len(b) for b in files.values())
    w.cycle.ran("submit", w.ns.digest)


def stage_intake(w: World) -> None:
    from cascade.lane_d.service import QUEUE_FILENAME

    with IntakeServer(w.ns.intake, w.ns.minter) as srv:
        res = _cli(w, "lane-d", "ready", srv.url, "--digest", w.ns.digest, "--context",
                   str(CONTEXT), "--licence", "MIT", "--attest", *_chain_args())
    assert w.ns.chain.commits and w.ns.chain.commits[-1][0] == res.data["commit_payload"], \
        "the CLI did not commit the intake's payload on chain"
    # the chain reveals the commitment; the intake's refresh tick sees it
    w.clock.block = w7a.REVEAL_BLOCK + 1
    w.ns.reads.block = w.clock.block
    w.ns.reads.reveals[(w.ns.hotkey, w.ns.digest)] = w7a.REVEAL_BLOCK
    w.ns.intake.tick()
    rows = json.loads((w.ns.intake_dir / QUEUE_FILENAME).read_text())["entries"]
    row = next(r for r in rows if r["digest"] == w.ns.digest)
    if row["state"] != "revealed" or row["reveal_block"] != w7a.REVEAL_BLOCK:
        raise CycleFailed(f"the intake did not reveal the entry: {row}")
    w.cycle.ran("intake", row["state"])


def stage_pool_build(w: World) -> None:
    boots = builder_release_check(w.tmp / "builder_host")
    w.cycle.ran("pool_build", f"{len(boots)} release syncs")


def frozen_snapshot(w: World) -> Path:
    """A second build from a frozen forge mirror (2026-11-27): snapshot 1's
    data re-cut 3 h later under a new name."""
    src = make_pool_d1(Path(w.fx["chain"]["pools"][1]))
    dst = w.tmp / "frozen" / "w7a-1-frozen"
    dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
    meta.update(snapshot=dst.name, as_of_ts=int(meta["as_of_ts"]) + 3 * 3600)
    (dst / "metadata.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return dst


def stage_pool_pin(w: World, *, snapshots=None) -> None:
    from cascade.pool import pool_d_publish as pub

    if snapshots is None:
        snapshots = [make_pool_d1(Path(w.fx["chain"]["pools"][s])) for s in (1, 2)]

    bucket = pub.FsPoolStore(w.tmp / "pool_bucket")
    w.ns.pins_path = w.tmp / "box_pins" / "pins.json"
    w.ns.pod_pool_root = w.tmp / "pod_pool"
    for src, from_block in zip(snapshots, (B1, B2), strict=False):
        snap = Path(src).name
        try:
            pub.publish_snapshot(src, bucket, prefix=pub.DEFAULT_PREFIX)
            pub.sync_pins(bucket, prefix=pub.DEFAULT_PREFIX, local_root=w.ns.pins_path.parent,
                          pins_path=w.ns.pins_path, from_block=from_block)
            pub.sync_pins(bucket, prefix=pub.DEFAULT_PREFIX, local_root=w.ns.pod_pool_root,
                          pins_path=w.ns.pod_pool_root / "pins.json", from_block=from_block)
        except pub.PublishError as e:
            raise CycleFailed(f"pool_pin refused snapshot {snap}: {e.code}: {e}") from e
    assert_pins_advance(w.ns.pins_path)
    w.cycle.ran("pool_pin", ", ".join(s for s, _ in pinned_first_seen(w.ns.pins_path)))


def _alias_entry(w: World) -> None:
    """The fixture's forecasts for challenger X serve the miner's real
    checkpoint digest (the eval host's table forecaster), and the metagraph
    holds the miner's hotkey at X's uid."""
    fx, dg = w.fx, w.ns.digest
    for snap in fx["chain"]["forecasts"]:
        fx["chain"]["forecasts"][snap][dg] = fx["chain"]["forecasts"][snap][fx["x"]]
    fx["entry"] = {**fx["entry"], "hotkey": w.ns.hotkey, "digest": dg}
    fx["x"] = dg


def _inputs_toml(w: World, d: Path, rig) -> Path:
    hosts = d / "eval_hosts.toml"
    hosts.write_text('[[host]]\nname = "lane-d-eval"\nhost = "203.0.113.9"\nport = 22\n'
                     'user = "root"\nremote_python = "/root/cascade/.venv/bin/python"\n'
                     'workdir = "/root/cascade"\nstage = "any"\n', encoding="utf-8")
    rd, t7 = d / "readiness.json", d / "t7.json"
    rd.write_bytes(rig.readiness.data)
    t7.write_bytes(rig.t7.data)
    p = d / "lane_d_inputs.toml"
    lines = {"intake_state_dir": str(w.ns.intake_dir), "pool_pins": str(w.ns.pins_path),
             "house_pointer": str(rig.house_path),
             "house_signer_ss58": w8c.house_key().ss58_address, "eval_hosts": str(hosts),
             "eval_cache_dir": "/root/lane_d_cache", "readiness": str(rd), "t7": str(t7),
             "eval_pool_root": str(w.ns.pod_pool_root)}
    p.write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in lines.items())
                 + "eval_cache_budget_bytes = 200000000000\n", encoding="utf-8")
    return p


def validator(w: World, name: str, *, offloaded: bool = True, ssh_run=None):
    """One validator: a W8c rig's fakes (chain reader, house pointer, bench
    pod), the PRODUCTION collaborators from an operator inputs TOML, and the
    production step. ``offloaded=False`` is the in-process reference path."""
    from cascade.validator.lane_d_inputs import build_lane_d_inputs, load_inputs_services
    from cascade.validator.lane_d_wiring import build_lane_d_step

    d = w.tmp / name
    rig = w8c.Rig(d, fx=w.fx)
    rig.reader.block_hash = lambda b: w8b.block_hash(int(b))
    rig.eval_host.hello_doc = pod_hello()
    services = load_inputs_services(rig.cfg, _inputs_toml(w, d, rig))
    services["bench_host"] = rig.bench_host
    host = services["eval_host"]
    host._run = ssh_run or FakePodSSH(rig.eval_host)
    if not offloaded:
        services["settle_on_host"] = False
    if not services["settle_on_host"]:
        # the pre-W-OFFBOX split: pools in this process, per-digest forecasts
        # from the eval host (the pod's own forecaster answers them)
        services["eval_host"] = rig.eval_host
    source = build_lane_d_inputs(rig.cfg, state_dir=rig.state_dir, chain_factory=rig.factory,
                                 expected_identity=w8c.identity(), **services)
    store = w8b.Store()
    step = build_lane_d_step(rig.cfg, state_dir=rig.state_dir, signer=w8b.signer(),
                             chain_factory=w8b.ReaderFactory(), store=store, inputs=source,
                             bench=source.bench)
    return SimpleNamespace(rig=rig, step=step, store=store, source=source, host=host,
                           pool_reads=[])


@contextlib.contextmanager
def box_pool_tripwire(roots: list[Path], hits: list[str]):
    """Raise :class:`BoxPoolRead` on any ``numpy.load`` of a file under
    ``roots`` (the box's pinned pool_d copies) in THIS process."""
    real = np.load
    roots = [Path(r).resolve() for r in roots]

    def guarded(file, *a, **k):
        if isinstance(file, str | os.PathLike):
            p = Path(file).resolve()
            if any(p.is_relative_to(r) for r in roots):
                hits.append(str(p))
                raise BoxPoolRead(f"the validator process loaded pinned pool_d file {p}")
        return real(file, *a, **k)

    w_mp = pytest.MonkeyPatch()
    w_mp.setattr(np, "load", guarded)
    try:
        yield hits
    finally:
        w_mp.undo()


def run_settlements(w: World, v, boundaries=(B1, B2), *, budget: int = BOX_STEP_BUDGET_BYTES,
                    tripwire: bool = True) -> None:
    from cascade.validator.lane_d_wiring import report_key

    for b in boundaries:
        w.clock.block = b
        with contextlib.ExitStack() as st:
            st.enter_context(box_guard())
            if tripwire:
                st.enter_context(box_pool_tripwire([w.ns.pins_path.parent], v.pool_reads))
            st.enter_context(box_budget(f"lane D settlement {b}", budget))
            try:
                v.step(b)
            except AssertionError:
                raise
            except Exception as e:  # noqa: BLE001 — every stage failure is the cycle's assertion
                raise CycleFailed(f"settle {b}: {type(e).__name__}: {e}") from e
        if report_key(b) not in v.store.objects:
            raise CycleFailed(f"settlement {b} published no lane D report")


def stage_settle(w: World) -> None:
    _alias_entry(w)
    w.ns.v = validator(w, "validator")
    run_settlements(w, w.ns.v)
    w.cycle.ran("settle", f"boundaries {B1}, {B2}")


def stage_publish(w: World) -> None:
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY
    from cascade.validator.lane_d_wiring import receipt_key, report_key

    objs = w.ns.v.store.objects
    rep = json.loads(objs[report_key(B1)])
    if rep.get("crowned") != w.ns.digest:
        raise CycleFailed(f"settlement {B1} did not crown the miner's entry: "
                          f"crowned={rep.get('crowned')!r} skipped={rep.get('skipped')}")
    later = json.loads(objs[report_key(B2)])
    assert (later.get("king_after") or {}).get("digest") == w.ns.digest, \
        f"the crowned entry did not keep KING_D at {B2}: {later.get('king_after')}"
    for b in (B1, B2):
        assert receipt_key(b) in objs, f"no lane D receipt was published for {b}"
    assert LANE_D_STATUS_KEY in objs, "no status/lane_d.json was published"
    w.ns.report = rep
    w.cycle.ran("publish", f"crowned {w.ns.digest[:12]}")


def _body(report: dict) -> bytes:
    return json.dumps({k: v for k, v in report.items() if k != "signature"}, sort_keys=True,
                      separators=(",", ":")).encode()


def stage_verify(w: World) -> None:
    from cascade.audit.lane_d import check_lane_d
    from cascade.validator.lane_d_wiring import report_key

    ref = validator(w, "reference", offloaded=False)
    run_settlements(w, ref, budget=1 << 40, tripwire=False)
    for b in (B1, B2):
        mine = json.loads(w.ns.v.store.objects[report_key(b)])
        theirs = json.loads(ref.store.objects[report_key(b)])
        assert _body(mine) == _body(theirs), (
            f"settlement {b}: a second validator re-derives a different report body")
    prior = [json.loads(w.ns.v.store.objects[report_key(B1)])]
    pins = json.loads(w.ns.pins_path.read_text())["pins"]
    res = check_lane_d(w.ns.report, w.ns.v.rig.cfg.lane_d,
                       signer_hotkey=w8b.signer().ss58_address, prior_reports=prior,
                       pool_dirs=[w.ns.pins_path.parent / p["dir"] for p in pins])
    if res.status != "PASS":
        raise CycleFailed(f"the outside validators' audit refuses the report: {res}")
    w.ns.ref = ref
    w.cycle.ran("verify", res.detail)


def stage_miner_status(w: World) -> None:
    from datetime import UTC, datetime

    from cascade.miner.live_status import status_snapshot
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

    with IntakeServer(w.ns.intake, w.ns.minter) as srv:
        res = _cli(w, "lane-d", "status", srv.url, "--hotkey", w.ns.hotkey)
    entries = res.data.get("entries") or []
    assert [e.get("digest") for e in entries] == [w.ns.digest], \
        f"`cascade lane-d status` does not show the miner's entry: {entries}"
    doc = json.loads(w.ns.v.store.objects[LANE_D_STATUS_KEY])
    snap = status_snapshot({"chain": {"current_block": int(doc["block"]),
                                      "era_blocks": w7a.ERA},
                            "lane_d_queue": doc},
                           now=datetime.now(UTC), max_age=180.0)
    lane = snap["lanes"]["d"]
    king = (lane or {}).get("king") or {}
    if king.get("digest") != w.ns.digest or king.get("hotkey") != w.ns.hotkey:
        raise CycleFailed(f"`cascade status` does not show the miner's crown: lanes.d={lane}")
    assert snap["freshness"]["lane_d_queue"]["fresh"], snap["freshness"]["lane_d_queue"]
    w.cycle.ran("miner_status", entries[0].get("state", "?"))


def run_cycle(w: World) -> Cycle:
    """Every stage in order, torch-free throughout (the whole cycle runs
    inside :func:`box_guard`); returns the cycle record."""
    with box_guard():
        for stage in (stage_submit, stage_intake, stage_pool_build, stage_pool_pin,
                      stage_settle, stage_publish, stage_verify, stage_miner_status):
            stage(w)
    w.cycle.assert_full(STAGES)
    return w.cycle

