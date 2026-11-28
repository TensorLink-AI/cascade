"""Lane C (MarketCast) full cycle on fake infrastructure (W-E2E).

One :class:`LaneCWorld` holds the shared fakes and runs the REAL code of
every stage:

========================  ===============================================
stage                     real code
========================  ===============================================
``submit``                miner ``recipe_error`` + ``upload_dir_to_hub`` +
                          ``format_finetune_commit`` (commitment on the
                          fake chain)
``intake``                ``FundingIntake.lane_c_fund`` over the REAL HTTP
                          server (``make_server``): signature, reveal,
                          GPU fit (#989) and sample geometry (#990) checks
``queue``                 ``LaneCQueue`` (the file the trainer shares)
``dispatch``              ``LaneCDriver.tick`` over ``RunnerLaneCOps``
                          (recipe fetch from the Hub, ``read_mkt_recipe``,
                          embargo ``scan_receipt``)
``train``                 ``train_mkt_leg`` (contract, scopes, rent SKUs,
                          verdict pickup); the pod is :class:`FakeLegPod`:
                          its failures go through the REAL
                          ``classify_leg_exception`` and
                          ``_record_funded_failure``
``harvest``               ``verify_harvested_mkt_checkpoint`` +
                          ``upload_dir_to_hub`` (in-memory Hub)
``settle_c``              ``LaneCDriver.settle`` (signed lane C manifest,
                          write-ahead) bound into a lane A
                          ``TrainingManifest`` (``lane_c_digest``)
``judge``                 main validator ``_lane_c_judge_round`` →
                          ``_mkt_settle`` → ``_mkt_publish_round`` with the
                          judge on a forked ``mkt_worker.Worker`` "pod"
                          (``MktSession`` + ``PodJudge``); the pinned base
                          resolved ON THE BOX (``resolve_pinned_base``),
                          fine-tunes fetched through ``fetch_from_hub``
``report``                the signed MarketCast report + sidecars
``record``                the trainer's next settlement records the
                          report's sha256 (``_mkt_arbiters``)
``verify``                a second (non-main) validator applies the
                          recorded report (``_mkt_read_recorded``,
                          signature + sha256) and must reach the same
                          judgement; an independent validator recomputes
                          the report on its own pod: byte-identical
``miner_status``          ``cascade lane-c status`` (the real CLI over the
                          real intake HTTP) shows the verdict / reason
========================  ===============================================

Fakes: chain (commitments, block hash), the Hub registry (an in-memory
``hippius_hub`` that, like hippius_hub >= 0.6, writes chunked-v2 objects
unless ``HIPPIUS_CHUNKED_WRITE=0``), the HF base snapshot, sr25519 (a
deterministic sha512 signer), the Lium pod (training is a numpy stand-in)
and the MarketCast pod's model loader (a torch-free forecaster). Nothing
imports torch or timestral: the box stages run under
:func:`~tests.e2e_lanes.harness.box_guard`."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import struct
import sys
import threading
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.e2e_lanes.harness import BlockClock, Cycle, box_budget, box_guard

REPO = Path(__file__).resolve().parents[2]
TARGET = "mkt:returns"
HK = "5E2eLaneCMinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAA"
COLDKEY = "5E2eLaneCMinerColdkeyAAAAAAAAAAAAAAAAAAAAAAAAA"
UID = 7
MAIN_VALIDATOR = "5E2eMainValidatorHotkeyAAAAAAAAAAAAAAAAAAAAAAA"
TRAINER = "5E2eTrainerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
GPU = "1xA100"
BASE_REPO = "e2e-org/timestral-28m"
BASE_REV = "0123456789abcdef0123456789abcdef01234567"
RECIPE_REPO = "e2e/mkt-recipe"
# the lane A era king the rolling scheduler hands the lane C driver
KING_REF = "cascade/ckpt-e2e-king@sha256:" + hashlib.sha256(b"e2e-king").hexdigest()
LANE_C_STAGES = ("submit", "intake", "queue", "dispatch", "train", "harvest", "settle_c",
                 "judge", "report", "record", "verify", "miner_status")
# a box stage's heap budget: the pod judge stays far under it, an in-process
# judge of the cycle's geometry (W-MKTOFF's pre-fix path) far over it
BOX_JUDGE_BUDGET = 6 << 20
EVAL_CONTEXT = 64
SNAPSHOT_MINUTES = 3800
# the 2026-11-28 10:54 geometry, scaled: contexts long enough that an
# in-process judge's covariate rows dwarf the pod judge's box footprint
WIDE_CONTEXT, WIDE_MINUTES = 512, 6000


# ── tiny checkpoints (hand-built safetensors: no library) ────────────────────


def safetensors_bytes(values) -> bytes:
    data = np.asarray(values, dtype="<f4").tobytes()
    header = json.dumps({"w": {"dtype": "F32", "shape": [len(values)],
                               "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


def first_value(weights: Path) -> float:
    raw = Path(weights).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    return float(np.frombuffer(raw[8 + n:8 + n + 4], dtype="<f4")[0])


BASE_WEIGHTS = safetensors_bytes([3.0] + [0.5] * 63)
BASE_CONFIG = b'{"arch": "timestral-28m-e2e"}'


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── the in-memory Hub (an hippius_hub stand-in) ──────────────────────────────

POINTER_MT = "application/vnd.hippius.pointer.v2"
PACK_MT = "application/vnd.hippius.pack.v1"


class MemHub:
    """``upload_folder`` / ``snapshot_download`` over a dict. Like hippius_hub
    >= 0.6.0 it writes a ``*.safetensors`` file as a chunked-v2 object (a
    pointer layer + packs) unless ``HIPPIUS_CHUNKED_WRITE=0``; like the
    pinned 0.5.1 its ``snapshot_download`` saves the pointer under the file's
    title. :meth:`registry_reader` is what ``cascade.shared.hippius``
    reassembles chunked objects through."""

    def __init__(self):
        self.objects: dict[str, dict] = {}          # manifest digest -> object
        self.pushes: list[tuple[str, str, bool]] = []

    def upload_folder(self, *, repo_id, folder_path, allow_patterns=None, ignore_patterns=None,
                      token=None, **_):
        import os

        chunked = os.environ.get("HIPPIUS_CHUNKED_WRITE", "1") != "0"
        files = {p.name: p.read_bytes() for p in sorted(Path(folder_path).iterdir())
                 if p.is_file() and not p.name.startswith(".")}
        titled, packs, layers = {}, {}, []
        for name, data in files.items():
            if chunked and name.endswith(".safetensors") and len(data) > 64:
                pointer, ps = _chunk(data)
                packs.update(ps)
                titled[name] = pointer
                layers.append({"mediaType": POINTER_MT, "digest": "sha256:" + _sha(pointer),
                               "size": len(pointer),
                               "annotations": {"org.opencontainers.image.title": name,
                                               "com.hippius.file.size": str(len(data)),
                                               "com.hippius.file.digest": "sha256:" + _sha(data),
                                               "com.hippius.chunk.count": str(len(json.loads(
                                                   pointer)["chunks"]))}})
            else:
                titled[name] = data
                layers.append({"mediaType": "application/octet-stream", "size": len(data),
                               "digest": "sha256:" + _sha(data),
                               "annotations": {"org.opencontainers.image.title": name}})
        layers += [{"mediaType": PACK_MT, "size": len(b), "digest": d} for d, b in packs.items()]
        manifest = json.dumps({"schemaVersion": 2,
                               "mediaType": "application/vnd.oci.image.manifest.v1+json",
                               "annotations": ({"com.hippius.layout": "chunked-v2"}
                                               if packs else {}),
                               "layers": layers}, sort_keys=True).encode()
        digest = "sha256:" + _sha(manifest)
        self.objects[digest] = {"repo": repo_id, "files": titled, "packs": packs,
                                "manifest": manifest}
        self.pushes.append((str(repo_id), digest, bool(packs)))
        return SimpleNamespace(oid=digest)

    def snapshot_download(self, *, repo_id, revision, local_dir, **_):
        obj = self.objects.get(str(revision))
        if obj is None or obj["repo"] != str(repo_id):
            raise RuntimeError(f"404 manifest unknown: {repo_id}@{revision}")
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, data in obj["files"].items():
            (d / name).write_bytes(data)
        return str(d)

    def registry_reader(self, ref):
        obj = self.objects[str(ref.digest)]

        class _Reader:
            def manifest_bytes(self, digest):
                return obj["manifest"]

            def blob_to(self, digest, fh):
                fh.write(obj["packs"][digest])

        return _Reader()

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("hippius_hub")
        mod.upload_folder = self.upload_folder
        mod.snapshot_download = self.snapshot_download
        return mod


def _chunk(data: bytes):
    cuts = sorted({0, len(data) // 3, 2 * len(data) // 3, len(data)})
    pieces = [data[a:b] for a, b in itertools.pairwise(cuts)]
    blob = b"".join(pieces)
    pd = "sha256:" + _sha(blob)
    rows, off = [], 0
    for p in pieces:
        rows.append({"digest": "sha256:" + _sha(p), "offset": off, "pack": pd, "size": len(p)})
        off += len(p)
    pointer = json.dumps({"chunks": rows, "file": {"digest": "sha256:" + _sha(data),
                                                   "size": len(data)},
                          "version": "chunked-v2"}, separators=(",", ":"),
                         sort_keys=True).encode()
    return pointer, {pd: blob}


# ── the fake chain + signatures ──────────────────────────────────────────────


class DetKey:
    """A deterministic stand-in for an sr25519 hotkey (sha512 of address +
    body); :func:`det_verify` is its verifier."""

    def __init__(self, ss58: str):
        self.ss58_address = ss58

    def sign(self, body: bytes) -> bytes:
        return hashlib.sha512(self.ss58_address.encode() + bytes(body)).digest()


def det_verify(ss58: str, message, signature_hex: str) -> bool:
    msg = message.encode() if isinstance(message, str) else bytes(message)
    return hashlib.sha512(ss58.encode() + msg).hexdigest() == str(signature_hex)


class FakeChain:
    def __init__(self, clock: BlockClock):
        self.clock = clock
        self.commits: list = []

    def commit(self, hotkey: str, payload: str, uid: int, coldkey: str) -> int:
        from cascade.shared.chain import Commitment

        block = self.clock.block
        self.commits.append(Commitment(uid=uid, hotkey=hotkey, coldkey=coldkey,
                                       payload=payload, commit_block=block))
        return block

    def reveal_block(self, hotkey, target, ref):
        from cascade.shared.lane_c_manifest import parse_finetune_commit

        for c in reversed(self.commits):
            if c.hotkey == hotkey:
                got = parse_finetune_commit(c.payload, mkt_suites=("returns",))
                return int(c.commit_block) if got == (target, ref) else None
        return None


class Store:
    """An S3 manifest bucket (text + bytes), with a write log."""

    def __init__(self):
        self.objs: dict[str, object] = {}
        self.puts: list[str] = []

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, str) else v.decode()

    def put_text(self, key, text, content_type="", acl=None):
        self.objs[key] = text
        self.puts.append(key)

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objs[key] = bytes(data)
        self.puts.append(key)


# ── the pods ─────────────────────────────────────────────────────────────────


def local_run(argv, timeout):
    """The MarketCast session's ssh / scp, run on this machine (the "pod"
    is a forked child sharing the filesystem): ``scp -r <srcs> host:<dest>``
    with ``scp`` semantics (one directory onto a new path = that path; else
    every source into the existing directory)."""
    import subprocess

    if argv[0] == "scp":
        rest, srcs, i = argv[1:-1], [], 0
        while i < len(rest):
            if rest[i] in ("-P", "-o", "-i"):
                i += 2
                continue
            if not rest[i].startswith("-"):
                srcs.append(rest[i])
            i += 1
        dest = Path(argv[-1].split(":", 1)[1])
        if len(srcs) == 1 and Path(srcs[0]).is_dir() and not dest.exists():
            shutil.copytree(srcs[0], dest)
        else:
            for src in map(Path, srcs):
                if src.is_dir():
                    shutil.copytree(src, dest / src.name)
                else:
                    shutil.copy2(src, dest / src.name)
        return subprocess.CompletedProcess(argv, 0, "", "")
    return subprocess.run(["bash", "-c", argv[-1]], capture_output=True, text=True,
                          timeout=timeout)


def e2e_loader(d: Path, device: str):
    """The MarketCast pod's model loader (``mkt_worker.LOADERS``) without
    timestral: a deterministic forecaster whose spread is the checkpoint's
    first weight (so the fine-tune's trained bytes decide its scores)."""
    from tests.accept.w_mktoff.helpers import Fc

    return Fc(first_value(Path(d) / "model.safetensors"))


OOM_STDERR = (REPO / "tests" / "accept" / "w_cmp" / "fixtures" / "mkt_oom_rtx4090.stderr")
NAN_STDERR = ("marketcast leg: non-finite training loss at step 1050\n"
              "miner submission rejected\n")


class FakeLegPod:
    """The miner's Lium pod running a MarketCast leg. ``script`` is one
    outcome per rent: ``"ok"`` trains (the fine-tune's first weight is
    1.0), ``"oom"`` / ``"nan"`` exit 1 with the real worker stderr. Every
    rent first does the leg's pre-rent work (``prep``: the corpus relay,
    ~1 h of CPU in production) and asks the REAL attach-only rent verdict
    (``leg_reattach.rent_refusal``) — what a re-attach with no ledgered pod
    refuses after the prep (#995)."""

    def __init__(self, world: LaneCWorld, script=("ok",)):
        self.w = world
        self.script = list(script)
        self.prep = 0
        self.rents: list[str] = []
        self.ledger: list = []                  # the funded-pod ledger rows

    def run_funded_leg(self, disp, gen, seeds, block, contract, suffix, **kw):
        from cascade.trainer import faults
        from cascade.trainer.leg_reattach import rent_refusal
        from cascade.trainer.remote import RemoteDispatchError, final_exception_line
        from cascade.trainer.runner_funded_legs import _leg_allocated

        r = self.w.runner
        assert kw.get("role") == "mkt_finetune" and kw.get("finetune_recipe")
        self.prep += 1
        refused = rent_refusal(prior_alloc=_leg_allocated(r, str(seeds.base_seed), gen.hotkey),
                               pod_live=None)
        if refused:
            raise RuntimeError(refused)
        skus = r.__dict__["_leg_local_obj"].mkt_rent_skus       # the MarketCast rent scope
        choice = kw.get("gpu_choice")
        sku = next((s for s in skus if choice is not None and s.lower() == choice.gpu.lower()),
                   skus[0])
        self.rents.append(sku)
        outcome = self.script.pop(0) if self.script else "ok"
        if outcome in ("oom", "nan"):
            # the 2026-11-27 22:10 OOM (rc 1) / the 2026-11-28 00:45 flat-context
            # NaN as the mkt-rc2 worker reported it (rc 3, "rejected")
            stderr, rc = (OOM_STDERR.read_text(), 1) if outcome == "oom" else (NAN_STDERR, 3)
            err = RemoteDispatchError(f"remote mkt_finetune on funded-x failed (rc={rc}) "
                                      f"{stderr[-300:]}",
                                      final_exception=final_exception_line(stderr), returncode=rc)
            err.role = "mkt_finetune"
            # what runner_funded_legs does with a worker exit
            q = r._funded_queue()
            prior = q.get(gen.hotkey) if q is not None else None
            miner_fault, error_class, burn = faults.classify_leg_exception(
                r, gen, seeds, err, rc=rc, text=str(err),
                stalled_before=bool(prior is not None and prior.last_error_class == "stall"),
                shards_leg=False, role="mkt_finetune")
            text = str(err)
            if error_class == faults.GPU_OOM_CLASS:
                text = f"{faults.gpu_oom_message(sku)}: {text}"
            if error_class == "infra":
                from cascade.funding.queue import OPERATOR_FAULT_CLASS

                error_class = OPERATOR_FAULT_CLASS if not burn else error_class
            r._record_funded_failure(gen.hotkey, text, miner_fault=miner_fault,
                                     error_class=error_class, burn=burn)
            raise err
        self.w.cycle.ran("train", f"{sku} step {kw.get('role')}")
        return self.w.harvest(gen, block, json.loads(kw["finetune_recipe"])["marketcast"])


class FakeRunner:
    """The trainer runner attributes the lane C leg path reads."""

    def __init__(self, world: LaneCWorld, pod: FakeLegPod):
        self.cfg = world.cfg
        self.work_root = world.root / "trainer"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.wallet = SimpleNamespace(hotkey=DetKey(TRAINER))
        self._funded_leg_failures: dict = {}
        self._mkt_harvested: dict = {}
        self._run_funded_leg = pod.run_funded_leg
        self._world = world

    def hub(self):
        from cascade.shared.hippius import HubConfig

        return HubConfig.from_storage(self.cfg.storage)

    def manifest_store(self):
        return self._world.store

    def _funded_queue(self):
        return self._world.queue

    def _dispatcher(self):
        return None

    def _load_funded_ledger(self):
        return list(self._world.pod.ledger)

    def _operator_leg_pod_prefix(self, round_id, hotkey):
        return f"operator-leg-{round_id}-{hotkey[:8]}"

    def _record_funded_failure(self, hotkey, msg, *, miner_fault, error_class, burn):
        from cascade.trainer.runner_funded_pods import _FundedPodMixin

        _FundedPodMixin._record_funded_failure(self, hotkey, msg, miner_fault=miner_fault,
                                               error_class=error_class, burn=burn)


def _cycle_ops_class():
    from cascade.trainer.lane_c_driver import RunnerLaneCOps

    class CycleOps(RunnerLaneCOps):
        """The production lane C ops, with the chain / clock / pool / spawn
        seams pointed at the fakes."""

        def __init__(self, world):
            super().__init__(world.runner, world.chain)
            self.w = world

        def commitments(self):
            return list(self.w.chain.commits)

        def queue(self):
            return self.w.queue                    # lane_c_queue.json beside the intake's

        def lane_a_busy(self, hotkey):
            return False

        def _seeds(self):
            return SimpleNamespace(base_seed=777), SimpleNamespace(start_block=self.w.era_start,
                                                                   index=1)

        def mkt_embargo_scan(self, tree, *, train_time, mc=None):
            from cascade.shared.marketcast_embargo import scan_receipt

            return scan_receipt(tree, self.w.snap, self.w.mc, train_time=float(train_time))

        def now(self):
            return float(self.w.train_time)

        def spawn(self, fn, name):
            if self.w.hold_legs:
                self.w.held.append(fn)             # the leg thread dies with a restart
                return
            self.w.cycle.ran("dispatch", name)
            fn()                                   # the leg thread, run inline

    return CycleOps


# ── the world ────────────────────────────────────────────────────────────────


def mc_table(base_sha256: str, **over):
    from cascade.shared.config.marketcast import parse_marketcast
    from tests.accept.w_mktoff import helpers as H

    raw = {"from_block": H.EB, "suites": ["returns"], "assets": {"crypto": list(H.ASSETS)},
           "frequencies": {"returns": ["1m", "5m"]}, "horizons": {"returns": ["1m", "5m"]},
           "eval_context_length": 64, "windows_per_cell": 6, "backtest_windows_per_cell": 6,
           "min_windows": 10, "min_clusters": 2, "start_margin_seconds": 60,
           "verdict_wait_blocks": 2 * H.EB, "max_infra_deferrals": 2, "min_time_blocks": 2,
           "min_dollar_volume_24h": 0.0, "forward_horizons": ["1m", "5m"],
           "base_ref": f"{BASE_REPO}@{BASE_REV}", "base_sha256": base_sha256,
           "funded_pod_skus": ["1xRTX4090", "1xA100", "1xH100"]}
    raw.update(over)
    return parse_marketcast(raw)


def recipe_obj(**over) -> dict:
    r = {"target": TARGET,
         "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                  "frequencies": ["1m", "5m"], "start": "2025-01-01", "end": "2025-03-01"},
         "context_length": 64, "horizon_mix": {"1m": 1.0, "5m": 1.0},
         "covariates": ["calendar", "session"],
         "augmentations": [],
         "objective": {"suite_weights": {"returns": 1.0}, "coverage_penalty": 0.1},
         "optimizer": {"lr": 1e-4, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 50,
                       "batch": 4, "weight_decay": 0.0, "clip": 1.0, "ema_decay": 0.9},
         "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}
    r.update(over)
    return r


def history_index() -> dict:
    """The public bar index the intake's geometry check reads (every
    recipe bank recorded since 2024)."""
    files = {}
    for a in ("BTC", "ETH", "SOL"):
        for tf in ("1m", "5m"):
            files[f"crypto/{a}/{tf}/2024-01.parquet"] = {
                "class": "crypto", "symbol": a, "timeframe": tf,
                "first": "2024-01-01T00:00:00Z", "last": "2026-08-31T23:59:00Z"}
    return {"schema": "cascade-marketcast-public-bars/1", "files": files}


class LaneCWorld:
    """Everything one lane C cycle runs on (see the module docstring)."""

    def __init__(self, tmp_path: Path, monkeypatch, *, pod_script=("ok",),
                 context: int = EVAL_CONTEXT, minutes: int = SNAPSHOT_MINUTES):
        from cascade.shared import hippius, sr25519
        from cascade.shared.config import load_chain_config
        from cascade.shared.config.lane_c import LaneCConfig
        from cascade.validator import mkt_worker
        from tests.accept.w_mktoff import helpers as H

        self.tmp = Path(tmp_path)
        self.root = self.tmp / "world"
        self.root.mkdir(parents=True)
        self.mp = monkeypatch
        self.cycle = Cycle("c")
        # the Hub, the HF base snapshot, the signatures, the pod model loader
        self.hub = MemHub()
        monkeypatch.setitem(sys.modules, "hippius_hub", self.hub.module())
        monkeypatch.setattr(hippius, "_chunked_registry_reader", self.hub.registry_reader)
        monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "e2e-token")
        # an operator environment whose hippius_hub (>= 0.6) writes chunked-v2
        # by default: cascade's own pushes must switch it off (#998)
        monkeypatch.setenv("HIPPIUS_CHUNKED_WRITE", "1")
        self.base_dir = self.root / "hf-base"
        self.base_dir.mkdir()
        (self.base_dir / "config.json").write_bytes(BASE_CONFIG)
        (self.base_dir / "model.safetensors").write_bytes(BASE_WEIGHTS)
        self.base_resolves = 0

        def snapshot_download(repo, *, revision, cache_dir=None, allow_patterns=None, **_):
            assert (repo, revision) == (BASE_REPO, BASE_REV), (repo, revision)
            self.base_resolves += 1
            return str(self.base_dir)

        hf = types.ModuleType("huggingface_hub")
        hf.snapshot_download = snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
        monkeypatch.setattr(sr25519, "verify_hotkey", det_verify)
        monkeypatch.setitem(mkt_worker.LOADERS, "timestral", e2e_loader)
        # config: the shipped chain.toml with a compressed MarketCast table
        self.mc = mc_table(_sha(BASE_WEIGHTS), eval_context_length=int(context))
        self.lc = LaneCConfig(from_block=H.EB, targets=(TARGET,), legs_per_target=1,
                              leg_hours=1.0)
        base = load_chain_config(REPO / "chain.toml")
        self.cfg = replace(base, marketcast=self.mc, lane_c=self.lc,
                           manifest=replace(base.manifest, trainer_hotkey=TRAINER),
                           netconfig=replace(base.netconfig, owner_hotkey=MAIN_VALIDATOR))
        # the chain clock: the leg dispatches 100 blocks before the anchor
        from cascade.shared.config import effective_epoch_blocks

        self.eb = int(effective_epoch_blocks(self.cfg.round, H.BLOCK))
        self.clock = BlockClock(H.BLOCK - 300, epoch_blocks=self.eb)
        self.era_start = H.EB
        self.chain = FakeChain(self.clock)
        self.store = Store()
        self.snap = H.snapshot(self.root / "snap", minutes=int(minutes))
        self.train_time = float(H.ANCHOR_T - 1200)
        # the funding queue the intake and the trainer share
        import time

        from cascade.funding.queue import LaneCQueue

        self.skew = 0.0                         # the queue clock's jump over a backoff
        self.queue = LaneCQueue(self.root / "lane_c_queue.json",
                                clock=lambda: time.time() + self.skew)
        self.pod = FakeLegPod(self, pod_script)
        self.runner = FakeRunner(self, self.pod)
        self.ops = _cycle_ops_class()(self)
        from cascade.trainer.lane_c_driver import LaneCDriver

        self.driver_path = self.root / "lane_c_driver.json"
        self.driver = LaneCDriver(self.cfg, self.ops, self.driver_path)
        from cascade.shared.manifest import format_trained_pointer

        self.king = format_trained_pointer(KING_REF)
        self.intake = None
        self.server = None
        self.ref = ""
        self.sessions: list = []
        self.validators: dict = {}
        self.hold_legs = False
        self.held: list = []
        self.peaks: list = []

    # ── stage: submit ───────────────────────────────────────────────────

    def submit(self, recipe: dict | None = None) -> str:
        from cascade.miner.lane_c import recipe_error
        from cascade.shared.hippius import upload_dir_to_hub
        from cascade.shared.lane_c_manifest import format_finetune_commit

        tree = self.root / "miner-recipe"
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir()
        (tree / "marketcast.json").write_text(json.dumps(recipe or recipe_obj()))
        why = recipe_error(self.cfg, tree, TARGET)
        assert why is None, f"the miner CLI refuses the cycle recipe: {why}"
        up = upload_dir_to_hub(tree, RECIPE_REPO)
        self.ref = up.ref.immutable_ref
        self.chain.commit(HK, format_finetune_commit(TARGET, self.ref, mkt_suites=("returns",)),
                          UID, COLDKEY)
        self.clock.advance(50)                       # the reveal
        self.cycle.ran("submit", self.ref)
        return self.ref

    # ── stage: intake (real HTTP) ───────────────────────────────────────

    def start_intake(self):
        from cascade.funding.gpu_choice import GpuCatalog
        from cascade.funding.intake import FundingIntake
        from cascade.funding.queue import FundedQueue
        from cascade.funding.vault import PayerKeyVault
        from cascade.shared.hippius import fetch_from_hub

        def fetch_recipe(ref):
            d = self.root / "intake-fetch" / _sha(ref.encode())[:12]
            shutil.rmtree(d, ignore_errors=True)
            fetch_from_hub(ref, d)
            return json.loads((d / "marketcast.json").read_text())

        self.intake = FundingIntake(
            FundedQueue(self.root / "q.json"), PayerKeyVault(dir=None),
            resolve_reveal=lambda hk, ref: None, require_signature=True,
            verify=lambda hk, msg, sig: sig == hashlib.sha256(hk.encode() + msg).hexdigest(),
            lane_c_queue=self.queue, resolve_lane_c_reveal=self.chain.reveal_block,
            lane_c_targets=(TARGET,), mkt_suites=("returns",),
            mkt_gpu_catalog=GpuCatalog(skus=self.mc.funded_pod_skus,
                                       default_max_price_per_hour=40.0),
            fetch_mkt_recipe=fetch_recipe, mkt_cfg=self.mc, fetch_mkt_history=history_index)
        self.server = self.intake.make_server("127.0.0.1", 0, per_ip_rpm=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        host, port = self.server.server_address[:2]
        self.intake_url = f"http://{host}:{port}"

    def fund(self, gpu: str = GPU) -> tuple[int, dict]:
        import urllib.error
        import urllib.request

        from cascade.miner.lane_c import build_lane_c_headers

        if self.intake is None:
            self.start_intake()
        headers = build_lane_c_headers(
            "lane-c-fund", HK, TARGET, self.ref, "sk-lium-e2e",
            lambda msg: hashlib.sha256(HK.encode() + msg).digest(), gpu=gpu)
        req = urllib.request.Request(self.intake_url + "/v1/lane-c/fund", data=b"",
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                status, body = resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            status, body = e.code, json.loads(e.read())
        if 200 <= status < 300:
            self.cycle.ran("intake", f"{status} {body.get('status')}")
            e = self.queue.get(HK)
            assert e is not None and e.status == "queued" and e.gpu == gpu, e
            self.cycle.ran("queue", e.ref)
        return status, body

    # ── stage: dispatch + train + harvest ───────────────────────────────

    def tick(self) -> list:
        started = self.driver.tick(self.clock.block, 1, self.king, self.era_start)
        return started

    def restart_trainer(self):
        """A trainer restart: a new driver over the same persisted state."""
        from cascade.trainer.lane_c_driver import LaneCDriver

        self.driver = LaneCDriver(self.cfg, self.ops, self.driver_path)
        return self.driver

    def harvest(self, gen, block: int, recipe: dict):
        """The trainer's harvest of the pod's checkpoint: verified, pushed."""
        from cascade.shared.hippius import upload_dir_to_hub
        from cascade.shared.manifest import format_trained_pointer
        from cascade.trainer.marketcast_runner import verify_harvested_mkt_checkpoint

        dest = self.root / "harvest" / gen.hotkey[:12]
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        weights = safetensors_bytes([1.0] + [0.25] * 63)     # the trained LoRA-merged base
        (dest / "model.safetensors").write_bytes(weights)
        (dest / "config.json").write_bytes(BASE_CONFIG)
        meta = {"base_ref": self.mc.base_ref, "base_sha256": self.mc.base_sha256,
                "target": TARGET, "covariates": list(recipe["covariates"]),
                "data_digest": _sha(json.dumps(recipe, sort_keys=True).encode())}
        (dest / "marketcast_meta.json").write_text(json.dumps(meta, sort_keys=True))
        mkt = {"base_ref": self.mc.base_ref, "base_sha256": self.mc.base_sha256,
               "checkpoint_sha256": _sha(weights)}
        receipt = SimpleNamespace(marketcast=mkt, tensor_digests={"model.safetensors":
                                                                  _sha(weights)})
        why = verify_harvested_mkt_checkpoint(self.runner, dest, receipt,
                                              base_weights=self.base_dir / "model.safetensors")
        assert why == "", f"harvest refused: {why}"
        up = upload_dir_to_hub(dest, f"cascade/ckpt-r777-mkt_finetune-{gen.hotkey[:8]}")
        self.runner._mkt_harvested[gen.hotkey] = dict(mkt)
        self.cycle.ran("harvest", up.ref.immutable_ref)
        return SimpleNamespace(trained_pointer=format_trained_pointer(up.ref.immutable_ref),
                               corpus_digest=_sha(b"corpus"), train_block=int(block),
                               gpu_name=GPU)

    # ── stage: trainer settlement ───────────────────────────────────────

    def settle(self, round_id: str) -> object:
        """The trainer's lane C settlement bound into the lane A manifest."""
        from cascade.shared.manifest import TrainingManifest

        block = self.clock.to_next_boundary()
        digest = self.driver.settle(round_id=round_id, block=block, era_index=1,
                                    base_pointer=self.king, created_block=block)
        if digest:
            self.cycle.ran("settle_c" if "settle_c" not in self.cycle.names() else "record",
                           round_id)
        return TrainingManifest(round_id=round_id, created_block=block,
                                contract_digest="1" * 64, base_arch_digest="2" * 64,
                                eval_dataset="e2e", entries=[], lane_c_digest=digest)

    # ── stage: validators ───────────────────────────────────────────────

    def validator(self, name: str, *, main: bool, on_pod: bool = True):
        if name in self.validators:
            return self.validators[name]
        from cascade.trainer.remote import RemoteHost
        from cascade.validator.lane_c import LaneCMixin
        from cascade.validator.marketcast import MarketCastMixin
        from cascade.validator.mkt_offload import MktSession, cache_root_of
        from cascade.validator.mkt_worker import Worker
        from tests.accept.w_mktoff import helpers as H

        world = self
        host = RemoteHost(name=f"mkt-{name}", host="localhost",
                          workdir=str(self.root / f"pod-{name}"))
        cache = Path(cache_root_of(host))
        cache.mkdir(parents=True, exist_ok=True)
        # the "pod" is a forked child of the box process: fork it inside the
        # box guard so the worker (its hello probes torch) never loads torch
        # on this machine either; it runs the validator's shipped package
        popen = H.pod_popen(lambda: Worker("cpu", cache), guard=box_guard)

        import scipy.stats  # noqa: F401 — the forecaster's; imported once, before the forks

        sess = MktSession(host, run=local_run, popen=popen)
        sess.start()
        self.sessions.append(sess)

        class Validator(LaneCMixin, MarketCastMixin):
            verify_signatures = True

            def _epoch_start_block(self, manifest):
                from cascade.shared.config import effective_epoch_blocks

                eb = effective_epoch_blocks(self.cfg.round, manifest.created_block)
                return manifest.created_block // eb * eb

            def _fetch_checkpoint_dir(self, entry):
                from cascade.shared.hippius import HubRef, fetch_from_hub
                from cascade.shared.manifest import parse_trained_pointer

                ref = parse_trained_pointer(entry.trained_pointer)
                dest = Path(self.cache_dir) / HubRef.parse(ref).digest.replace(":", "-")
                return fetch_from_hub(ref, dest)

        v = Validator()
        v.cfg = self.cfg
        v.cache_dir = str(self.root / f"val-{name}-cache")
        from cascade.shared.lane_c import LaneCState

        v._lane_c_state = LaneCState()
        v._lane_c_store = self.store
        v._mkt_store_obj = self.store              # every validator reads the one bucket
        v.mkt_eval_hosts = None
        v.__dict__["_mkt_sess"] = sess
        v.mkt_snapshot_fn = lambda block: (world.snap, f"pool_mkt/snapshots/block-{block}.tar",
                                           world.snap.digest)
        v.mkt_block_hash_fn = BlockClock.block_hash
        v.mkt_forecaster_fn = None
        v.bench_signer = DetKey(MAIN_VALIDATOR) if main else DetKey(f"5E2e{name}")
        v.mkt_judge_on_pod = on_pod
        v.commitment_history_fn = lambda: list(world.chain.commits)
        v.registration_at_block_fn = lambda hotkey, block: UID if hotkey == HK else None
        self.validators[name] = v
        return v

    def judge(self, name: str, manifest, *, main: bool, budget: int = BOX_JUDGE_BUDGET,
              on_pod: bool = True):
        """One validator's lane C settlement of ``manifest`` (a box stage)."""
        v = self.validator(name, main=main, on_pod=on_pod)
        try:
            with box_guard() as refused, box_budget(f"validator {name} lane C",
                                                    budget) as reading:
                rnd = v._lane_c_judge_round(manifest)
                v._lane_c_fold(v._lane_c(), rnd)
        finally:
            self.peaks.append((name, reading.peak))
        return rnd, refused

    def close(self):
        for s in self.sessions:
            s.close()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    # ── stage: miner-visible status ─────────────────────────────────────

    def miner_status(self, capsys) -> str:
        from cascade.miner.cli import main as cli_main

        capsys.readouterr()
        rc = cli_main(["lane-c", "status", self.intake_url, "--hotkey", HK])
        out = capsys.readouterr().out
        assert rc == 0, out
        return out

    def miner_evaluation(self, receipt: dict) -> str:
        """The forward-judging progress a miner reads: the trainer observes
        the validators' receipt and publishes ``status/lane_c.json``
        (``publish_lane_c_status``); the CLI view matches the exact attempt."""
        from cascade.miner.submission_status import submission_view
        from cascade.shared.submission_progress import receipt_progress
        from cascade.trainer.lane_c_driver import LANE_C_STATUS_KEY, publish_lane_c_status

        self.driver.observe_receipt(receipt)
        sched = SimpleNamespace(lane_c=self.driver, r=self.runner)
        assert publish_lane_c_status(sched, self.clock.block, min_interval_s=0.0)
        doc = json.loads(self.store.get_text(LANE_C_STATUS_KEY))
        snapshot = {"queues": {"c": doc}, "submission_receipt": receipt_progress(receipt),
                    "freshness": {"competition": {"fresh": True}}}
        return _render(submission_view(snapshot, hotkey=HK, lane="c"))


def _render(view) -> str:
    from cascade.miner.submission_status import render_submission

    lines: list[str] = []
    render_submission(SimpleNamespace(info=lines.append), view)
    return "\n".join(lines)


def report_key(rid: str) -> str:
    from cascade.shared.marketcast_report import mkt_report_key

    return mkt_report_key(rid)


def run_cycle(world: LaneCWorld, capsys, *, restart_mid_leg: bool = False) -> dict:
    """submit → intake → queue → dispatch → train → harvest → settle_c →
    judge → report → record → verify → miner_status. Returns what the test
    asserts on (the stages are recorded on ``world.cycle``).

    ``restart_mid_leg``: the trainer restarts right after the first dispatch
    (the leg's thread dies with it); the restarted driver re-attaches."""
    from cascade.shared.marketcast_report import load_mkt_report, mkt_text_digest

    out: dict = {}
    with box_guard():
        world.submit()
        status, body = world.fund()
    out["fund"] = (status, body)
    if not 200 <= status < 300:
        return out
    # dispatch + train (+ re-dispatch after an operator-side requeue)
    out["ticks"] = 0
    if restart_mid_leg:
        world.hold_legs = True
        world.clock.advance(1)
        world.tick()
        assert world.driver.st.open and world.held, "the leg was not dispatched"
        world.held.clear()
        world.hold_legs = False
        world.restart_trainer()
    import time

    for _ in range(4):
        world.clock.advance(1)
        e = world.queue.get(HK)
        if e is not None and e.not_before:          # an operator-fault backoff: wait it out
            world.skew = max(world.skew, e.not_before - time.time() + 1.0)
        world.tick()
        out["ticks"] += 1
        e = world.queue.get(HK)
        if world.driver.st.finished or e is None or e.status != "queued":
            break
    out["entry"] = world.queue.get(HK)
    world.skew = 0.0
    if not world.driver.st.finished:
        out["miner"] = world.miner_status(capsys)
        return out
    # settlement 1: the trainer publishes the lane C manifest; the main
    # validator judges it and publishes the MarketCast report on its pod
    with box_guard():
        m1 = world.settle("9100")
    out["m1"] = m1
    rnd1, refused = world.judge("main", m1, main=True)
    out["refused"] = list(refused)
    import copy

    item = copy.deepcopy(next((i for i in world.validators["main"]._lane_c().mkt_backlog
                               if str(i["round_id"]) == "9100"), None))
    if item is not None and any(c["entry"]["miner_hotkey"] == HK for c in item["candidates"]):
        world.cycle.ran("judge", "9100")         # registered on the main validator
    key = report_key("9100")
    text = world.store.objs.get(key)
    out["report_text"] = text
    if text is None:
        out["miner"] = world.miner_status(capsys)
        return out
    rep = load_mkt_report(text)
    out["report"] = rep
    entry = rep.entry_for(HK, world.driver.st.settled["9100"]["submission_identities"][0][
        "trained_pointer"], "judge")
    out["report_entry"] = entry
    if entry is not None and entry.windows and not entry.fault and entry.status != "infra_deferred":
        world.cycle.ran("report", mkt_text_digest(text))
    # settlement 2: the trainer records the report; every validator applies it
    with box_guard():
        m2 = world.settle("9101")
    out["m2"] = m2
    rnd_main, _ = world.judge("main", m2, main=True)
    # a non-main validator replays settlement 1 and 2 from the same bytes
    world.judge("peer", m1, main=False)
    rnd_peer, _ = world.judge("peer", m2, main=False)
    out["judgements"] = (rnd_main.rows, rnd_peer.rows)
    # an independent validator recomputes the report on its own pod
    other = world.validator("audit", main=True)
    other._mkt_store_obj = Store()
    out["recomputed"] = None
    if item is not None:
        with box_guard(), box_budget("audit recompute", BOX_JUDGE_BUDGET):
            other._mkt_publish_round(item, holders={})
        out["recomputed"] = other._mkt_store_obj.objs.get(key)
    rows_ok = (rnd_main.rows == rnd_peer.rows and any(
        r.get("hotkey") == HK or r.get("miner_hotkey") == HK for r in rnd_main.rows))
    if rows_ok and out["recomputed"] == text and "record" in world.cycle.names():
        world.cycle.ran("verify", mkt_text_digest(text))
    # the validators' receipt reaches the trainer's public status document
    receipt = {"round_id": "9101", "status": "scored", "lane_c": rnd_main.rows,
               "epoch_start_block": world.validators["main"]._epoch_start_block(m2)}
    out["receipt"] = receipt
    out["miner"] = world.miner_status(capsys)
    out["evaluation"] = world.miner_evaluation(receipt)
    if ("Verdict: published in round 9100" in out["miner"]
            and "Forward evaluation passed initially" in out["evaluation"]):
        world.cycle.ran("miner_status")
    return out
