#!/usr/bin/env python3
"""WB19: measure one lane B expert-only leg per candidate GPU type (LB6, plan
§12b.6; owner input OA8), plus the full lane A leg on 8xH100.

``run`` trains one leg ON A RENTED POD under the pinned leg contract, unchanged
(``--mode expert``: the lane B driver's ``for_hours`` derivation, stopped after
``--steps`` optimizer steps; ``--mode full``: the unreduced throne contract,
all its ``training_steps``), and writes one hashed ``laned_wb19_run/1``
record: the expert-only step rate, the leg overhead split from training, peak
memory at the contract micro-batching and the image qualification facts
(driver, CUDA, the torch that ran). ``run --dry-run --gpu-fixture FILE`` is the
CPU-only preflight: the same checks, the plan record, no measurement.

``combine`` assembles the runs into ``evidence/wb19.json`` (``laned_wb19/1``)
and the per-GPU step-rate table WB22 / WB23 read
(``cascade.funding.step_rates``).

Safety: ``run`` refuses on the production host before anything else (the
markers are only stat'ed, never read), every refusal comes before torch is
imported or anything is trained, and no env value but the image digest is
ever printed or recorded. ``tests/accept/wb19/README.md`` is the contract.

  CASCADE_TRAIN_IMAGE_DIGEST=sha256:<pin> python -P scripts/laned_wb19_measure.py run \\
      --chain-toml chain.deployed.toml --contract-block N --sku 8xH100 --mode expert \\
      --steps 500 --selection-json '{"0": [1, 4]}' --candidates evidence/owner/oa8.json \\
      --expect-contract-digest HEX --out wb19-8xH100.json
  python -P scripts/laned_wb19_measure.py combine --chain-toml chain.toml \\
      --runs wb19-*.json --candidates evidence/owner/oa8.json --out evidence/wb19.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path

_STARTED = time.time()          # "tool start": the boot phase of the wall split
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))   # a checkout run without an installed cascade

RUN_SCHEMA = "laned_wb19_run/1"
EVIDENCE_SCHEMA = "laned_wb19/1"
TABLE_SCHEMA = "laned_wb19_table/1"
LEG_SCHEMA = "laned_wb19_leg/1"
WP = "WB19"
FULL_SKU = "8xH100"
# trainer.contract.TRAIN_IMAGE_DIGEST_ENV, spelled here so the preflight never
# imports the trainer package (the real leg path asserts the two agree)
IMAGE_ENV = "CASCADE_TRAIN_IMAGE_DIGEST"
HOST_ROLE_ENV = "CASCADE_HOST_ROLE"
EXAMPLE_GENERATOR = REPO / "scripts" / "example_generator"
MEMORY_SAMPLE_SECONDS = 5.0

RUN_REFUSALS = frozenset({
    "prod_host", "wrong_contract", "gpu_not_candidate", "wrong_image", "no_gpu",
    "gpu_mismatch", "bad_selection", "bad_steps", "fixture_not_allowed",
    "leg_failed", "non_finite",
})
COMBINE_REFUSALS = frozenset({
    "bad_run", "not_pod", "wrong_image", "wrong_contract", "mixed_block", "duplicate_sku",
    "bad_note",
})
PROD_MARKERS = (
    "etc/cascade-ops.env",
    "etc/systemd/system/cascade-ops-follow.service",
    "etc/systemd/system/cascade-intake.service",
    "root/cascade/validator_state.db",
)
RUN_FIELDS = ("schema", "wp", "source", "mode", "sku", "contract_block", "gpu", "torch",
              "image_digest", "contract", "selection", "steps", "leg_config_digest", "wall",
              "steps_per_s", "tokens_per_s", "peak_memory_bytes", "finite", "tool",
              "body_sha256")
MEASURED = ("wall", "steps_per_s", "tokens_per_s", "peak_memory_bytes", "finite", "torch")

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_LAYER = re.compile(r"[0-9]+")


class MeasureRefused(Exception):
    """A refusal: ``code`` is a :data:`RUN_REFUSALS` or :data:`COMBINE_REFUSALS`
    member; the CLI exits 2 with ``REFUSED <code>: <message>``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ── hashing ─────────────────────────────────────────────────────────────────

def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def body_sha256(record: dict) -> str:
    """sha256 of the canonical JSON of ``record`` without ``body_sha256``."""
    return hashlib.sha256(canonical({k: v for k, v in record.items()
                                     if k != "body_sha256"})).hexdigest()


def _sealed(record: dict) -> dict:
    out = dict(record)
    out["body_sha256"] = body_sha256(out)
    return out


def leg_config_digest(*, mode: str, sku: str, contract_digest: str, image_digest: str,
                      steps: int, selection) -> str:
    leg = {"schema": LEG_SCHEMA, "mode": mode, "sku": sku, "contract_digest": contract_digest,
           "image_digest": image_digest, "steps": int(steps), "selection": selection}
    return hashlib.sha256(canonical(leg)).hexdigest()


def _write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ── the prod guard ──────────────────────────────────────────────────────────

def _present(path: Path) -> bool:
    """Stat only, never open. An OSError (e.g. a runner that may not stat
    /root) counts as absent."""
    try:
        return path.exists()
    except OSError:
        return False


def prod_reasons(env, fs_root) -> list[str]:
    """Why this host is the production box (empty = not prod)."""
    reasons = []
    if str((env or {}).get(HOST_ROLE_ENV, "")).strip().lower() == "prod":
        reasons.append(f"{HOST_ROLE_ENV} is prod")
    root = Path(fs_root)
    for marker in PROD_MARKERS:
        if _present(root / marker):
            reasons.append(f"prod marker present: /{marker}")
    return reasons


# ── the leg contract ────────────────────────────────────────────────────────

def _load_cfg(path: Path):
    from cascade.shared.config.loader import load_chain_config

    return load_chain_config(path)


def leg_contract(cfg, block: int, mode: str):
    """``expert``: the lane B driver's derivation; ``full``: the unreduced
    throne contract (the lane A leg)."""
    contract = cfg.throne_contracts_at(int(block))[0]
    if mode == "full":
        return contract
    return contract.for_hours(cfg.lane_b.leg_hours, guard_factor=cfg.round.heat_guard_factor,
                              guard_floor_seconds=cfg.round.heat_guard_floor_seconds)


def _contract_digest(contract) -> str:
    from cascade.shared.manifest import contract_digest

    return contract_digest(contract)


def _pinned_image(contract) -> str | None:
    m = _SHA256.search(str(getattr(contract, "train_image_digest", "") or "").strip().lower())
    return m.group(0) if m else None


def _contract_summary(cfg, contract, mode: str) -> dict:
    return {"digest": _contract_digest(contract),
            "training_steps": int(contract.training_steps),
            "max_train_seconds": int(contract.max_train_seconds),
            "micro_batches": int(contract.micro_batches),
            "batch_size": int(contract.batch_size),
            "context_length": int(contract.context_length),
            "leg_hours": float(cfg.lane_b.leg_hours) if mode == "expert" else None}


# ── candidates and GPUs ─────────────────────────────────────────────────────

def _read_candidates(path: Path | None) -> tuple[list[str], str | None]:
    """The OA8 record's ``gpu_types`` and the sha256 of its bytes."""
    if path is None:
        return [], None
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    types = doc.get("gpu_types") if isinstance(doc, dict) else None
    if not isinstance(types, list) or not all(isinstance(t, str) and t.strip() for t in types):
        raise ValueError(f"{path}: gpu_types must be a list of SKU strings")
    return [t.strip() for t in types], hashlib.sha256(raw).hexdigest()


def candidates(cfg, extra: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for sku in [*cfg.lane_b.funded_pod_skus, *extra]:
        if sku.lower() not in {s.lower() for s in out}:
            out.append(sku)
    return tuple(out)


def name_matches(base: str, name: str) -> bool:
    """``base`` equals one run of consecutive whitespace tokens of ``name``
    (case-insensitive): ``H100`` matches "NVIDIA H100 80GB HBM3", ``L40``
    never matches "NVIDIA L40S"."""
    want = base.strip().lower()
    toks = str(name).lower().split()
    return bool(want) and any("".join(toks[i:j]) == want for i in range(len(toks))
                              for j in range(i + 1, len(toks) + 1))


def _gpu_facts_from_fixture(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    gpus = doc.get("gpus") if isinstance(doc, dict) else None
    if not isinstance(gpus, list):
        raise MeasureRefused("gpu_mismatch", f"{path}: no gpus list")
    out = []
    for g in gpus:
        if not (isinstance(g, dict) and isinstance(g.get("name"), str)
                and isinstance(g.get("memory_total_bytes"), int)):
            raise MeasureRefused("gpu_mismatch", f"{path}: each gpu needs name and "
                                                 "memory_total_bytes")
        out.append({"name": g["name"], "memory_total_bytes": int(g["memory_total_bytes"])})
    return {"gpus": out, "driver": str(doc.get("driver") or ""),
            "cuda": str(doc.get("cuda") or "")}


def _nvidia_smi(env) -> str | None:
    return shutil.which("nvidia-smi", path=str((env or {}).get("PATH", "")))


def _gpu_facts_from_smi(smi: str) -> dict:
    run_env = {"PATH": str(Path(smi).parent)}
    q = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                       timeout=60, env=run_env)
    if q.returncode != 0:
        raise MeasureRefused("no_gpu", f"nvidia-smi failed (rc {q.returncode})")
    gpus, drivers = [], set()
    for line in q.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise MeasureRefused("no_gpu", f"unparseable nvidia-smi line {line!r}")
        name, mib, driver = parts
        gpus.append({"name": name, "memory_total_bytes": int(float(mib)) * 1024 * 1024})
        drivers.add(driver)
    head = subprocess.run([smi], capture_output=True, text=True, timeout=60, env=run_env)
    m = re.search(r"CUDA Version:\s*([0-9.]+)", head.stdout)
    return {"gpus": gpus, "driver": ",".join(sorted(drivers)), "cuda": m.group(1) if m else ""}


def check_gpus(sku: str, facts: dict) -> dict:
    """The visible GPUs against the layout SKU; the record's ``gpu``."""
    from cascade.provision.layout import split_layout

    base, n = split_layout(sku)
    gpus = facts["gpus"]
    names = {g["name"] for g in gpus}
    if len(gpus) != n:
        raise MeasureRefused("gpu_mismatch", f"{sku} needs {n} GPU(s); {len(gpus)} visible")
    if len(names) != 1:
        raise MeasureRefused("gpu_mismatch", f"{sku}: mixed GPU types {sorted(names)}")
    name = next(iter(names))
    if not name_matches(base, name):
        raise MeasureRefused("gpu_mismatch", f"{sku}: {name!r} is not a {base}")
    return {"name": name, "count": n, "driver": facts["driver"], "cuda": facts["cuda"],
            "memory_total_bytes": min(g["memory_total_bytes"] for g in gpus)}


# ── selection ───────────────────────────────────────────────────────────────

def parse_selection(raw: str | None) -> dict:
    """The worker's ``--selection-json`` shape, canonical (str keys, rows
    sorted ascending)."""
    if raw is None:
        raise MeasureRefused("bad_selection", "--mode expert needs --selection-json")
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise MeasureRefused("bad_selection", f"--selection-json is not JSON ({e})") from e
    if not isinstance(doc, dict) or not doc:
        raise MeasureRefused("bad_selection", "the selection must be a non-empty object")
    out = {}
    for key, rows in doc.items():
        if not isinstance(key, str) or not _LAYER.fullmatch(key):
            raise MeasureRefused("bad_selection", f"layer {key!r} is not a decimal ordinal")
        if not isinstance(rows, list) or not rows:
            raise MeasureRefused("bad_selection", f"layer {key}: rows must be a non-empty list")
        if not all(isinstance(r, int) and not isinstance(r, bool) and r >= 0 for r in rows):
            raise MeasureRefused("bad_selection", f"layer {key}: rows must be ints >= 0")
        if len(set(rows)) != len(rows):
            raise MeasureRefused("bad_selection", f"layer {key}: duplicate rows")
        out[str(int(key))] = sorted(rows)
    if len(out) != len(doc):
        raise MeasureRefused("bad_selection", "a layer ordinal is given twice")
    return out


# ── tool provenance ─────────────────────────────────────────────────────────

def _tool_commit(env) -> dict:
    """The checkout's commit and whether the tracked tree is clean. Unknown
    (no git, not a checkout) records an empty sha and ``git_dirty: true``."""
    git = shutil.which("git", path=str((env or {}).get("PATH", "")))
    if not git:
        return {"git_sha": "", "git_dirty": True}
    run_env = {"PATH": str(Path(git).parent), "GIT_CONFIG_NOSYSTEM": "1",
               "HOME": str(REPO)}
    try:
        sha = subprocess.run([git, "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=60, env=run_env)
        st = subprocess.run([git, "-C", str(REPO), "status", "--porcelain",
                             "--untracked-files=no"], capture_output=True, text=True,
                            timeout=60, env=run_env)
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "", "git_dirty": True}
    value = sha.stdout.strip()
    if sha.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        return {"git_sha": "", "git_dirty": True}
    return {"git_sha": value, "git_dirty": st.returncode != 0 or bool(st.stdout.strip())}


# ── finishing a record ──────────────────────────────────────────────────────

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def finish_record(plan: dict, result: dict) -> dict:
    """The plan plus the leg's result as a finished, re-hashed record.

    Raises :class:`MeasureRefused` ``leg_failed`` (the leg did not run its
    planned steps cleanly; wins over everything else) or ``non_finite``."""
    if not result.get("ok") or result.get("deadline_hit") is not False \
            or result.get("steps") != plan["steps"]:
        raise MeasureRefused("leg_failed", f"leg ok={result.get('ok')!r} deadline_hit="
                                           f"{result.get('deadline_hit')!r} steps="
                                           f"{result.get('steps')!r} (planned {plan['steps']}); "
                                           f"error: {result.get('error')}")
    wall = result.get("wall")
    if not isinstance(wall, dict) or not isinstance(wall.get("overhead_seconds"), dict):
        raise MeasureRefused("leg_failed", "the leg result carries no wall split")
    over = wall["overhead_seconds"]
    values = [result.get("final_loss"), wall.get("total_seconds"), wall.get("train_seconds"),
              over.get("boot"), over.get("corpus"), over.get("harvest")]
    if not all(_finite(v) for v in values):
        raise MeasureRefused("non_finite", "a non-finite loss or wall value")
    if wall["train_seconds"] <= 0 or any(v < 0 for v in values[1:]):
        raise MeasureRefused("non_finite", "train_seconds must be > 0 and every wall value "
                                           ">= 0")
    peak = result.get("peak_memory_bytes")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        raise MeasureRefused("leg_failed", f"peak_memory_bytes {peak!r} is not an int > 0")
    rec = dict(plan)
    c = plan["contract"]
    steps = int(plan["steps"])
    train = float(wall["train_seconds"])
    rec.update(
        torch=str(result.get("torch") or ""),
        peak_memory_bytes=peak,
        finite=True,
        wall={"total_seconds": float(wall["total_seconds"]), "train_seconds": train,
              "overhead_seconds": {k: float(over[k]) for k in ("boot", "corpus", "harvest")}},
        steps_per_s=steps / train,
        tokens_per_s=steps * int(c["batch_size"]) * int(c["context_length"]) / train,
    )
    return _sealed(rec)


# ── the real leg (pod only; torch is imported here and nowhere else) ────────

class _StepGate:
    """Wraps the backend's batch iterator: stops it after ``limit`` batches
    (one optimizer step each) and stamps the first batch (the backend's
    training clock starts on it). Batches pass through untouched."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.first_batch: float | None = None
        self.yielded = 0

    def wrap(self, original):
        def gated(*args, **kwargs):
            it = iter(original(*args, **kwargs))
            while self.yielded < self.limit:
                try:
                    batch = next(it)
                except StopIteration:
                    return
                if self.first_batch is None:
                    self.first_batch = time.time()
                self.yielded += 1
                yield batch
        return gated


@contextlib.contextmanager
def _gated(module, gate: _StepGate):
    original = module.iter_training_batches
    module.iter_training_batches = gate.wrap(original)
    try:
        yield
    finally:
        module.iter_training_batches = original


class _MemorySampler:
    """Peak ``memory.used`` over every GPU, sampled from nvidia-smi: the
    cross-process bound for a multi-GPU gang (its helper ranks are separate
    processes whose allocators this process cannot read)."""

    def __init__(self, smi: str) -> None:
        self._smi = smi
        self._stop = threading.Event()
        self.peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="wb19-mem")

    def _sample(self) -> None:
        q = subprocess.run([self._smi, "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=30, env={"PATH": str(Path(self._smi).parent)})
        for line in q.stdout.split():
            with contextlib.suppress(ValueError):
                self.peak = max(self.peak, int(float(line)) * 1024 * 1024)

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                self._sample()
            self._stop.wait(MEMORY_SAMPLE_SECONDS)

    def __enter__(self) -> _MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=60)


# The reference generator's own channel limit (scripts/example_generator/generator.py).
EXAMPLE_GENERATOR_CHANNELS = (8, 32)


def _example_channels(gen_cfg) -> int:
    """The channel count the reference generator must emit: the contract's
    ``max_channels``, refused unless it lies inside both the contract's
    ``[min_channels, max_channels]`` and the generator's own limit."""
    lo, hi = int(gen_cfg.min_channels), int(gen_cfg.max_channels)
    channels = hi
    ex_lo, ex_hi = EXAMPLE_GENERATOR_CHANNELS
    if not (lo <= channels <= hi and ex_lo <= channels <= ex_hi):
        raise RuntimeError(
            f"the contract's generator channels [{lo}, {hi}] admit no count the reference "
            f"generator can emit ([{ex_lo}, {ex_hi}]); refusing to build a corpus that "
            f"would fault every series")
    return channels


def _example_generator(work: Path, contract, gen_cfg) -> Path:
    """The repo's reference generator (owner code, still run sandboxed) set to
    full-context series at the contract's channel count, so every batch is the
    contract's ``batch_size × context_length`` points and every series passes
    the ``[min_channels, max_channels]`` check."""
    channels = _example_channels(gen_cfg)
    repo = work / "generator"
    shutil.copytree(EXAMPLE_GENERATOR, repo, dirs_exist_ok=True)
    conf_path = repo / "config.json"
    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    conf["min_length"] = conf["max_length"] = int(contract.context_length)
    conf["channels"] = channels
    conf_path.write_text(json.dumps(conf, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return repo


def _materialise_corpus(cfg, contract, seeds, repo: Path, out: Path, points: int) -> Path:
    """A ``shards`` corpus of ``points`` points from ``repo``, written the way
    the CPU generation tier writes one (sandboxed children, verified shards)."""
    from dataclasses import replace

    from cascade.generation.tier import generate_corpus

    gen_cfg = replace(cfg.generator_for_contract(contract), sandbox_strict=True)
    gcfg = replace(cfg.generation, target_points=int(points))
    if gcfg.children <= 0:
        gcfg = replace(gcfg, vcpus=max(1, os.cpu_count() or 1), fallback_vcpus=())
    idx = generate_corpus(repo, seeds.generation_seed, gen_cfg, gcfg, out, use_sandbox=True,
                          generator_ref="scripts/example_generator")
    if idx.get("faults"):
        raise RuntimeError(f"corpus generation stopped: {idx.get('stop_reason')} "
                           f"({len(idx['faults'])} fault(s))")
    return out


def _backend_module(contract):
    name = {"cascade_pyramid": "cascade.trainer.pyramid_trainer",
            "cascade_moe": "cascade.trainer.moe_trainer"}.get(
        str(getattr(contract, "base_arch", "") or ""), "cascade.trainer.toto2_trainer")
    module = importlib.import_module(name)
    if not hasattr(module, "iter_training_batches"):
        raise RuntimeError(f"{name} has no batch iterator to bound the leg by")
    return module


def _digest_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[p.relative_to(root).as_posix()] = h.hexdigest()
    return out


def _train_leg(plan: dict, *, cfg, contract, selection: dict | None, work_root: Path,
               corpus_dir: Path | None, smi: str, started: float) -> dict:
    import torch

    from cascade.trainer.contract import (
        TRAIN_IMAGE_DIGEST_ENV,
        RoundSeeds,
        assert_train_image,
    )
    from cascade.trainer.stream import open_round_stream
    from cascade.trainer.toto2_trainer import Toto2Trainer

    if TRAIN_IMAGE_DIGEST_ENV != IMAGE_ENV:
        raise RuntimeError(f"the trainer reads {TRAIN_IMAGE_DIGEST_ENV}, this tool {IMAGE_ENV}")
    assert_train_image(contract)
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device")
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    trainer = Toto2Trainer()
    seeds = RoundSeeds.derive(int(plan["contract_block"]), contract)
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=work_root, prefix=f"{plan['sku']}-{plan['mode']}-"))
    base_dir = None
    if plan["mode"] == "expert":
        if str(contract.base_arch) != "cascade_pyramid":
            raise RuntimeError(f"an expert leg needs the cascade_pyramid backend; the leg "
                               f"contract trains {contract.base_arch!r}")
        from cascade.trainer.pyramid_trainer import build_random_init

        base_dir = work / "base"
        base_dir.mkdir()
        build_random_init(contract, training_seed=seeds.training_seed, out_dir=base_dir)
    module = _backend_module(contract)
    boot_end = time.time()

    steps = int(plan["steps"])
    points = max(int(contract.train_tokens) if plan["mode"] == "full" else 0,
                 steps * int(contract.batch_size) * int(contract.context_length))
    repo = _example_generator(work, contract, cfg.generator_for_contract(contract))
    if contract.corpus_mode == "shards" and corpus_dir is None:
        corpus_dir = _materialise_corpus(cfg, contract, seeds, repo, work / "corpus",
                                         int(points * 1.05))
    marks: dict[str, float] = {}

    def marked(series):
        marks["first_pull"] = time.time()
        yield from series

    gate = _StepGate(steps)
    out_dir = work / "leg"
    sampler_cm = _MemorySampler(smi) if plan["mode"] == "full" and plan["gpu"]["count"] > 1 \
        else contextlib.nullcontext()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    with sampler_cm as sampler, open_round_stream(
            contract.corpus_mode, repo, seeds.generation_seed,
            cfg.generator_for_contract(contract), token_budget=int(contract.train_tokens),
            use_sandbox=True, blocked=cfg.static_guard.blocked,
            max_wall_seconds=contract.max_train_seconds,
            seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1),
            budget_denomination=getattr(contract, "budget_denomination", "points"),
            corpus_epochs=int(getattr(contract, "corpus_epochs", 0) or 0),
            corpus_dir=corpus_dir,
            continuation_protocol=str(getattr(contract, "continuation_protocol", "") or ""),
    ) as rs, _gated(module, gate):
        train_call = time.time()
        res = trainer.train(
            marked(rs.series()), contract, training_seed=seeds.training_seed,
            token_budget=int(contract.train_tokens), out_dir=out_dir,
            **({"warm_start_dir": base_dir} if base_dir is not None else {}),
            **({"expert_selection": {int(k): list(v) for k, v in selection.items()}}
               if selection is not None else {}))
    _digest_tree(out_dir)
    harvest_end = time.time()
    peak = max((int(torch.cuda.max_memory_allocated(i))
                for i in range(torch.cuda.device_count())), default=0)
    if sampler is not None:
        peak = max(peak, int(sampler.peak))
    if gate.first_batch is None or "first_pull" not in marks:
        raise RuntimeError("the leg trained no batch")
    m = res.metrics or {}
    train = float(m.get("train_seconds", res.train_seconds))
    boot = (boot_end - started) + (marks["first_pull"] - train_call)
    corpus = (train_call - boot_end) + (gate.first_batch - marks["first_pull"])
    harvest = max(0.0, harvest_end - (gate.first_batch + train))
    total = max(harvest_end - started, boot + corpus + train + harvest)
    return {"steps": int(m.get("steps", -1)), "deadline_hit": bool(m.get("deadline_hit")),
            "final_loss": float(m.get("final_loss", math.nan)), "torch": torch.__version__,
            "peak_memory_bytes": peak,
            "wall": {"total_seconds": total, "train_seconds": train,
                     "overhead_seconds": {"boot": boot, "corpus": corpus, "harvest": harvest}}}


def run_leg(plan: dict, **kw) -> dict:
    """The leg's result in ``finish_record``'s shape; never raises."""
    result = {"ok": False, "steps": 0, "error": None, "deadline_hit": False,
              "final_loss": None, "torch": None, "peak_memory_bytes": None, "wall": None}
    try:
        result.update(_train_leg(plan, **kw))
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 — any leg failure is leg_failed, reported
        result["error"] = f"{type(e).__name__}: {e}"[:800]
    return result


# ── run ─────────────────────────────────────────────────────────────────────

def _plan(args, env, fs_root) -> tuple[dict, dict]:
    """Every check, in the pinned order; the plan record and what the real
    leg needs. Raises MeasureRefused."""
    reasons = prod_reasons(env, fs_root)
    if reasons:
        raise MeasureRefused("prod_host", "this host is the production box ("
                             + "; ".join(reasons) + "); WB19 legs run on a rented pod")
    if args.mode == "full" and args.steps is not None:
        raise MeasureRefused("bad_steps", "a full leg runs the contract's training_steps; "
                                          "--steps is for --mode expert")
    if args.mode == "expert" and (args.steps is None or args.steps < 1):
        raise MeasureRefused("bad_steps", f"--mode expert needs --steps >= 1; got {args.steps}")
    if args.mode == "expert":
        selection = parse_selection(args.selection_json)
    elif args.selection_json is not None:
        raise MeasureRefused("bad_selection", "a full leg trains every row; --selection-json "
                                              "is for --mode expert")
    else:
        selection = None
    if args.gpu_fixture is not None and not args.dry_run:
        raise MeasureRefused("fixture_not_allowed", "--gpu-fixture is for --dry-run only; a "
                                                    "pod record carries probed GPU facts")
    if args.dry_run and args.gpu_fixture is None:
        raise MeasureRefused("fixture_not_allowed", "--dry-run needs --gpu-fixture")

    try:
        cfg = _load_cfg(args.chain_toml)
        contract = leg_contract(cfg, args.contract_block, args.mode)
        summary = _contract_summary(cfg, contract, args.mode)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise MeasureRefused("wrong_contract", f"{args.chain_toml} gives no leg contract at "
                                               f"block {args.contract_block}: {e}") from e
    v2_from = int(getattr(cfg.training, "v2_from_block", 0) or 0)
    if args.contract_block < v2_from:
        raise MeasureRefused("wrong_contract", f"block {args.contract_block} precedes [v2] "
                                               f"from_block {v2_from}")
    if args.expect_contract_digest is not None \
            and args.expect_contract_digest.strip().lower() != summary["digest"]:
        raise MeasureRefused("wrong_contract", f"{args.chain_toml} derives {args.mode} leg "
                                               f"contract {summary['digest']}, not the expected "
                                               f"{args.expect_contract_digest.strip()}")

    try:
        extra, _ = _read_candidates(args.candidates)
    except (OSError, ValueError) as e:
        raise MeasureRefused("gpu_not_candidate", f"--candidates unreadable: {e}") from e
    from cascade.funding.gpu_choice import GpuCatalog

    sku = GpuCatalog(skus=candidates(cfg, extra)).canonical(args.sku)
    if sku is None:
        raise MeasureRefused("gpu_not_candidate", f"{args.sku!r} is neither a [lane_b] "
                                                  "funded_pod_skus entry nor an OA8 gpu_type")

    pinned = _pinned_image(contract)
    runtime_raw = str(env.get(IMAGE_ENV, "") or "")
    m = _SHA256.search(runtime_raw.strip().lower())
    runtime = m.group(0) if m else None
    if pinned is None:
        raise MeasureRefused("wrong_image", "the leg contract pins no sha256 train image")
    if runtime is None:
        raise MeasureRefused("wrong_image", f"{IMAGE_ENV} is unset or carries no sha256 digest")
    if runtime != pinned:
        raise MeasureRefused("wrong_image", f"{IMAGE_ENV} is {runtime}, the leg contract pins "
                                            f"{pinned}")

    smi = None
    if args.dry_run:
        facts = _gpu_facts_from_fixture(args.gpu_fixture)
    else:
        smi = _nvidia_smi(env)
        if smi is None:
            raise MeasureRefused("no_gpu", "no nvidia-smi on PATH: this is not a GPU pod")
        facts = _gpu_facts_from_smi(smi)
    gpu = check_gpus(sku, facts)

    steps = int(args.steps) if args.mode == "expert" else int(contract.training_steps)
    plan = {"schema": RUN_SCHEMA, "wp": WP, "source": "dry_run" if args.dry_run else "pod",
            "mode": args.mode, "sku": sku, "contract_block": int(args.contract_block),
            "gpu": gpu, "torch": None, "image_digest": pinned, "contract": summary,
            "selection": selection, "steps": steps,
            "leg_config_digest": leg_config_digest(
                mode=args.mode, sku=sku, contract_digest=summary["digest"],
                image_digest=pinned, steps=steps, selection=selection),
            "wall": None, "steps_per_s": None, "tokens_per_s": None,
            "peak_memory_bytes": None, "finite": None, "tool": _tool_commit(env)}
    return _sealed(plan), {"cfg": cfg, "contract": contract, "selection": selection,
                           "smi": smi}


def _cmd_run(args, env, fs_root) -> int:
    plan, ctx = _plan(args, env, fs_root)
    if args.dry_run:
        record = plan
    else:
        result = run_leg(plan, cfg=ctx["cfg"], contract=ctx["contract"],
                         selection=ctx["selection"], work_root=args.work_dir,
                         corpus_dir=args.corpus_dir, smi=ctx["smi"], started=_STARTED)
        record = finish_record(plan, result)
    _write_atomic(args.out, record)
    summary = {"leg_config_digest": record["leg_config_digest"],
               "contract_digest": record["contract"]["digest"], "sku": record["sku"],
               "steps": record["steps"], "source": record["source"],
               "steps_per_s": record["steps_per_s"], "tokens_per_s": record["tokens_per_s"]}
    print(f"wrote {args.out}")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


# ── combine ─────────────────────────────────────────────────────────────────

def _load_run(path: Path) -> dict:
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise MeasureRefused("bad_run", f"{path}: unreadable ({e})") from e
    if not isinstance(rec, dict) or rec.get("schema") != RUN_SCHEMA or rec.get("wp") != WP:
        raise MeasureRefused("bad_run", f"{path}: not a {RUN_SCHEMA} record")
    missing = [k for k in RUN_FIELDS if k not in rec]
    if missing:
        raise MeasureRefused("bad_run", f"{path}: missing {missing}")
    if rec["body_sha256"] != body_sha256(rec):
        raise MeasureRefused("bad_run", f"{path}: body_sha256 does not match the body "
                                        "(hand-edited)")
    if rec["mode"] not in ("expert", "full"):
        raise MeasureRefused("bad_run", f"{path}: mode {rec['mode']!r}")
    if not (isinstance(rec["contract"], dict) and isinstance(rec["contract"].get("digest"), str)
            and isinstance(rec["gpu"], dict)):
        raise MeasureRefused("bad_run", f"{path}: malformed contract / gpu")
    return rec


def _overhead(rec: dict) -> float:
    o = rec["wall"]["overhead_seconds"]
    return o["boot"] + o["corpus"] + o["harvest"]


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


NOTE_MAX = 200


def _parse_notes(notes: Sequence[str], skus: Sequence[str]) -> dict[str, str]:
    """``SKU=TEXT`` notes keyed by the recorded SKU spelling (rev6.16)."""
    by_lower = {s.lower(): s for s in skus}
    out: dict[str, str] = {}
    for raw in notes:
        sku, sep, text = str(raw).partition("=")
        sku = sku.strip()
        if not sep or not sku:
            raise MeasureRefused("bad_note", f"--note {raw!r} is not SKU=TEXT")
        if sku.lower() not in by_lower:
            raise MeasureRefused("bad_note", f"--note {sku}: no expert run for that SKU")
        if not text.strip() or len(text) > NOTE_MAX:
            raise MeasureRefused("bad_note", f"--note {sku}: the text must be 1..{NOTE_MAX} "
                                             "characters")
        out[by_lower[sku.lower()]] = text
    return out


def combine(cfg, runs: list[dict], *, oa8_types: Sequence[str],
            oa8_sha256: str | None, notes: Sequence[str] = ()) -> dict:
    """The ``laned_wb19/1`` evidence from pod run records. Raises
    MeasureRefused (a :data:`COMBINE_REFUSALS` code). ``notes`` (``SKU=TEXT``)
    annotate measured rows; they never change a rate. Coverage may fail and
    the evidence is still usable (partial evidence, owner 2026-11-23, plan
    rev6.16): the reader loads it when coverage is the only failing check."""
    for r in runs:
        if r["source"] != "pod":
            raise MeasureRefused("not_pod", f"{r['sku']}/{r['mode']}: a {r['source']!r} "
                                            "record measures nothing")
        if any(r[k] is None for k in MEASURED):
            raise MeasureRefused("bad_run", f"{r['sku']}/{r['mode']}: a pod record without "
                                            "its measurements")
    blocks = sorted({int(r["contract_block"]) for r in runs})
    if len(blocks) > 1:
        raise MeasureRefused("mixed_block", f"runs at contract blocks {blocks}")
    block = blocks[0]
    contracts = {}
    for mode in ("expert", "full"):
        c = leg_contract(cfg, block, mode)
        contracts[mode] = (c, _contract_digest(c))
    pinned = _pinned_image(contracts["expert"][0])
    for r in runs:
        if r["image_digest"] != pinned:
            raise MeasureRefused("wrong_image", f"{r['sku']}/{r['mode']}: image "
                                                f"{r['image_digest']} is not the pin {pinned}")
        c, digest = contracts[r["mode"]]
        if r["contract"]["digest"] != digest:
            raise MeasureRefused("wrong_contract", f"{r['sku']}/{r['mode']}: contract "
                                                   f"{r['contract']['digest']} is not the "
                                                   f"{r['mode']} leg contract {digest}")
        if r["mode"] == "full" and r["steps"] != int(c.training_steps):
            raise MeasureRefused("wrong_contract", f"{r['sku']}/full ran {r['steps']} steps, "
                                                   f"the contract {c.training_steps}")
    expert = [r for r in runs if r["mode"] == "expert"]
    seen: set[str] = set()
    for r in expert:
        if r["sku"].lower() in seen:
            raise MeasureRefused("duplicate_sku", f"two expert runs for {r['sku']}")
        seen.add(r["sku"].lower())

    need = candidates(cfg, oa8_types)
    missing = [s for s in need if s.lower() not in seen]
    full = [r for r in runs if r["mode"] == "full" and r["sku"] == FULL_SKU]
    oom = [r["sku"] for r in expert
           if not r["peak_memory_bytes"] < r["gpu"]["memory_total_bytes"]]
    checks = [
        _check("coverage", not missing,
               f"missing expert runs: {missing}" if missing
               else f"every candidate measured: {list(need)}"),
        # The full lane A leg is OPTIONAL (owner 2026-11-23, "Yes skip it";
        # plan rev6.5): the measured [round] funded_sku_wall_seconds stands for
        # the full-leg wall. Zero full runs pass; one is validated as before;
        # two or more fail.
        _check("full_leg", len(full) <= 1,
               f"0 full {FULL_SKU} runs; skipped: owner_2026-11-23 (full leg optional)"
               if not full else
               f"{len(full)} full {FULL_SKU} run(s); at most one is allowed"),
        _check("memory", not oom,
               f"peak >= memory_total for {oom}" if oom
               else "every expert leg fits memory at the contract micro-batching"),
    ]
    rows = {r["sku"]: {"gpu_name": r["gpu"]["name"], "gpu_count": r["gpu"]["count"],
                       "driver": r["gpu"]["driver"], "cuda": r["gpu"]["cuda"],
                       "steps_per_s": r["steps_per_s"], "tokens_per_s": r["tokens_per_s"],
                       "overhead_seconds": _overhead(r),
                       "peak_memory_bytes": r["peak_memory_bytes"],
                       "memory_total_bytes": r["gpu"]["memory_total_bytes"],
                       "run_body_sha256": r["body_sha256"]} for r in expert}
    for sku, text in _parse_notes(notes, [r["sku"] for r in expert]).items():
        rows[sku]["note"] = text
    table = {"schema": TABLE_SCHEMA, "image_digest": pinned,
             "contract_digest": contracts["expert"][1], "contract_block": block, "rows": rows}
    lane_a = None
    if full:
        f = full[0]
        lane_a = {"sku": FULL_SKU, "train_seconds": f["wall"]["train_seconds"],
                  "total_seconds": f["wall"]["total_seconds"],
                  "overhead_seconds": _overhead(f),
                  "funded_sku_wall_seconds": dict(cfg.round.funded_sku_wall_seconds).get(
                      FULL_SKU)}
    ev = {"schema": EVIDENCE_SCHEMA, "wp": WP, "passed": all(c["passed"] for c in checks),
          "checks": checks, "runs": runs, "table": table, "lane_a_full_leg": lane_a,
          "oa8_sha256": oa8_sha256}
    return _sealed(ev)


def _cmd_combine(args, parser) -> int:
    try:
        cfg = _load_cfg(args.chain_toml)
    except (OSError, ValueError) as e:
        parser.error(f"--chain-toml {args.chain_toml} does not load: {e}")
    try:
        oa8_types, oa8_sha = _read_candidates(args.candidates)
    except (OSError, ValueError) as e:
        parser.error(f"--candidates unreadable: {e}")
    runs = [_load_run(p) for p in args.runs]
    ev = combine(cfg, runs, oa8_types=oa8_types, oa8_sha256=oa8_sha,
                 notes=tuple(args.note or ()))
    _write_atomic(args.out, ev)
    if args.table_out is not None:
        _write_atomic(args.table_out, ev["table"])
    print(f"wrote {args.out}" + (f" and {args.table_out}" if args.table_out else ""))
    print(json.dumps({"passed": ev["passed"], "runs": len(runs),
                      "checks": {c["name"]: c["passed"] for c in ev["checks"]}},
                     sort_keys=True))
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="laned_wb19_measure.py",
                                description="WB19 per-GPU expert-leg measurement.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure one leg on this pod (or --dry-run)")
    r.add_argument("--chain-toml", type=Path, required=True)
    r.add_argument("--contract-block", type=int, required=True,
                   help="the era start the leg contract is effective at (>= [v2] from_block)")
    r.add_argument("--sku", required=True, help="the layout SKU this pod was rented as")
    r.add_argument("--mode", choices=("expert", "full"), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, default=None,
                   help="optimizer steps an expert leg runs (>= 1)")
    r.add_argument("--selection-json", default=None,
                   help='expert rows {"<moe layer>": [ids]} (--mode expert)')
    r.add_argument("--candidates", type=Path, default=None,
                   help="the owner's OA8 record (evidence/owner/oa8.json)")
    r.add_argument("--expect-contract-digest", default=None,
                   help="refuse unless the derived leg contract digest is this")
    r.add_argument("--dry-run", action="store_true",
                   help="CPU-only preflight: the plan record, nothing measured")
    r.add_argument("--gpu-fixture", type=Path, default=None,
                   help="GPU facts JSON for --dry-run")
    r.add_argument("--work-dir", type=Path, default=Path("_wb19_work"),
                   help="where the leg's base init, corpus and checkpoint go (pod)")
    r.add_argument("--corpus-dir", type=Path, default=None,
                   help="a materialised shards corpus to train from instead of generating "
                        "one from scripts/example_generator")
    c = sub.add_parser("combine", help="assemble run records into evidence/wb19.json")
    c.add_argument("--chain-toml", type=Path, required=True)
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--candidates", type=Path, default=None)
    c.add_argument("--table-out", type=Path, default=None)
    c.add_argument("--note", action="append", default=None, metavar="SKU=TEXT",
                   help="annotate a measured row (e.g. a superseded path); repeatable")
    return p


def main(argv: Sequence[str] | None = None, *, env=None, fs_root="/") -> int:
    """``env`` replaces ``os.environ``; ``fs_root`` is where the prod markers
    are looked for. The CLI uses the real environment and ``/``."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    env = dict(os.environ if env is None else env)
    try:
        if args.cmd == "run":
            return _cmd_run(args, env, fs_root)
        return _cmd_combine(args, parser)
    except MeasureRefused as e:
        print(f"REFUSED {e.code}: {e.message}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
