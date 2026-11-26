"""``cascade-laned elastic``: elastic hosts for lane D (plan R13, W6d; owner 2026-11-24).

Two roles, each rented only while it has work:

* ``pool-d-builder`` -- the CPU ``pool_d`` builder (plan §1.7 "Build host",
  W6d: one snapshot every 3 h, each build < 3 h). **Shadeform, CPU only**:
  the client refuses any instance type with ``num_gpus > 0`` (owner
  2026-11-24: "GPU is ALWAYS Lium; Shadeform is CPU ONLY"). One VM comes up
  per 3-h build and is deleted after publish unless ``keep_warm``.
* ``lane-d-eval-host`` -- the lane D scoring GPU(s), **Lium only**, through
  the existing Lium adapter (:class:`cascade.provision.core.LiumProvider`:
  the digest-pinned worker image, quarantine of known-bad hosts, lemon
  recording). No GPU is ever rented on Shadeform; when Lium has no
  qualified GPU the controller waits and retries on Lium for
  ``retry_window_minutes`` and alerts. ``ceil(entries / per_host_entries)``
  hosts (capped by ``max_hosts`` and the spend caps) come up for a
  settlement with lane D work and go down when its scoring is done.

Spend guards (both roles): a hard ``max_price_per_hour`` and a
``monthly_cap_usd`` over this controller's own append-only spend ledger;
an offer above either is never rented; every failure after a create
deletes the instance (``try/finally``) and the delete is verified.
``--dry-run`` prints the chosen type and price and rents nothing.

Builder state lives in the private ``pool_d`` bucket (``[lane_d_ops]
pool_d_bucket``) under ``<pool_d_prefix>/_builder/``: the W6fs first-seen log
(``records.jsonl``, ``roots.jsonl``) and each published snapshot's
``metadata.json`` (the next build's ``--prior-dir`` frontier input). A
Shadeform volume would pin every build to one cloud and region, defeating the
cheapest-available pick, so object storage it is. ``state-push`` refuses
unless the bucket copy is a byte prefix of the local log (append-only) and
the bucket head is still the one ``state-pull`` saw (no concurrent builder).
Both stream (the log is 20 GB and growing; a single PUT that large is refused
``EntityTooLarge``): uploads go multipart above one part, the prefix check
hashes the local file against HEAD's sha256 instead of downloading the bucket
copy, unchanged files are not re-uploaded, and ``HEAD.json`` records each
file's sha256 and size so a restore verifies what it streamed down.

Multi-host scoring (owner 2026-11-24): :func:`assign_shards` splits a
cohort's entries over the hosts deterministically from the boundary block
hash and the entry ids (anyone re-derives it), and puts KING_D and the house
on EVERY host, so each challenger is paired with the king scored on its own
host in the same run (B6). The signed-report fields and the per-host pairing
inside the verdict are consensus code; they extend F5 (PR #830) and the plan,
not this operator tool (see docs/OPERATOR.md).

Credentials: env names only (``SHADEFORM_API_KEY``, ``SHADEFORM_SSH_KEY_ID``,
``LIUM_API_KEY`` / the Lium CLI's own config, ``HIPPIUS_S3_*``,
``TSFORGE_S3_*``). The builder env file is created by ``env-template`` with
names and empty values at mode 600; the owner fills it; this tool copies it to
the VM (mode 600) and never reads, prints or logs its values.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

SCHEMA = "laned_elastic/1"
STATE_SCHEMA = "laned_elastic_state/1"
PLAN_SCHEMA = "laned_eval_shards/1"
ROLES = ("pool-d-builder", "lane-d-eval-host")
BUILDER, EVAL_HOST = ROLES
BUILDER_PROVIDER = "shadeform"
EVAL_PROVIDER = "lium"
NAME_PREFIX = {BUILDER: "cascade-laned-pool-d", EVAL_HOST: "cascade-laned-eval"}
STATE_SUBDIR = "_builder"
DEFAULT_CONFIG = Path("deploy/laned/elastic.toml")

# Builder env file: the key NAMES the owner fills (values never touched here).
BUILDER_ENV_NAMES = ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY",
                     "TSFORGE_S3_ACCESS_KEY", "TSFORGE_S3_SECRET_KEY",
                     "TSFORGE_BUCKET", "TSFORGE_S3_ENDPOINT")

EXIT = {
    "ok": 0, "usage": 2, "no_capacity": 3, "price_cap": 4, "monthly_cap": 5,
    "gpu_refused": 6, "cohort_open": 7, "not_qualified": 8, "state_conflict": 9,
    "step_failed": 10, "teardown_unverified": 11, "host_mismatch": 12,
}


class ElasticError(RuntimeError):
    """A typed refusal; ``code`` is a key of :data:`EXIT`."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ── config ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuilderConfig:
    provider: str = BUILDER_PROVIDER
    min_vcpus: int = 8
    min_memory_gb: float = 32.0
    min_disk_gb: float = 150.0
    max_price_per_hour: float = 0.50
    monthly_cap_usd: float = 350.0
    build_hours: float = 3.0            # the expected bill per build (W6d wall < 3 h)
    ttl_hours: float = 3.5              # provider-side auto-delete backstop
    keep_warm: bool = False
    # Preferred types, cheapest-available first; [] = any CPU type meeting the spec.
    types: tuple[tuple[str, str], ...] = (("massedcompute", "cpu_medium"),
                                          ("verda", "cpu_16"),
                                          ("massedcompute", "cpu_small"))
    api_key_env: str = "SHADEFORM_API_KEY"
    ssh_key_id_env: str = "SHADEFORM_SSH_KEY_ID"
    ssh_user: str = "shadeform"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    workdir: str = "/home/shadeform/cascade"
    forge_dir: str = "/home/shadeform/tsforge"
    work_root: str = "/home/shadeform/pool_d_work"
    env_file: str = "/etc/cascade-laned-pool-d.env"     # on the box; owner fills
    remote_env_file: str = "/home/shadeform/.cascade-pool-d.env"
    release_dir: str = "/root/cascade"                    # the release the box runs
    chain_toml: str = "/root/cascade/chain.toml"          # the DEPLOYED chain.toml
    build_args: tuple[str, ...] = ()                       # extra pool_d_build flags


@dataclass(frozen=True)
class EvalCandidate:
    lium_gpu_type: str        # Lium ``gpu_type`` name
    gpu_model: str            # exact ``nvidia-smi --query-gpu=name`` it presents
    vram_gb: float
    profile: str = ""         # R12 profile (scripts/evalhost/profiles/*.toml)


@dataclass(frozen=True)
class EvalConfig:
    provider: str = EVAL_PROVIDER
    min_vram_gb: float = 96.0
    max_price_per_hour: float = 8.0
    monthly_cap_usd: float = 600.0
    settlement_hours: float = 3.0
    ttl_hours: float = 4.0
    per_host_entries: int = 2           # from A9 timing; owner sets it after A9
    max_hosts: int = 4
    retry_window_minutes: float = 180.0  # Lium-first: retry Lium this long, alert, never Shadeform
    retry_interval_seconds: float = 120.0
    boot_timeout_seconds: float = 1200.0
    image: str = ""                      # digest-pinned worker image (deployed provision.toml)
    ssh_pubkey_path: str = "~/.ssh/lium_cascade_ed25519.pub"
    ssh_key_path: str = "~/.ssh/lium_cascade_ed25519"
    ssh_user: str = "root"
    scoring_envs: str = "deploy/laned/scoring_envs.toml"
    # The public-mirror warm cache (owner 2026-11-24, #894): Lium persistent
    # volume HUIDs, host i of a cohort gets volume i (`lium up --volume
    # id:<HUID>`), mounted at public_cache_mount, where
    # deploy/laned/public_mirror.toml's cache_dir lives. A host past the list
    # scores from Hippius with its cache on the container disk (alerted).
    public_cache_volumes: tuple[str, ...] = ()
    public_cache_mount: str = "/workspace"
    # Used ONLY while scoring_envs is absent (F5, PR #830, not merged yet).
    fallback_qualified_gpu_models: tuple[str, ...] = ("NVIDIA H200",)
    candidates: tuple[EvalCandidate, ...] = (
        EvalCandidate("H200", "NVIDIA H200", 141.0,
                      "scripts/evalhost/profiles/elastic_h200.toml"),
        EvalCandidate("Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 96.0,
                      "scripts/evalhost/profiles/elastic_rtxpro6000.toml"),
    )


@dataclass(frozen=True)
class ElasticConfig:
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    eval_host: EvalConfig = field(default_factory=EvalConfig)
    state_dir: str = "~/.local/state/cascade-laned-elastic"


_SECRETISH = ("_secret", "_token", "_password", "api_key")


def _refuse_secrets(table: Mapping, where: str) -> None:
    for k, v in table.items():
        lk = str(k).lower()
        if lk.endswith("_env") or lk.endswith("_path") or lk.endswith("_file"):
            continue
        if any(s in lk for s in _SECRETISH):
            raise ElasticError("usage", f"[{where}] secret-shaped key {k!r} refused: "
                                        f"credentials are env-only (value not shown)")
        if isinstance(v, Mapping):
            _refuse_secrets(v, f"{where}.{k}")


def _num(table: Mapping, key: str, default, *, lo: float = 0.0, integer: bool = False):
    v = table.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo:
        raise ElasticError("usage", f"{key} must be a number >= {lo}; got {v!r}")
    if integer:
        if int(v) != v:
            raise ElasticError("usage", f"{key} must be an integer; got {v!r}")
        return int(v)
    return float(v)


def _builder_from(t: Mapping) -> BuilderConfig:
    d = BuilderConfig()
    known = set(BuilderConfig.__dataclass_fields__)
    unknown = sorted(set(t) - known)
    if unknown:
        raise ElasticError("usage", f"[pool_d_builder] unknown key(s): {unknown}")
    if t.get("provider", d.provider) != BUILDER_PROVIDER:
        raise ElasticError("usage", "[pool_d_builder] provider must be 'shadeform' "
                                    "(Shadeform is CPU only; Lium has no CPU pods)")
    types = t.get("types")
    if types is None:
        types_t = d.types
    else:
        if not isinstance(types, list):
            raise ElasticError("usage", "[pool_d_builder] types must be a list of tables")
        types_t = tuple((str(x["cloud"]), str(x["shade_instance_type"])) for x in types)
    out = replace(
        d,
        min_vcpus=_num(t, "min_vcpus", d.min_vcpus, lo=1, integer=True),
        min_memory_gb=_num(t, "min_memory_gb", d.min_memory_gb),
        min_disk_gb=_num(t, "min_disk_gb", d.min_disk_gb),
        max_price_per_hour=_num(t, "max_price_per_hour", d.max_price_per_hour),
        monthly_cap_usd=_num(t, "monthly_cap_usd", d.monthly_cap_usd),
        build_hours=_num(t, "build_hours", d.build_hours),
        ttl_hours=_num(t, "ttl_hours", d.ttl_hours),
        keep_warm=bool(t.get("keep_warm", d.keep_warm)),
        types=types_t,
        build_args=tuple(str(x) for x in t.get("build_args", d.build_args)),
        **{k: str(t.get(k, getattr(d, k))) for k in (
            "api_key_env", "ssh_key_id_env", "ssh_user", "ssh_key_path", "workdir",
            "forge_dir", "work_root", "env_file", "remote_env_file", "release_dir",
            "chain_toml")},
    )
    if out.max_price_per_hour <= 0 or out.monthly_cap_usd <= 0:
        raise ElasticError("usage", "[pool_d_builder] max_price_per_hour and monthly_cap_usd "
                                    "must be > 0 (a spend cap is never off)")
    if out.min_disk_gb < 150:
        raise ElasticError("usage", "[pool_d_builder] min_disk_gb must be >= 150 (plan R13)")
    if not 0 < out.build_hours < 3.0 + 1e-9:
        raise ElasticError("usage", "[pool_d_builder] build_hours must be in (0, 3] "
                                    "(W6d: a build finishes inside its 3-h period)")
    return out


_VOLUME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _volumes(v) -> tuple[str, ...]:
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) and _VOLUME_RE.match(x)
                                                   for x in v):
        raise ElasticError("usage", "[eval_host] public_cache_volumes must be a list of Lium "
                                    f"volume HUIDs; got {v!r}")
    if len(set(v)) != len(v):
        raise ElasticError("usage", "[eval_host] public_cache_volumes repeats a volume "
                                    "(one host per volume)")
    return tuple(v)


def _eval_from(t: Mapping) -> EvalConfig:
    d = EvalConfig()
    known = set(EvalConfig.__dataclass_fields__)
    unknown = sorted(set(t) - known)
    if unknown:
        raise ElasticError("usage", f"[eval_host] unknown key(s): {unknown}")
    if t.get("provider", d.provider) != EVAL_PROVIDER:
        raise ElasticError("usage", "[eval_host] provider must be 'lium' "
                                    "(GPU is always Lium; Shadeform is CPU only)")
    cands = t.get("candidates")
    if cands is None:
        cands_t = d.candidates
    else:
        cands_t = tuple(EvalCandidate(str(c["lium_gpu_type"]), str(c["gpu_model"]),
                                      float(c["vram_gb"]), str(c.get("profile", "")))
                        for c in cands)
    out = replace(
        d,
        min_vram_gb=_num(t, "min_vram_gb", d.min_vram_gb),
        max_price_per_hour=_num(t, "max_price_per_hour", d.max_price_per_hour),
        monthly_cap_usd=_num(t, "monthly_cap_usd", d.monthly_cap_usd),
        settlement_hours=_num(t, "settlement_hours", d.settlement_hours),
        ttl_hours=_num(t, "ttl_hours", d.ttl_hours),
        per_host_entries=_num(t, "per_host_entries", d.per_host_entries, lo=1, integer=True),
        max_hosts=_num(t, "max_hosts", d.max_hosts, lo=1, integer=True),
        retry_window_minutes=_num(t, "retry_window_minutes", d.retry_window_minutes),
        retry_interval_seconds=_num(t, "retry_interval_seconds", d.retry_interval_seconds),
        boot_timeout_seconds=_num(t, "boot_timeout_seconds", d.boot_timeout_seconds),
        fallback_qualified_gpu_models=tuple(
            str(x) for x in t.get("fallback_qualified_gpu_models",
                                  d.fallback_qualified_gpu_models)),
        candidates=cands_t,
        public_cache_volumes=_volumes(t.get("public_cache_volumes", d.public_cache_volumes)),
        **{k: str(t.get(k, getattr(d, k))) for k in (
            "image", "ssh_pubkey_path", "ssh_key_path", "ssh_user", "scoring_envs",
            "public_cache_mount")},
    )
    if not out.public_cache_mount.startswith("/"):
        raise ElasticError("usage", "[eval_host] public_cache_mount must be an absolute path")
    if out.max_price_per_hour <= 0 or out.monthly_cap_usd <= 0:
        raise ElasticError("usage", "[eval_host] max_price_per_hour and monthly_cap_usd "
                                    "must be > 0 (a spend cap is never off)")
    low = [c.lium_gpu_type for c in out.candidates if c.vram_gb < out.min_vram_gb]
    if low:
        raise ElasticError("usage", f"[eval_host] candidates below min_vram_gb "
                                    f"{out.min_vram_gb:g}: {low}")
    return out


def load_config(path: Path | str | None) -> ElasticConfig:
    if path is None:
        return ElasticConfig()
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ElasticError("usage", f"cannot read {p}: {e}") from None
    if raw.get("schema") != SCHEMA:
        raise ElasticError("usage", f"{p}: schema must be {SCHEMA!r}")
    _refuse_secrets(raw, p.name)
    unknown = sorted(set(raw) - {"schema", "state_dir", "pool_d_builder", "eval_host"})
    if unknown:
        raise ElasticError("usage", f"{p}: unknown table(s)/key(s): {unknown}")
    return ElasticConfig(builder=_builder_from(raw.get("pool_d_builder") or {}),
                         eval_host=_eval_from(raw.get("eval_host") or {}),
                         state_dir=str(raw.get("state_dir", ElasticConfig.state_dir)))


# ── qualified scoring environments (B5; F5's list when present) ────────────


def qualified_gpu_models(cfg: EvalConfig, *, root: Path | None = None) -> tuple[set[str], str]:
    """The GPU models a qualified scoring environment allows, and the source.

    F5's ``deploy/laned/scoring_envs.toml`` is authoritative when present:
    only ``status = "qualified"`` entries count (a ``candidate`` is listed but
    refused). Without it (F5 not merged) the config's
    ``fallback_qualified_gpu_models`` apply. The validator's handshake
    re-checks the booted host against the list at the boundary."""
    path = Path(cfg.scoring_envs)
    if root is not None and not path.is_absolute():
        path = root / path
    if path.is_file():
        try:
            doc = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ElasticError("usage", f"{path}: {e}") from None
        models = {str(m) for env in doc.get("env") or []
                  if env.get("status") == "qualified" for m in env.get("gpu_models") or []}
        return models, str(path)
    return set(cfg.fallback_qualified_gpu_models), "fallback_qualified_gpu_models"


def qualified_candidates(cfg: EvalConfig, *, root: Path | None = None
                         ) -> tuple[list[EvalCandidate], list[dict]]:
    models, _src = qualified_gpu_models(cfg, root=root)
    keep, refused = [], []
    for c in cfg.candidates:
        if c.vram_gb < cfg.min_vram_gb:
            refused.append({"gpu_type": c.lium_gpu_type, "reason": "vram_low"})
        elif c.gpu_model not in models:
            refused.append({"gpu_type": c.lium_gpu_type, "reason": "not_qualified"})
        else:
            keep.append(c)
    return keep, refused


# ── spend ledger and guards ────────────────────────────────────────────────


def month_start(ts: float) -> int:
    t = time.gmtime(int(ts))
    return calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))


class SpendLedger:
    """Append-only JSON lines: one ``open`` and one ``close`` row per instance."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def open(self, *, role: str, instance_id: str, provider: str, type_id: str,
             price_per_hour: float, ts: float) -> None:
        self._append({"event": "open", "role": role, "id": instance_id, "provider": provider,
                      "type": type_id, "price_per_hour": float(price_per_hour), "ts": int(ts)})

    def close(self, *, role: str, instance_id: str, ts: float) -> None:
        self._append({"event": "close", "role": role, "id": instance_id, "ts": int(ts)})

    def spent(self, role: str, now: float) -> float:
        """USD billed this UTC month (open instances accrue to ``now``)."""
        m0 = month_start(now)
        opened: dict[str, dict] = {}
        closed: dict[str, int] = {}
        for r in self.rows():
            if r.get("role") != role:
                continue
            if r["event"] == "open":
                opened[r["id"]] = r
            elif r["event"] == "close":
                closed[r["id"]] = int(r["ts"])
        total = 0.0
        for iid, r in opened.items():
            start = max(int(r["ts"]), m0)
            end = closed.get(iid, int(now))
            if end > start:
                total += float(r["price_per_hour"]) * (end - start) / 3600.0
        return total


def check_spend(*, price: float, cap_per_hour: float, ledger: SpendLedger, role: str,
                hours: float, monthly_cap: float, count: int, now: float) -> None:
    """Refuse (``price_cap`` / ``monthly_cap``) before anything is rented."""
    if not (math.isfinite(price) and 0 < price <= cap_per_hour):
        raise ElasticError("price_cap", f"${price:.2f}/h exceeds the ${cap_per_hour:.2f}/h cap")
    spent = ledger.spent(role, now)
    want = price * hours * count
    if spent + want > monthly_cap + 1e-9:
        raise ElasticError("monthly_cap", f"${spent:.2f} spent this month + ${want:.2f} "
                                          f"projected > ${monthly_cap:.2f} cap")


# ── Shadeform: CPU only (the pool_d builder) ───────────────────────────────


@dataclass(frozen=True)
class CpuOffer:
    cloud: str
    region: str
    type_id: str
    vcpus: int
    memory_gb: float
    disk_gb: float
    price_per_hour: float
    num_gpus: int = 0


def shadeform_cpu_offers(types_json: Mapping, cfg: BuilderConfig) -> list[CpuOffer]:
    """Available on-demand Shadeform CPU types meeting the builder spec,
    cheapest first (allowlist order breaks price ties). A type with
    ``num_gpus > 0`` is never an offer (CPU only), at any price."""
    allow = list(cfg.types)
    out: list[tuple[float, int, CpuOffer]] = []
    for t in types_json.get("instance_types", []) or []:
        conf = t.get("configuration") or {}
        try:
            gpus = int(conf.get("num_gpus", 0) or 0)
            price = int(t["hourly_price"]) / 100.0      # Shadeform reports CENTS
            vcpus = int(conf.get("vcpus", 0) or 0)
            mem = float(conf.get("memory_in_gb", 0) or 0)
            disk = float(conf.get("storage_in_gb", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if gpus > 0:
            continue
        key = (str(t.get("cloud")), str(t.get("shade_instance_type")))
        if allow and key not in allow:
            continue
        if (vcpus < cfg.min_vcpus or mem < cfg.min_memory_gb or disk < cfg.min_disk_gb
                or not 0 < price <= cfg.max_price_per_hour):
            continue
        region = next((a.get("region") for a in t.get("availability") or []
                       if a.get("available") and a.get("rental_type", "on_demand") == "on_demand"
                       and a.get("region")), None)
        if not region:
            continue
        out.append((price, allow.index(key) if allow else 0,
                    CpuOffer(key[0], str(region), key[1], vcpus, mem, disk, price, gpus)))
    out.sort(key=lambda x: (x[0], x[1]))
    return [o for *_, o in out]


def _shadeform_base():
    from ..provision.core import ShadeformProvider

    return ShadeformProvider


class ShadeformCpuOnly:
    """Shadeform for the ``pool_d`` builder: VM mode, CPU types only.

    Wraps :class:`cascade.provision.core.ShadeformProvider` (its HTTP session,
    ``wait_ready``, ``get_ip``, ``terminate``, ``pod_exists``). ``create``
    re-reads the type listing and refuses (``gpu_refused``) any type whose
    ``num_gpus > 0`` -- a GPU is never rented on Shadeform for lane D."""

    name = BUILDER_PROVIDER

    def __init__(self, cfg: BuilderConfig, *, inner=None):
        self.cfg = cfg
        if inner is None:
            key_id = os.environ.get(cfg.ssh_key_id_env, "").strip()
            inner = _shadeform_base()(api_key_env=cfg.api_key_env, ssh_key_id=key_id)
        self.inner = inner

    def types(self) -> dict:
        return self.inner._get("/instances/types", {"available": "true"})

    def offers(self) -> list[CpuOffer]:
        return shadeform_cpu_offers(self.types(), self.cfg)

    def create(self, offer: CpuOffer, *, name: str, ttl_seconds: float,
               spend_threshold_usd: float) -> str:
        if int(offer.num_gpus) > 0:
            raise ElasticError("gpu_refused", f"{offer.cloud}/{offer.type_id} has "
                                              f"{offer.num_gpus} GPU(s): Shadeform is CPU only")
        listed = [t for t in self.types().get("instance_types", []) or []
                  if (str(t.get("cloud")), str(t.get("shade_instance_type")))
                  == (offer.cloud, offer.type_id)]
        if not listed:
            raise ElasticError("no_capacity", f"{offer.cloud}/{offer.type_id} no longer listed")
        if any(int((t.get("configuration") or {}).get("num_gpus", 0) or 0) > 0 for t in listed):
            raise ElasticError("gpu_refused", f"{offer.cloud}/{offer.type_id} lists GPUs: "
                                              f"Shadeform is CPU only")
        key_id = getattr(self.inner, "ssh_key_id", "")
        if not key_id:
            raise ElasticError("usage", f"set ${self.cfg.ssh_key_id_env} (VM mode; the builder "
                                        f"never boots a docker image)")
        from ..provision.core import shadeform_delete_at

        body = {"cloud": offer.cloud, "region": offer.region,
                "shade_instance_type": offer.type_id, "shade_cloud": True, "name": name,
                "ssh_key_id": key_id, "auto_delete": {}}
        delete_at = shadeform_delete_at(ttl_seconds)
        if delete_at:
            body["auto_delete"]["date_threshold"] = delete_at
        thr = math.floor(float(spend_threshold_usd) * 100) / 100.0
        if thr >= 0.01:
            body["auto_delete"]["spend_threshold"] = f"{thr:.2f}"
        resp = self.inner._post("/instances/create", body)
        iid = (resp or {}).get("id")
        if not iid:
            raise ElasticError("step_failed", f"shadeform create returned no id (name {name})")
        return str(iid)

    def wait_ready(self, iid: str, *, timeout: float) -> bool:
        return self.inner.wait_ready(iid, timeout=timeout)

    def address(self, iid: str):
        return self.inner.get_ip(iid)

    def terminate(self, iid: str) -> None:
        self.inner.terminate(iid)

    def gone(self, iid: str) -> bool:
        return not self.inner.pod_exists(iid)


# ── Lium: the eval host GPU(s) ─────────────────────────────────────────────


@dataclass(frozen=True)
class GpuOffer:
    gpu_type: str
    executor_id: str
    price_per_hour: float
    candidate: EvalCandidate


def lium_gpu_offers(lium, cfg: EvalConfig, cands: Sequence[EvalCandidate],
                    exclude_ids: Sequence[str] = ()) -> list[GpuOffer]:
    """Qualified Lium executors under the $/h cap, cheapest first. Unpriced
    listings are dropped (a cap is never dodged by a missing field)."""
    from ..provision.core import executor_price_per_hour

    by_type = {c.lium_gpu_type: c for c in cands}
    if not by_type:
        return []
    out = []
    for sku, ex in lium.list_offers(tuple(by_type), gpus=1, exclude_ids=tuple(exclude_ids)):
        price = executor_price_per_hour(ex, gpu_count=1)
        if price is None or price > cfg.max_price_per_hour or sku not in by_type:
            continue
        out.append(GpuOffer(sku, str(ex.get("id")), float(price), by_type[sku]))
    out.sort(key=lambda o: (o.price_per_hour, o.gpu_type))
    return out


def make_lium(cfg: EvalConfig):
    from ..provision.core import LiumProvider

    return LiumProvider(max_price_per_hour=cfg.max_price_per_hour)


def hosts_needed(n_entries: int, cfg: EvalConfig) -> int:
    """``ceil(entries / per_host_entries)``, capped at ``max_hosts``; 0 = no work."""
    if n_entries <= 0:
        return 0
    return min(int(cfg.max_hosts), math.ceil(int(n_entries) / int(cfg.per_host_entries)))


def affordable_hosts(want: int, *, price: float, cfg: EvalConfig, ledger: SpendLedger,
                     now: float) -> int:
    left = cfg.monthly_cap_usd - ledger.spent(EVAL_HOST, now)
    per = price * cfg.settlement_hours
    return max(0, min(int(want), int(math.floor((left + 1e-9) / per)) if per > 0 else 0))


# ── deterministic cohort sharding (B6 extended to N hosts) ─────────────────


def assign_shards(block_hash: str, entry_ids: Sequence[str], n_hosts: int, *,
                  king: str, house: str | None = None) -> list[dict]:
    """Split the entries over ``n_hosts`` hosts; KING_D (and the house) on every host.

    Entries are ordered by ``sha256(block_hash ‖ "\\0" ‖ entry_id)`` and dealt
    round-robin, so the split depends only on the boundary's block hash and
    the entry ids (satellites re-derive it). Each challenger is then paired
    with the king forecast on its own host, in the same run (B6)."""
    if n_hosts < 1:
        raise ElasticError("usage", "n_hosts must be >= 1")
    ids = list(dict.fromkeys(str(e) for e in entry_ids))
    if len(ids) != len(entry_ids):
        raise ElasticError("usage", "duplicate entry id")
    bh = str(block_hash).lower().removeprefix("0x")
    order = sorted(ids, key=lambda e: (hashlib.sha256(f"{bh}\0{e}".encode()).hexdigest(), e))
    n = min(int(n_hosts), max(1, len(order)))
    shards = [{"shard": i, "entries": [], "anchors": [king] + ([house] if house else [])}
              for i in range(n)]
    for i, e in enumerate(order):
        shards[i % n]["entries"].append(e)
    return shards


def merge_shard_scores(shards: Sequence[Mapping], *, anchors: Sequence[str]) -> dict:
    """Pool per-host rows ``{"host", "rows": {model: {window_id: score}}}``.

    ``anchors`` (KING_D first, then the house) are scored on every host; each
    other model on exactly one host, on the same windows as that host's king.
    Returns ``{"anchors": {model: rows}, "members": {model: rows}, "host_of":
    {model: host}, "anchor_by_host": {model: {host: rows}}}`` -- the per-host
    king scores the signed report would record. When an anchor's rows differ
    between hosts, the per-host paired verdict is consensus work (F5 / a plan
    rev) and this refuses ``host_mismatch`` instead of guessing."""
    if not anchors:
        raise ElasticError("usage", "anchors must name KING_D")
    king = anchors[0]
    by_host: dict[str, dict[str, dict]] = {a: {} for a in anchors}
    members: dict[str, dict] = {}
    host_of: dict[str, str] = {}
    for sh in shards:
        host = str(sh["host"])
        rows = sh["rows"]
        for a in anchors:
            if a not in rows:
                raise ElasticError("host_mismatch", f"host {host} did not score anchor {a}")
            by_host[a][host] = dict(rows[a])
        for model, r in rows.items():
            if model in anchors:
                continue
            if model in members:
                raise ElasticError("host_mismatch", f"{model} scored on two hosts")
            if set(r) != set(rows[king]):
                raise ElasticError("host_mismatch", f"{model} on {host}: windows differ "
                                                    f"from the king's")
            members[model] = dict(r)
            host_of[model] = host
    out = {}
    for a, hosts in by_host.items():
        first = next(iter(hosts.values()))
        if any(v != first for v in hosts.values()):
            raise ElasticError("host_mismatch", f"{a}'s scores differ between hosts: the "
                                                f"per-host paired verdict is not built yet (F5)")
        out[a] = first
    return {"anchors": out, "members": members, "host_of": host_of,
            "anchor_by_host": by_host}


# ── local session state ────────────────────────────────────────────────────


def _state_path(cfg: ElasticConfig, role: str) -> Path:
    return Path(cfg.state_dir).expanduser() / f"{role}.json"


def load_state(cfg: ElasticConfig, role: str) -> dict:
    p = _state_path(cfg, role)
    if not p.is_file():
        return {"schema": STATE_SCHEMA, "role": role, "instances": [], "cohort": None}
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(cfg: ElasticConfig, role: str, state: dict) -> None:
    p = _state_path(cfg, role)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def ledger_of(cfg: ElasticConfig) -> SpendLedger:
    return SpendLedger(Path(cfg.state_dir).expanduser() / "spend.jsonl")


def alert(cfg: ElasticConfig, code: str, detail: str, *, now: float) -> dict:
    row = {"alert": code, "detail": detail, "ts": int(now)}
    p = Path(cfg.state_dir).expanduser() / "alerts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"ALERT": row}, sort_keys=True), file=sys.stderr)
    return row


# ── up / down ──────────────────────────────────────────────────────────────


@dataclass
class Clock:
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


def _teardown(provider, iid: str, *, ledger: SpendLedger, role: str, clock: Clock,
              tries: int = 3) -> bool:
    """Delete and VERIFY; the ledger row closes only once the instance is gone."""
    for _ in range(tries):
        with contextlib.suppress(Exception):    # verified below, retried
            provider.terminate(iid)
        try:
            if provider.gone(iid):
                ledger.close(role=role, instance_id=iid, ts=clock.now())
                return True
        except Exception:  # noqa: BLE001 -- treat an unreadable listing as not gone
            pass
        clock.sleep(5)
    return False


def builder_up(cfg: ElasticConfig, client: ShadeformCpuOnly, *, dry_run: bool,
               clock: Clock | None = None) -> dict:
    clock = clock or Clock()
    b = cfg.builder
    ledger = ledger_of(cfg)
    state = load_state(cfg, BUILDER)
    live = [i for i in state["instances"] if not i.get("closed")]
    if live:
        return {"role": BUILDER, "reused": True, "instance": live[0]}
    offers = client.offers()
    if not offers:
        raise ElasticError("no_capacity", "no available Shadeform CPU type meets the builder "
                                          "spec under the $/h cap")
    offer = offers[0]
    now = clock.now()
    check_spend(price=offer.price_per_hour, cap_per_hour=b.max_price_per_hour, ledger=ledger,
                role=BUILDER, hours=b.build_hours, monthly_cap=b.monthly_cap_usd, count=1,
                now=now)
    plan = {"role": BUILDER, "provider": BUILDER_PROVIDER, "cloud": offer.cloud,
            "region": offer.region, "type": offer.type_id, "vcpus": offer.vcpus,
            "memory_gb": offer.memory_gb, "disk_gb": offer.disk_gb,
            "price_per_hour": offer.price_per_hour,
            "projected_usd": round(offer.price_per_hour * b.build_hours, 4),
            "month_spent_usd": round(ledger.spent(BUILDER, now), 4)}
    if dry_run:
        return {**plan, "dry_run": True}
    name = f"{NAME_PREFIX[BUILDER]}-{int(now)}"
    left = b.monthly_cap_usd - ledger.spent(BUILDER, now)
    iid = client.create(offer, name=name, ttl_seconds=b.ttl_hours * 3600,
                        spend_threshold_usd=min(left, offer.price_per_hour * b.ttl_hours))
    ledger.open(role=BUILDER, instance_id=iid, provider=BUILDER_PROVIDER, type_id=offer.type_id,
                price_per_hour=offer.price_per_hour, ts=now)
    ok = False
    try:
        if not client.wait_ready(iid, timeout=1800):
            raise ElasticError("no_capacity", f"{iid} never became active")
        addr = client.address(iid)
        if addr is None:
            raise ElasticError("no_capacity", f"{iid} has no address")
        inst = {**plan, "id": iid, "name": name, "ip": addr.ip, "port": int(addr.ssh_port),
                "user": b.ssh_user, "started_ts": int(now)}
        state["instances"] = [inst]
        save_state(cfg, BUILDER, state)
        ok = True
        return {"role": BUILDER, "reused": False, "instance": inst}
    finally:
        if not ok and not _teardown(client, iid, ledger=ledger, role=BUILDER, clock=clock):
            alert(cfg, "teardown_unverified", f"builder {iid}", now=clock.now())


def eval_up(cfg: ElasticConfig, lium, *, entries: int, cohort: str, dry_run: bool,
            clock: Clock | None = None, root: Path | None = None) -> dict:
    """Rent ``hosts_needed(entries)`` qualified Lium GPUs for one cohort.

    Same-host rule (B6): a cohort's hosts are fixed for its run. ``up`` for a
    cohort that already has live hosts reuses them; a different cohort while
    one is open refuses ``cohort_open``. Lium only: with no qualified offer it
    retries for ``retry_window_minutes``, alerts, and refuses ``no_capacity``."""
    from ..provision.core import LaunchSpec

    clock = clock or Clock()
    e = cfg.eval_host
    ledger = ledger_of(cfg)
    state = load_state(cfg, EVAL_HOST)
    live = [i for i in state["instances"] if not i.get("closed")]
    if state.get("cohort") and live:
        if state["cohort"] != cohort:
            raise ElasticError("cohort_open", f"cohort {state['cohort']} still has "
                                              f"{len(live)} host(s) up; run down first")
        return {"role": EVAL_HOST, "reused": True, "cohort": cohort, "instances": live}
    want = hosts_needed(entries, e)
    if want == 0:
        return {"role": EVAL_HOST, "cohort": cohort, "hosts": 0, "reason": "no_lane_d_work"}
    cands, refused = qualified_candidates(e, root=root)
    if not cands:
        raise ElasticError("not_qualified", f"no candidate GPU runs a qualified scoring "
                                            f"environment: {refused}")
    if not dry_run and "@sha256:" not in e.image:
        raise ElasticError("usage", "[eval_host] image must be the digest-pinned worker "
                                    "image (source it from the deployed provision.toml)")
    deadline = clock.now() + e.retry_window_minutes * 60
    alerted = False
    while True:
        offers = lium_gpu_offers(lium, e, cands)
        if offers:
            break
        if not alerted:
            alert(cfg, "lane_d_eval_no_lium_gpu",
                  f"no qualified Lium GPU <= ${e.max_price_per_hour:.2f}/h for cohort "
                  f"{cohort}; retrying Lium for {e.retry_window_minutes:g} min (never Shadeform)",
                  now=clock.now())
            alerted = True
        if dry_run or clock.now() + e.retry_interval_seconds > deadline:
            raise ElasticError("no_capacity", "no qualified Lium GPU inside the retry window")
        clock.sleep(e.retry_interval_seconds)
    now = clock.now()
    price = offers[0].price_per_hour
    check_spend(price=price, cap_per_hour=e.max_price_per_hour, ledger=ledger, role=EVAL_HOST,
                hours=e.settlement_hours, monthly_cap=e.monthly_cap_usd, count=1, now=now)
    # Budget n hosts at the dearest of the n cheapest offers (>= 1: the single
    # host already passed check_spend above).
    worst = offers[min(want, len(offers)) - 1].price_per_hour
    n = max(1, affordable_hosts(want, price=worst, cfg=e, ledger=ledger, now=now))
    if n < want:
        alert(cfg, "lane_d_eval_hosts_capped", f"cohort {cohort}: {n} of {want} hosts "
                                               f"within the monthly cap", now=now)
    vols = e.public_cache_volumes
    if n > len(vols):
        alert(cfg, "lane_d_eval_no_volume", f"cohort {cohort}: {n - len(vols)} of {n} host(s) "
              f"without a public-cache Lium volume ([eval_host] public_cache_volumes); they "
              f"fetch every drawn series from Hippius", now=now)
    plan = {"role": EVAL_HOST, "provider": EVAL_PROVIDER, "cohort": cohort, "entries": entries,
            "hosts_wanted": want, "hosts": n,
            "volumes": [vols[i] if i < len(vols) else None for i in range(n)],
            "public_cache_mount": e.public_cache_mount,
            "offers": [{"gpu_type": o.gpu_type, "price_per_hour": o.price_per_hour}
                       for o in offers[:n]], "refused_types": refused,
            "month_spent_usd": round(ledger.spent(EVAL_HOST, now), 4)}
    if dry_run:
        return {**plan, "dry_run": True}
    pub = Path(e.ssh_pubkey_path).expanduser().read_text(encoding="utf-8").strip()
    rented: list[dict] = []
    claimed: list[str] = []
    ok = False
    try:
        for i in range(n):
            offers = lium_gpu_offers(lium, e, cands, exclude_ids=claimed)
            if not offers:
                raise ElasticError("no_capacity", f"Lium capacity gone after {i} host(s)")
            o = offers[0]
            check_spend(price=o.price_per_hour, cap_per_hour=e.max_price_per_hour,
                        ledger=ledger, role=EVAL_HOST, hours=e.settlement_hours,
                        monthly_cap=e.monthly_cap_usd, count=1, now=clock.now())
            name = f"{NAME_PREFIX[EVAL_HOST]}-{cohort}-{i}"
            spec = LaunchSpec(sku=o.gpu_type, count=1, image=e.image, ssh_pubkey=pub,
                              name_prefix=name, sku_choices=(o.gpu_type,),
                              exclude_ids=tuple(claimed), ttl_seconds=e.ttl_hours * 3600,
                              volumes=(vols[i],) if i < len(vols) else ())
            pod = lium.launch(spec)[0]
            landed = lium.price_of(pod)
            ledger.open(role=EVAL_HOST, instance_id=pod, provider=EVAL_PROVIDER,
                        type_id=o.gpu_type, price_per_hour=float(landed or o.price_per_hour),
                        ts=clock.now())
            rented.append({"id": pod, "gpu_type": o.gpu_type, "gpu_model": o.candidate.gpu_model,
                           "profile": o.candidate.profile,
                           "volume": vols[i] if i < len(vols) else None,
                           "price_per_hour": float(landed or o.price_per_hour)})
            claimed.append(o.executor_id)
            if landed is not None and landed > e.max_price_per_hour:
                raise ElasticError("price_cap", f"{pod} landed at ${landed:.2f}/h")
            if not lium.wait_ready(pod, timeout=e.boot_timeout_seconds):
                lium.record_executor_fault(pod, "lane_d_eval_boot_timeout")
                raise ElasticError("no_capacity", f"{pod} never became ready (lemon recorded)")
            addr = lium.get_ip(pod)
            if addr is None:
                lium.record_executor_fault(pod, "lane_d_eval_no_ports")
                raise ElasticError("no_capacity", f"{pod} RUNNING without ports (lemon)")
            rented[-1].update(ip=addr.ip, port=int(addr.ssh_port), user=e.ssh_user,
                              shard=i, started_ts=int(clock.now()))
        state.update(instances=rented, cohort=cohort)
        save_state(cfg, EVAL_HOST, state)
        ok = True
        return {**plan, "instances": rented}
    finally:
        if not ok:
            prov = _LiumTeardown(lium)
            for r in rented:
                if not _teardown(prov, r["id"], ledger=ledger, role=EVAL_HOST, clock=clock):
                    alert(cfg, "teardown_unverified", f"eval host {r['id']}", now=clock.now())


class _LiumTeardown:
    def __init__(self, lium):
        self.lium = lium

    def terminate(self, iid: str) -> None:
        self.lium.terminate(iid)

    def gone(self, iid: str) -> bool:
        exists = getattr(self.lium, "pod_exists", None)
        if exists is not None:
            return not exists(iid)
        return iid not in self.lium.list_tagged(iid)


def down(cfg: ElasticConfig, role: str, provider, *, cohort_done: bool = False,
         dry_run: bool = False, clock: Clock | None = None) -> dict:
    """Delete every live instance of ``role`` and verify each is gone.

    Eval host: refuses ``cohort_open`` unless ``cohort_done`` (B6: a cohort's
    hosts stay up until its scoring is done). Builder: honours ``keep_warm``
    only through :func:`cycle`; an explicit ``down`` always deletes."""
    clock = clock or Clock()
    state = load_state(cfg, role)
    live = [i for i in state["instances"] if not i.get("closed")]
    if role == EVAL_HOST and live and state.get("cohort") and not cohort_done:
        raise ElasticError("cohort_open", f"cohort {state['cohort']} is still scoring; pass "
                                          f"--cohort-done once its settlement has decided")
    if dry_run:
        return {"role": role, "dry_run": True, "would_delete": [i["id"] for i in live]}
    prov = provider if role == BUILDER else _LiumTeardown(provider)
    ledger = ledger_of(cfg)
    failed = []
    for inst in live:
        if _teardown(prov, inst["id"], ledger=ledger, role=role, clock=clock):
            inst["closed"] = int(clock.now())
        else:
            failed.append(inst["id"])
    if not failed and role == EVAL_HOST:
        state["cohort"] = None
    save_state(cfg, role, state)
    if failed:
        alert(cfg, "teardown_unverified", f"{role}: {failed}", now=clock.now())
        raise ElasticError("teardown_unverified", f"{failed} still listed after delete")
    return {"role": role, "deleted": [i["id"] for i in live]}


# ── remote steps (bootstrap / run) ─────────────────────────────────────────


def ssh_argv(inst: Mapping, *, key_path: str, known_hosts: str, remote: str) -> list[str]:
    """Pinned-host-key ssh (StrictHostKeyChecking=yes, per-host known_hosts)."""
    return ["ssh", "-p", str(inst["port"]), "-i", str(Path(key_path).expanduser()),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            f"{inst['user']}@{inst['ip']}", remote]


def scp_argv(inst: Mapping, *, key_path: str, known_hosts: str, src: str, dest: str) -> list[str]:
    return ["scp", "-P", str(inst["port"]), "-i", str(Path(key_path).expanduser()),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}", src,
            f"{inst['user']}@{inst['ip']}:{dest}"]


def known_hosts_for(cfg: ElasticConfig, inst: Mapping) -> str:
    return str(Path(cfg.state_dir).expanduser() / "known_hosts" / f"{inst['id']}")


def _evalhost_tool(release_dir: str) -> list[str]:
    return [sys.executable, "-P", f"{release_dir}/scripts/evalhost/evalhost.py"]


def pin_steps(cfg: ElasticConfig, inst: Mapping, release_dir: str) -> list[tuple[str, list[str]]]:
    """Scan the new host's key and pin it with R12's ``pin-host-key``."""
    kh = known_hosts_for(cfg, inst)
    scanned = kh + ".scan"
    return [
        ("keyscan", ["bash", "-c", f"ssh-keyscan -p {int(inst['port'])} "
                                   f"{shlex.quote(str(inst['ip']))} > {shlex.quote(scanned)}"]),
        ("pin_host_key", _evalhost_tool(release_dir) + [
            "pin-host-key", "--known-hosts", kh, "--host", str(inst["ip"]),
            "--port", str(inst["port"]), "--scanned", scanned]),
    ]


def builder_bootstrap_steps(cfg: ElasticConfig, inst: Mapping) -> list[tuple[str, list[str]]]:
    """Idempotent: the release + ``uv sync --frozen`` (deploy/pod_bootstrap.sh),
    the deployed chain.toml, the env file (mode 600), the state restore."""
    b = cfg.builder
    kh = known_hosts_for(cfg, inst)
    env = {"POD_IP": str(inst["ip"]), "POD_PORT": str(inst["port"]), "POD_USER": b.ssh_user,
           "POD_KEY": str(Path(b.ssh_key_path).expanduser()), "POD_WORKDIR": b.workdir,
           "POD_KNOWN_HOSTS": kh, "CASCADE_SRC_ROOT": b.release_dir,
           "CASCADE_CHAIN_TOML": b.chain_toml, "POD_STAGE": "pool_d"}
    envargs = ["env"] + [f"{k}={v}" for k, v in env.items()]
    remote_env = shlex.quote(b.remote_env_file)
    return pin_steps(cfg, inst, b.release_dir) + [
        ("release_sync", envargs + ["bash", f"{b.release_dir}/deploy/pod_bootstrap.sh"]),
        ("env_file", scp_argv(inst, key_path=b.ssh_key_path, known_hosts=kh,
                              src=b.env_file, dest=b.remote_env_file)),
        ("env_mode", ssh_argv(inst, key_path=b.ssh_key_path, known_hosts=kh,
                              remote=f"chmod 600 {remote_env}")),
        ("state_pull", ssh_argv(inst, key_path=b.ssh_key_path, known_hosts=kh,
                                remote=_remote(b, "state-pull --chain-toml chain.toml "
                                                  f"--forge-dir {shlex.quote(b.forge_dir)} "
                                                  f"--prior-root {shlex.quote(b.work_root)}"
                                                  "/prior"))),
        ("mirror_sync", ssh_argv(inst, key_path=b.ssh_key_path, known_hosts=kh,
                                 remote=_remote(b, "mirror-sync --chain-toml chain.toml "
                                                   f"--forge-dir {shlex.quote(b.forge_dir)}"))),
    ]


def _remote(b: BuilderConfig, cascade_laned_args: str) -> str:
    """A remote command run in the release dir with the env file loaded."""
    return (f"set -euo pipefail; cd {shlex.quote(b.workdir)}; set -a; "
            f". {shlex.quote(b.remote_env_file)}; set +a; "
            f".venv/bin/cascade-laned elastic {cascade_laned_args}")


def builder_run_script(b: BuilderConfig, *, as_of_ts: int, now_ts: int, started_ts: int) -> str:
    """The W6 sequence, verbatim from the W6fs / W6b / W6d READMEs:
    first_seen record, first_seen commit, pool_d_build, pool_d_publish publish
    (with the build wall), then state-push. Lane A's pool is fetched first
    (W6b replays it byte-for-byte)."""
    q = shlex.quote
    fs = f"{b.forge_dir}/first_seen"
    out = f"{b.work_root}/out/{as_of_ts}"
    lanea = f"{b.work_root}/lane_a_pool"
    py = ".venv/bin/python -P -m"
    extra = " ".join(q(x) for x in b.build_args)
    return "\n".join([
        "set -euo pipefail",
        f"cd {q(b.workdir)}",
        f"set -a; . {q(b.remote_env_file)}; set +a",
        f"rm -rf -- {q(lanea)} {q(lanea + '.tar')} {q(lanea + '.ref.json')} {q(out)}",
        f".venv/bin/cascade-laned elastic lane-a-pool --chain-toml chain.toml --out {q(lanea)}",
        f"{py} cascade.pool.first_seen record --forge-dir {q(b.forge_dir)} "
        f"--log-dir {q(fs)} --now-ts {int(now_ts)}",
        f"{py} cascade.pool.first_seen commit --log-dir {q(fs)} --now-ts {int(now_ts)}",
        "prior=()",
        f"for d in {q(b.work_root)}/prior/*/; do [ -f \"$d/metadata.json\" ] && "
        "prior+=(--prior-dir \"$d\"); done",
        " ".join(x for x in (
            f"{py} cascade.pool.pool_d_build --forge-dir {q(b.forge_dir)} --first-seen-dir {q(fs)}",
            f"--lane-a-pool {q(lanea)} --out {q(out)} --as-of-ts {int(as_of_ts)}",
            "--chain-toml chain.toml", extra, "\"${prior[@]}\"") if x),
        "finished=\"$(date +%s)\"",
        f"{py} cascade.pool.pool_d_publish publish --pool-d {q(out)} --chain-toml chain.toml "
        f"--started-ts {int(started_ts)} --finished-ts \"$finished\"",
        f".venv/bin/cascade-laned elastic state-push --chain-toml chain.toml "
        f"--forge-dir {q(b.forge_dir)} --pool-d {q(out)}",
    ])


def builder_run_steps(cfg: ElasticConfig, inst: Mapping, *, now_ts: int
                      ) -> list[tuple[str, list[str]]]:
    from ..pool.pool_d_publish import due_as_of

    b = cfg.builder
    kh = known_hosts_for(cfg, inst)
    script = builder_run_script(b, as_of_ts=due_as_of(now_ts), now_ts=now_ts,
                                started_ts=now_ts)
    return [("pool_d_build_publish",
             ssh_argv(inst, key_path=b.ssh_key_path, known_hosts=kh,
                      remote=f"bash -c {shlex.quote(script)}"))]


def eval_bootstrap_steps(cfg: ElasticConfig, inst: Mapping, *, release_dir: str
                         ) -> list[tuple[str, list[str]]]:
    """Per host: pin the key, the R12 on-rent check against the candidate's
    profile (GPU model / VRAM / driver / disk), then R12's release sync."""
    e = cfg.eval_host
    kh = known_hosts_for(cfg, inst)
    cap = known_hosts_for(cfg, inst) + ".smi"
    tool = _evalhost_tool(release_dir)
    probe = ssh_argv(inst, key_path=e.ssh_key_path, known_hosts=kh,
                     remote="nvidia-smi --query-gpu=name,memory.total,driver_version "
                            "--format=csv,noheader")
    return pin_steps(cfg, inst, release_dir) + [
        ("smi_capture", ["bash", "-c", " ".join(shlex.quote(a) for a in probe)
                         + f" > {shlex.quote(cap)}"]),
        ("on_rent_check", tool + ["on-rent-check", "--profile",
                                  f"{release_dir}/{inst['profile']}",
                                  "--nvidia-smi-query", cap]),
        ("venv_sync", tool + ["sync", "--release-dir", release_dir,
                              "--host", f"{inst['user']}@{inst['ip']}",
                              "--port", str(inst["port"]), "--known-hosts", kh,
                              "--dest", "/root/cascade"]),
    ]


Runner = Callable[[list[str]], int]


def _default_runner(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


def run_steps(steps: Sequence[tuple[str, list[str]]], *, dry_run: bool,
              runner: Runner | None = None) -> list[dict]:
    """Run each step in order; the first non-zero exit refuses ``step_failed``.
    ``--dry-run`` prints ``PLAN <step> <argv>`` and runs nothing."""
    runner = runner or _default_runner
    out = []
    for name, argv in steps:
        if dry_run:
            print(f"PLAN {name} {shlex.join(argv)}")
            out.append({"step": name, "dry_run": True})
            continue
        rc = runner(list(argv))
        out.append({"step": name, "rc": rc})
        if rc != 0:
            raise ElasticError("step_failed", f"{name} exited {rc}")
    return out


def cycle(cfg: ElasticConfig, client: ShadeformCpuOnly, *, dry_run: bool,
          runner: Runner | None = None, clock: Clock | None = None) -> dict:
    """One 3-hourly build: up → bootstrap → run → down (finally, unless keep_warm)."""
    clock = clock or Clock()
    up = builder_up(cfg, client, dry_run=dry_run, clock=clock)
    if dry_run:
        fake = {"id": "DRY-RUN", "ip": "0.0.0.0", "port": 22, "user": cfg.builder.ssh_user}
        steps = builder_bootstrap_steps(cfg, fake) + builder_run_steps(
            cfg, fake, now_ts=int(clock.now()))
        run_steps(steps, dry_run=True)
        return {**up, "keep_warm": cfg.builder.keep_warm}
    inst = up["instance"]
    result = {"up": up}
    try:
        result["bootstrap"] = run_steps(builder_bootstrap_steps(cfg, inst), dry_run=False,
                                        runner=runner)
        result["run"] = run_steps(builder_run_steps(cfg, inst, now_ts=int(clock.now())),
                                  dry_run=False, runner=runner)
    finally:
        if not cfg.builder.keep_warm or "run" not in result:
            result["down"] = down(cfg, BUILDER, client, clock=clock)
    return result


def eval_shard_plan(cfg: ElasticConfig, *, block_hash: str, entry_ids: Sequence[str],
                    king: str, house: str | None) -> dict:
    """``run`` for the eval host: the shard → host plan the validator consumes."""
    state = load_state(cfg, EVAL_HOST)
    live = sorted((i for i in state["instances"] if not i.get("closed")),
                  key=lambda i: int(i.get("shard", 0)))
    if not live:
        raise ElasticError("usage", "no eval host is up; run up first")
    shards = assign_shards(block_hash, entry_ids, len(live), king=king, house=house)
    for sh in shards:
        inst = live[sh["shard"]]
        sh.update(host=f"{inst['ip']}:{inst['port']}", instance=inst["id"],
                  gpu_model=inst["gpu_model"])
    return {"schema": PLAN_SCHEMA, "cohort": state.get("cohort"), "block_hash": block_hash,
            "shards": shards}


# ── builder-side state on object storage (runs ON the builder) ─────────────


# quarantine.jsonl (W6fs unreadable partitions) is newer than the first HEADs:
# a HEAD that does not name it restores it as empty (append-only from there).
FIRST_SEEN_FILES = ("records.jsonl", "roots.jsonl", "quarantine.jsonl")
PULLED_MARK = ".elastic_pulled.json"


def _state_key(prefix: str, *parts: str) -> str:
    return "/".join([prefix.strip("/"), STATE_SUBDIR, *parts])


def _head(store, prefix: str) -> dict | None:
    from ..pool.pool_d_publish import _get_json

    return _get_json(store, _state_key(prefix, "first_seen", "HEAD.json"))


PULLED_SCHEMA = "laned_elastic_pulled/2"
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_SCAN_CHUNK = 1 << 20


def _scan(path: Path, prefix_len: int = 0) -> tuple[str | None, str, int]:
    """One streaming pass over ``path`` (a missing file reads as empty):
    ``(sha256 of its first prefix_len bytes or None when shorter, sha256 of
    the whole file, size)``. Never holds more than one chunk."""
    h = hashlib.sha256()
    n = 0
    pre = _EMPTY_SHA if prefix_len <= 0 else None
    if path.is_file():
        with open(path, "rb") as f:
            while True:
                want = _SCAN_CHUNK
                if pre is None:
                    want = min(want, prefix_len - n)
                chunk = f.read(want)
                if not chunk:
                    break
                h.update(chunk)
                n += len(chunk)
                if pre is None and n == prefix_len:
                    pre = h.copy().hexdigest()
    return pre, h.hexdigest(), n


def _head_sizes(head: Mapping | None) -> Mapping:
    sizes = (head or {}).get("sizes")
    return sizes if isinstance(sizes, Mapping) else {}


def _read_mark(mark: Path) -> tuple[dict | None, dict]:
    """The HEAD ``state-pull`` saw and the byte sizes it restored. A mark
    written before ``PULLED_SCHEMA`` is the bare HEAD (sizes unknown)."""
    doc = json.loads(mark.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and doc.get("schema") == PULLED_SCHEMA:
        sizes = doc.get("sizes")
        return (doc.get("head") or None), (dict(sizes) if isinstance(sizes, Mapping) else {})
    return (doc or None), {}


def _object_size_or_none(store, key: str) -> int | None:
    from ..shared.hippius import ObjectNotFound

    try:
        return int(store.object_size(key))
    except ObjectNotFound:
        return None


def state_pull(store, *, prefix: str, log_dir: Path, prior_root: Path) -> dict:
    """Restore the first-seen log and every indexed snapshot's metadata.

    Each log file streams to a temp file next to it, is checked against the
    sha256 (and, when HEAD records it, the size) in ``HEAD.json``, is checked
    to extend any local copy (append-only), and only then replaces it. A
    local copy already equal to HEAD (a warm builder) is kept as is."""
    from ..shared.hippius import ObjectNotFound

    log_dir.mkdir(parents=True, exist_ok=True)
    head = _head(store, prefix)
    hsizes = _head_sizes(head)
    sizes: dict[str, int] = {}
    restored: list[str] = []
    kept: list[str] = []
    for name in FIRST_SEEN_FILES:
        local = log_dir / name
        key = _state_key(prefix, "first_seen", name)
        named = head is not None and name in head
        local_sha = local_size = None
        if local.is_file():
            _, local_sha, local_size = _scan(local)
        if (named and local_sha == head.get(name)
                and hsizes.get(name, local_size) == local_size
                and _object_size_or_none(store, key) == local_size):
            sizes[name] = int(local_size)
            kept.append(name)
            continue
        tmp = log_dir / f".{name}.pull.tmp"
        try:
            try:
                got = store.get_file_sha(key, tmp)
            except ObjectNotFound:
                if named:
                    raise ElasticError("state_conflict",
                                       f"HEAD names {name} but it is missing") from None
                tmp.write_bytes(b"")
                got = {"sha256": _EMPTY_SHA, "size": 0}
            if head is not None and not named:
                # A HEAD older than this file: only an empty copy is consistent.
                if got["size"]:
                    raise ElasticError("state_conflict", f"{name} is in the bucket but "
                                                         f"HEAD.json does not name it")
            elif named and (got["sha256"] != head.get(name)
                            or hsizes.get(name, got["size"]) != got["size"]):
                raise ElasticError("state_conflict", f"{name} does not match HEAD.json")
            if local_size is not None:
                pre, _, _ = _scan(tmp, local_size)
                if pre != local_sha:
                    raise ElasticError("state_conflict", f"local {name} is not a prefix of "
                                                         f"the bucket copy (append-only)")
            os.replace(tmp, local)
        finally:
            tmp.unlink(missing_ok=True)
        sizes[name] = int(got["size"])
        restored.append(name)
    (log_dir / PULLED_MARK).write_text(
        json.dumps({"schema": PULLED_SCHEMA, "head": head, "sizes": sizes}, sort_keys=True),
        encoding="utf-8")
    from ..pool.pool_d_publish import _get_json, _index_key

    index = _get_json(store, _index_key(prefix)) or {"snapshots": []}
    got = []
    for ent in index.get("snapshots") or []:
        snap = str(ent["snapshot"])
        try:
            data = store.get_bytes(_state_key(prefix, "prior", snap, "metadata.json"))
        except ObjectNotFound:
            continue
        d = prior_root / snap
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_bytes(data)
        got.append(snap)
    return {"head": head, "prior": got, "restored": restored, "kept": kept}


def _bucket_copy(store, *, key: str, name: str, head: Mapping | None, pulled: Mapping,
                 log_dir: Path) -> tuple[str, int]:
    """``(sha256, size)`` of the bucket copy ``state-push`` must extend.

    Named in HEAD: HEAD's sha256 (``state-pull`` verified the bytes against
    it) and the size it restored (else HEAD's, else ``HeadObject``) -- no
    download. Not named (no HEAD yet, or a file newer than HEAD): an absent or
    empty object is empty; anything else streams down once to hash it."""
    from ..shared.hippius import ObjectNotFound

    if head is not None and name in head:
        size = pulled.get(name, _head_sizes(head).get(name))
        if size is None:
            size = _object_size_or_none(store, key)
            if size is None:
                raise ElasticError("state_conflict", f"HEAD names {name} but it is missing")
        return str(head[name]), int(size)
    size = _object_size_or_none(store, key)
    if not size:
        return _EMPTY_SHA, 0
    tmp = log_dir / f".{name}.remote.tmp"
    try:
        try:
            got = store.get_file_sha(key, tmp)
        except ObjectNotFound:
            return _EMPTY_SHA, 0
        return str(got["sha256"]), int(got["size"])
    finally:
        tmp.unlink(missing_ok=True)


def state_push(store, *, prefix: str, log_dir: Path, pool_d: Path | None) -> dict:
    """Upload the grown first-seen log (append-only, compare-and-swap on HEAD)
    and the new snapshot's metadata.json.

    Streaming throughout (the log is 20 GB and growing): one local pass per
    file proves the bucket copy is a byte prefix (its sha256 from HEAD against
    the sha256 of the local file's first ``size`` bytes) and hashes the whole
    file; the upload goes through the store's ``put_file_sha`` (multipart above
    one part) and must report the same sha256 and size, and the stored object
    must have that size. A file whose sha256 and size equal the bucket copy's
    is not re-uploaded. ``HEAD.json`` keeps ``{name: sha256}`` and gains
    ``sizes`` ({name: bytes}); it is written last, so a failed push leaves the
    previous HEAD and the next push retries."""
    mark = log_dir / PULLED_MARK
    if not mark.is_file():
        raise ElasticError("state_conflict", "state-pull never ran on this builder")
    seen, pulled = _read_mark(mark)
    head = _head(store, prefix)
    if (head or None) != seen:
        raise ElasticError("state_conflict", "the bucket HEAD moved since state-pull "
                                             "(another builder?)")
    plan = []
    for name in FIRST_SEEN_FILES:
        key = _state_key(prefix, "first_seen", name)
        path = log_dir / name
        prev_sha, prev_size = _bucket_copy(store, key=key, name=name, head=head,
                                           pulled=pulled, log_dir=log_dir)
        pre, sha, size = _scan(path, prev_size)
        if pre != prev_sha:
            raise ElasticError("state_conflict", f"{name}: the bucket copy is not a prefix "
                                                 f"of the local log (append-only)")
        plan.append((name, key, path, sha, size, sha == prev_sha and size == prev_size))
    uploaded, skipped = [], []
    new_head: dict = {}
    sizes: dict[str, int] = {}
    for name, key, path, sha, size, same in plan:
        new_head[name], sizes[name] = sha, size
        if same and _object_size_or_none(store, key) == size:
            skipped.append(name)
            continue
        if path.is_file():
            got = store.put_file_sha(key, path)
        else:
            store.put_bytes(key, b"")
            got = {"sha256": _EMPTY_SHA, "size": 0}
        if got["sha256"] != sha or int(got["size"]) != size:
            raise ElasticError("state_conflict", f"{name} changed during state-push "
                                                 f"(scanned {size} bytes, uploaded "
                                                 f"{got['size']})")
        stored = _object_size_or_none(store, key)
        if stored != size:
            raise ElasticError("state_conflict", f"{name}: the bucket holds {stored} bytes "
                                                 f"after uploading {size}")
        uploaded.append(name)
    new_head["sizes"] = sizes
    out = {"head": new_head, "uploaded": uploaded, "skipped": skipped}
    if pool_d is not None:
        meta = json.loads((pool_d / "metadata.json").read_text(encoding="utf-8"))
        snap = str(meta["snapshot"])
        store.put_bytes(_state_key(prefix, "prior", snap, "metadata.json"),
                        (pool_d / "metadata.json").read_bytes())
        out["prior"] = snap
    store.put_bytes(_state_key(prefix, "first_seen", "HEAD.json"),
                    json.dumps(new_head, sort_keys=True).encode())
    mark.write_text(json.dumps({"schema": PULLED_SCHEMA, "head": new_head, "sizes": sizes},
                               sort_keys=True), encoding="utf-8")
    return out


def mirror_sync(client, *, bucket: str, forge_dir: Path) -> dict:
    """Delta-pull the forge mirror, never touching ``first_seen/`` (our own
    append-only log lives there). Download when missing or the size differs."""
    n_get = n_keep = 0
    pager = client.get_paginator("list_objects_v2")
    for page in pager.paginate(Bucket=bucket):
        for obj in page.get("Contents") or []:
            key = str(obj["Key"])
            parts = [p for p in key.split("/") if p]
            if not parts or parts[0] == "first_seen" or any(p in (".", "..") for p in parts):
                continue
            dest = forge_dir.joinpath(*parts)
            if dest.is_file() and dest.stat().st_size == int(obj.get("Size", -1)):
                n_keep += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            n_get += 1
    return {"downloaded": n_get, "unchanged": n_keep}


def _pool_d_store(chain_toml: Path):
    from ..pool.pool_d_publish import open_pool_store

    store = open_pool_store(chain_toml)
    return store, store.prefix


def _forge_client():
    from ..shared.hippius import S3Config, _s3_client

    endpoint = os.environ.get("TSFORGE_S3_ENDPOINT", "")
    bucket = os.environ.get("TSFORGE_BUCKET", "")
    if not endpoint or not bucket:
        raise ElasticError("usage", "TSFORGE_S3_ENDPOINT / TSFORGE_BUCKET not set in the "
                                    "builder env file")
    cfg = S3Config(endpoint=endpoint, region="decentralized", bucket=bucket,
                   access_key_env="TSFORGE_S3_ACCESS_KEY",
                   secret_key_env="TSFORGE_S3_SECRET_KEY")
    return _s3_client(cfg), bucket


def select_lane_a_snapshot(index: Sequence):
    """The newest REAL lane A index entry: smoke-test sentinels
    (``effective_block >= POOL_SENTINEL_MIN_BLOCK``, e.g. the 999999999999
    workflow_dispatch default of publish-pool.yml) are never a validator's
    pool, so W6b must never replay one (2026-11-25: the 2026-09-29 sentinel
    was picked and the replay refused with 4586 mismatching files)."""
    from ..shared.hippius import POOL_SENTINEL_MIN_BLOCK

    real = [s for s in index if int(s.effective_block) < POOL_SENTINEL_MIN_BLOCK]
    if not real:
        raise ElasticError("usage", "lane A pool index has no real (non-sentinel) snapshot")
    return max(real, key=lambda s: s.effective_block)


def lane_a_pool(chain_toml: Path, out: Path) -> dict:
    """Fetch the newest real lane A pool snapshot (sha-checked against the
    index) for W6b. Besides the unpacked ``out`` it keeps the verified tar
    (``<out>.tar``) and the index entry it resolved (``<out>.ref.json``), which
    ``pool_d_build``'s default snapshot mode re-verifies before it takes the
    selection from the snapshot."""
    from ..shared.config import load_chain_config
    from ..shared.hippius import (
        StorageError,
        pool_s3_store,
        read_pool_index,
        tar_cid_digest,
        unpack_tar_to_dir,
    )

    cfg = load_chain_config(chain_toml)
    store = pool_s3_store(cfg.storage)
    index = read_pool_index(store)
    if not index:
        raise ElasticError("usage", "lane A pool index is empty")
    meta = select_lane_a_snapshot(index)
    data = store.get_bytes(meta.key)
    got = tar_cid_digest(data)
    if got != meta.sha256:
        raise StorageError(f"pool_snapshot_digest_mismatch: {got} != {meta.sha256}")
    unpack_tar_to_dir(data, out)
    out = Path(out)
    tar = out.with_name(out.name + ".tar")
    tar.write_bytes(data)
    ref = {"effective_block": int(meta.effective_block), "sha256": meta.sha256,
           "key": meta.key, "tar": tar.name}
    out.with_name(out.name + ".ref.json").write_text(
        json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"effective_block": meta.effective_block, "sha256": meta.sha256, "out": str(out)}


def env_template(path: Path) -> dict:
    """Create the builder env file with key NAMES only (mode 600); never overwrite."""
    if path.exists():
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            raise ElasticError("usage", f"{path} has mode {mode:o}; it must be 600")
        return {"path": str(path), "created": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("# lane D pool_d builder credentials (owner fills; mode 600; never commit)\n")
        for name in BUILDER_ENV_NAMES:
            f.write(f"{name}=\n")
    return {"path": str(path), "created": True, "names": list(BUILDER_ENV_NAMES)}


def check_env_file(path: Path) -> None:
    if not path.is_file():
        raise ElasticError("usage", f"{path} missing; run `cascade-laned elastic env-template` "
                                    f"and fill it")
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        raise ElasticError("usage", f"{path} has mode {mode:o}; it must be 600")


# ── CLI ────────────────────────────────────────────────────────────────────


def _emit(doc) -> None:
    print(json.dumps(doc, indent=2, sort_keys=True, default=str))


def _cmd(args) -> int:
    try:
        return _dispatch(args)
    except ElasticError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT[e.code]


def _dispatch(args) -> int:
    act = args.action
    if act == "state-pull":
        store, prefix = _pool_d_store(args.chain_toml)
        _emit(state_pull(store, prefix=prefix, log_dir=args.forge_dir / "first_seen",
                         prior_root=args.prior_root))
        return 0
    if act == "state-push":
        store, prefix = _pool_d_store(args.chain_toml)
        _emit(state_push(store, prefix=prefix, log_dir=args.forge_dir / "first_seen",
                         pool_d=args.pool_d))
        return 0
    if act == "mirror-sync":
        client, bucket = _forge_client()
        _emit(mirror_sync(client, bucket=bucket, forge_dir=args.forge_dir))
        return 0
    if act == "lane-a-pool":
        _emit(lane_a_pool(args.chain_toml, args.out))
        return 0
    if act == "env-template":
        _emit(env_template(args.path))
        return 0
    cfg = load_config(args.config)
    role = args.role
    dry = bool(getattr(args, "dry_run", False)) or act == "quote"
    if role == BUILDER:
        if act in ("up", "cycle", "bootstrap") and not dry:
            check_env_file(Path(cfg.builder.env_file))
        client = ShadeformCpuOnly(cfg.builder)
        if act in ("quote", "up"):
            _emit(builder_up(cfg, client, dry_run=dry))
        elif act == "cycle":
            _emit(cycle(cfg, client, dry_run=dry))
        elif act == "down":
            _emit(down(cfg, BUILDER, client, dry_run=dry))
        else:
            inst = _one_live(cfg, BUILDER)
            steps = (builder_bootstrap_steps(cfg, inst) if act == "bootstrap"
                     else builder_run_steps(cfg, inst, now_ts=int(time.time())))
            _emit(run_steps(steps, dry_run=dry))
        return 0
    lium = make_lium(cfg.eval_host)
    if act in ("quote", "up"):
        _emit(eval_up(cfg, lium, entries=int(args.entries), cohort=str(args.cohort),
                      dry_run=dry))
    elif act == "down":
        _emit(down(cfg, EVAL_HOST, lium, cohort_done=bool(args.cohort_done), dry_run=dry))
    elif act == "bootstrap":
        state = load_state(cfg, EVAL_HOST)
        steps = [s for i in state["instances"] if not i.get("closed")
                 for s in eval_bootstrap_steps(cfg, i, release_dir=cfg.builder.release_dir)]
        _emit(run_steps(steps, dry_run=dry))
    elif act == "run":
        plan = eval_shard_plan(cfg, block_hash=args.block_hash, entry_ids=args.entry_id or [],
                               king=args.king, house=args.house)
        if args.out and not dry:
            args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        _emit(plan)
    else:
        raise ElasticError("usage", f"{act} is a builder action")
    return 0


def _one_live(cfg: ElasticConfig, role: str) -> dict:
    live = [i for i in load_state(cfg, role)["instances"] if not i.get("closed")]
    if not live:
        raise ElasticError("usage", f"no {role} is up; run up first")
    return live[0]


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Register ``elastic`` on the ``cascade-laned`` parser."""
    p = subparsers.add_parser("elastic", help="elastic lane D hosts (pool_d builder on "
                                              "Shadeform CPU; eval hosts on Lium)")
    acts = p.add_subparsers(dest="action", required=True)
    for name, hlp in (("quote", "print the chosen type and price; rents nothing"),
                      ("up", "rent (spend-capped; deleted on any failure)"),
                      ("bootstrap", "idempotent host setup"),
                      ("run", "builder: build + publish; eval host: write the shard plan"),
                      ("down", "delete and verify"),
                      ("cycle", "builder: up, bootstrap, run, down (the 3-hourly timer)")):
        a = acts.add_parser(name, help=hlp)
        a.add_argument("--role", choices=ROLES, required=True)
        a.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        a.add_argument("--dry-run", action="store_true")
        if name in ("quote", "up"):
            a.add_argument("--entries", type=int, default=0,
                           help="eval host: entries needing scoring this settlement")
            a.add_argument("--cohort", default="",
                           help="eval host: the settlement boundary / cohort id")
        if name == "down":
            a.add_argument("--cohort-done", action="store_true",
                           help="eval host: the settlement's scoring has decided")
        if name == "run":
            a.add_argument("--block-hash", default="")
            a.add_argument("--entry-id", action="append")
            a.add_argument("--king", default="")
            a.add_argument("--house", default=None)
            a.add_argument("--out", type=Path, default=None)
    a = acts.add_parser("state-pull", help="(on the builder) restore first_seen + priors")
    a.add_argument("--chain-toml", type=Path, required=True)
    a.add_argument("--forge-dir", type=Path, required=True)
    a.add_argument("--prior-root", type=Path, required=True)
    a = acts.add_parser("state-push", help="(on the builder) append-only upload")
    a.add_argument("--chain-toml", type=Path, required=True)
    a.add_argument("--forge-dir", type=Path, required=True)
    a.add_argument("--pool-d", type=Path, default=None)
    a = acts.add_parser("mirror-sync", help="(on the builder) forge delta pull, no first_seen/")
    a.add_argument("--chain-toml", type=Path, required=True)
    a.add_argument("--forge-dir", type=Path, required=True)
    a = acts.add_parser("lane-a-pool", help="(on the builder) fetch the newest lane A pool")
    a.add_argument("--chain-toml", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = acts.add_parser("env-template", help="create the builder env file (names, mode 600)")
    a.add_argument("--path", type=Path, required=True)
    p.set_defaults(func=_cmd)
    return p


__all__ = [
    "EXIT", "ROLES", "BuilderConfig", "Clock", "CpuOffer", "ElasticConfig", "ElasticError",
    "EvalCandidate", "EvalConfig", "GpuOffer", "ShadeformCpuOnly", "SpendLedger",
    "add_parser", "assign_shards", "builder_run_script", "builder_up", "check_spend",
    "cycle", "down", "eval_shard_plan", "eval_up", "hosts_needed", "lium_gpu_offers",
    "load_config", "merge_shard_scores", "mirror_sync", "qualified_candidates",
    "qualified_gpu_models", "shadeform_cpu_offers", "state_pull", "state_push",
]
