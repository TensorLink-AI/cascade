"""MarketCast (DEC-CA-0064) on the trainer orchestrator: the recipe fetch,
the leg dispatch on the miner's key and the harvest check.

A MarketCast leg rides the lane-C funded-leg machinery unchanged — the
lane-C queue (``POST /v1/lane-c/fund``), the same token budget per leg, the
payer pod on the miner's key, the miner's GPU choice (DEC-CA-0036 /
DEC-CA-0060), the identity + runtime pins, credential-free harvest — with
three differences:

* the base is the PINNED Timestral checkpoint (``[marketcast] base_ref``),
  never the era king, so a leg never ages out on a king change;
* the pod runs ``cascade-train-worker --role mkt_finetune`` (always
  ``--local-only``) with ``{"marketcast": <marketcast.json>,
  "has_generator": bool}``;
* the harvested checkpoint is checked as a Timestral fine-tune
  (:func:`verify_harvested_mkt_checkpoint`): the weights' sha256 is the one
  the worker reported, the architecture (tensor names / shapes / dtypes) is
  the base's, ``config.json`` is the base's, every weight is finite, and
  ``marketcast_meta.json`` passes :func:`cascade.shared.marketcast_recipe.
  meta_problem` (required fields, the pinned base, covariate names in the
  allow-list, size) — a malformed meta is the miner's fault here, not a
  validator-side hold later (review 2026-10-19).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("cascade.trainer.marketcast")

GENERATOR_MARKERS = ("generator.py", "pyproject.toml")


@dataclass(frozen=True)
class MktLegRecipe:
    """A validated MarketCast recipe as the lane-C driver carries it."""
    target: str
    payload: str                 # the marketcast.json text (canonical JSON)
    has_generator: bool
    # the trainer-signed dispatch time (unix): the pod clips every fetched
    # bar to before it and re-checks the embargo against it (review M12)
    train_time: float = 0.0
    # owner 2026-11-18: the warm-start checkpoint the TRAINER resolved from the
    # recipe's ``init`` against its lineage (:func:`resolve_mkt_init`)
    init_pointer: str = ""
    init_sha256: str = ""

    @property
    def leg_hours(self) -> float:
        """The recipe's own ``leg_hours`` (0 = absent: ``[lane_c] leg_hours``)."""
        try:
            return float(json.loads(self.payload).get("leg_hours") or 0.0)
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def to_json(self) -> str:
        return json.dumps({"marketcast": json.loads(self.payload),
                           "has_generator": bool(self.has_generator),
                           **({"train_time": float(self.train_time)} if self.train_time else {}),
                           **({"init_pointer": self.init_pointer,
                               "init_sha256": self.init_sha256} if self.init_pointer else {})},
                          sort_keys=True, separators=(",", ":"))


def has_generator_tree(d: Path) -> bool:
    """Whether a fetched recipe tree also ships a generator (a
    ``generator.py`` or a packaged tree at its root)."""
    return any((Path(d) / m).is_file() for m in GENERATOR_MARKERS)


LINEAGE_FILE = "marketcast_lineage.json"
_LINEAGE_LOCK = threading.Lock()


def load_mkt_lineage(root) -> dict:
    """The trainer's record of the MarketCast checkpoints IT produced:
    ``{trained_pointer: {"train_time", "checkpoint_sha256", "init_pointer"}}``
    (``{}`` when none / unreadable)."""
    from ..shared.atomic import read_text_with_backup

    d, _src = read_text_with_backup(Path(root) / LINEAGE_FILE, json.loads)
    return d if isinstance(d, dict) else {}


def record_mkt_lineage(root, pointer: str, *, train_time: float, checkpoint_sha256: str,
                       init_pointer: str = "") -> None:
    """Append one harvested MarketCast checkpoint to the lineage (atomic,
    with a ``.bak``). Serialised: concurrent lane-C legs finish on their own
    threads, and an unlocked read-modify-write would drop one of them (its
    checkpoint could then never be warm-started from)."""
    from ..shared.atomic import write_text_atomic

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with _LINEAGE_LOCK:
        d = load_mkt_lineage(root)
        d[str(pointer)] = {"train_time": float(train_time),
                           "checkpoint_sha256": str(checkpoint_sha256),
                           "init_pointer": str(init_pointer or "")}
        write_text_atomic(root / LINEAGE_FILE, json.dumps(d, sort_keys=True, indent=1),
                          backup=True)


def resolve_mkt_init(init: str, lineage: dict, *, submission_time: float | None) -> tuple[str, str]:
    """Owner 2026-11-18: ``(init_pointer, init_sha256)`` of a recipe's
    warm-start, or ``("", "")`` for the pinned base. Only a checkpoint THIS
    trainer produced (``lineage``) from a leg dispatched strictly before the
    submission is accepted: its own data ended ``embargo_days`` before ITS
    submission, so a chain of warm-starts never reads the current embargo
    gap, and a model trained off-platform cannot be smuggled in.
    :class:`~cascade.trainer.finetune.RecipeError` (miner fault) otherwise."""
    from .finetune import RecipeError

    if not init:
        return "", ""
    rec = lineage.get(str(init))
    if not isinstance(rec, dict):
        raise RecipeError(f"marketcast.json: init {init[:60]!r} is not a MarketCast checkpoint "
                          f"this trainer produced")
    if submission_time is not None and not float(rec.get("train_time", 0.0)) < float(
            submission_time):
        raise RecipeError("marketcast.json: init was dispatched after this submission")
    return str(init), str(rec.get("checkpoint_sha256", ""))


def read_mkt_recipe(d: Path, cfg, *, target: str, submission_time: float | None = None,
                    train_time: float | None = None, lineage: dict | None = None
                    ) -> MktLegRecipe:
    """``marketcast.json`` at the root of the fetched tree, validated with
    the shared rule; :class:`~cascade.trainer.finetune.RecipeError` (miner
    fault) on anything wrong. ``submission_time`` (the commitment's time on
    the dispatch clock) enforces ``data.end <= submission − embargo_days``
    HERE, on the trainer, before anything is rented (review M12) — the miner
    CLI's check is advisory."""
    from ..shared.marketcast_recipe import MARKETCAST_RECIPE_FILE, parse_marketcast_recipe
    from .finetune import RecipeError

    p = Path(d) / MARKETCAST_RECIPE_FILE
    if not p.is_file():
        raise RecipeError(f"the recipe tree has no {MARKETCAST_RECIPE_FILE}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        gen = has_generator_tree(d)
        rc = parse_marketcast_recipe(obj, cfg.marketcast, target=target,
                                     submission_time=submission_time, has_generator=gen)
    except (ValueError, OSError) as e:
        raise RecipeError(f"{MARKETCAST_RECIPE_FILE}: {e}") from e
    ip, isha = resolve_mkt_init(rc.init, lineage or {}, submission_time=submission_time)
    return MktLegRecipe(target=str(target), has_generator=gen,
                        payload=json.dumps(obj, sort_keys=True, separators=(",", ":")),
                        train_time=float(train_time or 0.0), init_pointer=ip, init_sha256=isha)


def base_weights_path(runner) -> Path:  # pragma: no cover — Hub download
    """The pinned base's ``model.safetensors`` on the orchestrator (fetched
    once at its commit, sha256-verified)."""
    from ..eval.timestral_adapter import WEIGHTS_FILE, verify_checkpoint
    from ..shared.config.marketcast import split_base_ref

    hit = getattr(runner, "_mkt_base_weights", None)
    if hit is not None:
        return hit
    mc = runner.cfg.marketcast
    repo, rev = split_base_ref(mc.base_ref)
    from huggingface_hub import hf_hub_download

    cache = str(Path(runner.work_root) / "_hf")
    p = Path(hf_hub_download(repo, WEIGHTS_FILE, revision=rev, cache_dir=cache))
    hf_hub_download(repo, "config.json", revision=rev, cache_dir=cache)   # beside the weights
    verify_checkpoint(p, mc.base_sha256)
    runner._mkt_base_weights = p
    return p


def verify_harvested_mkt_checkpoint(runner, dest: Path, receipt, *,
                                    base_weights: Path | None = None,
                                    base_config: dict | None = None) -> str:
    """Why the harvested fine-tune at ``dest`` must be refused ("" = fine)."""
    from ..eval.timestral_adapter import (
        CONFIG_FILE,
        META_FILE,
        WEIGHTS_FILE,
        safetensors_all_finite,
        same_structure,
        sha256_file,
    )
    from ..shared.marketcast_recipe import meta_problem

    mc = runner.cfg.marketcast
    m = dict(receipt.marketcast or {})
    w = Path(dest) / WEIGHTS_FILE
    if not w.is_file():
        return f"no {WEIGHTS_FILE} in the harvested checkpoint"
    if m.get("base_ref") != mc.base_ref or str(m.get("base_sha256", "")).lower() != \
            mc.base_sha256:
        return "the leg reports a base other than the pinned [marketcast] base"
    got = sha256_file(w)
    want = {str(m.get("checkpoint_sha256", "")),
            str(dict(receipt.tensor_digests or {}).get(WEIGHTS_FILE, ""))}
    if want != {got}:
        return f"{WEIGHTS_FILE} sha256 {got} != the worker's {sorted(want)}"
    bw = base_weights if base_weights is not None else base_weights_path(runner)
    diff = same_structure(bw, w)
    if diff:
        return f"not the base architecture: {diff[:3]}"
    try:
        cfg = json.loads((Path(dest) / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{CONFIG_FILE} unreadable"
    if base_config is None:
        base_config = json.loads((Path(bw).parent / CONFIG_FILE).read_text(encoding="utf-8")) \
            if (Path(bw).parent / CONFIG_FILE).is_file() else cfg
    if cfg != base_config:
        return f"{CONFIG_FILE} differs from the base's"
    try:
        raw = (Path(dest) / META_FILE).read_bytes()
        meta = json.loads(raw)
    except (OSError, ValueError):
        return f"{META_FILE} missing or unreadable"
    why = meta_problem(meta, mc, size=len(raw))
    if why:
        return why
    try:
        if not safetensors_all_finite(w):
            return "non-finite weights"
    except ValueError as e:
        return str(e)
    return ""


def mkt_leg_hours(cfg, recipe_leg_hours: float = 0.0) -> float:
    """A MarketCast leg's wall in hours: the recipe's own ``leg_hours`` (owner
    2026-11-16: the miner chooses, up to ``[marketcast] max_leg_hours``), else
    ``[lane_c] leg_hours``. The cap is re-applied here (a recipe parsed under
    a larger cap never outruns this release's)."""
    h = float(recipe_leg_hours or 0.0)
    if h <= 0.0:
        return float(cfg.lane_c.leg_hours)
    return min(h, float(cfg.marketcast.max_leg_hours))


def mkt_leg_contract(cfg, base, hours: float | None = None):
    """Red-team F5 (2026-10-20): the contract a MarketCast leg rents and
    harvests under — the throne contract at the LEG's wall (``hours``,
    default :func:`mkt_leg_hours`): token budget and wall guard scaled
    together with the ``[round]`` guard knobs, with the stream-CPU corpus (no
    CPU-tier shards for this leg). Owner 2026-11-16: the wall is the leg's own
    and may exceed the throne's pinned ``max_train_seconds`` (a 12 h recipe
    leg vs the 5 h throne; testnet's 300 s fast-round throne had silently cut
    every MarketCast leg to 5 min). The worker's own loop stops at the same
    wall (``worker.py``), so the pod, the harvest and the training agree on
    one budget."""
    from dataclasses import replace

    rnd = getattr(cfg, "round", None)
    h = float(hours) if hours else mkt_leg_hours(cfg)
    gf = float(getattr(rnd, "heat_guard_factor", 1.0) or 1.0)
    floor = int(getattr(rnd, "heat_guard_floor_seconds", 0) or 0)
    c = replace(base, corpus_mode="stream_cpu").for_hours(h, guard_factor=gf,
                                                          guard_floor_seconds=floor)
    return replace(c, max_train_seconds=max(int(round(gf * h * 3600.0)), floor))


def mkt_contract_block_kw(cfg, era) -> dict:
    """DEC-CA-0067: a MarketCast leg of an era past the automatic image repin
    carries ``contract_block`` = the era start, so the dispatcher's image hook
    checks the pod's worker code, pushes the post-switch chain.toml and the
    worker resolves (and verifies) the after image — the rent already used the
    era's pins (``funded_contract``). Any other leg: no key (byte-identical)."""
    from ..shared.era import image_after_armed, image_decoupled_era

    if getattr(getattr(cfg, "marketcast", None), "pod_image", ""):
        return {}       # its own image ([marketcast] pod_image): never the fleet's after pins
    start = int(getattr(era, "start_block", 0) or 0)
    if image_after_armed(cfg) and image_decoupled_era(cfg, start):
        return {"contract_block": start}
    return {}


def mkt_bars_store(storage, bucket: str):
    """The bar-recording bucket's store. The eval pool bucket (``[storage]
    pool_bucket``) is read exactly as every pool reader reads it — the pool
    endpoint and ``POOL_S3_*`` when set (:func:`~cascade.shared.hippius.
    pool_s3_store`; its primary, which presigns). Any other bucket:
    ``MKT_BARS_S3_ACCESS_KEY`` / ``MKT_BARS_S3_SECRET_KEY`` when set, else the
    trainer's ``HIPPIUS_S3_*``."""
    import os

    from ..shared.hippius import S3Config, S3Store, pool_s3_store

    if bucket == (getattr(storage, "pool_bucket", "") or "cascade-eval-pool"):
        st = pool_s3_store(storage, bucket=bucket)
        return getattr(st, "primary", st)
    own = bool(os.environ.get("MKT_BARS_S3_ACCESS_KEY"))
    base = S3Config.from_storage(storage, bucket=bucket)
    return S3Store(S3Config(
        endpoint=base.endpoint, region=base.region, bucket=bucket,
        access_key_env="MKT_BARS_S3_ACCESS_KEY" if own else base.access_key_env,
        secret_key_env="MKT_BARS_S3_SECRET_KEY" if own else base.secret_key_env))


def mkt_relay_corpus(cfg, finetune_recipe: str, *, store=None) -> dict:
    """``{"corpus_files": {mkt_bars.json: ...}}`` dispatch kwargs relaying
    the bar recording to a MarketCast pod (:mod:`cascade.pool.
    marketcast_relay`), ``{}`` with ``[marketcast] bar_relay`` off. The
    links are signed with the TRAINER's key; the pod gets only them."""
    import json

    from ..pool.marketcast_relay import RELAY_FILE
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from .marketcast_leg import training_relay_plan

    mc = cfg.marketcast
    if not mc.bar_relay:
        return {}
    bucket, prefix = mc.bar_relay.split("/", 1)
    if store is None:
        store = mkt_bars_store(cfg.storage, bucket)
    wrap = json.loads(finetune_recipe)
    train_time = float(wrap.get("train_time", 0.0) or 0.0) or None
    recipe = parse_marketcast_recipe(wrap["marketcast"], mc,
                                     has_generator=bool(wrap.get("has_generator")),
                                     submission_time=train_time)
    plan = training_relay_plan(store, prefix, mc, recipe, not_after=train_time)
    return {"corpus_files": {RELAY_FILE: json.dumps(plan, sort_keys=True)}}


def train_mkt_leg(ops, leg: dict, recipe: MktLegRecipe) -> dict:  # pragma: no cover — pods
    """One MarketCast leg on the MINER's key through the lane-C funded path
    (``RunnerLaneCOps``): returns the driver's result dict, with the
    ``marketcast`` record for the manifest entry."""
    from ..funding.gpu_choice import GpuChoice
    from ..funding.queue import OPERATOR_FAULT_CLASS
    from .faults import _FundedOperatorFallback
    from .finetune import leg_suffix
    from .lane_c_driver import LegFailure
    from .loop import ResolvedGenerator

    r = ops.r
    seeds, era = ops._seeds()
    if seeds is None or era is None:
        raise LegFailure("no era state yet", miner_fault=False, error_class="no_capacity",
                         burn=False)
    base = r.cfg.throne_contracts_at(int(era.start_block))[0]
    contract = mkt_leg_contract(r.cfg, base, mkt_leg_hours(r.cfg, recipe.leg_hours))
    gen = ResolvedGenerator(hotkey=leg["hotkey"], uid=int(leg["uid"]), ref=leg["ref"])
    r._funded_leg_failures.pop(leg["hotkey"], None)
    r._mkt_harvested.pop(leg["hotkey"], None)
    from .leg_pins import mkt_leg_scope

    try:
        with mkt_leg_scope(r):
            entry = r._run_funded_leg(
                r._dispatcher(), gen, seeds, int(leg["started_block"]), contract,
                leg_suffix(leg["target"], int(leg["uid"])),
                warm_start_ref=r.cfg.marketcast.base_ref, era_index=era.index,
                **mkt_contract_block_kw(r.cfg, era),
                role="mkt_finetune", finetune_recipe=recipe.to_json(), bench=False,
                funded_queue=ops.queue(),
                gpu_choice=GpuChoice(
                    gpu=str(leg.get("gpu", "") or ""),
                    allow_fallback=bool(leg.get("gpu_allow_fallback", False)),
                    max_price_per_hour=float(leg.get("gpu_max_price_per_hour", 0.0) or 0.0)))
    except _FundedOperatorFallback as e:
        q = ops.queue()
        recovery = q.get(gen.hotkey) if q is not None else None
        if recovery is not None and recovery.operator_billed:
            raise LegFailure("operator-funded MarketCast recovery is waiting for operator "
                             "capacity; the miner is not charged again", miner_fault=False,
                             error_class="operator_transient", burn=False) from e
        raise LegFailure(f"no marketplace capacity on the payer key ({e})",
                         miner_fault=False, error_class="no_capacity", burn=False) from e
    except Exception as e:
        verdict = r._funded_leg_failures.pop(leg["hotkey"], None)
        if verdict is None:
            raise LegFailure(f"marketcast leg failed (operator side): {e}", miner_fault=False,
                             error_class="operator_transient", burn=False) from e
        msg, miner_fault, error_class, burn = verdict
        raise LegFailure(msg or str(e), miner_fault=miner_fault, error_class=error_class,
                         burn=burn) from e
    mkt = r._mkt_harvested.pop(leg["hotkey"], None)
    if not mkt:
        raise LegFailure("marketcast leg returned without its harvest record",
                         miner_fault=False, error_class=OPERATOR_FAULT_CLASS, burn=False)
    # owner 2026-11-18: the warm-start is the TRAINER's (resolved before the
    # rent), stamped into the signed record and the lineage
    mkt = {**dict(mkt), **({"init_pointer": recipe.init_pointer} if recipe.init_pointer else {})}
    record_mkt_lineage(r.work_root, entry.trained_pointer, train_time=recipe.train_time,
                       checkpoint_sha256=str(mkt.get("checkpoint_sha256", "")),
                       init_pointer=recipe.init_pointer)
    return {"trained_pointer": entry.trained_pointer, "corpus_digest": entry.corpus_digest,
            "train_block": int(entry.train_block), "gpu_name": str(entry.gpu_name or ""),
            "marketcast": dict(mkt)}


__all__ = ["LINEAGE_FILE", "load_mkt_lineage", "record_mkt_lineage", "resolve_mkt_init",
           "MktLegRecipe", "has_generator_tree", "mkt_leg_contract", "mkt_leg_hours",
           "read_mkt_recipe", "train_mkt_leg",
           "verify_harvested_mkt_checkpoint"]
