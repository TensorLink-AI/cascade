"""``status/v2.json`` — the dashboards' live view of the v2 competition
(DEC-CA-0050): the Cascade-MoE-28M lane-A lineage and the domain-expert
lane B.

Unsigned and PRESENTATIONAL, like ``status/chain.json`` and the funded roster:
nothing a verdict depends on reads it. The trainer publishes it (it alone holds
the lane-B driver state and the king's routing probes); everything consensus-
relevant the pages show comes from the signed receipts instead.

Three pure builders and two publish entry points:

* :func:`king_routing_health` — per-MoE-layer load balance of the CURRENT king
  from its per-expert assignment counts (``RoutingProbe.affinity`` summed over
  the domains' affinity slices): max/mean, coefficient of variation, dead
  experts, normalised entropy, and a downsampled per-expert load heatmap.
* :func:`build_v2_status` — the whole document from the config, the block, the
  lane-B driver state (``LaneBDriverState`` or its JSON dict) and the health.
* :func:`v2_index_fields` — the compact lane-B columns a receipt contributes to
  its ``receipts/index.json`` row (``{}`` for every pre-lane-B receipt, so the
  index rows of archived rounds do not change).
* :func:`publish_v2_status` / :func:`publish_trainer_v2_status` — write the doc
  public-read (``publish_public_json``); the trainer entry is rate-limited and
  never raises (a dashboard must never stall a settlement).

Stdlib only at import time (numpy is imported lazily inside the health
builder), so ``summarize_receipt`` can call :func:`v2_index_fields` without
pulling in the eval stack.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime

log = logging.getLogger("cascade.shared.v2_status")

V2_STATUS_KEY = "status/v2.json"
SCHEMA = 1
HEAT_BINS = 32            # experts per layer are averaged into this many cells
HISTORY_MAX = 96          # routing-saturation rows kept in the public doc
OUTCOMES_MAX = 24         # dropped legs kept per domain
MIN_INTERVAL_S = 60.0     # leave headroom below the CLI's 180-second freshness limit

MODEL_NAME = "Cascade-MoE-28M"


# ── helpers ──────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _finite(x, nd: int = 6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return dict(getattr(obj, "__dict__", {}) or {})


def _first_attr(obj, names: tuple[str, ...]):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return n, v
    return None, None


# ── go-live ──────────────────────────────────────────────────────────────────


def v2_from_block(cfg) -> int:
    """The v2 activation block (``[v2] from_block``; 0 = not announced): a
    ``cfg.v2`` section when present, else the training contract's
    ``v2_from_block`` (where ``load_chain_config`` lands ``[v2] from_block``).
    Lane B has an independent launch and cannot stand in for v2's block."""
    v2 = getattr(cfg, "v2", None)
    fb = getattr(v2, "from_block", None) if v2 is not None else None
    if not fb:
        fb = getattr(getattr(cfg, "training", None), "v2_from_block", None)
    try:
        return max(0, int(fb or 0))
    except (TypeError, ValueError):
        return 0


def golive(cfg, block: int | None) -> dict:
    """``{from_block, state, blocks_to_go}`` with state ``unannounced`` (the
    0 placeholder), ``scheduled`` (block ahead) or ``live``."""
    fb = v2_from_block(cfg)
    opening = {}
    v2 = getattr(cfg, "v2", None)
    if getattr(v2, "commit_open_at_utc", "") or getattr(v2, "commit_from_block", 0):
        floor = int(v2.commit_from_block)
        opening = {"commit_open_at_utc": v2.commit_open_at_utc,
                   "commit_from_block": floor,
                   "admission_state": ("closed" if not floor else
                       "active" if fb > 0 and block is not None and int(block) >= fb else "queued")}
    if fb <= 0:
        date = getattr(getattr(cfg, "v2", None), "launch_at_utc", "")
        if date:
            return {**opening, "from_block": 0, "state": "scheduled", "blocks_to_go": None,
                    "launch_at_utc": date, "alignment": "first era at or after target"}
        return {**opening, "from_block": 0, "state": "unannounced", "blocks_to_go": None}
    if block is None or int(block) < fb:
        return {**opening, "from_block": fb, "state": "scheduled",
                "blocks_to_go": None if block is None else fb - int(block)}
    return {**opening, "from_block": fb, "state": "live", "blocks_to_go": 0}


# ── model ────────────────────────────────────────────────────────────────────

_PARAMS_CACHE: dict | None = None


def moe_model_card(contract=None) -> dict:
    """Architecture headline of the pinned MoE preset: total / active params
    (``moe_model.param_counts`` on a meta-device model — no memory, no
    weights), experts, top-k, MoE blocks. ``{}`` when torch is unavailable."""
    if getattr(contract, "base_arch", "") == "cascade_pyramid":
        try:
            import torch

            from ..trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel, param_counts

            config = CascadeMoEConfig.from_contract(contract)
            with torch.device("meta"):
                model = CascadeMoEModel(config)
            counts = param_counts(model)
            return {"name": contract.arch_preset, "total_params": counts["total"],
                    "active_params": counts["active"], "num_experts": config.num_experts,
                    "experts_per_layer": list(config.experts_per_layer),
                    "top_k": config.top_k, "moe_blocks": list(config.moe_blocks),
                    "num_layers": config.num_layers, "source": "configured_v2_contract"}
        except (ImportError, ValueError):
            return {}
    global _PARAMS_CACHE
    if _PARAMS_CACHE is not None:
        return dict(_PARAMS_CACHE)
    try:
        import torch

        from ..trainer.moe_model import CascadeMoEConfig, CascadeMoEModel, param_counts

        mc = CascadeMoEConfig()
        with torch.device("meta"):
            model = CascadeMoEModel(mc)
        pc = param_counts(model)
        _PARAMS_CACHE = {"name": MODEL_NAME, "total_params": int(pc["total"]),
                         "active_params": int(pc["active"]), "num_experts": mc.num_experts,
                         "top_k": mc.top_k, "moe_blocks": list(mc.moe_blocks),
                         "num_layers": mc.num_layers}
    except Exception as e:  # noqa: BLE001 — the card is cosmetic
        log.debug("v2 status: model card unavailable: %s", e)
        return {}
    return dict(_PARAMS_CACHE)


# ── routing health ───────────────────────────────────────────────────────────


def king_routing_health(load, *, bins: int = HEAT_BINS, layers: list[int] | None = None,
                        expert_counts: list[int] | None = None) -> dict:
    """Load-balance stats of one king from per-expert assignment counts.

    ``load`` is ``(n_moe_layers, num_experts)`` counts (a ``RoutingProbe``'s
    ``affinity``, or several summed). Per layer: ``max_over_mean`` (1.0 =
    perfectly balanced), ``cv`` (std/mean), ``dead`` (experts routed zero
    tokens), ``entropy`` (Shannon entropy of the load share ÷ log E; 1.0 =
    uniform) and ``heat`` — the experts in index order averaged into ``bins``
    cells, each cell's mean load ÷ the layer mean (1.0 = fair share). Empty
    input → ``{}``."""
    import numpy as np

    a = np.asarray(load, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return {}
    n_layers, max_experts = a.shape
    if expert_counts is not None and (len(expert_counts) != n_layers or
            any(not 1 <= count <= max_experts for count in expert_counts)):
        return {}
    out_layers = []
    for li in range(n_layers):
        E = expert_counts[li] if expert_counts is not None else max_experts
        nb = max(1, min(int(bins), E))
        edges = np.linspace(0, E, nb + 1).round().astype(int)
        row = a[li, :E]
        tot = float(row.sum())
        mean = tot / E if E else 0.0
        if tot <= 0:
            out_layers.append({"layer": int(layers[li]) if layers else li, "tokens": 0,
                               "max_over_mean": None, "cv": None, "dead": int(E),
                               "entropy": None, "heat": [0.0] * nb})
            continue
        p = row / tot
        nz = p[p > 0]
        ent = float(-(nz * np.log(nz)).sum() / math.log(E)) if E > 1 else 1.0
        heat = [round(float(row[edges[i]:edges[i + 1]].mean() / mean), 4)
                if edges[i + 1] > edges[i] else 0.0 for i in range(nb)]
        out_layers.append({
            "layer": int(layers[li]) if layers else li,
            "tokens": int(round(tot)),
            "max_over_mean": _finite(row.max() / mean, 4),
            "cv": _finite(row.std() / mean, 4),
            "dead": int((row == 0).sum()),
            "entropy": _finite(ent, 4),
            "heat": heat,
        })
    return {"experts": int(E), "bins": nb, "layers": out_layers}


def driver_king_load(driver, king_pointer: str, *, block: int | None = None,
                     allow_probe: bool = False):
    """Sum the cached routing probes of ``king_pointer`` (one per domain's
    affinity slice) held by a ``LaneBDriver`` → ``(load, n_domains)`` or
    ``(None, 0)``. With ``allow_probe`` a domain missing from the cache is
    probed through the driver's own (cached) ``_probe`` — the same probe the
    next tick's gate/selection reuses, so it is computed once per king."""
    import numpy as np

    if driver is None or not king_pointer:
        return None, 0
    lb = getattr(getattr(driver, "cfg", None), "lane_b", None)
    domains = list(getattr(lb, "domains", ()) or ())
    acc, n = None, 0
    probes = getattr(driver, "_probes", None) or {}
    lock = getattr(driver, "_cache_lock", None)
    for d in domains:
        hit = None
        try:
            if lock is not None:
                with lock:
                    items = list(probes.items())
            else:
                items = list(probes.items())
            for (ptr, dom, _digest), probe in items:
                if ptr == king_pointer and dom == d:
                    hit = probe
            if hit is None and allow_probe and block is not None:
                hit = driver._probe(king_pointer, king_pointer, d, int(block))
        except Exception as e:  # noqa: BLE001 — health is best-effort
            log.debug("v2 status: probe %s unavailable: %s", d, e)
            hit = None
        aff = getattr(hit, "affinity", None)
        if aff is None or not getattr(aff, "size", 0):
            continue
        aff = np.asarray(aff, dtype=np.float64)
        acc = aff if acc is None else acc + aff
        n += 1
    return acc, n


# ── lane-B view ──────────────────────────────────────────────────────────────

_PAYOUT_NAMES = {
    # canonical page key: attribute names in preference order (the go-live
    # build may re-denominate rounds → eras; both spellings are read)
    "half_life": ("half_life_eras", "half_life_rounds"),
    "floor": ("floor",),
    "ttl": ("win_ttl_eras", "ttl_eras"),
}


def _lb_targets(lb) -> tuple[str, ...]:
    """The lane-B slot list (DEC-CA-0059 ``targets``: domains then the
    ``horizon:<h>`` slots), else the domains (a pre-0059 config)."""
    return tuple(str(t) for t in (getattr(lb, "targets", None)
                                  or getattr(lb, "domains", ()) or ()))


def full_pay_tiers_view(cfg) -> list | None:
    """``[[lcb_threshold, full_eras], ...]`` of a section's gain-tiered
    full-pay period (owner 2026-10-20), None when it has none."""
    tiers = getattr(cfg, "full_pay_tiers", None)
    if not tiers:
        return None
    return [[_finite(t), int(n)] for t, n in tiers]


def payout_terms(lb) -> dict:
    """The lane-B payout schedule in one unit-tagged dict:
    ``{unit, full_pay_tiers, half_life, floor, ttl, share, slots,
    remainder}`` — ``full_pay_tiers`` the gain-tiered full-pay period
    (``[[lcb_threshold, full_eras], ...]``; a win's own tier is on its
    payout record)."""
    out: dict = {"full_pay_tiers": full_pay_tiers_view(lb)}
    # DEC-CA-0050 go-live: the decay is counted in ERAS (``half_life_rounds``
    # keeps its name) once the config carries the win TTL; the original build
    # counted settlements
    unit = "eras" if hasattr(lb, "win_ttl_eras") else "settlements"
    for key, names in _PAYOUT_NAMES.items():
        name, val = _first_attr(lb, names)
        out[key] = _finite(val) if val is not None else None
        if name and name.endswith("_eras"):
            unit = "eras"
    out["unit"] = unit
    out["share"] = _finite(getattr(lb, "share", None))
    # DEC-CA-0059: one slot per TARGET (domains + horizon slots)
    out["slots"] = len(_lb_targets(lb))
    out["remainder"] = str(getattr(lb, "remainder", "burn") or "burn")
    return out


def _sel_summary(sel) -> dict:
    """``{layer: [rows]}`` → ``{"per_layer": {layer: n}, "rows": {layer: [ids]}}``."""
    rows = {str(int(k)): sorted(int(x) for x in v) for k, v in (sel or {}).items()}
    return {"per_layer": {k: len(v) for k, v in rows.items()}, "rows": rows}


def _leg_row(sub: dict) -> dict:
    return {"hotkey": str(sub.get("hotkey", "")), "uid": int(sub.get("uid", -1)),
            "gen_ref": str(sub.get("gen_ref", "")),
            "commit_block": int(sub.get("commit_block", 0) or 0)}


def _queue_dict(st: dict) -> dict:
    import json

    q = st.get("queue")
    if isinstance(q, dict):
        return q
    try:
        return json.loads(q) if q else {}
    except (TypeError, ValueError):
        return {}


def _layer_min(rows_by_domain: dict, key: str) -> list:
    """Per-layer minimum of ``key`` across domains (the gate's binding value)."""
    cols = [list(m.get(key) or []) for m in rows_by_domain.values() if isinstance(m, dict)]
    if not cols:
        return []
    n = max(len(c) for c in cols)
    out = []
    for i in range(n):
        vals = [c[i] for c in cols if i < len(c) and c[i] is not None]
        out.append(_finite(min(vals), 4) if vals else None)
    return out


def _downsample(rows: list, n: int) -> list:
    if len(rows) <= n:
        return list(rows)
    step = (len(rows) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [rows[i] for i in idx]


def lane_b_view(lb, state, *, block: int | None) -> dict:
    """The lane-B part of the document from the config and the driver state."""
    st = _as_dict(state)
    domains = [str(d) for d in (getattr(lb, "domains", ()) or ())]
    targets = list(_lb_targets(lb))
    routing = st.get("routing") or {}
    q = _queue_dict(st)
    per: dict[str, dict] = {d: {"pending": [], "open": [], "finished": [], "pending_merge": False,
                                "selection": None, "dropped": []} for d in targets}

    def slot(d):
        return per.setdefault(str(d), {"pending": [], "open": [], "finished": [],
                                       "pending_merge": False, "selection": None,
                                       "dropped": []})

    for p in q.get("pending") or []:
        slot(p.get("domain", ""))["pending"].append(_leg_row(p))
    for o in q.get("open") or []:
        sub = o.get("sub") or {}
        row = {**_leg_row(sub), "base_era": int(o.get("base_era", 0) or 0),
               **_sel_summary(o.get("selection"))}
        s = slot(sub.get("domain", ""))
        s["open"].append(row)
        s["selection"] = row["rows"]
    for f in st.get("finished") or []:
        leg = f.get("leg") or {}
        sub = leg.get("sub") or {}
        row = {**_leg_row(sub), "base_era": int(leg.get("base_era", 0) or 0),
               "train_block": int(f.get("train_block", 0) or 0),
               **_sel_summary(leg.get("selection"))}
        s = slot(sub.get("domain", ""))
        s["finished"].append(row)
        s["selection"] = s["selection"] or row["rows"]
    for m in st.get("pending_merges") or []:
        # a winning delta waiting for the next era init: which domain is it?
        rp = str(m.get("rebased_pointer", ""))
        for d in list(per):
            if (f"-expert-{d}-" in rp
                    or any(w.get("domain") == d for w in m.get("qualified_wins", []))):
                per[d]["pending_merge"] = True
                per[d]["selection"] = per[d]["selection"] or _sel_summary(m.get("selection"))["rows"]
    for o in (st.get("outcomes") or [])[::-1]:
        if o.get("outcome") == "accepted":
            continue
        s = slot(o.get("domain", ""))
        if len(s["dropped"]) < OUTCOMES_MAX:
            s["dropped"].append({"round_id": str(o.get("round_id", "")),
                                 "hotkey": str(o.get("hotkey", "")),
                                 "uid": int(o.get("uid", -1)),
                                 "outcome": str(o.get("outcome", "")),
                                 "detail": o.get("detail")})
    for s in per.values():
        s["selection"] = _sel_summary(s["selection"]) if s["selection"] else None

    merges = []
    total_merged = 0
    for era, rec in sorted((st.get("merged_inits") or {}).items(), key=lambda kv: int(kv[0])):
        ms = rec.get("merges") or []
        doms = []
        for m in ms:
            rp = str(m.get("rebased_pointer", ""))
            doms.extend(d for d in domains if f"-expert-{d}-" in rp)
            doms.extend((m.get("acceptance") or {}).get("targets", []))
        total_merged += len(ms)
        merges.append({"era": int(era), "pointer": str(rec.get("pointer", "")),
                       "n": len(ms), "domains": sorted(set(doms)),
                       "rounds": sorted({str(m.get("round_id", "")) for m in ms if m.get("round_id")})})

    hist = []
    for r in st.get("routing_history") or []:
        doms = r.get("domains") or {}
        hist.append({"era": int(r.get("era", 0) or 0), "block": int(r.get("block", 0) or 0),
                     "lineage_tokens": int(r.get("lineage_tokens", 0) or 0),
                     "overlap": _layer_min(doms, "overlap"),
                     "jaccard": _layer_min(doms, "jaccard")})
    latest = None
    if routing.get("domains"):
        latest = {"era": int(routing.get("era", 0) or 0), "block": int(routing.get("block", 0) or 0),
                  "domains": {str(d): {"overlap": [_finite(v, 4) for v in (m.get("overlap") or [])],
                                       "jaccard": [_finite(v, 4) for v in (m.get("jaccard") or [])],
                                       "tokens": int(m.get("tokens", 0) or 0)}
                              for d, m in (routing.get("domains") or {}).items()}}
    opened = int(st.get("gate_opened_block", 0) or 0)
    return {
        "active": bool(getattr(lb, "active", lambda _b: False)(block)),
        "from_block": max(0, int(getattr(lb, "from_block", 0) or 0) - 1),
        "domains": domains,
        # DEC-CA-0059: every emission slot (domains, then ``horizon:<h>``)
        "targets": targets,
        "horizon_targets": [t for t in targets if t not in domains],
        "max_targets_per_leg": int(getattr(lb, "max_domains_per_leg", 1) or 1),
        "acceptance_from_block": int(getattr(lb, "acceptance_from_block", 0)),
        "payout": payout_terms(lb),
        "verdict": {"alpha": _finite(getattr(lb, "alpha", None)),
                    "gain_margin": _finite(getattr(lb, "gain_margin", None)),
                    "tau_domain": _finite(getattr(lb, "tau_domain", None)),
                    "tau_public": _finite(getattr(lb, "tau_public", None)),
                    "min_target_windows": getattr(lb, "min_target_windows", None),
                    "min_target_clusters": getattr(lb, "min_target_clusters", None)},
        "gate": {
            "open": opened > 0,
            "opened_block": opened,
            "reason": str(routing.get("reason", "") or ("open" if opened else "")),
            "lineage_tokens": int(st.get("lineage_tokens", 0) or 0),
            "lineage_meter": ("checkpoint_measured" if st.get("lineage_meter_version") == 1
                              else "legacy_budget_estimate"),
            "legacy_budget_tokens": int(st.get("legacy_budget_tokens", 0) or 0),
            "unmeasured_epochs": sum(
                "tokens_seen" not in row for row in (st.get("lineage_epochs") or {}).values()
                if isinstance(row, dict)),
            "open_tokens": int(getattr(lb, "open_tokens", 0) or 0),
            "routing_overlap": _finite(getattr(lb, "routing_overlap", None)),
            "routing_jaccard": _finite(getattr(lb, "routing_jaccard", None)),
            "latest": latest,
            "history": _downsample(hist, HISTORY_MAX),
        },
        "legs": {"legs_per_domain": getattr(lb, "legs_per_domain", None),
                 "leg_hours": _finite(getattr(lb, "leg_hours", None)),
                 "max_base_age_eras": getattr(lb, "max_base_age_eras", None),
                 "experts_min": getattr(lb, "min_experts", None),
                 "experts_max": getattr(lb, "max_experts", None)},
        "per_domain": per,
        "merges": merges,
        "merged_total": total_merged,
    }


def _pending_row(pe) -> dict:
    """A lane-C pending entry (``PendingEntry`` / its dict / a receipt-derived
    row) as ``{target, kind, hotkey, round_id, gain_lcb}``."""
    d = _as_dict(pe)
    hk = d.get("hotkey") or (d.get("entry") or {}).get("miner_hotkey", "")
    return {**d, "hotkey": str(hk or "")}


def lane_c_view(lc, state=None, *, block: int | None, trainer: dict | None = None,
                mc=None, era_blocks: int | None = None) -> dict:
    """The lane-C (DEC-CA-0056) part of the document: activation, the split,
    the payout and verdict terms, the leg policy and, per target, the slot
    HOLDER (``state``: a validator-side :class:`~cascade.shared.lane_c.
    LaneCState` or its JSON dict — the consensus record; None ⇒ holders
    unknown) and the trainer's leg counts (``trainer``: the trainer driver's
    ``LaneCDriver.status()`` dict — queue, open / finished legs, slots).
    ``payout.slots`` counts the targets ACTIVE at ``block`` (``mc`` =
    ``[marketcast]``; MarketCast-only lane C: none while it is off, so the
    page shows lane C's share burning). ``competitions`` (concurrent
    competitions, owner 2026-10-20): per configured competition its weight,
    normalised share of lane C (``frac``), state (pending | active |
    winding_down | ended at ``block``; ``era_blocks`` = the era length in
    force there, for the wind-down window), start / end blocks, wind-down
    rule, targets and per-target slot fraction."""
    from .lane_c import lane_c_split

    st = _as_dict(state)
    slots = st.get("slots") or {}
    pending = [_pending_row(pe) for pe in (st.get("pending") or ())]
    tr = dict(trainer or {})
    tr_targets = tr.get("targets") or {}
    targets = [str(t) for t in (getattr(lc, "targets", ()) or ())]
    per = {}
    for t in targets:
        w = slots.get(t) or {}
        legs = tr_targets.get(t) or {}
        per[t] = {
            "holder": ({"hotkey": str(w.get("hotkey", "")), "uid": int(w.get("uid", -1)),
                        "round_id": str(w.get("round_id", "")),
                        "win_block": int(w.get("win_block", 0) or 0),
                        "gain_lcb": _finite(w.get("gain_lcb"), 4),
                        "trained_pointer": str(w.get("trained_pointer", "")),
                        # receipt-derived holders carry the paid slot's age / multiplier
                        "age": w.get("age"),
                        "multiplier": _finite(w.get("multiplier")),
                        # the win's gain-tiered full-pay period (fixed at the win)
                        "tier": _finite(w.get("tier")), "full_eras": w.get("full_eras"),
                        "below_floor": bool(w.get("below_floor", False))} if w else None),
            "queued": int(legs.get("queued", 0) or 0), "open": int(legs.get("open", 0) or 0),
            "finished": int(legs.get("finished", 0) or 0),
            "leg_slots": int(legs.get("slots", getattr(lc, "legs_per_target", 0)) or 0),
            # entries carried to the next settlement: ``confirm`` (passed as the
            # target's best, re-scored next settlement) / ``deferred``
            "pending": [{"hotkey": str(pe.get("hotkey", "")), "kind": str(pe.get("kind", "")),
                         "round_id": str(pe.get("round_id", "")),
                         "gain_lcb": _finite(pe.get("gain_lcb"), 4)}
                        for pe in pending if str(pe.get("target", "")) == t],
        }
    fb = int(getattr(lc, "from_block", 0) or 0)
    comps, slot_frac = lane_c_split(lc, int(block or 0), int(era_blocks or 1), mc)
    for t in per:
        per[t]["competition"] = t.partition(":")[0]
        per[t]["slot_frac"] = _finite(slot_frac.get(t, 0.0))
    return {
        "active": bool(getattr(lc, "active", lambda _b: False)(block)),
        "from_block": fb,
        "state": ("off" if not fb else ("live" if block is not None and int(block) > fb
                                        else "scheduled")),
        "share": _finite(getattr(lc, "share", None)),
        "takes_from": str(getattr(lc, "takes_from", "lane_b")),
        "targets": targets,
        "payout": {"unit": "eras", "full_pay_tiers": full_pay_tiers_view(lc),
                   "half_life": _finite(getattr(lc, "half_life_rounds", None)),
                   "floor": _finite(getattr(lc, "floor", None)),
                   "ttl": getattr(lc, "win_ttl_eras", None),
                   "share": _finite(getattr(lc, "share", None)),
                   "slots": len(slot_frac),
                   "remainder": str(getattr(lc, "remainder", "burn"))},
        "verdict": {"alpha": _finite(getattr(lc, "alpha", None)),
                    "gain_margin": _finite(getattr(lc, "gain_margin", None)),
                    "min_target_windows": getattr(lc, "min_target_windows", None),
                    "min_target_clusters": getattr(lc, "min_target_clusters", None),
                    "n_windows": getattr(lc, "n_windows", None),
                    "guard_public": {t: list(v) for t, v in
                                     (getattr(lc, "guard_public", ()) or ())}},
        "legs": {"leg_hours": _finite(getattr(lc, "leg_hours", None)),
                 "legs_per_target": getattr(lc, "legs_per_target", None),
                 "max_base_age_eras": getattr(lc, "max_base_age_eras", None)},
        "competitions": [{"kind": c["kind"], "weight": _finite(c["weight"]),
                          "frac": _finite(c["frac"]), "state": c["state"],
                          "from_block": int(c["from_block"]),
                          "until_block": int(c["until_block"]),
                          "wind_down": c["wind_down"],
                          "wind_down_max_eras": int(c["wind_down_max_eras"]),
                          "targets": list(c["targets"]),
                          "slot_frac": _finite(c["slot_frac"])} for c in comps],
        "per_target": per,
        # where the holders came from: "validator" (a LaneCState), "receipt"
        # (the latest signed receipt's lane_c_pay / lane_c rows) or "" (none)
        "holders_source": str(st.get("source", "validator" if slots else "") or ""),
        "trainer": {"queue": dict(tr.get("queue") or {}),
                    "open": list(tr.get("open") or []),
                    "finished": int(tr.get("finished", 0) or 0),
                    "legs_started": int(tr.get("legs_started", 0) or 0),
                    "settlements": list(tr.get("settlements") or []),
                    "outcomes": list(tr.get("outcomes") or [])} if tr else {},
    }


def build_v2_status(cfg, *, block: int | None, state=None, health: dict | None = None,
                    era_index: int | None = None, king_pointer: str = "",
                    health_domains: int = 0, as_of: str | None = None,
                    lane_c_state=None, lane_c_trainer: dict | None = None,
                    lane_a: dict | None = None) -> dict:
    """The whole ``status/v2.json`` document. Pure given its inputs. The
    ``lane_c`` part appears once ``[lane_c] from_block`` is typed (an inert
    lane C leaves the document byte-identical); ``lane_a`` (the latest
    receipt's ladder / court / slash / king check, :func:`lane_a_view`)
    appears when given."""
    lb = getattr(cfg, "lane_b", None)
    doc = {
        "schema": SCHEMA,
        "as_of": as_of or _now(),
        "block": None if block is None else int(block),
        "netuid": getattr(getattr(cfg, "subnet", None), "netuid", None),
        "golive": golive(cfg, block),
        "model": moe_model_card(getattr(getattr(cfg, "training", None), "v2_contract", None)),
        "king": {"era": era_index, "pointer": str(king_pointer or ""),
                 "routing_health": health or {},
                 "health_source": (f"routing of {health_domains} domain affinity slice(s)"
                                   if health else "")},
        "lane_b": lane_b_view(lb, state, block=block) if lb is not None else {},
    }
    if lane_a:
        doc["lane_a"] = dict(lane_a)
    lc = getattr(cfg, "lane_c", None)
    if lc is not None and int(getattr(lc, "from_block", 0) or 0):
        eb = None
        if block is not None:
            try:
                from .era import era_length_blocks

                eb = int(era_length_blocks(cfg.round, int(block)))
            except Exception:  # noqa: BLE001 — no era grid: the wind-down window is unknown
                eb = None
        doc["lane_c"] = lane_c_view(lc, lane_c_state, block=block, trainer=lane_c_trainer,
                                    mc=getattr(cfg, "marketcast", None), era_blocks=eb)
    return doc


# ── receipt-derived views (lane-A ladder, lane-C holders) ────────────────────


def ladder_fields(emission) -> dict | None:
    """The lane-A ladder of a receipt's DEC-CA-0055 ``emission`` record
    (:func:`cascade.shared.emission.emission_record`) in one flat dict:
    ``king_share`` (the ladder's basis points as a fraction, else the policy's
    ``king_share``), ``king_share_bp`` / ``prev_bp``, the ladder ``event``
    (a crowning adds its margin-tiered start: ``crown_lcb`` / ``crown_units``
    / ``crown_tier`` / ``crown_start_bp``, None otherwise),
    the former-king court (``court_hotkey`` / ``court_until_block``), the key
    ``slashed`` flag with its ``slash_reason`` / ``slash_until`` / reign
    ``key_lapses`` (the escalating slash) and the policy terms (``former_share``, ``king_slash``,
    ``lane_a_cap``, ``prelaunch_burn``). ``None`` when there is no record."""
    if not isinstance(emission, dict) or not emission:
        return None
    pol = emission.get("policy") or {}
    lad = emission.get("ladder") or {}
    bp = lad.get("bp")
    share = (float(bp) / 10000.0 if bp is not None
             else _finite(pol.get("king_share")))
    return {"boundary": emission.get("boundary"),
            "king_since_block": emission.get("king_since_block"),
            "king_share": None if share is None else round(share, 6),
            "king_share_bp": bp, "prev_bp": lad.get("prev_bp"),
            "event": lad.get("event"),
            "crown_lcb": _finite(lad.get("crown_lcb")),
            "crown_units": lad.get("crown_units") or None,
            "crown_tier": _finite(lad.get("crown_tier")),
            "crown_start_bp": lad.get("crown_start_bp"),
            "court_hotkey": lad.get("court_hotkey") or None,
            "court_until_block": lad.get("court_until_block"),
            "slashed": bool(lad.get("slashed", False)),
            "slash_reason": lad.get("slash_reason") or None,
            "slash_until": lad.get("slash_until"),
            "key_lapses": int(lad.get("key_lapses", 0) or 0),
            "funded_by": lad.get("funded_by") or None,
            "former_share": _finite(pol.get("former_share")),
            "king_slash": _finite(pol.get("king_slash")),
            "lane_a_cap": _finite(pol.get("lane_a_cap")),
            "prelaunch_burn": _finite(pol.get("prelaunch_burn"))}


def king_check_fields(check) -> dict | None:
    """A receipt's DEC-CA-0055 ``king_check`` record, compacted: result,
    round, block, LCB, margin, king pointer / hotkey, reasons. ``None`` when
    absent."""
    if not isinstance(check, dict) or not check:
        return None
    return {"result": str(check.get("result", "") or ""),
            "round_id": str(check.get("round_id", "") or ""),
            "block": int(check.get("block", 0) or 0),
            "lcb": _finite(check.get("lcb")), "margin": _finite(check.get("margin")),
            "king_pointer": str(check.get("king_pointer", "") or ""),
            "king_hotkey": str(check.get("king_hotkey", "") or ""),
            "reasons": [str(r) for r in (check.get("reasons") or ())]}


def lane_a_view(receipt: dict | None) -> dict:
    """The ``status/v2.json`` ``lane_a`` block from a receipt body (signed
    JSON): the ladder share, court and slash, the last king check and the
    round. ``{}`` when the receipt carries neither record."""
    if not isinstance(receipt, dict):
        return {}
    ladder = ladder_fields(receipt.get("emission"))
    check = king_check_fields(receipt.get("king_check"))
    if ladder is None and check is None:
        return {}
    return {"round_id": str(receipt.get("round_id", "")),
            "epoch_start_block": receipt.get("epoch_start_block"),
            "ladder": ladder, "king_check": check}


def lane_c_state_from_receipt(receipt: dict | None) -> dict | None:
    """The lane-C slot HOLDERS from a receipt body, in the shape
    :func:`lane_c_view` reads (``{"slots": {target: {...}}, "pending": [...],
    "source": "receipt"}``).

    The trainer holds no validator slot state (the slot fold is
    ``cascade.validator.lane_c``), so the publisher reads the holders off the
    latest signed receipt: ``lane_c_pay.wins`` are the PAID live slots
    (holder hotkey / uid / win block; an expired or deregistered holder's slot
    burns and is not listed), enriched with the holder's ``confirmed`` verdict
    row when this receipt carries it (round, gain LCB, checkpoint). Rows with
    outcome ``pending`` / ``deferred`` are the entries carried to the next
    settlement. ``None`` when the receipt carries no lane-C record."""
    if not isinstance(receipt, dict):
        return None
    pay, rows = receipt.get("lane_c_pay"), receipt.get("lane_c")
    if not isinstance(pay, dict) and not isinstance(rows, list):
        return None
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    slots: dict = {}
    for w in (pay or {}).get("wins") or ():
        if not isinstance(w, dict) or not w.get("target"):
            continue
        t = str(w["target"])
        conf = next((r for r in rows if r.get("target") == t and r.get("hotkey") == w.get("hotkey")
                     and r.get("outcome") == "confirmed"), {})
        slots[t] = {"hotkey": str(w.get("hotkey", "")), "uid": int(w.get("uid", -1)),
                    "win_block": int(w.get("win_block", 0) or 0),
                    "age": w.get("age"), "multiplier": _finite(w.get("multiplier")),
                    "round_id": str(conf.get("origin_round") or receipt.get("round_id", "")
                                    if conf else ""),
                    "gain_lcb": w.get("gain_lcb", conf.get("gain_lcb")),
                    "tier": _finite(w.get("tier")), "full_eras": w.get("full_eras"),
                    "below_floor": bool(w.get("below_floor", False)),
                    "trained_pointer": str(conf.get("trained_pointer", "") or "")}
    pending = [{"target": str(r.get("target", "")),
                "kind": "confirm" if r.get("outcome") == "pending" else "deferred",
                "hotkey": str(r.get("hotkey", "")), "round_id": str(receipt.get("round_id", "")),
                "gain_lcb": r.get("gain_lcb")}
               for r in rows if r.get("outcome") in ("pending", "deferred")]
    return {"slots": slots, "pending": pending, "source": "receipt"}


def latest_receipt_body(store, anchor: str = "") -> dict | None:
    """The latest receipt body (raw JSON) from the manifest bucket: the
    anchor validator's single-writer trail first, then the shared pointer.
    Presentational (the status doc is unsigned): not signature-verified —
    the pages read the signed receipt themselves. Never raises."""
    import json

    try:
        from .hippius import RECEIPT_LATEST_KEY, receipt_latest_key

        keys = ([receipt_latest_key(anchor)] if anchor else []) + [RECEIPT_LATEST_KEY]
    except Exception:  # noqa: BLE001
        return None
    for key in keys:
        try:
            body = json.loads(store.get_text(key))
        except Exception:  # noqa: BLE001 — absent / unreadable ⇒ next
            continue
        if isinstance(body, dict):
            return body
    return None


# ── receipt index columns ────────────────────────────────────────────────────


def _failed_guard(row: dict) -> str:
    reason = str(row.get("reason") or "")
    for pfx in ("regression on domain ", "regression on "):
        if reason.startswith(pfx):
            return reason[len(pfx):].split(":", 1)[0].strip()
    return ""


def v2_index_fields(receipt) -> dict:
    """Lane-B columns for a receipt's ``receipts/index.json`` row — ``{}``
    when the receipt carries no lane-B record (every pre-v2 receipt).

    * ``lane_b_verdicts`` — per judged expert leg: hotkey, domain, stage,
      passed, reason, target LCB, the failed guard (if any), worst guard LCB;
    * ``lane_b_pay`` — the payout record verbatim (small: one row per domain);
    * ``emission_split`` — ``{lane_a, lane_b, burn}`` (plus ``lane_c`` once
      the receipt carries ``lane_c_pay``) fractions of the signed weight
      vector: lane B = the paid domain wins, lane C = the paid target slots,
      lane A = the court (``reward_uids``) net of any lane-B/C pay on the same
      uid, burn = the rest;
    * ``ladder`` / ``king_check`` — the DEC-CA-0055 lane-A records
      (:func:`ladder_fields`, :func:`king_check_fields`), whenever present.
    """
    rows = getattr(receipt, "lane_b", None)
    pay = getattr(receipt, "lane_b_pay", None)
    c_rows = getattr(receipt, "lane_c", None)
    c_pay = getattr(receipt, "lane_c_pay", None)
    out: dict = {}
    # DEC-CA-0055 lane-A records (present only on a receipt that carries them,
    # so archived rows do not change): the ladder share / court / slash and
    # the king check
    ladder = ladder_fields(getattr(receipt, "emission", None))
    if ladder is not None:
        out["ladder"] = ladder
    check = king_check_fields(getattr(receipt, "king_check", None))
    if check is not None:
        out["king_check"] = check
    if rows is None and pay is None and c_rows is None and c_pay is None:
        return out
    if c_rows is not None:
        out["lane_c_verdicts"] = [
            {"hotkey": r.get("hotkey"), "uid": r.get("uid"), "target": r.get("target"),
             "stage": r.get("stage"), "passed": bool(r.get("passed")), "reason": r.get("reason"),
             "gain_lcb": _finite(r.get("gain_lcb")), "gain": _finite(r.get("gain")),
             "phase": r.get("phase"), "outcome": r.get("outcome")}
            for r in c_rows if "error" not in r]
    if c_pay is not None:
        out["lane_c_pay"] = dict(c_pay)
    if rows is not None:
        vs = []
        for r in rows:
            guards = {**(r.get("guard_lcbs") or {}), **(r.get("public_lcbs") or {})}
            worst = min((v for v in guards.values() if isinstance(v, (int, float))
                         and math.isfinite(v)), default=None)
            v = {"hotkey": r.get("hotkey"), "domain": r.get("domain"),
                 "stage": r.get("stage"), "passed": bool(r.get("passed")),
                 "reason": r.get("reason"), "target_lcb": _finite(r.get("target_lcb")),
                 "failed_guard": _failed_guard(r) or None,
                 "worst_guard_lcb": _finite(worst)}
            if r.get("targets"):
                # DEC-CA-0059 multi-target leg: one row per declared target
                v["targets"] = list(r["targets"])
            vs.append(v)
        out["lane_b_verdicts"] = vs
    if pay is not None:
        out["lane_b_pay"] = dict(pay)
    if pay is not None or c_pay is not None:
        # lane C's pay counts as lane C, never as burn
        split = emission_split(list(getattr(receipt, "weights", ()) or ()),
                               list(getattr(receipt, "reward_uids", ()) or ()),
                               pay if pay is not None else {"wins": []}, lane_c_pay=c_pay)
        if split is not None:
            out["emission_split"] = split
    return out


def emission_split(weights: list[float], reward_uids: list[int], pay: dict,
                   lane_c_pay: dict | None = None) -> dict | None:
    """``{lane_a, lane_b, burn}`` of a combined weight vector. Exact: a paid
    win's slot is ``share/slots × multiplier`` — read from ``pay`` when it
    carries ``share``/``slots``, else recovered from a winner whose uid holds
    no lane-A weight (its whole weight is its lane-B pay). ``None`` when no
    exact decomposition exists (an overlapping uid with no recoverable slot)."""
    tot = sum(float(w) for w in weights)
    if tot <= 0:
        return None
    w = [float(x) / tot for x in weights]
    wins = [x for x in (pay.get("wins") or []) if isinstance(x, dict)]
    court = {int(u) for u in reward_uids}
    slot = None
    if pay.get("share") is not None and pay.get("slots"):
        slot = float(pay["share"]) / float(pay["slots"])
    else:
        for x in wins:
            u, m = int(x.get("uid", -1)), float(x.get("multiplier", 0) or 0)
            if 0 <= u < len(w) and u not in court and m > 0:
                slot = w[u] / m
                break
    lane_b = 0.0
    paid_on: dict[int, float] = {}
    for x in wins:
        u, m = int(x.get("uid", -1)), float(x.get("multiplier", 0) or 0)
        if not 0 <= u < len(w):
            continue
        if u in court:
            if slot is None:
                return None
            amt = slot * m
        else:
            amt = w[u]
        paid_on[u] = paid_on.get(u, 0.0) + amt
        lane_b += amt
    lane_c = 0.0
    if lane_c_pay is not None:
        # DEC-CA-0056: the record carries share and the per-target slot
        # fractions (competition weight, then per target), so each paid slot
        # is exact; a record without slot_frac splits share/slots equally
        c_share = float(lane_c_pay.get("share", 0.0) or 0.0)
        c_frac = lane_c_pay.get("slot_frac")
        c_slot = c_share / max(1, int(lane_c_pay.get("slots", 1) or 1))
        for x in lane_c_pay.get("wins") or []:
            u = int(x.get("uid", -1))
            if not 0 <= u < len(w):
                continue
            slot_c = (c_share * float(c_frac.get(str(x.get("target")), 0.0) or 0.0)
                      if isinstance(c_frac, dict) else c_slot)
            amt = slot_c * float(x.get("multiplier", 0) or 0)
            paid_on[u] = paid_on.get(u, 0.0) + amt
            lane_c += amt
    lane_a = sum(w[u] - paid_on.get(u, 0.0) for u in court if 0 <= u < len(w))
    burn = max(0.0, 1.0 - lane_a - lane_b - lane_c)
    out = {"lane_a": round(lane_a, 6), "lane_b": round(lane_b, 6), "burn": round(burn, 6)}
    if lane_c_pay is not None:
        out["lane_c"] = round(lane_c, 6)
    return out


# ── publish ──────────────────────────────────────────────────────────────────


def publish_v2_status(store, doc: dict) -> str:
    from .hippius import publish_public_json

    return publish_public_json(store, V2_STATUS_KEY, doc)


_LAST: dict[int, tuple[float, str]] = {}


def publish_trainer_v2_status(sched, block: int, *, min_interval_s: float = MIN_INTERVAL_S,
                              allow_probe: bool = True, clock=time.monotonic) -> bool:
    """The trainer hook: build + publish ``status/v2.json`` from a
    ``RollingScheduler`` (duck-typed: ``.cfg``, ``.lane_b`` driver or None,
    ``.state.current`` era with ``.index`` / ``.king()``, ``.r.manifest_store()``).

    Rate-limited per scheduler to ``min_interval_s``; also refresh unchanged
    state as a liveness heartbeat. Runs on mainnet before v2 is live — the page
    reads the go-live state (``unannounced`` / ``scheduled``). Never raises.
    Returns True when a document was uploaded."""
    import json

    key = id(sched)
    now = clock()
    last = _LAST.get(key)
    if last is not None and now - last[0] < min_interval_s:
        return False
    try:
        cfg = sched.cfg
        drv = getattr(sched, "lane_b", None)
        cur = getattr(getattr(sched, "state", None), "current", None)
        king = cur.king() if cur is not None and hasattr(cur, "king") else None
        ptr = str(getattr(king, "trained_pointer", "") or "")
        health, n = {}, 0
        if drv is not None and ptr:
            load, n = driver_king_load(drv, ptr, block=block, allow_probe=allow_probe)
            if load is not None:
                training = getattr(cfg, "training", None)
                if hasattr(training, "at_block"):
                    training = training.at_block(int(getattr(cur, "start_block", block)))
                mc = moe_model_card(training)
                health = king_routing_health(load, layers=mc.get("moe_blocks") or None,
                    expert_counts=mc.get("experts_per_layer"))
        # DEC-CA-0056: the trainer's lane-C driver (may be absent) feeds the
        # leg counts via ``status()``; its ``.st`` is the driver's leg state,
        # NOT the validators' slot state, so holders are not read from it
        # (they come from the latest receipt below)
        lc_drv = getattr(sched, "lane_c", None)
        lc_status = None
        if lc_drv is not None and callable(getattr(lc_drv, "status", None)):
            try:
                lc_status = lc_drv.status()
            except Exception as e:  # noqa: BLE001 — a status read never stalls the trainer
                log.debug("lane C status unavailable: %s", e)
        # The trainer has no validator slot state: lane-C holders and the
        # lane-A ladder / king check come from the latest receipt
        receipt = None
        try:
            anchor = str(getattr(getattr(cfg, "manifest", None), "validator_hotkey", "") or "")
            receipt = latest_receipt_body(sched.r.manifest_store(), anchor)
        except Exception as e:  # noqa: BLE001 — presentational
            log.debug("v2 status: latest receipt unavailable: %s", e)
        if lc_drv is not None and callable(getattr(lc_drv, "observe_receipt", None)):
            try:
                lc_drv.observe_receipt(receipt)
                lc_status = lc_drv.status()
            except Exception:  # noqa: BLE001 — telemetry cannot hold settlement
                log.warning("lane C submission receipt observation unavailable")
        doc = build_v2_status(cfg, block=int(block), state=getattr(drv, "st", None),
                              health=health, health_domains=n,
                              era_index=getattr(cur, "index", None), king_pointer=ptr,
                              lane_c_trainer=lc_status,
                              lane_c_state=lane_c_state_from_receipt(receipt),
                              lane_a=lane_a_view(receipt))
        from .submission_progress import receipt_progress

        doc["submission_receipt"] = receipt_progress(receipt)
        # Publish the era's actual init, including accepted expert merges,
        # before its first settlement. The trained king is a different pointer.
        if cur is not None and hasattr(cur, "warm_start_ckpt"):
            init = str(cur.warm_start_ckpt or "")
            doc["warm_start"] = {
                "pointer": init, "mode": "checkpoint" if init else "random_init",
                "size": str(getattr(cur, "warm_start_size", "") or ""),
                "era": int(cur.index), "era_start_block": int(cur.start_block),
                "generation": int(getattr(cur, "generation", 0)),
                "member_index": int(getattr(cur, "member_index", 0)),
            }
        # Agent-facing policy comes from the running trainer's resolved config;
        # never ask miners to infer live rules from their packaged chain.toml.
        doc["policy"] = {
            section: {name: getattr(obj, name) for name in names if hasattr(obj, name)}
            for section, obj, names in (
                ("training", getattr(cfg, "training", None),
                 ("target_train_hours", "ref_throughput_tokens_per_s", "training_steps", "batch_size", "context_length")),
                ("generation", getattr(cfg, "generation", None),
                 ("enabled", "window_seconds", "vcpus", "max_parallel_jobs", "target_points", "max_length")),
                ("lane_b", getattr(cfg, "lane_b", None),
                 ("min_experts", "max_experts", "max_domains_per_leg", "acceptance_from_block", "merge_guard_from_block")),
                ("round", getattr(cfg, "round", None), ("funded_field_cap", "finalist_cap")),
                ("marketcast", getattr(cfg, "marketcast", None),
                 ("from_block", "base_ref", "base_sha256", "base_arch", "suites")),
            ) if obj is not None
        }
        # Publish the actual effective era contract, not the pre-v2 base table.
        # Its digest is an integrity check; telemetry remains unsigned discovery.
        if hasattr(cfg, "throne_contracts_at"):
            from dataclasses import asdict

            from .manifest import contract_digest, contract_payload

            start = int(getattr(cur, "start_block", block))
            contract = cfg.throne_contracts_at(start)[0]
            doc["training_contract"] = {"schema": 1, "era_start_block": start,
                                        "body": contract_payload(contract),
                                        "digest": contract_digest(contract)}
            doc["policy"]["training"] = {name: getattr(contract, name) for name in
                ("target_train_hours", "ref_throughput_tokens_per_s", "training_steps", "batch_size", "context_length")}
            doc["policy"]["training"]["point_budget"] = contract.train_tokens
            doc["policy"]["training"]["budget_basis"] = "updates" if contract.training_steps else "reference_gpu_hours"
            for section in ("lane_b", "lane_c", "marketcast", "generator", "generation", "eval", "scoring"):
                doc["policy"][section] = asdict(getattr(cfg, section))
            doc["policy"]["generator"] = asdict(cfg.generator_for_contract(contract))
            doc["policy"]["generator"].pop("real_corpus_dir", None)
            doc["policy"]["round"].update({name: getattr(cfg.round, name) for name in
                ("reveal_margin_blocks", "funded_pod_skus", "heat_train_hours")})
        from ..funding.providers import ROLLING_PROVIDERS, spending_controls

        doc["operations"] = sched.r.__dict__.get("_operations_status")
        doc["launch_preflight"] = sched.r.__dict__.get("_v2_launch_preflight_status")
        doc["compute"] = {"rolling_providers": list(ROLLING_PROVIDERS),
                          "spending_controls": spending_controls(),
                          "total_cap_enforced_by_standard_fund": False,
                          "selection": "ordered_capacity_fallback",
                          "one_key_per_provider": True,
                          "fallback_after_uncertain_create": False}
        doc["agent_protocol"] = {"schema": 1, "guide": "llms.txt",
                                 "capabilities": ["status", "preflight", "verify_quick", "lane_b_test", "lane_c_test",
                                                  "submission_status"],
                                 "submission_status": {"schema": 1, "worker_liveness": "not_measured",
                                                       "pending_reveals": "redacted"}}
        burned = getattr(getattr(sched, "ops", None), "burned", None)
        doc["admission"] = {"lane_a_spent_hotkeys": sorted(burned()) if callable(burned) else None,
                            "one_live_leg_per_hotkey": True}
        queue = getattr(getattr(sched, "ops", None), "queue", None)
        if callable(queue):
            funded = queue()
            if funded is not None:
                doc["lane_a_queue"] = {"as_of": doc["as_of"], **funded.public_view()}
        body = json.dumps({k: v for k, v in doc.items() if k != "as_of"}, sort_keys=True)
        publish_v2_status(sched.r.manifest_store(), doc)
        _LAST[key] = (now, body)
        return True
    except Exception as e:  # noqa: BLE001 — a dashboard must never stall the trainer
        log.warning("v2 status publish failed (ignored): %s", e)
        _LAST[key] = (now, last[1] if last else "")
        return False


__all__ = [
    "V2_STATUS_KEY", "build_v2_status", "driver_king_load", "emission_split", "golive",
    "king_check_fields", "king_routing_health", "ladder_fields", "lane_a_view", "lane_b_view",
    "lane_c_state_from_receipt", "lane_c_view", "latest_receipt_body", "moe_model_card",
    "payout_terms",
    "publish_trainer_v2_status", "publish_v2_status", "v2_from_block", "v2_index_fields",
]
