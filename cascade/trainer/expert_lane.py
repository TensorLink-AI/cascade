"""Lane B (DEC-CA-0050) trainer-side planning: the opening gate, expert
selection on the king, the per-domain leg queue, and settling a finished leg
onto the CURRENT king (async delta rebase).

Everything here is trainer POLICY except what validators check: the merge
byte-check and the era-init rebuild (``apply_merges``, with the merge guards
``merge_row_overlap`` / ``check_finite_rows`` from ``[lane_b]
merge_guard_from_block``). The expert selection is NOT re-derived by
validators (DEC-CA-0050 "Trust model"): it is deterministic in the king
checkpoint + the affinity slice + the recorded exclusions, but trusted.

Affinity slice: the CONTEXT halves of the domain's pool series (no forecast
targets), first ``affinity_windows`` series of the domain by series id. The
routing decision reads inputs only, so nothing scored is trained on; the slice
is a pure function of the pool snapshot, so validators reproduce it.

Horizon targets (DEC-CA-0059, ``horizon:<h>``): the slice is the first
``affinity_windows`` contexts of ANY domain, and the routing counted is the
king's over the MASKED forecast patches of a horizon-``h`` decode — the
checkpoint wrapper's own window (context patches + ``ceil(h / patch)``
fully-masked patches, inputs ``[values × (1 − mask), mask]``), so the counts
read the mask and the context only, never a target value
(:func:`probe_horizon_routing`).

Multi-target legs (DEC-CA-0059): a leg declaring several targets trains the
UNION of each target's selection on the king (:func:`union_selection`), once,
under one leg budget; the union's rows are exclusive against every other open
leg (:meth:`LaneBQueue.start`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ── gate ──────────────────────────────────────────────────────────────────────


def selection_jaccard_layers(a: dict[int, list[int]], b: dict[int, list[int]],
                             ) -> dict[int, float]:
    """Per-layer Jaccard overlap of two expert selections (1.0 = same)."""
    out = {}
    for li in sorted(set(a) | set(b)):
        x, y = set(a.get(li, [])), set(b.get(li, []))
        out[int(li)] = 1.0 if not (x | y) else len(x & y) / len(x | y)
    return out


def topk_overlap_layers(a: list[np.ndarray], b: list[np.ndarray]) -> list[float]:
    """Token-level routing overlap per MoE layer (OLMoE "router saturation",
    arXiv 2409.02060): for each token ``|topk_a ∩ topk_b| / k``, averaged over
    the tokens. ``a``/``b`` are per-layer ``(N, k)`` expert ids of two models
    over the SAME tokens (top-k rows hold distinct ids). 1.0 = identical."""
    if len(a) != len(b):
        raise ValueError(f"layer count differs ({len(a)} vs {len(b)})")
    out = []
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            raise ValueError(f"top-k shapes differ ({x.shape} vs {y.shape}) — not the same tokens")
        if not x.size:
            out.append(1.0)
            continue
        hits = (x[:, :, None] == y[:, None, :]).sum(axis=(1, 2))
        out.append(float(hits.mean() / x.shape[1]))
    return out


@dataclass
class RoutingProbe:
    """One king's routing over one domain's affinity slice: per MoE layer the
    ``(N, k)`` top-k expert ids each token was routed to (``uint8`` at 256
    experts) and the per-expert assignment counts the ESFT selection ranks."""

    topk: list[np.ndarray]
    affinity: np.ndarray                      # (n_moe_layers, num_experts) int64

    @property
    def tokens(self) -> int:
        return int(self.topk[0].shape[0]) if self.topk else 0

    def select(self, cfg, exclude: dict[int, set[int]] | None = None) -> dict[int, list[int]]:
        # a zero-affinity expert never pads a lane-B selection (trainer
        # policy — validators do not re-derive rows, DEC-CA-0050)
        return select_routed_experts(self.affinity, p=cfg.esft_p, min_n=cfg.min_experts,
                                     max_n=cfg.max_experts, exclude=exclude)


def select_routed_experts(affinity, *, p: float, min_n: int, max_n: int,
                          exclude: dict[int, set[int]] | None = None) -> dict[int, list[int]]:
    """:func:`moe_model.select_domain_experts` that never pads up to
    ``min_n`` with experts the slice never routed to: a zero-affinity expert
    sees none of the domain's tokens, so training it is noise, and its rows
    would still be locked against every other leg. A layer then takes FEWER
    than ``min_n`` rows (possibly none). Implemented as extra exclusions so
    the digest-pinned model source (``moe_model.py`` feeds
    ``base_arch_digest``) and the direct lane's ``verify_selection`` replay
    stay byte-identical."""
    import torch

    from .moe_model import select_domain_experts

    aff = torch.as_tensor(affinity)
    banned: dict[int, set[int]] = {int(k): set(v) for k, v in (exclude or {}).items()}
    for li in range(aff.shape[0]):
        zero = {int(e) for e in torch.nonzero(aff[li] <= 0).reshape(-1).tolist()}
        if zero:
            banned.setdefault(li, set()).update(zero)
    return select_domain_experts(aff, p=p, min_n=min_n, max_n=max_n, exclude=banned or None)


def check_finite_rows(state: dict, selection: dict, names, *, what: str = "candidate") -> None:
    """Raise ``moe_model.MergeError`` when ``state`` holds a NaN / ±inf on a
    selected expert row — the consensus-gated merge guard (``[lane_b]
    merge_guard_from_block``), checked BEFORE the merge maths so the
    digest-pinned ``moe_model.py`` is untouched."""
    import torch

    from .moe_model import MergeError

    for li, pair in enumerate(names):
        rows = sorted(int(r) for r in selection.get(li, selection.get(str(li), [])))
        if not rows:
            continue
        idx = torch.tensor(rows, dtype=torch.long)
        for n in pair:
            if n in state and not bool(torch.isfinite(state[n][idx].float()).all()):
                raise MergeError(f"{n}: non-finite values on the {what}'s selected expert rows")


def routing_metrics(prev: RoutingProbe, cur: RoutingProbe, cfg) -> dict[str, list[float]]:
    """Per-layer ``{"overlap": [...], "jaccard": [...]}`` between two kings'
    routing of the same slice: token-level top-k overlap, and the Jaccard of
    their (unexcluded) ESFT selections."""
    jac = selection_jaccard_layers(prev.select(cfg), cur.select(cfg))
    return {"overlap": [round(v, 6) for v in topk_overlap_layers(prev.topk, cur.topk)],
            "jaccard": [round(jac[li], 6) for li in sorted(jac)],
            "tokens": min(prev.tokens, cur.tokens)}


def gate_open(lineage_tokens: int, routing: dict[str, dict | None], cfg) -> tuple[bool, str]:
    """Lane B opens once the king lineage has ``open_tokens`` (channel-time
    points) AND the lineage's routing has saturated: for every configured
    domain, EVERY MoE layer's routing of the domain's affinity slice agrees
    between the previous era's king and the current one — token-level top-k
    overlap >= ``routing_overlap`` and ESFT-selection Jaccard >=
    ``routing_jaccard``. ``routing[d]`` is :func:`routing_metrics` output, or
    absent/None when there is no earlier king to compare against (closed)."""
    if lineage_tokens < cfg.open_tokens:
        return False, f"lineage {lineage_tokens:.3g} < {cfg.open_tokens:.3g} tokens"
    missing = sorted(d for d in cfg.domains if not routing.get(d))
    if missing:
        return False, f"no previous king to compare routing on {missing}"
    empty = sorted(d for d in cfg.domains if not routing[d].get("tokens"))
    if empty:
        return False, f"empty affinity slice on {empty} (no routing evidence)"
    unstable = []
    for d in cfg.domains:
        m = routing[d]
        lo = [i for i, v in enumerate(m.get("overlap", [])) if v < cfg.routing_overlap]
        lj = [i for i, v in enumerate(m.get("jaccard", [])) if v < cfg.routing_jaccard]
        if lo or lj or not m.get("overlap") or not m.get("jaccard"):
            unstable.append(f"{d}(overlap<{cfg.routing_overlap} on layers {lo}, "
                            f"jaccard<{cfg.routing_jaccard} on layers {lj})")
    if unstable:
        return False, "routing unsaturated: " + "; ".join(unstable)
    return True, "open"


def union_selection(selections) -> dict[int, list[int]]:
    """Per-layer union of several expert selections (sorted ids) — the rows a
    multi-target leg trains (DEC-CA-0059). One selection ⇒ itself."""
    out: dict[int, set[int]] = {}
    for sel in selections:
        for li, rows in sel.items():
            out.setdefault(int(li), set()).update(int(x) for x in rows)
    return {li: sorted(v) for li, v in sorted(out.items())}


def routing_stale_union(pairs: list[tuple[RoutingProbe, RoutingProbe]],
                        selection: dict[int, list[int]], cfg,
                        exclude: dict[int, set[int]] | None = None) -> tuple[bool, dict]:
    """:func:`routing_stale` for a multi-target leg (DEC-CA-0059): ``pairs``
    holds ``(base, current)`` probes per declared target. The current king's
    re-planned selection is the union of each target's; the token overlap is
    the worst target's per layer. One pair ⇒ exactly :func:`routing_stale`."""
    if len(pairs) == 1:
        return routing_stale(pairs[0][0], pairs[0][1], selection, cfg, exclude=exclude)
    jac = selection_jaccard_layers({int(k): list(v) for k, v in selection.items()},
                                   union_selection(cur.select(cfg, exclude=exclude)
                                                   for _, cur in pairs))
    ov = [topk_overlap_layers(base.topk, cur.topk) for base, cur in pairs]
    detail = {"jaccard": [round(jac[li], 6) for li in sorted(jac)],
              "overlap": [round(min(col), 6) for col in zip(*ov, strict=True)] if ov else []}
    stale = (any(v < cfg.rebase_jaccard for v in detail["jaccard"])
             or any(v < cfg.rebase_overlap for v in detail["overlap"]))
    return stale, detail


def routing_stale(base: RoutingProbe, cur: RoutingProbe, selection: dict[int, list[int]],
                  cfg, exclude: dict[int, set[int]] | None = None) -> tuple[bool, dict]:
    """Rebase check: has the CURRENT king moved the leg's domain to different
    experts than its base king? Re-runs the ESFT selection on ``cur`` (same
    exclusions as the leg was planned under) against the leg's ``selection``
    per layer, plus the token-level top-k overlap of base vs current. Stale
    when any layer is below ``rebase_jaccard`` or ``rebase_overlap``. Returns
    ``(stale, {"jaccard": [...], "overlap": [...]})``."""
    jac = selection_jaccard_layers({int(k): list(v) for k, v in selection.items()},
                                   cur.select(cfg, exclude=exclude))
    detail = {"jaccard": [round(jac[li], 6) for li in sorted(jac)],
              "overlap": [round(v, 6) for v in topk_overlap_layers(base.topk, cur.topk)]}
    stale = (any(v < cfg.rebase_jaccard for v in detail["jaccard"])
             or any(v < cfg.rebase_overlap for v in detail["overlap"]))
    return stale, detail


# ── selection ─────────────────────────────────────────────────────────────────


def affinity_batches(contexts: list[np.ndarray], *, patch_size: int, max_patches: int,
                     batch: int = 16, scaler=None):
    """Standardised ``(patches, None)`` batches from 1-D context arrays, in the
    exact scaler the model trains under. Series are right-aligned to a common
    patch count per batch (shorter ones skipped below 2 patches)."""
    import torch

    from .moe_model import causal_standardize

    rows = []
    for c in contexts:
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        p = min(len(c) // patch_size, max_patches)
        if p >= 2:
            rows.append(c[-p * patch_size:])
    rows.sort(key=len)
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        w = min(len(r) for r in chunk)
        x = torch.as_tensor(np.stack([r[-w:] for r in chunk]), dtype=torch.float64)
        z, _, _ = (scaler or causal_standardize)(x)
        yield z.float().view(len(chunk), w // patch_size, patch_size), None


def load_king(king_dir: Path):
    import torch
    from safetensors.torch import load_file

    from .model_backend import checkpoint_backend

    backend = checkpoint_backend(king_dir)
    cfg = backend.CascadeMoEConfig.from_dict(json.loads((Path(king_dir) / "config.json").read_text())["moe"])
    model = backend.CascadeMoEModel(cfg)
    model.load_state_dict(load_file(str(Path(king_dir) / "weights.safetensors")))
    return model.to(torch.float32).eval()


def probe_routing(king_dir: Path, contexts: list[np.ndarray], cfg) -> RoutingProbe:
    """One forward of the king over a domain's affinity slice (the first
    ``affinity_windows`` contexts): the per-token routing and the per-expert
    counts the ESFT selection ranks (``bincount`` of the routed ids — the
    same integers :func:`moe_model.domain_expert_affinity` sums)."""
    from .moe_model import routing_topk

    model = load_king(king_dir)
    topk = routing_topk(model, affinity_batches(
        contexts[: cfg.affinity_windows], patch_size=model.cfg.patch_size,
        max_patches=model.cfg.context_length // model.cfg.patch_size,
        scaler=__import__(type(model).__module__, fromlist=["causal_standardize"]).causal_standardize))
    E = model.cfg.num_experts
    dtype = np.uint8 if E <= 256 else np.int32
    aff = np.stack([np.bincount(t.reshape(-1).numpy(), minlength=E) for t in topk]
                   ).astype(np.int64) if topk else np.zeros((0, E), dtype=np.int64)
    return RoutingProbe(topk=[t.numpy().astype(dtype) for t in topk], affinity=aff)


def horizon_of_target(target: str) -> int | None:
    """``h`` of a ``horizon:<h>`` target (DEC-CA-0059), else None."""
    from ..shared.config.v2 import HORIZON_TARGET_PREFIX

    if not str(target).startswith(HORIZON_TARGET_PREFIX):
        return None
    return int(str(target)[len(HORIZON_TARGET_PREFIX):])


def _stable_decode_steps() -> int:
    """The wrapper's ``STABLE_DECODE_STEPS`` (a constant of the generated
    ``forecast_wrapper.py`` the MoE checkpoint ships), read from the template
    itself so the horizon probe can never drift from the decode it mirrors."""
    import re

    from .toto2_trainer import _FORECAST_WRAPPER_PY

    m = re.search(r"^STABLE_DECODE_STEPS = ([0-9]+)$", _FORECAST_WRAPPER_PY, re.M)
    if m is None:
        raise RuntimeError("forecast wrapper template lost STABLE_DECODE_STEPS")
    return int(m.group(1))


def horizon_decode_window(horizon: int, *, patch_size: int, context_patches: int,
                          max_patches: int) -> tuple[int, int]:
    """``(context patches, masked patches)`` of the checkpoint wrapper's FIRST
    decode block at ``horizon`` (``toto2_trainer._decode_block_z`` /
    ``_decode_quantiles``): ``block = min(ceil(h / ps), STABLE_DECODE_STEPS //
    ps, max_patches − 2)`` masked patches after at most ``max_patches −
    block`` context patches."""
    stable = max(1, min(_stable_decode_steps() // patch_size, max_patches - 2))
    block = min(-(-int(horizon) // patch_size), stable)
    return max(0, min(int(context_patches), max_patches - block)), block


def horizon_affinity_batches(contexts: list[np.ndarray], *, horizon: int, patch_size: int,
                             context_patches: int, max_patches: int, batch: int = 16, scaler=None):
    """``(patches, mask, masked)`` batches for a horizon-``h`` decode: each
    context (right-aligned, standardised exactly as :func:`affinity_batches`)
    followed by ``masked`` all-zero, all-masked forecast patches. Inputs only —
    no target value is ever read."""
    import torch

    from .moe_model import causal_standardize

    ctx_p, block = horizon_decode_window(horizon, patch_size=patch_size,
                                         context_patches=context_patches,
                                         max_patches=max_patches)
    rows = []
    for c in contexts:
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        p = min(len(c) // patch_size, ctx_p)
        if p >= 2:
            rows.append(c[-p * patch_size:])
    rows.sort(key=len)
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        w = min(len(r) for r in chunk)
        x = torch.as_tensor(np.stack([r[-w:] for r in chunk]), dtype=torch.float64)
        z, _, _ = (scaler or causal_standardize)(x)
        n, cp = len(chunk), w // patch_size
        patches = torch.cat([z.float().view(n, cp, patch_size),
                             torch.zeros(n, block, patch_size)], dim=1)
        mask = torch.zeros(n, cp + block)
        mask[:, cp:] = 1.0
        yield patches, mask, block


def probe_horizon_routing(king_dir: Path, contexts: list[np.ndarray], cfg,
                          horizon: int) -> RoutingProbe:
    """The king's routing over the MASKED forecast patches of a horizon-``h``
    decode of the slice (DEC-CA-0059 horizon targets): per MoE layer the
    top-k of every masked-patch token only (context tokens are dropped), and
    their per-expert counts, which the ESFT selection ranks."""
    import torch

    model = load_king(king_dir)
    layers = model.moe_layers()
    k = model.cfg.top_k
    acc: list[list] = [[] for _ in layers]
    with torch.no_grad():
        for patches, mask, block in horizon_affinity_batches(
                contexts[: cfg.affinity_windows], horizon=horizon,
                patch_size=model.cfg.patch_size,
                context_patches=model.cfg.context_length // model.cfg.patch_size,
                max_patches=model.cfg.max_patches,
                scaler=__import__(type(model).__module__, fromlist=["causal_standardize"]).causal_standardize):
            model(patches, mask)
            n, P = patches.shape[0], patches.shape[1]
            for li, layer in enumerate(layers):
                # tokens are laid out (sequence, patch): keep the masked tail
                acc[li].append(layer.last_topk.cpu().view(n, P, k)[:, P - block:, :]
                               .reshape(-1, k))
    topk = [torch.cat(a) if a else torch.zeros((0, k), dtype=torch.int64) for a in acc]
    E = model.cfg.num_experts
    dtype = np.uint8 if E <= 256 else np.int32
    aff = np.stack([np.bincount(t.reshape(-1).numpy(), minlength=E) for t in topk]
                   ).astype(np.int64) if topk else np.zeros((0, E), dtype=np.int64)
    return RoutingProbe(topk=[t.numpy().astype(dtype) for t in topk], affinity=aff)


def probe_target_routing(king_dir: Path, contexts: list[np.ndarray], cfg,
                         target: str) -> RoutingProbe:
    """:func:`probe_routing` for a domain, :func:`probe_horizon_routing` for
    a ``horizon:<h>`` target."""
    h = horizon_of_target(target)
    if h is None:
        return probe_routing(king_dir, contexts, cfg)
    return probe_horizon_routing(king_dir, contexts, cfg, h)


# ── queue ─────────────────────────────────────────────────────────────────────


@dataclass
class LaneBSubmission:
    hotkey: str
    uid: int
    gen_ref: str
    domain: str
    commit_block: int
    # DEC-CA-0059: every declared target (``domain`` first); ``[]`` = the
    # single target ``domain``
    targets: list[str] = field(default_factory=list)

    @property
    def declared(self) -> tuple[str, ...]:
        return tuple(self.targets) if self.targets else (self.domain,)


def _targets_of(cfg) -> tuple[str, ...]:
    return tuple(getattr(cfg, "targets", None) or cfg.domains)


@dataclass
class OpenLeg:
    sub: LaneBSubmission
    selection: dict[int, list[int]]
    base_king_ref: str
    base_era: int
    # rows other legs held when this leg was planned (its selection skipped
    # them: open legs, finished-but-unsettled legs and pending merges) — the
    # rebase staleness check re-plans the domain on the current king under the
    # same exclusions, and the expert manifest records them
    exclude: dict[int, list[int]] = field(default_factory=dict)
    # DEC-CA-0059 item 8 miner-funded leg: the funded queue entry it consumed
    # ({"entry_ref", "gpu", "gpu_allow_fallback", "gpu_max_price_per_hour"});
    # {} = an operator-pool leg (``[lane_b] leg_funding = "operator"``)
    funding: dict = field(default_factory=dict)


DONE_KEEP = 4096
MAX_LEG_ATTEMPTS = 3


def submission_key(sub: LaneBSubmission) -> str:
    """One chain commitment's identity: hotkey + ref + commit block (a re-commit
    of the same ref at a NEW block is a new submission)."""
    return f"{sub.hotkey}|{sub.gen_ref}|{int(sub.commit_block)}"


@dataclass
class LaneBQueue:
    """FIFO per domain by commit block (never UID — UIDs recycle), at most
    ``legs_per_domain`` open legs per domain, and no two open legs owning
    the same expert row. A commitment is admitted ONCE: the chain keeps
    serving it every tick, so :meth:`submit` dedups against pending, in-flight
    (``open``) and finished (``done``) submissions by :func:`submission_key`."""

    pending: list[LaneBSubmission] = field(default_factory=list)
    open: list[OpenLeg] = field(default_factory=list)
    # submission keys whose leg finished with a checkpoint, or failed
    # MAX_LEG_ATTEMPTS times — never admitted again (bounded)
    done: list[str] = field(default_factory=list)
    # failed attempts per submission key (re-admitted until the cap)
    attempts: dict[str, int] = field(default_factory=dict)

    def submit(self, sub: LaneBSubmission, cfg) -> bool:
        """Queue ``sub`` unless it is already pending, in flight or done.
        Returns whether it was added."""
        unknown = [t for t in sub.declared if t not in _targets_of(cfg)]
        if unknown:
            raise ValueError(f"unknown lane-B target(s) {unknown!r}")
        cap = int(getattr(cfg, "max_domains_per_leg", 1) or 1)
        if len(sub.declared) > cap:
            raise ValueError(f"{len(sub.declared)} targets > max_domains_per_leg={cap}")
        key = submission_key(sub)
        if (key in self.done
                or any(submission_key(p) == key for p in self.pending)
                or any(submission_key(o.sub) == key for o in self.open)):
            return False
        self.pending.append(sub)
        return True

    def owned_rows(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for leg in self.open:
            for li, rows in leg.selection.items():
                out.setdefault(int(li), set()).update(rows)
        return out

    def next_dispatchable(self, cfg, eligible=None) -> list[LaneBSubmission]:
        """FIFO by commit block; a leg holds ``legs_per_domain`` capacity on
        EVERY target it declared (DEC-CA-0059), so it waits until all of
        them have room. ``eligible(sub)`` (DEC-CA-0059 item 8: funded, its
        hotkey free) filters BEFORE capacity is reserved, so an unfunded
        senior never holds a target's slot."""
        busy: dict[str, int] = {}
        for leg in self.open:
            for t in leg.sub.declared:
                busy[t] = busy.get(t, 0) + 1
        out = []
        for sub in sorted(self.pending, key=lambda s: (s.commit_block, s.hotkey)):
            if eligible is not None and not eligible(sub):
                continue
            if all(busy.get(t, 0) < cfg.legs_per_domain for t in sub.declared):
                out.append(sub)
                for t in sub.declared:
                    busy[t] = busy.get(t, 0) + 1
        return out

    def start(self, sub: LaneBSubmission, selection: dict[int, list[int]], base_king_ref: str,
              base_era: int, exclude: dict[int, set[int]] | None = None,
              held: dict[int, set[int]] | None = None, funding: dict | None = None) -> OpenLeg:
        """Open ``sub``'s leg on ``selection``. Refuses rows an open leg owns
        or ``held`` names (the driver's rows of finished-but-unsettled legs
        and of pending merges — exclusive until the era init consumes them)."""
        owned = self.owned_rows()
        for li, rows in (held or {}).items():
            owned.setdefault(int(li), set()).update(int(r) for r in rows)
        for li, rows in selection.items():
            if owned.get(int(li), set()) & set(rows):
                raise ValueError(f"selection overlaps an open leg's experts on layer {li}")
        self.pending.remove(sub)
        leg = OpenLeg(sub=sub, selection={int(k): list(v) for k, v in selection.items()},
                      base_king_ref=base_king_ref, base_era=base_era,
                      exclude={int(k): sorted(v) for k, v in (exclude or {}).items() if v},
                      funding=dict(funding or {}))
        self.open.append(leg)
        return leg

    def finish(self, leg: OpenLeg, *, ok: bool = True) -> None:
        """Close ``leg``. A leg that produced a checkpoint (``ok``) is DONE —
        its commitment is never admitted again. A failed leg frees its domain
        and its commitment may be re-admitted from the chain, at most
        ``MAX_LEG_ATTEMPTS`` times in all (a generator that fails every time
        must not hold a GPU lane forever)."""
        if leg not in self.open:
            return            # already expired/requeued (a late finish of an abandoned leg)
        self.open.remove(leg)
        key = submission_key(leg.sub)
        if not ok:
            self.attempts[key] = int(self.attempts.get(key, 0)) + 1
            if self.attempts[key] < MAX_LEG_ATTEMPTS:
                return
        self.attempts.pop(key, None)
        if key not in self.done:
            self.done.append(key)
            del self.done[:-DONE_KEEP]

    def requeue(self, leg: OpenLeg) -> None:
        """Put an open leg's submission back at its FIFO place, uncharged (no
        attempt counted): the leg was lost to the trainer, not to the miner —
        e.g. its thread died with the process."""
        if leg not in self.open:
            return
        self.open.remove(leg)
        key = submission_key(leg.sub)
        if key not in self.done and not any(submission_key(p) == key for p in self.pending):
            self.pending.append(leg.sub)

    def readmit(self, sub: LaneBSubmission) -> None:
        """Forget that ``sub`` finished, so its chain commitment is admitted
        again (a funded leg dropped unjudged at settlement through no fault
        of the miner — it retrains on the current king)."""
        key = submission_key(sub)
        self.done = [k for k in self.done if k != key]
        self.attempts.pop(key, None)

    def stale(self, leg: OpenLeg, era: int, cfg) -> bool:
        return era - leg.base_era > cfg.max_base_age_eras

    def expire_stale(self, era: int, cfg) -> list[OpenLeg]:
        """Close every open leg :meth:`stale` at ``era`` as a FAILED attempt
        (frees its domain slot and expert rows; re-admitted until
        ``MAX_LEG_ATTEMPTS``). Settlement drops a leg that old unjudged
        anyway, so nothing it could still deliver is lost."""
        out = [leg for leg in self.open if self.stale(leg, era, cfg)]
        for leg in out:
            self.finish(leg, ok=False)
        return out

    def to_json(self) -> str:
        return json.dumps({"pending": [asdict(p) for p in self.pending],
                           "open": [asdict(o) for o in self.open],
                           **({"done": list(self.done)} if self.done else {}),
                           **({"attempts": dict(self.attempts)} if self.attempts else {})},
                          sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> LaneBQueue:
        if not text.strip():
            return cls()
        o = json.loads(text)
        return cls(
            pending=[LaneBSubmission(**p) for p in o.get("pending", [])],
            open=[OpenLeg(sub=LaneBSubmission(**x["sub"]),
                          selection={int(k): v for k, v in x["selection"].items()},
                          base_king_ref=x["base_king_ref"], base_era=x["base_era"],
                          exclude={int(k): list(v)
                                   for k, v in (x.get("exclude") or {}).items()},
                          funding=dict(x.get("funding") or {}))
                  for x in o.get("open", [])],
            done=[str(k) for k in (o.get("done") or [])],
            attempts={str(k): int(v) for k, v in (o.get("attempts") or {}).items()},
        )


# ── settlement ────────────────────────────────────────────────────────────────


def settle_leg(king_now_dir: Path, king_base_dir: Path, leg_dir: Path,
               selection: dict[int, list[int]], out_dir: Path, *,
               refuse_nonfinite: bool = False) -> Path:
    """Write the leg carried onto the CURRENT king (``now + (leg − base)`` on
    its rows; byte-checked against its base). The result is a full checkpoint
    (current king's code/config) for the re-run verdict. Raises
    ``moe_model.MergeError`` on a leg that touched anything else (and, with
    ``refuse_nonfinite``, on a non-finite delta — the validators' merge guard)."""
    from safetensors.torch import load_file

    from .ckpt_digest import save_tensors_hashed
    from .moe_model import rebase_expert_delta

    now = load_file(str(Path(king_now_dir) / "weights.safetensors"))
    base = load_file(str(Path(king_base_dir) / "weights.safetensors"))
    leg = load_file(str(Path(leg_dir) / "weights.safetensors"))
    names = _moe_names(Path(king_now_dir))
    if refuse_nonfinite:
        check_finite_rows(leg, selection, names, what="leg")
    rebased = rebase_expert_delta(now, base, leg, selection, names)
    if refuse_nonfinite:
        check_finite_rows(rebased, selection, names, what="rebased leg")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "model.py", "forecast_wrapper.py"):
        shutil.copyfile(Path(king_now_dir) / f, out / f)
    save_tensors_hashed(rebased, out / "weights.safetensors")
    return out


def merge_rows(merge: dict) -> set[tuple[int, int]]:
    """The ``(MoE layer, expert row)`` cells one merge record writes."""
    return {(int(li), int(r)) for li, rows in (merge.get("selection") or {}).items()
            for r in rows}


def merge_row_overlap(merges: list[dict]) -> tuple[int, int] | None:
    """The first ``(layer, row)`` two of ``merges`` both write, else None.
    Two deltas summed onto one row were never judged together (DEC-CA-0050
    "disjoint merges commute" only holds for disjoint rows)."""
    seen: set[tuple[int, int]] = set()
    for m in merges:
        cells = merge_rows(m)
        hit = sorted(seen & cells)
        if hit:
            return hit[0]
        seen |= cells
    return None


def apply_merges(init_state: dict, merges: list[dict], load, names, *,
                 strict: bool = False) -> dict:
    """The merged era init: ``init + Σ (rebased − king)`` over each merge's
    rows, applied in list order (disjoint rows ⇒ order-free). ``load(pointer)``
    returns a state dict. Pure float32 CPU arithmetic — the trainer that builds
    the init and every validator that verifies it get identical bytes.

    ``strict`` (consensus, ``[lane_b] merge_guard_from_block``) refuses
    (``MergeError``) merges that overlap on any (layer, row) and any
    non-finite delta; the bytes of an accepted init are unchanged by it."""
    from .moe_model import MergeError, rebase_expert_delta

    if strict:
        hit = merge_row_overlap(merges)
        if hit is not None:
            raise MergeError(f"lane-B merges overlap on MoE layer {hit[0]} expert row {hit[1]}")
    state = init_state
    for m in merges:
        sel = {int(k): list(v) for k, v in m["selection"].items()}
        cand = load(m["rebased_pointer"])
        if strict:
            check_finite_rows(cand, sel, names, what="merge")
        state = rebase_expert_delta(state, load(m["king_pointer"]), cand, sel, names)
        if strict:
            check_finite_rows(state, sel, names, what="merged init")
    return state


# ── state at merge (DEC-CA-0050): what else an era init carries ──────────────
#
# ``apply_merges`` writes the SCORED weights (``weights.safetensors``, what
# validators rebuild bit-exactly). The init's lineage files must follow the
# same merge, or the next lane-A warm start would train on stale rows:
#
# * ``weights_stable.safetensors`` (the raw lineage weights a lane-A leg warm-
#   starts from under a power-EMA preset): the SAME delta on the same rows;
# * ``optimizer.safetensors`` (NorMuon momentum / row second moment; AdamW
#   moments, keyed ``<muon|adamw>.<param>.<state>``): the merged expert rows
#   are reset to zero — the moments were accumulated for the pre-merge row and
#   would push the merged row back (the direct lane's ``reset_edited_rows``);
# * ``power_ema.safetensors`` (``ema<σ>.<param>`` shadows + ``ema<σ>.__t__``):
#   the delta on every shadow's merged rows, so a ``power_ema_scope =
#   "lineage"`` average resumes around the merged weights (``"leg"`` restarts
#   the EMAs at the warm start, so there it is inert but kept consistent).


def merge_deltas(merges: list[dict], load, names) -> list[tuple[str, list[int], object]]:
    """``[(param name, rows, float32 delta)]`` per merge and MoE tensor, in
    merge order: exactly ``rebased − king`` on the merge's rows — the term
    :func:`apply_merges` adds (``load`` is cached by the caller)."""
    import torch

    out = []
    for m in merges:
        king, cand = load(m["king_pointer"]), load(m["rebased_pointer"])
        sel = {int(k): sorted(int(x) for x in v) for k, v in m["selection"].items()}
        for li, pair in enumerate(names):
            rows = sel.get(li, [])
            if not rows:
                continue
            idx = torch.tensor(rows, dtype=torch.long)
            for n in pair:
                out.append((n, rows, cand[n][idx].float() - king[n][idx].float()))
    return out


def apply_merge_deltas(state: dict, deltas, *, key=lambda n: n) -> dict:
    """``state`` with each delta added to its rows (float32 add, cast back to
    the stored dtype — :func:`moe_model.rebase_expert_delta`'s arithmetic).
    ``key(param name)`` maps a parameter to its tensor key in ``state``
    (identity for weights; the shadow key for a power EMA). A tensor absent
    from ``state`` is skipped."""
    import torch

    out = dict(state)
    copied: set[str] = set()
    for n, rows, delta in deltas:
        k = key(n)
        if k not in out:
            continue
        if k not in copied:
            out[k] = out[k].clone()
            copied.add(k)
        idx = torch.tensor(rows, dtype=torch.long)
        t = out[k]
        t[idx] = (t[idx].float() + delta).to(t.dtype)
    return out


def merge_power_ema(flat: dict, deltas) -> dict:
    """:func:`apply_merge_deltas` on every power-EMA shadow in ``flat``
    (``ema<σ>.<param>``; the prefixes are read off the ``.__t__`` keys)."""
    prefixes = [k[: -len(".__t__")] for k in flat if k.endswith(".__t__")]
    out = flat
    for pre in prefixes:
        out = apply_merge_deltas(out, deltas, key=lambda n, pre=pre: f"{pre}.{n}")
    return out


def reset_optimizer_rows(flat: dict, merges: list[dict], names) -> tuple[dict, int]:
    """``(flat, n)``: the flattened optimizer state (``_optim_state_tensors``
    keys ``muon.<param>.momentum`` / ``.row_v`` and ``adamw.<param>.exp_avg``
    / ``.exp_avg_sq``) with every merged expert row zeroed — a fresh
    optimizer's state for that row. ``n`` counts the tensors touched. Expert
    tensors are ``(E, …)`` in every layout (NorMuon steps the ``(E, d_out,
    d_in)`` view), so a row is the leading index everywhere. Step counters
    are left alone (they are per tensor, not per row)."""
    import torch

    rows_of: dict[str, set[int]] = {}
    for m in merges:
        for li, r in merge_rows(m):
            if li < len(names):
                for n in names[li]:
                    rows_of.setdefault(n, set()).add(r)
    out = dict(flat)
    touched = 0
    for k, t in flat.items():
        for n, rows in rows_of.items():
            if not any(k == f"{pre}.{n}.{st}" for pre, st in (
                    ("muon", "momentum"), ("muon", "row_v"),
                    ("adamw", "exp_avg"), ("adamw", "exp_avg_sq"))):
                continue
            if t.ndim == 0 or max(rows) >= t.shape[0]:
                raise ValueError(f"optimizer state {k} {tuple(t.shape)} has no expert row "
                                 f"{max(rows)}")
            z = t.clone()
            z[torch.tensor(sorted(rows), dtype=torch.long)] = 0
            out[k] = z
            touched += 1
    return out, touched


def _moe_names(ckpt_dir: Path) -> list[tuple[str, str]]:
    from .moe_model import CascadeMoEConfig

    cfg = CascadeMoEConfig.from_dict(json.loads((ckpt_dir / "config.json").read_text())["moe"])
    return [(f"blocks.{i}.ffn.w_in", f"blocks.{i}.ffn.w_out") for i in sorted(cfg.moe_blocks)
            if i < cfg.num_layers]
