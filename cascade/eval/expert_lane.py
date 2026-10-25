"""Lane B (DEC-CA-0050) — the domain-expert competition's verdict and payout.

A lane-B leg trains a trainer-selected expert subset of the king for ONE
declared domain. It is judged on two different kinds of statistical test, run
on the same paired-cluster bootstrap the lane-A duel uses:

* **Gain (superiority)** on the target domain's windows:
  ``LCB_α(rel) ≥ gain_margin``, ``rel = (king − chal) / king``, AND a strictly
  positive point estimate of ``rel`` — an unchanged model (identical scores,
  LCB = point = 0) never passes, whatever the margin. The target windows must
  come from at least ``min_target_windows`` windows over at least
  ``min_target_clusters`` distinct clusters (upstream sources): 100 windows of
  one feed are one observation to the cluster bootstrap, so fewer clusters is
  ``inconclusive`` (never a pass).
* **Guards (non-inferiority)** on every other pool domain and on each public
  benchmark metric (GIFT-Eval / BOOM / TIME / FEV): the candidate may not be
  worse by more than ``tau`` with confidence ``1 − α`` — equivalently
  ``LCB_α(rel) ≥ −tau`` (the upper bound on the DEGRADATION is ≤ tau).

The leg passes only if the gain test AND every guard pass. Requiring every
guard to reject its own null is an intersection-union test (Berger 1982): each
guard at level α keeps the family at level α, so the guards need NO
multiplicity correction — adding a benchmark never loosens or tightens the
others. (The gain test is a single hypothesis; no correction either.)

Small domains: a domain with fewer than ``min_guard_windows`` windows would
produce a bootstrap too wide to ever pass a non-inferiority bar (blocking
every leg) or, tested loosely, a free pass. Such domains are POOLED into one
``"_rest"`` guard instead, so they are still guarded in aggregate; only when
even the pool is too small is that recorded as unguarded.

Pure numpy, deterministic in ``seed`` (each test derives its own sub-seed), so
validators and ``cascade-audit`` replay it bit-for-bit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .bootstrap import (
    _bag_geomeans,
    _cluster_sums,
    _seed_to_int,
    cluster_codes,
    paired_bootstrap_lcb_aggregated,
)

REST_GUARD = "_rest"


@dataclass(frozen=True)
class ExpertLaneParams:
    alpha: float = 0.05
    B: int = 10_000
    gain_margin: float = 0.0            # target-domain superiority bar
    tau_domain: float = 0.01            # other-domain non-inferiority margin
    tau_public: float = 0.01            # public-benchmark non-inferiority margin
    min_target_windows: int = 100
    min_target_clusters: int = 5        # distinct sources among the target windows
    min_guard_windows: int = 50
    min_public_items: int = 10


DEFAULT_PARAMS = ExpertLaneParams()


@dataclass
class ExpertVerdict:
    passed: bool
    reason: str
    target_domain: str
    target_lcb: float
    target_n: int
    guard_lcbs: dict[str, float] = field(default_factory=dict)
    guard_n: dict[str, int] = field(default_factory=dict)
    public_lcbs: dict[str, float] = field(default_factory=dict)
    unguarded: list[str] = field(default_factory=list)
    target_gain: float = float("nan")    # point estimate of rel on the target windows
    target_clusters: int = 0
    inconclusive: str = ""               # "" | "windows" | "clusters"


def paired_geomean_lcb(king: Sequence[float], chal: Sequence[float], *, alpha: float,
                       B: int, seed: int | str) -> float:
    """LCB_α of ``1 − geomean(chal)/geomean(king)`` over paired items (e.g. the
    97 GIFT-Eval configurations, each already a ratio to seasonal-naive; lower
    better), items resampled with replacement. NaN when empty."""
    k = np.log(np.maximum(np.asarray(king, dtype=np.float64), 1e-12))
    c = np.log(np.maximum(np.asarray(chal, dtype=np.float64), 1e-12))
    if k.shape != c.shape or k.ndim != 1:
        raise ValueError(f"paired 1-D scores required; got {k.shape} vs {c.shape}")
    n = k.size
    if n == 0:
        return float("nan")
    idx = np.random.default_rng(_seed_to_int(seed)).integers(0, n, size=(B, n))
    d = (c[idx] - k[idx]).mean(axis=1)          # log geomean ratio chal/king per bag
    return float(np.quantile(1.0 - np.exp(d), alpha))


def paired_point_rel(kq, ka, km, cq, ca, cm, clusters=None) -> float:
    """The full-sample ``(king − chal) / king`` on the bootstrap's own metric
    (every cluster once — the identity bag). NaN when empty."""
    n = int(np.asarray(kq).shape[0])
    if n == 0:
        return float("nan")
    codes = cluster_codes(clusters, n)
    g = int(codes.max()) + 1
    idx = np.arange(g)[None, :]
    k = float(_bag_geomeans(*_cluster_sums(kq, ka, km, codes, g), idx)[0])
    c = float(_bag_geomeans(*_cluster_sums(cq, ca, cm, codes, g), idx)[0])
    return (k - c) / (k if abs(k) >= 1e-9 else 1e-9)


def _domain_lcb(mask, clusters, kq, ka, km, cq, ca, cm, p: ExpertLaneParams, seed: str) -> float:
    cl = None if clusters is None else [c for c, m in zip(clusters, mask, strict=True) if m]
    return paired_bootstrap_lcb_aggregated(
        kq[mask], ka[mask], km[mask], cq[mask], ca[mask], cm[mask],
        alpha=p.alpha, B=p.B, seed=seed, clusters=cl,
    )


def judge_expert_leg(
    *,
    target_domain: str,
    domains: Sequence[str],
    king_qloss: np.ndarray, king_abs_target: np.ndarray, king_mase: np.ndarray,
    chal_qloss: np.ndarray, chal_abs_target: np.ndarray, chal_mase: np.ndarray,
    clusters: Sequence | None = None,
    public: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    params: ExpertLaneParams = DEFAULT_PARAMS,
    seed: int | str = 42,
) -> ExpertVerdict:
    """Judge one lane-B leg. Per-window arrays are the private-pool scores of
    the king and the candidate on the SAME windows (``domains[i]`` labels window
    i); ``public`` maps a benchmark metric name (``"gift_crps"``, ``"fev_mase"``
    …) to paired per-item ``(king, chal)`` scores."""
    p = params
    dom = np.asarray(list(domains), dtype=object)
    kq, ka, km = (np.asarray(a, dtype=np.float64) for a in (king_qloss, king_abs_target, king_mase))
    cq, ca, cm = (np.asarray(a, dtype=np.float64) for a in (chal_qloss, chal_abs_target, chal_mase))
    if dom.shape[0] != kq.shape[0]:
        raise ValueError("domains must label every window")
    s = str(seed)

    tmask = dom == target_domain
    tn = int(tmask.sum())
    v = ExpertVerdict(passed=False, reason="", target_domain=target_domain,
                      target_lcb=float("nan"), target_n=tn)
    if tn < p.min_target_windows:
        v.reason = f"inconclusive: {tn} target windows < {p.min_target_windows}"
        v.inconclusive = "windows"
        return v
    tcl = None if clusters is None else [c for c, m in zip(clusters, tmask, strict=True) if m]
    v.target_clusters = tn if tcl is None else len(set(tcl))
    if v.target_clusters < p.min_target_clusters:
        v.reason = (f"inconclusive: {v.target_clusters} target clusters < "
                    f"{p.min_target_clusters}")
        v.inconclusive = "clusters"
        return v
    v.target_lcb = _domain_lcb(tmask, clusters, kq, ka, km, cq, ca, cm, p, f"{s}|gain|{target_domain}")
    v.target_gain = paired_point_rel(kq[tmask], ka[tmask], km[tmask], cq[tmask], ca[tmask],
                                     cm[tmask], tcl)

    # non-inferiority guards on every other domain (small ones pooled)
    small = np.zeros_like(tmask)
    for d in sorted({str(x) for x in dom if x != target_domain}):
        m = dom == d
        n = int(m.sum())
        if n >= p.min_guard_windows:
            v.guard_n[d] = n
            v.guard_lcbs[d] = _domain_lcb(m, clusters, kq, ka, km, cq, ca, cm, p, f"{s}|guard|{d}")
        else:
            small |= m
            v.unguarded.append(d)
    if small.any():
        n = int(small.sum())
        if n >= p.min_guard_windows:
            v.guard_n[REST_GUARD] = n
            v.guard_lcbs[REST_GUARD] = _domain_lcb(
                small, clusters, kq, ka, km, cq, ca, cm, p, f"{s}|guard|{REST_GUARD}")
            v.unguarded = []
    for name, (k, c) in sorted((public or {}).items()):
        if len(k) < p.min_public_items:
            v.unguarded.append(name)
            continue
        v.public_lcbs[name] = paired_geomean_lcb(k, c, alpha=p.alpha, B=p.B, seed=f"{s}|public|{name}")

    if not v.target_lcb >= p.gain_margin:              # NaN fails
        v.reason = f"no gain: target LCB {v.target_lcb:.4f} < margin {p.gain_margin}"
        return v
    if not v.target_gain > 0.0:                         # a zero effect never passes
        v.reason = f"no gain: target point gain {v.target_gain:.4f} <= 0"
        return v
    for d, lcb in v.guard_lcbs.items():
        if not lcb >= -p.tau_domain:
            v.reason = f"regression on domain {d}: LCB {lcb:.4f} < -{p.tau_domain}"
            return v
    for name, lcb in v.public_lcbs.items():
        if not lcb >= -p.tau_public:
            v.reason = f"regression on {name}: LCB {lcb:.4f} < -{p.tau_public}"
            return v
    v.passed = True
    v.reason = "merge"
    return v


# ── multi-target legs (DEC-CA-0059) ──────────────────────────────────────────


@dataclass
class TargetsVerdict:
    """One leg's verdict over its k declared targets: a per-target
    :class:`ExpertVerdict` (gain at ``alpha``/k; the guards, computed ONCE,
    are copied into every conclusive target's verdict) plus the shared guard
    outcome. A guard failure blocks every target; a target failing only its
    gain test is simply unpaid."""

    targets: tuple[str, ...]
    alpha: float                       # the per-target gain level (alpha / k)
    verdicts: dict[str, ExpertVerdict] = field(default_factory=dict)
    guard_reason: str = ""             # "" = every guard passed (or not computed)
    guards_computed: bool = False

    @property
    def k(self) -> int:
        return len(self.targets)

    @property
    def won(self) -> list[str]:
        return [t for t in self.targets if self.verdicts[t].passed]


HORIZON_PREFIX = "horizon:"


def _target_mask(target: str, dom: np.ndarray, hz: np.ndarray | None) -> np.ndarray:
    if target.startswith(HORIZON_PREFIX):
        h = int(target[len(HORIZON_PREFIX):])
        if hz is None:
            return np.zeros(dom.shape[0], dtype=bool)
        return hz == h
    return dom == target


def judge_expert_targets(
    *,
    targets: Sequence[str],
    domains: Sequence[str],
    king_qloss: np.ndarray, king_abs_target: np.ndarray, king_mase: np.ndarray,
    chal_qloss: np.ndarray, chal_abs_target: np.ndarray, chal_mase: np.ndarray,
    clusters: Sequence | None = None,
    public: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    params: ExpertLaneParams = DEFAULT_PARAMS,
    seed: int | str = 42,
    horizons: Sequence[int | None] | None = None,
    horizon_targets: Sequence[str] = (),
    guards_only: bool = False,
) -> TargetsVerdict:
    """Judge one lane-B leg on each of its declared ``targets`` (DEC-CA-0059).

    * **Per target** (a pool domain, or ``horizon:<h>`` = every window scored
      at horizon ``h``; ``horizons[i]`` labels window i): the same gain test
      as :func:`judge_expert_leg` — window / cluster floors, ``LCB ≥
      gain_margin`` and a positive point gain — at ``alpha / k`` for k
      declared targets (Bonferroni across the leg's own targets, on top of
      any other correction). Seeds ``<seed>|gain|<target>``.
    * **Guards, once**, at ``alpha`` (intersection-union: no correction):
      every pool domain not declared (small ones pooled into ``_rest``, as
      today), every configured horizon target not declared
      (``horizon_targets``), and every public metric — each ``LCB ≥ −tau``.
      Any guard failure blocks EVERY target of the leg.

    A single domain target with no horizon targets is exactly
    :func:`judge_expert_leg` (same seeds, guards, order and reasons)."""
    p = params
    tg = tuple(str(t) for t in targets)
    if (not tg and not guards_only) or len(set(tg)) != len(tg):
        raise ValueError(f"targets must be non-empty and unique; got {tg!r}")
    k = len(tg)
    pt = p if k <= 1 else replace(p, alpha=p.alpha / k)
    dom = np.asarray(list(domains), dtype=object)
    kq, ka, km = (np.asarray(a, dtype=np.float64) for a in (king_qloss, king_abs_target, king_mase))
    cq, ca, cm = (np.asarray(a, dtype=np.float64) for a in (chal_qloss, chal_abs_target, chal_mase))
    if dom.shape[0] != kq.shape[0]:
        raise ValueError("domains must label every window")
    hz = None
    if horizons is not None:
        hz = np.asarray([-1 if h is None else int(h) for h in horizons], dtype=np.int64)
        if hz.shape[0] != kq.shape[0]:
            raise ValueError("horizons must label every window")
    s = str(seed)
    out = TargetsVerdict(targets=tg, alpha=pt.alpha)

    conclusive = []
    for t in tg:
        tmask = _target_mask(t, dom, hz)
        tn = int(tmask.sum())
        v = ExpertVerdict(passed=False, reason="", target_domain=t,
                          target_lcb=float("nan"), target_n=tn)
        out.verdicts[t] = v
        if tn < p.min_target_windows:
            v.reason = f"inconclusive: {tn} target windows < {p.min_target_windows}"
            v.inconclusive = "windows"
            continue
        tcl = None if clusters is None else [c for c, m in zip(clusters, tmask, strict=True) if m]
        v.target_clusters = tn if tcl is None else len(set(tcl))
        if v.target_clusters < p.min_target_clusters:
            v.reason = (f"inconclusive: {v.target_clusters} target clusters < "
                        f"{p.min_target_clusters}")
            v.inconclusive = "clusters"
            continue
        v.target_lcb = _domain_lcb(tmask, clusters, kq, ka, km, cq, ca, cm, pt, f"{s}|gain|{t}")
        v.target_gain = paired_point_rel(kq[tmask], ka[tmask], km[tmask], cq[tmask],
                                         ca[tmask], cm[tmask], tcl)
        conclusive.append(v)
    if not conclusive and not guards_only:
        return out

    # ── guards, ONCE: undeclared domains (small ones pooled), undeclared
    # horizon targets, public metrics — all at alpha
    guard_lcbs: dict[str, float] = {}
    guard_n: dict[str, int] = {}
    public_lcbs: dict[str, float] = {}
    unguarded: list[str] = []
    declared = set(tg)
    small = np.zeros(dom.shape[0], dtype=bool)
    for d in sorted({str(x) for x in dom if x not in declared}):
        m = dom == d
        n = int(m.sum())
        if n >= p.min_guard_windows:
            guard_n[d] = n
            guard_lcbs[d] = _domain_lcb(m, clusters, kq, ka, km, cq, ca, cm, p, f"{s}|guard|{d}")
        else:
            small |= m
            unguarded.append(d)
    if small.any():
        n = int(small.sum())
        if n >= p.min_guard_windows:
            guard_n[REST_GUARD] = n
            guard_lcbs[REST_GUARD] = _domain_lcb(
                small, clusters, kq, ka, km, cq, ca, cm, p, f"{s}|guard|{REST_GUARD}")
            unguarded = []
    for t in horizon_targets:
        if t in declared:
            continue
        m = _target_mask(str(t), dom, hz)
        n = int(m.sum())
        if n >= p.min_guard_windows:
            guard_n[str(t)] = n
            guard_lcbs[str(t)] = _domain_lcb(m, clusters, kq, ka, km, cq, ca, cm, p,
                                             f"{s}|guard|{t}")
        else:
            unguarded.append(str(t))
    for name, (kk, c) in sorted((public or {}).items()):
        if len(kk) < p.min_public_items:
            unguarded.append(name)
            continue
        public_lcbs[name] = paired_geomean_lcb(kk, c, alpha=p.alpha, B=p.B,
                                               seed=f"{s}|public|{name}")
    out.guards_computed = True
    for d, lcb in guard_lcbs.items():
        if not lcb >= -p.tau_domain:
            what = "horizon" if d.startswith(HORIZON_PREFIX) else "domain"
            out.guard_reason = f"regression on {what} {d}: LCB {lcb:.4f} < -{p.tau_domain}"
            break
    if not out.guard_reason:
        for name, lcb in public_lcbs.items():
            if not lcb >= -p.tau_public:
                out.guard_reason = f"regression on {name}: LCB {lcb:.4f} < -{p.tau_public}"
                break

    for v in conclusive:
        v.guard_lcbs, v.guard_n = dict(guard_lcbs), dict(guard_n)
        v.public_lcbs, v.unguarded = dict(public_lcbs), list(unguarded)
        if not v.target_lcb >= pt.gain_margin:            # NaN fails
            v.reason = f"no gain: target LCB {v.target_lcb:.4f} < margin {pt.gain_margin}"
        elif not v.target_gain > 0.0:                     # a zero effect never passes
            v.reason = f"no gain: target point gain {v.target_gain:.4f} <= 0"
        elif out.guard_reason:
            v.reason = out.guard_reason                    # blocks every target
        else:
            v.passed = True
            v.reason = "merge"
    return out


# ── payout ───────────────────────────────────────────────────────────────────


def reward_multiplier(rounds_since_win: int, *, full_eras: int,
                      half_life_rounds: float = 1.0, floor: float = 0.1) -> float:
    """A lane-B / lane-C win's emission multiplier: 1.0 for its first
    ``full_eras`` eras (``rounds_since_win`` 0 … full_eras−1) — the win's
    RECORDED gain-tiered full-pay period (:func:`full_pay`) — then
    exponential decay with the given half-life down to ``floor``. Owner
    2026-10-08: "after 5 rounds it decays exponentially to 10%"; 2026-10-20:
    the full period is tiered by the win's gain LCB."""
    if rounds_since_win < 0:
        raise ValueError("rounds_since_win must be >= 0")
    if isinstance(full_eras, bool) or not isinstance(full_eras, int) or full_eras < 0:
        raise ValueError(f"full_eras must be an integer >= 0; got {full_eras!r}")
    if rounds_since_win < full_eras:
        return 1.0
    k = rounds_since_win - full_eras + 1
    return max(floor, 0.5 ** (k / half_life_rounds))


@dataclass(frozen=True)
class FullPay:
    """A win's gain-tiered full-pay period (owner 2026-10-20), fixed at the
    win and recorded on it. ``gain_lcb`` — the win's gain LCB on its target
    (None = not finite); ``tier`` — the threshold whose ``full_eras``
    applies; ``below_floor`` — an LCB under the lowest threshold (cannot
    clear a ``gain_margin`` at or above it; a config mismatch) paid at the
    lowest tier."""
    gain_lcb: float | None
    tier: float
    full_eras: int
    below_floor: bool = False

    def record(self) -> dict:
        """The fields a win / payout record stamps."""
        return {"gain_lcb": self.gain_lcb, "tier": float(self.tier),
                "full_eras": int(self.full_eras), "below_floor": bool(self.below_floor)}


def full_pay(gain_lcb: float | None, tiers: Sequence[Sequence]) -> FullPay:
    """CONSENSUS: the full-pay period of a win with gain LCB ``gain_lcb``:
    the ``full_eras`` of the HIGHEST threshold with ``gain_lcb >= threshold``
    (``tiers`` = ``(threshold, full_eras)`` pairs, thresholds descending —
    validated at load); below the lowest (or a non-finite LCB) ⇒ the lowest
    tier's, ``below_floor``. Pure — the audit re-derives it from the recorded
    LCB."""
    if not tiers:
        raise ValueError("full_pay needs at least one (threshold, full_eras) tier")
    x = None if gain_lcb is None or not math.isfinite(float(gain_lcb)) else float(gain_lcb)
    pairs = [(float(t), int(n)) for t, n in tiers]
    for t, n in pairs:
        if x is not None and x >= t:
            return FullPay(gain_lcb=x, tier=t, full_eras=n)
    t, n = pairs[-1]
    return FullPay(gain_lcb=x, tier=t, full_eras=n, below_floor=True)


def full_pay_mismatch(rec: Mapping, tiers: Sequence[Sequence]) -> str | None:
    """Why a recorded ``{gain_lcb, tier, full_eras, below_floor}`` is not
    :func:`full_pay` of its own ``gain_lcb`` under ``tiers`` (None =
    consistent). The audit's forged-tier check."""
    if not all(k in rec for k in ("gain_lcb", "tier", "full_eras")):
        return "carries no gain_lcb / tier / full_eras"
    want = full_pay(rec.get("gain_lcb"), tiers).record()
    for k in ("tier", "full_eras", "below_floor"):
        got = rec.get(k, False if k == "below_floor" else None)
        if got != want[k] or isinstance(got, bool) != isinstance(want[k], bool):
            return (f"{k} {got!r} != {want[k]!r} re-derived from gain_lcb "
                    f"{rec.get('gain_lcb')!r}")
    return None


@dataclass(frozen=True)
class LaneBWin:
    uid: int
    domain: str
    multiplier: float


def split_emissions(lane_a: Mapping[int, float], lane_b: Sequence[LaneBWin], *,
                    share_b: float, n_domain_slots: int, burn_uid: int,
                    remainder: str = "burn") -> dict[int, float]:
    """Combine both lanes into one weight vector summing to 1.

    Lane A's (normalised) court vector gets ``1 − share_b``; lane B's
    ``share_b`` is cut into ``n_domain_slots`` equal domain slots, each live win
    paid ``slot × multiplier``. The unpaid part of lane B (empty slots + decay)
    goes to ``remainder``: ``"burn"`` (the cap bites — decay is real) or
    ``"lane_a"`` (re-scales lane A). An empty lane A burns its share, as today.
    """
    if not 0.0 <= share_b <= 1.0:
        raise ValueError("share_b must be in [0, 1]")
    if remainder not in ("burn", "lane_a"):
        raise ValueError("remainder must be 'burn' or 'lane_a'")
    slots: dict[str, LaneBWin] = {}
    for w in lane_b:
        if w.domain in slots:
            raise ValueError(f"two live wins for domain {w.domain!r}")
        slots[w.domain] = w
    if len(slots) > n_domain_slots:
        raise ValueError("more live domains than slots")
    out: dict[int, float] = {}
    slot = share_b / n_domain_slots if n_domain_slots else 0.0
    paid_b = 0.0
    for w in slots.values():
        amt = slot * min(1.0, max(0.0, w.multiplier))
        out[w.uid] = out.get(w.uid, 0.0) + amt
        paid_b += amt
    left_b = share_b - paid_b
    share_a = 1.0 - share_b + (left_b if remainder == "lane_a" else 0.0)
    total_a = sum(lane_a.values())
    if total_a > 0:
        for uid, wt in lane_a.items():
            out[uid] = out.get(uid, 0.0) + share_a * wt / total_a
    else:
        out[burn_uid] = out.get(burn_uid, 0.0) + share_a
    if remainder == "burn" and left_b > 0:
        out[burn_uid] = out.get(burn_uid, 0.0) + left_b
    assert math.isclose(sum(out.values()), 1.0, rel_tol=1e-9)
    return out
