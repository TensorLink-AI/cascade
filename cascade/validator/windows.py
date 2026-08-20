"""Eval-window source — a rotating, private, per-round held-out set.

The KOTH comparison is only a *controlled* measurement if the king's model and
the challenger's model are scored on the same windows; it is only *contamination
resistant* if those windows are not a fixed public benchmark a generator can
distribution-match. cascade resolves both with a single rule:

    windows_for_round(round_seed) → a seeded slice of a private pool

* **Same within a round.** King and challenger are scored on the identical slice
  (one ``round_seed`` per round → one selection), so the eval cancels and the
  comparison is paired.
* **Agreed across validators.** ``round_seed`` is the round's chain block hash,
  so every honest validator draws the byte-identical slice and they converge on
  the same KOTH verdict.
* **Rotated across rounds.** A different ``round_seed`` permutes the pool
  differently, so the scored slice rotates — there is no fixed eval set to overfit
  (the TIME-benchmark philosophy: fresh data each time).

The pool itself is owner-controlled and **private** (``[eval] window_pool``).
Loading it — pulling the held-out corpus and slicing it into context/target
windows — is the integrator boundary; this module owns
the deterministic, validator-agreeing *selection and rotation*, which is pure and
unit-tested. :func:`build_windows_from_series` is the reference cutter the loader
can use to turn raw held-out series into :class:`EvalWindow` s.
"""

from __future__ import annotations

import hashlib
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..eval.window import EvalWindow


def _seed_to_int(seed: int | str) -> int:
    """Stable seed → 64-bit int (blake2b), matching the eval bootstrap so the
    same ``round_seed`` string maps identically everywhere a validator uses it."""
    if isinstance(seed, int):
        return seed
    h = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


@dataclass(frozen=True)
class MixParams:
    """Knobs for the jittered round mix (the DEC-TB-0003 port).

    ``from_block = 0`` disables the jitter entirely — the legacy uniform
    permutation runs, byte-identical to every historical round. When set, a
    round whose epoch-boundary block is ``>= from_block`` draws the jittered
    mix instead. Block-gating is what makes the rollout safe with external
    validators: a mixed fleet agrees on every round as long as every validator
    upgrades (and configures the same ``from_block``) BEFORE the activation
    block, and the audit replay of any historical round applies the rule that
    was active at that round's block.

    ``target_windows`` (0 = the caller's ``n_windows``) lets the jittered path
    sample fewer windows than the legacy verdict does. Calibration on the
    2026-08-19 production snapshot (2369 series): at n_windows=2000 the pool is
    ~84% served and per-domain caps crush the Dirichlet (transport p1–p99
    393–446 of 2000); at 1200 the mix genuinely rotates (64–430) while
    effective domains stay ≥ 5.0 and served windows stay 6× above
    ``[scoring] min_windows``.
    """

    from_block: int = 0
    jitter_alpha: float = 4.0     # symmetric Dirichlet over domains; expectation stays uniform
    block_slots: int = 8          # allocation unit: a (domain[, class]) cell is absent or ≥ a block
    class_keep_frac: float = 0.7  # per-round dgp_class activation (needs dgp_class pool labels)
    series_bag_frac: float = 0.7  # per-round series eligibility (salted-hash bag)
    target_windows: int = 0       # jittered draw size; 0 = caller's n_windows

    def active(self, block: int | None) -> bool:
        return self.from_block > 0 and block is not None and int(block) >= self.from_block


def _bag_keep(salt: int, key: str, frac: float) -> bool:
    """Per-round bag membership: salted hash of the stable series key.

    A hash, not a positional rng draw, so membership is independent of the
    order the pool happened to be enumerated in — only the rng-drawn salt
    varies per round, and it is consensus-safe by construction."""
    return (zlib.crc32(f"{salt}:{key}".encode()) % 10_000) < int(frac * 10_000)


def _split(total: int, k: int) -> list[int]:
    """Distribute ``total`` slots as evenly as possible into ``k`` groups."""
    if k == 0:
        return []
    base, extra = divmod(total, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _capped_jittered_split(
    n: int,
    caps: list[int],
    alpha: float | None,
    block: int,
    rng: np.random.Generator,
) -> list[int]:
    """Split ``n`` slots into ``len(caps)`` groups, Dirichlet-jittered around
    uniform (``alpha=None`` → exactly uniform), allocated in ``block``-slot
    units with a one-block floor per group.

    Unlike the TSBench original, every group is hard-capped at ``caps[i]``:
    cascade serves one window per series, so a cell's quota can never be
    duplicate-filled — overflow is redistributed to open groups by remaining
    weight (water-filling). The expected mix is therefore uniform *projected
    onto capacity*: a domain smaller than its uniform share is always fully
    eligible and the surplus flows to the larger domains.
    """
    k = len(caps)
    if k == 0:
        return []
    n = min(int(n), int(sum(caps)))
    counts = [min(block, c) for c in caps]
    if sum(counts) > n:
        # Too small to floor every group (heat-sized draws over many classes):
        # concentrate on a seeded subset in usable chunks, like the TSBench
        # small-n branch, instead of spraying token 1-slot counts.
        m = max(1, n // block)
        counts = [0] * k
        order = [int(i) for i in rng.permutation(k)]
        shares = _split(n, m)  # usable chunks for the chosen groups
        left = n
        for pos, i in enumerate(order):
            want = shares[pos] if pos < m else left  # cap-spill past the m chosen
            take = min(want, caps[i], left)
            counts[i] = take
            left -= take
            if left <= 0:
                break
        return counts
    if alpha is not None:
        w = rng.dirichlet(np.full(k, float(alpha)))
    else:
        w = np.full(k, 1.0 / k)
    for _ in range(64):  # water-filling: the open set shrinks; terminates early
        left = n - sum(counts)
        if left <= 0:
            break
        open_ = [i for i in range(k) if counts[i] < caps[i]]
        if not open_:
            break
        if left < block:
            j = max(open_, key=lambda i: (counts[i], -i))
            counts[j] += min(left, caps[j] - counts[j])
            continue
        wsum = float(sum(w[i] for i in open_))
        raw = np.array(
            [left * (float(w[i]) / wsum if wsum > 0 else 1.0 / len(open_)) for i in open_]
        )
        nb = left // block
        base = np.floor(raw / block).astype(int)
        rem = int(nb - base.sum())
        if rem > 0:
            frac = raw / block - base
            order = np.argsort(-frac, kind="stable")
            base[order[:rem]] += 1
        progressed = False
        for pos, i in enumerate(open_):
            add = min(int(base[pos]) * block, caps[i] - counts[i])
            counts[i] += add
            progressed = progressed or add > 0
        if not progressed:
            j = open_[0]
            counts[j] += min(block, caps[j] - counts[j])
    return counts


def _jittered_pick(
    pool: tuple[EvalWindow, ...],
    n: int,
    rng: np.random.Generator,
    mix: MixParams,
) -> list[EvalWindow]:
    """The jittered round draw: bag → Dirichlet domain mix → class rotation →
    capped even class split → without-replacement picks.

    Every draw flows through ``rng`` over sorted keys, so a given
    (pool, round_seed) yields byte-identical picks on every validator. Series
    identity is ``series_id`` (the pool's metadata key — stable across
    validators because the loader sorts filenames).
    """
    idx_by_id = {w.series_id: i for i, w in enumerate(pool)}
    ids = sorted(idx_by_id)
    if mix.series_bag_frac < 1.0:
        salt = int(rng.integers(0, 2**32))
        bagged = [s for s in ids if _bag_keep(salt, s, mix.series_bag_frac)]
        # An unlucky bag that empties the pool falls back to the full pool
        # rather than failing the round.
        if bagged:
            ids = bagged

    # The class tier only exists when the pool carries dgp_class labels;
    # falling back to `source` would fragment into ~1000 micro-cells on the
    # production pool and the activation cap would starve rounds below
    # [scoring] min_windows (measured in calibration: 67 of 256).
    has_classes = any(pool[idx_by_id[s]].metadata.get("dgp_class") for s in ids)
    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in ids:
        m = pool[idx_by_id[s]].metadata
        cls = str(m.get("dgp_class")) if has_classes else "_all"
        by_cell[(str(m.get("domain") or "unknown"), cls)].append(s)

    domains = sorted({d for d, _ in by_cell})
    dom_caps = [
        sum(len(v) for (d, _), v in by_cell.items() if d == dom) for dom in domains
    ]
    block = max(1, int(mix.block_slots))
    per_domain = _capped_jittered_split(n, dom_caps, mix.jitter_alpha, block, rng)

    picks: list[str] = []
    for dom, n_dom in zip(domains, per_domain):
        if n_dom == 0:
            continue
        all_classes = sorted({c for d, c in by_cell if d == dom})
        classes = list(all_classes)
        if has_classes and mix.class_keep_frac < 1.0 and len(classes) > 1:
            mask = rng.random(len(classes)) < mix.class_keep_frac
            kept = [c for c, keep in zip(classes, mask) if keep]
            classes = kept or [classes[int(rng.integers(0, len(classes)))]]
            # Rotation must never make the domain quota unfillable (cascade
            # cannot duplicate-fill): extend the kept set in seeded order over
            # the dropped classes until active capacity covers the quota.
            capacity = sum(len(by_cell[(dom, c)]) for c in classes)
            if capacity < n_dom:
                dropped = [c for c in all_classes if c not in classes]
                order = rng.permutation(len(dropped))
                for i in order:
                    c = dropped[int(i)]
                    classes.append(c)
                    capacity += len(by_cell[(dom, c)])
                    if capacity >= n_dom:
                        break
                classes.sort()
        if has_classes:
            # Never activate more classes than can each field one block,
            # weighted by how many series a class can field (capped at one
            # block) so tiny classes still get rounds, just less often — but
            # only while the trimmed set can still fill the quota.
            cap = max(1, n_dom // block)
            if len(classes) > cap:
                wts = np.array(
                    [min(len(by_cell[(dom, c)]), block) for c in classes], dtype=float
                )
                sel = rng.choice(len(classes), size=cap, replace=False, p=wts / wts.sum())
                trimmed = [classes[int(i)] for i in sorted(int(j) for j in sel)]
                if sum(len(by_cell[(dom, c)]) for c in trimmed) >= n_dom:
                    classes = trimmed
        cls_caps = [len(by_cell[(dom, c)]) for c in classes]
        per_class = _capped_jittered_split(n_dom, cls_caps, None, block, rng)
        for cls, n_cls in zip(classes, per_class):
            cell = by_cell[(dom, cls)]
            if n_cls >= len(cell):
                picks.extend(cell)  # without replacement, cell exhausted
            else:
                sel = rng.choice(len(cell), size=n_cls, replace=False)
                picks.extend(cell[int(i)] for i in sel)
    return [pool[idx_by_id[s]] for s in picks]


def round_composition(windows: list[EvalWindow], mix: MixParams | None = None) -> dict:
    """The realised composition of one round's served windows — the post-hoc
    manifest block (DEC-TB-0003). Post-hoc by construction: each round's
    Dirichlet draw is independent, so publishing it predicts nothing about the
    next round."""
    domains = dict(
        sorted(Counter(str(w.metadata.get("domain") or "unknown") for w in windows).items())
    )
    total = sum(domains.values())
    p = np.array([v / total for v in domains.values() if v], dtype=float)
    eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    out = {
        "n_windows": len(windows),
        "domains": domains,
        "effective_domains": round(eff, 2),
        "cadences": dict(
            sorted(Counter(str(w.metadata.get("freq") or "?") for w in windows).items())
        ),
        "n_dgp_classes": len(
            {w.metadata.get("dgp_class") for w in windows if w.metadata.get("dgp_class")}
        ),
        "n_clusters": len(
            {w.metadata.get("source") or w.series_id for w in windows}
        ),
    }
    if mix is not None:
        out["mix"] = {
            "from_block": mix.from_block,
            "jitter_alpha": mix.jitter_alpha,
            "block_slots": mix.block_slots,
            "class_keep_frac": mix.class_keep_frac,
            "series_bag_frac": mix.series_bag_frac,
            "target_windows": mix.target_windows,
        }
    return out


@runtime_checkable
class WindowSource(Protocol):
    """Produces the eval windows for one round, deterministic in ``round_seed``.

    ``block`` (the round's epoch-boundary block number) is an optional selector
    for sources that rotate a daily snapshot (the bucket pool); a static pool
    ignores it. Keeping it keyword-only + defaulted preserves the simple
    ``windows_for_round(seed, n)`` call for static sources.
    """

    def windows_for_round(
        self, round_seed: int | str, n_windows: int, *, block: int | None = None
    ) -> list[EvalWindow]:
        ...


@dataclass(frozen=True)
class RotatingWindowSource:
    """Select a rotating slice of a fixed private pool, seeded by the round.

    The pool is the materialised private held-out set (injected; the loader is a
    boundary). Selection is a seeded permutation prefix: deterministic in
    ``round_seed`` (so all validators agree and king/challenger pair), and
    rotating across rounds (so no fixed slice can be overfit). When the pool has
    fewer than ``n_windows`` entries the whole pool is returned in a seeded order
    and the KOTH ``min_windows`` gate decides whether the round is conclusive.
    """

    pool: tuple[EvalWindow, ...]
    # Where the pool came from, for the public round receipt: ``(ref, digest)``
    # — the Hub ``repo@digest`` and its OCI digest for a static pool. ("", "")
    # for an injected/test pool.
    provenance: tuple[str, str] = ("", "")
    # Jittered round mix (DEC-TB-0003 port). None or from_block=0 → the legacy
    # uniform permutation for every round, byte-identical to history. The gate
    # is the round's epoch-boundary BLOCK, so mixed validator fleets agree on
    # every round while they upgrade, and audit replay applies each round's
    # own rule.
    mix: MixParams | None = None

    def __post_init__(self) -> None:
        if not self.pool:
            raise ValueError("RotatingWindowSource pool is empty")

    def provenance_for_round(self, round_seed: int | str, *, block: int | None = None) -> tuple[str, str]:
        """``(pool_ref, pool_digest)`` recorded in the round receipt. A static
        pool's provenance is round-independent (``block`` ignored)."""
        return self.provenance

    def windows_for_round(
        self, round_seed: int | str, n_windows: int, *, block: int | None = None
    ) -> list[EvalWindow]:
        # ``block`` selects the daily snapshot for a bucket pool AND gates the
        # jittered mix; a static pool draws its slice from ``round_seed`` alone.
        if n_windows <= 0:
            raise ValueError(f"n_windows must be positive; got {n_windows}")
        if self.mix is not None and self.mix.active(block):
            n = self.mix.target_windows or n_windows
            rng = np.random.default_rng(_seed_to_int(round_seed))
            return _jittered_pick(self.pool, min(n, len(self.pool)), rng, self.mix)
        rng = np.random.default_rng(_seed_to_int(round_seed))
        perm = rng.permutation(len(self.pool))
        take = min(n_windows, len(self.pool))
        return [self.pool[i] for i in perm[:take]]


def build_windows_from_series(
    series: list[np.ndarray],
    *,
    context_length: int,
    horizon: int,
    metadata: list[dict] | dict | None = None,
    id_prefix: str = "w",
) -> list[EvalWindow]:
    """Cut held-out series into context/target :class:`EvalWindow` s.

    Each series contributes one window: the last ``horizon`` steps are the target
    and up to ``context_length`` steps before them are the history. Accepts 1-D
    ``(L,)`` or 2-D ``(C, L)`` series (1-D becomes a single channel). A series too
    short to yield ``horizon`` targets plus at least one context step is skipped.

    This is the reference cutter for a :class:`WindowSource` loader; it is pure
    and deterministic so the resulting pool is identical for every validator.
    """
    if context_length <= 0 or horizon <= 0:
        raise ValueError("context_length and horizon must be positive")
    out: list[EvalWindow] = []
    for i, raw in enumerate(series):
        arr = np.atleast_2d(np.asarray(raw, dtype=np.float64))   # (C, L)
        length = arr.shape[-1]
        if length < horizon + 1:
            continue
        ctx = min(context_length, length - horizon)
        history = arr[:, length - horizon - ctx : length - horizon]
        target = arr[:, length - horizon :]
        if metadata is None:
            md: dict = {}
        elif isinstance(metadata, dict):
            md = metadata
        else:
            md = metadata[i]
        out.append(
            EvalWindow(series_id=f"{id_prefix}{i}", history=history, target=target, metadata=md)
        )
    return out
