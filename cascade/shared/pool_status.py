"""Public eval-pool composition — ``status/pool.json``.

Validators score every round on a private, rotating pool of real-world series
(``docs/EVAL_POOL.md``). The pool's *contents* stay private — that is the
contamination lever — but its *shape* is exactly what a miner tuning a
generator needs to know: which domains, at which granularities, in what
proportion, and how those proportions drift as snapshots rotate. Until this
module that shape was only documented prose (a stale table in EVAL_POOL.md)
plus fully-revealed RETIRED snapshots on HF, weeks after the fact.

So ``cascade-pool publish`` — the owner's daily snapshot cron — also writes a
public-read composition doc into the MANIFEST bucket (the one the dashboards
already read anonymously): a rolling window of per-snapshot summaries, each
carrying ``effective_block`` (the same consensus key validators select
snapshots by), the data cutoff, series counts, window geometry, the tar
sha256 (cross-checkable against the signed manifest's ``eval_pool_sha256``
pin), and a ``breakdown`` of series counts per domain x granularity.

Aggregate counts only — never series identities, sources, or values: the
breakdown reveals nothing a reader of EVAL_POOL.md and the public forge
catalog could not already infer, and revealing WHICH series would hand
generators a distribution-matching target (the attack the pool's privacy
exists to prevent).

Everything here is presentational and unsigned, like the other ``status/``
docs: the signed manifest's pool pin remains the audit record, and a publish
failure must never disturb the snapshot publish it is describing (callers
wrap accordingly). Consumers map a round to its pool with the SAME rule
validators use — greatest ``effective_block <= the round's epoch-boundary
block`` (:func:`active_snapshot`).
"""

from __future__ import annotations

import json

POOL_STATUS_KEY = "status/pool.json"
POOL_STATUS_SCHEMA = 1

# Rolling window of snapshot summaries kept in the doc. Snapshots publish
# daily, so this is ~2 months of pool history; entries are a few hundred bytes.
POOL_STATUS_MAX_KEEP = 60


def build_pool_status_entry(
    *,
    effective_block: int,
    as_of: str,
    published_at: str,
    n_series: int,
    context_length: int,
    horizon: int,
    sha256: str = "",
    breakdown: dict | None = None,
) -> dict:
    """One snapshot's composition summary (pure — storage I/O stays with the
    caller).

    ``effective_block`` is the consensus join key (the epoch-boundary block
    from which the snapshot is active — the key validators select by), ``as_of``
    the harvest's freshness cutoff, ``published_at`` the publish wall-clock.
    ``breakdown`` is ``{domain: {freq: n_series}}`` straight from the builder's
    :class:`~cascade.pool.builder.BuildSummary.per_domain_freq`; ``sha256`` is
    the snapshot tar digest, cross-checkable against the signed manifest's
    ``eval_pool_sha256`` pin for the rounds it governs.
    """
    return {
        "effective_block": int(effective_block),
        "as_of": str(as_of),
        "published_at": str(published_at),
        "n_series": int(n_series),
        "context_length": int(context_length),
        "horizon": int(horizon),
        "sha256": str(sha256),
        "breakdown": {
            str(dom): {str(freq): int(n) for freq, n in freqs.items()}
            for dom, freqs in (breakdown or {}).items()
        },
    }


def read_pool_status(store: object) -> dict:
    """Read ``status/pool.json``; an empty doc if absent or malformed.

    Presentational, so unlike the pool bucket's ``pool/index.json`` a failed
    read degrades to empty rather than raising — the worst case is one publish
    cycle's history window resetting, never a consensus divergence.
    """
    empty = {"schema": POOL_STATUS_SCHEMA, "snapshots": []}
    try:
        text = store.get_text(POOL_STATUS_KEY)
    except Exception:  # noqa: BLE001 — any store hiccup reads as "no doc yet"
        return empty
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(obj, dict) or not isinstance(obj.get("snapshots"), list):
        return empty
    return obj


def update_pool_status(
    store: object,
    entry: dict,
    *,
    max_keep: int = POOL_STATUS_MAX_KEEP,
) -> dict:
    """Append/replace this snapshot's entry in ``status/pool.json``, public-read.

    Idempotent per ``effective_block`` (a re-published snapshot replaces its own
    entry — mirroring ``publish_pool_snapshot``), sorted by ``effective_block``
    (monotonic; the round id is a block-hash seed and must never order
    anything), capped at ``max_keep`` most recent. The owner publish cron is the
    only writer, so there is no cross-writer merge to lose. Returns the doc
    written.
    """
    from .heat_status import _publish_public_json

    block = int(entry["effective_block"])
    snaps = [s for s in read_pool_status(store).get("snapshots", [])
             if isinstance(s, dict) and _as_int(s.get("effective_block")) != block]
    snaps.append(dict(entry))
    snaps.sort(key=lambda s: _as_int(s.get("effective_block")) or 0)
    doc = {
        "schema": POOL_STATUS_SCHEMA,
        "updated_at": str(entry.get("published_at", "")),
        "snapshots": snaps[-max_keep:],
    }
    _publish_public_json(store, POOL_STATUS_KEY, doc)
    return doc


def snapshots(doc: object) -> list[dict]:
    """The doc's well-formed snapshot entries, sorted by ``effective_block``."""
    if not isinstance(doc, dict):
        return []
    snaps = [s for s in doc.get("snapshots", [])
             if isinstance(s, dict) and _as_int(s.get("effective_block")) is not None]
    return sorted(snaps, key=lambda s: _as_int(s.get("effective_block")))


def active_snapshot(doc: object, epoch_block: int) -> dict | None:
    """The entry describing the pool a round at ``epoch_block`` is scored on.

    The SAME deterministic rule validators apply to ``pool/index.json``
    (:func:`cascade.shared.hippius.select_snapshot`): greatest
    ``effective_block <= epoch_block``, falling back to the earliest entry when
    the round precedes them all; ``None`` only for an empty doc. Keeping the
    rules identical is the point — the dashboard's "this round's pool" must
    name the snapshot a validator would actually score on.
    """
    snaps = snapshots(doc)
    if not snaps:
        return None
    eligible = [s for s in snaps if _as_int(s.get("effective_block")) <= int(epoch_block)]
    return eligible[-1] if eligible else snaps[0]


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
