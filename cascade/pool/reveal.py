"""Public reveal of retired eval-pool snapshots — exact bytes, not a rebuild.

The daily publisher pins ``pool/snapshots/block-<N>.tar`` and validators score
that tar; its sha256 is the integrity hash in every round receipt. The reveal's
job is to let anyone verify what was scored — which is only possible if the
public copy is **byte-identical** to the pinned tar.

The original reveal *rebuilt* the prior day's pool from the forge mirror and
relied on the rebuild being deterministic. It is not: mirror partitions dated
day ``D`` keep landing throughout day ``D`` (feeds scrape at all hours) and
later (backfills), so the ~03:00 build that got pinned saw a partial day while
the next-morning rebuild saw the closed day (observed live: block-8740800.tar =
3,356 series vs its HF rebuild = 3,442). This module replaces the rebuild with
a mirror of the pinned tar itself: download, verify sha256 against the index,
unpack, and push both the files and the tar to the public HF dataset.

Safety guard — never reveal an active pool: entries are sorted by
``effective_block`` and the newest ``keep_newest`` (default 2) are withheld:
the newest may be pending/just-activated and the second-newest may still be
the active pool. Everything older cannot be selected by any current or future
round (validators pick the greatest ``effective_block <=`` the round's epoch
block). The guard is chain-free; under publisher gaps it errs conservative
(reveals later than strictly necessary, never earlier). Effective lag is
therefore ~48h in steady state, vs the old ~24h.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..shared.hippius import (
    PoolSnapshotMeta,
    StorageError,
    read_pool_index,
    tar_cid_digest,
    unpack_tar_to_dir,
)

log = logging.getLogger("cascade.pool")

# Anything at or above this is a workflow_dispatch smoke-test sentinel
# (999_999_999_999 in publish-pool.yml), never selectable by a validator:
# skip them so a sentinel can't consume a keep_newest slot or be revealed.
SENTINEL_MIN_BLOCK = 10**11

# Marker file inside each revealed dir; its presence makes re-runs idempotent
# and its content (the tar sha256) is the receipt-matching handle.
SHA_MARKER = "POOL_SHA256"


def reveal_dir_name(meta: PoolSnapshotMeta) -> str:
    """HF folder for one revealed snapshot. Date-prefixed so listings sort
    chronologically; block-suffixed so the mapping to a round's pinned
    ``pool_ref`` is explicit. Distinct from the legacy date-only dirs, which
    were non-exact rebuilds and are left untouched."""
    day = meta.as_of or "unknown"
    return f"snapshots/{day}-block-{meta.effective_block}"


def select_reveals(
    index: list[PoolSnapshotMeta],
    *,
    keep_newest: int = 2,
    sentinel_min_block: int = SENTINEL_MIN_BLOCK,
) -> list[PoolSnapshotMeta]:
    """The retired snapshots safe to reveal, oldest first.

    Pure selection logic (no IO): drop test sentinels, sort by
    ``effective_block``, withhold the newest ``keep_newest`` real entries.
    """
    real = sorted(
        (s for s in index if s.effective_block < sentinel_min_block),
        key=lambda s: s.effective_block,
    )
    if keep_newest <= 0:
        return real
    return real[:-keep_newest] if len(real) > keep_newest else []


def reveal_snapshots(
    store,
    hf_repo: str,
    *,
    hf_token: str,
    keep_newest: int = 2,
    max_reveal: int = 6,
    work_dir: Path | str = "./_reveal_stage",
) -> int:
    """Mirror retired snapshot tars from the pool bucket to the public HF repo.

    Idempotent: a snapshot whose ``SHA_MARKER`` already exists on HF is skipped,
    so each run only uploads what's missing (at most ``max_reveal``, oldest
    first, so a backlog backfills over a few runs). Returns the number revealed.
    Raises :class:`StorageError` if any downloaded tar fails its index sha256 —
    an integrity failure must be loud, never published.
    """
    from huggingface_hub import HfApi  # hippius extra; lazy like the HF script

    index = read_pool_index(store)
    candidates = select_reveals(index, keep_newest=keep_newest)
    if not candidates:
        log.info("reveal: no retired snapshots to reveal (index has %d entries)", len(index))
        return 0

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=False)

    revealed = 0
    newest_done: PoolSnapshotMeta | None = None
    for meta in candidates:
        if revealed >= max_reveal:
            log.info("reveal: max_reveal=%d reached; remaining backlog next run", max_reveal)
            break
        dir_name = reveal_dir_name(meta)
        if api.file_exists(hf_repo, f"{dir_name}/{SHA_MARKER}", repo_type="dataset"):
            newest_done = meta
            continue

        try:
            data = store.get_bytes(meta.key)
        except StorageError as e:
            # Pruned out-of-band tars are expected for old entries; skip, don't die.
            log.warning("reveal: %s not fetchable (%s); skipping", meta.key, e)
            continue
        got = tar_cid_digest(data)
        if got != meta.sha256:
            raise StorageError(
                f"reveal_digest_mismatch for {meta.key}: bucket tar {got} != index {meta.sha256}"
            )

        stage = Path(work_dir) / f"block-{meta.effective_block}"
        unpack_tar_to_dir(data, stage)
        tar_name = f"block-{meta.effective_block}.tar"
        (stage / tar_name).write_bytes(data)
        (stage / SHA_MARKER).write_text(meta.sha256 + "\n", encoding="utf-8")

        api.upload_folder(
            folder_path=str(stage),
            path_in_repo=dir_name,
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=(
                f"reveal exact snapshot block-{meta.effective_block} "
                f"(as_of={meta.as_of}, sha256={meta.sha256[:16]}…)"
            ),
        )
        log.info(
            "reveal: %s -> %s (%d series, sha256=%s)",
            meta.key, dir_name, meta.n_series, meta.sha256[:16],
        )
        revealed += 1
        newest_done = meta

    if newest_done is not None:
        import io

        api.upload_file(
            path_or_fileobj=io.BytesIO(_card(newest_done).encode("utf-8")),
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"refresh card (through block-{newest_done.effective_block})",
        )
    return revealed


def _card(latest: PoolSnapshotMeta) -> str:
    """Dataset card for the exact-bytes reveal era."""
    return f"""---
license: cc-by-4.0
pretty_name: cascade held-out eval pool (lagged exact reveal)
tags:
  - time-series
  - forecasting
  - benchmark
---

# cascade eval pool — lagged public reveal (exact bytes)

Retired snapshots of the **held-out evaluation pool** used by the
[cascade](https://github.com/TensorLink-AI/cascade) subnet. Each folder is a
**byte-identical mirror of the `pool/snapshots/block-<N>.tar` that validators
scored** — downloaded from the private pool bucket, sha256-verified against the
publisher index, and republished unmodified. A snapshot is revealed only after
a newer snapshot has superseded it, so no revealed pool can be selected by a
current or future round.

## Verifying a round receipt

1. Your receipt's pool provenance carries the snapshot tar's sha256.
2. Find the folder whose `POOL_SHA256` matches; `block-<N>.tar` in that folder
   is the exact artifact — hash it yourself to confirm.
3. Series order = filenames sorted lexicographically; `window_ids` are
   positional (`w<i>` = the i-th series in that order).

## Layout

```
snapshots/<as_of>-block-<N>/     one folder per revealed snapshot
  block-<N>.tar                  the EXACT tar validators scored
  POOL_SHA256                    its sha256 (matches receipts + publisher index)
  <series_id>.npy                the same files, unpacked for convenience
  metadata.json                  {{series_id: {{freq, seasonal_period, domain, source}}}}
  provenance.json                build config recorded at publish time
```

Legacy `snapshots/<YYYY-MM-DD>/` folders (through 2026-08-03) predate this
scheme: they were **rebuilds**, not byte mirrors, and are known to differ from
the scored tars (see the repo issue history). Use the `-block-<N>` folders for
exact replay.

## Latest revealed — `{latest.as_of}` (block {latest.effective_block})

- **series:** {latest.n_series}
- **sha256:** `{latest.sha256}`

> This is an **evaluation** set, not training data. Publishing it to train on
> would contaminate the benchmark it exists to measure.
"""
