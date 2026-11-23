# W15 acceptance tests: the prefetch cache

W15 (plan rev6 §3.0) depends on W4, which is on main
(`cascade/eval/lane_d_ingest.py`, `cascade/lane_d/storage.py`). These tests
are the spec. The builder may touch only the paths in `PATHS`.

## What the plan says

The W15 row, verbatim:

> | W15 | Prefetch cache, LRU with KING_D and house pinned | W4 | ci + pod | `scripts/laned/accept W15` |

§1.2 step 5, verbatim: "**Prefetch.** On reveal, **(rev5)** the launch eval
host (D11: valpriv-4090-class pinned-torch pod) prefetches the entry into a
sha-keyed, resumable cache, using presigned GETs. Hippius outages and slow
transfers are the history (§13)."

§10 "Data paths" 2, verbatim: "Vault → **(rev5)** the launch eval host
(valeval-laned from S8): presigned GET prefetch at reveal; sha-keyed LRU with
KING_D and the house pinned."

Also: §1.6 "Byte-based LRU (#467)"; §10.7 `lane_d_eval_pod` "disk ≥ cache
budget"; §13 "prefetch at reveal; resumable cache"; T7c (§4) "prefetch; box
RAM and disk flat"; §3.2 invariant 1 (unarmed, nothing changes).

## Contract: `cascade/eval/lane_d_cache.py` (new)

A digest is a lane D entry digest (W1.5/W4): 64 lowercase hex, the sha256 of
the canonical manifest `{"files": {name: {"sha256", "size"}}}` (sorted keys,
compact separators) over **every** regular file of the entry, as
`cascade.lane_d.intake.manifest_digest(checkpoint_manifest(dir))` computes it.

- **`CACHE_CODES`** is exactly `("bad_digest", "digest_mismatch",
  "over_budget")` (any order). **`CacheError(code, detail="")`** is a
  `RuntimeError` with `.code`.
- **`PrefetchCache(root, *, budget_bytes, king_d=None, house=None)`**
  - creates `root` (with parents); `.root` is that `Path`, `.budget_bytes` the
    budget. `budget_bytes` must be a positive `int` (not `bool`, float or str):
    else `ValueError` / `TypeError`.
  - **`.pinned`**: the `frozenset` of the set pins (`None` / `""` = no pin;
    king = house is one pin).
  - **`set_pinned(*, king_d, house)`** replaces the pinned set (it follows
    king and house changes). A pin may name a digest not yet cached; it
    applies when that entry is admitted. The old king/house become ordinary
    LRU entries.
  - **`put(digest, fill) -> Path`**: a cached digest that verifies is a hit:
    `fill` is not called, the entry becomes most recently used. Otherwise
    `fill(staging_dir)` writes the files into a staging directory under
    `root`; the cache hashes them (sha256 of every file, the manifest rule
    above) and admits them at **`root/<digest>`** only if they hash to
    `digest`, else `CacheError("digest_mismatch")`. If the entry fits only by
    evicting a pinned entry (or is larger than the budget) it is
    `CacheError("over_budget")` **and nothing is evicted**. Otherwise the least
    recently used **unpinned** entries are evicted (deleted from disk) until
    `total_bytes() + size <= budget_bytes`, then the entry is admitted as most
    recently used. A `fill` that raises propagates its own exception. Any
    refusal leaves no entry and no staging directory behind.
  - **`get(digest) -> Path | None`**: `None` when absent. Otherwise it
    **re-hashes every file on every read** (never trusting sizes or mtimes);
    if the entry's files no longer hash to `digest` (bytes changed, truncated,
    appended, a file removed or added, nested or not) the entry is deleted,
    `None` is returned (corrupt bytes are never served, pinned or not; the pin
    itself stays) and a WARNING `lane_d_cache_corrupt ... <digest> ...` is
    logged. A verified hit becomes most recently used and returns
    `root/<digest>`.
  - **`ensure(digest, fill) -> Path`**: `get`, else `put` (so a corrupt entry
    is refetched; a refetch that is also wrong is `digest_mismatch`, not
    served).
  - **`entries() -> list[str]`**: the cached digests, least recently used
    first. **`total_bytes() -> int`**: the sum of the entries' file sizes,
    always `<= budget_bytes`.
  - Every method validates digests first: anything but 64 lowercase hex is
    `CacheError("bad_digest")` before any `fill`, request or file access
    outside `root`.
  - **Resumable.** A new `PrefetchCache` on the same `root` recovers the
    entries and their LRU order. The order is recorded by the cache itself
    (a logical counter), **not** derived from file mtimes. Staging leftovers
    from a crash are neither entries nor served. If the reopened budget is
    smaller, LRU unpinned entries are evicted at open until it fits.
  - **Deterministic.** The same operation sequence on two roots gives the same
    `entries()`, `total_bytes()` and on-disk entry set after every step.
  - Logs (logger `cascade.eval.lane_d_cache`): INFO `lane_d_cache_evict ...
    <digest>` per eviction; WARNING `lane_d_cache_corrupt ... <digest>`;
    WARNING `lane_d_cache_refused <code> ...` per `put` refusal. These
    `lane_d_cache_*` tags need cascade-ops follower patterns (W11, build rule
    5).
- **`prefetch_grant(cache, grant, *, contract=None, expect=None) -> Path`**:
  `grant` is W4's presigned grant (`{"hotkey", "digest", "manifest_url",
  "files"}`). It validates `grant["digest"]` (`bad_digest`, before any
  request), then `cache.ensure(digest, fill)` where `fill` ingests the grant
  with W4's `ingest_presigned(..., contract=contract, expect=expect)`, so every
  W4 guard runs before admission. A hit makes **no** vault request. A W4
  refusal propagates its `IngestError` and admits nothing.
- **`open_prefetch_cache(lane_d_cfg, boundary, root, *, budget_bytes,
  king_d=None, house=None) -> PrefetchCache | None`**: `None`, creating
  nothing on disk, unless `lane_d_cfg.active(boundary)` (invariant 1: unarmed,
  armed for a later era, or closed past `until_block`).
- The module stays torch-free at import (the finite check inside W4's step 8
  is the only torch use, through `ingest_presigned`).

## What is pinned

| Requirement | Tests |
|---|---|
| Content addressing on write | `test_put_admits_exact_bytes_under_root_slash_digest`, `test_fill_whose_bytes_do_not_hash_to_the_digest_is_refused`, `test_digest_is_the_w4_manifest_digest`, `test_fill_that_raises_propagates_and_leaves_nothing` |
| Re-hash on read; corrupt refused and refetched, never served | `test_corrupt_entry_is_refused_on_read_and_dropped[*]`, `test_ensure_refetches_a_corrupt_entry_and_serves_only_verified_bytes[*]`, `test_refetch_that_is_also_wrong_is_refused_not_served`, `test_every_read_rehashes_not_only_the_first`, `test_rehash_does_not_trust_mtimes_or_sizes`, `test_corrupt_pinned_king_is_refused_and_refetched`, torch `test_prefetch_refetches_a_corrupt_entry` |
| LRU order | `test_lru_evicts_least_recently_used_first`, `test_get_refreshes_recency`, `test_put_hit_refreshes_recency`, `test_put_of_a_cached_digest_does_not_refill` |
| Byte budget | `test_eviction_is_by_bytes_not_by_count`, `test_total_bytes_never_exceeds_budget`, `test_entry_larger_than_the_budget_is_refused_and_evicts_nothing`, `test_budget_must_be_a_positive_int[*]` |
| KING_D and house never evicted | `test_king_d_and_house_are_never_evicted`, `test_pins_set_before_the_entry_arrives_apply_on_admission`, `test_new_entry_that_only_fits_by_evicting_a_pin_is_refused`, `test_king_equal_to_house_is_one_pinned_entry`, torch `test_prefetched_king_survives_lru_pressure` |
| Pinned set follows king/house changes | `test_pinned_set_follows_king_and_house_changes`, `test_clearing_a_pin_makes_it_evictable` |
| Resumable | `test_reopen_keeps_entries_bytes_and_lru_order`, `test_reopen_order_does_not_come_from_mtimes`, `test_reopen_ignores_staging_leftovers_from_a_crash`, `test_reopen_with_a_smaller_budget_evicts_lru_but_keeps_pins` |
| Determinism | `test_same_operations_give_the_same_cache` |
| Path safety | `test_bad_digest_refused_everywhere[*]`, `test_prefetch_refuses_a_grant_with_a_bad_digest_before_any_request` |
| Prefetch via W4 presigned ingest | `test_prefetch_hit_makes_no_request`, `test_prefetch_refused_by_ingest_admits_nothing[*]`, torch `test_prefetch_miss_ingests_admits_and_then_hits` |
| Invariant 1 pair | `test_inv1_prefetch_cache_unchanged_when_unarmed` + `test_inv1_prefetch_cache_armed_nonempty_lane_d_effect` |

## Fixtures and mocks

- The only mocked service is the vault bucket: `helpers.Vault`, a localhost
  presigned-URL server that records every request.
- Cache entries are deterministic four-file byte sets (`helpers.Entry`); the
  cache is content-addressed and never parses a checkpoint. The prefetch tests
  use W4's T5 fixture shape (the repo's real pyramid code bytes, a small
  config, a hand-written BF16 safetensors) with a matching `LaneDExpectation`.
- The tests compute digests themselves (`helpers.digest_of`) and check once
  that this equals W4's `manifest_digest`.

## Failure on main and the `torch` marker

- On main every test fails inside its body with `No module named
  'cascade.eval.lane_d_cache'` (`scripts/laned/accept W15 --expect-fail`).
- `test_w15_torch.py` is marked `torch`. The production box must never run
  it: run `-m "not torch"` locally (also inside `LANED_PYTEST_CMD`). CI
  installs torch.

## Score cache (rev6.18 B6)

Plan rev6.18 B6: every model in a cohort is forecast on one host in one run.
Tests: `test_w15_score_cache.py`.

- **`cascade/validator/lane_d_score_cache.py`** (new): `namespace(host_identity,
  package_hash)` (both 64 hex, else `ValueError`); `ScoreCache(root)` with
  `get(ns, digest, *, context, horizon, window_ids)` (exact float64 rows in
  request order, or `None` when any window is missing, the file is corrupt
  (dropped) or the namespace differs), `put(...)` (non-finite / misshaped
  arrays are never stored) and `retain(ns, keys)` (deletes other namespaces
  and unused keys).
- **`LaneDInputsSource(..., scoring_envs=, score_cache=)`**: with
  `scoring_envs` the hello is checked by `check_worker` at the boundary
  (`env_unqualified` otherwise, nothing forecast); `LaneDInputs.host` is the
  verified host. With a cache, a model is forecast only on a miss; a host
  change (another fingerprint, driver, env, GPU or torch build) misses for the
  whole cohort. `load_inputs_services` always loads the list (optional
  `scoring_envs` path key) and builds the cache from the optional
  `score_cache_dir` key.
