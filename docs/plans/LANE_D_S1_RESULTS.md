# Lane D S1 results

**Evidence rule.** Every number below cites its command, the pool snapshot
sha256, the host and a raw output file under `docs/plans/lane_d_s1/`. There
are no estimates, extrapolations or placeholders. Anything not run says
`not run: <blocker>`; anything below its floor says `below floor` plus the
counts.

Owner inputs are requested in `LANE_D_PROGRESS.md` → **Owner actions**
(OA1–OA7).

## B2: W1 measurements

| Measurement | Result | Command | Snapshot sha256 | Host | Raw output |
|---|---|---|---|---|---|
| Do first-seen or ingest timestamps exist in `cascade/pool/`? | **No.** No match for `first_seen`, `ingested_at`, `ingest_time`, `landed_at` or `LastModified` (grep exit 1) | `grep -rnE "first_seen\|ingested_at\|ingest_time\|landed_at\|LastModified" cascade/pool/` at repo `3c67b64` | n/a (code check) | droplet (prod box, read-only grep) | `lane_d_s1/b2_first_seen_grep.txt` |
| pool_mkt cadence | **Scheduled every 6 h (`30 */6 * * *`), but scheduled runs skip while `[marketcast] from_block = 0`, which is the current value on mainnet and testnet.** Today pool_mkt publishes only on manual dispatch | `grep -nE "cron\|schedule" .github/workflows/publish-mkt-pool.yml` plus `publish-mkt-pool.yml:84-95` | n/a (code check) | droplet (read-only) | `lane_d_s1/b2_pool_mkt_cadence.txt` |
| Forward-eligible windows and distinct sources per daily snapshot (6 h margin, h = 64) | **not run: blocked: needs owner (OA6)**, a non-box host to fetch ≥ 2 consecutive pinned `pool` snapshots. Pool records carry no timestamps, so this run can only produce a **snapshot-diff bound (upper bound on eligible counts)**, never measured first-seen | `python -I scripts/laned_w1_measure.py diff …` (script in this PR) | — | — | — |
| Floors met under the snapshot-diff upper bound? | **not run** (depends on the row above) | — | — | — | — |
| Same for pool_mkt at 6 h | **not run: blocked: needs owner (OA6)**, plus pool_mkt isn't published on schedule while MarketCast is off (row 2) | — | — | — | — |
| Series and sources with history ≥ 4k / 8k / 16k (+64), the launch `pool_d` | **not run: blocked: needs owner (OA5).** Not answerable from pool snapshots, which truncate every series to 4096 + 720 (`builder.py:264-266`). Needs the forge mirror (`TSFORGE_S3`) | — | — | — | — |
| Series and sources with history ≥ 32k / 64k / 128k (+64), for S8 | **not run: blocked: needs owner (OA5)** (same reason) | — | — | — | — |
| Launch `pool_d` size | **not run: blocked: needs owner (OA5)** | — | — | — | — |
| 16k-context batch memory fit on valpriv-4090 (24 GB) | **not run: blocked: needs owner (OA4)**, pod approval (valpriv-4090 outside lane A duel windows, or a rented pinned-torch pod) | — | — | — | — |

## B1: null FPR

| Measurement | Result |
|---|---|
| Prior-king pairs (null type 1, primary) | **not run: blocked: needs owner (OA1** pairs, **OA4** pod) |
| Seed replicas (null type 2) | **not run: blocked: needs owner (OA2)** |
| Perturbed king (null type 3, the easy case; no recommendation from it alone) | **not run: blocked: needs owner (OA4)** pod approval |
| Per-entry FPR, null LCB quantiles 50/90/95/99, disjoint/resampled spread ratio, confirmation-duel FPR, per-era FPR at k = 1 / 8 / 32, `d_margin` for per-era targets 1/5/10%, at min_clusters 20/30/40 | **not run** (depends on the rows above). The per-era target itself awaits **OA3** |

Other notes from the build:

- `score` doesn't pass a training contract to `evaluate_checkpoint`, so the
  config.json and weights-header pin isn't armed. The code-identity
  byte-check still runs.
- The snapshot sha256 the scripts record is a digest of the unpacked
  directory; the published tar sha256 is recorded as given.
- W1 holds two snapshots in float64, so it must run off the prod box (OA6).

The script (`scripts/null_fpr.py`) and its tests
(`tests/unit/test_null_fpr.py`) are in this PR. CI runs the tests; nothing
was executed on the prod box.

**Repo fact affecting B1 (verified by the script build):**

- `calib_draw` (`cascade/validator/calibration.py:52`, wrapping
  `windows.horizon_draw` at `windows.py:414`) cuts **one window per series**:
  the last 64 points.
- A pinned pool of about 4.6k series (snapshot 9509400: 4,633) therefore
  gives about **4.6k windows at h = 64, not ~12k**. That allows about **4
  disjoint 1000-window sets, not 12**, for the spread check. Fewer sets make
  the disjoint SD noisier; the script reports the set count with the ratio.
- Reaching ~12 disjoint sets would need several snapshots' windows, which
  conflicts with "one window list", or a smaller `--n`.

**Known limitation (B1):** the scoring snapshot isn't forward-only, and fresh
forward windows shift in distribution. A forward-window re-check happens at
the S6a shadow.
