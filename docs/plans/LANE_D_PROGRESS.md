# Lane D rev3 progress

Branch `claude/sleepy-cray-7uutm1`. One PR, owner merges. Report-only:

- no chain.toml values, consensus code, `cascade/pool/` or prod-box changes;
- W2–W12 are not built.

Status values: `todo` | `in progress` | `done` | `blocked: needs owner` |
`blocked: <other>`.

## Owner actions

These are the inputs the S1 runs need. They're never guessed; the dependent
runs stay blocked until each one is answered here.

1. **OA1, prior-king pairs (B1 null type 2).** List the checkpoint refs
   (`repo@sha256:…`) of ≥ 3 consecutive lane A v2 kings that are light
   fine-tunes of each other (same lineage, adjacent reigns). Write them here
   as `king_t, king_t+1` pairs.
2. **OA2, seed replicas (B1 null type 2b).** Do any same-corpus,
   different-seed replicas exist (DEC-CA-0061, docs/OPERATOR.md
   seed-noise)? If yes, give their refs. If none, write "none".
3. **OA3, per-era false-crown target.** The recommended value is 5%. Confirm
   or set a value; it gates S4 (T14) and sizes `d_margin`.
4. **OA4, pod approval for B1 scoring and the B2 16k memory fit.** Either:
   - (a) use valpriv-4090 **outside lane A duel windows** (no rental; it
     touches an eval pod), or
   - (b) approve renting one pinned-torch pod (torch 2.4.1+cu124 capable,
     ≥ 24 GB), with a cost cap.

   State which, plus the time window.
5. **OA5, forge-mirror read access for B2.** The B2 history-length counts
   (≥ 16k, and the S8 lengths) and first-seen checks can't come from pool
   snapshots: they truncate history to 4096+720 and carry no timestamps.
   Approve running `scripts/laned_w1_measure.py` (or a mirror variant) on
   GitHub Actions or the builder with the `TSFORGE_S3` read credentials. Say
   where.
6. **OA6, pinned pool snapshots for B2's snapshot-diff bound.** Approve
   fetching ≥ 2 consecutive pinned daily `pool` snapshots and one `pool_mkt`
   run of snapshots to a non-box host (GitHub Actions or the builder), and say
   where. They're never fetched to the prod box.
8. **OA8, pod approval for LB6 (WB19).** Approve one pinned-torch testnet pod (8×H100, matching the lane A full-leg SKU) for an expert-only leg vs a full leg on the same SKU. Give a cost cap.
7. **OA7, input plan.** The brief's PLAN START/END block contained only the
   template placeholder. I used the latest local plan (v2 design + the v3
   loop-readiness §3) as commit 1. If you meant a different v2 file, say so
   and the PR is rebuilt on it.

## Items

| Item | Status | Notes |
|---|---|---|
| Commit 1: input plan unchanged | done | `docs/plans/LANE_D_PLAN.md` (see OA7) |
| A1 forward-only: data time AND first-seen; `max_mirror_lag_hours`; first-seen WP | done | first-seen absence verified (no `first_seen`/`ingested_at` in `cascade/pool/`) |
| A2 repeated attempts, confirmation duel, per-era margin | done | |
| A3 horizon h = 64 | done | already in v2 §1.1 "Horizon" |
| A4 testnet exit criteria and floors | done | repo fact corrected: pool_mkt scheduled runs skip while `[marketcast] from_block = 0` (both networks) → new WP R15 |
| A5 descope long context to 4k/8k/16k; S8 | done | |
| A6 BF16 only | done | |
| A7 unpaid share [DECIDE] | done | |
| A8 coldkey caps per lane + resolution [DECIDE] | done | |
| A9 verifiability | done | |
| A10 low-frequency check | done | |
| A11 smaller fixes | done | |
| B1 script + tests (`scripts/null_fpr.py`) | done | py_compile + ruff clean; 10 tests; CI runs them (never run on the prod box) |
| B1 run | blocked: needs owner | OA1, OA2, OA3, OA4 |
| B2 script + tests (`scripts/laned_w1_measure.py`) | done | py_compile + ruff clean; 8 tests; must run off-box (OA6) |
| B2: first-seen exists? | done | No: `grep -rnE "first_seen\|ingested_at\|ingest_time\|landed_at\|LastModified" cascade/pool/` finds nothing |
| B2 run: snapshot-diff bound, pool_mkt | blocked: needs owner | OA6 |
| B2 run: history ≥ 4k…128k, pool_d size | blocked: needs owner | OA5 |
| B2 run: 16k memory fit | blocked: needs owner | OA4 |
| C: DEC-CA-0070 draft (proposed) | done | 0069 is taken by unmerged PR #482 |
| §15 incentive review folded in as [DECIDE] D25–D32 | done | |
| Lifecycle trace `docs/plans/lane_d_lifecycle.md` | in progress | |
| rev4 lane B (§12b, WB10–WB21, TB10–TB22, SB track, D35–D42) | done | |
| DEC-CA-0071 amendment rev4 (on top of PR #482's text) | done | |
| LB6 expert-leg cost measurement | blocked: needs owner | OA8 |
| PR | todo | |
