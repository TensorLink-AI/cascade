# scripts/

Operator and maintenance tools. None of them run as part of the trainer,
validator or miner; each file's docstring has its full usage.

Start with [the operator runbook](../docs/OPERATOR.md) for service procedures.
Use `--help` and inspect outputs before any script that uploads or publishes.
Local analysis and `check_docs.py` do not establish that a deployment is live.

## Archive and champions

| Script | What it does |
|---|---|
| `sync_king.py` | Archive the current king generator into `champions/` with provenance. |
| `backfill_champions.py` | Backfill `champions/archive/` with every reign in the receipts' throne history. |
| `champions_archive.py` | Helpers shared by the two above (one folder per reign). |
| `scrape_kings.py` | Scrape kings and every other revealed generator into the private R2 archive. |
| `archive_sync.py` | Copy immutable manifests and receipts to the separate-account archive bucket. |

## Dashboards and published references

| Script | What it does |
|---|---|
| `publish_website.py` | Publish the dashboards to the manifest bucket, public-read. |
| `publish_chain_status.py` | Publish the dashboards' live `status/chain.json`. |
| `publish_round_bench.py` | Publish a round's public-benchmark report as a canonical `BenchReport`. |
| `publish_reference_bench.py` | Publish an official-checkpoint benchmark as a dashboard reference. |
| `wrap_official_toto2.py` | Package the official Toto2 weights in cascade's checkpoint layout for benching. |
| `refresh_leaderboard_reference.py` | Refresh the GIFT-Eval leaderboard reference the scoreboard reads. |
| `refresh_pareto_reference.py` | Refresh the benchmark Pareto model catalog and parameter metadata without publishing. |
| `make_bench_public.sh` | Make a bucket's `benchmarks/*` objects public-read. |

## Release and contract pins

| Script | What it does |
|---|---|
| `repin_worker_image.sh` | Re-pin `[training] train_image_digest` to a published worker image tag. |
| `worker_code_fingerprint.py` | Derive `[round] worker_code_fingerprint` for a pinned worker image. |
| `check_tls_trust.py` | Offline CA-store qualification for installed workers, benchmark environments and CI; `--system-only` checks before dependency installation. |
| `check_docs.py` | Offline check of current guides: CLI options/coverage, local links, loaded config values and reward tables/examples derived from payout code. |

## Analysis and probes

| Script | What it does |
|---|---|
| `replay_margin_decay.py` | Replay signed receipts under candidate margin-decay schedules. |
| `bench_seed_noise.py` | Calibrate the DEC-CA-0061 bench-gate tolerance on same-recipe seed replicas (report-only). |
| `bench_timing.py` | Time the benchmark sweep at `--workers 1` vs auto/N (optionally paired), check the reports are byte-identical, size the DEC-CA-0063 budgets (report-only). |
| `lane_b_horizon_jaccard.py` | Per-layer Jaccard of the lane-B horizon-target expert selections on one king checkpoint (DEC-CA-0059 "measure before arming horizons"). |
| `compute_promotion_error_vectors.py` | Precompute per-window error vectors for promotion selection. |
| `measure_reveal_jitter.py` | Measure timelock reveal jitter on a live network (sizes `reveal_margin_blocks`). |
| `probe_hub_enumeration.py` | Check whether the Hub registry lets third parties enumerate repos. |
| `scan_generator.py` | Static security scan of an untrusted generator tree; executes nothing. |

## Examples and config templates

- `example_generator/` — the minimal generator used by the miner docs and tests.
- `provision.example.toml`, `remote_hosts.example.toml` — templates to copy.
