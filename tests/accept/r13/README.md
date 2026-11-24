# R13 acceptance tests: elastic lane D hosts

R13 (plan §3.0 R13 row: "`pool_d` builder VM provisioning: disk, wall, retry,
credentials placement (no secrets printed), systemd timer"; §1.7 "Build
host"; W6d: one `pool_d` snapshot every 3 h, each build < 3 h) plus the owner
decisions of 2026-11-24:

- the `pool_d` builder runs on **Shadeform, CPU only**, elastically: up for
  each 3-h build, down after publish unless `keep_warm`;
- **GPU is always Lium; Shadeform is CPU only.** The lane D eval host(s) rent
  through the existing Lium adapter; with no qualified Lium GPU the controller
  retries Lium for `retry_window_minutes`, alerts, and never falls back to a
  Shadeform GPU. The Shadeform client refuses any type with `num_gpus > 0`;
- many submissions ⇒ `ceil(entries / per_host_entries)` hosts (capped by
  `max_hosts` and the spend caps), the cohort split deterministically from the
  boundary block hash and the entry ids, KING_D and the house on every host.

The tool is `cascade-laned elastic` (`cascade/laned/elastic.py`), configured
by `deploy/laned/elastic.toml`. Owner execution WP: the live run writes
`evidence/r13.json`; these tests pin the CI-checkable parts.

## Ground rules (the tests enforce them)

- **Offline.** Fake Shadeform HTTP surface and fake Lium adapter
  (`helpers.py`); no network, no torch, nothing rented, no keys read.
- **Spend guards.** A per-role `max_price_per_hour` and `monthly_cap_usd`
  (both > 0, never off) over an append-only spend ledger; an offer over either
  is never rented; `--dry-run` / `quote` prints the chosen type and price and
  creates nothing.
- **Always delete on failure.** Any failure after a create deletes the
  instance and verifies it is gone (alert `teardown_unverified` otherwise).
- **Host keys pinned.** Every ssh/scp carries `StrictHostKeyChecking=yes`
  and a per-host `UserKnownHostsFile`; keys are pinned with R12's
  `pin-host-key`. Never `accept-new` or `/dev/null`.
- **First-seen append-only.** Builder state lives in the `pool_d` bucket under
  `<pool_d_prefix>/_builder/`; `state-push` refuses (`state_conflict`) a log
  the bucket copy is not a byte prefix of, or a HEAD that moved since
  `state-pull`; `mirror-sync` never touches `first_seen/`.
- **Same host per cohort (B6).** A cohort keeps its hosts; `down` refuses
  (`cohort_open`) until `--cohort-done`; `up` for another cohort refuses.
- **Qualified environments (B5).** F5's `deploy/laned/scoring_envs.toml` is
  authoritative when present (only `status = "qualified"` counts); otherwise
  `fallback_qualified_gpu_models`.

## Exit codes

| code | meaning |
|---|---|
| 0 | ok |
| 2 | usage / config |
| 3 | `no_capacity` |
| 4 / 5 | `price_cap` / `monthly_cap` |
| 6 | `gpu_refused` (Shadeform GPU type) |
| 7 | `cohort_open` |
| 8 | `not_qualified` |
| 9 | `state_conflict` |
| 10 | `step_failed` |
| 11 | `teardown_unverified` |
| 12 | `host_mismatch` |

## What is pinned

| Row text | Test |
|---|---|
| disk ≥ 150 GB, caps, providers fixed | `test_shipped_config_parses_with_the_owner_defaults`, `test_config_refusals` |
| cheapest available CPU type, never a GPU | `test_cpu_offers_cheapest_available_and_never_a_gpu_type`, `test_shadeform_client_refuses_gpu_types` |
| dry-run rents nothing | `test_dry_run_prints_choice_and_rents_nothing`, `test_dry_run_rents_nothing` |
| $/h and monthly caps | `test_price_cap_and_monthly_cap_refuse_before_renting`, `test_ledger_counts_only_this_month`, `test_monthly_cap_limits_host_count`, `test_over_cap_offers_are_never_rented` |
| delete on failure | `test_failed_boot_deletes_and_verifies`, `test_unverified_delete_alerts`, `test_cycle_deletes_on_failure_even_with_keep_warm`, `test_lemon_is_recorded_and_every_rented_host_deleted` |
| the W6 build sequence, elastic cycle, keep-warm | `test_run_script_is_the_w6_sequence`, `test_cycle_builds_then_deletes`, `test_keep_warm_keeps_the_vm_after_a_good_build` |
| credentials placement (names only, mode 600) | `test_env_template_names_only_mode_600` |
| first-seen append-only on object storage | `test_r13_state.py` |
| Lium only, retry then alert | `test_eval_role_is_lium_only_no_shadeform_gpu_path`, `test_no_lium_gpu_retries_lium_then_alerts` |
| qualified list | `test_qualified_list_from_scoring_envs`, `test_fallback_list_only_without_scoring_envs`, `test_not_qualified_refuses` |
| host count, same host per cohort | `test_hosts_needed`, `test_up_rents_n_hosts_cheapest_first_under_cap`, `test_same_hosts_for_a_cohort_and_down_waits_for_it` |
| deterministic shards, king on every host, 3 hosts = 1 host | `test_assignment_is_deterministic_and_anchors_every_host`, `test_three_identical_hosts_give_the_one_host_verdict`, `test_merge_refuses_a_king_that_differs_between_hosts`, `test_shard_plan_maps_shards_to_live_hosts` |
| R12 kit for the eval host | `test_eval_bootstrap_uses_the_r12_kit` |

Not here (consensus, owner/plan decision): the signed report's per-model host
fields and per-host king scores, and a verdict that pairs each challenger with
its own host's king when hosts differ. These extend F5 (PR #830) and need a
plan revision; `merge_shard_scores` refuses `host_mismatch` until then.
