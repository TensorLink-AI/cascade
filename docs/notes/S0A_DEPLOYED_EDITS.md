# S0a: edits the owner types into the deployed chain.toml (F6, rev6.18 B9/B12)

Owner action, at the next quiet restart (no legs in flight, outside the
validator scoring window). Never a script on the box: edit
`/root/cascade/chain.toml` by hand, then run

```
cascade-audit decided --config /root/cascade/chain.toml --state-dir /root/cascade   # exit 0
```

`tests/accept/r6x/test_r6x_f6.py` parses the TOML block below and checks
that it is exactly what turns `decided` green and keeps the init gate's
legacy window identical to the release file.

## Lines to type

Each key goes in the named table; a key that is already there is replaced
(the deployed file has `image_decoupled_from_block = 0` and no
`prelaunch_burn_from_block` / `init_gate_retired_from_block` line today).

```toml
[v2]
prelaunch_burn_from_block = 9486900
image_decoupled_from_block = 9493200

[scoring]
init_gate_retired_from_block = 4294967295
```

- `[v2] prelaunch_burn_from_block`, `image_decoupled_from_block`: the two
  decided votes `cascade-audit decided` reports as `[TYPE]` today
  (`prelaunch-95-r2`, `v2b-69630f92`; lock 9486000, applied 9486900 /
  9493200). Main already carries both. Simulated on a copy of the deployed
  file (2026-11-23): `decided` exit 1 before, exit 0 after (also with
  `--state-dir /root/cascade`).
- `[scoring] init_gate_retired_from_block` (B12): type it **with the F6
  release**, same value as the release `chain.toml`. The release no longer
  reads `[scoring] init_gate_mode` (retired, ignored), so without this line
  the deployed validator stops recording the legacy shadow floor at the
  restart instead of at the block, and `cascade-audit koth-params` (run
  against the release file) flags its receipts. `4294967295` is the
  far-future placeholder: the owner picks the real era boundary when the F6
  release is cut and types it in both files.

## Typed into main by F6 (no deployed edit)

Main now carries the deployed values (main <- deployed):

| key | value |
|---|---|
| `[round] funded_sku_wall_seconds."8xH100"` | 13600 (was 8801 in main) |
| `[round] funded_sku_wall_seconds."8xH200"` | 12270 (was 8333 in main) |
| `[round] funded_sku_wall_measured_image` | `sha256:5d41d94b…a67641` (rc15) |
| `[training] budget_denomination_after` / `_block` | `"points+mv20"` / 9165600 (owner decision) |
| `[static_guard] packed_sources` / `_from_block` | `"reject"` / 9169200 (owner decision) |

## Operator-only keys (19): no edit, no marker

These deployed-only (or deployed-valued) keys are box money/ops policy that
no other validator reads. Main stays unchanged and the deployed values stay.

`eval.calib_num_samples`, `eval.calib_windows`, `generation.exclude_offers`,
`round.funded_host_blocklist`, `round.operator_balance_reserve_usd`,
`round.operator_daily_spend_cap_usd`, `round.operator_inventory_host_files`,
`round.operator_lanes_operator_billed_only`, `round.operator_max_price_per_hour`,
`round.operator_standing_pods_usd_per_hour.{valeval-blackwell2,valpriv-4090}`,
`round.ops_continuation_{image_digest,interval_seconds,max_bytes}`,
`round.payer_projected_rate_per_hour`, `round.progress_first_step_seconds`,
`round.receipt_watch_validator_hotkey`, `round.funded_max_leg_cost_usd`,
`round.funded_ready_timeout_seconds`.

**Do not add `# LIVE OVERRIDE` markers to them.** The plan's "dedrift
`operator` marker" does not exist: `cascade.laned.dedrift.classify` decides a
marked key's class by rule, and it classes 15 of these as `main`
(`release_only_key`) and the two `eval.calib_*` as `overlay`. A marker would
put them into R6x's `main_drift` and fail the S0a evidence. Unmarked, R6
never classifies them, and the R6x consensus rule (`r6x-consensus/1`)
already excludes the `[round]` rental/payment knobs and the operator-only
`[generation]` table; `eval.calib_*` equal their code defaults (256, 32), so
their effective values match main.
