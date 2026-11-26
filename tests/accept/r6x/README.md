# R6x acceptance tests: the contract they pin

R6x is the owner's S0a execution (plan §3.0 R6x row, §5 row 2 "S0a
De-drift"): R6's main PR merged, the deployed `chain.toml` reconciled. Its
Accept is

```
cascade-audit decided --config /root/cascade/chain.toml --state-dir /root/cascade   # exit 0
scripts/laned/accept R6x --evidence evidence/s0a_diff.json                          # consensus-key diff empty
```

These tests are the spec. The implementation PR may touch only the paths in
`PATHS` (the release `chain.toml` and the unit tests that anchor on its old
values). Conventions are W0c's (`tests/accept/w0c/README.md`). R6x owns no
T-rows and no invariants.

## What runs where

- `test_main_types_the_s0a_main_class_values`: the release `chain.toml`
  carries every main-class drift value R6 found on the deployed file
  (`fixtures/expected_main_values.json`, public values only). It fails on
  main until the S0a main PR merges (the expect-fail right reason).
- Collector/verifier tests: built from R6's fixtures (`tests/accept/r6/fixtures`)
  in a throwaway git repo. No host file is read.
- `test_laned_evidence_s0a_diff`: skipped without `LANED_EVIDENCE`. With it
  (`accept R6x --evidence evidence/s0a_diff.json`) it runs the verifier on
  the owner's evidence against this checkout. With `LANED_S0A_DEPLOYED=<file>`
  as well (on the box), it also re-derives the deployed side from the file.

## F6 (plan rev6.18 B9 main side, B12): `test_r6x_f6.py`

- **B9 main <- deployed.** The release `chain.toml` carries
  `[round] funded_sku_wall_seconds` `8xH100 = 13600`, `8xH200 = 12270`,
  `funded_sku_wall_measured_image` (rc15), and the two owner keys at their
  live values (owner 2026-11-23): `[training] budget_denomination_after =
  "points+mv20"`, `_block = 9165600`; `[static_guard] packed_sources =
  "reject"`, `_from_block = 9169200`. `cascade-audit decided` (the in-process
  `decided.drift_report` against the shipped record) has no failing finding.
- **Deployed edits.** `docs/notes/S0A_DEPLOYED_EDITS.md` has exactly one
  ` ```toml ` block: the two decided votes (`[v2] prelaunch_burn_from_block =
  9486900`, `image_decoupled_from_block = 9493200`) and `[scoring]
  init_gate_retired_from_block` at main's value. Main with the votes untyped
  and the block dropped (the deployed state) has 2 failing findings; with the
  block's lines applied, none. With `LANED_S0A_DEPLOYED=<file>` (on the box)
  the same holds for the real deployed file.
- **B12.** `[scoring] init_gate_mode` / `init_gate_tolerance` are no
  `ScoringConfig` fields, are listed in `RETIRED_KEYS["scoring"]` and change
  nothing when a file still carries them; `KothParams` keeps `init_gate_mode`
  (archived params replay) but not the tolerance. `koth_params(block)` is
  `"shadow"` before `init_gate_retired_from_block` and `"off"` (the
  drop-when-default value) from it; `0` = off everywhere. The archived
  shadow-era receipt fixture (`fixtures/receipt_shadow_init_gate.json`:
  mainnet round 9550553995117568107, entry rows trimmed, re-signed by
  `//Alice` with the pre-F6 code) loads with no unknown field, re-dumps
  byte-identically, verifies (received bytes and re-built body) and passes
  `check_koth_params` against the release file.
- The 19 operator-only keys get no marker (see the notes file: dedrift has no
  operator marker; a `LIVE OVERRIDE` marker would class 15 of them `main`).

## `s0a_evidence.py`

`collect` (the owner, on the box) and `verify` (CI, or by hand). Both read
only. Run with `env PYTHONPATH=<checkout> python -P tests/accept/r6x/s0a_evidence.py ...`.

`collect --deployed F --repo R --main-ref REF --state-dir D --effective-block N --out FILE`:

1. Reads `chain.toml` and `decided_votes.toml` from commit `REF` of repo `R`
   (never a work tree).
2. Runs R6 (`cascade.laned.dedrift.dedrift`) on `F` against that `chain.toml`
   with `--effective-block N`: classification counts, the main-class markers
   that still drift (`main_drift`), and the overlay body.
3. Runs the `cascade-audit decided` checks in process
   (`decided.drift_report` on `F` + `decided.state_report` on `D`, record from
   the commit's `decided_votes.toml`): exit code 1 if any finding fails, the
   failing findings' names, and the sha256 of the rendered report. It does
   not load `.env` files (the CLI's `main` does).
4. Computes the consensus-key diff (below) between `F` and
   `main ⊕ overlay.overrides`.
5. Writes `laned_s0a_diff/1`. `passed` is true iff the diff is empty,
   `main_drift` is empty and the audit exit code is 0. Exit 0 iff passed.

No config value is written except the overlay body (which is published
anyway). The consensus side is a sha256 fingerprint per key; operator-only
keys are never fingerprinted.

### Consensus keys (`r6x-consensus/1`)

Every leaf `[table] key` of either file, except:

- R6's operator-only class (`[generation]`, `[lane_d_ops]`, trainer-only
  recovery policy in `[round]`, `eval.gift_gate_data_dir`,
  `eval.calib_horizons`);
- the trainer's own `[round]` rental and payment knobs: a `[round]` key
  outside `netconfig.ALLOWED["round"]` that is neither a decided block nor a
  `decided.PIN_KEYS` pin.

A key's value is its **effective** value: the parsed `ChainConfig` attribute
(so a key absent from one file reads as its code default), else the raw TOML
value, else an "absent" sentinel. Overlay-class drift is compared against
`main ⊕ overlay`, because the overlay will carry it; base drift survives any
overlay (`build_config` merges onto `effective_raw(base)`), so it must be
zero.

### `laned_s0a_diff/1`

| key | value |
|---|---|
| `schema`, `wp`, `passed` | `"laned_s0a_diff/1"`, `"R6x"`, `true` |
| `deployed_sha256` | sha256 of the deployed file bytes |
| `main_commit` | full sha of the release commit |
| `main_chain_sha256`, `decided_votes_sha256` | sha256 of those files at `main_commit` |
| `effective_block` | the overlay's `effective_block` (an era start) |
| `overlay` | the R6 overlay body (`cascade-netconfig/2`) |
| `classification` | `{sha256, counts{main,overlay,operator-only}, markers, main_drift: []}` |
| `consensus_rule` | `"r6x-consensus/1"` |
| `deployed_fingerprints` | sorted `[table, key, sha256]` for every consensus key of either side |
| `consensus_diff` | `[]` (each row would be `{table, key, deployed, main}` fingerprints) |
| `audit_decided` | `{command, exit_code: 0, output_sha256, fails: []}` |

`verify` refuses (exit 1) unless every field has its type and value above,
the counts add up to `markers`, every fingerprinted key is a consensus key,
the overlay passes `check_body`, carries only overlay-class keys and has the
recorded `effective_block`, **and** the release side re-derives: the two
file hashes match `main_commit`, the overlay validates on that `chain.toml`,
every consensus key of `main ⊕ overlay` has a recorded fingerprint, and
every recorded fingerprint equals the release one (the diff, recomputed, is
empty). With `--deployed F` it also checks `F`'s sha256 and recomputes the
deployed fingerprints.

## Running

```
nice -n 19 env PYTHONPATH= python -m pytest --noconftest -p no:cacheprovider -q tests/accept/r6x/
```

Nothing heavy: no torch, no network, no chain.

## Failure on main

`test_main_types_the_s0a_main_class_values` fails with an `AssertionError`
naming the three keys main does not carry yet (`round.funded_operator_fallback`,
`round.funded_pod_skus`, `storage.hf_backup_repo`). Everything else passes or
skips. That is a right reason under W0c's expect-fail rule.
