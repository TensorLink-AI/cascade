# R11 acceptance tests: shipped-config startup gates (T26)

R11 builds a harness that runs a `chain.toml` through every real validator and
trainer startup gate at a pinned chain head (plan §3.0 R11 row, §4 T26, §13
"startup gates at externals", §14.1). These tests are the spec. The builder
may touch only the paths in `PATHS`: `cascade/laned/startup_gates.py`, and
`cascade/laned/__init__.py`, which is shared with R1, R2, R6 and R10. Keep
that file and the other WPs' files intact. R11 owns T26 and no invariants.

**Why.** The 2026-11-13 incidents were configs that loaded fine but that a
startup gate refused at the live head:

- the 128-byte readiness-note crash-loop (`v2b_note_problem`, PR #457);
- one release later, the calendar-launch trial (`check_calendar_launch`,
  PR #460).

CI ran neither gate against the shipped file at a post-launch head (OPSLOG
2026-11-13 13:10 LESSON).

## Command

```
python -P -m cascade.laned.startup_gates --config FILE --head-block N \
    --role validator|trainer [--head-timestamp SECONDS] [--state-dir DIR]
```

- **cwd and paths.** It runs with cwd = the repo root, and a relative
  `--config` resolves against the cwd.
- **Arguments.**
  - `--head-block N` is the injected chain head. It is required, and the
    harness never reads a head from the chain.
  - `--head-timestamp SECONDS` is the chain time of that head, in UNIX
    seconds. The fixture chain serves `block_timestamp(b) = SECONDS - 12 * (N - b)`.
    Without it, a block-timestamp read raises, just as on a chain that can't
    serve one. The real gate then keeps the calendar trial.
  - `--state-dir DIR` holds the node's persisted activation records, under
    the file names the services read: `activation_state.json`,
    `activation_forfeit_state.json`, `activation_margin_v2_state.json`,
    `activation_prelaunch_burn_state.json`,
    `activation_image_decoupled_state.json` and
    `activation_shared_ladder_state.json`, in the real `ActivationStore`
    format. It stands in for the validator's state-db directory and the
    trainer's work root. If it's absent, the node is fresh with no records.
    The harness never writes into it.
- **No network, no torch.** The tests run the harness under a guard that
  raises on any socket connect and on any `torch` import. Chain reads are
  served from the injected head:
  - `current_block` and `finalized_block` return N;
  - timestamps come from `--head-timestamp`;
  - commitments are empty;
  - the node has no wallet and no hotkey.

  Nothing reads S3 or HF.
- **Output.** Stdout carries exactly one line per entry of `GATES[role]`, in
  order:
  `<PASS|FAIL|WARN|SKIP> <gate>[: <reason>]`. Logging and diagnostics go to
  stderr.
  - The reason is the gate's refusal text, flattened to one line (newlines
    become spaces), so it contains the real message.
  - A FAIL stops the sequence, as `main.py` does. Every later gate prints
    `SKIP`.
  - An **advisory** gate is one whose refusal the service catches and
    carries on past. It prints `WARN` and does not stop the sequence.
- **Exit codes:**

  | Code | Meaning |
  |---|---|
  | 0 | no FAIL |
  | 1 | any FAIL |
  | 2 | usage error (missing `--head-block`, a role other than `validator`/`trainer`, or a `--config` that doesn't exist) |

## Gate list (`cascade.laned.startup_gates.GATES`)

`GATES` maps `"validator"` and `"trainer"` to an ordered sequence of gate
objects, each with:

- `name` (unique within its role);
- `calls`, a non-empty tuple of the callee names the gate runs, exactly as
  the startup path spells them;
- `advisory` (bool).

Each gate calls the real function or functions. Nothing is re-implemented.

The required names are below. Others are free, for example `vote_pin` or
`bench_signer`. `require_vote_pin` must be in some gate's `calls`, and it
must run before the loader.

| role | order (required names) | real function |
|---|---|---|
| both | `code_path` first | `verify_release_code_path` (the #431 guard; `CASCADE_RELEASE_PATH` / first `PYTHONPATH` entry) |
| both | `loader` before `launch_ready` | `load_chain_config` under `require_vote_pin()` (it runs the `[lane_d]` parse and the v2b / shared-ladder pin checks) |
| both | `launch_ready` | `assert_launch_ready(cfg, role=...)` |
| validator | `launch_ready` < `activation_note` < `calendar_launch`; `activation_note` < `v2b_typed_notice` | `_activation_startup`: `v2b_note_problem` at the head (its worst case covers head + 216000), `v2b_typed_notice_problem`, `prelaunch_retirement_problem`; then `run_forever`'s `check_calendar_launch` |
| trainer | `launch_ready` < `calendar_launch` < `worker_release` < `v2b_typed_notice` | `run_forever`: `check_calendar_launch`, `check_worker_release` (**advisory**: `run_forever` logs and continues), then `_image_decoupled_startup`'s `v2b_typed_notice_problem` |

The only advisory gate is the trainer's `worker_release`. The validator has
none.

## Kept in sync with `main.py` (AST)

The startup path is these scopes (`helpers.STARTUP_SCOPES`). For a
`run_forever`, only the statements before its first top-level `while` count.

- **validator:**
  - `cascade/validator/main.py:main`;
  - `ValidatorRunner.run_forever`;
  - `ActivationMixin._activation_startup`;
  - `ActivationMixin._v2b_typed_notice_startup`.
- **trainer:**
  - `cascade/trainer/main.py:main`;
  - `TrainerRunner.run_forever`;
  - `TrainerRunner._image_decoupled_startup`.

Two checks keep the list in sync:

1. Every name in a gate's `calls` is called somewhere on that role's startup
   path.
2. Every call on the path whose last name segment matches
   `^(verify_|require_|assert_|check_)|_problem$|^load_chain_config$` is in
   some gate's `calls`.

The one exclusion is the validator's `verify_direct_receipt`, the offline
`--direct-receipt` replay mode. A new `check_*` / `*_problem` gate added to
`main.py` or a startup method therefore fails the tests until the harness
runs it. A self-test plants such a call to prove the scanner sees it, and
also proves that a call inside the poll loop is ignored.

## What the tests pin

| Test | Pins |
|---|---|
| `test_startup_gates_module_exists`, `test_gate_list_*`, `test_gate_calls_*` | the module, `GATES`, the required names, the order, and the advisory set |
| `test_every_listed_gate_*`, `test_every_startup_gate_call_*`, `test_scanner_*` | the AST sync with the real startup path |
| `test_t26_shipped_config_passes_every_gate` | T26: `chain.toml` and `chain.testnet.toml`, both roles, at head 9,557,440 (a 2026-11-21 mainnet head, 12 s/block from the 9,547,200 era start) and at head + 216,000, with chain time. Exit 0, every gate `PASS` (`worker_release` may `WARN`) |
| `test_t26_*_without_head_timestamp` | both shipped files type `[v2] from_block`, so they start without chain time |
| `test_crashloop_20261113_*` | the R2 fixture (`tests/accept/r2/fixtures/crashloop_20261113.json`): the pre-decision release with `[v2] image_decoupled_from_block = 0`, the node's rolling and pre-launch records in `--state-dir`, at head 9,493,750. `FAIL activation_note` whose reason contains the logged refusal and the logged note. The gates before it pass. The text is checked against the live `v2b_note_problem` |
| `test_460_*` | the pre-decision release, v2b typed, dated launch pending, at head 9,500,400: `FAIL calendar_launch` ("image_decoupled_from_block=9493200 precedes [v2] from_block") for both roles without chain time, and `PASS` with chain time past `launch_at_utc` (the #460 fix) |
| `test_invalid_lane_d_key_fails_at_loader` | `[lane_d] share = 1.5` gives `FAIL loader` with the real parse error |
| `test_loader_runs_under_the_vote_pin` | a wrong `[v2] image_decoupled_feature` gives `FAIL loader` "… refusing to start". That happens only with `require_vote_pin()` in force, which proves the harness arms it first |
| `test_placeholder_fails_at_launch_ready` | `[manifest] trainer_hotkey = ""` gives `FAIL launch_ready` |
| `test_shadowed_release_fails_at_code_path` | `CASCADE_RELEASE_PATH` set to another directory gives `FAIL code_path` first, then every other gate `SKIP` |
| `test_head_block_is_required`, `test_unknown_role_*`, `test_missing_config_*` | exit 2 |

Run (prod box, tests only, no torch):
`nice -n 19 env PYTHONPATH=<wt> /root/cascade/.venv/bin/python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -q <wt>/tests/accept/r11/`
