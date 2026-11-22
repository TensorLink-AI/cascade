# WB17 acceptance tests: the lane B opening rule (D41 / LB4; TB17, TB18)

WB17 is in the plan §3.0 lane B track. Its rules come from §12b.4 LB4, D41
(decided: "(a) `open_tokens` + a passing lane A king check") and the T-rows
TB17 and TB18. The `[lane_b] open_rule` key itself is WB10's parse
(`cascade/shared/config/lane_b.py`). These tests are the spec. The builder
may touch only the paths in `PATHS`.

Plan text pinned verbatim (`test_plan_rows_pinned`):

- WB17: `| WB17 | Opening rule (D41, LB4): `open_rule` implementation; driver
  state records the rule and the satisfying evidence | `lane_b_driver` state
  `gate_rule`, `gate_evidence` | WB10 | ci + testnet | `scripts/laned/accept
  WB17` (TB17–TB18) |`
- TB17: `Opening rule (D41 default) on fixtures: tokens met + king-check pass |
  Gate opens; driver state records rule + evidence`
- TB18: `Tokens met, king check failing (or tokens short) | Gate stays closed`
- LB4: `The new key is `[lane_b] open_rule` (`"tokens_king_check"` |
  `"tokens"` | `"cogate"`; default `"cogate"`, so unarmed digests don't move).`

"D41 default" in TB17 is the decided option (a), `tokens_king_check`. The
shipped config default stays `cogate`.

On main, every failure is an `AssertionError`:

- `tokens_king_check` and `tokens` are parsed but ignored: the driver still
  runs the co-gate, so with one king the gate stays closed ("no previous king
  to compare routing");
- `LaneBDriverState` has no `gate_rule`, `gate_evidence` or
  `gate_closed_reason`;
- `RunnerLaneBOps` has no `king_check`.

The config tests, `test_inv1_open_rule_unchanged_when_unarmed` and the
offline `test_verifier_*` tests pass on main. `test_evidence_*` skips.

## What runs for real, and what is stubbed

The harness is the same as WB15's and WB18's.

- **Real:** funding goes through `FundingIntake.lane_b_fund` into the
  `cascade.funding.queue.LaneBQueue` file. The commitment is built by
  `format_expert_targets`. The config comes from `parse_lane_b`. The gate and
  dispatch run in `LaneBDriver.tick`, over its real state file.
- **Stubbed (external or GPU only):**
  - the driver's ops: commitments, pod dispatch, and **`king_check()`**. It
    returns the latest lane A king check record the validators stamped on
    their signed receipt (DEC-CA-0055 `king_check`: `result`, `round_id`,
    `block`, `king_pointer`, ...), or `None` when none is readable;
  - the king forward, through a torch-free `RoutingProbe`. Every king routes
    the same way, so the co-gate saturates whenever it is measured;
  - the reveal oracle and the signature check.
- **No torch:** an autouse guard makes any torch import raise.
- **No sentinel:** `suite_targets` is off, so `block_hash` must never be read.

## The contract

1. **Rule forms.** `open_rule` is one of `tokens_king_check`, `tokens` or
   `cogate`, and the parsed `LaneBConfig.open_rule` keeps it. The default is
   `cogate`.
2. **Invalid rule refused at load.** Any other value, including a non-string,
   raises `ValueError` naming `open_rule` from `parse_lane_b`.
3. **`tokens_king_check` (D41 a).** The gate opens on a tick where
   `lineage_tokens >= open_tokens` **and** `ops.king_check()` returns a record
   with `result == "pass"`. No previous king is needed, so it can open on the
   first king. The funded entry starts in the same tick.
4. **`tokens` (D41 b).** The gate opens when `lineage_tokens >= open_tokens`
   (equality counts), with no king check needed.
5. **The record (TB17).** On opening, `LaneBDriverState` holds:
   - `gate_opened_block`, the opening tick's block;
   - `gate_rule`, the configured rule;
   - `gate_evidence`, a non-empty object with `lineage_tokens` (the count
     that met the rule) and `open_tokens` (the threshold). Under
     `tokens_king_check` it also has `king_check`, the passing record with at
     least `result`, `round_id` and `king_pointer` copied from it. That is the
     check that satisfied the rule, not an earlier failing one;
   - `gate_closed_reason == ""`.

   All of these are persisted in `lane_b_driver.json`.
6. **Strict JSON, deterministic.** `gate_evidence` survives a strict
   `json.dumps(allow_nan=False)` / `json.loads` round trip unchanged. It
   holds only plain dicts, lists, strings, ints, finite floats and bools,
   with string keys. The same chain facts in another work directory, at
   another wall-clock time, give the same bytes.
7. **Closed (TB18).** While the rule is unmet, the tick returns `[]`,
   dispatches nothing and opens no leg. The funded entry stays `queued` with
   `attempts == requeues == 0` and `last_error_class == ""`. `gate_rule` is
   `""` and `gate_evidence` is `{}`. The typed reason `gate_closed_reason` is
   persisted:
   - `tokens_short`: `lineage_tokens < open_tokens`, under either rule. It is
     checked before the king check, so it wins when both fail;
   - `king_check_not_passed`: the record's `result` is `fail` or `deferred`;
   - `king_check_unavailable`: `ops.king_check()` returned `None`.

   These three are the whole enum. Repeated closed ticks stay closed and
   uncharged.
8. **Evidence that stops satisfying the rule: the gate stays open.** The plan
   doesn't address this case. The driver documents the gate as a one-way latch
   ("lane B never re-closes"), and D45 puts lane B live at full share from the
   block the rule is satisfied. So a later failing check doesn't re-close the
   gate: new entries still start, and `gate_opened_block`, `gate_rule` and
   `gate_evidence` never change after opening.
9. **Survives restart.** A new `LaneBDriver` on the same state file has the
   same `gate_opened_block`, `gate_rule` and `gate_evidence`. It dispatches
   without reading the rule's facts again (no king check needed).
10. **Production wiring.** `RunnerLaneBOps.king_check()` returns the record
    from the runner's verified receipt read
    (`runner._receipt_king_check()[0]`, in `runner_promotion.py`), or `None`
    when that read returns `None`.

## Invariants (1, paired per invariant 10)

| test | pins |
|---|---|
| `test_inv1_open_rule_unchanged_when_unarmed` | `open_rule` absent (`cogate`): today's gate. One king stays closed, the next era's king opens it with tokens met, tokens short stays closed, and the king check is never read |
| `test_inv1_open_rule_armed_nonempty_lane_d_effect` | `tokens_king_check`: the same facts open the gate on the first king, and the king check is read |

## Testnet evidence (L4)

WB17 runs "ci + testnet". `scripts/laned/accept WB17 --evidence
evidence/wb17_gate.json` exports `$LANED_EVIDENCE`. `test_evidence_*` then
checks it with `helpers.evidence_problems`. Without the env var it skips, so
the WP can reach L3 first. The file is strict JSON:

```
{"schema": "laned_wb17_gate/1", "source": "testnet", "netuid": 259,
 "open_rule": "tokens_king_check", "open_tokens": <int > 0>,
 "king_check": <the receipt king_check record the gate read, result "pass">,
 "closed": <the testnet trainer's lane_b_driver.json while closed (TB18)>,
 "opened": <the same file after the gate opened (TB17)>}
```

The verifier requires:

- `closed.gate_opened_block == 0`, a typed `closed.gate_closed_reason`, and an
  empty record;
- `opened.gate_opened_block > 0`, `opened.gate_rule == open_rule` and
  `opened.gate_closed_reason == ""`;
- `gate_evidence.lineage_tokens >= open_tokens == gate_evidence.open_tokens`;
- `gate_evidence.king_check` passing, with the receipt record's `round_id`
  and `king_pointer`.

The `test_verifier_*` tests check, offline, that the verifier rejects every
tampered field.

## Not pinned

These are left to the builder:

- extra keys in `gate_evidence`, such as the block, era, king or routing
  numbers;
- whether a passing check for a king pointer other than the tick's king
  counts (the fixtures always match);
- whether `tokens` or a latched gate reads `ops.king_check()`;
- what `cogate` records in `gate_rule`, `gate_evidence` and
  `gate_closed_reason` (its decisions are pinned unchanged);
- the gate fields in `status()` and `status/lane_b.json`, and log wording;
- how an older state file that latched under `cogate` (with no `gate_rule`)
  is shown.

## Files

| file | pins |
|---|---|
| `PATHS` | the modules WB17 may change |
| `helpers.py` | the scenario (real intake, queue, parser, driver), the stub probe and ops (`king_check`), the torch guard, the strict-JSON helpers, the testnet evidence verifier |
| `test_wb17_open_rule.py` | the plan text, contract items 1–10, the invariant pair |
| `test_wb17_evidence.py` | the testnet evidence check (skips without `$LANED_EVIDENCE`) and the offline verifier tests |
