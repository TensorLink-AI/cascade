# WB15 acceptance tests: row claims and overlap deferral (LB2, TB16, TB21)

WB15 is in the plan §3.0 lane B track, with the rules in §12b.2 LB2
"Overlaps" and T-rows TB16 / TB21. It makes the trainer read the rows that
WB14 stores. RV4 note 9 says that today the rows are stored but ESFT
silently overrides them. These tests are the spec. The builder may touch
only the paths in `PATHS`.

On main, every failure is an `AssertionError`:

- a declared entry trains an ESFT pick instead of its rows;
- an overlapping later claim starts anyway, because main re-plans it with
  ESFT around the holder;
- `status()` has no `deferred` list.

`test_plan_rows_pinned` and `test_inv1_esft_only_unchanged_when_unarmed`
pass on main.

## What runs for real, and what is stubbed

- **Real:** funding goes through `FundingIntake.lane_b_fund` with the WB14
  `X-Lane-B-Declared-Rows` header, into the `cascade.funding.queue.LaneBQueue`
  file. Planning is `LaneBDriver.tick` over `expert_lane.LaneBQueue`. Status
  is `LaneBDriver.status()` and `publish_lane_b_status` (`status/lane_b.json`).
  The config comes from `parse_lane_b`.
- **Stubbed, external or GPU only:**
  - the driver's `LaneBOps`: chain commitments, storage, and pod dispatch.
    A dispatched leg stays open until the test returns it through the
    driver's own `done` callback.
  - the reveal oracle and the signature check.
  - an in-memory object store.
  - the king forward. `expert_lane.probe_target_routing` is swapped for a
    torch-free `RoutingProbe` subclass. Its ESFT `select` takes the 6
    highest-affinity rows per layer (row 0 is highest) that aren't excluded,
    and it records each exclusion set.
- **No torch:** an autouse guard makes any torch import raise.

## The contract

1. **A declared entry trains exactly its stored rows.** The leg's
   `selection` equals the entry's `declared_rows`, with empty layers
   dropped, and the leg isn't re-picked by ESFT.
2. **Claim order is the earliest reveal.** The key is the funded entry's
   `reveal_block`, then the hotkey. The plan only says "earliest reveal".
   The tie rule is the one the code already uses for reveal seniority
   (`funding.queue.select_field`, `LaneBQueue.next_dispatchable`). Funding
   order and chain-row order don't matter.
3. **Rows are exclusive against open legs and finished-but-unsettled legs**
   (DEC-CA-0059), whichever mode holds them. An open ESFT leg defers a later
   declaration. An open leg also defers a claim that revealed earlier but
   was funded after the leg started.
4. **A deferred claim is neither rejected nor charged.** Its funded entry
   stays `queued` with `attempts == requeues == 0` and
   `last_error_class == ""`. It is never in flight, it stays pending in the
   driver queue, and it writes no `outcomes` row. This holds across repeated
   ticks.
5. **Status.** `status()["deferred"]` is a list. The same list is in the
   published `status/lane_b.json`. Each row has these fields:

   | field | value |
   |---|---|
   | `hotkey` | the deferred hotkey |
   | `entry_ref` | its funded entry ref |
   | `deferred` | `"rows_held"` |
   | `holder` | `{"hotkey", "entry_ref"}` of the holder |
   | `expected_release` | `{"kind": "verdict" or "merge", "era": int}` |

   For an open or finished-but-unsettled holder, the kind is `verdict` and
   `base_era <= era <= base_era + max_base_age_eras`. The holder itself is
   never listed. With nothing deferred, the list is `[]`. It may be absent
   only in the all-ESFT unarmed case.
6. **Both modes coexist (TB21).** In one tick, declared entries are planned
   first, even when an ESFT entry revealed earlier. ESFT entries are then
   planned excluding exactly the held rows: open, finished and declared
   holders. The rows of a *deferred* claim aren't excluded, because a
   deferred claim holds nothing. The ESFT leg's `exclude` records the held
   rows.
7. **Release.** When the holder's leg ends (a miner fault fails its entry),
   the deferred claim starts on the next tick with its own rows, and the
   `deferred` list empties.
8. **Determinism.** The same plan comes out under permuted funding order
   and chain order, and under `PYTHONHASHSEED` 0 / 1 / 4242. The hash-seed
   check runs `python -P -m tests.accept.wb15.helpers` in fresh
   interpreters.

## Invariants (1, paired per invariant 10)

- **`test_inv1_esft_only_unchanged_when_unarmed`:** with no declared rows,
  planning is today's. The first ESFT leg has no exclusions, the second
  excludes exactly the first's rows, and nothing is deferred.
- **`test_inv1_esft_only_armed_nonempty_lane_d_effect`:** with a declared
  entry, the ESFT plans exclude its rows and an overlapping declaration is
  deferred.

## Not pinned

These are left to the builder:

- pending-merge holders (`kind: "merge"`), because building one needs
  `settle`, which needs torch;
- whether a funded declared claim whose hotkey is busy holds rows;
- the exact `era` within the bound above;
- any extra fields on a deferred row.

## Files

| file | pins |
|---|---|
| `helpers.py` | the scenario (real intake, queue, driver), the stub probe, the torch guard, the hash-seed entry point |
| `test_wb15_row_claims.py` | the plan text (TB16 / TB21 / WB15 verbatim); contract items 1–8; the invariant pair |
