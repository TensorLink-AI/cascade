# W5h acceptance tests: queue selection

W5h (plan rev6.4 §3.0 W5h row, §1.2 "Rate limits", "Enforcement" and
"Coldkey resolution" (D22 option b), §1.5 "Maximum reveal delay" and
"State" (A8, A9, B8, B10), T6b, T18b, T18c, T18d) owns the validator's queue
selection inside `settle_lane_d`. These tests are the spec. The builder may
touch only the paths in `PATHS`. Every other lane D WP's accept dir (W5a–W5g,
W7a–W7d, W9, W14, W17, W12b) must stay green.

The W5h row, verbatim:

> Queue selection: FIFO by T; `max_judged_per_coldkey_per_settlement`, the
> global cap, coldkey re-check at reveal (`coldkey_changed`),
> `coldkey_unresolved` bound, `max_reveal_delay_days` close; typed skip
> reasons in the report (A8, A9) **(rev5)** `deregistered` void (court slot
> burns, with W8) and `ready_expired` (`ready_ttl_blocks`) handling, both
> typed in state and report.

## Fixtures

Every case is a real `settle_lane_d` call over real `pool_d/1` snapshots
(numpy only), reusing W5g's fixtures (`tests/accept/w5g/helpers.py`: the
armed `[lane_d]` parse with floors lowered to 24 windows / 4 sources, the
snapshot builder, the quantile forecasts, the signer). There is no external
service in these tests: no bench is needed, because nothing reaches the gate
except in the court-burn case, which reuses W5g's `Flow` and its recorded
bench host.

`helpers.early_pool` is a snapshot whose windows are all too early for an
entry revealed at `T_X`. Such an entry stays `pending_floor` (selected, never
frozen, never judged), however many settlements pass. The reveal-delay tests
need that.

## Contract

Unless stated otherwise, "typed" means the same `(status, reason)` appears in
`lane_d_state.json` and in the signed report's `entries`. A skip is listed in
`skipped` exactly once, with that reason. It is not in `judged`, has no
verdict and no rows, and the report validates against `lane-d-report/1`
(`helpers.assert_skip`).

### FIFO and the caps (§1.2 "Enforcement", T6b)

- Open entries are walked **FIFO by T**, the observed reveal (`reveal_ts`,
  consistent with `reveal_block` in every fixture), never by input order or
  digest order.
- `max_entries_per_settlement` (global) and
  `max_judged_per_coldkey_per_settlement` (per `coldkey_at_ready`) cap the
  selection. An entry over either cap is `pending` / `coldkey_cap_settlement`.
- It is **not charged**: its `used_windows` stay empty, its frozen set has no
  windows, and it has no rows.
- It is **carried**: it is judged at a later settlement once its coldkey has
  room. In the test, the coldkey's two judged entries lose at boundary 1.
- The caps are in the signed `params`.

On main all of this already holds (`_select` in `lane_d.py`), so these tests
pass and pin it.

### The coldkey re-check at reveal (D22 option b, A8)

The validator's re-check reaches `settle_lane_d` on the **entry record**: an
optional key `coldkey_at_reveal` in each `entries` item. It holds the coldkey
ss58 the validator read for the hotkey at reveal, inside the unpruned window.
`None` means the read failed or the coldkey is unknown.

| `coldkey_at_reveal` | outcome |
|---|---|
| key absent | no re-check. The entry is selected exactly as on main. This keeps every earlier WP's fixtures valid, and W8b's production wiring always sends the key (not pinned here) |
| equal to `coldkey_at_ready` | selected as usual |
| a different coldkey | **`closed` / `coldkey_changed`**, never judged. It holds no cap, so with a per-coldkey cap of 1 the coldkey's next entry is judged |
| `None` | **unresolved**, see below |

`coldkey_at_ready` empty or missing is unchanged: a registration refusal,
`closed` / `coldkey_unresolved` at once, whatever the bound (W5g pins this
case). The signed `entries[].coldkey` is always the value recorded at ready,
never the re-check value.

### The `coldkey_unresolved` bound (D22: "never a silent pending")

- An entry whose re-check is `None` is **open**: `pending` /
  `coldkey_unresolved`, skipped, never selected, never charged
  (`used_windows` empty), and it holds no cap slot.
- Each such settlement logs a record at WARNING or above whose message starts
  with **`lane_d_coldkey_unresolved`** and contains the digest's first 12 hex
  characters. That is the plan's cascade-ops alert name. It is also listed in
  `tests/accept/w3/fixtures/lane_d_intake_tags.py`, and W11 adds the follower
  pattern.
- **Resolution.** A later settlement whose `entries` lists the same digest
  with a readable `coldkey_at_reveal` resolves it. If that equals
  `coldkey_at_ready`, the entry is selected like any other and judged. If it
  differs, the entry is `closed` / `coldkey_changed`.
- **The bound.** With `[lane_d] max_coldkey_unresolved_settlements = N`
  (W2, default 4, `>= 1`), the entry is held open through its first N − 1
  unresolved settlements. At its **N-th** unresolved settlement it closes,
  `closed` / `coldkey_unresolved` (plan: `closed_unjudged:
  coldkey_unresolved`), skipped and alerted there too. With N = 1 it closes
  at the first one. A closed entry is never judged later.
- How the count is kept in state is not pinned. It must survive between
  settlements, because each settlement is a new call on the state file.

### The reveal-delay close (§1.5, A9, T18b)

- An entry's reveal deadline is `reveal_block + max_reveal_delay_days *
  7200`, in 12 s blocks. That is W17's `reveal_deadline_block`.
- At a settlement whose `boundary >= deadline`, an entry that is still open
  and pending closes **before selection**: `closed` /
  `reveal_delay_exceeded` (plan: `closed_unjudged: reveal_delay_exceeded`),
  skipped, never judged. That holds even when fresh windows would freeze it
  at that settlement.
- One boundary earlier (`boundary < deadline`) nothing closes. The fixture's
  deadline lands exactly on `boundary(9)`, so the limit is inclusive.
- **Logged:** a record at WARNING or above whose message starts with a
  `lane_d_` tag and contains `reveal_delay_exceeded` and the digest's first
  12 hex characters.
- **Slots freed:** with a per-coldkey cap of 1, the coldkey's next entry is
  judged at that settlement.
- The limit follows the key: two days leave `boundary(9)` inside it.
- How a `confirm_pending` or held (`bench_hold`) entry meets the limit is not
  pinned. Only `pending` entries are tested.

### Voids (rev5: B8, B10; T18c, T18d)

- **`deregistered`.** An open entry whose hotkey is not in the settlement's
  `uid_of` is `closed` / `deregistered`, skipped, and frees its coldkey slot.
  A crowned KING_D whose hotkey deregisters has a court slot with
  `uid: null` and `burn_reason: "deregistered"` (B8, with W8).
- **`ready_expired`.** An entry with `reveal_block - ready_block >
  ready_ttl_blocks` is `closed` / `ready_expired` and frees its slot. Exactly
  `ready_ttl_blocks` is still in time, the same as the intake's
  `reveal_block <= ready_expires_block` (W3).

These hold on main (`_register`, `_court`). The tests pin them.

### Where the code lives (keeps W5g green)

W5g's `test_every_reason_the_settlement_can_emit_has_a_case` reads
`cascade/validator/lane_d.py` and fails on any reason literal outside its
cases. It lists `coldkey_changed` and `reveal_delay_exceeded` as owned by
W5h. So:

- **`cascade/validator/lane_d_queue.py` (new)** holds the queue selection
  `settle_lane_d` calls: FIFO by T, both caps, the re-check, the unresolved
  bound and the reveal-delay close. Its non-docstring string literals include
  `coldkey_changed`, `reveal_delay_exceeded`, `coldkey_cap_settlement` and
  `coldkey_unresolved`. It imports neither torch nor bittensor.
- `lane_d.py` imports it and carries no `coldkey_changed` or
  `reveal_delay_exceeded` literal. That test passes on main and must stay
  true.
- Function names inside `lane_d_queue` are not pinned.

## Not pinned

- `max_open_per_coldkey`. It is advisory, at the intake (W3,
  `coldkey_cap_open`).
- How the production loop reads `coldkey_at_reveal` (W8b).
- The satellites' re-derivation. W9's `check_lane_d` already treats a
  `coldkey_unresolved` skip as a non-candidate. `cascade/audit/lane_d.py` is
  in PATHS only in case a W5h rule needs it there.
- Tie-breaking between equal T values.

## Invariants

None owned, and no `unchanged_when_unarmed` names, so no invariant 10
pairing. Every fixture is armed (`from_block = ARM_BLOCK`).

## Failure on main (36 collected, 13 fail, 23 pass)

| test | on main |
|---|---|
| `test_t6b_a_coldkey_changed_between_ready_and_reveal_is_refused`, `test_a_changed_coldkey_entry_frees_its_coldkey_slot`, `test_an_unresolved_entry_that_resolves_to_another_coldkey_is_changed` | assert: `coldkey_at_reveal` ignored, so the entry is judged (`confirm_pending`) |
| `test_t6b_an_unreadable_recheck_is_coldkey_unresolved_and_alerted`, `test_an_unresolved_entry_is_not_charged_and_holds_no_cap`, `test_an_unresolved_entry_resolved_later_is_judged`, `test_t6b_coldkey_unresolved_closes_after_the_bound[1,3]` | assert: judged, not `coldkey_unresolved` |
| `test_t18b_an_entry_pending_at_the_limit_closes_reveal_delay_exceeded`, `test_t18b_the_close_comes_before_judging_and_frees_the_coldkey_slot` | assert: stays `pending_floor` / is judged; the coldkey's next entry is capped |
| `test_the_queue_module_exists_and_imports_light` | `ModuleNotFoundError: cascade.validator.lane_d_queue` (in PATHS) |
| `test_the_w5h_reasons_are_emitted_from_the_queue_module`, `test_settle_lane_d_uses_the_queue_module` | assert: no such file / no import |

The FIFO, cap, void, edge, back-compat and key-parse tests pass on main.

A scratch control passed all 36 tests: a ~60-line `lane_d_queue.py` plus a
3-line call in `settle_lane_d`, in a throwaway worktree. The W5g, W7a, W9
and W17 accept dirs stayed green with it. The control kept two extra
per-entry state fields (a re-check flag and an unresolved count), and
`lane_d_state` accepted them.
