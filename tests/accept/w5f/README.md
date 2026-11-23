# W5f acceptance tests: house entry and refresh

W5f (plan rev6 §3.0 W5f row, §1.4 "The house", §2 items 1-2, §1.5 "State",
§4 T10, §3.1, §17) pins how the house enters lane D and what a new house
pointer does at a settlement boundary. These tests are the spec. The builder
may touch only the paths in `PATHS`. The W1.5, W5a-W5e, W7a-W7c, W8 and W3b
tests must stay green.

Plan text these tests pin:

- §1.4: "the house is the checkpoint digest lane A currently treats as king
  ... It's refreshed whenever that digest changes ... It enters at that
  settlement, and its T is the block at which that digest was published."
  "If the house holds KING_D, the lane D king slot goes unpaid. Prior miner
  kings keep their decaying tails." D21 decided: burn. D7/D9: the house is
  never in the court and never a prior king.
- §2 item 2: "The house can retake KING_D when a better lane A king beats it
  on forward windows. The lane D king slot then burns."
- T10: "House pointer swapped to a stronger checkpoint | House retakes; slot
  burns; prior miner tails keep paying".

Every settlement goes through the real `settle_lane_d` on fixture pools,
forecasts and a recorded bench (the only external services). No state is
hand-written.

## Contract

### 1. The signed house pointer (`cascade/validator/lane_d_house.py`, new)

- `HOUSE_SCHEMA = "lane-d-house/1"`.
- `HOUSE_FIELDS = ("schema", "digest", "reveal_ts", "signer", "signature")`.
  A pointer has exactly these keys.
- `HOUSE_ERROR_CODES`: exactly `unreadable`, `bad_json`, `duplicate_key`,
  `not_object`, `unknown_schema`, `missing_field`, `unknown_field`,
  `bad_value`, `bad_signature`, `wrong_signer`.
- `class LaneDHouseError(ValueError)` with `.code` in `HOUSE_ERROR_CODES`.
- `sign_house_pointer(*, digest, reveal_ts, signer) -> dict`. `signer` is a
  keypair. It returns the pointer, with `signer` set to the key's ss58 and
  `signature` the sr25519 signature (64 bytes as 128 lower-case hex, no
  `0x`) over the **canonical body**. The canonical body is the pointer
  without `signature`, serialised as `json.dumps(sort_keys=True,
  separators=(",", ":"), ensure_ascii=True, allow_nan=False)` in UTF-8, the
  same canonical JSON the lane D report uses. A satellite can build and
  check the pointer on its own.
- `verify_house_pointer(pointer, *, signer_ss58) -> {"digest", "reveal_ts"}`
  returns the house that enters: the pointer's digest and T, nothing else.
  The input is never mutated. Each refusal below has exactly one defect, so
  the order of the checks is free:
  - not a dict: `not_object`;
  - a missing key: `missing_field`;
  - an extra key: `unknown_field`;
  - `schema` other than `HOUSE_SCHEMA`: `unknown_schema`;
  - `digest` not exactly 64 lower-case hex (no `sha256:` prefix, no
    upper case), or `reveal_ts` not a non-negative int (a bool, float or
    string is refused), or `signer` empty, or `signature` not 128 lower-case
    hex: `bad_value`;
  - `signer` ≠ `signer_ss58`: `wrong_signer`;
  - a signature that does not verify over the canonical body (tampered
    digest or T, or signed by another key while claiming the pinned
    signer): `bad_signature`.
- `load_house_pointer(path, *, signer_ss58)` accepts a str or a path. It
  reads the file strictly, then verifies it:
  - an absent path or a directory: `unreadable`;
  - empty, truncated or non-UTF-8 bytes, or `NaN`: `bad_json`;
  - a repeated key at any depth: `duplicate_key`. A pointer can't carry two
    digests and let the loader pick one;
  - a JSON non-object: `not_object`;
  - otherwise the `verify_house_pointer` codes.
- **Deterministic.** sr25519 signatures are randomised, but two signatures
  of the same pointer verify to the same house and settle to
  byte-identical lane D states.

D46 (T_house = the latest commit block in the house's lineage) is computed
by whoever produces the pointer. W5f takes T from the signed pointer as
given.

### 2. `settle_lane_d(house=...)` fails closed

- `house` is the verified house `{"digest", "reveal_ts"}`, which is what
  `verify_house_pointer` and `load_house_pointer` return.
- Armed, the settlement refuses anything else with `LaneDHouseError` before
  any state, report or pool is read or written:
  - not a dict: `not_object`;
  - a missing key: `missing_field`;
  - an extra key: `unknown_field`;
  - a digest that is not 64 lower-case hex, or a `reveal_ts` that is not a
    non-negative int: `bad_value`.
- On refusal the state file and every persisted report stay byte-identical.
  On a first settlement, no state file and no reports directory are
  created. A valid call after a refusal settles exactly as on an untouched
  copy.
- Unarmed (invariant 10 pair), any house, even an invalid one, returns
  `report=None, pay=None, receipt=None` and writes nothing.

### 3. Refresh while the house holds KING_D

- A new house at boundary B moves the throne in place at B, with no duel.
  The state's `house` and `king` become the new digest and T
  (`king.house = true`, `king.hotkey = null`), and `prior_kings` is not
  touched. The old house is never a prior king.
- The report's `house` and `king_after` are the new house record
  `{digest, hotkey: null, house: true, reveal_ts}`, and `crowned` is
  `null`.
- The slot burns: `court_uids[0] = None`, `burn_reason = "house_king"`.
- Idempotent: re-applying the same pointer leaves `house`, `king`,
  `prior_kings` and `entries` unchanged.
- `check_lane_d` PASSes every such report.

### 4. Refresh while a miner holds KING_D

- **Second bar from that boundary.** The miner king keeps the throne. Every
  miner cohort judged at B has `house_digest` = the new house. The new
  house is scored on the cohort's windows, and the old house is not scored
  at all.
- **The house enters as a challenger at B** (§1.4 "It enters at that
  settlement", §2 item 2). It is judged against KING_D forward-only, under
  the same rules as an entry: eligibility for `max(T_house, T_K)`, the
  floors, and a first clear followed by the mandatory confirmation on the
  next disjoint set.
  - It is judged in **its own cohort**: `members == [house]`, `king_digest`
    = KING_D, and no `house_digest` (the house is never its own second
    bar).
  - Its verdict has `digest` = the house, `king_digest` = KING_D and no
    `house` judgement. Its rows cover the cohort's windows.
  - The same pointer at the next boundary is the **same house entry**: its
    first clear is kept, and the confirmation's `first_set_sha256` /
    `first_king_digest` point back to it.
- **A weaker house** loses its duel (`cleared: false`). It never takes
  KING_D, never sits in the court and is never a prior king.
- **T10, a stronger house.** After two real miner crowns (X, then Y, so
  KING_D = Y and `prior_kings = [X]`), the pointer swaps to H1. H1 first
  clears against Y, and Y still reigns. H1 then confirms on a set disjoint
  from its first, and **retakes**:
  - `king_after` is the house record of H1, and `king_before` is Y;
  - in state, `king` = H1 with `house: true` and `hotkey: null`;
  - `prior_kings` = `[Y, X]`: the dethroned miner king becomes the newest
    prior king;
  - `court_uids == [None, uid(Y), uid(X)]`. Slot 0 is H1 with
    `burn_reason = "house_king"`. Y and X are paid (`burn_reason` null),
    and `share` is unchanged.
- **After the retake**, a further pointer (H2) is a house-held refresh. It
  happens in place (no verdict for H2), and the tails stay
  `[None, uid(Y), uid(X)]`. Neither H0 nor H1 is ever a prior king.
- A miner entry whose digest is the swapped house's is refused at dedup
  (`skipped` has `duplicate_digest`) and closed, never judged.
- **Restart-safe.** The state file and its reports are copied after the
  first clear, and a fresh call settles the confirmation boundary there. It
  gives the same report body (everything except the randomised
  `signature`), the same pay and the same state.
- Every report from these chains PASSes `check_lane_d` and `verify_report`.

**Not pinned:**

- **The gate for the house challenger** (the plan's "Proposed design changes" item 17, "what
  replaces gate vs house when the house is the challenger"). The fixtures
  give a bench on which the challenger beats every other digest, so gated
  and ungated designs both pass. If it is gated, its anchor can only be
  KING_D.
- **Whether a newer pointer replaces a pending house entry.**
- **`report.crowned` on a retake.**

## On main

- `cascade.validator.lane_d_house` doesn't exist. Every pointer and
  fail-closed test fails on its `ModuleNotFoundError`, which is in `PATHS`.
- A miner-held refresh never judges the house, so the challenger and T10
  tests fail on an `AssertionError`.
- The house-held refresh, second-bar, court, audit, restart, dedup and
  unarmed-pair tests already pass. W5f owns them as its assertions.

## Running

```
LANED_PYTEST_CMD="nice -n 19 env PYTHONPATH=<wt> <venv>/python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'" \
  python -P scripts/laned/accept W5f --expect-fail --root <wt>
```
