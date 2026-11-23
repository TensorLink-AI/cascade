# W7d acceptance tests: the independent satellite replay

W7d (plan rev6 §3.0 W7d row, depends on W7c) is the satellite half of
§14.3: "Satellites verify the signature, check that the parameters stated in
the report equal `config_at(boundary)` from the adopted overlay, check
structural rules, re-run the bootstrap from signed rows, and derive
weights." It also covers §1.5 ("re-derive the lane D weights from the
receipt chain"), invariant 7 ("Replay compares structured fields only") and
invariant 12 (below). These tests are the spec. The builder may touch only
the paths in `PATHS`. The W7a, W7b, W7c and W5e tests must stay green.

Every chain is real `settle_lane_d` runs on a synthetic pool (numpy only, no
torch), published as W7c persists it. The bench pod is the one mocked
external service. The signer is a deterministic test keypair.

## Invariant 12 and the import graph

Plan §3.2 invariant 12, verbatim: **"Audit and satellite-replay code never
import the producer's eligibility, verdict or weight functions."** (W7d,
W9; import-graph check)

The invariant bans producer code, not shared pure primitives. W7d allows
exactly these (`helpers.ALLOWED_PRIMITIVES`), plus the parent packages
`cascade`, `cascade.eval`, `cascade.shared`, `cascade.shared.config` and the
replay's own package `cascade.satellite`:

| allowed module | why |
|---|---|
| `cascade.eval.bootstrap` | `cohort_maxt_lcbs`, the generic max-T statistic (the audit shares it too) |
| `cascade.eval.crps` | the bootstrap's own dependency |
| `cascade.shared.sr25519` | `verify_hotkey`, the one sr25519 verifier (DEC-CA-0051) |
| `cascade.shared.config.lane_d` | the parsed `[lane_d]` config type the replay is handed |

Besides these, the modules under `cascade/satellite/` import only the
standard library and numpy. Forbidden (`helpers.FORBIDDEN_PACKAGES`; a name
is forbidden when it equals an entry or sits below it):

| forbidden | what it holds |
|---|---|
| `cascade.validator` | `lane_d` (settlement, verdicts), `lane_d_frozen`, `lane_d_forward` (eligibility), `lane_d_bench` (`paired_gate`), `lane_d_gate` (the draw), `lane_d_report`, `lane_d_state`, `weights`, ... |
| `cascade.shared.emission` | `lane_d_vector`, `lane_parts` (weights) |
| `cascade.shared.chain` | `decayed_share_vector` (weights) |
| `cascade.shared.lane_b`, `cascade.shared.lane_c` | other lanes' weights |
| `cascade.audit` | `check_lane_d`: the other independent re-derivation. Two independent implementations cross-check each other only if neither leans on the other (and `cascade.audit.checks` imports `cascade.shared.chain`) |
| `cascade.lane_d`, `cascade.laned` | the intake service and operator tooling |
| `cascade.eval.lane_d_scorer`, `lane_d_ingest`, `lane_d_t7`, `lane_d_handshake` | eval-host scoring (satellites can't re-derive targets, §1.5) |
| `cascade.trainer`, `torch` | |

So the replay re-implements from their specs whatever it needs of these:
the canonical JSON, the gate draw (W3b1, as `cascade/audit/lane_d.py`
documents it), the `paired_gate` LCB rule (W3b2), the court
and the decayed pay split, the W7c genesis state and digests, and the
`lane-d-state/1` / `lane-d-report/1` / receipt schema names.

The check has three views, and every one must hold:

1. **Static, direct.** Every import in every `cascade/satellite/**/*.py`, at
   any depth (function-level imports included), is stdlib, numpy, or the
   allowlist above. Names are imported from the allowed modules themselves
   (`from cascade.eval.bootstrap import cohort_maxt_lcbs`), never through a
   package re-export (`from cascade.eval import ...` of a name).
2. **Static, transitive.** The closure over repo modules from those files
   (a module implies its parent packages, function-level imports followed)
   holds no forbidden module. No dynamic import (`importlib`,
   `import_module`, `__import__`, `module_from_spec`, `exec`, `eval`) appears
   under `cascade/satellite/`, since the static graph can't follow it.
3. **Runtime.** A fresh interpreter that imports the module loads no
   forbidden module and no `bittensor` (the SDK loads only for a signature
   check). One that replays a whole chain, with signatures, states and
   metagraphs, still holds no forbidden module or torch.

## Contract: `cascade/satellite/lane_d_replay.py`

```python
replay_chain(reports, receipts, cfg, *, signer_ss58: str,
             uid_of: Mapping[int, Mapping[str, int]] | None = None,
             states: Mapping[int, bytes] | None = None
             ) -> tuple[SettlementReplay, ...]
```

- `reports` is the published reports directory (W7c `report_path` layout:
  `lane_d_report_<boundary>.json` files, `<state stem>_reports/`), or a
  mapping `{boundary: report dict}`. Files are parsed strictly: bad UTF-8,
  bad JSON, a non-object, NaN/Infinity and a duplicate key at any depth make
  the settlement `report_unreadable`.
- `receipts` are the W7c receipt records, in settlement order. There is one
  result per receipt, in the same order, with `boundary` = the receipt's.
- `cfg` is the parsed `[lane_d]` config (`cascade.shared.config.lane_d`) in
  force for the chain (`config_at(boundary)`).
- `signer_ss58` is the pinned report signer. The report's own `signer` is
  never trusted.
- `uid_of` (optional) is the metagraph `{hotkey: uid}` at each boundary.
  When it's given, the court's uids are re-derived from it. Without it the
  stated uids are taken as signed and checked only for consistency.
- `states` (optional) is the published state snapshot bytes per boundary
  (§10.8). See "Posterior state".
- It is pure. It never writes a file and never mutates an input.

`SettlementReplay` is a frozen dataclass with at least these fields:

| field | value |
|---|---|
| `boundary` | int |
| `status` | `MATCH`, `MISMATCH` or `UNVERIFIABLE` (exported module constants, three distinct strings) |
| `mismatches` | tuple of field paths, no repeats. Non-empty iff `MISMATCH`. Empty otherwise |
| `reason` | one of `UNVERIFIABLE_REASONS` iff `UNVERIFIABLE`, else `None` |
| `crowned` | the re-derived crown (digest or `None`) |
| `king_after` | the re-derived KING_D digest after the settlement |
| `outcomes` | `{digest: outcome}` re-derived for every verdict of the settlement |
| `gate` | `{digest: gate outcome}` re-derived for every gate result (`{}` without a gate) |
| `pay` | tuple of `(digest, uid or None, share)` per court slot, king first. A `None` uid burns |
| `report_sha256` | sha256 of the report's canonical body (every field but `signature`) |
| `posterior_state_sha256` | the receipt's posterior, once verified (below) |

The derived fields are pinned on `MATCH`. Their values on other statuses are
the builder's choice. `UNVERIFIABLE_REASONS` contains at least:

| reason | when |
|---|---|
| `unknown_schema` | `schema` isn't `lane-d-report/1` (§14.3 item 3: refused cleanly, never a fork) |
| `report_missing` | no published report for the receipt's boundary |
| `report_unreadable` | the file doesn't parse strictly (above) |
| `signature_unverifiable` | `SignatureVerifierUnavailable` (the SDK is missing). Never a `MATCH`, never a `MISMATCH` |
| `prior_unverified` | an earlier settlement of this call wasn't `MATCH` |

Once a settlement isn't `MATCH`, every later one is `UNVERIFIABLE` with
`prior_unverified`: the chain state it builds on is untrusted.

### Field paths (invariant 7: structured comparisons only)

A mismatch is a path into the report, in W7a's form: dotted keys with `[i]`
list indices, for example `verdicts[0].lcb` or `pay.court[0].burn_reason`.
Receipt fields are `receipt.<field>` and state fields are `state.<field>`.
Each path matches `^[a-z][a-z0-9_]*(\[[0-9]+\]|\.[a-z0-9_]+)*$`. There's no
free text in anything compared. When a test expects a mismatch "at P", some
reported path equals P or lies below it (`P.x`, `P[i]`). The deepest path is
the builder's choice. `extensions` is never consensus (W7a): its content
never changes a re-derived value, and a re-signed report that only adds
`extensions` still MATCHes.

### Per-settlement re-derivation (from the signed report and `cfg`)

At least the scope of `check_lane_d` (`cascade/audit/lane_d.py` steps 1-7),
re-implemented independently:

| rule | mismatch at |
|---|---|
| the signer is `signer_ss58` and the sr25519 signature over the canonical body verifies | `signature` or `signer` |
| every `params` field equals `cfg` (§14.3 item 2; the W7a list) | `params.<key>` |
| `rows_sha256` is the canonical digest of `rows` | `rows_sha256` |
| per cohort: forward-only on both clocks against `max(T) + margin`, the floors, one finite row per (window, scored model) | `cohorts[i]…` or `rows…` |
| per verdict: max-T LCB, gain and clear re-run from the rows and `seed`. With a miner on KING_D the house is a second bar, judged the same way (`cohorts[i].house_digest`, `verdicts[j].house`), present exactly then | `verdicts[j].lcb` / `.gain` / `.cleared` / `.outcome` / `.house…`, `cohorts[i]…` |
| the §1.3b gate: `block` = boundary; `suite` / `eligible` re-drawn from `block_hash`, `gate_suites`, `gate_weights`, the sentinel; each result's outcome and `blocked_by` from its four LCBs against `-tau_public`; the candidates' outcomes from their gate results; no gate, no crown | `gate.suite`, `gate.eligible`, `gate.results[k]…`, `verdicts[j].outcome` |
| the crown is the highest-LCB passing first-verdict candidate (first clear ∧ A10 not `fail` ∧ gate `pass`; rev6.11 (b)), or a re-gated held candidate that passes. No gate, no crown | `crowned` |
| `king_after` is the crowned entry (`{digest, hotkey, house: false, reveal_ts}` from its entry) or else `king_before` | `king_after` |

A satellite can't refute a stated `bench_hold` (the bench is box-side, and a
held result has no LCBs). A hold can never crown, though, and the replay
re-derives everything that follows from it.

### Chain rules (across the receipts)

| rule | mismatch at |
|---|---|
| C1 receipts: `receipt.boundary == report.boundary`; `receipt.report_sha256` is the body digest of that report; the first `prior_state_sha256` is the W7c genesis digest of the first report's `house` (`{"schema": "lane-d-state/1", "boundary": 0, "house": {digest, reveal_ts}, "king": {digest, reveal_ts, "hotkey": null, "house": true}, "prior_kings": [], "entries": {}}`, canonical JSON); every later prior is the previous posterior | `receipt.boundary`, `receipt.report_sha256`, `receipt.prior_state_sha256` |
| C2 `king_before` is the previous `king_after`. The exception is when the house held KING_D and the report's `house` digest changed: then it's the new house (`{**house}`, §1.4 refresh). The first report's `king_before` is its house | `king_before` |
| C3 `prior_kings`: after a crown by a miner over a miner king, `[king_before] + previous prior_kings`, cut to `reward_prior_kings`. Otherwise it's unchanged (`[]` at genesis) | `prior_kings` |
| C4 **(rev6.11 (b): no D20, no confirmation duel)** every verdict is a first verdict, judged against `king_before`, and an entry is judged at most once in the chain (a loser is never re-judged; a held clear is re-gated without a verdict, C6). A report carrying a confirmation stage (a cohort or verdict `stage` other than `first`, a verdict `confirmation` block, or an entry `confirm_pending`) is refused: UNVERIFIABLE `confirmation_stage_present`, never a MATCH | `cohorts[i]…`, `verdicts[j]…` |
| C5 pay: `pay.share == cfg.share`. The court is `[king_after] + prior_kings[:reward_prior_kings]`, king first, by recency. Each slot is `{digest, hotkey (None for the house), uid, burn_reason}`: `uid` is `uid_of[boundary][hotkey]` when the metagraph is given, and `burn_reason` is `None` iff `uid` is an int, else `house_king` (house slot) or `deregistered` | `pay.share`, `pay.court…` |
| C6 held: a gate result whose digest has no verdict in this report is a held candidate only if an earlier report of the chain left that digest with a cleared first verdict `bench_hold` (gate absent or `bench_hold`). Nothing in between may have decided it, and its entry must be open (`pending`) in the previous report | `gate…` |

**Pay shares.** Slot `i` of the court is worth `share * king_decay**i /
sum_j king_decay**j`. A `None` uid burns. The shares sum to `share`. The
tests check them against an independent computation and against the
producer's `emission.lane_d_vector` (the test may import it, the replay may
not).

### Posterior state

The posterior digest is always bound: by the next receipt's prior (C1), and
by the state snapshot when `states` is given. With `states[b]`:

- `sha256(states[b]) == receipt.posterior_state_sha256`, else
  `receipt.posterior_state_sha256`;
- the bytes are the canonical JSON of what they parse to, else `state…`;
- the state's consensus projection is re-derived from the signed report and
  receipt. `state.boundary`; `state.house` = report `house` `{digest,
  reveal_ts}`; `state.king` = report `king_after` (digest, hotkey, house,
  reveal_ts); `state.prior_kings` = report `prior_kings` (same keys);
  `state.entries[d].status` / `.reason` = the report's entries;
  `state.last_report` = `{boundary, rows_sha256, report_sha256 (the body
  digest), prior_state_sha256 (the receipt's prior)}`. A difference is at
  `state.<field>…`.

Fields the report doesn't carry (`used_windows`, `frozen_set`, `confirm`,
`gate_hold`) are bound by the digest only. W7c: a rebuild from reports isn't
defined.

## Fixture chains (`helpers.CHAINS`)

X is sharp and revealed after arming. Y is sharper and revealed after b2.
H0 and H1 are bad houses. Bench: Y beats X, X beats both houses.

**rev6.11 (b): no confirmation duel; a crown is decided at the first
verdict (first clear ∧ A10 ∧ gate).** The floors are 48 windows / 4 sources
and each snapshot holds 24 fresh windows, so X waits on the floors at b1
(snapshot 1 only) and has its first, and only, verdict at b2. Y is eligible
from snapshot 2 on as fresh data; with the 48 h embargo (rev6.12 (a), every
cadence) snapshot 1 is eligible for Y as embargo windows (their targets
straddle Y's `b`), so Y's b3 set is 24 embargo + 48 fresh windows. Every
fixture feed pins `embargo_hours = 48` in the snapshot's `feeds` (W6b).

| chain | settlements |
|---|---|
| `gated` | b1 X floor wait (nothing judged, ungated call); b2 X's first verdict vs H0: clear, gate `pass`, outcome `crowned`; b3 judges nothing |
| `held` | b1 floor wait; b2 X's first verdict clears with no gate inputs (outcome `bench_hold`, X stays `pending`, nothing crowns); b3 X re-gated without a duel, `pass`, crowned (a gate result, no verdict) |
| `house_bar` | X crowned at b2; Y's first verdict at b3 against KING_D = X **and** the house: crowned. X becomes a prior king, so the court pays `[Y, X]` by decay |
| `refresh` | b1 X floor wait vs H0; b2 the house refreshes to H1 (the throne moves in place) and X's first verdict is against H1: crowned; b3 judges nothing |

The D20 chains (`d20_restart`, `d20_original`) are gone with D20.
`confirm_on_king_change` is a parsed legacy key with no effect and leaves
the report's `params` echo.

**Not corrected here (flag):** the house-challenger chains below are built
with the W5f helpers (`tests/accept/w5f/helpers.py`), whose chains still
crown over two settlements (first clear, then confirmation). They follow
whatever the W5f tests-correction makes them; this batch can't touch
`tests/accept/w5f/`.

Low frequency: on main `lowfreq` is report-only (`no_windows` /
`not_candidate`, W3b5 not built). The fixtures are hourly, so the replay
only meets `no_windows` / `not_candidate`, and no `lowfreq` class is pinned
here. `test_w7d_fixture_chains.py` checks every chain fact above, every
chain rule C1–C4 on the fixtures, and that `check_lane_d` PASSes every
honest report. It passes on main.

## House-challenger chains (W5f follow-up, `house_helpers.HOUSE_CHAINS`)

W5f (§1.4 "The house", §2 items 1-2, §4 T10) makes a new house a challenger
while a miner holds KING_D. The replay re-derives it from the chain, never
from the report's say-so:

- a new house digest while a miner holds KING_D, and is neither KING_D nor
  a prior king, opens a house challenger. A new pointer ends an earlier one.
  While the house holds KING_D a new pointer moves the throne in place (C2,
  no challenger);
- the challenger is judged alone in its own cohort: `members == [house]`,
  `king_digest` = KING_D (as for an entry), no `house_digest` and no
  `verdicts[j].house` (the house is never its own bar), its T the report's
  `house.reveal_ts`. Its first verdict, hold and used windows are its own:
  never those of a miner entry with the same digest;
- a confirmed, passing house **retakes** KING_D: `king_after` is the house
  record, `crowned` is null, its verdict's outcome is `crowned`, KING_D's
  miner king becomes the newest prior king (C3), and the court's first slot
  burns (`house_king`). A lost house is never judged again;
- an entry is `closed` / `duplicate_digest` exactly when it's a copy: a new
  entry naming the house or KING_D (refused at registration), an open entry
  naming a new house, or one already closed so;
- with `states`, `state.house_entry` is present exactly while there's a
  challenger, with the house's digest and T.

Every chain is real `settle_lane_d` runs built with the W5f helpers
(`tests/accept/w5f/helpers.py`), signed by the W5f test key:

| chain | settlements |
|---|---|
| `house_loses` | X crowned over H0 (b1, b2); b3 a weak H1w is judged alone against X and loses, Y first-clears against X with H1w as the second bar; b4 Y confirms, crowned |
| `house_retakes` | X then Y crowned (b1..b4); b5 a sharp H1 first-clears against Y; b6 confirms and retakes (prior kings [Y, X], slot burns); b7/b8 Z (sharper) first-clears and confirms against the house, crowned |
| `copy_at_swap` | X crowned; b3 a miner submits H1's digest at the swap (`duplicate_digest` at registration), H1 first-clears; b4 H1 retakes (prior kings [X]) |
| `open_copy` | X crowned; b3 an entry with H1's digest first-clears while H0 is the house; b4 the pointer names H1: the entry closes `duplicate_digest` and the house starts at stage `first`; b5 H1 retakes |

Forgeries, each re-signed and its receipt re-issued, are MISMATCH at the
path: a house cohort with an extra member (`cohorts[i].members`), a retake
with `crowned` set (`crowned`), a retake without the dethroned miner king in
the prior kings, court adjusted to match (`prior_kings`), and a
`duplicate_digest` close of an entry that isn't a copy (`entries[i]`).

## Armed contexts (rev6.6, owner 2026-11-23): `test_w7d_armed.py`

Plan rev6.6 §1.1: the main validator records the armed set and its evidence
in the signed report, and satellites **verify** it rather than re-deriving
it from `cfg.contexts`. `params.contexts` stays the declarable set
(`expected_params` keeps copying `cfg.contexts`). The replay re-implements the
rule itself, with no import of `cascade.eval.lane_d_scorer` (invariant 12).

- A report with neither `armed_contexts` nor `armed_evidence` has nothing to
  verify (the legacy, pre-rev6.6 form).
- Otherwise both must be present, or the absent one is a mismatch at
  `armed_contexts` or `armed_evidence`. They must also be consistent with the
  report and with `cfg`:
  - `armed_evidence.min_long_series == cfg.min_long_series`;
  - `armed_evidence.min_sources == 20`;
  - `readiness_sha256` and `t7_sha256` are 64 lower-case hex or `null`;
  - `variants` is exactly `cfg.contexts`, ascending, each
    `{context, series, sources, t7_fits, armed}` (non-negative ints, bools);
  - each `armed == (series >= min_long_series and sources >= min_sources and
    t7_fits)`;
  - `armed_contexts` is the armed variants, ascending and non-empty (an
    empty set can't score anything).

  A violation is a MISMATCH at `armed_contexts` or below `armed_evidence`
  (for example `armed_evidence.variants[2].armed`,
  `armed_evidence.min_long_series`).
- A strict subset of `cfg.contexts` whose evidence supports it MATCHes.
- The tests splice the fields into an honest chain's settlement 2, then
  re-sign it with the validator's key and re-issue its receipt
  (`helpers.doctored`). On main the replay ignores the fields: the consistent
  cases MATCH (pass) and every lie fails on its MISMATCH assertion.

## rev6.11 / rev6.12: crown at the first verdict, embargo, fresh share, draw, clamp

Plan rev6.9 (W5i), rev6.10 (a)/(b), rev6.11 (a)/(b), rev6.12 (a)/(b) and the
W7d row. `test_w7d_rev611.py` and `test_w7d_windows.py`; the field names are
the ones the W5j / W5i / W7a tests pin.

**No confirmation stage.** A report with a cohort or verdict `stage` other
than `first`, a verdict `confirmation` block, or an entry
`confirm_pending` is UNVERIFIABLE `confirmation_stage_present` (in
`UNVERIFIABLE_REASONS`); every later settlement is `prior_unverified`.
Crowning is re-derived as first clear ∧ A10 (`lowfreq.result` not `fail`) ∧
gate `pass`: a crown whose A10 result is `fail`, or with no gate, is a
MISMATCH at `crowned` (or under the verdict / `gate`).

**Per-window record (signed, W7a).** Each judged window carries
`embargo_hours` (the snapshot's pinned `e`, ≤ `params.embargo_hours`),
`freq` (the pool `freq`), `kind` (`forward` | `backtest`), `timeframe`
(= `freq`), `bin` (`forward` for the h = 64 cell; `short` / `medium` /
`long` for the rev6.9 slow bins), `h`, `origin` (the target start, =
`min_target_ts`) and `embargo_window` (bool). `params` echoes
`embargo_hours` (48), `min_fresh_window_share` (0.5) and
`max_windows_per_settlement` (`cfg`), and no `embargo_window_share_cap`
(rev6.12 (b) replaced it) and no `confirm_on_king_change`.

**Eligibility and class (rev6.12 (a): every cadence).** With `b =
cohorts[i].t_max + 3600 · forward_margin_hours` and `e` the window's
`embargo_hours`: `min_first_seen_ts + 3600 e > b` and `min_target_ts > b −
3600 e`, else a MISMATCH. A window is fresh iff `min_target_ts > b` and
`min_first_seen_ts > b` (the e = 0 rule); otherwise it is an embargo window
(a target straddling `b` included), and `embargo_window` must say so. There
is no `freq` term: the embargo applies to every cadence.

**Fresh share (rev6.12 (b)).** On every judged set, F / (F + E) ≥
`min_fresh_window_share`. A set below it is refused: a MISMATCH, or
UNVERIFIABLE `fresh_share_short`.

**The draw (W5j).** The draw record (a top-level `draw`, or one per cohort
under `cohorts[i].draw`) holds `boundary_block_hash`, `seed` =
`f"{boundary_block_hash}:lane_d_windows"`, `max_windows_per_settlement`,
`quota` and `capacity` as `{"<timeframe>|<bin>": int}` and `embargo_windows`
(the count). The replay re-derives the picks from the block hash and the
signed candidate list (fresh-first per cell, scarce cells whole, surplus
re-split, `max_source_share` after with no refill, embargo windows above
`F · s / (1 − s)` dropped last-drawn first) and the total ≤
`max_windows_per_settlement`. A changed seed, block hash or quota, or a
drawn window dropped from the set, is a MISMATCH (at `draw…` /
`cohorts[i]…`). The draw chain (`test_w7d_windows.py`: `min_windows` 24,
`max_windows_per_settlement` 32, X judged once at b2 over 48 candidates)
gives the same picks in two independent runs, ≤ 32 windows, no source above
`max_source_share`, and MATCHes.

**The clamp (rev6.9 W5i).** `combined_log_ratio(r_fwd, r_bt, *, n_fwd,
n_bt, cap, floor) -> float`, a module-level function of
`cascade.satellite.lane_d_replay` (re-implemented there, invariant 12):
`c = min(cap, n_bt / (n_bt + n_fwd))` (0 when `n_bt = 0`), and `log R =
(1 − c) · log r_fwd + c · log clamp(r_bt, floor, 1 / floor)`. Off (cap 0 or
no backtest window) it is exactly `log r_fwd`. Perfect memorisation with
the forward ratio equal gains exactly `1 − floor^cap` (< `d_margin` 0.01 at
cap 0.05, floor 0.85). The replay applies it to every verdict whose report
carries backtest windows (the fixtures have none; W5i's tests build them).

Not pinned here: e = 0 forced on sub-hourly series (reversed by rev6.12
(a)); the RNG consumption order of the draw beyond determinism and the
properties above (the W5j tests pin it; the honest chains MATCH only if the
replay agrees with the producer).

### rev6.11 tests-correction (failing on main at `315d240`)

`helpers.py` changed (48-window floors, the `refresh` chain replacing the
D20 chains, `embargo_hours` in every fixture feed, the new UNVERIFIABLE
reason), so every file runs in the tests-correction classification. On main
the producer first-clears X at b2 (`confirm_pending`) and never confirms
(snapshot 3 alone is below the floors), and the reports carry no embargo or
draw fields. 61 tests fail on main, each on an `AssertionError` (or on
`AttributeError: module 'cascade.satellite.lane_d_replay' has no attribute
'combined_log_ratio'`, in PATHS); 96 pass. The fixture-fact guards
(`_has_gate`, `_has_house_bar`, the two-slot court) turn a missing fixture
fact into an assertion.

## Tests

| row | test |
|---|---|
| import graph (invariant 12) | `test_w7d_import_graph.py` |
| MATCH on every honest chain (files, mapping, with states and metagraphs); re-derived crown, king, outcomes, gate, pay, body digest; purity; invariant 7 (`extensions`, path grammar); `PYTHONHASHSEED` 0 / 1 / 4242 give equal results | `test_w7d_match.py` |
| doctored-but-re-signed: crown (withheld, self-consistent withheld, double, `king_after`), verdict (outcome, clear, lcb), rows (resealed, unsealed, dropped), gate outcome (against its LCBs, pass below tau, suite, held re-gate, C6 non-held result), pay (share, uid vs metagraph, burn reason, extra slot, order), params, signature, `king_before`, `prior_kings`, house bar (judgement, dropped bar), the first-verdict king after a house refresh (rev6.11, replacing the D20 cases), receipts (body, link, genesis); downstream `prior_unverified` | `test_w7d_mismatch.py` |
| UNVERIFIABLE reasons; posterior state binding | `test_w7d_unverifiable.py` |
| W5f house challenger: honest chains MATCH (files, mapping, states and metagraphs) with the re-derived outcomes, crown, king and pay; the forgeries above; the fixture facts | `test_w7d_house.py` |
| rev6.11: no confirmation stage (`confirmation_stage_present`), crown = first clear ∧ A10 ∧ gate, the clamp | `test_w7d_rev611.py` |
| rev6.10 / rev6.12: embargo record and class, fresh share, the draw record and a capped draw | `test_w7d_windows.py` |

## Not pinned

- Whether a report that fails the W7a structure is `MISMATCH` (and at which
  path) or `UNVERIFIABLE`. The tests never send one, apart from the unknown
  schema.
- What a satellite does with a boundary at which `cfg` says lane D doesn't
  judge, receipts out of order, or a report with no receipt.
- The exact path set of a mismatch beyond the one each test names, and the
  derived fields of a non-`MATCH` result.
- A CLI. `python -P -m cascade.satellite.lane_d_replay ...` is the builder's
  choice.
- Re-scoring rows from forecasts or a revealed pool snapshot (D17, D23).
  The replay works from signed rows.

## Why each test fails on main

`cascade/satellite/` doesn't exist.

| test | failure on main |
|---|---|
| `test_w7d_match.py`, `test_w7d_mismatch.py`, `test_w7d_unverifiable.py` | `ModuleNotFoundError: No module named 'cascade.satellite'` (in PATHS). The subprocess cases fail on their return-code assert |
| `test_w7d_import_graph.py` | assert: the module file is absent, there's nothing to check, the subprocess import fails |
| `test_w7d_fixture_chains.py` | passes: it checks the fixtures |
