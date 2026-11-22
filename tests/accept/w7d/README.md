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
documents it), the `paired_gate` LCB rule (W3b2), the D20 rule, the court
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
| the crown is the highest-LCB passing confirmation, or a re-gated held candidate that passes | `crowned` |
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
| C4 D20: a verdict at stage `first` is judged against `king_before`. A stage `confirm` verdict needs an earlier first clear (`confirm_pending`) of that digest in this chain; the most recent one counts (a `restart` discards the older). Its `confirmation.first_set_sha256` / `first_king_digest` are that cohort's `set_sha256` / king. Its cohort king is `king_before` under `restart` (a king change sends the entry back to stage `first`) or the first-clear king under `original`. Its windows share none with any earlier cohort of that digest | `verdicts[j].confirmation…`, `verdicts[j].stage`, `cohorts[i]…` |
| C5 pay: `pay.share == cfg.share`. The court is `[king_after] + prior_kings[:reward_prior_kings]`, king first, by recency. Each slot is `{digest, hotkey (None for the house), uid, burn_reason}`: `uid` is `uid_of[boundary][hotkey]` when the metagraph is given, and `burn_reason` is `None` iff `uid` is an int, else `house_king` (house slot) or `deregistered` | `pay.share`, `pay.court…` |
| C6 held: a gate result whose digest has no verdict in this report is a held candidate only if an earlier report of the chain left that digest with a cleared `confirm` verdict `confirmed` (gate absent or `bench_hold`). Nothing in between may have decided it, and its entry must be `confirm_pending` in the previous report | `gate…` |

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

| chain | settlements |
|---|---|
| `gated` | b1 X first clear vs H0 (ungated call); b2 X confirms, gate `pass`, crowned; b3 judges nothing |
| `held` | b1 first clear; b2 X confirms with no gate inputs (`bench_hold`, `confirmed`, nothing crowns); b3 X re-gated without a duel, `pass`, crowned (a gate result, no verdict) |
| `house_bar` | X crowned at b2; Y judged against KING_D = X **and** the house at b3 (first clear) and b4 (confirm, crowned). X becomes a prior king, so the court pays `[Y, X]` by decay |
| `d20_restart` | b1 X first clear vs H0; b2 the house refreshes to H1 and X restarts at stage `first` vs H1; b3 confirms vs H1, crowned |
| `d20_original` | `confirm_on_king_change = "original"`: b1 first clear vs H0; b2 house H1, X confirms vs H0 (H1 a second bar), crowned |

Low frequency: on main `lowfreq` is report-only (`no_windows` /
`not_candidate`, W3b5 not built). The fixtures are hourly, so the replay
only meets `no_windows` / `not_candidate`, and no `lowfreq` class is pinned
here. `test_w7d_fixture_chains.py` checks every chain fact above, every
chain rule C1–C4 on the fixtures, and that `check_lane_d` PASSes every
honest report. It passes on main.

## Tests

| row | test |
|---|---|
| import graph (invariant 12) | `test_w7d_import_graph.py` |
| MATCH on every honest chain (files, mapping, with states and metagraphs); re-derived crown, king, outcomes, gate, pay, body digest; purity; invariant 7 (`extensions`, path grammar); `PYTHONHASHSEED` 0 / 1 / 4242 give equal results | `test_w7d_match.py` |
| doctored-but-re-signed: crown (withheld, self-consistent withheld, double, `king_after`), verdict (outcome, clear, lcb), rows (resealed, unsealed, dropped), gate outcome (against its LCBs, pass below tau, suite, held re-gate, C6 non-held result), pay (share, uid vs metagraph, burn reason, extra slot, order), params, signature, `king_before`, `prior_kings`, house bar (judgement, dropped bar), D20 (`original` first king, `restart` discarded set), receipts (body, link, genesis); downstream `prior_unverified` | `test_w7d_mismatch.py` |
| UNVERIFIABLE reasons; posterior state binding | `test_w7d_unverifiable.py` |

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
