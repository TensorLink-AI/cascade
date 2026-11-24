# WB20 acceptance tests: the contract they pin

WB20 writes the lane B docs (plan §3.0 WB20 row, §11.2 "(rev4) Lane B docs",
§12b). The builder may touch only the four guides in `PATHS`:
`docs/MINER.md`, `docs/INTERFACE.md`, `docs/AGENT_MINING.md` and
`docs/VALIDATOR.md`. These are doc **contract** tests, not prose review.

**The code is the truth.** No list of keys or codes is hardcoded. `helpers.py`
derives them from main on every run (torch-free imports and AST reads):

| fact | source |
|---|---|
| lane B track keys and defaults (WB10) | `cascade/shared/config/lane_b.py` `LANE_B_TRACK_DEFAULTS` |
| every `[lane_b]` key a node reads | those keys + the reads in `v2.parse_lane_b` and `lane_b.parse_lane_b_track` |
| `open_rule` values (WB17) | `LANE_B_OPEN_RULES` |
| intake contamination refusals (WB13) | every `"code"` value in `FundingIntake._lane_b_contamination` (`cascade/funding/intake.py`) |
| declared-rows refusals (WB14) | `cascade/trainer/declared_rows.py` `DECLARED_ROWS_CODES` |
| deferrals (WB15, WB18) | `cascade/trainer/row_claims.py` `ROWS_HELD`, `SENTINEL_TARGET` |
| `gate_closed_reason` values (WB17) | `cascade/trainer/expert_lane.py` `GATE_CLOSED_REASONS` (AST) |
| funding expiry | the `last_error_class` of `FundedQueue.expire_stale` (`LaneBQueue` inherits it) |
| `contamination` receipt keys (WB13) | the dict `scan_repo` returns (`cascade/interface/contamination_scan.py`) |
| header and signed tag (WB14) | `LANE_B_DECLARED_ROWS_HEADER`, `_DECLARED_ROWS_SIGNED_TAG` in `intake.py` |
| suite targets and commit codes (WB18) | `v2.SUITE_TARGET_SUITES`, `SUITE_TARGET_PREFIX` (`suite:`), `SUITE_CODE_PREFIX` (`s-`) |
| experts per layer and the clamp (WB14) | `model_experts_per_layer()`, `layer_clamp()` under the default fractions |

When the code adds a key or a code, the docs must follow. `contamination_budget_exceeded`
(WB13 RV6) is not in the code today, so no doc may name it. Once it lands,
`_lane_b_contamination` returns it and MINER/INTERFACE must document it.

## Definitions

- **Lane B section:** a heading (outside fenced code) whose text matches
  `lane[ -]?b` (case-insensitive). It runs to the next heading of the same or
  a higher level. A doc may have several, and all of them count. Each of the
  four docs must have at least one.
- **Documented token:** the token appears as an inline code span, either the
  whole span or a whole word inside one (for example `` `deferred: rows_held` ``),
  or as a whole word on a fenced line.
- **Key with default:** one line of the section has a code span naming the
  key (`` `k` ``, `` `[lane_b] k` ``, or `` `k = v` ``) and either a span
  `` `[lane_b] k = <toml>` `` / `` `k = <toml>` `` or another span on the same
  line whose TOML value equals the parser default. Types must match, so
  `false` is not `0`. Floats compare by value, so `6` equals `6.0` and
  `27_000` equals `27000`. `contamination_min_hits` is
  `{exact = 1, normalised = 3}` and `contamination_hashset` is `""`.
  `funded_pod_skus` has no literal default. The parser starts it from the
  same file's `[round] funded_pod_skus`, so its line must carry the span
  `` `[round] funded_pod_skus` ``.
- **Paragraph:** a blank-line separated block. A table or a list counts as one.

## What each doc must document

| requirement | MINER | INTERFACE | AGENT_MINING | VALIDATOR |
|---|:-:|:-:|:-:|:-:|
| a lane B section | ✓ | ✓ | ✓ | ✓ |
| every WB10 track key with its default | ✓ | | | |
| every intake contamination refusal code | ✓ | ✓ | `benchmark_series_embedded` | |
| every `declared_rows_*` code | ✓ | ✓ | | |
| `rows_held`, `sentinel_target` | ✓ (also as `deferred: <code>`) | | ✓ | |
| `gate_closed_reason` and every value of it | ✓ | | | ✓ |
| `funding_expired` | ✓ | | | |
| `sentinel_target` explained in the paragraph that names `sentinel_max_defer_eras` | ✓ | | | |
| `X-Lane-B-Declared-Rows` with an example the WB14 parser accepts | ✓ | ✓ | ✓ | |
| the signed `rows=` suffix and `declared_rows` | | ✓ | | |
| "ordinal" and `0..N-1` on one line, plus per-layer clamp lines (`E` with `lo..hi` for each distinct layer size), plus `min_experts_frac` / `max_experts_frac` | | ✓ | ✓ | |
| `repo_too_large` in the paragraph that names `suite_target_max_repo_mb` and its default | | ✓ | | |
| `contamination` and each receipt key | | ✓ | | ✓ |
| `suite:<s>` for every suite | ✓ | ✓ | ✓ | |
| `s-<s>` for every suite | ✓ | ✓ | | |
| `open_rule` and every value of it | ✓ | | | ✓ |
| `gate_rule`, `gate_evidence` | | | | ✓ |
| `contamination_hashset` named in a paragraph that says "release-only", plus `declared_rows` and `hashset_version` | | | | ✓ |
| one paragraph with "sentinel", "guard", "target", "era" and a negation (neither/nor/not/never): the era's sentinel is neither guard nor target (W-S2, WB18) | ✓ | | | ✓ |

The split follows §11.2. MINER is the miner-facing reference: keys, every
code a miner can receive, deferrals and the sentinel. INTERFACE covers the
wire format: the embedded-benchmark ban and the `declared_rows` header.
AGENT_MINING covers the agent workflow for declaring rows. VALIDATOR covers
what validators check: the hash-set pin, declared rows and the opening rule.

**Declared-rows examples.** These are every `X-Lane-B-Declared-Rows: {json}`
in the section, with the header, colon and JSON object on one line. Each
example is run through `parse_declared_rows` under the default clamp. An
example whose line names a `declared_rows_*` code must be refused with one
of the codes it names. Any other example must be accepted. At least one
accepted example is required. The routing artefact (WB12,
`status/lane_b_routing-<king8>.json`) is not built, so nothing requires it,
and the stale rule below applies to it.

## Stale mentions (all four docs)

- Anywhere in the doc: every `[lane_b] <key>` must be a key some parser
  reads. Every `X-Lane-B-*` header must be one the code knows. Every
  `deferred: <x>` must be a deferral code. Every `open_rule = "<x>"` must be
  an open rule.
- In lane B sections: every snake_case word inside an inline code span must
  exist under `cascade/**/*.py`, as a string constant, an identifier or a
  whole word of a string constant, or as a key or string value of
  `chain.toml` / `chain.testnet.toml`. Every `suite:<x>` and every span
  `s-<x>` must name a real suite. `<suite>` placeholders are allowed.

## Links, leaks, check_docs

- Every relative link in the four docs resolves, anchors included. The slug
  algorithm is the same as `scripts/check_docs.py`'s.
- The four docs leak nothing. That means no IPv4 address (the RFC 5737
  ranges, 127/8 and 0.0.0.0 are allowed), no token-shaped secret (AWS, HF,
  GitHub, `sk-`, private key blocks, Slack/Discord webhooks, Telegram bot
  tokens) and no non-placeholder `HIPPIUS_*` / `HF_TOKEN` / `WANDB_API_KEY` /
  `LIUM_API_KEY` / `BT_WALLET_PASSWORD*` / `CASCADE_OPS_WEBHOOK_URL`
  assignment. They also have no `user@host` login, no `*.internal` /
  `.local` / `.lan` style hostname and no `ssh -p N user@host`.
- `python -P scripts/check_docs.py` exits 0 (CLI flags, links, config and
  reward references).

## Failure on main

INTERFACE.md and VALIDATOR.md have no lane B section. MINER.md §13 and
AGENT_MINING.md "Lane B" predate the rev4 features. About 44 tests fail, all
with `AssertionError`. The code-truth self-check, the stale, link and leak
tests, check_docs and the AGENT_MINING/MINER section-exists tests pass on
main. There are no import, fixture or syntax errors. The PATHS are docs
only, so no import can fail for the right reason. A scratch control
satisfied every test: minimal conforming sections appended to the four docs
passed all 60. Injected stale keys, codes, headers, IPs, a wrong default and
a refused example each failed.

## rev6.22 (`test_wb20_lane_b_slots_docs.py`)

`docs/MINER.md` states full pay until replaced (no decay, no expiry; ties
keep the holder; empty slots burn; `since_era`), every-slot testing in two
max-T families with the bars and cap read from `LaneBConfig`, take-all, and
multi-target funding with a repeated `--target` (checked against the parser).
The reward reference blocks are regenerated from `scripts/check_docs.py`
(B is untiered; the decay example is C's).
