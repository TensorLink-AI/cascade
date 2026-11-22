# R2x acceptance tests: S0b live run (owner WP)

R2x is the S0b live run (plan §3.0 R2x row, §5 row 3 S0b, §14.1, §17.1 L4):
decode the live mainnet readiness notes, add the `cfg-` overlay segment
with the R2 simulator, and show that **no still-pending segment would be
trimmed**. It runs in `owner (chain read)`: the owner (or a box with a chain
endpoint) runs the collector; CI only verifies its evidence. These tests are
the spec. The builder may touch only `PATHS`: `cascade/laned/s0b_collect.py`.

R2x is external: its Accept is the evidence verifier here. The
`test_evidence_*` tests **skip** (with a clear reason) when `$LANED_EVIDENCE`
is unset, so the collector can merge and `accept R2x` goes green (L3) before
the owner's run. L4 needs a green `accept R2x --evidence
evidence/s0b_notes.json` run on main (W0c). When `$LANED_EVIDENCE` **is**
set, a missing file, invalid JSON, `source: "fixture"` or any hand-edited
field fails hard. The `test_verifier_*` and collector tests run offline from
a fixture and never reach a chain.

## The collector

```
python -P -m cascade.laned.s0b_collect --network finney --netuid 91 \
    --cfg <cfg> --out evidence/s0b_notes.json [--dry-run [--fixture SNAPSHOT]]
```

- **Reads only, through `ChainClient`** (`cascade/shared/chain.py`), all at
  one head block: `current_block()`, `block_hash(head)`,
  `validator_stakes(head)` (the typed metagraph's stake and permit, the
  numbers the activation tally uses) and `read_plain_commitments(head)` (one
  `Commitments::CommitmentOf` query map; the plain store readiness notes live
  in). `ChainClient(netuid, network)` is built **without a wallet**. No
  `bittensor` import, no raw `subtensor`/`query`, no setter, no wallet
  argument (`test_collector_uses_only_chainclient_reads_and_no_wallet`).
- **Rows:** every hotkey with `validator_permit` and `stake > 0`, largest
  stake first. `note_state` is `ok` (a v1 readiness note,
  `activation.parse_signals`), `missing` (no plain commitment) or
  `not_a_note` (some other plain commitment). `note_hex` is the commitment's
  decoded UTF-8 text re-encoded as hex.
- **Simulation:** for each `ok` note, R2's `decode_note` and
  `simulate_note(note, head, cfg, 0, 0, worst_case=True)`: the new `cfg-`
  segment unlocked, then judged at its longest, as the live startup gate
  `v2b_note_problem` does. `headroom_bytes = 128 - simulate.bytes`.
- **`--cfg`:** `cfg-<8 hex>`, `<8 hex>`, the 64-hex overlay body digest
  (`netconfig.cfg_feature_name`), or `none`. With `none`
  (`target_cfg: null`), the simulation uses the probe name `cfg-ffffffff`,
  the same one `v2b_note_problem` uses. Every `cfg-<8 hex>` name is 12 bytes,
  so the trim result and the headroom are the same for any target: `none` is
  a valid S0b run before S0c has produced the seq-1 digest.
- **`--dry-run`** opens no connection. Without `--fixture` it prints the
  fetch plan (`would_fetch`) and writes nothing. With `--fixture` (a snapshot
  in the shape of `fixtures/chain_snapshot.json`: the four reads' return
  values) it writes the evidence it would write, with `source: "fixture"`,
  which the verifier never accepts.
- **Stdout** is one JSON object (`dry_run`, `network`, `netuid`,
  `would_fetch`, `fixture`, `out`, `gate_clear`, `passed`, and the summary).
- **Exit codes:** 0 = no pending segment trimmed and at least one note
  checked (`gate_clear`); 1 = a pending segment would be trimmed or no
  permitted validator has a note; 2 = usage, input or chain error. In live
  mode `--out` is required.

## Evidence `laned_s0b_notes/1`

```json
{"schema": "laned_s0b_notes/1", "wp": "R2x", "source": "chain",
 "network": "finney", "netuid": 91, "head_block": 9612345,
 "block_hash": "0x<64 hex>", "fetched_at": "<ISO 8601 with offset>",
 "tool": {"module": "cascade.laned.s0b_collect", "git_sha": "<40 hex>", "git_dirty": false},
 "target_cfg": null,
 "simulate_params": {"cfg_feature": "cfg-ffffffff", "cfg_source": "probe",
                     "cfg_lock": 0, "cfg_activation": 0, "worst_case": true},
 "hotkeys": [{"hotkey": "5…", "stake": 480000.0, "validator_permit": true,
              "note_state": "ok|missing|not_a_note", "note_hex": "…" | null,
              "decoded": <R2 decode object> | null,
              "simulate": <R2 simulate object> | null, "headroom_bytes": 18 | null}],
 "counts": {"permitted": 4, "ok": 2, "missing": 1, "not_a_note": 1},
 "headroom": {"min_bytes": 18, "min_hotkey": "5…"},
 "trimmed_pending": [{"hotkey": "5…", "feature": "v2b-…"}],
 "passed": true}
```

## The verifier (`helpers.evidence_problems`)

It rejects the file unless all of these hold:

- the keys above; `schema`, `wp: "R2x"`, `source: "chain"`,
  `(network, netuid) == ("finney", 91)`; `head_block > 0`; `block_hash`
  `0x` + 64 hex; `fetched_at` ISO 8601 with an offset;
- `tool.git_sha` is a commit in this repository that contains
  `cascade/laned/s0b_collect.py`, and `git_dirty` is false;
- `simulate_params` is exactly the worst-case rule above, for `target_cfg`
  or the probe;
- each row is permitted with stake > 0, and its `decoded`, `simulate` and
  `headroom_bytes` equal what the **R2 decoder re-derives from the recorded
  `note_hex`** (`missing` rows carry nulls; `not_a_note` rows must not parse
  as a readiness note);
- `counts`, `headroom`, `trimmed_pending` (each `ok` row's
  `simulate.pending_trimmed`, in row order) and `passed`
  (`trimmed_pending == []` and at least one `ok` row) are recomputed and must
  match.

The S0b exit gate is `test_evidence_no_pending_segment_trimmed`:
`trimmed_pending == []` and `passed: true`. If it fails, type the decided
blocks of the named segments in main first (plan §5 row 3), release, and
re-run the collector.

Run (prod box, tests only, offline):
`nice -n 19 env PYTHONPATH= /root/cascade/.venv/bin/python -m pytest --noconftest -p no:cacheprovider -q tests/accept/r2x/`
