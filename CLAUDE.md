# CLAUDE.md — cascade

Working notes for AI-assisted sessions on this repo. Keep entries short; each
records a DECISION and its revisit condition, not general documentation.

## Design decisions

### Throughput policy: "wall is the law" (2026-07-14)
`ref_throughput_tokens_per_s` (185k) is deliberately calibrated to a WELL-FED
trainer on the reference GPU, not to the median miner pipeline (~80k measured
across 554 testnet runs). Combined with `max_train_seconds` = the budget hours,
this makes generator throughput a compute MULTIPLIER: a miner's realized
training compute is proportional to their pipeline speed (a median generator
trains ~43% of the token budget before the wall stops it; `deadline_hit` in the
run record). This is intentional — the competition prices data quality × 
generation speed, and slow generation self-penalizes; do NOT "fix" mass
deadline_hits by loosening the wall or shrinking the budget.

Revisit when: enough duel telemetry exists to judge whether throughput is
dominating data quality in throne outcomes (see the starvation telemetry in
the S3 log sink `logs/round-<id>/<role>-<size>.jsonl`, and per-round roll-up
lines once the telemetry PR lands). The alternative ("budget is the law": set
ref to the measured median so equal-compute duels are the norm and speed is
only a floor) is one config line + coordinated trainer+validator restart at a
boundary — `[training]` keys fold into contract_digest.

Pre-mainnet: the full checklist lives in docs/MAINNET_LAUNCH.md (netuid 91
decided 2026-07-14; L40S pin + rebuilt worker-image digest + container sandbox
+ pool-publish cron are the blockers; gift gate launches shadow → enforce).

## Operational invariants (hard-learned)

- `[training]` edits change `contract_digest` → the VALIDATOR must restart too,
  or it rejects every manifest (`contract_digest_mismatch`).
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- Never restart the provisioner inside its pre-boundary trigger window.
