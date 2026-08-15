---
id: DEC-CA-0017
type: decision
title: "Validator build stamp on receipts — unsigned, outside canonical_body"
status: active
date: 2026-08-15
tags: [receipts, validator, observability, dashboard]
revisit_when: "anything consensus- or slashing-grade ever needs to KNOW a validator's build (not just ask it) — then the stamp must move to a signed drop-when-default field inside canonical_body, which requires every signature-verifying reader to upgrade BEFORE any validator stamps"
relations: {refines: DEC-CA-0012}
---
Every published receipt now carries an **unsigned** top-level `"build"` key —
the short git hash of the tree the validator ran (`running_build()` in
`cascade/shared/receipt.py`; `CASCADE_BUILD` env overrides for rsync'd non-git
deployments; `"unknown"` if neither). `update_receipt_index` carries it into
`receipts/index.json`, and the dashboard's Validators panel renders each
validator's latest stamp — so "who has upgraded?" is one read of the index.

**Why it exists.** Arming DEC-CA-0012's `finalists > 1` required knowing which
of the 6 external validators ran a build ≥ `1bb14ed` (PR #173 + #196), and
there was no way to tell remotely: no validator release tags, and at k=1 the
cohort receipts are deliberately bit-identical to pre-cohort receipts
(`cohort_k` drops when 0). This closes that gap permanently for every release
after it ships.

**Why unsigned / outside `canonical_body`.** `load_receipt` reads only known
keys and `verify_receipt_signature` checks `canonical_body()` only, so an
unknown top-level key has zero compat blast radius: old validators, auditors,
and dashboards parse and verify stamped receipts unchanged — no
`receipt_version` bump (a bump is a flag-day break: `load_receipt` hard-rejects
mismatches), no two-step release. The rejected alternative — a signed
drop-when-default field inside `canonical_body` (the `cohort_k` pattern) —
would require every signature-verifying reader to update BEFORE any validator
stamps, i.e. exactly the coordination problem this exists to observe.

**Limits, stated up front.** Assertable, not attested: an operator can stamp
anything, so it answers "did you pull?", never slashing-grade proof. No
retroactive signal: an absent stamp means only "code older than this change".
Self-adopting: nothing ever requires the field; each validator stamps whenever
it next pulls.
