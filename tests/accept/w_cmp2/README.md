# W-CMP2 acceptance tests: the multi-instance lane C runner

W-CMP2 (plan rev6.25, DEC-CA-0075) depends on W-CMP. Owner 2026-11-24: "the
goal is that after this release a NEW lane C competition (new domain, new
standard pool from a builder job, plus a spec) launches by overlay with no
validator update."

| File | Pins |
|---|---|
| `test_w_cmp2_validator.py` | an `energy` spec instance on a GENERIC standard pool (an energy builder's `load` channel read as the close) is registered, rowed by the main validator under its own report key, recorded by `reports[energy]`, judged, confirmed, slotted, paid by its competition weight and replayed PASS by `check_marketcast(kind="energy")` (a forged pass FAILs); `mkt`'s rows, reports and state are byte-identical with the instance configured beside it (signatures aside); registry lookups |
| `test_w_cmp2_trainer_config.py` | a spec compiles into an instance (kind, targets, competition, table inheriting the pinned base; `mkt` unchanged; pending before its start); bad instances (no pool, off-grid start, unknown metric / channel) are typed refusals whose share burns; reserved ids refused; the overlay adds and removes an instance and never starts one retroactively; the trainer dispatches, stamps the instance's embargo, settles the leg and records `reports[energy]`; the pod's training data is the pool clipped to the recipe range before the dispatch; the audit reads instance rows as MarketCast rows |

The instance kind is registered per test and unregistered after it (the kind
registry is process-global). Local run: `-m 'not torch'`, `--noconftest`.
