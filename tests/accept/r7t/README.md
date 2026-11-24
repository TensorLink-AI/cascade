# R7t acceptance tests: the testnet overlay identity

R7t is the owner's S0c testnet drill (plan §3.0 R7t row: "needs testnet
`[netconfig] owner_hotkey`"). Its drill Accept is
`scripts/laned/accept R7t --evidence evidence/s0c_testnet.json`. These tests
pin the CI-checkable prerequisite only: the release `chain.testnet.toml`
names the key every testnet node verifies overlays against. Conventions are
W0c's (`tests/accept/w0c/README.md`): every `cascade.*` import is inside a
test body. Nothing touches a chain, a store or a wallet; the only keypair is
a throwaway `//Alice` dev key.

## The rule (cascade/shared/config/sections.py `NetConfigConfig`, v2.py `check_bench_source`)

`[netconfig] owner_hotkey` is the owner's MAIN validator hotkey (the key that
signs the receipts satellites replay and the validator bench report), not
the subnet-owner hotkey. When both are set it must equal `[manifest]
validator_hotkey`. Owner 2026-11-24, testnet netuid 259:

- validator0 hotkey `5F1VmMSaPPi2qm91HE8VyMiHoCz4YxGaexJ3qmbyAPBUcUtR` =
  `[manifest] validator_hotkey` -> `owner_hotkey`;
- owner hotkey `5Cyver53HZaxdn6j9K2W9LyaoGxjz1W3cXHoKcaDyzfjfpHs` (uid 0) is
  `[manifest] trainer_hotkey` and must NOT be the overlay signer.

## Tests

- the testnet file loads with `owner_hotkey` set, enabled, equal to
  `[manifest] validator_hotkey`, distinct from `trainer_hotkey`; mainnet
  `chain.toml` keeps its own main validator key;
- `netconfig.plan_overlay` on the step-1 rehearsal overlay
  (`deploy/overlays/launch-20261125.pre.testnet.toml` from #870, else the
  mirrored `fixtures/pre.testnet.toml`) at testnet era start 8524200 is VALID
  and names `bittensor:test` / netuid 259;
- `verify_overlay` against the configured key refuses that body signed by any
  other key (a dev key, or one claiming the trainer hotkey as signer).

Every test fails on main (owner key `""`), per the FEATURE rule.
