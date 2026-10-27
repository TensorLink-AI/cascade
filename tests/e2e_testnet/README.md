# Live testnet end-to-end suite (v2 Lane A)

Drives the real mechanism on testnet and asserts each step:
CLI → intake → chain commitment and timelock reveal → trainer admission and
pod rental → training → settlement manifest → every validator's receipt and
weights → audit → pod teardown.

Run from the repo root with the testnet `.env` sourced and the trainer,
both validators and the intake running:

```bash
set -a; . ./.env; set +a
.venv/bin/python -m tests.e2e_testnet.run --stages preflight,admission   # free
.venv/bin/python -m tests.e2e_testnet.run --spend                        # full run, rents GPUs
.venv/bin/python -m tests.e2e_testnet.run --spend --resume               # keep tracking a run
```

| Stage | Cost | Checks |
|---|---|---|
| `preflight` | free | services up, intake answers, v2 active, status feed fresh, launch preflight not blocked, contract digest = feed, image pinned and resolvable on GHCR, worker code fingerprint = source, wallets registered, provider keys present, pod baseline |
| `admission` | free | `cascade verify` accepts the 8-channel fixture and rejects univariate, packed-source, nondeterministic and network-access generators; `submit --no-commit` dry run |
| `legs` | **paid** | one funded submission per provider scenario, reveal, stage tracking to a terminal status, trained, payer-billed, provider used |
| `settlement` | free | each entry judged in a round; both validators' receipts exist and agree; on-chain weights = receipt reward uids; `duel` and `round` views |
| `validator` | free | receipt index, latest receipt per validator and its signer, `status/chain.json`, `bench_v` reports when `bench_source = "validator"`, king ladder when armed |
| `audit` | free | `cascade-audit round <id> --tier 0` for every settled e2e round, per validator |
| `cleanup` | free | no testnet pods left beyond the start-of-run baseline |

Provider scenarios (each needs its own registered miner wallet, since a
hotkey holds one commitment). A scenario whose provider key is not set is
skipped, not failed.

| Scenario | Wallet | Providers | Flow |
|---|---|---|---|
| `lium` | miner0 | lium | `submit` (funded) |
| `shadeform` | miner1 | shadeform | `submit --no-fund` then `fund --gpu RTX4090 --allow-fallback` |
| `fallback` | miner2 | lium → shadeform | `fund --gpu L40S`; reports which provider rented (fallback only happens when Lium has no L40S) |
| `vast` | miner3 | vast | needs `VAST_API_KEY` and a registered miner3 |
| `runpod` | miner4 | runpod | uncapped `--max-price 0`; needs `RUNPOD_API_KEY` and a registered miner4 |

Output: `_e2e/report.json` (every check with PASS/FAIL/WARN/SKIP) and
`_e2e/state.json` (submitted refs, phase history, judged rounds). Exit status
is 1 if any check failed.
