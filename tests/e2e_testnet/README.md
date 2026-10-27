# Live testnet end-to-end suite (v2 Lane A)

Drives the real mechanism on testnet and asserts each step:
CLI → intake → chain commitment and timelock reveal → trainer admission and
pod rental → training → settlement manifest → every validator's receipt and
weights → audit → pod teardown.

Run from the repo root with the testnet `.env` sourced and the trainer,
both validators and the intake running:

```bash
set -a; . ./.env; set +a
.venv/bin/python -m tests.e2e_testnet.run --stages preflight,admission   # no GPU rental; admission uploads a fixture
.venv/bin/python -m tests.e2e_testnet.run --spend                        # full run, rents GPUs
.venv/bin/python -m tests.e2e_testnet.run --spend --resume               # keep tracking a run
```

| Stage | Cost | Checks |
|---|---|---|
| `preflight` | free | services up, intake answers, v2 active, status feed fresh, launch preflight not blocked, contract digest = feed, image pinned and resolvable on GHCR, worker code fingerprint = source, wallets registered, provider keys present, pod baseline |
| `admission` | free | `cascade verify` accepts the 32-channel fixture and rejects univariate, packed-source, nondeterministic and network-access generators; `submit --no-commit` dry run |
| `legs` | **paid** | one funded submission per provider scenario, reveal, stage tracking to a terminal status, trained, payer-billed, provider used |
| `settlement` | free | each entry judged in a round; both validators' signed receipts bind the exact ref and agree; current chain weight values match within u16 rounding; `duel` and `round` views |
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

Preflight runs before any paid stage, including `--spend --resume`; a failed
preflight/admission blocks submissions. The harness accepts only `--network
test`. Use `--resume` when a state file contains entries. Submission intent is
persisted before the CLI call: a crash with an unresolved intent requires
chain/intake reconciliation, not automatic resubmission. State/report writes
are atomic. Admission has no GPU rental but does upload an unfunded artifact;
it is not a read-only stage.

Tracking reads current `data.submissions[]`, requires its own freshness and
matches hotkey/ref/lane. Settlement requires a signed **scored** receipt whose
manifest actually includes that exact challenger; eligibility or a rejection
is not successful training evidence. Receipt history is checked when tracking
starts late. On-chain checks compare normalized values as well as UIDs; a
newer receipt makes a historical current-weight check explicitly skipped.
Pod inventory covers Lium, Shadeform, Vast and Runpod; listing failures cannot
establish an empty account or a successful cleanup.

The carried king does not need a challenger fund request. CPU generation is
operator-funded. Mainnet v2 GPU training tries the king's deposited provider
key, then operator recovery if that key fails. A key-caused fallback retains
the existing key-lapse reward policy, including for the carried king. Ensure
the operator account can pay for it. Shipped testnet disables king_ladder and
CPU shards; the harness reports those coverage gaps instead of claiming to
qualify mainnet funding or CPU generation. Run a separately configured
rehearsal for those paths. The same applies to the production model's actual
GPU memory/throughput and prepared Targon evaluation hosts.
