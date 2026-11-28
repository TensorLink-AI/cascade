# Staging cycle for lanes B, C and D (W-E2E tier b)

Retrospective 2026-11-28 item P1b. The lanes went live before any of them had
run the whole path once, so mainnet became their integration test. This
runbook is the gate: one lane cycle on **testnet** with **one cheap Lium GPU**,
run before

- a lane opens or re-opens,
- an image repin,
- any release that touches a lane's trainer, validator, intake or worker path,

and nightly while a lane is changing. The CI tier
(`tests/accept/w_e2e/`, every PR, fakes, torch-free) proves the code path. This
tier proves the deployed release, the pinned images and the real hosts.

Runner: `tests/e2e_testnet/lanes.py`. Report: `_e2e/lanes_<lane>_report.json`
(cite it in the release checklist).

## 0. Dry run (free, any time, any box)

```bash
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane b
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane c
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane d
```

A dry run reads git, `chain.testnet.toml` and `deploy/provision.testnet.toml`,
checks the guards, and parses every command of the cycle with the real
`cascade` CLI parser. It prints the plan and writes the report. It rents,
signs, uploads and commits nothing. CI runs the same dry run
(`test_w_e2e_staging_dry_run.py`).

## 1. Before a real run (owner or operator)

| Check | How |
|---|---|
| Testnet services run the release under test | trainer, both validators and the intake on testnet are on the same release sha; pass it as `--expect-release <sha>` |
| The lane is armed on testnet | `[lane_b|lane_c|lane_d] from_block > 0` in the deployed testnet config. Lane D ships `from_block = 0`; arming it is a signed testnet overlay (Approve tier), never this runner. |
| Lium balance | Check the staging Lium account's balance first (memory: an experiment drained Lium and Lium stopped every pod, validator eval pods included). Budget about $2-5 per lane run. |
| Nothing on mainnet is in a settlement window | the staging run shares no host with mainnet, but it shares the operator's attention |
| Keys | `LIUM_API_KEY` for the **staging miner** account in the environment. The runner checks presence only and passes the env NAME (`--provider-key-env lium=LIUM_API_KEY`), never the value. Miner wallets are testnet wallets (`miner0` by default). |
| No box load | do not run this from the 7 GB mainnet box; use the testnet operator host or an admin machine |

## 2. Real run (paid: one GPU, <= $1/h)

```bash
set -a; . ./.env.testnet; set +a
# lane B: a generator tree, one domain
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane b --spend \
    --intake https://<testnet-intake> --hotkey <miner ss58> \
    --repo-dir tests/e2e_testnet/generators/mv32 --target energy \
    --gpu RTX4090 --max-price 1.0 --expect-release <sha>
# lane C: a MarketCast recipe tree (marketcast.json), one mkt target
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane c --spend \
    --intake https://<testnet-intake> --hotkey <miner ss58> \
    --repo-dir <recipe tree> --target mkt:market --hub-repo <ns/repo> \
    --gpu RTX4090 --max-price 1.0 --expect-release <sha>
# lane D: an entry dir from `cascade lane-d export` (no GPU rented by the runner)
.venv/bin/python -P -m tests.e2e_testnet.lanes --lane d --spend \
    --lane-d-intake https://<testnet-lane-d-intake> --hotkey <miner ss58> \
    --entry-dir <entry dir> --context 4096 --expect-release <sha>
```

The runner refuses: a network other than `test`, a config whose netuid is not
259, a GPU outside the cheap single-GPU list (RTX4090, L40S, RTX3090, A10,
RTX A6000), a price cap of 0 (uncapped) or above $1/h. A failed preflight
blocks every paid stage. Lium only: the GPU tier never uses Shadeform.

## 3. What each lane checks

| Stage | Lane B | Lane C | Lane D |
|---|---|---|---|
| preflight | guards, lane armed, release sha, image pin, key present, every CLI command parses | same | same (no GPU) |
| submit | `lane-b verify-repo`, `deploy --expert-domain --reveal-now`, `lane-b fund --train-hours 0.5` | `lane-c check`, `upload`, `commit --reveal-now`, `fund` | `lane-d verify`, `submit`, `ready --attest` |
| track | queue row to a terminal status (each phase change is logged) | same | same |
| settle | the verdict on the status row: rebase, bench, judge and merge on the eval pod | the MarketCast report from the mkt pod | the pool_d build / pin and the eval-host settlement |
| verify | every testnet validator's acceptance of the expert manifest (cascade-ops status) | `cascade-audit marketcast` | `cascade-audit latest --tier 0` (the receipt's lane D record) |
| status | `cascade lane-b status --hotkey` shows the verdict | `lane-c status` | `lane-d status` |
| cleanup | no pod left on the staging key (`lium ps`) | same | n/a |

## 4. After the run

- `_e2e/lanes_<lane>_report.json` `ok: true` is the gate. Attach it to the
  release checklist or the PR that needed it.
- A FAIL is an incident: OPSLOG entry (symptom, stage, cause), and a CI-tier
  regression fixture in `tests/accept/w_e2e/` in the same PR as the fix, so
  the CI tier catches that class from then on.
- Terminate anything left over at once.
