# W-E2E acceptance tests: one full cycle per lane before a lane takes money

Owner 2026-11-28, retrospective item P1 (failure class C1). Lanes B, C and D
went live at block 9586800 before any of them had run submit → train → judge
→ verdict once, and every paid leg found the next broken stage (one 3 h
settlement each). These tests run that whole path on every PR.

## Two tiers

| Tier | Where | What |
|---|---|---|
| (a) CI | `tests/accept/w_e2e/` + drivers in `tests/e2e_lanes/`; CI job `e2e-lanes` (torch-free install, 10 min budget, part of the required `test` check) | the real intake, trainer, validator and miner CLI code through a whole cycle; fakes only for pods, registry, chain and S3 |
| (b) staging | `tests/e2e_testnet/lanes.py` + `tests/e2e_testnet/LANES_RUNBOOK.md` | one lane on testnet with ONE cheap real Lium GPU (<= $1/h); dry-run by default (CI runs the dry run); `--spend` is manual / nightly |

## The cycles

| Lane | Stages (`harness.Cycle.assert_full`) |
|---|---|
| B | submit → intake (HTTP) → funded queue → dispatch (gate, routing probe on the eval pod, ESFT selection) → generation (Shadeform stand-in) → train (Lium stand-in, real `_run_funded_leg`) → harvest + Hub upload → settlement (bench, prepare-ahead, rebase / merge check / judge on the eval pod, publish) → validator (own fetch + scoring, rows byte-identical) → `cascade lane-b status` |
| C | submit (recipe check, Hub upload, commit) → intake (HTTP; GPU fit #989, sample geometry #990) → queue → dispatch → train (real `train_mkt_leg` + fault classifier) → harvest (Hub) → signed lane C manifest → judge (validator publish path, judge on a forked MarketCast pod) → signed report → recorded in the next manifest → verify (peer validator from the recorded bytes; independent re-judge byte for byte) → `cascade lane-c status` |
| D | `lane-d verify` / `submit` (real intake over HTTP, filesystem vault) → `ready --attest` (fake chain reveal) → pool_d build with per-build release sync → snapshot publish + pin sync (box + eval host) → settlement (production `load_inputs_services`, eval host's real `lane_d_forecasts settle` in a forked child) → signed report, receipts, `status/lane_d.json` → second validator byte-identical + `check_lane_d` → `cascade lane-d status` |

Box-side stages run inside `harness.box_guard()` (a call-time import of
torch, timestral, safetensors, transformers or triton raises
`BoxImportRefused`, naming file:line) and `harness.box_budget()` (traced heap
peak per stage; fixtures sized so a pod-side load done in-process goes far
over).

## Regression fixtures (2026-11-25..28)

Fixed on main ⇒ the test runs the cycle as shipped (passes) and under a
revert patch of the fix (must fail by assertion):

| Bug | Lane(s) | Caught as |
|---|---|---|
| pod reply with download chatter on stdout (#983) | B, D | the first pod / eval-host reply fails to parse |
| chunked-v2 checkpoint (#998) | B, C | validator `checkpoint_tampered` / fine-tune unloadable |
| box-side timestral import in a settlement path (#1003) | C | `BoxImportRefused`; report `infra_deferred` (retro C4: still quiet on main) |
| box-side model work (#952) | B | `BoxImportRefused` |
| in-process covariate / pool load over the box budget (#1005, #1000) | C, D | `BoxBudgetExceeded`, pinned-file tripwire |
| requeued leg re-attach with no pod (#995) | B, C | generate-then-refuse, work done twice, no verdict |
| GPU OOM classified as infra (#989) | B, C | re-rented instead of ending with `gpu_oom` |
| NaN loss blamed on the miner (#991) | C (B: never the miner's) | the entry ends as a miner fault |
| frozen pool_d mirror / bootstrap-only builder sync (#976, #994) | D | stale data / stale code refused |

Still present on main when the harness first ran, fixed in this PR:

- `cascade/trainer/expert_lane.py`: the trainer box imported torch and
  `moe_model` at every lane B dispatch for the ESFT selection; now numpy, the
  same steps (torch equivalence test runs in CI, where torch is installed).
- `cascade/trainer/lane_b_driver.py`: a settled funded lane B entry never
  stored its verdict (looked up by the rebased pointer); the row now carries
  `trained_leg_pointer` (display telemetry, never signed).

## Running locally (prod box rules)

Never torch on the 7 GB box (the guard enforces it). One file at a time,
niced, with the lock-matching venv (bittensor 11.3):

```bash
nice -n 19 env PYTHONPATH=$PWD /root/cascade-v2/.venv/bin/python -P -m pytest -q \
    tests/accept/w_e2e/test_w_e2e_lane_b_cycle.py -p no:cacheprovider
```

No chain, storage, pod, torch or real key. Adding a lane stage or a new
failure class: add the stage to the lane's driver and a revert-patch fixture
here in the same PR as the fix.
