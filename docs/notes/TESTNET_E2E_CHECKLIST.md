# Testnet end-to-end checklist: lane D (and lane B, sentinel) on netuid 259

Written 2026-11-22 against `main` @ `28a1e98`. Every item cites where it comes
from: a plan section, row or test id in `docs/plans/LANE_D_PLAN.md` (written
"plan §x" below), or a `file:line` on main. WP levels are copied from the
status board (`status/laned_build.json` on branch `laned/build-state`,
generated 2026-11-22T23:52Z). **They go stale.** Re-read the board before each
phase (plan §17.6). This note is a working checklist, not a source of truth:
where it disagrees with the plan, the plan wins.

> **What you need before starting**
> 1. A **separate testnet host** ("testnet-ctl", a cheap CPU VM, plan §10.3), with no mainnet key, wallet, `.env` or bucket credential on it (plan §4 "never the mainnet wallet").
> 2. **Testnet-only wallets** made by the owner on their own machine and registered on **netuid 259** (`chain.testnet.toml:12`), plus testnet-scoped Hippius/R2 credentials in the environment only.
> 3. **Owner approval for rentals** with TTLs: testnet-eval (a 24 GB pinned-torch pod), the `pool_d` builder VM, and the T3/T7 pods (plan §10.5 S3 rows, §10.3).
> 4. **The WPs below built on main.** Today no lane D row can run live: the validator never calls `settle_lane_d` outside tests (W8b, L0), and the intake (W3, L1), miner CLI (W10, L0) and `pool_d` builder (R13, W6a–W6d, L0) are not built. See Phase 7.
> 5. **Owner decisions written down:** the W3a storage path (`decisions/STORAGE-LANE-D.md`, still `status: proposed`) and the testnet `[netconfig] owner_hotkey` (`chain.testnet.toml:930`, currently `""`).

Legend: **Who** is `owner` (signs, spends, pastes remote commands, merges) or
`agent` (prepares, dry-runs, reads, writes evidence drafts). Placeholders look
like `<testnet-validator-hotkey>`. This note never contains real secrets,
wallet paths or IPs, and must never be given any.

---

## Phase 0: box prerequisites (testnet-ctl)

| ☐ | What | Command / key | Who | Evidence | Ref |
|---|---|---|---|---|---|
| ☐ | Separate host from production. It runs the testnet validator, the lane D intake and cascade-ops, at the RC commit, with testnet wallets only | — | owner | host recorded in `evidence/r14.json` (R14) | plan §10.3 "testnet-ctl" |
| ☐ | Linux with systemd. The service units are systemd units | `deploy/cascade-provisioner.service`; the W3 unit `deploy/cascade-laned-intake.service` (after W3) | owner | — | `tests/accept/w3/README.md` "Service, systemd unit and Caddy route" |
| ☐ | Python 3.11 | `.python-version` = `3.11`; `pyproject.toml:6` | owner | `python3.11 --version` output | CLAUDE.md reproducibility contract |
| ☐ | A stable `uv` ≥ 0.13.4 | `uv --version` | owner | output | `deploy/pod_bootstrap.sh:12-17` (the bootstrap refuses an older one) |
| ☐ | **No torch/GPU stack on testnet-ctl unless it does GPU work.** The pinned `torch==2.4.1+cu124` goes on **testnet-eval**, the pod | `uv.lock:2378-2380`; `pyproject.toml:27` (`train` extra) | owner | R12 evidence (Phase 4) | plan §10.2 "Launch eval host", §10.3 "testnet-eval" |
| ☐ | Disk and RAM. The plan gives **no number** for testnet-ctl. What it does pin: the intake unit `MemoryMax` ≤ 512M; T7c needs box RSS and disk deltas ≤ 64 MB during a 2.5 GB upload. Size to that and record it | `free -m`, `df -h` | owner | R14 evidence | `tests/accept/w3/README.md` (unit, T7c); plan §4 T7c |
| ☐ | **No mainnet material on the host:** no mainnet coldkey or hotkey, no mainnet `HIPPIUS_*`/`BACKUP_*` keys, no `/etc/cascade-ops.env` copied from production | — | owner | statement in R14 evidence | plan §4 "Test hotkeys are owner-registered; never the mainnet wallet"; §10.3 |

## Phase 1: wallets and chain

All key creation and every signed extrinsic is **owner, on the owner's own
machine** (CLAUDE.md hard rules). Agents never read keyfiles.

| ☐ | What | Command / key | Who | Evidence | Ref |
|---|---|---|---|---|---|
| ☐ | Testnet coldkey(s) and hotkeys. Minimum set: **main validator**, **trainer**, **≥ 1 satellite**, **test miners**. T6b needs 3 miner hotkeys under one coldkey plus a second coldkey; T18c needs a miner hotkey you are willing to deregister | owner's own `btcli` | owner | hotkey SS58s (public) listed in `evidence/r14.json` | plan §4 T6, T6b, T18c; §10.3 "Test miners: us" |
| ☐ | Register every hotkey on netuid 259 and stake the validator(s) above the permit threshold | owner's `btcli` against `--network test` | owner | uid per hotkey (chain read) | `docs/VALIDATOR.md:150-151`; `chain.testnet.toml:12` |
| ☐ | Pin the trainer signer | `[manifest] trainer_hotkey = "<testnet-trainer-hotkey>"` | owner | diff | `chain.testnet.toml:544` (today pins an existing testnet key; keep it or replace it) |
| ☐ | Pin the main validator | `[manifest] validator_hotkey = "<testnet-validator-hotkey>"` | owner | diff | `chain.testnet.toml:545` |
| ☐ | **Set the overlay signer.** Today `""`, which makes the whole overlay path inert: "nothing fetched, signalled or applied". It must equal `[manifest] validator_hotkey` | `[netconfig] owner_hotkey = "<testnet-validator-hotkey>"` (a release-level edit of the file, not an overlay) | owner (Approve-tier edit) | diff + `cascade-validator --offline` output | `chain.testnet.toml:921-930`; plan R7t, R14; `docs/OPERATOR.md` items 6–7 |
| ☐ | Satellites need `owner_hotkey` **and** `trainer_hotkey`, or they refuse to start | — | owner | satellite startup log | `docs/VALIDATOR.md` "Satellite" section (`chain.toml must carry [netconfig] owner_hotkey and [manifest] trainer_hotkey`) |
| ☐ | Build the testnet grid file for every block computation. Derive it from `[round]` and **check it against the observed chain**: `{"schema":"laned_grid/1","round":{"epoch_blocks":150,"epoch_blocks_prev":300,"epoch_activation_block":600,"era_settlements":4},"segments":[{"from_block":0,"epoch_blocks":300,"era_blocks":1200},{"from_block":600,"epoch_blocks":150,"era_blocks":600}]}` | `cascade-laned blocks next-era-start <block> --grid <grid.json> --config chain.testnet.toml` | agent | grid JSON | `chain.testnet.toml:182-195`; `tests/accept/r1/README.md` "The live grid"; plan §5 "Block arithmetic" |
| ☐ | Confirm the testnet anchors are already in the past, so nothing re-judges: `[v2] from_block = 8190000`, `[lane_b] from_block = 8190000`, `merge_guard_from_block = 8280000` | chain head read | agent | head block | `chain.testnet.toml:618, 754, 792`; plan §4 Setup |

## Phase 2: code

| ☐ | What | Command | Who | Evidence | Ref |
|---|---|---|---|---|---|
| ☐ | **Release = the RC** (main head with lane D code, **unarmed**). Record the 40-hex commit | `git -C <release-dir> rev-parse HEAD` | agent | commit in R14 / X3 evidence | plan §10.3 "RC commit"; §10.5 S3 "deploy the RC" |
| ☐ | **Stage order:** S3 begins after S2 (build) and S2b (the mainnet catch-up release, R9, L0 today). S2b is a mainnet step. On testnet it matters because the **old-code** nodes in T23/T24 should run the release mainnet runs today (the pre-lane-D code). Keep that worktree too | `git -C <repo> worktree add <old-release-dir> <old-release-ref>` | agent | both commits recorded | plan §5 rows 6–8; §17.2 S2b; §4 T23, T24 |
| ☐ | Frozen venv, never a resolver run. testnet-ctl gets no `train` extra; testnet-eval does | `uv sync --frozen` (plus the role's extras) | owner/agent | `uv.lock` sha | CLAUDE.md contract; plan §10.2 |
| ☐ | **Code-path guard:** every service starts with `python -P` (or the entry point), with `PYTHONPATH` or `CASCADE_RELEASE_PATH` naming the release dir. **Never** set `CASCADE_ALLOW_CODE_PATH_MISMATCH` | env in the unit | owner | startup log; a refused start exits 78 | `cascade/shared/codepath.py:13-14, 28-29`; `tests/accept/w3/README.md` (exit 78) |
| ☐ | Offline config load for each role | `cascade-validator --offline --chain-toml chain.testnet.toml`; `cascade-trainer --offline --chain-toml chain.testnet.toml` | agent | stdout | `docs/VALIDATOR.md:217`; `docs/OPERATOR.md` "Before go-live" (offline loads) |
| ☐ | Overlay policy lists `lane_d` on this release | `cascade-validator --netconfig-policy --chain-toml chain.testnet.toml` | agent | stdout | plan §5 row 10 (S5 gate), X5 |
| ☐ | Deploy gate before each restart, and verify after it | `python -P scripts/deploy_gate.py check --service validator --from <running-ref> --to <rc-ref> --root <testnet-root>` then `… verify --service validator --release <rc-dir> --root <testnet-root>`. Services: `trainer`, `validator`, `intake-laned`, `eval-host` | agent (read-only) | GO/REFUSE text | `scripts/deploy_gate.py:9-11, 93, 383-397`; plan R8 |
| ☐ | **Caveat (gap):** `deploy_gate` and `deploy/laned/window.py` place settlement windows on a **hard-coded 900-block grid**. Testnet settles every 150 blocks, so their `settlement` rule is wrong on testnet. Place boundaries with `cascade-laned blocks` instead, and never `--ack` without the owner | — | agent | note in X3 evidence | `scripts/deploy_gate.py:34, 147-148`; `deploy/laned/window.py:12`; `chain.testnet.toml:182` |

## Phase 3: config (`chain.testnet.toml` and overlays)

### 3.1 Keys to set on the deployed testnet file (release-level edits; Approve-tier)

| ☐ | Key | Today | Set to | Ref |
|---|---|---|---|---|
| ☐ | `[netconfig] owner_hotkey` | `""` (`:930`) | `<testnet-validator-hotkey>` | plan R7t (it cites `:903`; on main the key is at `:930`), R14 |
| ☐ | `[manifest] trainer_hotkey` / `validator_hotkey` | existing testnet keys (`:544-545`) | `<testnet-trainer-hotkey>` / `<testnet-validator-hotkey>` | `docs/VALIDATOR.md` "Configure" step 1 |
| ☐ | `[lane_d_ops] eval_host` | `""` (`:954`) | `<testnet-eval host entry>` ("set on the deployment (testnet-eval, R14)") | `chain.testnet.toml:954`; plan W2b |
| ☐ | `[lane_d_ops] vault_bucket`, `vault_prefix`, `pool_d_bucket`, `pool_d_prefix` | already testnet names (`:950-953`) | keep; confirm the buckets exist | plan §10.3; W2b |
| ☐ | `[lane_b] contamination_hashset` (only for the TB rows) | `""` (`:799`), **release-only** (an overlay setting it is refused) | the WB11 artefact's version + sha256 | plan WB10, WB11; §12b.1 |
| ☐ | `[marketcast] from_block` | `0` (`:1022`) | **leave at 0**. pool_mkt runs by manual `workflow_dispatch` (owner) until R15 builds a lane D schedule that does not arm MarketCast | plan §4 Setup (rev3, verified); R15 |
| ☐ | Storage tables | testnet buckets (`:507-538`) | keep. See the `hub_namespace` warning in Phase 7 | `chain.testnet.toml:488-538` |

After every edit, repeat the Phase 2 offline loads. Diff the deployed file
against git before reasoning from the repo copy (CLAUDE.md).

### 3.2 Overlays (signed by the testnet main validator; owner executes)

Generate the bodies with R7 (`cascade/laned/overlay.py`, L4). Each one is
validated by the real `netconfig` checks. The publish wrapper is a dry run
unless given `--execute --confirm <token>`.

| ☐ | Overlay | Command | Who | Evidence | Ref |
|---|---|---|---|---|---|
| ☐ | **S0c benign seq-1** (restates current values only) | `cascade-laned overlay benign --config chain.testnet.toml --grid <grid.json> --after <head> --out <dir>`, then `cascade-laned overlay publish --config chain.testnet.toml --overrides <dir>/overrides.toml --effective-block <era-start> --network test --wallet-name <testnet-validator-wallet> --wallet-hotkey <testnet-validator-hotkey>` (dry run), then the owner adds `--execute --confirm <token>` | agent → owner | `evidence/s0c_testnet.json` | plan §5 row 4 (S0c); R7, R7t |
| ☐ | **S0c refused overlay**: one release-only entry, refused whole | `cascade-laned overlay refused …` (same flags) → publish as above | agent → owner | same file | plan §5 row 4 ("The refused overlay applies nothing") |
| ☐ | **Lane D arming** (one overlay): `[lane_d] from_block` = a 600-block era boundary at least one era out, `share = 0.25`, `takes_from = "lane_a"`; every lane A key byte-identical to seq N-1 | `--netconfig-overlay` body via `overlay publish`; R3 lint refuses a lane D overlay touching other keys | agent → owner | overlay seq + effective block in X3 evidence | plan §4 Setup; §5 row 11; D1; R3 |
| ☐ | **Testnet-only lane D values** in the same overlay. Testnet can't meet the mainnet floors (1000 windows / 20 sources) in a day, so the plan allows **testnet-only floors** and an hourly `pool_d` with a 30 min margin | `[lane_d] min_windows`, `min_clusters`, `forward_margin_hours` (overlay-settable; every key but `remainder`) | owner picks the values | values in X3 evidence | plan §4 Setup (rev3 floors, hourly `pool_d`, margin 30 min); `chain.testnet.toml:958-962` |
| ☐ | **Weakened testnet house**: an early lane A leg, so an honest fine-tune can beat it. House changes are simulated by swapping the house pointer by overlay | testnet overlay | owner | house ref in X3 evidence | plan §4 Setup |
| ☐ | **Sentinel arming** (TS2): `[v2] sentinel_from_block` = an era boundary | overlay (`[v2]` runtime policy) | agent → owner | overlay seq | `chain.testnet.toml:682-686`; plan §16; W-S1 |
| ☐ | **Lane B suite targets** (TB rows): `[lane_b] suite_targets = true` once D41's gate (`open_rule`) is satisfied, with `contamination_hashset` already in the release (3.1) | overlay for `suite_targets`; release for the hash set | owner | overlay seq | `chain.testnet.toml:798-806`; plan §5 row 12c SB3/SB5; D37, D41 |
| ☐ | **Staged kill overlay** (T21/T27) | `cascade-laned overlay kill --config chain.testnet.toml --grid <grid.json> --after <block> --previous <seq-N body> --out <dir>` | agent (stage) → owner (publish) | latency in X3 evidence | plan §5 "Kill switch and latency"; R7 |

## Phase 4: services

| ☐ | Service | Command / unit | Who | Evidence | Ref |
|---|---|---|---|---|---|
| ☐ | **Testnet validator** (main) | `cascade-validator --chain-toml chain.testnet.toml --network test --wallet-name <testnet-validator-wallet> --wallet-hotkey <testnet-validator-hotkey>` under systemd; restarts via `deploy/laned/restart_validator.sh --dry-run …` first | owner (start); agent (dry-run) | `deploy_gate verify` output | `docs/VALIDATOR.md:228-230`; R5 |
| ☐ | **Testnet trainer** (lane A/B legs, house source) | `cascade-trainer --chain-toml chain.testnet.toml --network test --wallet-name <testnet-trainer-wallet> --wallet-hotkey <testnet-trainer-hotkey>` | owner | trainer log apply-block line | `cascade/trainer/main.py:87-96`; plan §5 row 4 ("the trainer logs the same apply block") |
| ☐ | **Satellite(s)**: one on the RC, one on the old release (T23), one docker-based in `/app` (T29) | `cascade-satellite --chain-toml chain.testnet.toml --network test --wallet-name <testnet-satellite-wallet> --wallet-hotkey <testnet-satellite-hotkey>` | owner | satellite status JSON (`--once`) | `docs/VALIDATOR.md:753-760`; plan §4 T23, T29 |
| ☐ | **Lane D intake** (after W3 merges): `deploy/cascade-laned-intake.service`, `ExecStart=<abs>/python -P -m cascade.lane_d --host 127.0.0.1 --port <N> --state-dir <abs> --chain-toml <abs>/chain.testnet.toml`, `Environment=PYTHONPATH=<abs release>`, credentials via `EnvironmentFile`; Caddy route `deploy/caddy/lane-d-intake.caddy` (`/v1/lane-d/*`) | owner | `intake-laned up:` line; `deploy_gate.py verify --service intake-laned` | plan W3; `tests/accept/w3/README.md` "Service, systemd unit and Caddy route" |
| ☐ | **cascade-ops on testnet.** Defaults are mainnet-shaped, so set: `CASCADE_ROOT=<testnet-root>`, `CASCADE_SETTLEMENT_BLOCKS=150`, `CASCADE_ERA_BLOCKS=600`, `CASCADE_ROSTER_URL=<testnet manifests roster URL>` (the default points at the **mainnet** `cascade-manifests` bucket), `CASCADE_OPS_STATE=<dir>`; then install the follow service and health timer | `cascade-ops follow --once --from-start --no-alerts`; `systemctl enable --now cascade-ops-follow.service cascade-ops-health.timer`; `cascade-ops health` | owner (install); agent (read) | health exit code | `/root/cascade-ops/cascade_ops/config.py:53-68`; cascade-ops README "Install" |
| ☐ | Alert destination for testnet: the owner fills a testnet `/etc/cascade-ops.env` (mode 600). An agent never writes or prints it. Until it's filled, alerts reach `alerts.jsonl` only | — | owner | a delivered test alert | plan §10.7 "Alert delivery"; cascade-ops README "Alerts" |
| ☐ | **Lane D monitors** (after W11): `lane_d_report`, `lane_d_pending`, `lane_d_eval_pod`, `lane_d_bench_hold`, `pool_d_stale`, `primary_store_failed`, `box_ram`, follower pattern for every lane D log line | `cascade-ops health` | agent | health output | plan §10.7; W11 (L0) |
| ☐ | **testnet-eval** (forecast host): a 24 GB pinned-torch pod, rented just in time with a TTL. The committed `launch_eval` profile pins **exactly** `NVIDIA GeForce RTX 4090`, ≥ 24000 MiB, driver ≥ 550.54.14 and 9.2 GB free on `/root`. Any other GPU fails `wrong_gpu` and needs its own committed profile | `scripts/evalhost/setup_eval_host.sh --profile scripts/evalhost/profiles/launch_eval.toml --pod <name> --host <user>@<host> --port <port> --known-hosts <file> --release-dir <rc-dir> --work-dir <dir> --dry-run`, then `--print` for the owner to paste; recovery: `scripts/evalhost/recover_eval_host.sh … --configured <host>:<port> --user <user>` | agent (dry-run/print) → owner (paste) | `evidence/r12_pod.json` (`laned_r12_pod/1`); `scripts/laned/accept R12 --evidence evidence/r12_pod.json` | plan §10.3, R12, D11; `scripts/evalhost/profiles/launch_eval.toml`; `tests/accept/r12/README.md` |
| ☐ | **Testnet bench** (gate suites): valeval-blackwell4 **only outside lane A's king-check and dethrone windows** (check `v2_check_due`), or reduced-series suites on testnet-eval. On valeval-blackwell4, use the `bench_gate` profile (exact RTX PRO 6000 Blackwell Workstation, 9.2 GB free on `/data`, 17747 bench files, suite ids pinned) | `setup_eval_host.sh --profile scripts/evalhost/profiles/bench_gate.toml … --dry-run` | agent → owner | `evidence/r12_pod.json` for that pod | plan §10.3 "Testnet bench"; `scripts/evalhost/profiles/bench_gate.toml` |
| ☐ | **`pool_d` builder VM** (R13): own disk, `TSFORGE_S3` + Hippius pin credentials only, systemd timer; **hourly** testnet builds under the testnet prefix | (R13 not built) | owner | `evidence/r13.json` | plan §10.2 "pool_d builder", §10.3, R13 |
| ☐ | **Testnet stack evidence** (R14): testnet-ctl, testnet-eval, `owner_hotkey` set, hourly `pool_d`, pod TTLs, verified teardown | `scripts/laned/accept R14 --evidence evidence/r14.json` | owner | `evidence/r14.json` (schema not yet on main: no `tests/accept/r14/`) | plan R14 |
| ☐ | Every pod gets a TTL, an on-pod check before termination, and verified teardown | — | owner | teardown line per pod in R14 / X3 evidence | plan §10.3 last bullet; §10.5 S3 |

## Phase 5: the §4 matrix on testnet

The §4 matrix has **62 rows**: 38 lane D rows (T1–T29 with suffixes; the
board's `t_rows` total is 38), 19 lane B rows (TB10–TB27, TB15b) and 5
sentinel rows (TS1–TS5). Every row must pass on testnet before the mainnet
block is announced (plan §5 go-live rule; §17.3 "S3 needs 100%").

**How a row is recorded.** WPs with a named testnet/pod evidence file have
their own Accept (given below). Every row's pass also goes into
`evidence/s3_matrix.json`, verified by `scripts/laned/accept X3 --evidence
evidence/s3_matrix.json` (plan X3). **X3's schema isn't on main yet** (there's
no `tests/accept/x3/`), so record each row now with: row id, release commit,
block range, observed outcome, log/report paths, and pass/fail. "Level" means
the owning WP's level on the board; a row can't run live until its WPs are at
L2 or above on main **and** W8b (production wiring) is built.

### 5.1 Lane D settlement path

| ☐ | Row | Pass criterion (quoted from plan §4) | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | T1 | "KING_D = house; slot burns; lane A verdicts and court unchanged" (2 eras, house only) | W8 (L5), W8b (L0), X3 | X3 |
| ☐ | T2 | Exact copy of the house: "Refused at dedup (no eval spend)" | W3 (L1) | W3 testnet evidence, case `T2` → `duplicate_weights` (`laned_w3_testnet/1`) |
| ☐ | T3 | Honest fine-tune of the weakened house on public data: "Wins forward duel and gate; crowned; published; lane D share paid". Needs one rented GPU for a few hours, **public data only** | W5e (L5), W10 (L0), W16 (L0), W13 (L0), W8b (L0) | X3 (≥ 1 real crowning) |
| ☐ | T4 | Domain-overfit model: "Loses, or blocked by the drawn suite, with the reason logged" | W5e (L5) | X3 |
| ☐ | T4b | Regresses only on low-frequency series: "Blocked by `lowfreq_regression`" | W3b5 (L4), W5g (L0) | X3 |
| ☐ | T5 | Malformed set (extra `.py`, edited `model.py`, wrong `config.json`, `.pt`/`.bin`/pickle, wrong dtype or shape, NaN, oversize): "Each refused at ingest with a specific code; nothing executed". Artefacts made by script | W4 (L3) | `scripts/laned/accept W4` + X3 |
| ☐ | T6 | Second entry per hotkey; coldkey cap; upload after `ready`: "Refused; refused; `digest_mismatch` at ingest" | W3 (L1) | W3 cases `T6_second_entry` → `hotkey_entry_open`, `T6_coldkey_cap` → `coldkey_cap_open` |
| ☐ | T6b | 3 entries from one coldkey; coldkey changed between ready and reveal; unresolvable coldkey: "2 judged, 1 skipped `coldkey_cap_settlement` and carried; `coldkey_changed` refusal; `coldkey_unresolved` alert, then closed after the bound" | W5h (L0), W3 (L1), W5g (L0) | X3 |
| ☐ | T7c | 2.5 GB BF16 direct upload (W3a path); prefetch: "box RAM and disk flat" | W3 (L1), W3a (L3), W15 (L0) | W3 case `T7c` → `ok`, `weights_bytes` ≥ 2,000,000,000, box RSS/disk delta ≤ 64 MB |
| ☐ | T8 | Forward rule, both clocks (`publish_ahead`, `late_mirror`, `before_margin`, missing `frontier`/unlisted feed, over-lag, both T orders, forecast product, backfill): "Eligible only if **both** clocks … are after wall(max(T_E, T_K)) + `forward_margin` …; the late-mirror, publish-ahead, over-lag, forecast-product and backfill cases are all ineligible" | W5a (L4), W6b (L0), W6fs (L0), W9 (L0) | X3 |
| ☐ | T9 | Three entries, different T: "Intersection set; latest-T deferred if below floors; max-T crowns the best" | W5c (L5) | X3 |
| ☐ | T9b | Clear, then house refresh before confirmation: "first clear discarded, recorded in state and report" (D20 restart) | W5e (L5), W5f (L0) | X3 |
| ☐ | T10 | House pointer swapped to a stronger checkpoint (by overlay): "House retakes; slot burns; prior miner tails keep paying" | W5f (L0), W8 (L5) | X3 (≥ 1 house retake) |
| ☐ | T13 | "Lane A verdicts and court before and after arming: Identical" | W1.5 (L4), W8 (L5) | X3 |
| ☐ | T16 | Published house resubmitted + 1-ulp perturbation: "Exact refused; perturbation loses to the house" | W3 (L1) | W3 case `T16_front_run` → `duplicate_weights` + X3 |
| ☐ | T17 | Checkpoint making a suite partial or NaN: "`checkpoint_fault`; KING_D not held" | W3b2 (L3), W3b4 (L5), W5e (L5), W5g (L0) | X3 |
| ☐ | T18 | `ready` in an era's last settlement; `until_block` set with entries pending: "Carries across eras; at `until_block` pending entries close unjudged and unpaid" | W2 (L5), W5b (L5), W7a (L5) | X3 |
| ☐ | T18b | Pending at `max_reveal_delay_days`: "`closed_unjudged: reveal_delay_exceeded`, logged, slots freed" | W5h (L0), W5g (L0) | X3 |
| ☐ | T18c | Pending entry's hotkey deregisters; a prior king deregisters: "Entry voided `deregistered`, slot freed; the prior king's slot burns; satellite replay agrees" | W5h (L0), W8 (L5), W7d (L1) | X3 |
| ☐ | T18d | Readied but not revealed within `ready_ttl_blocks`: "`ready_expired`, open slot freed, nothing judged" | W3 (L1) | W3 case `T18d` → `ready_expired` |
| ☐ | T18e | One source over `max_source_share`: "Excess windows from that source dropped deterministically (seeded by the settlement); the set re-checks its floors" | W5b (L5) | X3 |

### 5.2 Gate, bench, context and eval host

| ☐ | Row | Pass criterion | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | T7 | Wall time and memory on the launch eval host for 4k/8k/16k; 4k vs declared-128k copy on ≤ 4096 windows: "Identical scores on short windows; a variant is armed only if it fits the budget" | W5d (L3) | `scripts/laned/accept W5d --evidence evidence/t7.json` (`laned_t7/1`, pod run) |
| ☐ | T7b | "Gate: two validators derive the same suite from the block hash; regression vs KING_D or vs house blocks; the era's sentinel suite swept and published only, never gating" | W3b1 (L5), W3b2 (L3), W3b3 (L5), W3b4 (L5) | X3 |
| ☐ | T14 | Null FPR per null-pair type, confirmation-duel FPR, per-era FPR at k = 32 (`scripts/null_fpr.py`): "**S4 gate:** per-era FPR ≤ the owner target (D24, OA3) at the chosen `d_margin`" (target 5%) | S1 B1; needs OA1, OA2, OA4 (open) | X3 |
| ☐ | T19 | Eval worker on stale code: "Handshake mismatch ⇒ CRIT; nothing scored" | W4 (L3), R8 (L5), R12 (L3) | X3 (+ the R12 handshake capture) |
| ☐ | T28 | Eval host updated before and after the validator: "Handshake fails closed when mismatched; works in the documented order" (eval host → validator, plan §5 row 10) | R8 (L5), W4 (L3) | X3 |

### 5.3 Sentinel (§16)

| ☐ | Row | Pass criterion | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | TS1 | Sentinel draw on two validators; recorded value replayed when the hash is unreadable: "Identical" | W-S1 (L5) | `scripts/laned/accept W-S1` + X3 |
| ☐ | TS2 | Lane A with sentinel armed: "Three-suite verdicts; sentinel-only regression blocked iff ratchet on" (`sentinel_ratchet = true`, `:686`) | W-S1 (L5) | X3 |
| ☐ | TS3 | Lane A sentinel unarmed: "Byte-identical" | W-S1 (L5) | X3 |
| ☐ | TS4 | Lane B entry targeting the sentinel: "Deferred (D35)" | WB18 (L2), W-S2 (L2) | X3 |
| ☐ | TS5 | "Draw determinism and uniformity: As stated" | W-S1 (L5) | X3 |

### 5.4 Lane B (SB3: TB10–TB27, TS1–TS5, LB1 contamination check)

| ☐ | Row | Pass criterion | Owning WPs (level) |
|---|---|---|---|
| ☐ | TB10 | Exact GIFT/BOOM/TIME/FEV window as a literal: "Refused at intake, `benchmark_series_embedded`, before any compute" | WB13 (L2), WB11 (L3) |
| ☐ | TB11 | Same window ×3.7, +100, rounded to 3 dp, in `config.json`: "Refused (normalised match)" | WB13 (L2), WB11 (L3) |
| ☐ | TB12 | Clean generator; hash-set version change: "Passes; the receipt records the hash-set version used" | WB13 (L2) |
| ☐ | TB13 | Rows out of range / key outside ordinals 0–6: "`declared_rows_out_of_range`; `declared_rows_bad_layer`" | WB14 (L5) |
| — | TB14 | Only if D40 (b). **D40 is decided (a), router rows frozen**, so the row is not applicable unless D40 changes | WB16 (L0); plan §17.4 D40 |
| ☐ | TB15 | Counts outside 4..19 / 8..38 / 16..76 on layers with ≥ 1 row: "`declared_rows_count`" | WB14 (L5) |
| ☐ | TB15b | Rows on 2 of 7 layers, 0 on the rest: "Accepted (ESFT-consistent)" | WB14 (L5) |
| ☐ | TB16 | Overlapping rows: "The earliest reveal holds; the other shows `deferred: rows_held` with the holder and expected release" | WB15 (L5) |
| ☐ | TB17 | Tokens met + king-check pass: "Gate opens; driver state records rule + evidence" (`open_tokens` testnet = 1e8, `:774`) | WB17 (L3) |
| ☐ | TB18 | Tokens met, king check failing (or tokens short): "Gate stays closed" | WB17 (L3) |
| ☐ | TB19 | Suite-target entry on the current sentinel suite: "`deferred: sentinel_target`, not charged; judged in a later era" | WB18 (L2) |
| ☐ | TB20 | Past `sentinel_max_defer_eras`: "Judged on its target, logged" | WB18 (L2) |
| ☐ | TB21 | No declared rows: "ESFT selection excluding held rows (both modes coexist)" | WB14 (L5), WB15 (L5) |
| ☐ | TB22 | Lane B unarmed, all new keys default: "Byte-identical digests, receipts and weights" | WB13 (L2) |
| ☐ | TB23 | Lane B leg on a cheap GPU in `[lane_b] funded_pod_skus` (`:809`): "Rents that type; `gpu_name` recorded; a lane A leg requesting the same type is still refused" | WB22 (L0), WB19 (L0, OA8) |
| ☐ | TB24 | GPU type listed but no measured step rate: "Funding refused with a typed reason; no rental" | WB22 (L0) |
| ☐ | TB25 | `train_hours` 6.0 and 6.5: "6.0 admitted … guard ≤ `max_leg_seconds`; 6.5 refused at funding, nothing charged" | WB23 (L0) |
| ☐ | TB26 | 6 h leg funded late in an era: "Held to the next era's pre-train window (cross-era admission); landing boundary shown to the miner" | WB23 (L0) |
| ☐ | TB27 | Leg record over the consensus caps: "Validator refuses the entry; satellite replay agrees" | WB23 (L0) |

Each TB row's CI half is `scripts/laned/accept <WB WP>`. The live result goes
into X3 (plan §17.2 "SB3 as S3 for TB10–TB27 and TS1–TS5").

### 5.5 Recovery, restart, audit and upgrade

| ☐ | Row | Pass criterion | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | T11 | Validator restart mid-judgement: "No re-score, no double crown" | W7c (L5) | X3 |
| ☐ | T12 | Satellite replay + `cascade-audit`; one tampered row: "Identical weights; `check_lane_d` passes; tamper fails" | W7d (L1), W9 (L0) | X3 |
| ☐ | T20 | Hippius primary write fails, R2 succeeds: "Alert fires". The plan doesn't say how to induce the failure; the agent proposes a method and the owner approves it | W11 (L0) (`primary_store_failed`) | X3 |
| ☐ | T22 | Readiness note of the exact mainnet shape plus `cfg-`: "No pending segment trimmed" | R2 (L5) | `python -P -m cascade.laned.notes simulate --text <note> --head-block <head> --cfg cfg-<8hex> --worst-case` (exit 0); X3 |
| ☐ | T23 | Old-code satellite with the new main validator: "Matches until lane D arms; burns only after `from_block`; recovers after upgrade" | X3 | X3 |
| ☐ | T24 | Old-code trainer with a `[lane_d]` overlay published: "Trainer refuses that overlay" | R3 (L5), W2 (L5) | X3 |
| ☐ | T25 | Upgrade from a **copy of mainnet state**: "Clean start; identical next verdict". The owner takes the snapshot on production, read-only (it bundles state files only, never keys, `.env`, configs or logs). Then restore on testnet-ctl | `python -P -m cascade.laned.restore snapshot --source <mainnet-node-dir> --config chain.toml --out <dir> --block <block>` (owner, prod, read-only); `… verify --bundle <b>`; `… restore --bundle <b> --target <testnet-node-dir> --config chain.testnet.toml`; `… check --target … --config …`; `… verify-evidence --evidence evidence/t25.json --bundle <b>` | R10 (L3) | `scripts/laned/accept R10 --evidence evidence/t25.json` (`laned_t25/1`) |
| ☐ | T26 | Shipped chain.toml through every validator and trainer startup gate at the live head: "Starts (the #457/#460 class)" | R11 (L5) | `scripts/laned/accept R11` (CI) + the Phase 2 testnet starts |
| ☐ | T29 | Docker-based satellite (code in `/app`) following the announcement verbatim: "Upgrades and signals `cfg-`". Needs the §11.3 announcement draft | X3 | X3 |

### 5.6 Kill switch

| ☐ | Row | Pass criterion | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | T21 | Kill switch through the signed overlay: "Share returns to lane A from the stated era; no release needed" | R7 (L4), R3 (L5), W2 (L5) | `cascade-laned overlay kill …` → publish; X3 |
| ☐ | T27 | Revert overlay; stop-publishing path: "Overlay kill latency measured. Stop-publishing: the old vector (incl. the old lane D king) is re-asserted for `stale_after_blocks`, then the **whole** vector burns". After W14 is on main, add the lane D-only burn row. On testnet `stale_after_blocks = 3600` is **6 testnet eras** (`:940`), so budget about 12 h | R7 (L4), W14 (L0) | X3 |

### 5.7 Latency and the weight loop

| ☐ | Row | Pass criterion | Owning WPs (level) | Accept / evidence |
|---|---|---|---|---|
| ☐ | T15 | 2 h lane D sweep across weight intervals: "No weight gap > `weight_set_interval_blocks`" (180, `chain.testnet.toml:554`) | W19 (L0); W8b's weight-loop bound | `scripts/laned/accept W19` + X3 |

### 5.8 Other WPs that run on testnet

These WP Accepts also need a testnet run (plan §3.0 "Runs in" includes
`testnet`). The plan names no evidence file for them, so they are recorded
in X3: W3, W8b, W10, W13, W14, W19, WB17, W-EMA, R7t (`evidence/s0c_testnet.json`),
R10 (`evidence/t25.json`), R14 (`evidence/r14.json`) and R15
(`evidence/r15.json`: "≥ 4 consecutive 6 h snapshots published on testnet").

## Phase 6: S3 exit gate

Quoted, plan §4: **"Exit gate:** every row passes. … That includes T7, T7b,
T7c and T14–T20 …, plus the new T4b, T6b, T9b and T18b. At least one real
crowning (T3) and one house retake (T10) are observed end-to-end. No
unexplained ERROR for 6 testnet eras."

Plan §5 row 8: S3 = "§4 matrix including T22–T29"; exit "§4 exit gate".
Plan §17.2: "S3 | X3 at L4, with every §4 T-row passed". Plan X3: "each row's
recorded pass, 6 clean eras, ≥ 1 crowning, ≥ 1 house retake".

- [ ] Every one of the 38 lane D rows passed (board counter `t_rows` = 38/38; §17.3 "S3 needs 100%")
- [ ] Including T7, T7b, T7c, T14, T15, T16, T17, T18, T19, T20, T4b, T6b, T9b, T18b (named in the gate)
- [ ] Including T22–T29 (§5 row 8)
- [ ] ≥ 1 real crowning observed end-to-end (T3)
- [ ] ≥ 1 house retake observed end-to-end (T10)
- [ ] 6 consecutive testnet eras (6 × 600 blocks) with no unexplained ERROR in validator, trainer, intake or satellite logs
- [ ] `evidence/s3_matrix.json` written; `scripts/laned/accept X3 --evidence evidence/s3_matrix.json` exits 0 (X3 at L4)
- [ ] S0c testnet part done first: R7t at L4 (`seq-1 adopted ≥ 51%; refused body applied nothing`) (§17.2 S0c)
- [ ] Lane B (SB3, plan §5 row 12c): TB10–TB27 (TB14 n/a under D40 (a)) and TS1–TS5 green, **and** LB1's contamination check passing
- [ ] Calibrations testnet must supply before the mainnet block (`d_margin`, gate tolerances, sentinel) recorded (§5 go-live rule, "What replaces shadow evidence")
- [ ] Owner sign-off: "All §4 T-rows green on testnet before the block is announced" (§5 row 11)

## Phase 7: known blockers and must-nots

### 7.1 WPs not built yet (board 2026-11-22T23:52Z)

| Blocks | WP (level) | Ref |
|---|---|---|
| **Every live lane D row.** Nothing outside tests calls `settle_lane_d` | W8b (L0) | plan W8b row |
| Entries cannot be submitted (T2, T6, T7c, T16, T18d, T3) | W3 intake (L1), W10 miner CLI (L0), W16 reference fine-tune (L0) | plan W3, W10, W16 |
| No `pool_d` to judge on (all forward rows, T8) | R13 (L0), W6a/W6b/W6c/W6d/W6fs (L0); W1 `blocked:owner` (OA5, OA6) | plan §3.0; §17.4 OA5/OA6 |
| Testnet stack and pool_mkt cadence | R14 (L0), R15 (L0) | plan R14, R15 |
| S0c drill | R7t (L0, needs `owner_hotkey`) | plan R7t |
| Audit / replay (T12, T18c) | W9 (L0), W7d (L1) | plan W9, W7d |
| Monitors (T20, alerts) | W11 (L0) | plan W11, §10.7 |
| Kill-switch lane D-only burn (T27 extra row) | W14 (L0) | plan W14 |
| Weight loop off the lane D thread (T15) | W19 (L0) | plan W19; invariant 6 |
| King publication (T3 "published") | W13 (L0) | plan W13 |
| House refresh (T9b, T10), holds/reasons (T4b, T6b, T17, T18b) | W5f (L0), W5g (L0 on the board; its tests-only PR #664 is on main), W5h (L0) | plan W5f, W5g, W5h |
| Prefetch (T7c) | W15 (L0) | plan W15 |
| Lane B GPU list / leg length (TB23–TB27) | WB22, WB23 (L0), WB19 (L0, `blocked:owner` OA8) | plan WB19, WB22, WB23 |
| T14 null FPR | OA1, OA2, OA4 open | plan §17.4 |
| Evidence schemas not on main for R7t, R14, R15, X3 (no `tests/accept/<wp>/`) | — | `ls tests/accept/` on main |

### 7.2 Owner steps

- [ ] **W3a storage choice.** Run the live probe: `python -P -m cascade.laned.storage_probe --bucket <throwaway lock-enabled bucket> --prefix <throwaway prefix>/ --other-bucket <bucket outside the sub-token> --out evidence/w3a_probe.json --confirm-live`, then `… verify --evidence evidence/w3a_probe.json` the same day. Fill `decisions/STORAGE-LANE-D.md` "Chosen path" and "Evidence sha256" (`cascade/laned/storage_probe.py:16-26`; plan W3a). W3, W6d, W10, W13 and W18 wait on it.
- [ ] **Testnet `[netconfig] owner_hotkey`** (`chain.testnet.toml:930`), equal to `[manifest] validator_hotkey` (`docs/OPERATOR.md` item 6).
- [ ] Rentals with cost caps and TTLs: testnet-ctl, testnet-eval, the builder, the T3 fine-tune pod and the T7 pod (plan §10.5 S3; §10.6 "per testnet day ~$10–30 plus a few hours of 80 GB GPU").
- [ ] Every signed overlay publish (`--execute --confirm <token>`) and every remote paste (R12 `--print` blocks) (plan §3 "Owner steps"; §10.4 owner-paste scripts).
- [ ] pool_mkt runs by `workflow_dispatch` until R15 (plan §4 Setup).
- [ ] T25 mainnet-state snapshot on production, read-only (R10).
- [ ] Testnet alert webhook (plan §10.7).

### 7.3 Things that must not happen on testnet

- [ ] **No mainnet wallet or key** on any testnet host or pod (plan §4; §10.3).
- [ ] **Never omit `--network test`.** `cascade-validator`, `cascade-trainer`, `cascade-satellite` and `cascade-audit` all default to `finney` (`cascade/validator/main.py:21`, `cascade/trainer/main.py:94`, `cascade/validator/satellite.py:777`, `cascade/audit/main.py:155`).
- [ ] **Never point at mainnet buckets.** Mainnet is `cascade-manifests`, `cascade-logs`, `cascade-lane-d-vault`, `cascade-lane-d-pool` (`chain.testnet.toml:507-508, 950-952` comments). The cascade-ops roster URL defaults to the mainnet bucket (`cascade_ops/config.py:67-68`), so override it.
- [ ] **Shared Harbor namespace.** `[storage] hub_namespace = "cascade"` is the same project as mainnet (`chain.testnet.toml:501-504`: "no separate cascade-testnet Harbor project exists yet"). Testnet pushes land beside mainnet artefacts. Ask the owner whether to create `cascade-testnet` before T3/T13 pushes.
- [ ] **No lane A interference.** Use valeval-blackwell4 only outside lane A king-check and dethrone windows (plan §10.3). No real 8×H100 lane A dethrone on testnet (plan §4). The T3 fine-tune uses **public data only** (plan §10.3).
- [ ] **No mainnet state written back.** A T25 restore goes only into a testnet node dir (`cascade/laned/restore.py` `restore` help), and validator state is never restored across a settlement (plan §10.8).
- [ ] **No shadow periods.** Testnet uses real duels and real payouts, never `share = 0` "would-be" eras (plan §5 go-live rule, D45).
- [ ] **No edits to pinned deps or contract fields** to make a testnet row pass (CLAUDE.md reproducibility contract). `[lane_b] contamination_hashset` changes only by release (plan WB10).
- [ ] **No lane A keys** in a lane D overlay (plan §5 row 11; R3).
