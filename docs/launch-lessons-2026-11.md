# Launch lessons, 2026-11 (lanes B/C/D)

Lanes B, C and D launched at mainnet era 9586800 (2026-11-25 22:27 UTC). These
are the failure classes we hit between 2026-11-24 and 2026-11-27, from the
OPSLOG and PRs #944–#969. Owner, 2026-11-27: "make a note of all the issues we
faced and feed that into CI/CD, linting etc so we don't make the same mistakes
on future upgrades."

Each row gives the symptom, the root cause, the fix PR and the guard that now
catches the class. A **test** runs in CI's unit or accept jobs. An **IRnnn**
is a lint rule in `scripts/check_incident_rules.py` (see
[INCIDENT_RULES.md](INCIDENT_RULES.md)), run by CI's `lint` job. A
**checklist** item could not be automated without GPUs or live state, so it
is listed under [Pre-upgrade checklist](#pre-upgrade-checklist).

## Failure classes

| # | Symptom | Root cause | Fix | Guard |
|---|---|---|---|---|
| 1 | The first funded lane B entry sat `queued` through two settlements. | `_point_cfg_consumers` re-pointed only the scheduler and promotion. The lane B/C drivers kept the startup config, where lanes B/C were off; the overlay that turned them on never reached them. | #951 | test `tests/unit/test_runner_cfg_consumers_follow_overlay.py`: every `runner.<attr> = Cls(cfg, …)` in `trainer/main.py` must be named by the re-point hook. Also `tests/unit/test_lane_drivers_follow_overlay.py`. |
| 2 | Lane B could never open. | The open gate needed 2e11 lineage tokens. With lane A frozen, lineage stays at 1e11. The default was never checked against the frozen state. | overlay seq 4 (`open_rule = "tokens"`) | checklist: evaluate every gate's default against the live state (frozen lanes included). |
| 3 | Lane B probe, judge, merge, rebase and bench staging loaded the 120m king on the 7 GB box. The validator judge did the same. OOM risk at the first judged leg. | Lane B code was written as if it ran where lane A's eval runs. Model work lived on box-side paths. | #952, #953, #954, #955 (trainer); #965 (validator) | test `tests/unit/test_box_side_torch_free.py`: 29 box-side modules import with torch, safetensors, transformers and triton refused. The function paths are pinned by `tests/accept/wb27/test_wb27_validator_remote.py` and `tests/unit/test_lane_b_offload.py`. |
| 4 | Lane B eval pod disk at 98%. | The king (14 GB, with a 5 GB optimizer) was pulled whole, with no disk guard. | #955: disk guard, LRU eviction, selective fetch | test `tests/unit/test_lane_b_offload.py`. cascade-ops pod-disk check. |
| 5 | Two "bad hosts" were blocked for 7 days. They were healthy. | The ProgressWatchdog judged a lane B leg on lane A's 23,841 steps. That leg had no `train_hours`, so it had a 3 h wall at an expert step rate. | #957 (wall-bounded steps) | test `tests/accept/wb27/test_wb27_watchdog.py`. |
| 6 | More lane A assumptions on lane B/C legs: MarketCast legs killed at 45 min, lane A detached timeout, lane A wall on cost and approvals, rent type not pinned. | The same class as row 5, found by audit. | #958 | test `tests/accept/wb27/test_wb27_lane_assumptions.py`. |
| 7 | No lane B make-good could ever run ("paused after 3 operator-side failures"). | `approval_on_file` checked the entry ref `energy\|<gen ref>`, but the approval was filed under the bare `gen.ref`. | #959 | test `tests/accept/wb27/test_wb27_approval_ref.py`. |
| 8 | An expert leg trained on 1 of 8 GPUs (about 8× slow). The make-good was aborted. | The pinned worker image rc15 was built 10-31 from 5e3fade. Main fixed the gang in #748 on 11-23, but no image carrying it was pinned. Claude also told the owner the images were current without checking the digest → commit mapping. | #960 (rank rule + `GangShortfall`); image rc3 from 641e9938 | script `scripts/check_image_freshness.py` (CI `lint`, warn-only): lists the worker-side files changed since each pin's build commit. cascade-ops `image_fresh`. checklist: rebuild from the deployed SHA. |
| 9 | Make-goods could rent unpinned cheaper GPU types. | Approvals carried no GPU type. | #961 | test `tests/accept/wb27/test_wb27_approval_skus.py`. |
| 10 | Bitwise qualification rc15 → 7f24e243 FAILED at world 8. The repin was halted and later forced by the owner's override. | Numerics changed somewhere on main since 5e3fade (not found). Qualification ran only after the image was built. | owner override `image_after_unqualified_override` | checklist: qualify before scheduling the switch. |
| 11 | The repin left `ops_continuation_image_digest` on rc15, so every continuation leg would have died at worker start. The WB19 rate table was keyed to rc15, so nothing was measured. | The `[v2] *_after` path covers 4 of the 5 pins. Rate tables are keyed by digest. | hand edit at the switch; WB19 re-key (open) | `check_image_freshness.py` reads `ops_continuation_image_digest`. checklist: 5 pins + rate table + wall gate flip together. |
| 12 | Lane D public half failed with "No module named 'pyarrow'". | The `launch_eval` profile synced only the `train` extra. | #963 (profiles + worker self-check) | test `tests/accept/w8c/test_w8c_host_modules.py`. IR001 (bare `uv sync --frozen`). |
| 13 | Lane D did not settle for 7 retries after seq 5 (`public_weight 0.5`), and its 0.5 share burned. | The public half was never measured at real scale: 10k random windows, serial ranged GETs, cold cache, against a 3600 s eval timeout. | seq 6 (`public_weight 0`); #967 (warm sweep, parallel reads, score cache) | tests `tests/accept/w6p/test_w6p_public_warm.py`, `w8c/test_w8c_public_parallel.py`, `w15/test_w15_public_score_cache.py`. checklist: time the real-scale path on the real host before arming. |
| 14 | Orphan `lane_d_forecasts` processes on the 4090; retries stacked three copies. | `subprocess.run(timeout=)` kills only the local ssh, not the remote process. | #969 (`timeout -k 30 <T-30>` + `flock` on the host) | **IR008**: a function that builds a `<remote_python> -m …` command must bound it on the host (`timeout`) or launch it detached. Open: `lane_d_bench_host`, `r2_transfer.relay_command` (`KNOWN VIOLATION pending owner`). |
| 15 | No absent records were published for boundaries the worker gave up on, so receipt replay skipped them forever. The court was carried forward, so nodes diverged. | `absent_settlement` was never called. `court_for` answered with a record at or before b, but the audit needs one exactly at b. | #969 | tests `tests/accept/w8b/test_w8b_exact_boundary_absent.py`, `w19/test_w19_abandon.py`. |
| 16 | The runway guard said "fine" with $194 against a real $15/h burn. All pods, the validator eval pods included, were about 10 h from a Lium stop. | It counted `operator_standing_pods_usd_per_hour` ($1.49/h, a stale list) instead of the account's real pods. | #966 (listing-based burn) | test `tests/accept/wb27/test_wb27_money_guard.py`. checklist: money caps vs `lium ps`. |
| 17 | Wrong cost projection and ceilings for operator legs. | One rate × one wall applied to every GPU type (per-type pricing mixup). The lane C payer check used the lane A $15/h knob. | #966 (per-type rate × per-type wall); #964 (MarketCast at its own cap) | tests `tests/accept/wb27/test_wb27_money_guard.py`, `test_wb27_lane_c_launch.py`. |
| 18 | Every mkt eval pod failed its health gate (replace → terminate → retry every 600 s). Lane C never got a verdict. | The gate used the fleet image and fingerprint pins for pods that boot `[marketcast] pod_image`. | #964 (`stage_gate_pins`) | test `tests/unit/test_stage_pods_gate_on_their_own_pins.py`: every config section with its own `pod_image` needs a gate stage that returns that digest. |
| 19 | The mkt provisioner reaped a pod it never rented. `restart_provisioner.sh` could stop `cascade-mkt-provisioner`. | The orphan scope was any provisioner-shaped pod. `collect_pids` matched the mkt provisioner's cmdline. | #943 (write-ahead rent intents); #968 item 7 | tests `tests/unit/test_provision_loop.py`, `tests/accept/w_cmp/test_w_cmp_lane_c_eval_offload.py` (live decoy processes). IR003 (literal `pgrep`/`pkill -f`). |
| 20 | Every mkt round rented a pod even with no `pool_mkt` snapshot. | Prechecks ran after the lease. | #968 items 1–4 | test `tests/accept/w_cmp/test_w_cmp_lane_c_eval_offload.py`. |
| 21 | A rent timed out at 600 s on its first pull of a new (uncached) image. | `funded_ready_timeout_seconds` was sized for cached images. | owner: 600 → 1800 s | checklist: first-pull ready timeout. |
| 22 | A lane B expert leg ran about 5× slower per step than lane A, with GPUs about 20% busy. | Expert legs skip `accelerate()`, so the reference `MoEFFN.forward` runs a per-expert Python loop with `.item()` syncs. | open (lane B fast path + parity test → new image + requalification) | checklist: the fast kernels on every role. |
| 23 | pool_d builder OOM three times, then EntityTooLarge on the state push. | The first-seen record/index held all 7,002 feeds, and the 20 GB push was a single PUT. The real data size was never tried before launch. | #944, #948, #949 | tests `tests/unit/test_first_seen_*_memory.py`, `tests/unit/test_pool_d_state_streaming.py`. checklist: real-scale data. |
| 24 | PATHS merge conflicts on every update-branch between PRs on the same WP (#964/#966). | Two PRs append lines to the same `tests/accept/<wp>/PATHS`. | this PR | `.gitattributes` `tests/accept/*/PATHS merge=union`. GitHub's web conflict check ignores merge drivers, so update the branch locally (`git merge origin/main`). Pinned by `tests/accept/w_launch/test_w_launch_lessons_guards.py`. |
| 25 | Knobs that silently do nothing (risk with each new cap: #966 `miner_make_good_cap_usd`, #968 `mkt_lease_max_hours`). | A dataclass field with no `load_chain_config` parse. | #966 and #968 each shipped the parse + a round-trip test | IR007 (every config dataclass field is parsed somewhere). checklist: a loader round-trip test per knob. |
| 26 | Lane D's share burns at a validator restart before its first publication. | The court is memory-only. | operational rule | checklist: restart a validator only right after a lane D publish, never near a boundary. |

## 2026-11-27 root-cause analysis

Owner, 2026-11-27 ~20:50: "run a subagent for root cause analysis, make sure
we have patches, and add any lessons to lint". Sources: the OPSLOG entries of
2026-11-27 and PRs #960–#983. Every fix PR below was checked: it is merged on
`origin/main` and ships a test. Root-cause lines are given at the parent of
the fix commit. The guards marked **new** are added in this PR.

### OPEN (no merged fix + test)

- **MarketCast eval-pod cache on the root overlay.** This one is latent: no
  incident yet. `cascade/validator/mkt_offload.py:338` (`upload`) caches
  checkpoints under `<workdir>/_mkt_offload/cache` without the #982
  data-mount placement, so it has the same failure class as row 33. It is
  pinned as a `KNOWN GAP` in
  `tests/unit/test_eval_pod_cache_dirs_on_data_mount.py`.
- **Validator RSS growth.** 2026-11-27 09:22, 3.7 → 4.5 GB with 3.1 GB of swap
  in use. It was handled by a restart only, and no cause was found (a lane D
  retry loop is suspected).
- **pool_d pin sync pins nothing and logs no reason.** Seen 18:20 at
  09f19baa, after the #976 fresh-sync build. There is no fix and no test.
- **No pool_d DATA-age health check.** cascade-ops `pool_d_stale` reads the
  snapshot's age. The 11-26 freeze renamed the same data every 3 h, so that
  check stayed green. #976 refuses non-advancing snapshots at publish. The
  wanted cascade-ops check (max `first_seen` age, WARN > 9 h, CRIT > 12 h) is
  not built.
- **rc15 → main bitwise qualification FAIL at world 8.** This carries over
  from row 10. The numerics change since 5e3fade has not been found, and the
  repin ran on the owner's override.

### Incidents

| # | Symptom | Root cause (file:line) | Fix | Guard |
|---|---|---|---|---|
| 27 | Expert legs trained on 1 of 8 GPUs (about 8× slow). A make-good was aborted. | The pinned worker image rc15 (5e3fade, 10-31) has `world = gang_size(...) if slices is None else 1` at `cascade/trainer/pyramid_trainer.py:192`. #748 fixed it on main, but no image carrying the fix was pinned. | #960 (rank rule + `GangShortfall`), image rc3 | test `tests/accept/wb27/test_wb27_gang_ranks.py`. `scripts/check_image_freshness.py` (row 8). checklist 1, 14. |
| 28 | The lane B leg ran at 0.34 steps/s against lane A's ~1.9, with GPUs ~20% busy. | Expert legs skip `accelerate()` (`cascade/trainer/pyramid_dp.py:598-600`, "lane A only"), so `MoEFFN.forward` runs a per-expert Python loop with `.item()` syncs (`cascade/trainer/pyramid_model.py:289`). | #970 (lane B fast path + parity test) | test `tests/accept/wb27/test_wb27_expert_fast_path.py`. checklist 12. |
| 29 | The first rc4 leg died in its first backward: "expected scalar type Float but found BFloat16". About 5 min of 8xH200 was lost. | `_ExpertMatmul.backward` (`cascade/trainer/expert_training.py:31`) mixed the bf16 autocast forward with fp32 leaves. #970's tests ran in fp32 only. | #974 | test `tests/accept/wb27/test_wb27_expert_mm_autocast.py` (CPU bf16 autocast). **new** `tests/unit/test_autograd_functions_have_bf16_tests.py`: every custom `autograd.Function` in `cascade/` needs a bf16-autocast backward test. It fails on the #974 parent. checklist 14. |
| 30 | A finished 14 GB leg was harvested over tar-over-ssh Australia → NYC at ~0.5 MB/s. The 1 h stall watchdog would have torn the pod down and burned the run. It was rescued by hand through presigned R2. | The harvest stall path (`cascade/trainer/remote.py:1810` → `cascade/trainer/runner_funded_legs.py:611-680`) treated a slow transfer of a finished run as a failed leg: teardown, burn. | #979 (stall → pod → R2 fallback, pod kept, unburned) | test `tests/accept/wb27/test_wb27_harvest_r2_fallback.py`. |
| 31 | The leg's checkpoint was rejected after harvest: expected `ebfed637…`, receipt `a3ea79b2…`. | Trainer vs worker lane B contract. `apply_spec_levers` (`cascade/shared/lane_b_levers.py:371`) folded the default levers into a no-spec leg. The worker folds them only `if lane_b_spec:` (`cascade/trainer/worker.py:305`). | #980 | test `tests/accept/wb27/test_wb27_lane_b_receipt_contract.py`. **new** `tests/unit/test_lane_b_trainer_worker_contract_parity.py`: runs the worker's own expert-contract block (AST-extracted, torch-free) on the dispatch's exact `--recipe-json` for no-spec, `train_hours`, levers, and `train_hours` + levers. It fails on the #980 parent with the live diff (`power_ema_*`). |
| 32 | A trainer restart did not re-attach the finished leg ("ReattachMissing"). It removed the 8xH200 and requeued the leg, so the approval would have retrained it (~$100). | Re-attach re-derived the era from the current scheduler (`cascade/trainer/lane_b_driver.py:2231`, `era = sched.state.current`) and its contract block (`:2322`). 9594000 → 9597600 changed the run tag, so the finished run was "not found". It also did not recognise a finished run (progress.json + "worker done" + checkpoint). | #982 (`_dispatch_era`; finished-run harvest) | tests `tests/accept/wb27/test_wb27_reattach_finished_and_cache_disk.py::test_lane_b_reattach_rederives_the_dispatch_era`, `::test_reattach_harvests_the_finished_expert_run_on_the_live_dispatch_dir`, `::test_a_requeued_fresh_dispatch_harvests_the_finished_run_instead_of_retraining` (the 5GMzsP fixture). checklist 15. |
| 33 | The 16:27 lane B settlement failed: "pod disk short … 6.8 GB free < 26.0 GB needed". The 43 GB `/` was at 86% while a 159 GB `/workspace` sat empty. | `OFFLOAD_DIR = "_lane_b_offload"` under `<workdir>` (`cascade/trainer/lane_b_offload.py:76`, also `eval_offload` / `lane_b_bench`) went on the root overlay. | #982 (`cascade/shared/pod_data_root.py`) | test `…/test_wb27_reattach_finished_and_cache_disk.py::test_offload_dir_is_linked_onto_the_larger_data_mount`. **new** `tests/unit/test_eval_pod_cache_dirs_on_data_mount.py`: an AST inventory of `f"{host.workdir}/…"` paths in the eval-pod modules. Each must be placed or reviewed. It fails on the #982 parent. |
| 34 | The 19:27 settlement rebase failed: "pod reply is not JSON". The settlement published without lane B. | `json.loads(proc.stdout)` (`cascade/trainer/lane_b_offload.py:376`) over `<worker> && cat result.json`. `hippius_hub/file_download.py:392` printed "Downloading … (parallel)..." to stdout first. | #983 (`pod_reply_json`) | test `tests/accept/wb27/test_wb27_pod_reply_json.py`. **new IR009**: `json.loads` of an ssh reply's raw stdout. Cat-only reply, last-line parse or tolerant parser only. It flags the #983 parent at line 376. |
| 35 | Lane D pool_d was frozen from 11-26 00:00: every snapshot had the same data and an empty `frontier`, so no window could be drawn. | The builder ran `state_pull` + `mirror_sync` only at bootstrap (`cascade/laned/elastic.py:1026-1033`), never per build. Its forge mirror froze at 11-25 17:59. | #976 (sync every build; publish/pin refuse non-advancing snapshots) | tests `tests/accept/w6d/test_w6d_data_advances.py`, `tests/accept/r13/test_r13_builder.py`, `test_r13_state.py`. OPEN: a cascade-ops data-age check. |
| 36 | The lane D public half timed out (3600 s) on every retry, and its 0.5 share burned. | See row 13. | #967 | row 13 tests. |
| 37 | The money guard said "fine" against a real $15/h burn. Per-type pricing was wrong. | See rows 16–17 (`operator_standing_pods_usd_per_hour`; one rate × one wall). | #966 | `tests/accept/wb27/test_wb27_money_guard.py`. |
| 38 | Every mkt eval pod failed its health gate. | See row 18 (fleet pins for `[marketcast] pod_image` pods). | #964 | `tests/unit/test_stage_pods_gate_on_their_own_pins.py`, `tests/accept/wb27/test_wb27_lane_c_launch.py`. |
| 39 | The validator lane B judge, merged-init and acceptance checks loaded checkpoints on the 7 GB box. | See row 3. | #965 | `tests/accept/wb27/test_wb27_validator_remote.py`, `tests/unit/test_box_side_torch_free.py`. |
| 40 | `receipt-replay` crashed on lane D receipts that were not yet readable (pending), which blocked #960/#961/#963. | `scripts/replay_mainnet_receipts.py` treated a pending or unreadable receipt as a failure. | #962 | test `tests/unit/test_replay_pending_receipts.py`. |
| 41 | Process: rc4 was deployed to a paid leg with only fp32 tests of its new kernel. | No production-dtype test and no real-environment smoke run before a paid leg. | rc5 got a 2xRTX4090 30-step smoke run (owner ask) | **new** bf16 guard (row 29). checklist 14. |
| 42 | Process: the trainer was restarted mid-harvest without checking that the re-attach would recognise the finished run. The pod was removed. | Restart discipline covered "legs in flight" but not "finished, unharvested or unuploaded runs". | #982 (re-attach of finished runs) | checklist 15. |
| 43 | Process: Claude suggested re-running a finished paid leg whose checkpoint existed (owner: no re-run). | Recovery was framed as a retrain, not a rescue of the verified checkpoint (receipt digests, pending re-upload). | operator rescue via `_reupload_pending` (12:15 entry) | checklist 16. |

## Pre-upgrade checklist

Run it before any release, image repin or overlay that changes a lane.

1. **Image built from the deployed SHA.** Rebuild the worker image (and the
   mkt image) from the exact commit the trainer will run. Run
   `python3 scripts/check_image_freshness.py --config <deployed chain.validator.toml>`;
   it must print `OK` for every pin. Before saying "images are current",
   check the digest → build commit mapping (row 8).
2. **All five pins flip together.** Flip the `[v2] *_after` image, the
   fingerprint, the measured-wall image, `prior_train_image_digests` and
   `[round] ops_continuation_image_digest`. Flip the rate tables keyed by
   digest (WB19) in the same step. Never do this inside a seed epoch.
3. **Qualification first.** Do the bitwise world-8 qualification and the
   wall/rate measurement on the new image before scheduling the switch. An
   override is the owner's decision and is logged.
4. **Box-load rule.** Nothing on the box loads torch, a model or a
   checkpoint. New box-side modules go into `BOX_MODULES` in
   `tests/unit/test_box_side_torch_free.py`. Never run torch-backed tests on
   the box.
5. **First-pull ready timeout.** A new image is uncached on every host. The
   ready timeout (`funded_ready_timeout_seconds`, now 1800 s) must cover a
   cold pull of the image's size.
6. **Money caps vs the real pods.** Compare `lium ps` (every pod, including
   the owner's) with the runway's burn, the daily cap and the per-type
   projection for the GPU types the legs will actually rent. Check the Lium
   balance before renting.
7. **External-validator readability.** Everything an external validator
   needs must be readable with its credentials: overlays, receipts, the public
   mirror ACL, published absent records. Consensus constants are released
   first, then activated.
8. **Lane-aware paths.** For each lane the change touches, trace the effective
   (overlay) config through rates, steps, walls, approvals and health-gate
   pins. Never assume lane A constants. A driver built only when the BASE
   config arms its lane cannot be armed by an overlay alone.
9. **Gate defaults vs the live state.** Every open gate or threshold must be
   reachable in the current state, frozen lanes included (row 2).
10. **Real-scale data.** Time every data path (public half, pool_d build,
    state push) at full production size on the real host before arming its
    weight.
11. **Eval-host deps.** Install every profile with the extras its worker
    imports. The worker self-check (`w8c_host_modules`) must pass on the host.
12. **Fast kernels on every role.** A new training role must run the
    accelerated path, or the slowdown must be measured and its wall sized
    for it.
13. **Restart windows.** No trainer restart with legs in flight. No validator
    restart in a scoring window, or before lane D's first publish.
14. **Before a paid leg on a new image or kernel.** Do a short
    real-environment smoke run of any new image before a paid leg: the real
    `worker.main` role with the leg's own args, a few dozen steps, on the
    target GPU type, checking all GPUs busy, no dtype error and the expected
    steps/s. New kernels also need a CI test in the production dtype and
    autocast path (bf16), not only fp32 (rows 29, 41).
15. **Before a trainer restart.** List finished but unharvested or unuploaded
    runs (`lane_b_driver.json` "finished", `_funded_harvest/*.pending.json`,
    live pods holding a done worker). Confirm that the restarted code's
    re-attach recognises each one under its DISPATCH era and contract block,
    never the current one. If it does not, wait or rescue the run first
    (row 32).
16. **A finished paid leg is never retrained.** When a leg's checkpoint
    exists and its receipt digests match, rescue it: R2 or presigned copy,
    then sha256 against the receipt, then the pending re-upload. Withdraw or
    cap the approval first so that no re-rent can pass (row 43).

## Not automated (and why)

- Bitwise qualification, wall and step-rate measurement, the fast-kernel
  speed and first-pull timing all need GPUs and real pods (checklist 3, 5, 12).
- Money caps vs real burn need the live Lium listing. #966 makes the runtime
  guard read it (checklist 6).
- Real-scale timing needs production data (checklist 10).
- Gate reachability depends on the live chain state (checklist 9).
- Image freshness is warn-only in CI, because the repo `chain.toml` is a
  template and the deployed file decides (checklist 1).
