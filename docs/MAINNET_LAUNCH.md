# Mainnet launch checklist — netuid 91

Reviewed 2026-07-14 against everything the testnet campaign surfaced (PRs
#76–#93). Items are ordered: blockers → decisions → ops. `chain.toml` carries
the decided values as ready-to-uncomment lines where they would break unit
fixtures if active pre-launch.

## Config blockers (chain.toml)

- [x] `[subnet] netuid = 91` — set.
- [ ] `[round] commit_floor_block` = the announced go-live block. Pre-live
      commits (squatters, rehearsals) never compete and never burn their one
      submission (#93).
- [ ] `[training] expected_gpu = "NVIDIA L40S"` — uncomment at launch.
      Enforcing: folds into contract_digest; king+challenger hardware pairing.
- [ ] `[training] train_image_digest` — REBUILD ghcr.io/tensorlink-ai/
      cascade-worker from the LAUNCH COMMIT and pin the new digest. The
      existing worker-v0.1.0 (sha256:bd7e853b…) predates #76–#93 (CPU-cap fix,
      sandbox hardening, telemetry) and must not be pinned.
- [ ] `[training] ref_throughput_tokens_per_s` — re-measure on a real L40S with
      a saturating generator (expect ~170–185k; see CLAUDE.md "wall is the
      law" — the value is deliberately capability-calibrated, NOT median-miner).
- [ ] `[generator] sandbox_mode = "container"`, `sandbox_image = <digest-pinned
      sandbox image>`, `sandbox_strict = true`. Subprocess mode leaves host RAM
      uncapped for GPU-profile generators and has no device boundary; container
      mode is the real isolation (cgroups + no --gpus + ro rootfs). The sandbox
      image is NOT yet published (only a local test image exists).
- [ ] `[eval]`/`[storage]`: mainnet eval-pool bucket + `cascade-pool publish`
      as a MONITORED cron (never manual); `window_pool` stays empty (bucket
      wins; static pools are an overfitting target). Mainnet manifest/logs/
      receipts buckets + R2/HF mirror repos provisioned (the HF-outage
      fallbacks are only as good as the mirrors actually existing).

## Decisions (defaults exist; choose deliberately)

- [ ] `gift_gate_mode`: launch `"shadow"`, flip `"enforce"` after ~a week of
      duel calibration (first live data point 2026-07-14: challenger passed at
      lcb 0.126 vs tol 0.03 — tolerance may want tightening). Enforce mode
      requires every validator to pin `gift_gate_data_dir` (or run an eval
      pod) — document in validator onboarding.
- [ ] `dethrone_cp` (currently 1): one round win takes the throne. Consider 2
      for throne stability at 24h rounds with real emissions.
- [ ] `cascade_enabled` (+ `cascade_reign_days = 7`): armed at launch or after
      the first stable reign.
- [ ] `corpus_mode = "stream_cpu"` stays for launch (byte-exact audits are the
      trust story; #88 closed the CPU-mode GPU escape). `stream_gpu` is a
      post-launch experiment — see CLAUDE.md and the generator evaluation notes.

## Ops (not toml, still launch-gating)

- [ ] Build + publish digest-pinned **worker image** from the launch commit
      (unlocks two blockers above and the provisioner's image-boot mode).
- [ ] Build + publish digest-pinned **sandbox container image**.
- [ ] systemd units for trainer / validator / provisioner on the mainnet box
      (tmux is testnet-grade; `deploy/cascade-provisioner.service` ships in the
      repo — After=network-online, Restart=on-failure, EnvironmentFile for
      provider keys; write matching units for trainer+validator).
- [ ] `WANDB_API_KEY` in the environment (wandb has been silently OFF —
      `[wandb] enabled` alone does nothing without the key).
- [ ] Provisioner `provision.toml` for mainnet: heats SKU ladder as on testnet;
      FINAL stage provisioner-managed (one 2-GPU L40S pod, expected_gpu-matched)
      once the worker image exists; eval stage as on testnet.
- [ ] File the bittensor upstream bug: `decode_revealed_commitment` chokes on
      py-substrate-interface's raw (UTF-8-valid) rendering — payload-length
      lottery skips miners. #92 shields OUR readers; other validators' stacks
      would still skip miners until fixed upstream.
- [ ] Validator onboarding doc: pinned SDK (10.5.0), bench data pinning for
      enforce-mode gift gate, eval-offload option.

## Already correct in the template

`epoch_blocks = 7200` (24h) · `max_train_seconds = 10800` (= 3h budget, "wall
is the law") · `one_submission_per_hotkey = true` · quality gates
(`max_abs_value`, dup fraction) · dependency/static-guard blocklists ·
`min_windows = 200` · bootstrap-LCB scoring · `bittensor==10.5.0` pinned (#91).
