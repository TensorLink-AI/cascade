# W-LAUNCH acceptance tests: the Wed 2026-11-25 launch blocks

Owner decision 2026-11-24: every new competition (lane B expert legs, lane C
MarketCast, lane D) starts at one era start, the first after Wed
2026-11-25 18:00 UTC — mainnet **9586800** (2663 × 3600), testnet
**8524200** (14207 × 600).

- `chain.toml` types `[lane_c]` and `[marketcast] from_block = 9586800`
  (moved from 9579600). Lane B and lane D are NOT typed there.
- `deploy/overlays/launch-20261125.{pre.,}{mainnet,testnet}.toml`: the
  ready-to-sign two-step overlay. Step 1 (`.pre.`) carries every non-lane-D
  key (lane B `from_block` on mainnet, the lane B / C split shares); step 2
  restates step 1 byte-identically and adds `[lane_d]` (`from_block`, the
  D 50 % share). Two steps because the R3 lint (`non-lane-d`) refuses a
  `[lane_d]` overlay that changes any other key relative to seq N-1.

The tests run the real loader, `netconfig.plan_overlay` (allow-list, merged
load checks, era start, retroactivity) and `cascade.laned.overlay_lint` on
the shipped files. On a base without this change they fail with
`AssertionError` (missing overlay file, old block).

#894 (owner 2026-11-24): lane D launches LIVE-ONLY — both step-2 overlays set
`[lane_d] public_weight = 0.0`. `test_w_launch_lane_d_launches_live_only` pins
it and that the lane D judge (`_public_half`) and inputs source (`_public`)
need no public index, eval-host public scores or block hash at 0 (at 0.5 the
same calls fail closed).

Owner 2026-11-24 22:30: lane D goes live at launch. Step 1 and then step 2
(`launch-20261125.*`, `[lane_d] from_block = 9586800`) are both published at
least one era before the launch. Step 2 must clear a gate: at least 2/3 of
stake signals the #913 release (`upgrade_tracker`). The gate is enforced by
`cascade-laned overlay publish` whenever an overlay turns LANE D OFF -> ON,
and `--force-upgrade-threshold` overrides it.

`deploy/overlays/lane-d-step2.{mainnet,testnet}.toml` is the fallback for a
step 2 that misses the launch. It is the same body with `[lane_d] from_block =
"next-era"`, which `--effective-block next-era --head-block H` resolves to the
first era start at least one era past H.

`test_w_launch_deferred_step2.py` pins:
- both bodies;
- the gate on the launch step 2 and on the fallback;
- the plan lines `LANE D: OFF -> ON` and `LANE A SHARE: 0.5 -> 0.0`;
- the resolution;
- the R3 lint on the resolved fallback.

cascade#909 (owner 2026-11-25): lane B's king / candidate / anchor public rows
are benched on an EVAL POD named by `cascade-trainer --lane-b-bench-hosts`
(launch: `targon-eval.toml`, lane A's bench pod), never on the CPU-only trainer
box. `cascade/trainer/lane_b_bench.py` caches rows by checkpoint content and
benches on one background worker; a settlement never benches inline (a full
four-suite sweep is ~1 h, past `LANE_B_SETTLE_MAX_SECONDS`), queues every
missing candidate in one pass and fails closed (legs kept, nothing published).
No pod-side lock: assumes lane A sweeps are quiet after launch; add a lock if
lane A resumes. `test_w_launch_laneb_bench.py` pins it with fakes.

#933 (launch-critical): with lane A at 0 % after the launch, the top-incentive
hotkey is a lane B/C/D payee, so the trainer must never take lane A's king from
the on-chain incentive. `RollingScheduler._king_hotkey` resolves from signed
sources only: the receipt trail's king, then the trainer's persisted era king,
else HOLD (no king, no king leg, ERROR log + a `lane_a_king_unresolved`
critical operator alert, cleared once a receipt names a king). The metagraph
read stays as a log-only cross-check. `test_w_launch_lane_a_king.py` pins:
- no receipt: the persisted king is kept through settlements and the era roll;
- a lane D payee with top incentive is never chosen;
- first run, no receipt and no state: hold + alert;
- receipt names the king: the king whatever the incentive says, and a run whose
  later receipts name no king publishes byte-identical manifests.

Lane A shutdown (owner 2026-11-25, "lane A should shut down when the other
lanes start"): `[round] lane_a_until_block` (0 = off; 9586800 on the
operator's deployed chain.toml). It is trainer + intake policy only: it is not
an overlay key, not in `_UNTIL_KEYS` and not a digest input, and no validator
reads it. From that block:
- the intake refuses a new lane A `/v1/fund` and a `/v1/submit` carrying a
  payer key with `409 lane_ended` (`lane` = `"a"`) before the key is vaulted;
- a plain `/v1/submit` (store only) stays open, because lane B legs may name a
  vault ref;
- not yet started lane A entries are released, with nothing charged; their keys
  are left to the vault TTL;
- the trainer seats no new lane A challenger. In-flight legs, the king leg and
  king-only settlements continue.

`test_w_launch_lane_a_cutoff.py` pins it.

Provisioner orphan scope (mainnet 2026-11-25 07:50): the dedicated MarketCast
offload provisioner (`provision.mkt.toml`, eval off, mkt on) reaped
`cascade-1795592862-mkt-0`, a Lium pod it never rented, because the reaper
treated every `cascade-<n>-<stage>` name missing from its own ledger as an
orphan. Two changes fix this:
- Every launch first writes a `RentIntent` (provider, name prefix, stage) to
  the ledger, before the provider call. When the launch returns, the intent is
  replaced by the pod ids in one atomic save.
- The reaper terminates a pod only when it matches a recorded intent of THIS
  instance for a stage the instance has enabled. Other provisioner-named pods
  get one WARNING and are left alone.

`test_w_launch_mkt_orphan_scope.py` pins:
- a foreign mkt pod is left alone;
- an mkt pod with this instance's intent is reaped;
- heat, final and eval pods are never touched by an mkt-only instance;
- the intent is on disk before `launch` is called, and a crash after create
  is reaped on restart.
Launch miner docs + CLI (owner 2026-11-25). Every miner-facing guide states
that lane A is closed to new challengers from 9586800, the step 1 / step 2
splits (A 50 / B 35 / C 15, then A 0 / B 35 / C 15 / D 50 once lane D is
armed), that D's 50% stays with the lane A king until the first lane D crown,
the 24 h launch embargo and that the lane D public weight is set by overlay.
- The generated `cascade:rewards:summary` block is computed by
  `scripts/check_docs.py` from `chain.toml` plus the mainnet launch overlays
  (`launch_configs`), so every copy states the LIVE split; the allocation,
  king-start and slot-tiers tables stay the checked-in template.
- The CLI help (top level, `deploy`, `submit`, `fund`) says lane A is closed.
  `cascade deploy` (without `--expert-domain`) and `cascade submit` (unless
  store only, `--no-fund --no-commit`) read the chain head and refuse a lane A
  entry at or past the cutoff BEFORE any upload, intake POST or chain commit
  (`lane_a_closed`, exit 1; `--allow-closed-lane-a` overrides). The cutoff is
  `[round] lane_a_until_block` when the loaded config types it, else the
  published per-netuid cutoff (mainnet 91: 9586800); the miner's checked-in
  chain.toml keeps the key at 0.
- The overlay sync: both mainnet launch steps and the next-era fallback step 2
  restate `[v2] reign_warm_start_blocks = 0` from the published step 1
  (cfg-961291bc); without it the R3 `non-lane-d` lint refuses step 2.

`test_w_launch_miner_cli.py` and `test_w_launch_overlay_sync.py` pin it.
