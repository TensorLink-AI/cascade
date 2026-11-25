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
