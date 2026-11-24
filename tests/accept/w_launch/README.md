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

Owner decision 2026-11-24 (option 1): Wednesday is step 1 only. Lane D's step
2 is deferred to `deploy/overlays/lane-d-step2.{mainnet,testnet}.toml`, which
carries the launch step-2 body with `[lane_d] from_block = "next-era"`.
`test_w_launch_deferred_step2.py` pins that body and checks that
`cascade-laned overlay publish --effective-block next-era --head-block H`
resolves the start to the first era start at least one era past H. It also
checks the plan shows `LANE D: OFF -> ON` and `LANE A SHARE: 0.5 -> 0.0`, and
that arming lane D is refused below 2/3 of stake signalling the release
(`upgrade_tracker`) unless `--force-upgrade-threshold` is passed. The resolved
body passes the R3 lint after step 1. `launch-20261125.mainnet.toml` stays as
is, marked superseded for mainnet.
