# W-IMG acceptance tests: one worker image per seed epoch

Plan rev6.24 (e), WP W-IMG. Depends on W-OV only for sequencing; the code is
independent. Runs in CI and on testnet. **This WP is a deliberate
reproducibility-contract decision, signed off by the owner.**

## Owner decision (2026-11-24, verbatim)

> "can we relax need for image to be pinned"
>
> "we need a system so that we roll all these lanes out, validators only need
> to update for major upgrades, not if we tweak training image or comp rules"
>
> "we just log what image pin was used, it's auditable"

## The rule

- **Lane A.** The image stays inside each leg's `contract_digest`, so duel
  reproducibility is intact. From the first v2 seed epoch starting at or after
  the typed `[v2] image_epoch_from_block` (release-then-activate):
  - every entry of a settlement ran the image its declared contract names;
  - that image is the same for the whole seed epoch;
  - it changes only at a seed-epoch boundary.
  There is no fixed validator pin. The trainer binds the `[v2] *_after`
  release at the next seed-epoch start. The qualification record stays per
  image.
- **Unmeasured walls** (owner 2026-11-24, "YES to the 5 h cap"). A funded or
  king rent on an image with no measured pod walls is rented instead of
  refused:
  - the leg wall is capped at `[round] unmeasured_image_wall_cap_seconds`
    (18000);
  - the leg's record carries `walls_measured: false` and the cap;
  - 0 restores the old refusal.
  Measured images are unchanged.
- **Lanes B, C, D.** The image is recorded, never enforced:
  - `ExpertEntry.train_image_digest`;
  - `LaneCEntry.train_image_digest`;
  - the lane D report's `extensions.eval_image`.
  Each is dropped at its default, so every earlier record keeps its bytes.
- **Before activation.** Everything is byte-identical:
  - the switch block is still `image_decoupled_from_block`;
  - the audit adds no row;
  - no record is stamped;
  - contract digests are unchanged.

## Tests

- `test_w_img_lane_a.py`:
  - one image passes, mixed images are refused, and a mid-epoch change is
    refused;
  - a new image at the next seed epoch passes with no config change;
  - the rule starts only at a seed-epoch boundary;
  - the audit per receipt and across receipts;
  - the trainer binding;
  - byte-identity before activation.
- `test_w_img_lanes_bcd.py`: lanes B, C and D record the image, dropped when it
  is the default.
- `test_w_img_wall_cap.py`: a measured image is unchanged, an unmeasured one
  rents under the cap (the leg wall, the watchdog and the record), and cap 0
  refuses as before.
- `test_w_img_config.py`: the loader, the shipped files are off, and the keys
  are release-only.

No torch: never run torch-backed files or `make laned-e2e` on the ops box.
