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
- `test_w_img_era_king_fixed.py` (#934): the trainer side of "the image
  changes only at a seed-epoch boundary":
  - a king the validators hold (published, or carried into a later era of
    its seed epoch) is never dropped or retrained after a restart onto a
    new image or contract;
  - an image-only repin keeps proving it with the image it was dispatched
    under, so settlements keep the SAME king pointer;
  - an unprovable change HOLDS the settlement with a loud error; the next
    seed epoch trains its king under the new contract;
  - the normal rollover is unchanged, and a king nobody holds yet (the
    epoch's first era, nothing published) still retrains as before.
- `test_w_img_lock_frozen.py` (owner 2026-11-28): the image's venv equals
  `uv.lock`:
  - the comparer (`cascade.shared.lockdrift`) finds version drift, packages
    the lock does not know and a wrong git commit, numerics-relevant first;
  - `deploy/Dockerfile` installs with `uv sync --frozen` and fails the build on
    drift, and `deploy/pod_bootstrap.sh` checks after its sync;
  - no install path re-resolves the main project (`pip install -e`), and the
    benchmarks sidecar stays frozen on its own lock;
  - the health gate's `lock_match` refuses a marked pod that drifted and
    reports a legacy image (no marker) without refusing it.

No torch: never run torch-backed files or `make laned-e2e` on the ops box.

## Retro item 8: images built, qualified and smoke-tested by the pipeline (owner 2026-11-28)

`test_w_img_qualify_pipeline.py`:

- a bitwise-only `/1` record no longer qualifies an `*_after` image. A `/2`
  record must carry:
  - the publish-worker build manifest;
  - a passing 30-step smoke for every fleet role (the bf16 autocast backward,
    the full gang);
  - measured walls on the SKU's own GPU count;
  - the golden files;
- `[marketcast] pod_image` is refused without `qualifications/mkt-<to12>.json`.
  That record needs the flat-context forward to be bit-identical to the
  previous MarketCast image's, and a pipeline build;
- an override needs a date at most 14 days ahead. It stops being honoured at
  that date;
- the build manifest is the checkout's lock-frozen build: fingerprint, lock,
  marker, arch list. CI emits it from a Dockerfile probe stage, without
  pulling the image;
- the smoke verdicts catch the three paid-leg classes: the rc15 single-GPU
  leg, the rc4 bf16 crash and the lane C NaN;
- the job plans one cheap Lium pod and emits ONE config patch (the 5 pins +
  the walls);
- docs/OPERATOR.md orders the release: build, qualify, walls, record, patch,
  repin at a seed-epoch start.
