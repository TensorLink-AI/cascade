# Shadow scratch control — DEC-CA-0014 Stage 1 (operations)

What ships: every M-th warm-started round (`[telemetry]
scratch_shadow_every_rounds`; 0 = off) the trainer additionally trains the
reigning king's generator **from scratch** — random init, the round's own
seeds, the identical training contract — strictly after the manifest
publishes, benches it on GIFT-Eval/BOOM/TIME, and publishes a trainer-signed,
explicitly-labeled telemetry report:

- `benchmarks/scratch/round-<id>.json` — the signed per-round document
  (`cascade.shared.scratch_report`; `kind = "scratch_shadow"`,
  `telemetry_only = true` inside the signed body). Carries the scratch
  checkpoint's six numbers plus, when the round's duel bench produced them,
  the lineage king's six as an in-document reference — one fetch, one point on
  both curves.
- `benchmarks/scratch/index.json` — unsigned presentational roll-up: per
  round, `scratch_geomean`, `king_geomean`, and their `gap`. The trend in
  `gap` is the whole Stage-1 output: compounding / flat / closing decides
  whether the Stage-2 reseed valve is ever armed (DEC-CA-0014).

## Why it is consensus-inert (the invariants tests pin)

- `[telemetry]` tier: the knob never enters `contract_digest`
  (`test_telemetry_key_never_touches_contract_digest`).
- The scratch run is never a manifest entry, never a `BenchReport` entry
  (validators parse that schema with strict role validation on the
  promotion-provenance path — extending it would be a fleet-lockstep change;
  a new key namespace is invisible to deployed consumers), and is never
  handed to `promotion.record_bench` — the promotion candidate pool cannot
  see it.
- It runs strictly post-publish on the bench thread with the bench's
  swallow-everything contract: a scratch failure cannot delay, fail, or
  modify a round.
- Shippable trainer-side unilaterally: the remote leg drives the PINNED
  worker image's existing CLI (`--role king --repo-suffix -scratch
  --train-hours <target_train_hours>`, no `--warm-start-ref`) — no pod-side
  code change, so no image re-pin, so no `contract_digest` event.

Two accepted telemetry-grade asymmetries of the pinned-worker route, both
documented in `loop._scratch_shadow_remote`: the run's S3/wandb label is the
screen-path key `heat-<king_hotkey>` (a key a king never otherwise uses — the
first-class `scratch-king-<size>` label needs a worker CLI flag, which waits
for the next routine image re-pin), and the worker skips `assert_train_image`
on the screen path (the pod was launched from the pinned image regardless).
The budget is byte-identical either way: `for_hours(target_train_hours)`
yields the same token count, and the wall guard clamps to
`max_train_seconds` at the shipped `heat_guard_factor >= 1.0`.

## GPU budget (the Stage-1 cost note)

A shadow costs one full training leg + one bench sweep, every M rounds.

**Where it runs — recommended: spare final-lane time (implemented).** The
scratch leg targets the king's own final pod after its duel bench, inside the
provisioner's existing `bench_pending`/`bench_complete` hold. Mainnet
timeline on that pod after the manifest: king's duel bench ~45–60 min →
scratch train 3 h (fixed token budget) → scratch bench ~45–60 min ≈ **5 h**,
inside the shipped `[eval] bench_hold_max_hours = 6.0` (the challenger's
bench runs on its own pod in parallel and releases normally). Cost at the
DEC-CA-0014 starting cadence M=4: ~4 extra L40S pod-hours per 48 h — about
1/12 of a lane, or ≈ +65 % on the king-pod's per-shadow-round rental and
≈ +17 % on final-stage GPU averaged over the cycle. Heat-fleet spend, which
dominates on big fields, is untouched. If the hold ever races the cap
(slow SKU, HF re-download), raise `bench_hold_max_hours` to 8.0 — an
`[eval]`-tier, owner-local value — rather than shrinking the leg.

**Alternative — one extra rented pod.** A third final-class lane rented only
on shadow rounds isolates the leg completely (no hold coupling, no cap risk)
at the price of provisioner policy work (a new stage class) plus boot/rsync
overhead per shadow. Not warranted at M=4; revisit if M drops below the
point where holds overlap round boundaries.

Testnet: the 0.75 h budget makes the whole leg ~50 min with the capped bench
(`cascade_bench_max_series = 3`) — cheap enough to run at M=2.

## Testnet validation (one full cycle before mainnet arming)

`chain.testnet.toml` on this branch arms `scratch_shadow_every_rounds = 2`.
Restart the testnet trainer with it (validators need nothing — that is the
point) and check, over one promotion cycle:

1. A warm-started round on an even epoch index logs
   `scratch shadow leg starting` and trains the extra leg post-publish; the
   checkpoint lands at `ckpt-r<seed>-king-<size>-scratch`.
2. `benchmarks/scratch/round-<id>.json` publishes signed and labeled;
   `benchmarks/scratch/index.json` grows a row with a finite `gap`.
3. The round's manifest and `benchmarks/round-<id>.json` are shape-identical
   to a non-shadow round (no extra entries); the validator logs no new
   warnings; the validator's reign log and the next `PromotionRecord` contain
   no scratch pointer.
4. The bench hold releases normally (`bench_complete.json`) and the next
   round dispatches clean — the shadow never bleeds into it.
5. A random-init round and an odd-epoch round both skip, with the reason
   logged.
6. After ≥2 shadow points, the index's two curves populate and the wandb
   `scratch_shadow` events mirror them.

Then mainnet arming is a one-line `[telemetry]` change (owner sign-off,
`scratch_shadow_every_rounds = 4` per DEC-CA-0014) plus a trainer restart —
no validator involvement, no digest change, no lockstep.

## OPSLOG entries (paste into /root/OPSLOG.md when deploying)

```
YYYY-MM-DD  DEC-CA-0014 Stage 1 BUILT (branch claude/margin-decay-dec-ca-0014-qorsop).
  Shadow scratch control: [telemetry] scratch_shadow_every_rounds (0=off; testnet
  armed at 2), trainer-side only. Every M-th warm-started round trains the king's
  generator from scratch post-publish on the king's final pod (inside the bench
  hold; ~5h at mainnet budget, fits the 6h hold) and publishes signed
  benchmarks/scratch/round-<id>.json + index. Never enters manifest/BenchReport/
  promotion pool — consensus-inert, no digest change, no validator deploy.
  Remote leg rides the PINNED worker CLI (role=king, repo-suffix=-scratch,
  train-hours=target ⇒ byte-identical budget); telemetry label for that leg is
  heat-<king_hotkey> until the next routine image re-pin adds a first-class flag.
  NEXT: one full testnet cycle per docs/SCRATCH_SHADOW.md, then owner decision
  on mainnet M=4.

YYYY-MM-DD  OPEN DESIGN THREAD margin decay: replay harness landed
  (scripts/replay_margin_decay.py — pure margin_for_tenure replay over recorded
  tenure+LCBs, cohort-aware, floor>0 guardrail, near-miss report). Owner-picked
  candidate staged in chain.testnet.toml (start=0.02 end=0.010 warmup=12);
  DEC-CA-0016 node PROPOSED; mainnet plan is release-then-activate across all 6
  external validators (docs/MARGIN_DECAY_ROLLOUT.md — read the verdict-fork
  blast radius section before any deploy step). NEXT: run the harness over the
  mainnet receipt trail (--fetch --verify), owner picks (end, warmup), testnet
  ramp, then the coordinated release.
```
