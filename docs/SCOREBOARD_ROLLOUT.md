# Stakeholder scoreboard — rollout runbook

Everything the stakeholder scoreboard (`cascade/website/stakeholders.html`)
needs to go fully live. All code is on this branch; this file lists the
**operational** steps, in order, with what each unlocks. Steps are
independent — do them in any order; the page degrades honestly (pending
states) for anything not yet done.

The page computes every figure from public objects in the manifest bucket
(`chain.toml [storage] manifest_bucket`). Nothing is typed in by hand;
publishing the right objects is all there is to it.

## Credentials / infra needed

| What | Needed for |
|---|---|
| `HIPPIUS_S3_ACCESS_KEY` / `HIPPIUS_S3_SECRET_KEY` | every publish step |
| Chain access (public RPC; read-only, no wallet) | economics feed |
| Trainer wallet (optional) | signing reference bench docs |
| GPU pod + `cascade-bench` docker image (or `uv sync --project benchmarks` + `cascade-benchmark-download`) | official-Toto2 baselines |
| Hugging Face access to `Datadog/Toto-2.0-4m` (and other released sizes) | official-Toto2 baselines |

## 1. Publish the page itself

```bash
python scripts/publish_website.py --only stakeholders
```

Unlocks: everything below appears as it lands; until then the live site
serves the old page. Re-run after any future page change.

## 2. Economics cell (alpha price + emissions)

The cell reads the `economics` block of `status/chain.json`. Publishers on
this branch include it automatically; the currently deployed validator
predates it. Either:

```bash
# zero-risk, immediate — read-only on chain, no wallet:
python scripts/publish_chain_status.py --loop 60
```

or restart the validator on this branch (its poll-cadence publish then
carries economics; no `[training]` change is involved, so the
contract-digest restart coupling is not tripped).

Verify: `curl <endpoint>/<bucket>/status/chain.json | jq .economics`

## 3. Round benchmark reports (GIFT-Eval / BOOM / TIME)

The page probes `benchmarks/round-<round_id>.json` for the newest 48 scored
rounds (misses among the newest 6 are re-probed; older misses are treated
as permanently absent — some rounds legitimately have no report). If the
production pipeline already publishes these objects, nothing to do; the
tile and section light up on their own.

If it does not (nothing in this repo publishes them automatically — the
trainer's `bench_hook` is log-only), publish per round from the sidecar
reports:

```bash
python scripts/publish_round_bench.py --round-id <id> \
    --entry role=king,size=toto2-4m,report=/out/king-report.json \
    --entry role=challenger,size=toto2-4m,report=/out/chal-report.json \
    --wallet-name trainer --wallet-hotkey default
```

The `size=` stamp per entry is required — it is what lets the page place
each result on the ladder once rounds train at more than one size (and
what advances the hero to the 22M bout when the time comes). Round docs
are immutable once published; the script refuses to overwrite without
`--force`.

Verify: pick a `round_id` from `receipts/index.json` and
`curl <endpoint>/<bucket>/benchmarks/round-<id>.json`.

## 4. Official-Toto2 baselines (the rallying metric)

Once per released official size (start with 4M). Three commands:

```bash
# a) wrap the official weights in cascade's checkpoint layout.
#    --template is any cascade round checkpoint of the SAME size.
#    On a state-dict key mismatch the script prints the full diff and
#    refuses to guess — write a --map-file from the diff and re-run.
python scripts/wrap_official_toto2.py \
    --template /ckpts/<any-cascade-4m-checkpoint> \
    --official /hf/Datadog/Toto-2.0-4m \
    --source "Datadog/Toto-2.0-4m@hf:<revision>" \
    --out /ckpts/official-toto2-4m

# b) score it with the same battery as every round checkpoint (~1h/4090):
docker run --rm --gpus all -v /ckpts/official-toto2-4m:/ckpt:ro -v /out:/out \
    cascade-bench:<tag> /ckpt /out/report.json --device cuda --batch-size 512

# c) publish the reference doc (refuses incomplete batteries; --wallet-*
#    optionally signs with the trainer hotkey):
python scripts/publish_reference_bench.py /out/report.json \
    --preset toto2-4m --source "Datadog/Toto-2.0-4m@hf:<revision>" \
    --wallet-name trainer --wallet-hotkey default
```

Unlocks: the HERO number above the scoreboard (the page's single headline:
score-parity vs the official Toto2 at the largest shared size) and the
"Is it catching the official Toto2?" section, both computed against
Cascade's best published checkpoint at the same size. Until then the hero
shows the target statement with an honest em-dash. Check which official
sizes actually exist on Hugging Face before planning rungs beyond 4M.

Also bench the GENESIS KING once per size — it anchors the hero's
completion meter at the fixed day-one origin. Its checkpoint is a normal
cascade artifact (fetch the day-one king's trained pointer from the
earliest receipt), so steps (a)/wrapping are NOT needed — score it with
the sidecar as-is, then:

```bash
python scripts/publish_reference_bench.py /out/genesis-report.json \
    --preset toto2-4m --kind genesis \
    --source "metro-v1:trained:hippius:cascade/<genesis-ckpt>@sha256:<digest>"
```

Until the genesis doc lands, the meter falls back to the earliest
published round result at that size.

## 5. Pretraining-progress targets (one constant, with citation)

The "How far along is the pretraining?" section computes
the Cascade side entirely from the training contract and the signed
warm-start record — nothing to operate. Its completion bar, however, needs
the OFFICIAL release's per-size step count: fill `TOTO2_PRETRAIN` in
`cascade/website/stakeholders.html` with the step count from the Toto 2.0
tech report / model card and cite it in `source`. A unit test refuses a
filled value without a citation; until filled the section shows an explicit
pending state. Never estimate these values.

## 6. Verify

Open the published page and check, in order:

1. Top bar shows LIVE; X / Discord / GitHub glyphs present; "Mine with
   your agent" links to `docs/MINER.md`.
2. Scoreboard row 1: miners entered / lead changes / better than day one /
   model scale — all live (these only need `receipts/index.json`).
3. Row 2: benchmark or vs-official cell live after step 3/4; compute
   efficiency live; carry-forward "not active yet" (correct until
   warm-start ships); economics live after step 2.
4. Every number's provenance is described under "Where each number comes
   from" — if a claim and its source disagree, the page is wrong; fix the
   page, not the copy.

## Design constraints (do not regress)

- Exactly 8 scoreboard cells, 4×2 desktop; order fixed.
- No invented numbers anywhere: a metric without a published source shows
  an honest pending state.
- Public benchmarks (incl. the official-Toto2 duel) are report-only —
  rounds settle on the hidden pool; keep that sentence in the methodology.
- The technical dashboard (`index.html`) stays the deep-dive; don't
  duplicate it here.
