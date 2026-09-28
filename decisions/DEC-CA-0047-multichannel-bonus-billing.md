---
id: DEC-CA-0047
type: decision
title: "Token billing with a proportional width bonus: budget_denomination = \"points+mv<PCT>\" bills every channel and discounts C > 1 series by 100/(100+PCT), so the budget (not the wall, not the GPU drawn) decides every leg's tokens"
status: proposed
date: 2026-09-27
tags: [training, multivariate, budget, contract, fairness, trainer, audit]
revisit_when: "an ablation at equal tokens shows two-channel corpora do not beat univariate ones (then the bonus pays for nothing and the rule should be plain \"points\"), or funded legs land on SKUs whose speed spread makes even token billing wall-bound"
relations: {revises: DEC-CA-0042, depends_on: [DEC-CA-0041, DEC-CA-0001], relates_to: [DEC-CA-0036, DEC-CA-0037]}
---

## Decision (owner 2026-09-27, mechanism only — arming is a separate cut)

Add a third budget denomination, `points+mv<PCT>` (PCT an integer 1..100):

* every values entry is a budget point (`points` billing: a `(C, L)` series costs `C×L`);
* a series with `C > 1` is billed `ceil(C×L × 100 / (100 + PCT))` points;
* the trainer's per-batch counter applies the same per-row rule, so the stream's stop and the
  loop's stop agree exactly and `cascade-audit` replays the same consumed prefix;
* univariate series are bit-identical to `points` (every historical round);
* the field stays drop-when-default (`points`); arming `points+mv<PCT>` is a deliberate
  contract_digest bump (trainer config, worker image, audit, miner scorer all read the string).

The repo ships the mechanism with the default unchanged. Arming on mainnet is the owner's
separate decision: testnet first, digest recompute, worker image rebuild + re-pin, era-boundary
restart. Validators need no update — since block 8942400 the signed manifest carries the contract
body and validators gate only the locked terms.

## Why

DEC-CA-0042's series-points billing rewards width by letting a `C`-channel corpus train `C×` the
tokens per fixed step count. Measured 2026-09-27 (scratchpad ablation, private verdict windows):
the two-channel king beats its own single-channel form by ~1 % and beats the previous
single-channel king by ~0.5 %; the gain concentrates on multichannel eval windows; a duplicated
channel collapses the model (−12 %). So the incentive works. But under series-points the extra
tokens are wall-bound: a `C = 2` corpus fills its budget in ~5 h on an H100 and stops at the
5 h wall with ~65 % of its steps on an RTX4090. Funded legs land on whatever SKU the marketplace
has, so the token count a miner trains depends on the GPU drawn, not the code. That is unfair
in exactly the place the incentive bites.

Token billing removes the GPU dependence (every width trains the same tokens, the budget binds
before the wall on every allowed SKU) but removes the width reward with it. The proportional
bonus restores a bounded reward: an all-multichannel corpus trains `PCT %` more tokens on any
GPU; a corpus stacking a share `s` of its points earns `1 / (1 − s·PCT/(100+PCT))`; a token
gesture (one stacked series) earns nothing measurable; and because every channel is billed, a
junk or copied second channel spends real budget and the bonus cannot cover it — the rule is
self-policing without any statistical coupling test (which is gameable in both directions).

## Activation: a scheduled switch, not a flip

`[training] budget_denomination_after = "points+mv<PCT>"` + `budget_denomination_after_block = <era
start block>`. The two schedule fields are NEVER part of `contract_digest` (pinning the schedule moves no
in-flight digest); the contract EFFECTIVE at a block is `TrainingContractConfig.at_block(block)`. Every leg
carries its ERA's start block (`cascade-train-worker --contract-block`), so the king pre-train (which
launches before the boundary), the challengers, the settlement manifest (`contract_body` / `contract_digest`
at the era start) and the audit (replays the published body) agree by construction. Legs of eras that
started before the gate finish under the old rule; the first era starting at/after the gate trains, settles
and publishes under the new one. The worker image must carry this code before the gate block.

`[static_guard] packed_sources = "reject"` + `packed_sources_from_block = <block>` gates the admission
rejection the same way (orchestrator-side only; no image dependency).

## Not decided here

* The PCT value. 20 reproduces the ~1.2× the current king realises at 22 % stacking under
  series-points, applied instead to full stacking on any GPU.
* Whether two channels help at EQUAL tokens — untested. One leg (own generator on both lanes,
  `points` billing) answers it and is the gate for arming.
* Wall scaling with width (DEC-CA-0042's follow-up) — unnecessary under token billing.
