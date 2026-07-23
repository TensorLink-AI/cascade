---
id: DEC-CA-0004
type: decision
title: "Cascade promotion: persist the king, never vacate the throne"
status: active
date: 2026-07-22
tags: [cascade, warm-start, koth, incentives, consensus]
revisit_when: "evidence that a reigning king's promoted init confers a relative advantage on the king (it shouldn't — both roles train from the shared init), or a redesign of the KOTH margin schedule that makes tenure unbounded"
relations: {}
---
On a Cascade promotion the throne PERSISTS: the king stays crowned and only
the reign clock + checkpoint log reset (re-crown the same king). The vacate
action is removed — not made configurable.

Vacate was justified as fairness ("the king's incumbency advantage is now
shared, so it must re-earn the throne"), but in cascade there is no
model-side incumbency advantage to give back: miners compete via
GENERATORS, and both king and challenger models are trained fresh every
round from one shared identical init (the controlled-experiment invariant),
before and after a promotion. Promotion moves the shared baseline for
everyone at once. The re-opened race therefore has the same entrants, same
generators, same shared init — the old king just wins again. And it earns
throughout the vacancy anyway: with `ChampionState()` vacant,
`_king_uid_to_vote` falls back to the manifest king, which is the old king
(trainer keys off on-chain incentive). So vacate re-runs a race the king
already won, keeps paying it, and breaks the state machine doing it:

- STALL (observed live, structural not incidental): with a vacant state,
  `state.apply_round` fills the throne only via its dethrone branch; an
  incumbent win returns `new_king_hotkey=None`, `genesis()` is never called
  in the live loop, and `_cascade_round`'s re-crown guard requires
  `self.state.king_hotkey is not None` — so the reign clock stays null and
  promotions stop firing after ~1–2 cycles.
- `ChampionState()` also wipes `former_kings` (the rewarded court) and
  tenure/streaks, dropping the anti-flap margin back to `start` for the very
  king most proven — pointless dethrone-cheapening plus lost court rewards.
- Every vacate adds a throne-handoff + trainer/validator king re-sync window
  per cycle (the 2026-07-20 divergence class).

Persist does not over-entrench: the dethrone path is untouched (a better
generator dethrones any round) and `margin_for_tenure` is a CAPPED affine
ramp saturating at `end` — tenure never compounds beyond it. Persist also
DELETES Problem 2c of [[DEC-CA-0003]] (synchronized trainer vacate) — no
handoff exists to synchronize; remaining reign-clock work is block-anchoring
it + the manifest-derived reign log.

NOT configurable: promotion behavior is consensus-critical — validators
splitting on vacate-vs-persist fork their champion states and weight
vectors. One behavior, hardcoded. Implementation: `cascade_check` re-crowns
(`crown(king_hotkey=state.king_hotkey, now=now)`) instead of `vacate()`;
`_apply_cascade` stops clearing `ChampionState` (log-only); module docstring
"Action" paragraph updated; tests assert a second cascade fires
`reign_days` after the first with no dethrone in between (the freeze
regression). See [[DEC-CA-0002]].
