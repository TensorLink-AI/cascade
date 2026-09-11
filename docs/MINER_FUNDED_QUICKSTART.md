# Miner quickstart: funded legs, private submissions, multivariate data

Short version of what changes for miners from **Fri 2026-09-11 ~20:30 UTC (block 9046800)**.
Long form: [MINER.md](MINER.md) §6b, [MINER_FUNDED_ROUNDS.md](MINER_FUNDED_ROUNDS.md), [INTERFACE.md](INTERFACE.md).

| | block | UTC |
|---|---|---|
| last operator-funded round | 9043200 | Fri 2026-09-11 ~08:30 |
| **first funded round** — funding + private submissions live | **9046800** | **Fri 2026-09-11 ~20:30** |
| multivariate eval windows scored jointly | 9068400 | Mon 2026-09-14 ~20:30 |

Rounds stay on the 12h grid (~08:30 / ~20:30 UTC boundaries).

**Intake (mainnet, netuid 91): `https://submissions.cascadesub.net`** — live now; `GET /v1/queue` shows the live queue.

---

## 1. Fund your leg with a Lium key

From 9046800 a revealed submission competes only after you fund its training leg. The pod
that trains your generator rents on **your** [Lium](https://lium.io) API key; the operator
pays for the king, the evals and the validators. No heat screen — every funded, seated
entrant duels the king.

```bash
export LIUM_API_KEY=sk-...                     # env only; the CLI never takes it as an argument
cascade deploy ./my-generator --hub-repo <ns/name> --wallet-name w --wallet-hotkey h   # as before
cascade fund https://submissions.cascadesub.net --ref <repo@digest> --wallet-name w --wallet-hotkey h
# → fund: queued
cascade queue --intake https://submissions.cascadesub.net --hotkey <your ss58>    # live queue + last roster, your rows marked
cascade fund https://submissions.cascadesub.net --ref <repo@digest> --withdraw --wallet-name w --wallet-hotkey h   # leave while still queued
```

**Rules**

- Hotkey must be **registered** (`403 not_registered`); the ref must match your **revealed** commitment (`403 not_revealed`).
- **Seat order = reveal block.** Up to **8 seats a round**, clamped to live GPU capacity. Unseated entries wait, unburned, in the same order.
- **One GPU type per round**, the most available of `RTX4090, RTX3090, L40S, L40, A6000`; king and challengers share it. Keep **~4h of that GPU's Lium price** on your account: ~3h leg + ~1h benching your own checkpoint on the same pod.
- **Never burned by infrastructure:** dead pod, sold-out market, rate limit → re-queued, nothing spent. Bad/revoked key → entry released as `auth`, fix and fund again. Your generator crashing on the pod = your run, spent.
- **One submission per hotkey, spent only when judged.** A hotkey that competed in any round up to Fri 08:30 is spent. Use a **fresh hotkey, reveal after block 9043200, fund before 9046800** to be in the first funded round.
- **Key custody:** held ≤ 36h in a sealed vault (0600, never logged), used only to rent/tear down your pod, forgotten on withdraw. The request is signed by your hotkey over the key hash + timestamp + ref. Use `https://` only.
- **No funded entry at a boundary ⇒ no round.** The king holds.

Terminal states you can see in `cascade queue`: `auth`, `ref_mismatch` (you re-revealed a different ref), `burned`, `funding_expired` (entry outlived the 36h TTL). All re-fundable.

---

## 2. Private submissions (code stays private unless it takes the throne)

Skip the public Hub repo: POST the code straight to the operator's private vault, fund in the same request.

```bash
export LIUM_API_KEY=sk-...
cascade submit ./my-generator https://submissions.cascadesub.net --wallet-name w --wallet-hotkey h
# → verifies locally, ZIPs the repo, stores it privately, chain-commits
#   vault/direct@sha256:<hex>, funds the leg (auto-queues when the reveal lands)
```

Useful flags: `--no-fund` (fund later with `cascade fund`), `--no-commit` (print the chain payload, touch nothing),
`--reveal-now` / `--next-epoch` / `--blocks-until-reveal N` (reveal timing; default is the timed reveal),
`--skip-runtime` / `--skip-verify` (the trainer still verifies).

- **Losers are never published.** A king's code publishes only when it is **deposed** (`champion_publish = "dethrone"`) to `champions/` on the manifest bucket; `cascade fetch king` resolves it anonymously.
- **Earliest upload owns a digest.** Another hotkey committing your digest is dropped at field entry; byte-copies still die at dedup.
- Submit (or fund) within the 36h key TTL of your reveal; the default timed reveal leaves ample headroom.
- On chain it is an ordinary `metro-v1:gen:hippius:` commit under the reserved `vault/direct` repo — validators need nothing.

---

## 3. Multivariate series (`max_channels = 32`)

`generate()` may now yield **`(C, L)` float arrays, C ≤ 32** — the channels are the variates of one
series. A 1-D `(L,)` yield is still one channel; a univariate generator is unchanged and byte-identical.

```python
def generate(self, n_series, rng):
    for _ in range(n_series):
        L = int(rng.integers(self.min_len, self.max_len + 1))
        yield self.coupled_system(L, rng)        # np.ndarray, shape (C, L), C <= 32
```

- **A channel costs what it trains:** a `(C, L)` series bills `C×L` points of the token budget. No discount, no penalty.
- **Step count is independent of C:** batches fill to `batch_size // C` series (`batch_denomination = "sequences"`), so a C=32 corpus gets the same ~number of optimizer steps as a univariate one. Width buys cross-channel signal per step, not fewer steps.
- **From 9068400 (Mon ~20:30 UTC)** the duel scores multivariate eval windows jointly: all channels forecast in one pass, the window counts **once** with its channels averaged in. Until a multivariate eval pool is published this changes nothing.
- **Only coupled channels teach anything.** At C=1 the variate attention receives no gradient. Stacking unrelated series into one array is legal but useless — channel-redundancy telemetry (`channel_corr_mode = "shadow"`, `max_channel_corr = 0.999`) labels near-duplicate channels now; enforcement follows calibration.
- Optional `mask` field: `(C, L)` uint8 parallel to `values`, 1 = missing (see INTERFACE.md).
- Eval-side windows stay ≤ 8 channels regardless of your C.

---

## Friday checklist

1. Fresh registered hotkey if yours has competed.
2. Lium API key with ~4h of RTX4090-class balance.
3. Latest `cascade` CLI (`cascade fund`, `cascade submit`, `cascade queue` must exist).
4. Reveal after Fri ~08:30 UTC (9043200); `cascade fund` or `cascade submit` before ~20:30 UTC (9046800).
5. Watch `cascade queue --intake https://submissions.cascadesub.net --hotkey <you>` and `cascade round`.
