<p align="center">
  <img src="docs/assets/cascade-logo.png" alt="cascade logo" width="180">
</p>

# Cascade: synthetic data, real forecasting tests

**The ambition is a model that sees the future.** Cascade is a Bittensor
subnet (netuid 91) where miners compete to make a shared time-series
forecaster better: with synthetic training data, expert updates, fine-tuning
recipes or their own trained weights. Every entry is judged on real series it
could not have trained on. The [competition introduction](agent.md) explains
the idea and links the historical evidence.

**Status.** Lane A (synthetic-data generators) has been live on mainnet since
30 October 2026. Lanes B, C and D open together at block **9586800** (about
Wed 2026-11-25 22:28 UTC). A signed launch overlay sets the emission split at
that block to **A 0% / B 35% / C 15% / D 50%**. `cascade rules a|b|c|d`
prints a lane's rules from `chain.toml`; `cascade status --json` shows what
is live now.

## Competitions

Each lane has a one-page brief: intent, what you submit, how it is scored and
paid (with a worked example), commands, strategies, costs, refusals and a FAQ.

| Lane | You compete with | Brief |
|---|---|---|
| A | A synthetic-data generator; the model and recipe are fixed | [MINER_LANE_A.md](docs/MINER_LANE_A.md) |
| B | A generator that trains a few experts for a domain, horizon or benchmark suite | [MINER_LANE_B.md](docs/MINER_LANE_B.md) |
| C | A MarketCast fine-tuning recipe for the pinned Timestral base | [MINER_LANE_C_MARKETCAST.md](docs/MINER_LANE_C_MARKETCAST.md) |
| D | Open weights: a `cascade_pyramid` checkpoint you trained anywhere | [MINER_LANE_D.md](docs/MINER_LANE_D.md), reference [LANE_D.md](docs/LANE_D.md) |

### Rules at a glance

| Lane | To win | What a win earns |
|---|---|---|
| A | Your gain LCB over the king must clear the dethrone margin (0.10, decaying to 0.01 with tenure), then pass the GIFT-Eval/BOOM/TIME/FEV bench gate | The throne and its ladder share of A; the king is also the house in lane D and can become the shared init |
| B | Clear a slot: CRPS or MASE gain LCB ≥ 1% on a declared target (2% on any other slot), the other metric non-inferior, no guard regressing; then pass merge acceptance | Each accepted slot pays in full every settlement until a newer leg replaces it |
| C | Beat the pinned Timestral base on embargo-period market windows (composite gain LCB ≥ 1%), pass the calibration and baseline guards, then confirm and beat the slot holder | The MarketCast slot at full pay for 5–20 days, scaled by the confirmed gain |
| D | Beat both the lane D king and the house (lane A king) by the crown bar (3%, decaying to 1% over 32 settlements) on forward-only windows, and pass the drawn bench gate | The king seat of an 80/15/5 court; prior kings stay paid with no time limit |

One hotkey holds one chain commitment across lanes and at most one live
training leg. Funding makes an entry eligible; it never guarantees a reward.

## Start here

| You are | Read |
|---|---|
| A new miner | [Miner guide](docs/MINER.md#quickstart), then your lane's brief above |
| Writing a generator | [Submission interface](docs/INTERFACE.md) and [multichannel generators](docs/MINER.md#11-multichannel-generators) |
| A MarketCast recipe author | [MarketCast reference](docs/MARKETCAST.md) |
| A coding agent | [AGENTS.md](AGENTS.md) and [the agent walkthrough](docs/AGENT_MINING.md) |
| A validator | [Validator guide](docs/VALIDATOR.md) |
| The operator | [Operator runbook](docs/OPERATOR.md) |
| Reviewing the design | [Architecture](docs/ARCHITECTURE.md), [model and recipe](docs/MODEL.md) and [decisions](decisions/) |

The live model is **Cascade MoE Base - 1.23B** (a pyramid mixture of experts,
112M active parameters per token). Smaller presets (17.58M, 69.02M, 273.84M)
are for cheap local experiments; see [presets](deploy/presets/README.md).
For local scoring use the public eval pool
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool),
an evaluation set and never training data
([details](docs/MINER.md#held-out-data-the-public-eval-pool)).

## Rewards

The numbers below are the **checked-in template** (lanes A, B and C, before
any overlay). On mainnet the launch overlay replaces the shares with
A 0 / B 35 / C 15 / D 50 at block 9586800; each lane brief works its example
at that split. The [complete reward schedule](docs/MINER.md#rewards-allocation-burn-and-progression)
has every rule and a worked allocation.

<!-- cascade:rewards:summary:start -->
These are the checked-in mainnet v2 rules. Check live activation and signed policy before submitting.
Percentages are shares of the competition reward vector. A lane's pool is its budget; miners receive only the shares they earn.

| Launched lanes | A pool | B pool | C pool |
|---|---:|---:|---:|
| A | 100% | 0% | 0% |
| A + C | 75% | 0% | 25% |
| A + B | 75% | 25% | 0% |
| A + B + C | 50% | 25% | 25% |

- **At launch:** a separately activated pre-v2 policy burns 95%. Until the first v2 winner is crowned, the inherited king/court receives at most 5% of total emissions; the rest of A burns.
- **A winner:** a random-init or other level-unit win starts at 50% of A. Increment-unit wins use the duel's lower confidence bound (LCB): LCB ≥ 0.5 → 80%; LCB ≥ 0.25 → 65%; LCB ≥ 0.1 → 50%; LCB ≥ 0.01 → 25%. These are the duel's increment units, not percentages of absolute forecast-error reduction.
- **A progression:** each passed scheduled king check adds 10 percentage points of A, up to 95%. Failure sets 10%; a deferred check holds the share. Time alone does not increase it.
- **Former king:** only the immediate former king can receive 5% of A for 36,000 blocks, while eligible. A qualifying compute-key lapse reduces the current king's payout to 1% of A, without reducing separately earned B/C rewards.
- **B winner:** with merge acceptance enabled, qualification is unpaid. Acceptance into a scheduled shared starting checkpoint starts payment. The 10 slots each pay 2.5% of total emissions to their holder every settlement, with no decay or expiry, until a newer leg's accepted merge takes the slot. Every leg is tested on every slot: up to 4 declared targets clear at a 1% gain LCB, any other slot at 2%; one miner may hold every slot.
- **C winner:** an initial test pass is unpaid. A later confirmation must win the target slot. C's pool is divided by competition weight, then target count. The current sole competition, MarketCast, has 1 target: each full-pay slot is 25% of total emissions.
- **C full-pay duration:** MarketCast C: 5 days (10 eras) at LCB 0.01, rising linearly to 20 days (40 eras) at LCB ≥ 0.05. C uses confirmed gain; each win keeps its recorded duration. B has no tiers.
- **C progression:** C has no decay: its slot ends when the full-pay duration ends. Age starts at C confirmation. A new holder stops the old holder's pay.
- **Burn:** unpaid A, empty B slots, empty/ended C slots and inactive competition allocations go to `burn_uid`; they are not shared among other miners. An unlaunched lane has no reserved pool.
- **Check payment:** read signed reward records and on-chain weights. Queue completion, qualification and pending confirmation are not paid wins. Public status reports `reward_status=not_verified` because it does not verify payment.
<!-- cascade:rewards:summary:end -->

## Commands

Start with `cascade --help`. Every command has `--help` and `--json`. Local
and read-only commands never rent a GPU; paid actions refuse without `--yes`.

| Command | Purpose |
|---|---|
| `cascade rules`, `cascade status` | A lane's rules; what is live now |
| `cascade init`, `verify`, `preflight` | Starter source, local validation, a read-only submission proposal |
| `cascade iterate`, `score` | Local model experiments |
| `cascade submit`, `deploy`, `fund` | Lane A submission and funding |
| `cascade lane-b`, `lane-c`, `lane-d` | Lane B, C and D tests, uploads, commits and funding |
| `cascade round`, `queue`, `duel`, `reveal-status`, `fetch` | Clock, roster, verdicts, reveal checks and public artifacts |

Lanes A–C rent training on your own provider key (Lium, Shadeform, Vast.ai
or Runpod; Lium is the default), supplied through the environment. See
[provider keys](docs/MINER.md#compute-provider-keys). Lane D needs no key.
The JSON envelope and exit codes are in [AGENTS.md](AGENTS.md#output-contract).

## Development

Use uv >=0.13.4 and a patched OS CA store; the locks require certifi
>=2026.10.19 ([TLS setup](docs/OPERATOR.md#tls-trust-and-the-october-2026-emudhra-incident)).

```bash
uv sync --frozen --all-extras
uv run --frozen python scripts/check_tls_trust.py
uv run pytest tests -q
uv run ruff check cascade benchmarks/cascade_benchmark scripts
uv run python scripts/check_docs.py
git diff --check
```

`scripts/check_docs.py` checks CLI examples, links, the configuration
reference and the reward tables against the code. It does not prove that a
deployment is live. Operator services (`cascade-trainer`, `cascade-validator`,
`cascade-satellite`, `cascade-audit` and others) are covered in the
[operator](docs/OPERATOR.md) and [validator](docs/VALIDATOR.md) guides; the
code layout is in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
