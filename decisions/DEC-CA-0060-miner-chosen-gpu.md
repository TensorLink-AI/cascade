---
id: DEC-CA-0060
type: decision
title: "Every leg keeps the same token budget; a miner may choose which GPU type (and later how many GPUs) its own key rents, trading money for speed, never for compute"
status: proposed
date: 2026-10-19
tags: [funding, provisioning, lanes]
revisit_when: "multi-GPU legs are wanted (needs a data-parallel worker); or a chosen type keeps missing the wall (warn harder or refuse it at fund time)"
relations: {depends_on: [DEC-CA-0036, DEC-CA-0042], relates_to: [DEC-CA-0059, DEC-CA-0056]}
---

## Decision

Owner (2026-10-19): "They all get same token budget but if they want to rent
more they can."

1. **Budget unchanged.** Every miner-funded leg (lane A challenger, lane B,
   lane C) trains the contract's token budget; the wall stays the law
   (DEC-CA-0001).
2. **Miner picks the GPU.** The fund request may name a preferred GPU type
   from `[round] funded_pod_skus`; the trainer rents that type on the
   miner's key instead of the cheapest fitting one. Not available ⇒ the
   leg waits and requeues like any capacity miss (optional fallback to the
   cheapest type if the miner allows it).
3. **Miner's money, miner's caps.** The operator price caps protect miners
   from the default choice; a miner who names a type accepts its price, up
   to an optional per-request cap they set.
4. **Slow choices are the miner's risk.** A type that cannot finish the
   budget inside the wall (`funded_sku_wall_seconds`) is warned at fund time;
   a leg that stops at the wall is judged as it stands, as today.
5. **Multi-GPU later.** The leg worker is single-GPU today (no data-parallel
   path outside the direct ~220M stack); renting several GPUs per leg needs
   that worker first.

## Shipped

Trainer, intake and miner CLI only. No validator change and no consensus
change: the manifest, receipts and verdict path are untouched, and every
funded leg trains the contract's token budget inside its wall (DEC-CA-0001).

- **Default `allow_fallback = false`** (this settles the earlier open
  question). A chosen type that is unavailable makes the leg wait and requeue
  as a capacity miss: no burn, nothing rented. Fallback to the cheapest type
  is opt-in per request.
- **Intake** (`cascade/funding/intake.py`). `/v1/fund` and `/v1/lane-c/fund`
  parse the optional GPU headers before the signature check, so an unknown
  SKU is refused with 400 `unknown_gpu` and nothing is queued or vaulted. The
  allowed set is `[round] funded_pod_skus` when `funded_sku_per_leg` is on;
  with it off, every choice is refused (`gpu_choice_unavailable`). The
  catalog is built in `cascade/funding/main.py` from chain.toml.
  `--trust-refs` (dev, no chain config) accepts any well-formed SKU.
- **Fund-time warning** (`gpu_choice.wall_warning`). A reply carries
  `"warning"` when the chosen SKU has no `funded_sku_wall_seconds` entry
  (assumed wall-bound) or its measured wall reaches the contract's
  `max_train_seconds`. The request is never refused for this.
- **Queue** (`cascade/funding/queue.py`). `FundedEntry` gains `gpu`,
  `gpu_allow_fallback` and `gpu_max_price_per_hour`. Every accepted fund
  replaces them, so the latest request wins and one without a GPU resets to
  the default. They survive requeue and expiry, and carry into a re-commit
  parked behind an in-flight leg. They are written to the queue file only
  when set, so an entry without a choice serialises exactly as before.
- **Rent** (`cascade/trainer/runner_funded_pods.py` `_rent_funded_host`,
  `cascade/trainer/runner_capacity.py` `_chosen_gpu_rent`):
  - A chosen type rents `skus=(sku,)` under price caps
    `{max_price_per_hour: <miner cap or 0>, max_leg_cost_usd: 0}`. The wall
    table is kept so listings still sort by leg cost.
  - No capacity: the capacity wait polls that type under the miner's cap
    until its own latest safe start, then the leg requeues with
    `no_capacity` and `burn=False`. With `allow_fallback`, the next attempt
    rents the default path at once (`funded_pod_skus`, operator caps).
  - The default path (no choice) is byte-for-byte the old code path.
  - The choice is read from the hotkey's in-flight lane-A entry for this ref
    (`_entry_gpu_choice`). Lane C passes its own entry's choice
    (`lane_c_driver.train_leg`). The king's own-key leg passes the default
    choice: it answers no fund request.
  - A choice whose type was later removed from `funded_pod_skus`, or a
    network with `funded_sku_per_leg` off, rents the default path, with a log
    line.
  - The owner-armed operator-lane hybrid (`funded_operator_fallback`) is
    unchanged. It is operator-billed and never charges the miner.
- **Lane B** (2026-10-19): lane B is miner-funded (DEC-CA-0059 item 8).
  `POST /v1/lane-b/fund` takes the same GPU headers and v3 rule; the choice
  rides the lane-B queue entry and the open leg (`OpenLeg.funding`) into
  `_run_funded_leg(gpu_choice=…)`, never lane A's. `cascade lane-b fund`
  accepts `--gpu`, `--allow-fallback`, `--max-price`.
- **Status.** The `gpu` field (empty means the default) appears in:
  - `/v1/queue` rows, plus `gpu_allow_fallback` and
    `gpu_max_price_per_hour` when set;
  - `/v1/lane-c/queue`;
  - the published roster's `in_flight` / `waiting` rows
    (`rolling.py _roster`);
  - `status/lane_c.json` open legs;
  - `cascade queue`, `cascade lane-c status`.
- **Single GPU, verified.**
  - A funded pod is rented with `gpus_per_pod=1`
    (`provision/funded.py:225`).
  - It is dispatched with `lane_count=1`
    (`trainer/runner_funded_legs.py:100`) on `cuda_device="0"`
    (`trainer/runner_funded_pods.py:710`).
  - The only `torch.distributed` path is the direct ~220M stack
    (`direct/torch_worker.py:91`).

## Interfaces

Request headers (all optional; on `POST /v1/fund` and `POST /v1/lane-c/fund`):

| header | value | default |
|---|---|---|
| `X-Gpu-Type` | a SKU from `[round] funded_pod_skus` (case-insensitive; stored in the catalog's spelling) | none: cheapest fitting type, operator caps |
| `X-Gpu-Allow-Fallback` | `1`/`true`/`yes` or `0`/`false`/`no` | `false` |
| `X-Gpu-Max-Price` | the miner's own USD per GPU-hour cap on the chosen type, `>= 0` (0 = none) | none |

`X-Gpu-Allow-Fallback` and `X-Gpu-Max-Price` without `X-Gpu-Type` are a 400
(`gpu_options_without_gpu`). Other errors: `unknown_gpu`, `bad_gpu_fallback`,
`bad_gpu_max_price`, `gpu_choice_unavailable`.

Signed message (`canonical_fund_message`):

- **No GPU header (unchanged v2 bytes):**
  `cascade-<action>:v2:<hotkey>:<ref>:<ts>:<key_digest>`
- **Any GPU header (v3):**
  `cascade-<action>:v3:<hotkey>:<ref>:<ts>:<key_digest>:gpu=<X-Gpu-Type>;fallback=<X-Gpu-Allow-Fallback>;max_price=<X-Gpu-Max-Price>`

The v3 values are the header strings exactly as sent (stripped; an absent
header is empty). Lane C signs `<target>|<ref>` in the ref field, as before.
Adding, stripping or editing any GPU header breaks the signature.

Reply: the existing body, plus `gpu`, `allow_fallback`, `max_price_per_hour`
and an optional advisory `warning` when a type was chosen. Without a choice,
the reply is unchanged.

Miner CLI: `cascade fund` and `cascade lane-c fund` accept `--gpu SKU`,
`--allow-fallback` and `--max-price USD`, and print the echoed choice and
any `WARNING:`.

## Open

- **Multi-GPU legs.** They need a data-parallel leg worker. Today's worker
  is single-GPU (see Shipped), and renting several GPUs per leg is out of
  scope until that worker exists.
- **`cascade submit` with a key** parks a `pending_reveal` entry with the
  default choice. To choose a type, re-fund with `cascade fund --gpu` once
  the reveal lands.
- **Fallback and the miner's cap.** Whether a fallback rent should also
  honour the miner's `--max-price`. Today it uses the default path's
  operator caps.
