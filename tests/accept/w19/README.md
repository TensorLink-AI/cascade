# W19 acceptance tests: lane D work runs off the weight/poll thread

W19 (plan rev6 §3.0 W19 row, §3.2 invariant 6, §4 T15) depends on W5e. These
tests are the spec. The builder may touch only the paths in `PATHS`.

## What the plan says

The W19 row, verbatim:

> | W19 | Lane D work runs **off the weight/poll thread**; the weight-loop keep-alive (#471) is in the release (invariant 6, T15) | W5e | ci + testnet | `scripts/laned/accept W19` (a fixture slow lane D step leaves weight-set timing unchanged) |

Invariant 6 (§3.2), verbatim:

> 6. **Weights never wait for lane D** (off the poll thread; keep-alive). **(rev3)** #471 is merged (0c52d07) but not deployed (rc37 lacks it); it ships with S2b.
>    (W19, X3 T15)

T15 (§4), verbatim:

> | T15 | 2 h lane D sweep across weight intervals | No weight gap > `weight_set_interval_blocks` |

§1.6 capacity row, verbatim: "| Weight loop | must never wait on lane D |
Lane D runs off the poll thread; keep-alive (#471) live before S6 (T15)."
The rev6.2 W8b row (proposed) says the production wiring of
`settle_lane_d` relies on this: "lane D's bench work runs off the weight
loop (W19) or is bounded so weight reasserts are never delayed past a
stated bound (2026-11-22: weights went stale ≥ 2 h during a synchronous
king-check sweep)."

## Today's code (main)

- `cascade/validator/loop.py` `run_forever` (`:1105-1115`) is one thread:
  `_poll_once` → `_reassert_weights_at_head` → `_vbench_lease_tick` →
  `sleep(poll_seconds)`. Any step added inside that loop delays the next
  weight re-assert by its own duration.
- `cascade/validator/weights.py`: `_maybe_reassert_weights` re-pushes every
  `weight_set_interval_blocks` (180); `_weights_keepalive_tick` (`:317`, #471)
  re-pushes the last landed vector during a long blocking step, and
  `validator_bench.py:901` passes it as the offloaded sweep's `on_wait`.
- Nothing calls `settle_lane_d` outside tests yet (W8b, rev6.2 proposed,
  wires it). So W19 provides the off-thread seam W8b plugs the settlement
  into.

## Contract

### `cascade/validator/lane_d_worker.py`: `LaneDWorker(step)`

`step` is a callable `step(boundary: int)`; its return value is ignored.

- `submit(boundary) -> bool`: if no step is running and `boundary` has not
  completed, starts `step(boundary)` on a new **daemon** thread (never the
  caller's) and returns `True` **without waiting for it**; otherwise returns
  `False` and starts nothing (one step at a time, no pile-up). Never raises
  because of the step.
- `running -> bool`: a step is in flight.
- `join(timeout=None) -> bool`: waits at most `timeout` seconds; `True` iff
  no step is in flight afterwards (immediately `True` when idle).
- A boundary whose step **returned** is completed and never run again by
  this worker. A boundary whose step **raised** is not completed (fail-closed:
  retried at a later submit, W8b "retry next tick"); the exception is kept as
  `last_error`, which a later successful step resets to `None`.

### `ValidatorRunner` (`cascade/validator/loop.py`)

- A dataclass field `lane_d_step: Callable[[int], object] | None = None`:
  the lane D settlement step (W8b wires `settle_lane_d` here; tests inject
  fixtures). `None` ⇒ no lane D work at all.
- `_lane_d_tick(client) -> None`: called by `run_forever` every poll (the
  poll loop calls `self._lane_d_tick(...)` next to
  `self._reassert_weights_at_head(...)`). It returns promptly and never
  raises. When `lane_d_step` is set and the current settlement boundary
  (`_reassert_boundary(current_block)`, the grid in force) is armed
  (`cfg.lane_d.active(boundary)`; the settlement AT `from_block` belongs to
  the era before), it submits that boundary to the runner's `LaneDWorker`,
  kept as `self._lane_d_worker`. Unarmed (`from_block = 0`) or with no step:
  nothing is submitted and no thread starts.
- Weight pushes stay on the poll thread: the step's thread never pushes
  weights through the loop.

### Weight timing (the Accept fixture)

With a lane D step that hangs (or a real W3b2/W5e `SweepSession` whose bench
host hangs: the 2 h lane D bench sweep), driving the poll loop's lane D tick
and head re-assert once per poll (10 blocks, `poll_seconds = 120`) across
two armed settlements:

- the blocks at which weights are pushed are **identical** to the same loop
  with no lane D step (baseline);
- every poll finishes within 1 s of wall clock;
- no gap between pushes exceeds `weight_set_interval_blocks` (on the poll
  grid: ≤ interval + one poll − 1 block), across > 2 h of blocks (T15);
- a raising step never reaches the weight loop (timing still identical);
- the step for B1 runs exactly once while hung (B2's tick starts nothing);
  once it ends, the next armed boundary's step starts.

### The keep-alive (#471) stays in the release

`WeightsMixin._weights_keepalive_tick` exists on `ValidatorRunner`, re-pushes
the last landed vector once the interval has passed (not before), never
raises before the first push, and `validator_bench.py` passes it as the bench
sweep's `on_wait`. These pass on main; they pin that W19 does not lose it.

### Unarmed (invariant 1) and its pair (invariant 10)

`test_weight_timing_unchanged_when_unarmed` (unarmed with a step injected:
the step is never called, no thread starts, weight timing is the baseline) is
paired with `test_weight_timing_armed_nonempty_lane_d_effect` (armed: the
step runs once per armed boundary off the poll thread, weight timing is still
the baseline). Every invariant 6 test is named `test_inv6_*`.

## Fixtures and services

The chain is the only external service: `helpers.FakeChain` is the
`ChainClient` duck type the weight loop calls (block clock, metagraph,
`set_equal_share_weights` / `set_weight_vector`), no real chain and no
signing. The validator is a real `ValidatorRunner` over the shipped
`chain.toml` (loader parse; `[lane_d]` armed at block 9,720,000 for the armed
cases), with `state_db_path` in the test's tmp dir so nothing outside it is
read or written. Hung fixtures are released at teardown.

On main the runner tests fail with `AssertionError` (no `lane_d_step` field /
`_lane_d_tick`), the worker tests with `ModuleNotFoundError` for
`cascade.validator.lane_d_worker` (in `PATHS`), the `run_forever` check with
`AssertionError`; the keep-alive tests pass.

### Not pinned here (flagged)

- Which chain client the step uses. The weight loop's client is driven from
  the poll thread; W8b's wiring decides how the step reads the boundary
  block hash, and it should not share a non-thread-safe websocket with the
  poll thread.
- Catching up a boundary missed while a step was still running is W8b's
  (restart-safe, idempotent settlement is W7c's).
- The testnet half of `Runs in` (`ci + testnet`) is X3's T15 row.

## PATHS

- `cascade/validator/lane_d_worker.py` (new): the runner.
- `cascade/validator/loop.py`: the field, `_lane_d_tick` and the call in
  `run_forever`'s poll loop (`:1105-1115`).
