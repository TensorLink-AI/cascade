# R8 acceptance tests: the contract they pin

R8 (plan §3.0 R8 row, §3 "Blast radius by path", §10.5 S5, §10.8) extends
`scripts/deploy_gate.py` to the two new lane D services. The tests also pin
the CI-checkable part of T19 and T28 (eval host kept in step with the
validator). R8 owns no invariants. The builder may touch only the paths in
`PATHS`: `scripts/deploy_gate.py` and, if existing unit tests have to change,
`tests/unit/test_deploy_gate.py`.

The gate is loaded by file path (`importlib`) and its CLI runs as
`python -P scripts/deploy_gate.py ...` with `cwd = tmp_path`. `--repo` and
`--root` always point at throwaway dirs in `tmp_path`, so nothing reads the
production box. `verify` scans `/proc` read-only, as it does today.

## 1. `--service intake-laned|eval-host`

- `check --service S --from A --to B --repo R --root DIR` accepts
  `S ∈ {trainer, validator, intake-laned, eval-host}` with the **same flags**
  as today. It exits 0 (GO) or 1 (REFUSE), never 2. The first stdout line
  starts with `"<S> "` and lists areas as `name(count)`. The last line is
  `GO` or `REFUSE`. An empty `--root` has no chain clock, so it prints
  `REFUSE [clock]`.
- A diff that adds `cascade/lane_d/intake.py` shows `lane_d_intake(1)` on
  that first line.
- `verify --service S --release DIR --root DIR` accepts the same four
  services. When no such process exists, it exits 1 with a `VERIFY FAIL`
  line, never a usage error (2) and never a traceback. How `verify` finds the
  eval-host or intake process is the builder's choice.

## 2. Guards per service (`gate(...)`, same signature as today)

- `intake-laned` and `eval-host` are refused with rule `scoring` while the
  validator is still judging a round. §10.5 S5 says every restart happens
  outside scoring windows, and the eval host serves that scoring. Once the
  round's `weights set` line appears, the refusal clears.
- The settlement and clock guards apply to both new services unchanged.
- For `--service trainer|validator`, a diff in `cascade/lane_d/` adds a note
  containing `intake-laned`: the intake service must be deployed as well.
- T19/T28: a `--service validator` diff touching
  `cascade/validator/private_worker.py` (the code the eval host runs) adds a
  note containing `eval-host`: the eval host must be synced, eval host first.

## 2b. The settlement blackout and the block clock (#787, `test_r8_settlement_scaling.py`)

- The blackout scales with the settlement length in force at the
  extrapolated block (`Grid.length_at`): mainnet's 3 h (900-block)
  settlement keeps exactly 20 min before and 15 min after a boundary (pinned
  to the block: 726 of 900 blocks open); a longer one (12 h) keeps the same
  minutes (capped); a shorter one gets the same fractions (testnet's 150
  blocks: 3.3 / 2.5 min), so a restart window always exists.
- The block clock is the NEWEST sample of the validator log and
  `trainer.log`; a sample older than 120 min is `REFUSE [clock]`.
- `check --validator-log FILE` names the validator's log (default
  `<root>/validator.log`; testnet's main validator writes `val0.log`).

## 3. AREAs

- **(i) `lane_d_intake`**: a new key in `AREAS` that matches everything under
  `^cascade/lane_d/`. It does **not** match `cascade/validator/lane_d*.py`,
  `cascade/eval/lane_d_scorer.py`, `cascade/shared/config/lane_d.py` or
  `cascade/lane_d_other.py`.
- **Decision: `ops_tooling`**: the rollout tooling package `cascade/laned/`
  (R1/R6/R2 tools, not on main yet) classifies into a new AREA `ops_tooling`.
  It is not `consensus` and not `lane_d_intake`. It is an AREA rather than an
  allow-list entry because it is executable code that the owner runs.
- **(ii) Every path classifies.** Every path from `git ls-files` (run at the
  repo root) matches at least one AREA, or this test's own allow-list. The
  allow-list lives in the test, so a builder can't widen it. It holds docs
  and tests only: `^docs/`, `^decisions/`, any `*.md`, `LICENSE`/`NOTICE`,
  `llms.txt`, `^tests/`, `^<dir>/tests/` and `.gitignore`. Today 325 tracked
  code and config paths fall outside both. Examples are `cascade/miner/`,
  `cascade/pool/`, `deploy/`, `scripts/`, `benchmarks/`, `base_generator/`,
  `champions/`, `.github/workflows/`, `.dockerignore`, `.python-version` and
  `decided_votes.toml`. The builder chooses the AREA names for them.
- **(iii) `WORKER_IMAGE` covers the on-pod import closure.** The test computes
  the closure itself with a static AST walk from `cascade/trainer/worker.py`.
  The walk:
  - follows absolute and relative `cascade.*` imports;
  - includes lazy imports inside functions (this is how
    `expert_training.py` is reached, via `moe_trainer`/`pyramid_trainer`);
  - includes every parent package `__init__.py`;
  - skips `if TYPE_CHECKING:` bodies.

  Every file in the closure must match a `WORKER_IMAGE` pattern. The closure
  over-approximates `worker.py --role expert`, which is the safe direction.
  Today it is 271 files across most of `cascade/`, because `worker.py`
  imports `trainer.main`. That matches reality: the Dockerfile bakes the
  whole repo (`COPY . /root/cascade`), and `worker_code_fingerprint` hashes
  every `cascade/**/*.py`. The existing `WORKER_IMAGE` paths
  (`deploy/Dockerfile`, `worker.py`, `cascade/interface/`, `uv.lock`,
  `pyproject.toml`) stay.

## Failure on main (expected, all `AssertionError`)

17 failed and 17 passed. The tests for the existing services, the negative
classification checks and the closure sanity check pass. The failures:

- argparse rejects `intake-laned`/`eval-host` on `check` and `verify`;
- there is no `scoring` refusal and no `intake-laned`/`eval-host` notes;
- there is no `lane_d_intake` or `ops_tooling` AREA;
- 325 paths are unclassified;
- 265 of the 271 closure files are outside `WORKER_IMAGE`.
