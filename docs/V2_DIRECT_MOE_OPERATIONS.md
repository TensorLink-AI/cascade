# Direct runtime operations

The separate direct runtime uses payer-funded Lium, Vast.ai or RunPod pods,
SSH dispatch and harvest, Hippius storage, a signed journal and receipt-replay
validators. It is distinct from rolling A/B/C funding. Live activation is
refused by `DirectConfig.validate()` in this release. This guide describes
configuration, isolated operation and recovery; the [protocol guide](V2_DIRECT_MOE.md)
explains job rules and miner commands.

**Read by task:** [local validation](#run-the-pinned-validation) ·
[configuration and staging](#runtime-and-staging-commands) ·
[dataset provenance](#fev-provenance) · [recovery](#recovery-and-invariants).
The commands below include state changes and paid operations; choose only the
steps authorized for your isolated test. They do not open rolling A/B/C mining.

## Run the pinned validation

From the repository root:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/cascade-v2-pinned uv sync --frozen \
  --extra train --extra dev --extra chain --extra hippius --extra deploy --extra pool-forge
CASCADE_SANDBOX_TEST_IMAGE=cascade-miner:harness PYTHONDONTWRITEBYTECODE=1 /tmp/cascade-v2-pinned/bin/python -m pytest \
  tests/unit tests/integration/test_direct_torch.py tests/integration/test_direct_runtime.py \
  -q -rs -p no:cacheprovider
UV_PROJECT_ENVIRONMENT=/tmp/cascade-v2-bench uv sync --project benchmarks --frozen --extra fev --extra time
```

The benchmark tests need pytest and its dependencies, at the versions in the root
`uv.lock`. Install those exact test-only pins without changing the benchmark runtime:

```sh
python3 - <<'PYTEST_PINS'
import tomllib
packages = tomllib.load(open("uv.lock", "rb"))["package"]
names = {"pytest", "iniconfig", "packaging", "pluggy", "pygments"}
with open("/tmp/direct-bench-test-requirements.txt", "w") as output:
    output.write("\n".join(p["name"] + "==" + p["version"] for p in packages if p["name"] in names))
PYTEST_PINS
uv pip install --python /tmp/cascade-v2-bench/bin/python -r /tmp/direct-bench-test-requirements.txt
```

Then run:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/cascade-v2-bench/bin/python -m pytest benchmarks/tests -q -p no:cacheprovider
```

The command above uses the existing local `cascade-miner:harness` image. On another
host, supply a suitable existing Python+NumPy sandbox image; absence is a skip, not
an isolation pass. Production worker images must be pinned/qualified separately.

The distributed test launches two real Gloo processes, executes the MoE forward,
backward and optimizer steps, kills a rank, and compares resumed weight, optimizer
and router tensors exactly with uninterrupted execution. Its model is intentionally
tiny and is **not a competing proxy model**. The single-CUDA test runs only when
existing hardware is available and explicitly skips otherwise. The lifecycle test
uses actual trained checkpoints/forecasts, two founders, current-king rescoring,
promotion, an exactly-once **local finality fixture**, and the next warm-start block.
Its permissive private threshold and public panel values are test inputs, not policy.

Native FEV validation loads a local Parquet dataset through installed FEV and scores
an actual MoE checkpoint. It is not the full FEV benchmark. The four native adapters
remain subject to the full pinned benchmark dataset sweep before release.

## Runtime and staging commands

The runtime is `cascade.direct`; `cascade-trainer --direct-runtime-config FILE`
invokes the same scheduler. It is separate from the historical rolling competition.
The old `--direct-simulation` command remains explicitly a counter simulation.

An operator must supply `/etc/cascade/direct.json` with these required objects:

- `state_dir`, `poll_seconds`, `ledger_policy` (the exact validated fields in
  `funding/direct_ledger.initial_state`), and `bootstrap_payer`.
- `storage`: `kind=hippius`, bucket and prefix (endpoint/region default to Hippius
  S3; credentials from `HIPPIUS_S3_ACCESS_KEY`/`HIPPIUS_S3_SECRET_KEY` on the
  orchestrator only) and `public_read: true` so auditors can fetch checkpoints to
  reproduce scores; or `kind=file,root=...` for local CPU validation. Upload the
  private panel with a credentialed client WITHOUT a public ACL — never through this
  store while `public_read` is on.
- `provider`: `kind=pods`, `computes` (the marketplaces offered to miners: any of
  `lium`, `vast`, `runpod`), `netuid`, `ssh_pubkey_file` (the orchestrator key the
  worker image installs from `$SSH_PUBKEY`) and named `PodProfile` objects: SKU,
  gang size (`world_size`, rented as ONE pod with that many GPUs), GPU name, memory,
  precision, hourly ceiling in micro-USD, maximum seconds, digest-pinned worker image
  and measured qualification digest. Profiles are shared across marketplaces.
- `vault_dir` and `vault_ttl_hours`: the intake's payer vault. Lium keys live in the
  root (the rolling lane's vault, unchanged); Vast and RunPod keys in `vast/` and
  `runpod/` subdirectories. Seal with `CASCADE_VAULT_KEY_FILE` on intake and scheduler.
- `dispatch`: `remote_python` and `workdir` inside the worker image, and `key_path`
  (the orchestrator's private SSH key). Pods receive no credentials: inputs are
  staged over SSH and checkpoints harvested to Hippius by the orchestrator.
- `journal`: Hippius S3 `bucket` the signed journal is published to (`--publish`).
- `evaluation`: guard contract with all four suites/cases/source groups and explicit
  tolerances; private policy; content-addressed private panel and case metadata;
  initial artifact map; sample count, seed, device; and `sidecar` (benchmark Python,
  data directory, exact revisions, timeout, device, pinned FEV manifest).
  `fixture_panels` is for local validation only and cannot count as native evidence.
- `trainer_wallet`: explicit wallet name/hotkey/path for signed journal export.
  Configure the corresponding public trainer hotkey separately in validator/audit.

Rewards are paid through **emissions**, never transfers: `--payout UNITS` records a
payout boundary and settles it in the journal (throne/royalties per the reward
policy; the continuation pool split pro rata across earned, unpaid entitlements;
the rest burns). Validators derive the identical weight vector from the journal.

Training specs pin the full architecture/config/source/parameter counts, complete
optimizer config, objective, precision, seed, data source, context/channels/schedule,
world size, microbatch, exact scalar budget, checkpoint cadence and implementation
hash (`cascade.direct.fingerprint.implementation_digest()`). Founder initialization
is CPU-seeded on every hardware profile. Later blocks must use the actual promoted
weights **and optimizer**. Specialist specs additionally pin selection/domain and
accepted initial state; selection/provenance uses `cascade.direct.experts.select`.

With these files supplied, the following are the exact preparation/recovery commands:

```sh
/opt/cascade/.venv/bin/python -m cascade.direct.initialize --spec /etc/cascade/founder-work.json --output /var/lib/cascade/genesis.json
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --cohort /etc/cascade/cohort.json
/opt/cascade/.venv/bin/cascade-intake --direct-config /etc/cascade/direct.json --host 127.0.0.1 --port 8080
# Miners: `cascade direct deposit-key` (POST /v2/direct/credentials) files a
# lium|vast|runpod key, then `cascade direct fund` posts a signed fund-job envelope
# to /v2/direct/fund whose terms name that provider (see V2_DIRECT_MOE.md).
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --bind /etc/cascade/admitted-work.json
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --status
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --serve
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --resume JOB --evidence INCIDENT_REFERENCE
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --settle JOB
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --advance JOB
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --merge EXPERT_JOB_1 EXPERT_JOB_2 --optimizer-policy reset_edited_rows
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --payout EMISSION_UNITS
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --publish
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --export /var/lib/cascade/receipt.json
/opt/cascade/.venv/bin/cascade-audit direct-moe /var/lib/cascade/receipt.json --trainer-hotkey TRAINER_SS58
# Owner (main validator): sign + publish the next network config version (rules.json below).
/opt/cascade/.venv/bin/cascade-validator --publish-direct-config rules.json --direct-config-bucket CONFIG_BUCKET --wallet-name OWNER_W --wallet-hotkey OWNER_H
# Validators: fetch the owner config, verify + replay the trainer's scores against it
# (no evaluation) and set weights. The two flags below are all they ever configure.
/opt/cascade/.venv/bin/cascade-validator --direct-owner-hotkey OWNER_SS58 --direct-config-bucket CONFIG_BUCKET --wallet-name W --wallet-hotkey H
# (Without an owner config: --direct-journal-bucket BUCKET --direct-trainer-hotkey TRAINER_SS58.)
# After rotating `evaluation.private_panel` to a fresh panel, reveal the retired one:
/opt/cascade/.venv/bin/python -m cascade.direct --config /etc/cascade/direct.json --reveal-panel /etc/cascade/retired-panel.json
# Anyone: reproduce every recorded score from the public checkpoints (GPU recommended).
cascade-audit direct-moe --journal-bucket BUCKET --trainer-hotkey TRAINER_SS58 --reproduce [--sidecar sidecar.json] [--device cuda]
```

**Owner network config (DEC-CA-0052 §8).** The main validator owns every rule
the other validators apply. `rules.json` holds exactly `trainer` (journal signer
ss58), `journal` (`{"bucket", "endpoint"?, "region"?}`), `burn_uid`, `burn_all`
(kill switch: every validator burns), `ledger_policy` (must equal the journal's),
`reward_policy` (or null), `guards` (allowed `digest(GuardContract.to_dict())`),
`private_policies` (allowed `digest(PrivatePolicy)` as recorded in decisions),
`private_panels` (append-only schedule `[{"panel": digest(panel descriptor),
"from_block": N}]`, strictly increasing blocks, one active at a time),
`reveal_after` (seconds) and `poll_seconds`; the publish command adds `protocol`, `seq` and `prev`. Validators
re-read it every poll, refuse an older `seq`, and refuse a journal that used an
unlisted guard/private policy or a different ledger/reward policy (weights stay
as they were). Add a new guard or policy BEFORE the trainer uses it; drop an old
one only after its cohort has rotated out, or the whole journal is refused.
Rotating the trainer hotkey, moving the journal bucket or the burn uid is a
republish: no validator restart and no `chain.toml` edit.

Chain anchor (§9): publishing also commits `cascade-direct:1:<digest>` from the
owner hotkey, which must be a dedicated registered hotkey (the command refuses to
overwrite another commitment, e.g. a readiness note). Validators accept only the
committed config and keep the last one under `--cache-dir`. A new panel must
start at least 10 blocks after the current block when you publish (the command
checks this; validators refuse a panel starting before its commitment block).
The trainer stamps each decision with the chain block it was scored at: set
`"chain": {"network": "finney", "chain_toml": "..."}` in the trainer config.

Auditability (§9): schedule a new private panel in `private_panels` BEFORE the
trainer scores with it (otherwise the journal is refused); never delete old
entries; keep checkpoints
public-read (`storage.public_read`), or a credential-free HEAD fails and the
journal is refused; and rotate + `--reveal-panel` every panel within
`reveal_after` of its first use, or validators burn the boundary until the
reveal lands (they recover on the next poll after it does).

**Rounds (§11).** Start a journal with `round_origin` (a live era start),
`round_blocks` (6 eras = 3 days) and `era_blocks` in its ledger policy and a
`"chain"` block source in the trainer config. Each round: `--advance` the
previous round's winner after the round closes (refused earlier), then open the
next `--cohort` (one per round; reuse the anchor with a new seed when no one
won), and run `--payout` at most once per era. With rounds, the stale burn is
"a full round without a promotion of the king's own"; `stale_after` is unused.

**Stale-king burn (§10).** Set `stale_after` (journal clock, seconds) and
`stale_burn` (e.g. `"9/10"`) in the reward policy: a king whose generator has
not produced a new promotion within `stale_after` keeps only 10% of the throne
share; the rest burns. The reward policy is fixed before the first admission, so
set these when starting the journal.

**Trust model (DEC-CA-0052 §7).** Validators check that the journal replays from
its own scores; reproduction checks that the scores are true. Each decision binds
the checkpoints, architecture, panel digests (or sidecar revisions), samples, seed
and device it was scored with. `--reproduce` reports per decision and half:
PASS (scores within `--tolerance`, verdict identical), FAIL (divergent score or
verdict, mis-bound or unavailable checkpoint; exit 1) or WARN (panel not yet
revealed, no sidecar for the bound revisions, or a pre-binding decision). Rotate
private panels on a schedule so every decision becomes reproducible; a panel is
burned the moment it is revealed.

Install `deploy/direct-v2/cascade-direct.service` only after filling/validating the
configuration and setting state-directory ownership. The service intentionally has
no paid authorization flag. Renting payer-funded pods requires an explicitly
authorized invocation with `--allow-paid`. It does not activate a mainnet contract.
Review the exact configuration and paid limits before authorizing that invocation.

The initializer output's `manifest` must be installed in the evaluation initial-artifact
map under its returned `anchor`; use that anchor in the founding cohort and admissions.
Create the cohort before intake accepts funding. Marketplace keys travel only in the
credentials request and live only in the vault; never in a signed envelope or journal.

## FEV provenance

`cascade_benchmark.direct_cli` requires a JSON manifest with `tasks: {path,sha256}`
and `datasets: [{path,sha256}, ...]`. Tasks must use literal local Parquet/Arrow files,
with no mutable Hub dataset configuration. Hashes are verified before evaluation;
its canonical manifest SHA256 is the required FEV revision. The existing historical
CLI remains unchanged. The benchmark lock now includes FEV 0.10.0; existing dependency
pins were retained. GIFT, TIME and BOOM keep their existing recorded dataset revisions.
The source-group and fixed-case contract must still be approved; pinning a dataset
alone does not establish statistical independence or approved scope.

## Recovery and invariants

- Keep ledger/WAL and the per-marketplace provider journals on durable storage.
  Snapshot them consistently; never delete a provider intent to retry it.
- Marketplaces have no client-token idempotency: the pod NAME derives from the
  reservation key (`cascade-n<netuid>-direct<hex>-<payer slug>`, outside the rolling
  reaper's pattern), a launch is submitted at most once per key, and every later call
  looks the name up. A definitive refusal (auth, sold out, balance) is recorded as
  rejected; an ambiguous one keeps the submission and is reconciled by name; a launch
  that never appears is NOT refunded. The pod's platform id and SSH host key are pinned
  on first sight; a different pod under the same name is attributed to the payer.
- Teardown needs the payer's key (their pod is invisible to ours): the vault TTL must
  outlive every legitimate pod lifetime. A missing key leaves the release pending and
  must alert. Termination is confirmed by re-listing, never by the delete call.
- Worker-host flock plus completion markers prevent duplicate torchrun launches for
  a job/fence; a repeated SSH `start` attaches. Inputs are staged by digest; harvested
  checkpoint objects are re-hashed before upload to Hippius and before the controller
  records the pointer; stale-fence output is refused. The CUDA supervisor enforces the
  lease deadline and the scheduler's `reap_expired` terminates overdue pods.
- Conservative allocation estimates are committed once against the job cap; they are
  not invoices and are not refunded implicitly.
- `--status` and each scheduler tick emit job state, attempts, consumed count, fence,
  failure attribution and next retry. Alert on capacity pause, funding cure, evaluation
  outage, unreconciled reservation, stalled checkpoints and pending teardown.
  Queue scarcity does not erase conditional rights or fabricate a failed experiment.
- Do not change worker implementation or scientific configuration to resume a saved
  treatment. The code fingerprint and generator identity are checked on continuation.
- Payout boundaries settle entitlements in the journal; there is no transfer outbox.
- Expert merge verifies frozen slices, rejects overlap, resets only edited expert
  optimizer rows and scores the combined manifest before recording promotion/rewards.

## Configuration constraints

Supply explicit scientific, hardware, funding and reward policies; fixture
values are not production defaults. Source/image fingerprints and hardware
profiles must match the worker treatment. A complete configuration does not
bypass the closed activation guard or qualify real provider recovery,
checkpoint throughput, benchmark coverage or weight-setting behaviour.
