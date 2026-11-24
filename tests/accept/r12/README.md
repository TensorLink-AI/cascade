# R12 acceptance tests: the eval-host provisioning kit

R12 (plan §3.0 R12 row, §10.2 "Launch eval host", §10.4 owner-paste scripts,
§10.5 S4, §5 row 9 S4, T19) builds the kit that brings an eval host into
service and back after a fault: an on-rent check, pinned known_hosts with
endpoint resolution, a venv sync, the bring-up handshake, owner-paste setup
and recovery scripts, and the live-run evidence. Runs in `ci (scripts) + pod
+ owner`. Accept: `scripts/laned/accept R12` (shellcheck + dry-run) and
`--evidence evidence/r12_pod.json` for a live run. Depends on W4 (the code
handshake, `cascade/eval/lane_d_handshake.py`).

These tests are the spec. The builder may touch only `PATHS`:
`scripts/evalhost/`. The common conventions (PATHS syntax, exit codes,
expect-fail right reasons) are in `tests/accept/w0c/README.md`.

## Ground rules (the tests enforce them)

- **Offline, captures only.** Every check parses CAPTURED pod output
  (`nvidia-smi`, `df`, `ssh-keyscan`, `lium ps`, the suite identities, the
  worker hello). The tests put `ssh scp rsync sftp ssh-keyscan lium nvidia-smi
  uv docker curl wget btcli sudo systemctl nc` first on `PATH` as shims that
  record their argv and exit 97; no kit step in these tests may call one.
  `torch` and `bittensor` are unimportable (the kit runs on the 7 GB box).
- **No secrets.** `HIPPIUS_*`, `HF_TOKEN`, `WANDB_API_KEY`, `LIUM_API_KEY`,
  `BT_WALLET_PASSWORD`, `CASCADE_OPS_WEBHOOK_URL` are set to sentinel values
  that must never appear in any output or evidence. No output (and no
  non-comment line of the shell scripts) contains `set -x`/xtrace,
  `printenv`, `cat <file>.env`, `BT_WALLET_PASSWORD`, `coldkey`,
  `.bittensor/wallets` or `PRIVATE KEY`.
- **Host keys are always pinned.** Every `ssh`/`scp`/`rsync` command line the
  kit prints carries `UserKnownHostsFile=<the --known-hosts path, literal>`
  and `StrictHostKeyChecking=yes`. Never `StrictHostKeyChecking=no|accept-new`
  or `UserKnownHostsFile=/dev/null`, anywhere.
- **`--dry-run` everywhere**: it runs nothing external, writes nothing, and the
  same inputs give the same output.

## Files

| file | what |
|---|---|
| `scripts/evalhost/evalhost.py` | the tool, stdlib + `cascade.eval.lane_d_handshake`; run as `python -P scripts/evalhost/evalhost.py <subcommand>` with the repo on `PYTHONPATH` (it may also add its own repo root) |
| `scripts/evalhost/setup_eval_host.sh` | owner-paste bring-up |
| `scripts/evalhost/recover_eval_host.sh` | owner-paste recovery |
| `scripts/evalhost/profiles/bench_gate.toml` | the bench gate host |
| `scripts/evalhost/profiles/launch_eval.toml` | D11's launch eval host |

Shared code may live in other files under `scripts/evalhost/`. Every `*.sh`
there must be `bash -n` and `shellcheck -x` clean (the shellcheck test skips
with an explicit reason when `shellcheck` isn't installed; the ubuntu-latest
CI runner has it).

## Exit codes (typed)

| code | meaning |
|---|---|
| 0 | ok |
| 1 | `evidence`: the pod did not pass |
| 2 | usage or unparseable input (bad capture, bad profile, missing input) |
| 10 / 11 / 12 / 13 | on-rent: `wrong_gpu` / `vram_low` / `driver_low` / `disk_low` |
| 20 / 21 | pin-host-key: `host_key_changed` / `fingerprint_mismatch` |
| 30 / 31 / 32 / 33 / 34 | resolve-endpoint: `same_pod_repin` / `endpoint_key_changed` / `not_found` / `not_running` / `endpoint_changed_unverified` |
| 40 / 41 | handshake: `suite_identity_mismatch` / `code_mismatch` |

Every subcommand prints one JSON object on stdout (except `sync`, below).

## Profiles (`laned_evalhost_profile/1`, TOML)

```toml
schema = "laned_evalhost_profile/1"
name = "bench_gate"
gpu_model = "<exact nvidia-smi name>"
min_vram_mib = 97000            # per GPU, nvidia-smi memory.total (MiB)
min_driver = "580"              # dotted version, compared numerically
disk_path = "/data"
min_disk_free_bytes = 9200000000   # two checkpoints + headroom
bench_files = 17747             # optional
[suite_identities]              # optional
gift = "30841734"
boom = "69325b54"
time = "83e3d0b3"
fev = "6de245af"
```

A wrong `schema` or a missing key is a usage error (2). The shipped profiles
pin real history (`test_r12_profiles.py`):

- **`bench_gate.toml`**: `gpu_model` contains `RTX PRO 6000 Blackwell`
  (the Ada fallback differs at 1e-8 and must not share a gate);
  `min_vram_mib` in [90000, 97887]; `min_driver` in [580, 580.173.02] (both
  580.173.02 and 595.99.02 ran; 570/535 could not bench);
  `min_disk_free_bytes` in [9.2e9, 14.8e9] (OPSLOG 2026-11-22: 5.7 GB free
  is the failure class, 14.8 GB free was healthy); the four suite identities
  above; `bench_files = 17747`. It accepts both known-good pods and refuses the
  Ada, the 4090, driver 570/535 and the 5.7 GB disk.
- **`launch_eval.toml`** (D11: valpriv-4090, pinned torch 2.4.1+cu124):
  `gpu_model` contains `RTX 4090`, never `Blackwell` (sm_120 cannot run the
  pinned torch); `min_vram_mib` in [24000, 24564];
  `min_disk_free_bytes ≥ 9.2e9`; no suite identities.

## `on-rent-check`

```
evalhost.py on-rent-check --profile P --nvidia-smi-L F --nvidia-smi-query F --df F
evalhost.py on-rent-check --profile P --dry-run
```

Captures:

- `nvidia-smi -L`: `GPU <i>: <name> (UUID: GPU-…)` per GPU.
- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`:
  `<name>, <MiB>, <driver>` per GPU.
- `df -B1 --output=avail,target <disk_path>`: a header line, then
  `<avail bytes> <mount>`.

Output: `{"ok", "reasons": [{"code", "detail"}], "observed": {"gpu_models":
[…], "vram_mib": [ints], "driver": str, "disk_free_bytes": int}, "required":
{"gpu_model", "min_vram_mib", "min_driver", "min_disk_free_bytes"}}`.

- `wrong_gpu`: no GPU listed, or any GPU in either capture whose name is not
  **exactly** `gpu_model` (the list and the query must agree).
- `vram_low`: any GPU below `min_vram_mib`. `driver_low`: the driver below
  `min_driver`, compared as integer tuples (`580.173.02 ≥ 580.95`).
  `disk_low`: avail below `min_disk_free_bytes`. All limits are inclusive.
- `reasons` lists every failing code in the order wrong_gpu, vram_low,
  driver_low, disk_low, each with a non-empty `detail`; the exit code is the
  first one's. An unparseable capture is a usage error (2), never a pass.
- `--dry-run` prints `{"dry_run": true, "would_run": [...]}` holding exactly
  the three capture commands above (`df` with the profile's `disk_path`).

## `pin-host-key`

```
evalhost.py pin-host-key --known-hosts F --host H --port N --scanned F
    [--fingerprint SHA256:…] [--previous HOST:PORT] [--dry-run]
```

`--scanned` is captured `ssh-keyscan -p N H` output (`#` comment lines
ignored; no key is a usage error, 2). Entries use the OpenSSH host pattern:
`[H]:N`, or bare `H` for port 22; hashed entries (`|1|salt|hmac`) are
recognised. Fingerprints are OpenSSH's `SHA256:<base64, no padding>` of the
key blob. Output: `{"action", "fingerprints" (scanned, sorted),
"pinned_fingerprints" (already pinned for H:N, sorted), "reason"?,
"dry_run"}`.

- No entry for H:N: appends the scanned line(s) (`added`), creating the file
  if needed and keeping existing bytes as a prefix.
- The same key: `unchanged`, the file byte-identical (idempotent).
- A different key for a key type already pinned: `refused`,
  `reason: host_key_changed`, exit 20, the file untouched.
- `--fingerprint` not among the scanned keys: `refused`, exit 21, nothing
  written.
- `--previous HOST:PORT` (the Lium NAT case): the scanned key must equal a key
  pinned for the previous endpoint; then `repinned` (new line added, old line
  kept), else exit 20. A previous endpoint with nothing pinned is refused
  (non-zero), nothing written.
- `--dry-run` reports the same action and exit code and writes nothing.

## `resolve-endpoint`

```
evalhost.py resolve-endpoint --lium-ps F --pod NAME --configured HOST:PORT
    [--known-hosts F --scanned F]
evalhost.py resolve-endpoint --pod NAME --configured HOST:PORT --dry-run
```

`--lium-ps` is captured `lium ps` JSON: an array of pods, each with `huid`,
`name`, `status` and either `ssh_cmd` (`ssh <user>@<host> -p <port>`) or
`host` + `ssh_port`. The pod matches `--pod` by `name` or `huid`. Output:
`{"verdict", "host", "port", "endpoint_changed", "previous" (the configured
endpoint), "fingerprints" (scanned, when given)}`.

| verdict | exit | when |
|---|---|---|
| `same_endpoint` | 0 | RUNNING at the configured host:port (and, with a scan, the pinned key) |
| `same_pod_repin` | 30 | the endpoint moved and the key scanned at the new endpoint equals the key pinned for the configured one |
| `endpoint_key_changed` | 31 | a scan whose key differs from the pinned key (moved or not) |
| `not_found` | 32 | no pod with that name/huid: only now may it be called gone |
| `not_running` | 33 | listed but not `RUNNING` |
| `endpoint_changed_unverified` | 34 | the endpoint moved and no scan was given |

`--dry-run`: `{"dry_run": true, "would_run": [...]}` with a `lium ps …`
command.

## `sync`

```
evalhost.py sync --release-dir D --host USER@HOST --port N --known-hosts F --dest DIR --dry-run
```

The dry-run prints plain lines: one `COPY <relpath>` per file, sorted and
unique, then `RUN <command>` lines. The copied set is **exactly**:

- `pyproject.toml`, `uv.lock` (both required: otherwise exit 2, no COPY),
  `.python-version` when present;
- every regular file under `cascade/` and `benchmarks/` (the bench code),
  except: symlinks (never followed); any path component `__pycache__`,
  `.venv` or `.git`; `*.pyc`/`*.pyo`; and secret-named files anywhere —
  basename `.env`, `.env.*`, `*.env`, `*.pem`, `*.key`, `id_rsa*`,
  `id_ecdsa*`, `id_ed25519*`, `known_hosts*`, `.netrc`, or containing
  `secret` or `credential` (case-insensitive).

- (#784) the project build's own inputs, DERIVED from the release's
  `pyproject.toml`: `[project] readme` (a path or a table's `file`) and every
  source of `[tool.hatch.build.targets.wheel.force-include]` (on this repo
  `README.md`, `chain.toml`, `chain.testnet.toml`, `docs/AGENT_MINING.md`).
  Each must be a regular file inside the release, not secret-named; a missing
  one is exit 2 with no COPY (its `uv sync --frozen` would fail on the host).

Nothing else (`.env`, other docs, tests, scripts, wallets). A file's
contents are never printed. The `RUN` lines hold a pinned `rsync`/`scp`
transfer to `--dest`, then `uv sync --frozen` on the host (never `uv lock`,
`uv add`, `uv pip`, or `uv sync` without `--frozen`); every ssh/rsync/scp line
names the port (`-p N`) and pins the host key. On this repo the set covers
every tracked non-symlink file under `cascade/` and `benchmarks/`.

`--extra NAME` (repeatable; only an `[project.optional-dependencies]` group of
the release's pyproject, else exit 2) plans `uv sync --frozen --extra NAME`.
A profile's optional `uv_extras` list (`profile-info --get uv_extras`, space
separated) is what the owner scripts pass: `launch_eval.toml` has
`uv_extras = ["train"]` (the pinned torch 2.4.1+cu124 lives in that extra;
#784: a plain sync left the eval host without torch), `bench_gate.toml` none.

## `handshake`

```
evalhost.py handshake --profile P [--identities F] [--hello F --release-dir D] [--dry-run]
```

- `--identities`: the pod's captured suite identities,
  `{"suites": {"gift": "<hex>", …}, "bench_files": N}`. Each pinned suite
  matches when its observed hex (≥ 8 chars, lowercase) starts with the 8-hex
  pin. A missing suite or a short value is a mismatch; when the profile pins
  `bench_files`, a different or missing count is the mismatch `bench_files`.
  Required when the profile has `suite_identities` (else 2).
- `--hello`: the pod's captured `worker_hello()` JSON (W4). It is checked with
  `cascade.eval.lane_d_handshake.check_handshake(hello,
  expected=local_identity(<release-dir>))`; a mismatch carries W4's
  `HandshakeMismatch.fields`. A non-object hello fails closed (41).
  A profile without `suite_identities` needs `--hello` (else 2).
- Output: `{"ok", "mismatched": [sorted suite names and/or "bench_files"],
  "suites": {suite: {"pinned", "observed", "ok"}}, "code": null | {"ok",
  "fields": [...]}}`. Both checks run when both inputs are given; exit 40
  when any suite mismatches, else 41 when the code does, else 0.
- `--dry-run`: `{"dry_run": true, "would_run": [...]}`.

## `evidence`

```
evalhost.py evidence --profile P --pod NAME --host H --port N --captures DIR --out FILE
    [--release-dir D] [--dry-run]
```

`--captures` holds `nvidia_smi_L.txt`, `nvidia_smi_query.txt`, `df.txt`,
`ssh_keyscan.txt` and, when present, `identities.json` and `hello.json`. It
re-runs the checks above and prints the evidence JSON; without `--dry-run` it
also writes it to `--out` (the same object). Exit 0 when `passed`, else 1.
`--dry-run` writes nothing.

```json
{"schema": "laned_r12_pod/1", "wp": "R12", "source": "pod",
 "checked_at": "<ISO 8601 with offset>",
 "tool": {"git_sha": "<40 hex HEAD of the kit's repo>", "git_dirty": false},
 "profile": "scripts/evalhost/profiles/<name>.toml",
 "profile_sha256": "<sha256 of the profile bytes>",
 "pod": {"name": "valeval-blackwell4", "host": "<ip>", "port": 41100},
 "captures": {"nvidia_smi_L": "…", "nvidia_smi_query": "…", "df": "…",
              "ssh_keyscan": "…", "identities": {…} | null, "hello": {…} | null},
 "on_rent": <on-rent-check output>,
 "host_key": {"fingerprints": ["SHA256:…"]},
 "handshake": <handshake output>,
 "passed": true}
```

`profile` is recorded repo-relative. `passed` = on-rent ok, at least one
scanned key for `[host]:port`, and the handshake ok.

## The verifier (`helpers.evidence_problems`)

It rejects the file unless: every key above is present; `schema`, `wp`,
`source: "pod"`, `checked_at` with an offset; `tool.git_sha` is a commit of
this repository containing `scripts/evalhost/evalhost.py` and `git_dirty` is
false; `profile` is under `scripts/evalhost/profiles/` at that commit and
`profile_sha256` matches it; and, re-derived from the recorded captures by
the kit with that committed profile: `on_rent.ok`, its reason codes and
`observed`; `host_key.fingerprints` (computed from `ssh_keyscan` for the
pod's endpoint); the suite-identity verdict; and, when a hello is recorded
(required for a profile without suite pins), `hello.git_head` is a commit
here whose `cascade/` tree (`git archive`) has W4's `package_hash` equal to
`hello.package_hash`. `passed` must equal the re-derived verdict and be true.
No sentinel or live secret value and no private key may appear anywhere.

## The owner-paste scripts

```
setup_eval_host.sh   --profile P --pod NAME --host USER@HOST --port N \
    --known-hosts F --release-dir D --work-dir W  (--dry-run | --print)
recover_eval_host.sh --profile P --pod NAME --configured HOST:PORT --user USER \
    --known-hosts F --release-dir D --work-dir W  (--dry-run | --print)
```

- Bash, `#!/usr/bin/env bash`, `set -euo pipefail`; `--help` exits 0 and
  names `--dry-run` and `--print`; a missing required flag exits non-zero.
- `--dry-run` prints `PLAN <step> <command>` lines; `--print` prints a
  self-contained bash block (`set -euo pipefail`, every value expanded to its
  literal, `bash -n` clean) with a `# STEP <step>` line before each step, for
  the owner to review and paste. Neither runs anything or writes any file;
  both are deterministic. Every `evalhost.py` line uses `python -P`.
- **Setup** steps, in order: `known_hosts` (`ssh-keyscan -p N HOST` into
  the work dir, then `evalhost.py pin-host-key`: the key is pinned before any
  ssh), `on_rent` (`nvidia-smi -L`, the query, `df`, then
  `evalhost.py on-rent-check`), `venv_sync` (`evalhost.py sync`),
  `handshake` (`evalhost.py handshake`), `evidence` (`evalhost.py evidence
  … --out evidence/r12_pod.json`).
- **Recovery** steps, in order: `resolve_endpoint` first (`lium ps`, then
  `evalhost.py resolve-endpoint`: never declare the pod dead before it),
  `known_hosts`, `on_rent` (including `df`: the two-checkpoint headroom),
  then `handshake` last among them (`venv_sync` may sit before it).
- Idempotent: keys are pinned only through `pin-host-key`, never appended
  to the known-hosts file directly (`>> <known-hosts>`).
- Never destructive: no `lium rm|down|stop|terminate|delete|kill`, no
  `rm -rf`, `shred` or `mkfs` (disk is reported, freed only on the owner's
  yes; never terminate a pod without an on-pod worker check).

## Evidence run

The owner runs the printed setup block against the rented host; its last
step writes `evidence/r12_pod.json`. `accept R12 --evidence
evidence/r12_pod.json` then exports `LANED_EVIDENCE` and the
`test_evidence_*` tests verify it. Without `LANED_EVIDENCE` they skip.

## Failure on main

`scripts/evalhost/` doesn't exist, so every test fails with an
`AssertionError` naming the missing kit file (`R12 kit file … is missing`),
except the two `test_evidence_*` tests, which skip without
`LANED_EVIDENCE`. Run (prod box, offline, nothing heavy):
`nice -n 19 env PYTHONPATH=<wt> /root/cascade-v2/.venv/bin/python -P -m pytest --noconftest -p no:cacheprovider -m 'not torch' -q tests/accept/r12/`
