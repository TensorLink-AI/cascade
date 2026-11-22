# W3 acceptance tests: the lane D intake service

W3 (plan §1.2 "Flow" and "Ownership and dedup", §1.5 "State" B10, §3.0 W3
row, §10.2, §11.2 B13, T2/T6/T6b/T7c/T16/T18d) builds the `cascade/lane_d/`
intake service `cascade-laned-intake`: `POST /v1/lane-d/submit`,
`POST /v1/lane-d/ready` and `GET /v1/lane-d/queue`, behind Caddy on the box.
These tests are the spec. The builder may touch only `PATHS`:
`cascade/lane_d/` (new modules `service.py`, `main.py`, `__main__.py`, plus
any helper modules), `deploy/cascade-laned-intake.service` and
`deploy/caddy/lane-d-intake.caddy`.

`cascade/lane_d/intake.py` and `storage.py` already exist on main and other
WPs import them (W4: `upload_checkpoint`, `FilesystemStorage`,
`fetch_presigned`; W1.5). The builder may reuse and extend them but keeps
every existing public name working.

Conventions follow `tests/accept/w0c/README.md`. No PATHS module is imported
at module level: on main every failure is a missing
`cascade.lane_d.service` / `cascade.lane_d.main` (in `PATHS`) or an
`AssertionError`. The only fakes are external services: the chain reads,
the Hippius grant minter and a failing storage wrapper, each a subclass of
the producer's ABC. The vault is the real fs adapter (`FilesystemStorage`).
The checkpoint "files" are inert bytes. Nothing uploaded is ever imported or
run. There's no network: the entry-point tests run with `torch` and
`bittensor` unimportable and every socket connect refused.

## Upload path (W3 is gated on W3a)

`decisions/STORAGE-LANE-D.md` names the one upload path on its `Chosen path:`
line. **The owner hasn't filled it yet**, so
`test_upload_path_is_the_decision_records_choice` fails until the W3a live
run is recorded. Until then the W3 builder records
`evidence/blocked/w3.md` (`Reason: owner`, W3a). It never picks a path.

- `service.UPLOAD_PATH_IDS == ("presigned_multipart", "single_presigned_put", "upload_relay")`.
- `service.UPLOAD_PATH` is the chosen id, and `service.OFFERED_UPLOAD_PATHS == (UPLOAD_PATH,)`.
- `service.GrantMinter` is an ABC with a class attribute `upload_path` and
  two methods:
  - `mint(prefix, files) -> dict`, where `files` is `{name: size}` for the
    four checkpoint files. It returns JSON-serialisable grants: presigned
    part URLs, PUT URLs or relay tickets, whatever the path needs.
  - `finalize(prefix, files) -> None`, called once at a successful `ready`
    before the object-list check. For multipart this completes the upload,
    since the intake completes it itself. It may raise
    `intake.IntakeError(code)`, which is answered as `409 <code>`.
- **Exactly one** concrete `GrantMinter` subclass ships under `cascade.lane_d`
  (the Hippius one), and its `upload_path == UPLOAD_PATH`.
  `LaneDIntake(minter=...)` raises `ValueError` for a minter of any other
  path.
- A request may declare `X-Upload-Path` (unsigned). Any value other than
  `UPLOAD_PATH` is `400 upload_path_not_offered` and nothing is minted.

## The app

`service.LaneDIntake(*, storage, reads, minter, cfg, ops, state_dir, now,
per_ip_rpm=DEFAULT_PER_IP_RPM, per_hotkey_rpm=DEFAULT_PER_HOTKEY_RPM)` is a
**WSGI callable**. The arguments are:

- `storage`: a `LaneDStorage`;
- `cfg`: a `LaneDConfig`;
- `ops`: a `LaneDOpsConfig` (W2b);
- `now`: a callable returning unix seconds (the deterministic test clock).

`service.LaneDReads` is an ABC: the chain and registry reads the intake makes.
A concrete subclass backed by `ChainClient` ships in `cascade.lane_d`; the
tests only check that it exists.

| method | returns |
|---|---|
| `current_block()` | the chain head |
| `coldkey_of(hotkey)` | the owning coldkey ss58, or `None` iff the hotkey is not registered on the netuid |
| `reveal_block(hotkey, digest)` | the block of the hotkey's lane D reveal of this digest, else `None` |
| `published_weights()` | the weights sha256 set of every published checkpoint (lane A legs and kings, lane D kings) and the current house |

Every response is JSON. A refusal is
`{"code": <code>, "message": ...}` and every code is in
`service.REFUSAL_CODES` (a typed tuple; the test driver checks every 4xx/5xx).
Routes: `POST /v1/lane-d/submit`, `POST /v1/lane-d/ready`,
`GET /v1/lane-d/queue`. Any other path is `404 not_found`; the wrong method is
`405 method_not_allowed`.

## Ownership

Headers: `X-Miner-Hotkey` (ss58), `X-Content-Digest` (the 64 lowercase hex
manifest digest, no `sha256:` prefix), `X-Context-Length` (decimal int),
`X-Timestamp` (unix seconds), `X-Signature` (hex sr25519 by the hotkey).
The signed bytes are `service.canonical_message(action, hotkey, digest,
context, timestamp, body)`:

```
cascade-lane-d-<action>:v1:<hotkey>:<digest>:<context>:<timestamp>:<sha256 hex of the body>
```

Here `action` is `submit` or `ready`, and the signature binds the body (the
manifest, the licence attestation). Verification goes through
`cascade.shared.sr25519.verify_hotkey`. The refusals:

| condition | answer |
|---|---|
| no hotkey | `400 missing_hotkey` |
| digest not 64 hex | `400 bad_digest` |
| context missing or not an int | `400 bad_context` (or `bad_signature`) |
| timestamp outside ±`service.FRESHNESS_SECONDS` (300) or unparseable | `400 stale_timestamp` |
| missing, garbage or wrong-key signature; any signed field changed | `400 bad_signature` |
| timestamp not strictly greater than the last verified one for (action, hotkey) | `409 replayed_timestamp` |
| verifier unavailable (`SignatureVerifierUnavailable`) | `503 signature_unavailable`, never `bad_signature` |
| hotkey unregistered (`coldkey_of` is `None`) | `403 not_registered` |
| `ready` by a hotkey that didn't submit the digest | `403 not_owner` (the uploader must be the committer) |
| `submit` of a digest another hotkey already submitted | `409 digest_owned` |

## Submit

The body is the canonical manifest `{"files": {name: {"sha256", "size"}}}`,
the same form as `intake.canonical_manifest`.

- sha256 of its canonical JSON must equal `X-Content-Digest`, else
  `400 digest_mismatch`.
- Unparseable, not an object, a size that isn't a positive int, or a sha256
  that isn't 64 hex: `400 bad_manifest`.
- The file names must be exactly `intake.ALLOWED_FILES`, else
  `400 file_not_allowed`.
- The context must be in `cfg.contexts`, else `400 context_not_armed`.
- Nothing is minted on any refusal.

On success it answers `201 {"status": "granted", "upload_path", "hotkey",
"digest", "context", "prefix", "grants"}`:

- `prefix` is `f"{ops.vault_prefix}/{hotkey}/{digest}/"`;
- `grants` is the minter's `mint(prefix, {name: size})` output;
- the intake writes the canonical manifest to `prefix + "manifest.json"` in
  storage itself.

Resubmits:

- A resubmit of the same digest before `ready` is granted again, as a retry.
- A different digest before `ready` **replaces** the hotkey's entry, so
  there's at most one record per hotkey. The replaced digest is then
  `404 unknown_entry` at ready.
- While the hotkey has an open entry, a new submit is
  `409 hotkey_entry_open` (T6).

## Dedup (before any eval spend)

Dedup is by the manifest's `weights.safetensors` sha256:

- At submit, it's `409 duplicate_weights` if the sha is in
  `published_weights()` (T2, T16) or equals an **open** entry's weights from
  another hotkey.
- It's re-checked against `published_weights()` at `ready`: a king crowned
  between submit and ready is a `duplicate_weights` too.
- A one-byte perturbation isn't a duplicate; that case belongs to the
  margin.
- An expired entry's weights no longer block.
- If `published_weights()` fails, the answer is `503 dedup_unavailable`.

## Ready

The body is `{"licence": <id>, "attestation": <text>}`; unparseable is
`400 bad_body`. The steps, in order:

1. **B13 licence attestation.** The licence must be one of
   `service.LICENCES == ("Apache-2.0", "OpenMDW-1.1")` (D13), else
   `400 licence_not_open`. The attestation must be exactly
   `service.LICENCE_ATTESTATION`, else `400 licence_attestation_missing`.
   That's one statement of at least 80 characters naming both licences,
   "training data" and restricted-licence "teacher" outputs (plan §11.2).
   The record keeps `licence` and `licence_attestation_sha256` (sha256 of
   the statement's UTF-8).
2. The context must equal the submit's, else `400 context_mismatch`.
3. `finalize`.
4. The object list is checked **without downloading** anything but
   `manifest.json`, giving `409 missing_object`, `409 size_mismatch` or
   `409 stray_object`.
5. **A8:** `coldkey_at_ready = coldkey_of(hotkey)` is read from the chain
   **at ready**, not at submit.
6. `ready_block = current_block()` and
   `ready_expires_block = ready_block + cfg.ready_ttl_blocks`.

On success it answers `200 {"status": "ready", "hotkey", "digest",
"context", "coldkey_at_ready", "ready_block", "ready_expires_block",
"weights_sha256", "licence", "licence_attestation_sha256",
"commit_payload"}`. The `commit_payload` starts
`metro-v1:weights:c<context>:hippius:` and contains the digest.

A repeated ready of a ready entry answers 200 with the **same**
`ready_block`: a repeat never extends the TTL. A refused ready leaves the
entry `submitted`.

## Caps (A8) and backpressure

"Open" means `ready` or `revealed` (plan §1.2: ready and not yet judged).

- **`409 coldkey_cap_open`** is sent when the coldkey already has
  `cfg.max_open_per_coldkey` open entries. The body carries
  `"advisory": true`, `"open": n` and `"cap": k`, and has no
  `commit_payload`.
- **What "advisory" means (plan §1.2 "Enforcement"):**
  - The intake refuses the ready, so the entry stays `submitted`, no
    `ready_block` is recorded and no commit payload is issued. An honest
    miner can't proceed.
  - It is **not** consensus. The intake's view is never read by satellites
    or by scoring.
  - The authority is the validator's queue selection (W5h). It skips an
    over-cap entry with `coldkey_cap_settlement`, uncharged, and carries
    it to the next settlement.
  - The miner may retry once one of the coldkey's open entries expires or
    closes.
- **`503 queue_full`** is sent when `ops.queue_max_pending` entries are
  already open (global backpressure). It's checked at ready.
- `LaneDIntake.close(hotkey, digest, reason)` moves an entry to `closed` and
  frees its slots. Wiring it to the validator's judged status is W5h/W17's
  job, not W3's.

## TTL (B10, T18d)

The intake evaluates TTL on every request against `current_block()`. A
`ready` entry becomes:

- `revealed` if `reveal_block(...)` is not `None` and is
  `<= ready_expires_block`. A revealed entry never expires at the intake.
- `ready_expired` if `current_block() > ready_expires_block` with no reveal
  inside the window. A reveal after the window doesn't revive it. The
  hotkey's open slot and the coldkey count are freed, a new submit is
  accepted, and a ready of the expired entry is `409 ready_expired`.

Expiry is persisted: a restart with a lagging chain read can't revive the
entry. A failed `reveal_block` read never expires an entry; the answer is
`503 chain_unavailable` or the entry stays open.

## Queue

- The state is one JSON file, `state_dir / service.QUEUE_FILENAME`, written
  atomically: a temp file plus `os.replace`, e.g.
  `cascade.shared.atomic.write_text_atomic`.
- Every accepted submit and ready persists before it answers.
- A failed write answers `503 state_unavailable`. The file on disk is then
  the previous one, and the in-memory state is rolled back too.
- A new `LaneDIntake` on the same `state_dir` resumes everything: entries,
  states, ownership, caps and ready blocks.
- A corrupt queue file (and backup, if any) makes the constructor **raise**.
  It never starts with an empty queue.
- `GET /v1/lane-d/queue` answers `200 {"entries": [...]}`. Each row has
  `hotkey`, `digest`, `state` (`submitted|ready|revealed|ready_expired|closed`),
  `context`, `coldkey_at_ready`, `ready_block`, `ready_expires_block`,
  `weights_sha256`, `licence` and `licence_attestation_sha256`. Ready-time
  fields are `null` before ready.

## Secrets

Grants (presigned URLs, signatures, relay tickets) appear **only** in the
submit response. They're never in a log record at any level, a file under
`state_dir`, or the queue view.

## Rate limits

- **Per IP:** `per_ip_rpm` requests per sliding 60 s window, counting every
  request (refused, unknown route, queue view). Over the limit it's
  `429 rate_limited` with `Retry-After` (≥ 1).
- **Client IP:** the last `X-Forwarded-For` entry when `REMOTE_ADDR` is
  loopback (Caddy). Otherwise it's `REMOTE_ADDR`, and the header is ignored.
- **Per hotkey:** `per_hotkey_rpm` per 60 s across IPs. It counts **only
  signature-verified** requests, so a forger can't spend a victim's budget.
- `DEFAULT_PER_IP_RPM` and `DEFAULT_PER_HOTKEY_RPM` are ints in [1, 600].

## RAM budget

The box has 7 GB and the validator scores on its CPU.

- `service.MAX_BODY_BYTES` is an int in [4096, 65536]. A
  `Content-Length` above it is `413 body_too_large` **without reading the
  body**. A negative or unparseable length is a 4xx, also unread.
- The app never reads past `Content-Length`.
- The rate limiter tracks at most `service.RATE_LIMIT_MAX_KEYS` keys (an int
  in [1024, 16384]); `LaneDIntake.rate_limit_keys()` reports the count.
- There's at most one record per hotkey, however often it resubmits.
- The unit sets `MemoryMax` ≤ 512M.

## Fail closed

Every error from storage, the chain or the minter is a typed 503 that changes
nothing. A transient error is never read as "absent", so it's never
`missing_object`, never `not_registered` and never an expiry.

| failing call | answer |
|---|---|
| storage (`put`, `get`, `list`, `size`) | `503 storage_unavailable` |
| `current_block`, `coldkey_of` (and `reveal_block`) | `503 chain_unavailable` |
| `mint` or `finalize` | `503 grants_unavailable` |
| `published_weights` | `503 dedup_unavailable` |

## Service, systemd unit and Caddy route

`cascade/lane_d/main.py` defines:

- `DEFAULT_HOST == "127.0.0.1"` and `DEFAULT_PORT`;
- `build_arg_parser()`, with flags `--host`, `--port`, `--state-dir` and
  `--chain-toml`;
- `main(argv)`. It calls
  `cascade.shared.codepath.verify_release_code_path("cascade-laned-intake")`
  (the PR #431 guard) before any chain read or bind, and logs a startup
  line containing `intake-laned up:`, which `deploy_gate verify` looks for.

`cascade/lane_d/__main__.py` runs `main`:

- `python -P -m cascade.lane_d --help` exits 0 with `torch` and `bittensor`
  unimportable.
- Launched against another declared release (`CASCADE_RELEASE_PATH`), it
  exits **78** and prints `REFUSING TO START` and the service name.
- No module in `cascade/lane_d/` imports torch, bittensor or the validator
  package at module level.
- Every lane D log call starts with a tag from
  `fixtures/lane_d_intake_tags.py` (build rule 5, `follower_patterns.py`).
  cascade-ops (W11) adds the same tags.

`deploy/cascade-laned-intake.service`:

- It has `[Unit]`, `[Service]` and `[Install]` sections, with
  `WantedBy=multi-user.target`.
- `ExecStart=/abs/.../python -P -m cascade.lane_d --host 127.0.0.1 --port N
  --state-dir /abs --chain-toml /abs`. That matches `deploy/laned/lib.sh`'s
  process pattern and `deploy_gate`'s `argv[0]` ends-with-`python` rule.
- `Environment=PYTHONPATH=/abs` declares the release, and the file never
  mentions `CASCADE_ALLOW_CODE_PATH_MISMATCH`.
- `Restart=on-failure` and `RestartPreventExitStatus` includes 78, so a
  refused start stays down.
- `MemoryMax` ≤ 512M and `OOMScoreAdjust=-500`.
- No secret-shaped `Environment=` names. Credentials come via
  `EnvironmentFile`, env-only.
- No wallet, coldkey or `.bittensor` reference. The intake signs nothing.
- The port differs from the lane A intake's default.

`deploy/caddy/lane-d-intake.caddy` is a snippet the box's site block
imports. It has exactly one `handle /v1/lane-d/* { ... }` (not
`handle_path`), and inside it:

- `request_body { max_size X }` with X ≤ `MAX_BODY_BYTES`;
- exactly one `reverse_proxy 127.0.0.1:N` (or `localhost:N`), where N is the
  unit's `--port`.

It never routes the lane A/B/C paths. Nothing is reloaded by the tests.

**Open item for the owner (not pinned here):** `deploy_gate verify`
(R8) requires a `.scope` cgroup, because `lib.sh` restarts via
`systemd-run --scope`, while a `.service` unit runs in a `.service` cgroup.
Either the S5 restart keeps `lib.sh`'s scope launch and the unit is for boot
and recovery, or R8's verify learns the unit. The builder states which in
the PR.

## Testnet evidence

`test_evidence_testnet_run` **skips** while `$LANED_EVIDENCE` is unset. When
it's set, the file must have:

- `schema: "laned_w3_testnet/1"`, `wp: "W3"`, `source: "testnet"`,
  `network: "test"`, `netuid: 259`;
- `upload_path` equal to the decision record's choice;
- a 40-hex `intake_commit`;
- one `cases` row per id in `helpers.EVIDENCE_CASES`, with `expected` and
  `observed` codes and a matching `http_status`:
  - T2 `duplicate_weights`;
  - T6 `hotkey_entry_open` and `coldkey_cap_open`;
  - T16 `duplicate_weights`;
  - T18d `ready_expired`;
  - T7c `ok`;
- `t7c.weights_bytes` ≥ 2,000,000,000, with `box_rss_delta_mb` and
  `box_disk_delta_mb` ≤ 64 (box RAM and disk flat);
- no credential- or signature-shaped string;
- `passed: true`.

The checker itself is tested offline.

## Invariants

None owned and no `unchanged_when_unarmed` names, so no invariant-10
pairing. The tests use an armed-looking `LaneDConfig` (`from_block = 1`).
W3 doesn't pin behaviour while lane D is unarmed.

## Failure on main

Most tests fail with `ModuleNotFoundError: cascade.lane_d.service` (or
`cannot import name 'service' from 'cascade.lane_d'`). The rest fail as
follows:

- the unit and Caddy tests: `AssertionError`, because the files are missing;
- the entry-point tests: no `cascade.lane_d.__main__` or `main.py`;
- the decision test: unfilled record.

The evidence-checker and follower-tag tests pass on main, and the evidence
test skips.
