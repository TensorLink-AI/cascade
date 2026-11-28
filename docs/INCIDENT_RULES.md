# Incident rules

When an incident could have been caught by a pattern check, it gets a lint
rule. `scripts/check_incident_rules.py` runs those rules in CI's `lint` job,
next to `check_docs.py`. It needs only the standard library and imports nothing
from cascade, and it scans the whole repository in under a second:

```bash
python3 scripts/check_incident_rules.py            # exit 1 on any hit
python3 scripts/check_incident_rules.py --list     # every rule with its incident
python3 scripts/check_incident_rules.py --rule IR003 deploy/restart_validator.sh
```

The checker reads tracked files (`git ls-files`). It never reads `tests/`,
`evidence/`, `docs/plans/`, `docs/archive/`, this file or the checker itself,
because those quote bad patterns on purpose. It checks executable text:

- shell scripts and systemd units, with comment lines skipped and `\`
  continuations joined;
- Python lines that are not comments;
- Markdown shell code blocks;
- for IR001 only, inline Markdown code spans, because the #892 instruction
  was one.

`tests/unit/test_incident_rules.py` has a positive, a negative and an
allowlist case for every rule, and it also runs the checker over the whole
repository.

## Exceptions

Put an allowlist comment on the line of the hit or on the line above it.
Name the rule and give the reason:

```bash
# incident-ok: IR004 KH_OPTS pins a per-pod known_hosts when POD_KNOWN_HOSTS is set
```

In Markdown, use `<!-- incident-ok: IR001 <reason> -->`. An allowlist comment
that names no reason, or names an unknown rule, is itself a hit (IR000). A
known violation waiting on an owner decision is allowlisted with
`KNOWN VIOLATION pending owner` in its reason, so it stays visible in the
source.

## Adding a rule

Every incident that a pattern check could have caught gets a rule **in the
same PR as its fix** (one PR per feature: the rule, its tests and the fix go
together).

1. Name the failure class, not the instance. For example, "a frozen sync
   without extras", not "line 307 of VALIDATOR.md".
2. Add a `Rule` to `RULES` in `scripts/check_incident_rules.py`. Give it the
   next free `IR` number, a one-line title, the incident (date, issue or PR),
   the fix text and the file globs. Then add a `check_file(rel, text)`
   function (regex or a small `ast` / `tomllib` check) or a
   `check_tree(files)` function for a whole-package check like IR007. Prefer
   `command_lines()` so that comments and prose don't trip the rule.
3. Add tests to `tests/unit/test_incident_rules.py`: at least one hit taken
   from the incident's own text, one near-miss that must pass, and one
   allowlisted case.
4. Run the checker on main. Fix the true hits in the same PR, allowlist the
   justified ones with a reason, and list any you left in the PR body.
5. Add a section here: the incident, the rule, and how to fix a hit.

A class that no pattern can catch (a wrong number, a logic race) belongs in
a test or a runtime guard, not here. Say which one in the incident's OPSLOG
entry.

## IR001 — bare `uv sync --frozen`

**Incident.** 2026-11-24, issue #892 (found in the b274626d testnet rollout).
The validator upgrade step in `docs/VALIDATOR.md` said `git pull`, then
`uv sync --frozen`. On a venv installed with `--extra train --extra hippius
--extra chain`, that command makes the venv match the lock exactly, so it
**uninstalls** the extras, bittensor included. Running processes kept going
until their next restart or lazy import, then failed with
`No module named 'bittensor'`. #895 fixed the docs. #897 fixed the same bug
in the lane D eval-host restart (`deploy/laned/lib.sh`).

**Rule.** A `uv sync … --frozen` command that names no `--extra`,
`--all-extras`, `--inexact` or `--group` is a hit, unless its extras are
parametrised (`{extra}`, `$EXTRAS`). It applies in `docs/`, `README.md`,
`deploy/` and `scripts/`. An inline Markdown instruction passes when its
paragraph says which extras to keep.

**Fix.** Repeat the install extras, as in `uv sync --frozen --extra train
--extra hippius --extra chain`. On a host whose profile installed extras the
script doesn't know about, use `uv sync --frozen --inexact`.

## IR002 — `python -m` without `-P`

**Incident.** 2026-11-04. The intake was launched as `python -m
cascade.funding.main` with cwd `/root/cascade`. The cwd became `sys.path[0]`
and shadowed the release on `PYTHONPATH`, so the intake ran Oct-30 code for
a day. The #419 withdraw guard never ran and a miner lost $90.54. #431
added a startup code-path guard (`cascade/shared/codepath.py`).

**Rule.** In `deploy/` and in `scripts/` shell files, `python -m <module>`
is a hit unless the flags include `-P` or `-I`.

**Fix.** Use `python -P -m …`, or the venv entry point (`.venv/bin/cascade-…`).

## IR003 — `pgrep -f` / `pkill -f` with a literal pattern

**Incident.** This happened repeatedly from 2026-09-14 to 2026-11-24 (OPSLOG).
`pgrep -f`/`pkill -f` matches full command lines, so it also matches the
shell that runs it whenever that shell's command line contains the pattern.
Tool shells were killed (exit 144), a deploy script killed itself, and
`restart_*` scripts listed the invoking shell as the service. On 2026-11-24
the EXP-LB-KING experiment driver's waits stalled on a self-matching
`pgrep`.

**Rule.** In `deploy/` and `scripts/`, `pgrep`/`pkill` with `-f` and a
literal pattern that has no `[…]` class is a hit. A pattern held in a
variable passes. Whoever builds the variable must bracket it or exclude
`$$`/`$PPID`, as `deploy/restart_*.sh` do.

**Fix.** Bracket one character, as in `pgrep -f 'cascade[-]trainer'`. The
regex is unchanged, but the literal text no longer matches itself. You can
also kill by a pid file. The bracket protects only the pattern: a command
line that also contains the plain name elsewhere still self-matches, so put
the kill in a script file.

## IR004 — ssh host key not pinned

**Incident.** 2026-09-15, round 9075600. Lium reuses NAT `ip:port` pairs
across containers, and each container has a fresh host key. The JIT king
pod was reached through the shared `~/.ssh/known_hosts` with
`StrictHostKeyChecking=accept-new`, and its first `mkdir` failed with "Host
key verification failed". The king leg died and the pod sat idle. The fix
pinned the scanned key per pod (`pinned_host_key`).

**Rule.** In `deploy/` and `scripts/`, `StrictHostKeyChecking=no` and
`UserKnownHostsFile=/dev/null` are hits. So is
`StrictHostKeyChecking=accept-new` on a line that names no
`UserKnownHostsFile`, because that means the shared file.

**Fix.** Pin the pod's scanned key with `-o UserKnownHostsFile=<per-pod
file> -o StrictHostKeyChecking=yes`. Hand-run ssh in an operator shell is
outside this rule.

## IR005 — bulk git staging

**Incident.** 2026-08-31, PR #239. A `git add -u` in a lint-fix commit
staged the deployment's uncommitted live `chain.toml` override and pushed it
into public main. PR #240 restored the default.

**Rule.** In `scripts/`, `deploy/` and Markdown shell blocks, `git add
-u`/`-A`/`--all`/`--update`/`.` and `git commit -a`/`--all` are hits.

**Fix.** Stage explicit paths, and check the commit's file list before you
push.

## IR006 — Shadeform as a GPU provider

**Incident.** Owner rule, 2026-11-24: "lium is always the one for GPU,
shadeform only CPU". It came after a Shadeform eval-host fallback was
proposed. Shadeform GPU pods had already cost rounds: a docker-mode cached
image, CDI-mode containers that never started, and failed gates. The lane D
elastic code refuses a Shadeform type with `num_gpus > 0`
(`cascade/laned/elastic.py`).

**Rule.** A TOML table in `deploy/` or `scripts/` is a hit when its
`provider`/`providers` names `shadeform` and the same table sets a positive
`gpus`, `gpus_per_pod`, `num_gpus` or `gpu_count`.

**Fix.** Remove `shadeform` from that GPU table's providers. CPU tables (the
pool_d builder, `[generation] providers`) are fine.

**Open.** The `[provisioner.eval]` table in `deploy/provision.mainnet.toml`
still lists `["lium", "shadeform"]`. It is allowlisted as `KNOWN VIOLATION
pending owner`: the template predates the rule, and the deployed
`provision.toml` governs.

## IR007 — config dataclass field the loader never parses

**Incident.** 2026-08-27. DEC-CA-0033's `[training]` knobs (#223) and
DEC-CA-0034's `[round]`/`[scoring]` init-gate fields (#224) landed as
dataclass fields with no `load_chain_config` parse. Setting them in
`chain.toml` did nothing, and the tests stayed green because they built the
contracts directly. #225 fixed both.

**Rule.** Every field of a dataclass in `cascade/shared/config/*.py` must
appear somewhere in that package as a string literal (a `t.get("field")`
read) or as a keyword argument (`Config(field=…)`). Classes parsed
generically through `fields(Cls)` are exempt. Runtime-only fields, which are
set in code and never read from `chain.toml`, are listed with their setter in
`IR007_RUNTIME_ONLY` in the checker.

**Fix.** Add the loader parse and a loader round-trip test (patch the repo's
`chain.toml` text, load it, assert the field). For a new runtime-only field,
add `# incident-ok: IR007 runtime-only: <who sets it>` on the field.

**Limits.** This is a presence check. A field whose name happens to be some
other key's string passes. The round-trip test is still the real guard.

## IR008 — remote python over ssh with no on-host `timeout`

**Incident.** 2026-11-27, PR #969. `SSHEvalHost._ssh` (lane D eval host)
passed `timeout=` to `subprocess.run`. That kills only the local `ssh`. The
remote `lane_d_forecasts public` kept running on the 4090, and each retry
started another copy beside it (three were stacked; two orphans were killed by
hand at 01:15). #969 bounds the request on the host with `timeout -k 30
<T-30>` and serialises it with `flock`.

**Rule.** In `cascade/`, a function that reads `.remote_python` and builds a
`-m <module>` command must also carry a `timeout` command token, or launch
detached (`nohup`/`setsid`) with its own deadline. Reviewed builders that
cannot outlive their caller (a `--help` probe, a stdin `--serve` session, the
detached leg launch) are listed with a reason in `IR008_REVIEWED` in the
checker; open ones say `KNOWN VIOLATION pending owner` there
(`lane_d_bench_host.SSHBenchHost.__call__`, `r2_transfer.relay_command`).

**Fix.** Wrap the remote argv in `timeout -k <grace> <T - grace>` as
`SSHEvalHost.remote_command` does, and take a `flock` when a retry must never
run beside an orphan.

**Limits.** The check is per function: a command built in one helper and sent
in another is checked where the `-m` string is built.

## IR009 — an ssh reply's raw stdout parsed as JSON

**Incident.** 2026-11-27, PR #983. The lane B settlement rebase ran
`<worker> && cat result.json` in one ssh command and parsed the whole stdout
with `json.loads`. While the step pulled an uncached 16 GB checkpoint,
`hippius_hub/file_download.py` printed `Downloading <file> (parallel)...` to
stdout ahead of the JSON. The parse failed ("pod reply is not JSON"), and the
19:27 settlement published without lane B. #983 parses the last JSON line
(`lane_b_offload.pod_reply_json`).

**Rule.** In `cascade/` and `scripts/`, a function that makes an ssh call (a
call whose name contains `ssh`) must not pass `X.stdout` or
`X.stdout.strip()` straight to `json.loads`. Three shapes pass:

- `X` comes from its own ssh call whose command is a literal `cat <file>`
  and nothing else (no `&&`, `;` or `|`);
- the parse picks a line (`.splitlines()[-1]`);
- the reply goes through a tolerant parser (`pod_reply_json`).

**Fix.** Read the result file with a separate `cat`-only ssh call, or parse
the last line. Remote modules log to stderr. Library chatter is outside our
control, so the reader must tolerate it.

**Limits.** The check is per function and follows only plain `X = <call>`
assignments. A reviewed exception (`validator_bench`'s stdlib-only
`bench_identity` probe) carries an `incident-ok: IR009` comment.

## IR010 — pool or checkpoint arrays loaded in the validator process

**Incident.** 2026-11-28. The validator's off-thread lane D settlement step
loaded every pinned `pool_d` snapshot into the validator process at once (13
pins, about 1.7 GB on disk plus about 17 MB of JSON metadata each). The
validator's RSS went from 1.05 GB to more than 3.65 GB on each step. The 7 GB
box also runs the trainer and the intake, and it OOM-killed the validator at
04:39 and 05:41, 22 to 35 s after `lane D: step for settlement … started`. No
log line named the cause. W-OFFBOX moved the pool phase to the lane D eval
host (`cascade.eval.lane_d_settle`). The step now logs its peak RSS and
reports an ERROR over the box budget (`offthread_worker.STEP_RSS_BUDGET_BYTES`).

**Rule.** In `cascade/validator/`, a call to a pool or checkpoint loader is a
hit: `load_pool`, `load_pool_for`, `load_pool_series`, `_load_pools`,
safetensors `load_file` / `safe_open`, or `np.load` / `numpy.load` /
`torch.load`. Two shapes pass:

- `np.load(..., mmap_mode="r")`, which pages the data in on demand;
- a reviewed bounded load that carries
  `# incident-ok: IR010 <why it is bounded>`, for example one cached file, a
  loader whose callers are checked, or code that runs on a pod and not on the
  box.

**Fix.** Load and score on the eval host, or stream one file at a time with
`mmap_mode` under a hard bound and a test that checks peak memory as the
input grows (`tests/accept/w_offbox`).

**Limits.** The check matches call names only. A load reached through another
module (`cascade/eval/`, `cascade/pool/`) is not a hit. The step's runtime RSS
log catches that case. The existing sites were allowlisted on 2026-11-28 with
`baseline 2026-11-28` in their reasons. They each load one snapshot or one
checkpoint pair, and each one should be reviewed.
