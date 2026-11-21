# R5 acceptance tests: the contract they pin

R5 ships the previous-release worktrees and restart scripts for the four
services lane D's S5 upgrade restarts (plan §3.0 R5 row, §5 row 10, §10.5 S5,
§10.8 Rollback, §13). The builder may touch only the paths in `PATHS`
(`deploy/laned/`). It owns no T-rows and no invariants. The common
conventions (PATHS syntax, exit codes, expect-fail right reasons) are in
`tests/accept/w0c/README.md`.

The tests only ever run the scripts with `--dry-run`, against a fixture box
root, a fixture git repo and a fixture venv in `tmp_path`. A real restart is
never executed. Every command a real run would use to change the box
(`systemd-run`, `systemctl`, `choom`, `ssh`, `scp`, `rsync`, `pkill`,
`sqlite3`, `uv`, `busctl`, `nohup`, `sudo`, `docker`, `lium`, `btcli`) is put
first on `PATH` as a shim that records its argv and exits 97. A dry-run that
calls any of them fails.

## Files

| file | what it is |
|---|---|
| `deploy/laned/restart_eval_host.sh` | the eval host (remote pod, over ssh) |
| `deploy/laned/restart_validator.sh` | the main validator |
| `deploy/laned/restart_intake_laned.sh` | the lane D intake (`cascade-laned-intake`) |
| `deploy/laned/restart_trainer.sh` | the trainer |
| `deploy/laned/upgrade_s5.sh` | runs the four in the S5 order |

Shared code may live in other files under `deploy/laned/` (for example a
sourced `lib.sh`). Every `*.sh` under `deploy/laned/` must be
`shellcheck -x` clean and `bash -n` clean. The shellcheck test skips with an
explicit message when `shellcheck` isn't installed; the CI runner has it.
The stub scan also runs over `deploy/laned/`, so no `TODO`/`FIXME` there.

## CLI (all four restart scripts)

```
restart_<svc>.sh --dry-run --root ROOT --repo REPO --venv VENV \
    --release-ref REF --release-dir DIR \
    --previous-ref REF --previous-dir DIR [--ack RULE ...]
restart_eval_host.sh ... --host USER@HOST --port N --known-hosts FILE
upgrade_s5.sh        <the union of the above>
```

- `--root` is the box root (`/root/cascade` in production): logs, `_train_work/`
  and the §10.8 state files are read from it, and `_state_backups/` is under it.
- `--repo` is the git repo the release worktrees are made from.
- `--venv` is the pinned venv whose `bin/python` and console scripts run the
  service. The fixture venv's `bin/python` is a symlink to the test's Python;
  its console scripts are empty executables.
- `LANED_NOW` (env, ISO 8601 UTC such as `2026-11-16T12:00:00Z`) replaces the
  clock. Without it the script uses the real UTC time. The snapshot timestamp
  `<ts>` is that time as `%Y%m%dT%H%M%SZ`.
- A missing required flag is a usage error: non-zero exit (2 is fine).
- `--ack RULE` names a refusal rule the owner accepted (as in
  `deploy_gate.py --ack`); it is forwarded to the `deploy_gate check` step.

## Exit codes

0: the plan is GO (and with `--dry-run`, printed). 1: refused. 2: usage.
`--dry-run` changes nothing in any case: no file under the root, repo or
worktrees is created or modified, no worktree is added, no shim is called,
and two runs with the same inputs print the same plan.

## The dry-run plan

Every planned action is one output line starting `PLAN <step> `, in this
order of steps:

1. `PLAN worktree` lines.
   - The release: when `--release-dir` doesn't exist, the line contains
     `git -C <repo> worktree add --detach <release-dir> <release-ref>`. When it
     exists and its `HEAD` is the commit of `--release-ref`, the line contains
     `keep` and the dir. When it exists at another commit, the script refuses
     (non-zero, `REFUSE`, the dir named).
   - The previous release (the rollback target, §10.8): the same create-or-keep
     rule, with `--previous-dir` and `--previous-ref`.
   - An unknown `--release-ref` or `--previous-ref` is refused (non-zero).
   - No output line ever contains `worktree remove` or `worktree prune`.
2. `PLAN check`: the line contains `python -P`,
   `<release-dir>/scripts/deploy_gate.py check --service <gate-svc>`,
   `--from <previous-ref>`, `--to <release-ref>` and each `--ack RULE` given.
3. `PLAN snapshot` lines, one per §10.8 state file, copying from `<root>` to
   `<root>/_state_backups/<ts>/`: `validator_state.db`, `cascade_state.json`,
   `warm_start_init.json`, `lane_d_state.json`, `netconfig_state.json`, every
   `activation_*_state.json`, `satellite_state.json`, `decided_votes.toml`.
   A present file's line contains `<root>/<name>` and
   `<root>/_state_backups/<ts>/<name>`. An absent fixed name's line contains
   the name and the word `absent`. Every service snapshots, the eval host too.
4. `PLAN restart`: the exact restart command(s).
   - Box services: the command contains `PYTHONPATH=<release-dir>`,
     `<venv>/bin/python -P` and the service's entry point
     (`cascade-validator`, `cascade-trainer`, or `cascade-laned-intake` /
     `cascade.lane_d` for the intake). Validator and trainer run under
     `systemd-run --scope`; the intake under `systemd`. The plan pins the OOM
     score with `-500` (`choom -n -500` or `OOMScoreAdjust=-500`). No
     `python -m` without `-P` (the cwd-shadowing incident, §13).
   - Eval host: every `ssh` in the plan pins the host key with
     `-o UserKnownHostsFile=<known-hosts>` and `-o StrictHostKeyChecking=yes`,
     names `-p <port>` and `<host>`, and the plan syncs the venv with
     `uv sync --frozen`. Never `StrictHostKeyChecking=no` or
     `UserKnownHostsFile=/dev/null`.
5. `PLAN verify`: the line contains `python -P`,
   `<release-dir>/scripts/deploy_gate.py verify --service <gate-svc>` and
   `--release <release-dir>`.

`<gate-svc>` is `eval-host`, `validator`, `intake-laned` or `trainer` (the
`deploy_gate` names R8 adds for the new services).

Nothing in the scripts or the plan reads or names wallet keys: no
`coldkey`, `BT_WALLET_PASSWORD`, `.bittensor/wallets` or `cat .env`.

## Refusals (quiet-window rules, §10.5 S5, §13, owner rules)

Each is checked by every script, `--dry-run` included, before any plan is
GO. A refusal exits 1 and prints `REFUSE [<rule>]` with a reason.

| rule | fixture that triggers it |
|---|---|
| `freeze` | `LANED_NOW` between Wed 22:00 and Thu 08:00 UTC |
| `settlement` | the last `block N` line in `validator.log` (else `trainer.log`) places `LANED_NOW` within 20 min before a 900-block boundary, or within 15 min after one (12 s blocks) |
| `clock` | neither log has a `block N` line, so the window can't be placed |
| `scoring` | `validator.log` has a `new manifest round=R` with no later `round=R weights set` |
| `in_flight` | `_train_work/funded_queue.json` has an entry with `status` `in_flight` and a pod-holding `phase` (for example `training`) |
| `harvest` | `_train_work/era_state.json` has a non-empty `verify_queue` |

Log lines use the box format `YYYY-MM-DD HH:MM:SS,mmm <level> <text>`.
`--ack freeze` lets the freeze fixture through (exit 0, plan printed).

## `upgrade_s5.sh`

It runs the four restart scripts in the S5 order, **eval host → validator →
lane D intake → trainer**, with the same flags, and stops at the first one
that refuses (non-zero). In a dry-run the `PLAN restart` lines appear in that
order.

## Not pinned here

The exact service argument lines (wallet names, ports, `--chain-toml`), the
copy tool for the snapshot (`cp -a` or `sqlite3 .backup`), and the bench-sweep
window (no detector exists yet). `deploy_gate.py` itself is R8.
