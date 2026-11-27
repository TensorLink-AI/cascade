"""scripts/check_incident_rules.py: one positive, negative and allowlist case per rule.

Every rule is an incident (docs/INCIDENT_RULES.md). The last test runs the
checker over the repository itself, so a new hit fails the unit suite as well
as the CI lint step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_incident_rules", ROOT / "scripts" / "check_incident_rules.py")
cir = importlib.util.module_from_spec(_SPEC)
sys.modules["check_incident_rules"] = cir
_SPEC.loader.exec_module(cir)


def _rule(rule_id):
    return next(r for r in cir.RULES if r.id == rule_id)


def hits(tmp_path, files: dict[str, str], rule_id: str):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return [(h.rule, h.path, h.line) for h in cir.check(tmp_path, sorted(files),
                                                        (_rule(rule_id),))]


# ── IR001 bare `uv sync --frozen` ──────────────────────────────────────────────

@pytest.mark.parametrize("rel,text,line", [
    # the #892 form: inline upgrade instruction with no extras in the paragraph
    ("docs/VALIDATOR.md", "Upgrade: `git pull`, then `uv sync --frozen`.\n", 1),
    ("docs/X.md", "intro\n\n```bash\ngit pull\nuv sync --frozen\n```\n", 5),
    ("deploy/laned/lib.sh", 'ssh "$h" "cd $D && uv sync --frozen"\n', 1),
    ("scripts/x.py", 'cmd = f"cd {d} && uv sync --frozen"\n', 1),
    ("deploy/x.sh", "PATH=$HOME/.local/bin:$PATH uv sync --frozen\n", 1),
])
def test_ir001_flags_a_bare_frozen_sync(tmp_path, rel, text, line):
    assert hits(tmp_path, {rel: text}, "IR001") == [("IR001", rel, line)]


@pytest.mark.parametrize("rel,text", [
    ("docs/VALIDATOR.md", "Upgrade: `uv sync --frozen --extra train --extra chain`.\n"),
    ("docs/VALIDATOR.md", "`uv sync --frozen` pins the lock. Repeat the extras.\n"),
    ("docs/X.md", "```sh\nUV_PROJECT_ENVIRONMENT=/v uv sync --frozen \\\n  --extra train\n```\n"),
    ("deploy/x.sh", "uv sync --frozen --inexact\n.bootstrap/uv sync --frozen --all-extras\n"),
    ("deploy/x.sh", "# a comment about uv sync --frozen\n"),
    ("scripts/x.py", 'help = "copy the release and uv sync --frozen on the host"\n'),
    ("scripts/x.py", 'cmd = f"cd {d} && uv sync --frozen{extra}"\n'),
    ("docs/X.md", "`uv sync --project env --frozen --group qualify`\n"),
    ("cascade/x.sh", "uv sync --frozen\n"),          # outside the rule's globs
    ("tests/x.md", "`uv sync --frozen`\n"),           # globally excluded
])
def test_ir001_passes_commands_that_keep_extras(tmp_path, rel, text):
    assert hits(tmp_path, {rel: text}, "IR001") == []


def test_ir001_allowlist(tmp_path):
    text = "# incident-ok: IR001 sidecar has no extras\nuv sync --frozen\n"
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR001") == []
    md = "Do `uv sync --frozen`. <!-- incident-ok: IR001 fresh host, nothing installed -->\n"
    assert hits(tmp_path, {"docs/X.md": md}, "IR001") == []


# ── IR002 `python -m` without -P ───────────────────────────────────────────────

def test_ir002_flags_python_m_without_P(tmp_path):
    unit = "[Service]\nExecStart=/opt/c/.venv/bin/python -m cascade.direct --serve\n"
    sh = "cd /root/cascade\nnohup python3 -u -m cascade.funding.main &\n"
    assert hits(tmp_path, {"deploy/x.service": unit, "deploy/run.sh": sh}, "IR002") == [
        ("IR002", "deploy/run.sh", 2), ("IR002", "deploy/x.service", 2)]


@pytest.mark.parametrize("text", [
    "ExecStart=/root/cascade/.venv/bin/python -P -m cascade.lane_d --port 1\n",
    "python3 -u -P -m cascade.x\n",
    "python -I -m cascade.x\n",
    "python -IP -m cascade.x\n",
    "# python -m cascade.x\n",
    ".venv/bin/cascade-validator --wallet-name x\n",
])
def test_ir002_passes_P_and_entry_points(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR002") == []


def test_ir002_allowlist(tmp_path):
    text = "# incident-ok: IR002 stdlib module, no cascade import\npython -m http.server\n"
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR002") == []


# ── IR003 self-matching pgrep/pkill -f ─────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "TRAINER_PID=\"$(pgrep -f 'cascade-trainer' | head -1)\"\n",
    "pkill -f run_ctx8064.py\n",
    "pgrep -af cascade.funding.main\n",
    "pkill -KILL -f -- 'cascade-benchmark'\n",
])
def test_ir003_flags_a_literal_pattern(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR003") == [("IR003", "deploy/x.sh", 1)]


@pytest.mark.parametrize("text", [
    "pgrep -f 'cascade[-]trainer'\n",
    "pkill -f 'bin/cascade[-]benchmark( |$)' 2>/dev/null\n",
    'pgrep -f -- "$pat" || true\n',
    "pgrep -fc cascade[.]trainer[.]worker\n",
    "pgrep cascade-trainer\n",                       # no -f: matches the process name only
    "kill \"$(cat /run/x.pid)\"\n",
    "# pgrep -f cascade-trainer\n",
])
def test_ir003_passes_bracketed_variable_or_no_f(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR003") == []


def test_ir003_python_strings(tmp_path):
    bad = 'busy = run("pgrep -f cascade_benchmark >/dev/null")\n'
    ok = 'sweep = f"for q in $(pgrep -f {pattern}); do kill $q; done"\n'
    assert hits(tmp_path, {"scripts/a.py": bad, "scripts/b.py": ok}, "IR003") == [
        ("IR003", "scripts/a.py", 1)]


def test_ir003_allowlist(tmp_path):
    text = "pkill -f stray-thing  # incident-ok: IR003 run from a script file, never a shell\n"
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR003") == []


def test_ir003_suggests_the_bracket_form():
    assert cir._bracketed("cascade-trainer") == "[c]ascade-trainer"


# ── IR004 unpinned ssh host keys ───────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "ssh -o StrictHostKeyChecking=no root@pod true\n",
    "ssh -o UserKnownHostsFile=/dev/null root@pod true\n",
    "scp -o StrictHostKeyChecking=accept-new f root@pod:/x\n",
])
def test_ir004_flags_unpinned_keys(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR004") == [("IR004", "deploy/x.sh", 1)]


@pytest.mark.parametrize("text", [
    "ssh -o UserKnownHostsFile=/tmp/pod.kh -o StrictHostKeyChecking=yes root@pod true\n",
    "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH root@pod true\n",
    'argv = ["ssh", "-o", f"UserKnownHostsFile={kh}", "-o", "StrictHostKeyChecking=yes"]\n',
])
def test_ir004_passes_pinned_keys(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR004") == []


def test_ir004_allowlist(tmp_path):
    text = ("# incident-ok: IR004 KH_OPTS pins POD_KNOWN_HOSTS\n"
            'SSH=(-o StrictHostKeyChecking=accept-new "${KH_OPTS[@]}")\n')
    assert hits(tmp_path, {"deploy/x.sh": text}, "IR004") == []


# ── IR005 bulk git staging ─────────────────────────────────────────────────────

@pytest.mark.parametrize("rel,text", [
    ("scripts/x.sh", "git add -u\n"),
    ("scripts/x.sh", "git -C /root/cascade add -A && git commit -m x\n"),
    ("scripts/x.sh", "git add .\n"),
    ("scripts/x.sh", 'git commit -am "lint"\n'),
    ("docs/OPERATOR.md", "```bash\ngit add --update\n```\n"),
])
def test_ir005_flags_bulk_staging(tmp_path, rel, text):
    assert len(hits(tmp_path, {rel: text}, "IR005")) == 1


@pytest.mark.parametrize("text", [
    "git add cascade/x.py tests/unit/test_x.py\n",
    "git commit --amend --no-edit\n",
    "git -C /w commit -F /tmp/msg\n",
    "git add ./docs/x.md\n",
])
def test_ir005_passes_explicit_paths(tmp_path, text):
    assert hits(tmp_path, {"scripts/x.sh": text}, "IR005") == []


def test_ir005_allowlist(tmp_path):
    text = "git add -A  # incident-ok: IR005 throwaway clone in a temp dir\n"
    assert hits(tmp_path, {"scripts/x.sh": text}, "IR005") == []


# ── IR006 Shadeform GPUs ───────────────────────────────────────────────────────

def test_ir006_flags_a_shadeform_gpu_table(tmp_path):
    text = ('[provisioner.eval]\nsku = "RTX 3090"\ngpus_per_pod = 1\n'
            'providers = ["lium", "shadeform"]\n')
    assert hits(tmp_path, {"deploy/provision.x.toml": text}, "IR006") == [
        ("IR006", "deploy/provision.x.toml", 4)]


def test_ir006_flags_nested_and_array_tables(tmp_path):
    text = '[[a.types]]\nprovider = "shadeform"\nnum_gpus = 8\n'
    assert hits(tmp_path, {"scripts/x.toml": text}, "IR006") == [("IR006", "scripts/x.toml", 2)]


@pytest.mark.parametrize("text", [
    '[provisioner.eval]\ngpus_per_pod = 1\nproviders = ["lium"]\n',
    '[pool_d_builder]\nprovider = "shadeform"\n[[pool_d_builder.types]]\ncloud = "m"\n',
    '[generation]\nproviders = ["targon", "shadeform"]\n',
    '# providers = ["lium", "shadeform"]\n# gpus_per_pod = 1\n',
])
def test_ir006_passes_cpu_and_lium(tmp_path, text):
    assert hits(tmp_path, {"deploy/x.toml": text}, "IR006") == []


def test_ir006_allowlist(tmp_path):
    text = ('[e]\ngpus = 1\n# incident-ok: IR006 owner-approved one-off\n'
            'providers = ["shadeform"]\n')
    assert hits(tmp_path, {"deploy/x.toml": text}, "IR006") == []


# ── IR007 config fields the loader never parses ────────────────────────────────

SECTIONS = '''from dataclasses import dataclass


@dataclass(frozen=True)
class KnobConfig:
    parsed_knob: int = 0
    kwarg_knob: int = 0
    orphan_knob: int = 0
'''
LOADER = '''from .sections import KnobConfig


def load(raw):
    k = raw.get("knob", {})
    return KnobConfig(parsed_knob=int(k.get("parsed_knob", 0)), kwarg_knob=1)
'''


def test_ir007_flags_a_field_the_loader_never_reads(tmp_path):
    files = {"cascade/shared/config/sections.py": SECTIONS,
             "cascade/shared/config/loader.py": LOADER}
    assert hits(tmp_path, files, "IR007") == [
        ("IR007", "cascade/shared/config/sections.py", 8)]


def test_ir007_passes_once_the_loader_parses_it(tmp_path):
    loader = LOADER.replace("kwarg_knob=1", 'kwarg_knob=1, orphan_knob=k.get("orphan_knob")')
    files = {"cascade/shared/config/sections.py": SECTIONS,
             "cascade/shared/config/loader.py": loader}
    assert hits(tmp_path, files, "IR007") == []


def test_ir007_passes_generic_fields_loops_and_the_runtime_registry(tmp_path):
    generic = LOADER + "\n\ndef dump(c):\n    return [f.name for f in fields(KnobConfig)]\n"
    files = {"cascade/shared/config/sections.py": SECTIONS,
             "cascade/shared/config/loader.py": generic}
    assert hits(tmp_path, files, "IR007") == []
    assert "ActivationConfig.resolved_block" in cir.IR007_RUNTIME_ONLY


def test_ir007_allowlist(tmp_path):
    sections = SECTIONS.replace(
        "    orphan_knob: int = 0\n",
        "    # incident-ok: IR007 runtime-only: set by the worker\n    orphan_knob: int = 0\n")
    files = {"cascade/shared/config/sections.py": sections,
             "cascade/shared/config/loader.py": LOADER}
    assert hits(tmp_path, files, "IR007") == []


# ── IR008 remote python over ssh with no on-host bound ─────────────────────────

# The pre-#969 SSHEvalHost._ssh (cascade/validator/lane_d_inputs.py at 0b617360):
# a local subprocess timeout only; the remote lane_d_forecasts outlived it.
IR008_PRE_969 = '''\
class SSHEvalHost:
    def _ssh(self, host, args, payload):
        cmd = (f"cd {shlex.quote(host.workdir)} && "
               + " ".join(shlex.quote(a) for a in
                          [host.remote_python, "-m", "cascade.eval.lane_d_forecasts", *args]))
        proc = self._run(build_ssh_argv(host, cmd), input=payload, timeout=self.timeout_s)
'''
# The #969 fix: the request is bounded on the host.
IR008_FIXED = '''\
class SSHEvalHost:
    def remote_command(self, host, args):
        argv = ["timeout", "-k", "30", str(self.timeout_s - 30)]
        argv += [host.remote_python, "-m", "cascade.eval.lane_d_forecasts", *args]
        return f"cd {shlex.quote(host.workdir)} && " + " ".join(shlex.quote(a) for a in argv)
'''


def test_ir008_flags_the_pre_969_eval_host_call(tmp_path):
    files = {"cascade/validator/lane_d_inputs.py": IR008_PRE_969}
    assert hits(tmp_path, files, "IR008") == [("IR008", "cascade/validator/lane_d_inputs.py", 2)]


@pytest.mark.parametrize("text", [
    IR008_FIXED,
    # detached launch with its own deadline
    "def launch(host):\n    return f'nohup {host.remote_python} -m cascade.trainer.worker &'\n",
    # a local python -m (no remote_python): not a remote command
    "def local():\n    return [sys.executable, '-m', 'cascade.eval.x']\n",
    # remote python -c probe, no -m
    "def probe(host):\n    return f'{host.remote_python} -c \"import cascade\"'\n",
])
def test_ir008_passes_bounded_detached_and_local(tmp_path, text):
    assert hits(tmp_path, {"cascade/x.py": text}, "IR008") == []


def test_ir008_allowlist_and_reviewed_table(tmp_path):
    text = "# incident-ok: IR008 exits in seconds\n" + IR008_PRE_969.split("\n", 1)[1].lstrip()
    assert hits(tmp_path, {"cascade/x.py": text}, "IR008") == []
    for key, reason in cir.IR008_REVIEWED.items():
        rel, fn = key.split("::")
        src = (ROOT / rel).read_text()
        assert f"def {fn}(" in src, f"IR008_REVIEWED names a function that is gone: {key}"
        assert len(reason) > 10



# ── IR009 an ssh reply's raw stdout parsed as JSON ─────────────────────────────

# The pre-#983 LaneBOffload step (cascade/trainer/lane_b_offload.py at ea860300):
# the worker and the `cat result.json` share one ssh stdout.
IR009_PRE_983 = '''\
class LaneBOffload:
    def _step(self, host, op, run_dir, stdin):
        cmd = (f"cd {q(host.workdir)} && {py} -m cascade.trainer.lane_b_offload {op} && "
               f"cat {q(run_dir + '/result.json')}")
        proc = ok(self._ssh(host, cmd, left(), stdin), "worker")
        doc = json.loads(proc.stdout)
'''


def test_ir009_flags_the_pre_983_pod_reply(tmp_path):
    files = {"cascade/trainer/lane_b_offload.py": IR009_PRE_983}
    assert hits(tmp_path, files, "IR009") == [("IR009", "cascade/trainer/lane_b_offload.py", 6)]
    stripped = IR009_PRE_983.replace("json.loads(proc.stdout)", "json.loads(proc.stdout.strip())")
    assert hits(tmp_path, {"cascade/y.py": stripped}, "IR009") == [("IR009", "cascade/y.py", 6)]


@pytest.mark.parametrize("text", [
    # #983: the tolerant parser
    IR009_PRE_983.replace("json.loads(proc.stdout)", "pod_reply_json(proc.stdout)"),
    # a separate cat-only ssh call (bench_hook / lane_b_bench / eval_offload)
    "def report(host, path):\n"
    "    cat = run_ssh(build_ssh_argv(host, f\"cat {shlex.quote(path)}\"), 120)\n"
    "    return json.loads(cat.stdout)\n",
    # a last-line parse (eval_offload.check_private_worker)
    "def probe(host):\n    proc = run_ssh(build_ssh_argv(host, cmd), 30)\n"
    "    return json.loads(proc.stdout.strip().splitlines()[-1])\n",
    # a local subprocess: no ssh in the function
    "def local():\n    run = subprocess.run(argv, capture_output=True)\n"
    "    return json.loads(run.stdout)\n",
])
def test_ir009_passes_tolerant_cat_only_and_local(tmp_path, text):
    assert hits(tmp_path, {"cascade/x.py": text}, "IR009") == []


def test_ir009_cat_chained_after_a_worker_is_still_a_hit(tmp_path):
    text = ("def step(host):\n"
            "    proc = run_ssh(build_ssh_argv(host, f\"cat a.json && {py} -m w\"), 9)\n"
            "    return json.loads(proc.stdout)\n")
    assert hits(tmp_path, {"cascade/x.py": text}, "IR009") == [("IR009", "cascade/x.py", 3)]


def test_ir009_allowlist(tmp_path):
    text = IR009_PRE_983.replace(
        "        doc = json.loads(proc.stdout)\n",
        "        # incident-ok: IR009 stdlib-only module, prints only its JSON\n"
        "        doc = json.loads(proc.stdout)\n")
    assert hits(tmp_path, {"cascade/x.py": text}, "IR009") == []

# ── allowlist hygiene (IR000) and the repository itself ────────────────────────

def test_allowlist_needs_a_known_rule_and_a_reason(tmp_path):
    text = ("# incident-ok: IR001\nuv sync --frozen\n"
            "# incident-ok: IR999 typo'd rule id\n")
    p = tmp_path / "deploy" / "x.sh"
    p.parent.mkdir(parents=True)
    p.write_text(text)
    got = sorted((h.rule, h.line) for h in cir.check(tmp_path, ["deploy/x.sh"]))
    assert got == [("IR000", 1), ("IR000", 3), ("IR001", 2)]


def test_every_rule_is_documented():
    book = (ROOT / "docs" / "INCIDENT_RULES.md").read_text()
    for r in cir.RULES:
        assert f"## {r.id}" in book, f"{r.id} has no section in docs/INCIDENT_RULES.md"


def test_the_repository_has_no_incident_rule_hits():
    found = cir.check(ROOT)
    assert not found, "\n".join(map(str, found))


def test_cli_exit_codes(tmp_path, capsys):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "x.sh").write_text("git add -u\n")
    assert cir.main(["--root", str(tmp_path), "deploy/x.sh"]) == 1
    assert "IR005" in capsys.readouterr().out
    (tmp_path / "deploy" / "x.sh").write_text("git add deploy/x.sh\n")
    assert cir.main(["--root", str(tmp_path), "deploy/x.sh"]) == 0
    assert cir.main(["--list"]) == 0
