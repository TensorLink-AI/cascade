"""W-LAUNCH: the 2026-11-27 root-cause analysis as CI guards (owner 2026-11-27
~20:50, "make sure we have patches, and add any lessons to lint").

docs/launch-lessons-2026-11.md gains a dated section: one row per 2026-11-27
incident with its root cause, merged fix PR and guard, and an OPEN list. The
new automated guards are IR009 (an ssh reply's raw stdout parsed as JSON, #983)
and three torch-free unit scans named below. Each test fails on a base without
this PR by assertion. All torch-free: AST/regex checks only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "launch-lessons-2026-11.md"


def _cir():
    path = ROOT / "scripts" / "check_incident_rules.py"
    spec = importlib.util.spec_from_file_location("check_incident_rules_rca1127", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    rules = {r.id: r for r in mod.RULES}
    assert "IR009" in rules, "no incident rule for JSON parsed from an ssh reply's raw stdout"
    return mod, rules["IR009"]


# The pre-#983 lane B offload step (lane_b_offload.py:371-376 at ea860300).
PRE_983 = '''\
class LaneBOffload:
    def _call(self, host, op, run_dir, stdin):
        cmd = (f"{src}cd {q(host.workdir)} && mkdir -p {q(root)} && {worker} && "
               f"cat {q(run_dir + '/result.json')}")
        proc = ok(self._ssh(host, cmd, left(), stdin), "worker")
        doc = json.loads(proc.stdout)
'''


def test_ir009_flags_the_19_27_settlement_reply(tmp_path):
    cir, rule = _cir()
    rel = "cascade/trainer/lane_b_offload.py"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(PRE_983)
    assert [(h.rule, h.line) for h in cir.check(tmp_path, [rel], (rule,))] == [("IR009", 6)]
    (tmp_path / rel).write_text(PRE_983.replace("json.loads(", "pod_reply_json("))
    assert cir.check(tmp_path, [rel], (rule,)) == []


def test_ir009_is_clean_on_the_repository():
    cir, rule = _cir()
    found = cir.check(ROOT, None, (rule,))
    assert not found, "\n".join(map(str, found))


def test_ir009_is_documented():
    book = (ROOT / "docs" / "INCIDENT_RULES.md").read_text()
    assert "## IR009" in book, "IR009 has no section in docs/INCIDENT_RULES.md"


FIX_PRS_1127 = ("#960", "#962", "#964", "#965", "#966", "#967", "#970", "#974", "#976",
                "#979", "#980", "#982", "#983")
NEW_GUARDS = ("IR009", "test_autograd_functions_have_bf16_tests",
              "test_lane_b_trainer_worker_contract_parity",
              "test_eval_pod_cache_dirs_on_data_mount")


def _section() -> str:
    assert DOC.is_file(), "missing docs/launch-lessons-2026-11.md"
    text = DOC.read_text()
    head = "## 2026-11-27 root-cause analysis"
    assert head in text, "lessons doc has no 2026-11-27 RCA section"
    return text.split(head, 1)[1].split("\n## ", 1)[0]


def test_lessons_doc_has_the_2026_11_27_rca_section():
    sec = _section()
    assert "### OPEN" in sec, "the RCA section does not list its OPEN incidents first"
    assert sec.index("### OPEN") < sec.index("### Incidents")
    for pr in FIX_PRS_1127:
        assert pr in sec, f"the 2026-11-27 section has no row for {pr}"
    for guard in NEW_GUARDS:
        assert guard in sec, f"the 2026-11-27 section does not name the guard {guard}"
        if guard.startswith("test_"):
            assert (ROOT / "tests" / "unit" / f"{guard}.py").is_file(), f"missing {guard}.py"


def test_checklist_has_the_pre_paid_leg_smoke_run():
    text = DOC.read_text()
    checklist = " ".join(text.split("## Pre-upgrade checklist", 1)[1]
                         .split("\n## ", 1)[0].split())
    assert "short real-environment smoke run of any new image before a paid leg" in checklist
    assert "DISPATCH era" in checklist, "no pre-restart re-attach check (row 32)"
    assert "never retrained" in checklist, "no rescue-not-retrain rule (row 43)"
