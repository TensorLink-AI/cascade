"""W3: the service entry point with the code-path guard, the systemd unit and
the Caddy route (static tests on the files; nothing is installed, started or
reloaded). Contract: README.md "Service", "systemd unit", "Caddy route"."""

from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .helpers import CADDY, REPO, TAGS_FILE, UNIT

# deploy/laned/lib.sh laned_proc_patterns (intake_laned): how the restart and
# deploy_gate find the process.
PROC_PATTERNS = (
    r"^[^ ]*python[0-9.]*( -[A-Za-z]+)* [^ ]*bin/cascade-laned-intake( |$)",
    r"^[^ ]*python[0-9.]*( -[A-Za-z]+)* -m cascade\.lane_d(\.[a-z_]+)?( |$)",
)
SECRET_ENV = re.compile(r"(HIPPIUS_|AWS_|SECRET|TOKEN|PASSWORD|PASSWD|_KEY\b|API_KEY|"
                        r"BT_WALLET|WANDB_|HF_TOKEN)", re.IGNORECASE)
MEMORY_MAX_BYTES = 512 * 1024 * 1024

_SITECUSTOMIZE = """\
import os, socket
_m = os.environ.get("W3_NET_BLOCK_MARKER")
if _m:
    open(_m, "w").close()
def _refuse(*a, **k):
    raise OSError("W3 test: network connect is blocked")
socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
socket.create_connection = _refuse
"""


# ── the unit file ────────────────────────────────────────────────────────────


def parse_unit(text: str) -> dict[str, dict[str, list[str]]]:
    joined = re.sub(r"\\\n", " ", text)
    out: dict[str, dict[str, list[str]]] = {}
    sec = None
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
            continue
        assert sec is not None and "=" in line, f"unparseable unit line: {raw!r}"
        k, v = line.split("=", 1)
        out[sec].setdefault(k.strip(), []).append(" ".join(v.split()))
    return out


def _unit() -> dict:
    assert UNIT.is_file(), f"{UNIT.relative_to(REPO)} is missing (W3 PATHS)"
    return parse_unit(UNIT.read_text(encoding="utf-8"))


def _exec_argv(unit: dict) -> list[str]:
    starts = unit["Service"].get("ExecStart", [])
    assert len(starts) == 1, starts
    return shlex.split(starts[0])


def _env_assignments(unit: dict) -> dict[str, str]:
    env = {}
    for v in unit["Service"].get("Environment", []):
        for tok in shlex.split(v):
            k, _, val = tok.partition("=")
            env[k] = val
    return env


def _size(v: str) -> int:
    m = re.fullmatch(r"(\d+)([KMG]?)", v.strip())
    assert m, f"unparseable size {v!r}"
    return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]


def test_unit_has_the_three_sections():
    u = _unit()
    for sec in ("Unit", "Service", "Install"):
        assert sec in u, sec
    assert u["Install"].get("WantedBy") == ["multi-user.target"]


def test_unit_runs_python_dash_P_module_form_on_loopback():
    argv = _exec_argv(_unit())
    assert argv[0].startswith("/") and argv[0].endswith("/python"), argv[0]
    assert argv[1:4] == ["-P", "-m", "cascade.lane_d"], argv
    line = " ".join(argv)
    assert any(re.search(p, line) for p in PROC_PATTERNS), line
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1", argv
    assert "--port" in argv and argv[argv.index("--port") + 1].isdigit(), argv
    for flag in ("--state-dir", "--chain-toml"):
        assert flag in argv and argv[argv.index(flag) + 1].startswith("/"), (flag, argv)


def test_unit_declares_the_release_for_the_code_path_guard():
    u = _unit()
    env = _env_assignments(u)
    assert env.get("PYTHONPATH", "").startswith("/"), env
    assert "CASCADE_ALLOW_CODE_PATH_MISMATCH" not in env
    assert "CASCADE_ALLOW_CODE_PATH_MISMATCH" not in UNIT.read_text(encoding="utf-8")
    # a refused start (exit 78, EX_CONFIG) stays down instead of looping
    from cascade.shared.codepath import EXIT_CODE_PATH_MISMATCH

    prevent = " ".join(u["Service"].get("RestartPreventExitStatus", [])).split()
    assert str(EXIT_CODE_PATH_MISMATCH) in prevent, prevent
    assert u["Service"].get("Restart") == ["on-failure"]


def test_unit_ram_budget_and_oom_score():
    u = _unit()["Service"]
    assert "MemoryMax" in u, "the RAM budget is a MemoryMax= in the unit"
    assert _size(u["MemoryMax"][-1]) <= MEMORY_MAX_BYTES, u["MemoryMax"]
    assert u.get("OOMScoreAdjust") == ["-500"], u.get("OOMScoreAdjust")


def test_unit_carries_no_secret_and_no_wallet():
    env = _env_assignments(_unit())
    text = UNIT.read_text(encoding="utf-8")
    bad = [k for k in env if SECRET_ENV.search(k)]
    assert bad == [], f"secret-shaped Environment= names in the unit: {bad}"
    for needle in ("wallet", "coldkey", "BT_WALLET_PASSWORD", ".bittensor"):
        assert needle not in text, needle


# ── the Caddy route ──────────────────────────────────────────────────────────


def _caddy() -> str:
    assert CADDY.is_file(), f"{CADDY.relative_to(REPO)} is missing (W3 PATHS)"
    lines = [ln.split("#", 1)[0] for ln in CADDY.read_text(encoding="utf-8").splitlines()]
    return "\n".join(lines)


def _caddy_bytes(v: str) -> int:
    m = re.fullmatch(r"(\d+)\s*(B|KB|KiB|MB|MiB)?", v.strip())
    assert m, f"unparseable Caddy size {v!r}"
    mult = {None: 1, "B": 1, "KB": 1000, "KiB": 1024, "MB": 1000 ** 2, "MiB": 1024 ** 2}
    return int(m.group(1)) * mult[m.group(2)]


def test_caddy_routes_only_lane_d_paths_to_the_intake():
    text = _caddy()
    handles = re.findall(r"^\s*(handle\S*)\s+(\S+)\s*\{", text, flags=re.M)
    assert handles == [("handle", "/v1/lane-d/*")], handles
    ups = re.findall(r"reverse_proxy\s+(\S+)", text)
    assert len(ups) == 1, ups
    host, _, port = ups[0].rpartition(":")
    assert host in ("127.0.0.1", "localhost"), ups
    argv = _exec_argv(_unit())
    assert port == argv[argv.index("--port") + 1], (port, argv)
    for other in ("/v1/submit", "/v1/fund", "/v1/lane-b", "/v1/lane-c"):
        assert other not in text, other


def test_caddy_body_cap_is_within_the_app_cap():
    from cascade.lane_d import service

    m = re.search(r"request_body\s*\{[^}]*max_size\s+([^\n}]+)", _caddy())
    assert m, "the route sets request_body { max_size ... }"
    assert _caddy_bytes(m.group(1)) <= service.MAX_BODY_BYTES


def test_intake_port_differs_from_the_lane_a_intake():
    argv = _exec_argv(_unit())
    from cascade.funding.main import build_arg_parser

    lane_a = build_arg_parser().parse_args(["--queue-path", "/nonexistent/q.json"]).port
    assert int(argv[argv.index("--port") + 1]) != lane_a


# ── the entry point ──────────────────────────────────────────────────────────


def _sandbox(tmp_path: Path) -> Path:
    d = tmp_path / "w3_sandbox"
    for pkg in ("torch", "bittensor"):
        (d / pkg).mkdir(parents=True, exist_ok=True)
        (d / pkg / "__init__.py").write_text(
            f"raise ImportError('W3 test: {pkg} is blocked for the intake entry point')\n")
    (d / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    return d


def _run(tmp_path: Path, *args: str, release: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("AWS_") or k.startswith("CASCADE_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(_sandbox(tmp_path)), str(REPO)])
    env["CASCADE_RELEASE_PATH"] = release
    marker = tmp_path / "net_block_marker"
    env["W3_NET_BLOCK_MARKER"] = str(marker)
    proc = subprocess.run([sys.executable, "-P", "-m", "cascade.lane_d", *args],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120, check=False)
    assert marker.is_file(), "the network block (sitecustomize) did not load"
    return proc


def test_entry_point_help_without_torch_or_bittensor(tmp_path):
    proc = _run(tmp_path, "--help", release=str(REPO))
    assert proc.returncode == 0, proc.stderr[-3000:]
    for flag in ("--host", "--port", "--state-dir", "--chain-toml"):
        assert flag in proc.stdout, flag


def test_cli_defaults_bind_loopback():
    from cascade.lane_d.main import DEFAULT_HOST, DEFAULT_PORT, build_arg_parser

    assert DEFAULT_HOST == "127.0.0.1"
    ns = build_arg_parser().parse_args(["--state-dir", "/x", "--chain-toml", "/y"])
    assert ns.host == DEFAULT_HOST and ns.port == DEFAULT_PORT


def test_shadowed_release_refuses_to_start_with_exit_78(tmp_path):
    """The PR #431 guard: launched against another declared release, the
    service exits EX_CONFIG before it reads the chain or binds a port."""
    from cascade.shared.codepath import EXIT_CODE_PATH_MISMATCH

    elsewhere = tmp_path / "another-release"
    elsewhere.mkdir()
    state = tmp_path / "state"
    proc = _run(tmp_path, "--state-dir", str(state), "--chain-toml", str(REPO / "chain.toml"),
                "--port", "0", release=str(elsewhere))
    assert proc.returncode == EXIT_CODE_PATH_MISMATCH, (proc.returncode, proc.stderr[-3000:])
    assert "REFUSING TO START" in proc.stderr
    assert "cascade-laned-intake" in proc.stderr


def test_main_calls_the_guard_and_logs_the_up_line():
    path = REPO / "cascade" / "lane_d" / "main.py"
    assert path.is_file(), "cascade/lane_d/main.py is missing (W3 PATHS)"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", getattr(c.func, "attr", "")) for c in calls]
    assert "verify_release_code_path" in names, names
    guard = next(c for c in calls
                 if getattr(c.func, "id", getattr(c.func, "attr", "")) == "verify_release_code_path")
    assert isinstance(guard.args[0], ast.Constant) and guard.args[0].value == "cascade-laned-intake"
    # deploy_gate verify looks for "<service> up:" in <root>/intake-laned.log
    assert "intake-laned up:" in src


def test_lane_d_log_lines_are_tagged_and_covered(tmp_path):
    """Build rule 5 over cascade/lane_d/ with this WP's tag list."""
    proc = subprocess.run(
        [sys.executable, "-P", str(REPO / "scripts" / "laned" / "follower_patterns.py"),
         "--root", str(REPO), "--source", "cascade/lane_d", "--patterns", str(TAGS_FILE)],
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_intake_package_imports_no_torch_and_no_validator():
    """Static: no module in cascade/lane_d/ imports torch or the validator
    package at module level (the intake is its own small service)."""
    bad = []
    for p in sorted((REPO / "cascade" / "lane_d").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [("." * node.level) + node.module]
            for n in names:
                if n.split(".")[0] in ("torch", "bittensor") or "validator" in n.split("."):
                    bad.append(f"{p.name}: {n}")
    assert bad == [], bad
    assert (REPO / "cascade" / "lane_d" / "main.py").is_file()
