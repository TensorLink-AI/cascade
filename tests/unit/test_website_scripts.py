"""Every function the website pages call is defined on the page.

The pages are static assets with no build step, so a sweep that deletes a
helper but leaves one caller behind only fails in a visitor's browser (the
cleanup sweep removed ``ckptRoundNo`` while ``wsLabel`` still called it). This
is a cheap lexical check: strip comments and string literals from each inline
``<script>``, collect every bare ``name(`` call, and require a definition for
it on the page (``function name``, a ``var/let/const`` binding, or a
parameter) unless it is a browser/JS builtin.

Lexical, not a parser: a false "undefined" means the scanner needs a new
builtin or binding form, never that the test should be skipped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAGES = [
    REPO / "cascade" / "website" / "index.html",
    REPO / "cascade" / "website" / "stakeholders.html",
]

_KEYWORDS = frozenset([
    "if", "for", "while", "switch", "catch", "function", "return", "typeof", "new",
    "delete", "void", "in", "of", "do", "else", "try", "throw", "await", "async",
    "yield", "with", "super", "this"
])
_BUILTINS = frozenset([
    "parseInt", "parseFloat", "isNaN", "isFinite", "String", "Number", "Boolean",
    "Array", "Object", "Math", "JSON", "Date", "fetch", "setTimeout",
    "clearTimeout", "setInterval", "clearInterval", "encodeURIComponent",
    "decodeURIComponent", "encodeURI", "Promise", "Set", "Map", "RegExp", "Error",
    "console", "document", "window", "alert", "requestAnimationFrame",
    "cancelAnimationFrame", "URL", "URLSearchParams", "Symbol", "BigInt",
    "getComputedStyle", "localStorage", "AbortController", "TextDecoder", "Intl",
    "structuredClone", "queueMicrotask", "atob", "btoa", "Uint8Array",
    "Float64Array", "IntersectionObserver", "ResizeObserver", "MutationObserver",
    "matchMedia", "navigator", "location", "history", "performance", "Blob",
    "confirm", "prompt", "WeakMap", "Proxy", "Reflect"
])


def _inline_scripts(html: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    ]


def _strip(js: str) -> str:
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js = re.sub(r"(?m)(^|[^:\\])//.*$", r"\1", js)
    js = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', js)
    js = re.sub(r"'(?:\\.|[^'\\\n])*'", "''", js)
    return re.sub(r"`(?:\\.|[^`\\])*`", "``", js, flags=re.S)


def _defined(js: str) -> set[str]:
    names = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", js))
    names |= set(re.findall(r"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)", js))
    for m in re.finditer(r"\b(?:var|let|const)\s+([^;]*)", js):
        names |= set(re.findall(r"(?:^|,)\s*([A-Za-z_$][\w$]*)\s*(?==|,|$|;)", m.group(1)))
    for m in re.finditer(r"function\s*[\w$]*\s*\(([^)]*)\)", js):
        names |= set(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
    for m in re.finditer(r"\(([^()]*)\)\s*=>", js):
        names |= set(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
    names |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", js))
    names |= set(re.findall(r"catch\s*\(\s*([\w$]+)", js))
    return names


def _called(js: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", js)}


def undefined_calls(html: str) -> set[str]:
    js = _strip("\n;\n".join(_inline_scripts(html)))
    return _called(js) - _defined(js) - _KEYWORDS - _BUILTINS


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_called_function_is_defined_on_the_page(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    assert _inline_scripts(html), f"{page.name} has no inline script to check"
    assert undefined_calls(html) == set()


def test_scanner_catches_a_deleted_helper() -> None:
    """The regression this file exists for: a caller outliving its helper."""
    page = (
        "<script>function roundNoOf(r){return r;}\n"
        "function wsLabel(ref){ var rn=ckptRoundNo(ref); return roundNoOf(rn); }</script>"
    )
    assert undefined_calls(page) == {"ckptRoundNo"}


def test_index_keeps_ckpt_round_no() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    assert "function ckptRoundNo(" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_inline_scripts_parse(page: Path, tmp_path: Path) -> None:
    """``node --check`` over every inline script (joined as the browser runs
    them): a syntax error blanks the whole page, and the lexical scan above
    cannot see one."""
    js = "\n;\n".join(_inline_scripts(page.read_text(encoding="utf-8")))
    f = tmp_path / (page.stem + ".js")
    f.write_text(js, encoding="utf-8")
    out = subprocess.run([shutil.which("node"), "--check", str(f)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
