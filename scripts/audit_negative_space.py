#!/usr/bin/env python3
# VENDORED from the negative-space-programming skill. Do not hand-edit.
# Refresh: cp ~/.claude/skills/negative-space-programming/scripts/audit_negative_space.py scripts/
# pyright: reportAttributeAccessIssue=false, reportReturnType=false
"""Audit Python source for missing negative space.

Checks (all AST-based, standard library only):

  NSP001  function with fewer than the minimum number of runtime checks
  NSP002  assert on a tuple literal -- always true, never fires
  NSP003  side effect inside a check -- behaviour changes when checks are off
  NSP004  compound check `a and b` -- split so the failure names itself
  NSP005  unbounded `while True` loop -- no explicit iteration bound
  NSP006  bare `except:` -- catches SystemExit and KeyboardInterrupt too
  NSP007  swallowed exception -- handler neither raises nor logs
  NSP008  bare `assert` outside tests (--strict only) -- removed by python -O

Usage:
    python audit_negative_space.py PATH [PATH ...]
    python audit_negative_space.py src/ --min-assertions 2
    python audit_negative_space.py src/ --strict --json
    python audit_negative_space.py src/ --select NSP005,NSP007

Exits 1 if any finding is reported, 0 if clean. Suppress a line with a trailing
`# nsp: ignore` or `# nsp: ignore NSP005`.

This file audits clean against itself at `--min-assertions 1`, the right dial
for dispatch-heavy code; `assets/examples/after.py` is clean at the default 2.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

CHECK_FUNCS = frozenset({"require", "invariant", "precondition", "postcondition",
                         "ensure", "check", "unreachable", "verify", "_require"})
LOG_ATTRS = frozenset({"debug", "info", "warning", "warn", "error", "exception",
                       "critical", "log", "print", "capture_exception"})
MUTATORS = frozenset({"append", "extend", "insert", "pop", "remove", "clear",
                      "update", "add", "discard", "sort", "write", "writelines",
                      "send", "close", "commit", "execute", "setdefault",
                      "popitem", "seek", "flush", "put", "get_nowait"})
BOUND_HINTS = frozenset({"bounded", "range", "islice", "takewhile", "zip"})
DENSITY_EXEMPT_DECORATORS = frozenset({"overload", "abstractmethod", "property",
                                       "setter", "fixture", "singledispatch"})
SKIP_DIRS = frozenset({".venv", "venv", "__pycache__", "build", ".git",
                       "site-packages", ".tox", ".mypy_cache"})

SEVERITY = {
    "NSP000": "error",
    "NSP001": "warning", "NSP002": "error", "NSP003": "error", "NSP004": "warning",
    "NSP005": "error", "NSP006": "error", "NSP007": "error", "NSP008": "warning",
}


class AuditBug(AssertionError):
    """The auditor itself is in an impossible state."""


def _require(condition: object, message: str) -> None:
    """Local `require` so this script stays dependency-free. Survives -O."""
    if not condition:
        raise AuditBug(message)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    severity: str
    symbol: str
    message: str

    def render(self) -> str:
        return (f"{self.path}:{self.line}: {self.code} [{self.severity}] "
                f"{self.symbol}: {self.message}")


def _is_test_file(path: Path) -> bool:  # nsp: ignore NSP001  pure name predicate, no state
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py")
            or "tests" in path.parts or "test" in path.parts
            or name == "conftest.py")


def _suppressions(source: str) -> dict[int, set[str]]:
    """Map line number -> suppressed codes ('*' means every code)."""
    _require(isinstance(source, str), "source must be text")
    out: dict[int, set[str]] = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        marker = line.find("# nsp: ignore")
        if marker == -1:
            continue
        rest = line[marker + len("# nsp: ignore"):].strip()
        codes = {c.strip(" ,") for c in rest.split() if c.strip(" ,").startswith("NSP")}
        out[lineno] = codes or {"*"}
    _require(all(v for v in out.values()), "empty suppression set would silence nothing")
    return out


def _calls_named(node: ast.AST, names: frozenset[str]) -> bool:
    _require(isinstance(node, ast.AST), f"expected an AST node, got {type(node).__name__}")
    _require(names, "empty name set would always return False")
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
            if isinstance(func, ast.Attribute) and func.attr in names:
                return True
    return False


def _is_check_stmt(node: ast.stmt) -> bool:
    """An assert, a call to a check helper, or a guard clause that raises."""
    _require(isinstance(node, ast.stmt), f"expected a statement, got {type(node).__name__}")
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and _calls_named(node, CHECK_FUNCS):
        return True
    if isinstance(node, ast.If) and len(node.body) == 1:
        return isinstance(node.body[0], ast.Raise)
    return False


def _count_checks(fn: ast.AST) -> int:
    """Checks anywhere in the function, excluding nested function bodies."""
    _require(isinstance(fn, ast.AST), "expected an AST node")
    total = 0
    stack: list[ast.AST] = [fn]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.stmt) and _is_check_stmt(child):
                total += 1
            stack.append(child)
    _require(total >= 0, "check count went negative")
    return total


def _body_statements(fn: ast.AST) -> list[ast.stmt]:
    """Body with the docstring removed."""
    _require(hasattr(fn, "body"), f"{type(fn).__name__} has no body")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                       and isinstance(s.value, ast.Constant)
                                       and isinstance(s.value.value, str))]
    _require(len(body) <= len(fn.body), "docstring filter added statements")
    return body


def _is_trivial(fn: ast.AST) -> bool:
    """Stubs, protocol members, and one-statement wrappers are exempt from density."""
    _require(isinstance(fn, ast.AST), "expected an AST node")
    body = _body_statements(fn)
    if not body:
        return True
    if len(body) == 1:
        # A single-statement wrapper has little space to get wrong -- unless that
        # statement is a loop or a try, which carry their own negative space.
        return not isinstance(body[0], (ast.While, ast.For, ast.AsyncFor, ast.Try))
    return False


def _decorator_names(fn: ast.AST) -> set[str]:
    _require(isinstance(fn, ast.AST), "expected an AST node")
    _require(hasattr(fn, "decorator_list"), f"{type(fn).__name__} cannot be decorated")
    names: set[str] = set()
    for dec in fn.decorator_list:
        for sub in ast.walk(dec):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
    return names


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    """True when the handler neither re-raises nor reports."""
    _require(isinstance(handler, ast.ExceptHandler), "expected an except handler")
    _require(handler.body, "an except handler cannot have an empty body")
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise):
            return False
    if _calls_named(handler, LOG_ATTRS):
        return False
    meaningful = [s for s in handler.body
                  if not isinstance(s, (ast.Pass, ast.Continue, ast.Break))]
    return not meaningful


def _loop_is_bounded(loop: ast.While) -> bool:
    """A `while True` counts as bounded when its body checks an iteration count
    or walks a bounded iterator."""
    _require(isinstance(loop, ast.While), "expected a while loop")
    _require(loop.body, "a while loop cannot have an empty body")
    for sub in ast.walk(loop):
        if isinstance(sub, ast.stmt) and _is_check_stmt(sub):
            return True
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in BOUND_HINTS:
                return True
    return False


class Auditor(ast.NodeVisitor):
    def __init__(self, path: Path, min_assertions: int, strict: bool):
        _require(min_assertions >= 0, f"min_assertions must be non-negative, got {min_assertions}")
        _require(path.suffix == ".py", f"not a Python file: {path}")
        self.path = path
        self.min_assertions = min_assertions
        self.strict = strict
        self.is_test = _is_test_file(path)
        self.findings: list[Finding] = []
        self._scope: list[str] = []

    def add(self, node: ast.AST, code: str, message: str) -> None:
        _require(code in SEVERITY, f"unknown finding code {code}")
        _require(getattr(node, "lineno", 0) > 0, f"{code} on a node with no line number")
        self.findings.append(Finding(
            path=str(self.path), line=node.lineno, code=code,
            severity=SEVERITY[code], symbol=".".join(self._scope) or "<module>",
            message=message,
        ))

    # --- scopes -----------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        depth = len(self._scope)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()
        _require(len(self._scope) == depth, "scope stack unbalanced after class")

    def _visit_function(self, node) -> None:
        depth = len(self._scope)
        self._scope.append(node.name)
        if not self.is_test:
            self._check_density(node)
        self.generic_visit(node)
        self._scope.pop()
        _require(len(self._scope) == depth, "scope stack unbalanced after function")

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _check_density(self, node) -> None:
        _require(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)),
                 "density applies to functions only")
        if _is_trivial(node) or node.name.startswith("__"):
            return
        if _decorator_names(node) & DENSITY_EXEMPT_DECORATORS:
            return
        count = _count_checks(node)
        if count < self.min_assertions:
            stmts = len(_body_statements(node))
            self.add(node, "NSP001",
                     f"{count} runtime check(s) in {stmts} statements; minimum is "
                     f"{self.min_assertions}. What must never be true when this returns?")

    # --- checks -----------------------------------------------------------
    def visit_Assert(self, node: ast.Assert) -> None:
        _require(isinstance(node, ast.Assert), "visit_Assert on a non-assert node")
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.add(node, "NSP002",
                     "assert on a tuple literal is always true; drop the parentheses "
                     "or use require(cond, msg)")
        self._check_compound(node, node.test, "assert")
        self._check_side_effect(node, node.test, "assert")
        if self.strict and not self.is_test:
            self.add(node, "NSP008",
                     "plain assert is removed by python -O; use require() for a check "
                     "that must survive production")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        _require(isinstance(node, ast.Expr), "visit_Expr on a non-expression node")
        if isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in CHECK_FUNCS and node.value.args:
                self._check_compound(node, node.value.args[0], name)
                self._check_side_effect(node, node.value.args[0], name)
        self.generic_visit(node)

    def _check_compound(self, node: ast.AST, test: ast.expr, kind: str) -> None:
        _require(isinstance(test, ast.expr), "a check's condition must be an expression")
        _require(kind, "every finding needs the kind of check that produced it")
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            self.add(node, "NSP004",
                     f"compound {kind}: split into {len(test.values)} checks so a "
                     f"failure names the half that broke")

    def _check_side_effect(self, node: ast.AST, test: ast.expr, kind: str) -> None:
        _require(isinstance(test, ast.expr), "a check's condition must be an expression")
        for sub in ast.walk(test):
            if isinstance(sub, ast.NamedExpr):
                self.add(node, "NSP003",
                         f"walrus assignment inside {kind}; the binding vanishes when "
                         f"checks are disabled")
                return
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr in MUTATORS:
                self.add(node, "NSP003",
                         f"{kind} calls .{sub.func.attr}(), which mutates state; "
                         f"checks must be pure")
                return

    def visit_While(self, node: ast.While) -> None:
        _require(isinstance(node, ast.While), "visit_While on a non-loop node")
        test = node.test
        infinite = isinstance(test, ast.Constant) and bool(test.value) is True
        if infinite and not _loop_is_bounded(node):
            self.add(node, "NSP005",
                     "`while True` with no iteration bound; add an explicit maximum "
                     "and assert it, or mark it `# nsp: ignore NSP005` if it is a "
                     "deliberate event loop")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        _require(isinstance(node, ast.ExceptHandler), "visit_ExceptHandler on a non-handler")
        if node.type is None:
            self.add(node, "NSP006",
                     "bare `except:` also catches SystemExit and KeyboardInterrupt; "
                     "name the exceptions you expect")
        elif _handler_swallows(node):
            self.add(node, "NSP007",
                     "handler neither raises nor logs; the failure is erased. Re-raise, "
                     "log with context, or comment why it is ignorable")
        self.generic_visit(node)


def audit_file(path: Path, min_assertions: int, strict: bool) -> list[Finding]:
    _require(path.is_file(), f"not a file: {path}")
    _require(min_assertions >= 0, f"min_assertions must be non-negative, got {min_assertions}")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Finding(str(path), exc.lineno or 0, "NSP000", "error",
                        "<module>", f"could not parse: {exc.msg}")]

    auditor = Auditor(path, min_assertions, strict)
    auditor.visit(tree)

    suppressed = _suppressions(source)
    kept = [f for f in auditor.findings
            if not ({"*", f.code} & suppressed.get(f.line, set()))]
    _require(len(kept) <= len(auditor.findings), "suppression invented findings")
    return kept


def collect(paths: list[str]) -> list[Path]:
    _require(paths, "no paths given")
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        _require(p.exists(), f"path does not exist: {raw}")
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*.py")
                                if not (SKIP_DIRS & set(q.parts))))
        elif p.suffix == ".py":
            files.append(p)
    _require(all(f.suffix == ".py" for f in files), "collected a non-Python file")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="files or directories to audit")
    parser.add_argument("--min-assertions", type=int, default=2,
                        help="minimum runtime checks per non-trivial function (default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="also flag plain `assert` outside tests (NSP008)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--select", default="",
                        help="comma-separated codes to report, e.g. NSP005,NSP007")
    args = parser.parse_args(argv)

    _require(args.min_assertions >= 0, "--min-assertions cannot be negative")
    selected = {c.strip().upper() for c in args.select.split(",") if c.strip()}
    _require(selected <= set(SEVERITY), f"unknown code(s): {sorted(selected - set(SEVERITY))}")

    files = collect(args.paths)
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path, args.min_assertions, args.strict))
    if selected:
        findings = [f for f in findings if f.code in selected]
    findings.sort(key=lambda f: (f.path, f.line, f.code))

    if args.json:
        print(json.dumps({"files_scanned": len(files),
                          "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        for f in findings:
            print(f.render())
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        summary = ", ".join(f"{c}={n}" for c, n in sorted(counts.items())) or "none"
        print(f"\n{len(files)} file(s) scanned, {len(findings)} finding(s): {summary}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
