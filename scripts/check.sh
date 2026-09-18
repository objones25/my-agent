#!/usr/bin/env bash
# The offline gate, in one place. The pre-commit hook and the CI workflow both
# call this and re-list nothing, so the sequence cannot drift between them.
#
# Excluded deliberately: `-m live` and `-m eval`. They need the network and cost
# money, so they stay a human decision.
set -euo pipefail

step() { printf '\n=== %s ===\n' "$1"; }

step "ruff"
uv run ruff check .

step "mypy (strict)"
uv run mypy

step "pyright (standard) -- both checkers, on purpose: they disagree (F16)"
uv run pyright

step "pytest (offline; includes the src doctests)"
uv run pytest -q

step "coverage"
uv run pytest --cov -q

step "negative-space audit (gate)"
uv run python scripts/audit_negative_space.py src/ \
    --select NSP002,NSP003,NSP005,NSP006,NSP007

step "negative-space audit (advisory)"
uv run python scripts/audit_negative_space.py src/ \
    --select NSP001 --min-assertions 2 || true

printf '\nall gates passed\n'
