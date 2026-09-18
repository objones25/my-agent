#!/usr/bin/env bash
# The offline gate, in one place. The pre-commit hook and the CI workflow both
# call this and re-list nothing, so the sequence cannot drift between them.
#
# Excluded deliberately: `-m live` and `-m eval`. They need the network and cost
# money, so they stay a human decision.
set -euo pipefail

step() { printf '\n=== %s ===\n' "$1"; }

# First, because every step below is `uv run`, and `uv run` silently re-locks a
# stale lockfile as a side effect. That would let the gate itself mutate
# uv.lock and say nothing, and CI -- which runs `uv sync --locked` -- would then
# be the first thing to notice. Checking here keeps the two from disagreeing.
step "uv lock --check"
uv lock --check

step "ruff"
uv run ruff check .

step "mypy (strict)"
uv run mypy

step "pyright (standard) -- both checkers, on purpose: they disagree (F16)"
uv run pyright

step "pytest (offline; includes the src doctests)"
uv run pytest -q

# Reported, not gated. `[tool.coverage.report]` sets no `fail_under`, so this
# step cannot fail anything the plain pytest above did not already catch -- it
# exists to print the numbers where a reader of the gate output will see them.
# Deliberate: a coverage floor nobody agreed to is a floor someone lowers. It
# stays a separate run rather than folding --cov into the step above so that the
# suite is first measured exactly as a developer runs it, with no coverage
# tracing attached.
step "coverage (reported, not gated)"
uv run pytest --cov -q

step "negative-space audit (gate)"
uv run python scripts/audit_negative_space.py src/ \
    --select NSP002,NSP003,NSP005,NSP006,NSP007

step "negative-space audit (advisory)"
uv run python scripts/audit_negative_space.py src/ \
    --select NSP001 --min-assertions 2 || true

printf '\nall gates passed\n'
