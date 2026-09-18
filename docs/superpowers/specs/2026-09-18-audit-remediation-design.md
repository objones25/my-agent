# Audit remediation: close the gaps the gates do not cover

A full audit of the repo on 2026-09-18 found every mechanical gate green — 240 tests, both
type checkers, ruff, the NSP CI gate, the test audit, recorded versions matching the installed
wheels, no tracked secrets, `SecretStr` verified non-leaking. The findings are all in what those
gates *do not reach*: one real contract defect, three structural gaps, and documentation that has
drifted from the code.

This spec covers all eight findings, ordered by dependency rather than severity.

## What this builds

1. **`AgentConfig` that cannot be invalidated after construction.** `frozen=True` today blocks
   rebinding a field but not mutating the list behind it, so `__post_init__`'s validation is
   defeatable on the two capability-relevant fields.
2. **Two speculative helpers removed.** `check_shape` and `check_finite` have zero call sites and
   serve a domain the project has deliberately not chosen.
3. **The contract helpers inside the default test gate.** `negative_space.py` is the only module
   with no test file, and its doctests are invisible to `uv run pytest`.
4. **The mirror's error paths tested.** `on_chain_error` and `on_llm_error` are driven zero times,
   in the module whose whole rationale is that crashed runs matter most.
5. **One gate, three layers.** A single script owning the command sequence, invoked by a
   pre-commit hook and by a GitHub Actions workflow.
6. **A README, and documentation that matches the code.**

## Out of scope

- `DTZ001` in `test_run_log_path_rejects_a_naive_timestamp`. A naive datetime is that test's input.
  The project's ruff config does not select `DTZ`, so a `noqa` would trip `RUF100`.
- The weave shutdown wait (F23). Diagnosed, unfixable from code, already documented.
- `ModelConfig`. Every field is a scalar, so its "cannot be invalidated after construction" claim
  is already true. Only `AgentConfig` needs the change.
- Any new agent capability. The allowlist, the permission defaults and the backend are untouched.
- Moving the socket guard to a root `conftest.py` — see Open questions.
- The live and eval suites. They cost money and need the network; the gate stays offline.

## Verified facts

Checked against the installed wheels and this repo on 2026-09-18. Re-verify if these move.

**`--doctest-modules` does not reach `src/` on its own.** `testpaths = ["tests"]` confines
collection, so adding the flag to `addopts` collects **zero** src doctests
(`uv run pytest --doctest-modules --collect-only -q | grep -c negative_space` → `0`). `src` has to
join `testpaths`.

**Three of the four doctests fail under pytest while passing under `python -m doctest`.** The
expected text is `negative_space.CheckFailed: ...`; pytest imports the module as
`my_agent.negative_space`, so the actual exception line is
`my_agent.negative_space.CheckFailed: ...`. No literal satisfies both runners, which is why the
examples get rewritten rather than the flag simply added.

**The doctest inventory is four examples across four helpers**, of which three raise —
`require`, `bounded` and `check_finite` — and those three are exactly the three that fail.
`check_shape` has no raising example and is the one that passes. After Task 2 removes
`check_shape` and `check_finite`, **two raising examples remain**, both currently failing.

**`unreachable()` has no doctest at all** and is not among the collected items, so it has no
automated coverage of any kind today. Task 3's test file is its first.

**The whole offline gate sequence takes 6 seconds** (ruff, mypy, pyright, pytest, NSP gate;
pyright is the slowest single step at 1.6s). This is the measurement that removes the need for
fast/slow tiers in the check script.

**`require()` survives `python -O`** — it raises `CheckFailed` rather than using `assert`. Also,
the suite no longer "passes silently" under `-O` as CLAUDE.md claims: pytest emits
`PytestConfigWarning`, `filterwarnings = ["error"]` makes it fatal, and the run exits `1`.

**`create_deep_agent` has 16 keyword-only parameters**, not the 17 CLAUDE.md:165 states. The same
file's Verified API Facts list is correct (diffed name by name against `inspect.signature`).

**`pre-commit` is neither installed nor declared** in `[dependency-groups] dev`. Task 5 adds it.
Environment: uv 0.6.1, Python 3.13.2, `.python-version` pinned to `3.13`.

**`WEAVE_TRACE_LANGCHAIN` defaults on.** `weave/integrations/langchain/langchain.py:472` sets it
to `"true"` when unset, so its absence from `.env.example` is a documentation nicety, not a
blocker — `WeaveTracing.activate()`'s failure message names it.

## Architecture

### Task 1 — `AgentConfig` immutability

`AgentConfig` declares `tools`, `middleware` and `permissions` as `Sequence`, stores whatever the
caller passed, and validates it in `__post_init__`. Demonstrated hole:

```
constructed with 1 rule; __post_init__ validated it
caller mutated the list afterwards -> 2 rules now
frozen?  FrozenInstanceError (assignment blocked)
_agent_kwargs accepted the poisoned list: ['FilesystemPermission', 'str']
```

`__post_init__` coerces each sequence field to a tuple **before** validating, so the object that
gets validated is the immutable one:

```python
object.__setattr__(self, "tools", tuple(self.tools))
object.__setattr__(self, "middleware", tuple(self.middleware))
object.__setattr__(self, "permissions", tuple(self.permissions))
```

`object.__setattr__` because the dataclass is frozen. Downstream is unaffected: `_agent_kwargs`
already does `list(config.permissions) or None`, and `create_deep_agent` takes `Sequence` for
`tools` and `middleware`.

### Task 2 — remove the speculative helpers

Delete `check_shape` and `check_finite` from `negative_space.py` and from `__all__`. Both have
zero call sites in `src/` and `tests/`; every apparent use elsewhere is prose in a comment. They
are tensor-shape and NaN-loss helpers for a domain CLAUDE.md says is undecided on purpose, which
is the YAGNI rule the project applies everywhere else.

`bounded()` and `unreachable()` stay. `bounded()` is the agent-loop bound CLAUDE.md explicitly
promises; `unreachable()` is a three-line idiom. Both get real tests in Task 3.

Deleting `check_shape` removes a cited worked example in two places: `CLAUDE.md:238`
("`check_shape` in `negative_space.py` is the worked example") and `docs/findings.md:174`, which
names both `check_shape` and `compiled_tool_names`. The replacement is ready-made —
`capabilities.compiled_tools` uses the same explicit-`raise` pattern for the same reason (mypy
cannot narrow through a helper call) and carries a comment saying so. Both citations re-point at
`compiled_tools`; findings.md keeps its second example but names the function that now holds the
code, since `compiled_tool_names` delegates to it.

### Task 3 — the contract helpers inside the gate

Three changes, in this order:

1. **Rewrite the two remaining raising doctests to be runner-agnostic** — `require`'s and
   `bounded`'s, the survivors of Task 2. Instead of a `Traceback` block whose exception line
   depends on the module's import name:

   ```
   >>> try:
   ...     list(bounded(range(10), 3, name="retries"))
   ... except CheckFailed as exc:
   ...     print(exc)
   retries exceeded its bound of 3 iterations
   ```

   This asserts the message *exactly* — stricter than the `...` traceback form — and passes under
   both runners.

2. **Make pytest collect them.** `testpaths = ["tests", "src"]` and `--doctest-modules` in
   `addopts`. Collecting `src/` imports every module, which runs the load-time contract checks;
   they pass today and failing loudly is the intent.

3. **Add `tests/test_negative_space.py`**, restoring the one-file-per-module rule. Doctests are
   documentation that happens to run; the failure paths, boundaries and category discipline belong
   in real tests: `require` raising `CheckFailed` and carrying its message, `CheckFailed` being an
   `AssertionError` subclass, `unreachable` always raising, `bounded` at `limit` and `limit + 1`,
   `bounded` rejecting a limit below 1, and `bounded` being lazy (it must not consume the iterable
   before yielding).

Then **drop `uv run python -m doctest ...` from CLAUDE.md's command list.** Two runners with
divergent expectations is the defect; one runner is the fix.

### Task 4 — the mirror's error paths

`tests/test_mirror.py` drives seven of nine callbacks. Add coverage for `on_chain_error` and
`on_llm_error`, asserting the same properties the existing `on_tool_error` test does: the record
type, `error_type`, the message, and that `name` is absent (per the module's documented contract
that only `*_start` records carry a name).

### Task 5 — one gate, three layers

**`scripts/check.sh`** is the single source of truth. No arguments, no tiers — the whole offline
sequence takes 6 seconds. `set -euo pipefail`, echo each step, exit non-zero on the first failure:

```
uv run ruff check .
uv run mypy
uv run pyright
uv run pytest -q                 # includes the doctests after Task 3
uv run pytest --cov -q
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007
uv run python scripts/audit_negative_space.py src/ --select NSP001 --min-assertions 2 || true
```

The last line stays advisory, matching CLAUDE.md. The live and eval suites are excluded
deliberately: they need network and cost money.

**`.pre-commit-config.yaml`** gets one `repo: local` hook of `language: system` that runs
`scripts/check.sh`, with `pass_filenames: false` and `always_run: true`. It does not re-list the
commands. `pre-commit` joins `[dependency-groups] dev`.

**`.github/workflows/ci.yml`** installs uv, runs `uv sync`, then `scripts/check.sh`. Nothing else
— the workflow must not become a second copy of the sequence. There is no remote yet; this exists
so the gate works the day one appears.

`scripts/` already holds a vendored file excluded from ruff and mypy; the new script is shell, so
neither tool is affected.

### Task 6 — README and documentation drift

**`README.md` is 0 bytes** while `pyproject.toml` declares it, and `description` is still
`"Add your description here"`. The README covers: what the project is (a deep-agent harness built
on contracts, evals and observability, with the domain deliberately undecided), how to install and
run it, the command list, a short architecture orientation pointing at CLAUDE.md and
`docs/findings.md`, and its status. It does not duplicate CLAUDE.md.

Drift fixes: CLAUDE.md:165 `17` → `16`; the `-O` claim corrected to what actually happens;
`WEAVE_TRACE_LANGCHAIN` added to `.env.example` with a note that weave defaults it on; the
`negative_space.py` helper lists at CLAUDE.md:210 and :475 updated for the deletion; the doctest
command removed from the command block; `scripts/check.sh` documented as the way to run everything.

## Testing

Every task follows the project's established cycle, the same one the existing plan document uses:
write the failing test, run it and watch it fail for the right reason, implement the minimum,
watch it pass, **mutation-check** by reverting the change and confirming the test goes red, then
the full gate and a commit.

Specific tests that must exist and must be shown to fail first:

| Task | Test | Mutation that must turn it red |
|---|---|---|
| 1 | Post-construction mutation of `permissions` is rejected | Remove the tuple coercion |
| 1 | Same for `middleware` and `tools` | Coerce only `permissions` |
| 3 | `bounded` yields exactly `limit` items and raises on `limit + 1` | Remove the bound check |
| 3 | `bounded` is lazy | Materialise the iterable inside `bounded` |
| 3 | `unreachable` always raises | Make it return `None` |
| 4 | `on_chain_error` writes a `chain_error` record with `error_type` | Drop the handler |
| 4 | `on_llm_error` likewise | Drop the handler |

Task 5 is verified by running the script, then deliberately breaking one gate (a ruff violation in
a scratch file) and confirming a non-zero exit — the script is only worth having if it fails.

Task 2's verification is the absence of regression: the full gate stays green after the deletion,
and the doc citations resolve to real code.

## Documentation changes

- `README.md` — written from empty.
- `CLAUDE.md` — the `17` → `16` fix, the `-O` claim, the helper lists, the command block gaining
  `scripts/check.sh` and losing the standalone doctest command.
- `.env.example` — `WEAVE_TRACE_LANGCHAIN`.
- `docs/findings.md` — the `check_shape` worked-example citation re-pointed. No new F-number: none
  of this is newly verified library behaviour. The two doctest gotchas in Verified facts above are
  *pytest configuration* facts and belong in this spec and in CLAUDE.md's testing section, not in a
  findings entry about a dependency.
- `pyproject.toml` — a real `description`, `pre-commit` in dev deps, `testpaths`, `addopts`.

## Open questions

- **The socket guard does not cover `src/` doctests.** `tests/conftest.py` is directory-scoped, so
  once `src` joins `testpaths` the doctests run without the network block or the RNG seeding. Fine
  for pure contract helpers, and moving the guard to a root `conftest.py` would change what applies
  to every test in the project. Deliberately deferred; noted here so it is a decision rather than
  an oversight.
- **Three config gaps were found and are not addressed:** `xfail_strict` is unset, coverage has no
  `fail_under`, and the import mode is the default `prepend` rather than `importlib`. None has bitten
  yet — there are no `xfail`s in the suite, and coverage is reported rather than gated. Worth doing
  when a coverage floor is actually wanted; a floor nobody agreed to is a floor someone lowers.
- **`bounded()` still has no call site** after this work. It is kept on CLAUDE.md's explicit promise
  that it is the agent-loop bound. If the next domain slice does not use it, deleting it is the
  honest follow-up.
