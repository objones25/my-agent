# Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight findings from the 2026-09-18 audit: make `AgentConfig`'s validation
undefeatable, delete two speculative helpers, bring the contract helpers inside the default test
gate, test the mirror's error paths, give the project a real CI gate in three layers, and fix the
documentation that has drifted from the code.

**Architecture:** Six independent tasks in dependency order. Task 1 is a contract fix in
`agent.py`. Task 2 deletes code and must precede Task 3, which writes the tests that would
otherwise cover it. Task 3 changes pytest configuration so `src/` doctests run in the default
suite. Task 4 adds two tests. Task 5 puts the whole gate sequence in one shell script that a
pre-commit hook and a GitHub Actions workflow both call, so the command list exists once. Task 6
is documentation.

**Tech Stack:** Python 3.13.2, uv 0.6.1, pytest 9.1.1, mypy 2.3.1 (strict), pyright 1.1.414
(standard), ruff 0.16.8, deepagents 0.7.15, langchain-core 1.6.3.

**Spec:** `docs/superpowers/specs/2026-09-18-audit-remediation-design.md`

## Global Constraints

- Never write an API call from memory. `ctx7` for intent, then `inspect` against the installed
  wheel for truth. `inspect` wins.
- `require()` from `my_agent.negative_space` for programmer errors; never bare `assert` in `src/`.
  `python -O` deletes `assert` statements entirely.
- A test expecting a tripped `require()` names `CheckFailed`, never `AssertionError` — the latter
  is its base class and is also satisfied by a bare `assert`.
- `filterwarnings = ["error"]` is set. A new warning fails the build; fix it or scope an ignore
  matched on category **and** message **and** module. Do not widen the setting.
- Every task ends with the full gate green and one commit. Commit messages: imperative subject,
  prose body explaining *why*, no bullet lists, ASCII only (`--` not an em dash), ending with
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- After writing a test and before implementing, run it and confirm it fails **for the intended
  reason**. After it passes, revert the implementation and confirm the test goes red. A test that
  survives its mutant is decorative and must be rewritten.
- Do not add an agent capability. The tool allowlist, the permission defaults and the backend are
  out of scope.
- The live (`-m live`) and eval (`-m eval`) suites are excluded from every gate: they need the
  network and cost money.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/my_agent/agent.py` | Modify | `AgentConfig.__post_init__` coerces its sequence fields to tuples before validating |
| `src/my_agent/negative_space.py` | Modify | Lose `check_shape` and `check_finite`; two doctests rewritten runner-agnostically; `__main__` doctest block removed |
| `tests/test_negative_space.py` | Create | First unit tests for the contract helpers; restores one-file-per-module |
| `tests/test_agent.py` | Modify | Three tests that the config cannot be invalidated after construction |
| `tests/test_mirror.py` | Modify | Two tests for `on_chain_error` and `on_llm_error` |
| `pyproject.toml` | Modify | `--doctest-modules`, `testpaths`, `pre-commit` dev dep, real `description` |
| `scripts/check.sh` | Create | The single source of truth for the offline gate sequence |
| `.pre-commit-config.yaml` | Create | One local hook that calls `scripts/check.sh` |
| `.github/workflows/ci.yml` | Create | Installs uv, syncs, calls `scripts/check.sh` |
| `README.md` | Modify | Written from empty |
| `CLAUDE.md` | Modify | Drift fixes; command block gains the script and loses the doctest command |
| `.env.example` | Modify | `WEAVE_TRACE_LANGCHAIN` |
| `docs/findings.md` | Modify | Worked-example citation re-pointed |

---

## Task 1: `AgentConfig` cannot be invalidated after construction

**Files:**
- Modify: `src/my_agent/agent.py` — `AgentConfig.__post_init__`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `AgentConfig.tools`, `.middleware`, `.permissions` are `tuple` after construction,
  whatever sequence type the caller passed. No signature changes: the fields stay annotated
  `Sequence[...]`, and `as_kwargs()` keeps returning them unchanged.

**Background the implementer needs:**

`AgentConfig` is `@dataclass(frozen=True, slots=True)`. `frozen=True` blocks *rebinding* a field
but does nothing about mutating the object a field points at. The three sequence fields are stored
exactly as the caller passed them, so validation in `__post_init__` is defeatable:

```
constructed with 1 rule; __post_init__ validated it
caller mutated the list afterwards -> 2 rules now
frozen?  FrozenInstanceError (assignment blocked)
_agent_kwargs accepted the poisoned list: ['FilesystemPermission', 'str']
```

`object.__setattr__` is how a frozen dataclass sets a field, and it works with `slots=True` —
verified. Coerce **before** the validation loops so the object that gets validated is the
immutable one.

`ModelConfig` needs no equivalent change: every one of its fields is a scalar, so its docstring's
claim that it "cannot be invalidated after construction" is already true.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`, in the `AgentConfig` section alongside the existing validation tests:

```python
def test_agent_config_freezes_the_rules_it_was_given(
    deny_secrets: FilesystemPermission,
) -> None:
    """`frozen=True` stops the field being rebound, not the list behind it being
    mutated. Without coercion a caller passes validation and then appends
    anything they like, and `_agent_kwargs` forwards it to create_deep_agent."""
    rules = [deny_secrets]
    config = AgentConfig(permissions=rules)

    rules.append("not a permission at all")  # type: ignore[arg-type]

    assert config.permissions == (deny_secrets,)


def test_agent_config_freezes_the_middleware_it_was_given() -> None:
    entries = [TodoListMiddleware()]
    config = AgentConfig(middleware=entries)

    entries.append("not middleware")  # type: ignore[arg-type]

    assert len(config.middleware) == 1


def test_agent_config_freezes_the_tools_it_was_given() -> None:
    @tool
    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    tools = [echo]
    config = AgentConfig(tools=tools)

    tools.append("not a tool")  # type: ignore[arg-type]

    assert config.tools == (echo,)
```

`TodoListMiddleware` and `tool` are already imported in this file. `deny_secrets` is a fixture in
`tests/conftest.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -q -k freezes`
Expected: 3 failures. `config.permissions` is the caller's list, now two items long, so
`assert config.permissions == (deny_secrets,)` fails on both length and type.

- [ ] **Step 3: Write the implementation**

In `src/my_agent/agent.py`, replace the opening of `AgentConfig.__post_init__`:

```python
    def __post_init__(self) -> None:
        # Coerce before validating. `frozen=True` stops a field being rebound but
        # not the sequence behind it being mutated, so validating the caller's own
        # list leaves every check below defeatable by a later append -- and for
        # `permissions` and `middleware` that means a capability decision that
        # can be changed after it was reviewed. `object.__setattr__` is how a
        # frozen dataclass assigns; it works with `slots=True`.
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "middleware", tuple(self.middleware))
        object.__setattr__(self, "permissions", tuple(self.permissions))

        require(self.name != "", "name must not be empty")
```

Leave the rest of the method unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -q`
Expected: all pass, including the existing `_agent_kwargs` and `build_agent` tests — `list()` of a
tuple works exactly as before.

- [ ] **Step 5: Mutation-check the tests**

Delete the `permissions` coercion line only, leaving the other two:

Run: `uv run pytest tests/test_agent.py -q -k freezes`
Expected: `test_agent_config_freezes_the_rules_it_was_given` fails, the other two pass. This is
what proves each test covers its own field rather than all three passing on one line.

Restore the line.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run ruff check . && uv run mypy && uv run pyright && uv run pytest -q
git add src/my_agent/agent.py tests/test_agent.py
git commit
```

Commit subject: `Make AgentConfig's validation undefeatable`. Body: explain that `frozen=True`
froze the binding and not the sequence, so a caller could pass validation and then append to the
permissions list, and `_agent_kwargs` would forward a `str` into `create_deep_agent`. Note that
`ModelConfig` needs nothing because all its fields are scalars.

---

## Task 2: Delete the two speculative helpers

**Files:**
- Modify: `src/my_agent/negative_space.py` — remove `check_shape`, `check_finite`, and their
  `__all__` entries; drop the now-unused `math` and `Sequence` imports
- Modify: `CLAUDE.md` — lines ~210, ~238, ~475
- Modify: `docs/findings.md` — the F10 worked-example citation

**Interfaces:**
- Consumes: nothing.
- Produces: `my_agent.negative_space.__all__ == ["CheckFailed", "bounded", "require", "unreachable"]`.
  Task 3 writes tests for exactly those four names and must not reference the deleted two.

**Background the implementer needs:**

`check_shape` and `check_finite` have **zero call sites** in `src/` or `tests/` — every apparent
use elsewhere is prose inside a comment. They are tensor-shape and NaN-loss helpers, and CLAUDE.md
says the project's domain is deliberately undecided, so they are speculative by the project's own
YAGNI rule. `bounded()` and `unreachable()` stay: CLAUDE.md explicitly promises `bounded()` as the
agent-loop bound, and `unreachable()` is a three-line idiom.

This file is **not** vendored. Its docstring says "Copy this module into your project", which is
the source skill's instruction to an adopter, not a do-not-edit marker — and the project's copy has
already diverged from the skill asset by 48 lines (PEP 695 `bounded[T]`, a sorted `__all__` for
`RUF022`, tighter `Sequence[int | str | None]` typing). Only `scripts/audit_negative_space.py` is
marked vendored in CLAUDE.md. Editing this file is established practice.

Deleting `check_shape` removes a cited worked example in two places. The replacement already
exists: `capabilities.compiled_tools` uses the same explicit-`raise` pattern for the same reason
(mypy cannot narrow through a helper call) and carries a comment saying so.

- [ ] **Step 1: Confirm there is nothing to break**

Run:
```bash
grep -rn "check_shape\|check_finite" src/ tests/ | grep -v "^src/my_agent/negative_space.py"
```
Expected: no output. If anything appears, stop — the premise of this task is wrong.

- [ ] **Step 2: Delete the two functions**

In `src/my_agent/negative_space.py`:

Remove `"check_finite",` and `"check_shape",` from `__all__`, leaving:

```python
__all__ = [
    "CheckFailed",
    "bounded",
    "require",
    "unreachable",
]
```

Delete the whole `def check_shape(...)` function (through its `return bindings`) and the whole
`def check_finite(...)` function. Then narrow the imports, since `math` and `Sequence` were only
used by the deleted code:

```python
from collections.abc import Iterable, Iterator
from typing import NoReturn
```

`Any` also becomes unused — confirm with ruff in Step 4 and remove it if so.

Also update the module docstring's first line, which now overstates what the file is:

```python
"""Runtime checks that survive ``python -O``.

Adapted from the negative-space-programming skill and owned here: this copy has
diverged deliberately (PEP 695 generics, a sorted ``__all__``, and only the
helpers this project actually uses). Standard library only, no dependencies.
```

- [ ] **Step 3: Re-point the documentation citations**

`CLAUDE.md`, the negative-space bullet (~line 210):

```markdown
- `src/my_agent/negative_space.py` holds `require()`, `unreachable()` and `bounded()`. Use these,
  not bare `assert` — `python -O` deletes `assert` statements entirely, condition and message both,
  and some container images set `PYTHONOPTIMIZE`.
```

`CLAUDE.md`, the mypy-narrowing bullet (~line 238) — replace the last sentence:

```markdown
  `compiled_tools` in `capabilities.py` is the worked example.
```

`CLAUDE.md`, the repo layout line (~line 475):

```
  negative_space.py   # contract helpers: require/unreachable/bounded
```

`docs/findings.md`, the F10 entry (~line 174) — replace the final sentence:

```markdown
`compiled_tools` in `capabilities.py` is the worked example.
```

- [ ] **Step 4: Run the full gate**

Run: `uv run ruff check . && uv run mypy && uv run pyright && uv run pytest -q`
Expected: all green. Ruff will name any import left unused — remove exactly those.

Then confirm the citations resolve to real code:
```bash
grep -n "def compiled_tools" src/my_agent/capabilities.py
```
Expected: one hit.

- [ ] **Step 5: Commit**

```bash
git add src/my_agent/negative_space.py CLAUDE.md docs/findings.md
git commit
```

Commit subject: `Delete the two contract helpers nothing calls`. Body: zero call sites, helpers for
a domain the project deliberately has not chosen, and the worked-example citation re-pointed at
`compiled_tools`, which already uses the same explicit-raise pattern for the same mypy reason.

---

## Task 3: Bring the contract helpers inside the default gate

**Files:**
- Modify: `src/my_agent/negative_space.py` — two doctests rewritten; `__main__` block removed
- Modify: `pyproject.toml` — `addopts`, `testpaths`
- Create: `tests/test_negative_space.py`
- Modify: `CLAUDE.md` — command block

**Interfaces:**
- Consumes: Task 2's `__all__` of four names.
- Produces: `uv run pytest` collects the `src/` doctests. No source signatures change.

**Background the implementer needs:**

Three measured facts shape this task.

1. **`--doctest-modules` alone collects nothing.** `testpaths = ["tests"]` confines collection, so
   `uv run pytest --doctest-modules --collect-only -q | grep -c negative_space` returns `0`. `src`
   has to join `testpaths`.
2. **The existing doctests fail under pytest.** They expect `negative_space.CheckFailed: ...`.
   `python -m doctest src/my_agent/negative_space.py` imports the file as top-level
   `negative_space`, so that matches; pytest imports it as `my_agent.negative_space`, so it does
   not. No literal satisfies both runners.
3. **The `try/except` form satisfies both**, and asserts the message *exactly* rather than eliding
   it behind `...`. Verified under pytest and under `python -m doctest`.

After Task 2, only `require` and `bounded` have raising examples. `unreachable` has no doctest at
all and no automated coverage of any kind — this task is its first.

The module ends with a `if __name__ == "__main__":` block calling
`doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)`. That flag ignores the exception's
module *and its message*, which is why the block passes where pytest fails. Once pytest runs the
doctests, the block is a second runner with weaker assertions — delete it.

Note a limitation and do not try to fix it here: `tests/conftest.py` is directory-scoped, so the
socket guard and RNG seeding will not apply to `src/` doctests. That is fine for pure contract
helpers and is recorded in the spec's Open Questions.

- [ ] **Step 1: Rewrite the two raising doctests**

In `src/my_agent/negative_space.py`, `require`'s docstring:

```python
    """Fail unless ``condition`` is truthy.

    >>> require(1 < 2)
    >>> try:
    ...     require(2 < 1, "ordering broken")
    ... except CheckFailed as exc:
    ...     print(exc)
    ordering broken

    Keep one predicate per call: ``require(a); require(b)`` reports which half
    failed, ``require(a and b)`` does not.
    """
```

And `bounded`'s:

```python
    >>> list(bounded(range(3), 5))
    [0, 1, 2]
    >>> try:
    ...     list(bounded(range(10), 3, name="retries"))
    ... except CheckFailed as exc:
    ...     print(exc)
    retries exceeded its bound of 3 iterations
    """
```

Then delete the trailing block entirely:

```python
if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)
    raise SystemExit(1 if failures else 0)
```

- [ ] **Step 2: Verify the doctests now pass under pytest**

Run: `uv run pytest --doctest-modules src/my_agent/negative_space.py -q`
Expected: `2 passed` — `require` and `bounded`. Before the rewrite this command reported failures
on both.

- [ ] **Step 3: Make the default suite collect them**

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-ra --strict-markers --strict-config --doctest-modules -m 'not live and not eval'"
testpaths = ["tests", "src"]
```

Leave every other key alone.

- [ ] **Step 4: Verify collection and that the suite is still green**

Run: `uv run pytest --collect-only -q | grep -c negative_space`
Expected: `2` (was `0`).

Run: `uv run pytest -q`
Expected: 245 passed (243 before this task's tests, plus the two doctests), 2 deselected.

- [ ] **Step 5: Write the failing unit tests**

Create `tests/test_negative_space.py`:

```python
"""Unit tests for the contract helpers.

The doctests in `negative_space.py` are documentation that happens to execute,
and they now run in this suite. What they read badly for lives here: the failure
paths, the boundaries, and the category discipline every other module's
`require()` calls depend on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from my_agent.negative_space import CheckFailed, bounded, require, unreachable


def test_require_passes_a_truthy_condition(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    assert_does_not_raise(lambda: require(True, "should not fire"))


def test_require_raises_check_failed_with_its_message() -> None:
    with pytest.raises(CheckFailed, match="ordering broken"):
        require(False, "ordering broken")


def test_require_has_a_message_even_when_none_is_given() -> None:
    """A bare `require(x)` that fires must still say something."""
    with pytest.raises(CheckFailed, match="requirement failed"):
        require(False)


def test_check_failed_is_an_assertion_error() -> None:
    """Existing handlers and `pytest.raises(AssertionError)` must keep working —
    which is also why a test expecting a tripped require() must name
    CheckFailed, the narrower type."""
    assert issubclass(CheckFailed, AssertionError)


def test_unreachable_always_raises() -> None:
    with pytest.raises(CheckFailed, match="reached unreachable code"):
        unreachable()


def test_unreachable_carries_its_message() -> None:
    with pytest.raises(CheckFailed, match="match was exhausted"):
        unreachable("match was exhausted")


def test_bounded_yields_everything_inside_the_bound() -> None:
    assert list(bounded(range(3), 5)) == [0, 1, 2]


def test_bounded_allows_exactly_the_limit() -> None:
    """The boundary, stated on its own: off-by-one here would either reject a
    legal loop or let one extra iteration through."""
    assert list(bounded(range(3), 3)) == [0, 1, 2]


def test_bounded_rejects_one_item_past_the_limit() -> None:
    with pytest.raises(CheckFailed, match="exceeded its bound of 3"):
        list(bounded(range(4), 3))


def test_bounded_names_the_loop_in_its_failure() -> None:
    with pytest.raises(CheckFailed, match="retries exceeded"):
        list(bounded(range(10), 3, name="retries"))


@pytest.mark.parametrize("limit", [0, -1], ids=["zero", "negative"])
def test_bounded_rejects_a_bound_that_permits_no_iterations(limit: int) -> None:
    with pytest.raises(CheckFailed, match="bound must be at least 1"):
        list(bounded(range(3), limit))


def test_bounded_is_lazy() -> None:
    """It wraps producers whose length you did not compute, so it must not
    consume the iterable to check the bound — that would hang on exactly the
    infinite producer it exists to catch."""
    consumed: list[int] = []

    def producer() -> Iterator[int]:
        for i in range(100):
            consumed.append(i)
            yield i

    first = next(bounded(producer(), 5))

    assert first == 0
    assert consumed == [0]
```

`assert_does_not_raise` is an existing fixture in `tests/conftest.py`, so it needs no import.

- [ ] **Step 6: Run the new tests to verify they fail, then pass**

The helpers already exist, so these are characterisation tests and will pass immediately — except
`test_bounded_rejects_a_bound_that_permits_no_iterations`, which needs `list(...)` because
`bounded` is a generator and its `require` does not run until first iteration. Confirm:

Run: `uv run pytest tests/test_negative_space.py -q`
Expected: all pass. If the `limit` test errors instead of failing, the `list()` wrapper is missing.

- [ ] **Step 7: Mutation-check the tests**

Three mutants, restoring the file between each:

1. In `bounded`, delete `if count > limit: raise CheckFailed(...)`.
   Expected red: `test_bounded_rejects_one_item_past_the_limit`,
   `test_bounded_names_the_loop_in_its_failure`, and `bounded`'s doctest.
2. In `bounded`, change `require(limit >= 1, ...)` to `require(limit >= 0, ...)`.
   Expected red: the `zero` case of the bound-permits-no-iterations test.
3. In `unreachable`, replace `raise CheckFailed(...)` with `return None`.
   Expected red: both `unreachable` tests. (mypy will also object, since the return type is
   `NoReturn` — that is the point of the annotation and worth noting in the commit.)

A mutant that kills nothing means the test is decorative; rewrite it rather than accepting it.

- [ ] **Step 8: Update the command block in CLAUDE.md**

Remove this line entirely — the doctests now run in the default suite, and two runners with
divergent expectations was the defect:

```
uv run python -m doctest src/my_agent/negative_space.py   # contract helpers' doctests
```

Add a line to CLAUDE.md's testing section:

```markdown
- **The `src/` doctests run in the default suite** (`--doctest-modules`, with `src` in
  `testpaths`). They are written as `try/except` + `print` rather than as `Traceback` blocks,
  because the expected exception line differs between runners: pytest imports the module as
  `my_agent.negative_space`, `python -m doctest` as `negative_space`. The `try/except` form also
  asserts the message exactly, where `...` elides it.
```

- [ ] **Step 9: Full gate, then commit**

```bash
uv run ruff check . && uv run mypy && uv run pyright && uv run pytest -q
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007
python ~/.claude/skills/pytest-expert/scripts/audit_tests.py tests/
git add src/my_agent/negative_space.py tests/test_negative_space.py pyproject.toml CLAUDE.md
git commit
```

Commit subject: `Bring the contract helpers inside the test gate`. Body: the module every other
module's contracts depend on was the only one with no test file, pytest collected zero doctests,
and gutting `bounded()` left ruff, mypy and pytest all green — only a manual command caught it.
Record both measured gotchas (`testpaths` confinement and the qualname mismatch) and why the
standalone doctest command is gone.

---

## Task 4: Test the mirror's error paths

**Files:**
- Modify: `tests/test_mirror.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing other tasks depend on.

**Background the implementer needs:**

`tests/test_mirror.py` drives seven of `JsonlMirror`'s nine callbacks. `on_chain_error` and
`on_llm_error` are driven zero times — in the module whose docstring says "the runs most worth
reading back are the ones that crashed". Both write through the same `_write` helper as the
existing `on_tool_error`, recording `error_type` and `error`.

The file already has what you need: a `records(stream)` helper returning parsed JSON objects, and
`mirror` / `stream` fixtures. Model the new tests on the existing one:

```python
    mirror.on_tool_error(ValueError("permission denied"), run_id=RUN_ID)
    written = records(stream)[0]
    assert written["event"] == "tool_error"
    assert written["error_type"] == "ValueError"
    assert "permission denied" in written["error"]
```

The module documents that `*_end` and `*_error` records carry no `name` field — only `*_start`
records do — so assert that too. It is the property that makes a reader join an error back to its
start by `run_id`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mirror.py`, next to the existing `on_tool_error` test:

```python
def test_chain_error_is_recorded_with_its_type_and_message(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    """A crashed chain is the case the mirror exists for, and it was the one
    callback no test drove."""
    mirror.on_chain_error(RuntimeError("the agent exploded"), run_id=RUN_ID)

    written = records(stream)[0]

    assert written["event"] == "chain_error"
    assert written["error_type"] == "RuntimeError"
    assert "the agent exploded" in written["error"]


def test_llm_error_is_recorded_with_its_type_and_message(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    mirror.on_llm_error(TimeoutError("router timed out"), run_id=RUN_ID)

    written = records(stream)[0]

    assert written["event"] == "llm_error"
    assert written["error_type"] == "TimeoutError"
    assert "router timed out" in written["error"]


def test_error_records_carry_no_name_to_join_on(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    """Documented contract: only `*_start` records carry a name, so an error is
    joined back to its start by `run_id`. A name here would make the log look
    self-describing when it is not."""
    mirror.on_chain_error(RuntimeError("boom"), run_id=RUN_ID)
    mirror.on_llm_error(RuntimeError("boom"), run_id=RUN_ID)

    assert all("name" not in record for record in records(stream))
```

- [ ] **Step 2: Run the tests to verify they fail**

These cover existing behaviour, so they pass immediately. That is expected for characterisation
tests — the mutation check in Step 4 is what proves they are real. Confirm they pass:

Run: `uv run pytest tests/test_mirror.py -q -k "chain_error or llm_error or no_name"`
Expected: 3 passed.

- [ ] **Step 3: Confirm the coverage gap is closed**

Run: `uv run pytest --cov -q | grep mirror`
Expected: the `Missing` column no longer lists the `on_chain_error` and `on_llm_error` `_write`
lines (309 and 449 before this task).

- [ ] **Step 4: Mutation-check the tests**

Restoring between each:

1. In `on_chain_error`, change `"chain_error"` to `"chain_failed"`.
   Expected red: `test_chain_error_is_recorded_with_its_type_and_message`.
2. In `on_llm_error`, drop `error_type=type(error).__name__,`.
   Expected red: `test_llm_error_is_recorded_with_its_type_and_message`.
3. In `_write`, add `name="chain"` unconditionally.
   Expected red: `test_error_records_carry_no_name_to_join_on`.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run ruff check . && uv run mypy && uv run pyright && uv run pytest -q
git add tests/test_mirror.py
git commit
```

Commit subject: `Test the mirror's error paths`. Body: two of nine callbacks were driven zero
times, and they were the two the module exists for; note the `name`-absence contract as the thing
that makes an error joinable to its start.

---

## Task 5: One gate, three layers

**Files:**
- Create: `scripts/check.sh`
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` — `pre-commit` in dev deps
- Modify: `CLAUDE.md` — command block

**Interfaces:**
- Consumes: Task 3's pytest configuration, so `uv run pytest` already includes the doctests and
  the script needs no separate doctest command.
- Produces: `scripts/check.sh`, exit 0 on success and non-zero on the first failure. Both the hook
  and the workflow call it and nothing else.

**Background the implementer needs:**

There is no CI: no `.github/workflows`, no pre-commit, no Makefile, nox or tox. Around ten gates
are documented in CLAUDE.md and every one is run by hand.

The whole offline sequence takes **6 seconds** end to end (pyright is the slowest single step at
1.6s). That measurement is why the script takes no arguments and has no fast/slow tiers — there is
nothing to optimise around, and a tier flag would be a second thing to keep in sync.

The one rule that matters: **the command list lives in the script only.** A hook or a workflow that
re-lists the commands is a second copy that will drift. The advisory `NSP001` line keeps its
`|| true`, matching CLAUDE.md.

There is no git remote yet, so the workflow will not run until one exists. It is written now so it
works the day one appears.

- [ ] **Step 1: Write the script**

Create `scripts/check.sh`:

```bash
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
```

Then: `chmod +x scripts/check.sh`

- [ ] **Step 2: Verify it passes, then verify it can fail**

Run: `./scripts/check.sh`
Expected: every step, then `all gates passed`, exit 0. Confirm with `echo $?`.

Now prove it fails. Create a file with a deliberate ruff violation:

```bash
printf 'import os\n' > /tmp/gate_probe.py && cp /tmp/gate_probe.py ./gate_probe.py
./scripts/check.sh; echo "exit: $?"
rm ./gate_probe.py
```
Expected: the ruff step reports `F401` and the script exits non-zero **without** running mypy. A
gate that cannot fail is not a gate.

- [ ] **Step 3: Add pre-commit to the dev dependencies**

In `pyproject.toml`, `[dependency-groups] dev`, keeping the list alphabetical:

```toml
dev = [
    "mypy>=2.3.1",
    "pre-commit>=4.0.0",
    "pyright>=1.1.414",
    "pytest>=9.1.1",
    "pytest-asyncio>=1.4.0",
    "pytest-cov>=7.1.0",
    "pytest-mock>=3.15.1",
    "ruff>=0.16.8",
]
```

Run: `uv sync`
Then: `uv run pre-commit --version` — expected: a version, not `Failed to spawn`.

Note: `uv sync` may move other pinned versions. If it does, re-run `./scripts/check.sh` and update
the version block in CLAUDE.md and `docs/findings.md` to match, as CLAUDE.md requires after any
`uv sync` that moves versions.

- [ ] **Step 4: Write the pre-commit config**

Create `.pre-commit-config.yaml`:

```yaml
# One hook, calling the one script. Deliberately not a list of ruff/mypy hooks:
# that would be a second copy of the gate sequence, free to drift from
# scripts/check.sh and from CI.
repos:
  - repo: local
    hooks:
      - id: offline-gate
        name: offline gate (scripts/check.sh)
        entry: scripts/check.sh
        language: system
        pass_filenames: false
        always_run: true
```

Install it: `uv run pre-commit install`

- [ ] **Step 5: Verify the hook runs**

Run: `uv run pre-commit run --all-files`
Expected: the hook passes, with the script's step output visible.

- [ ] **Step 6: Write the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
# Calls scripts/check.sh and nothing else, so CI cannot drift from what a
# developer runs locally.
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install the locked environment
        run: uv sync --locked

      - name: Offline gate
        run: ./scripts/check.sh
```

`uv python install` reads `.python-version`, which is pinned to `3.13`. `uv sync --locked` fails if
`uv.lock` is out of date, which is what you want in CI.

- [ ] **Step 7: Check the workflow is valid YAML**

Run:
```bash
uv run python -c "import pathlib,sys; d=pathlib.Path('.github/workflows/ci.yml').read_text(); print('non-empty:', bool(d.strip()))"
```
There is no YAML parser in the dependency set and adding one for this is not worth it; the
workflow cannot be executed without a remote anyway. Read it once against the
`astral-sh/setup-uv` README to confirm the action inputs, and note in the commit that it is
unexercised until a remote exists.

- [ ] **Step 8: Update the command block in CLAUDE.md**

Add at the top of the commands block, above `uv sync`:

```
./scripts/check.sh                        # every offline gate, in order. What CI and the
                                          # pre-commit hook both run; the list lives only here.
```

And add a line to the testing section:

```markdown
- **The gate sequence lives in `scripts/check.sh` and nowhere else.** The pre-commit hook and
  `.github/workflows/ci.yml` both call it and re-list nothing, because three copies of a command
  list is three things to forget. The live and eval suites stay out: they need the network and
  cost money.
```

- [ ] **Step 9: Commit**

```bash
git add scripts/check.sh .pre-commit-config.yaml .github/workflows/ci.yml pyproject.toml uv.lock CLAUDE.md
git commit
```

Commit subject: `Put the gate in one script, and have three layers call it`. Body: ten gates were
documented and every one was manual; the sequence now lives in one file that a hook and a workflow
both call, so it cannot drift; the six-second measurement is why there are no tiers; and the
workflow is unexercised until a remote exists. Say that the script was verified by breaking a gate
and watching it exit non-zero.

---

## Task 6: A README, and documentation that matches the code

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` — `description`
- Modify: `CLAUDE.md` — line ~165, the `-O` claim
- Modify: `.env.example`

**Interfaces:**
- Consumes: Tasks 1-5, since the README describes the finished state including `scripts/check.sh`.
- Produces: nothing.

**Background the implementer needs:**

`README.md` is **0 bytes** while `pyproject.toml` declares it as the project readme, and
`description` is still the scaffold's `"Add your description here"`.

Two drift items were measured. `create_deep_agent` has **16** keyword-only parameters, not the 17
CLAUDE.md:165 claims — and the same file's Verified API Facts list is correct, so the file
contradicts itself. And CLAUDE.md warns that under `python -O` "the assertions inside the tests
vanish and everything passes"; in fact the run exits **1**, because pytest emits
`PytestConfigWarning` and `filterwarnings = ["error"]` makes it fatal. Reality is safer than the
doc, and the doc should say so.

`WEAVE_TRACE_LANGCHAIN` is read by the code and named in `WeaveTracing.activate()`'s failure
message but is absent from `.env.example`. Weave sets it to `"true"` itself when unset
(`weave/integrations/langchain/langchain.py:472`), so document it as optional rather than required.

The README must not duplicate CLAUDE.md. CLAUDE.md is the contributor's contract; the README is
the front door.

- [ ] **Step 1: Write the README**

Create `README.md`:

```markdown
# my-agent

A deep-agent harness built on [`deepagents`](https://github.com/langchain-ai/deepagents), driven by
contracts, tests and observability rather than by feature count.

**The agent's domain is deliberately undecided.** What is being built here is the harness: the
capability allowlist, the model wiring, the run bounds, the observability seams, and the tests that
keep all of it honest. One placeholder tool is enough to exercise it. When a real domain is chosen,
it should slot in behind the existing protocols without any existing file changing.

## What is actually interesting here

- **A capability allowlist that is proved, not requested.** `create_deep_agent` enables a shell
  `execute` tool with no opt-in. It is withheld, and the absence is asserted against the compiled
  graph *and* every subagent graph — because deepagents gives its general-purpose subagent its own
  filesystem middleware, so the parent's allowlist is not the whole story.
- **Bounds that belong to the thing they bound.** One turn goes through `run_turn`, which always
  sends a step limit and attaches a wall-clock deadline, so no caller can forget either.
- **Library behaviour recorded rather than assumed.** `docs/findings.md` holds twenty-three
  verified findings, each with how it was checked and what the code does about it.
- **Tests that are checked for being able to fail.** Changes here are verified by reverting the fix
  and watching a test go red.

## Requirements

- Python 3.13 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- A Hugging Face token with Inference Providers access

## Setup

```bash
uv sync
cp .env.example .env    # then fill in HF_TOKEN
```

Only `HF_TOKEN` is required. Tracing switches on if you supply the LangSmith or W&B keys; see
`.env.example`.

## Running it

```bash
uv run my-agent                  # live checks: one per finding in docs/findings.md
uv run my-agent "your prompt"    # one ordinary turn
```

## Development

```bash
./scripts/check.sh               # every offline gate, in order -- what CI runs
uv run pytest                    # the offline suite on its own
uv run pytest -m live > live.log 2>&1   # hits the real router; redirect, never pipe
```

`scripts/check.sh` is the single source of truth for the gate: the pre-commit hook and
`.github/workflows/ci.yml` both call it and re-list nothing.

## Where to look

| Path | What it holds |
|---|---|
| `CLAUDE.md` | The contributor's contract: architecture, non-negotiables, verified API facts |
| `docs/findings.md` | F1-F23: verified library behaviour and what the code does about each |
| `docs/superpowers/specs/` | Design documents |
| `src/my_agent/capabilities.py` | The allowlist and the proof it held |
| `src/my_agent/run.py` | One bounded turn |

## Status

Pre-domain. The harness works end to end against the Hugging Face router, with LangSmith and W&B
Weave tracing verified to coexist over the same run.
```

- [ ] **Step 2: Fix the project description**

In `pyproject.toml`:

```toml
description = "A contract- and eval-driven deep-agent harness on deepagents, with the domain deliberately undecided"
```

- [ ] **Step 3: Fix the two drift items in CLAUDE.md**

Line ~165 — `17` becomes `16`:

```markdown
`create_deep_agent` has 16 keyword-only parameters. Threading them through `build_agent` one at a
```

The `-O` warning in the testing section — replace it with what actually happens:

```markdown
- Do not run the suite under `python -O`: it exits 1 rather than lying to you. pytest emits
  `PytestConfigWarning` because `assert` statements in test bodies are not executed, and
  `filterwarnings = ["error"]` makes that fatal. The reason the flag is dangerous elsewhere still
  stands — `-O` deletes every `assert`, which is why `src/` uses `require()` instead.
```

- [ ] **Step 4: Document `WEAVE_TRACE_LANGCHAIN`**

Append to the Weave block in `.env.example`:

```
# Optional. Weave's LangChain integration is gated on this and weave sets it to
# "true" itself when unset, so you normally need not. Set it to a falsey value
# and WeaveTracing.activate() will fail loudly rather than trace nothing.
# WEAVE_TRACE_LANGCHAIN=true
```

- [ ] **Step 5: Verify every claim the README makes**

The README states counts and command names; check them rather than trusting the draft:

```bash
grep -c "^## F" docs/findings.md                  # expect 23, matching "twenty-three"
./scripts/check.sh >/dev/null && echo "check.sh works"
uv run python -c "import inspect; from deepagents import create_deep_agent as c; \
print(sum(1 for p in inspect.signature(c).parameters.values() if p.kind == p.KEYWORD_ONLY))"
```
Expected: `23`, `check.sh works`, `16`. Fix the README or CLAUDE.md to match whatever these print
— the measurement wins.

- [ ] **Step 6: Full gate, then commit**

```bash
./scripts/check.sh
git add README.md pyproject.toml CLAUDE.md .env.example
git commit
```

Commit subject: `Write the README, and fix what the docs got wrong`. Body: the readme was 0 bytes
while pyproject declared it; the "17 keyword parameters" claim contradicted the same file's own
verified list; the `-O` warning described a silent pass that is actually a hard failure. Note that
the README's counts were verified against the code rather than written from the draft.

---

## Self-Review

**Spec coverage.** Each of the spec's six architecture sections maps to the task of the same
number. The spec's eight findings map as: #1 → Task 1; #4 → Task 2; #3 → Task 3; #5 → Task 4;
#2 → Task 5; #6, #7, #8 → Task 6. The spec's Out of scope list is untouched by every task. The
spec's three Open Questions remain open by design and are not tasks.

**Placeholders.** None. Every code step carries the code; every verification step carries the
command and the expected output.

**Type and name consistency.** `AgentConfig.tools/.middleware/.permissions` are tuples after
Task 1 and are only ever read via `as_kwargs()` and `list(...)` afterwards. Task 2 leaves
`__all__` at four names and Task 3's test file imports exactly those four. `compiled_tools` is the
name used in both re-pointed citations and it exists. `scripts/check.sh` is the only place the gate
sequence appears, and Tasks 5 and 6 both refer to it by that path.

**One risk worth flagging to the executor:** Task 5 Step 3 runs `uv sync` to install
`pre-commit`, which may move other pinned versions. CLAUDE.md requires the recorded version block
to be re-verified after any such move, so that step says so explicitly. If versions do shift, the
version table in CLAUDE.md and `docs/findings.md` must be updated in the same commit.
