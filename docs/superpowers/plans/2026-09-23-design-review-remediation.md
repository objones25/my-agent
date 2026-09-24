# Design review remediation (Tasks 2-5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three latent gaps and three drifts the 2026-09-23 design review found, none of which touch subagents: one exception base for the run bounds, one reader for per-call token usage, `capabilities.py`'s parameter checks routed through `contracts.py`, and three small drifts.

**Architecture:** Each change is a small edit inside an existing module, apart from one new module (`src/my_agent/usage.py`) that both `run.py` (core) and `mirror.py` (observability) import, so neither has to import the other. Every behavioural change lands test-first. A final task re-measures the tripped-check counts CLAUDE.md states and updates the docs.

**Tech Stack:** Python 3.13, langchain-core 1.6.3, deepagents 0.7.15, pytest 9.1.1, mypy 2.3.1 (strict), pyright 1.1.414, ruff 0.16.8, all through `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-23-design-review-remediation-design.md`. This plan covers the spec's **Tasks 2, 3, 4 and 5 only** (numbered 1-4 here, plus a docs task). The spec's Task 1 (subagents) waits on its open questions 1, 2 and 4 and gets its own plan once they are settled. The spec says Tasks 2-5 "do not touch subagents and do not wait on the open questions".

## Global Constraints

- Read `CLAUDE.md` before starting. It is the contributor's contract, and every rule below comes from it.
- `./scripts/check.sh` must be green after every task. It runs uv lock check, ruff, mypy (strict), pyright, pytest, coverage and the negative-space audit.
- Use `require()` from `my_agent.negative_space`, never a bare `assert`, in `src/`. Programmer errors use `require()` (raises `CheckFailed`). Operating errors are typed exceptions.
- **Every new `require()` lands with a test that trips it.** A test expecting a tripped `require()` uses `pytest.raises(CheckFailed, ...)`, never `AssertionError`.
- **Every check message must be one no other check could produce.** Two sites with the same message cannot be told apart by a test.
- **One test file per source module.** A new module gets a new test file.
- Never write an API call from memory. Confirm it with `uv run python -c "import inspect, X; print(inspect.signature(X.thing))"`. Every library name this plan uses was checked against the installed wheels on 2026-09-23: `langchain_core.messages.ai.UsageMetadata`, `langchain_core.outputs.LLMResult(generations=...)`, `ChatGeneration`, `ChatGenerationChunk` (subclasses `ChatGeneration`), `AIMessageChunk` (subclasses `AIMessage`).
- `filterwarnings = ["error"]` is on. A new warning fails the build. Fix it, do not silence it.
- Run the full suite with `uv run pytest -q` and a single test with `uv run pytest tests/test_x.py::test_name -q`. Never use `python -O`.
- One commit per task. `main` is protected, so work on the worktree's branch. End every commit message with this line:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Do not refactor `main.py` beyond the one edit in Task 1. It is scaffolding (CLAUDE.md, "Testing and evals").

## Review Focus

These five inputs are implied by the spec but no step tests them directly. Each one now has a test in the task that owns the code.

1. **A streamed call** (`ChatGenerationChunk` carrying an `AIMessageChunk`) with usage. `call_usage` must read it, because a streamed turn is otherwise invisible to the token bound. Test added in Task 2.
2. **A generation with no message** (a plain `Generation`, as a non-chat model returns). `call_usage` must skip it, not crash. Test added in Task 2.
3. **Usage on a later choice only.** The first choice carries no usage and a later one does, so `call_usage` must return the later one's rather than report the call as unmeasured. Test added in Task 2.
4. **A call that reported no usage** must still log `"usage": null` in the mirror, exactly as today, so existing log readers see no change. Test added in Task 2.
5. **A `CheckFailed` on the single-prompt path** must still crash `main()`, not be reported as a spent budget. Narrowing the `except` clause must not widen what it swallows. Test added in Task 1.

---

## File Structure

| File | Change | Responsibility after the change |
|---|---|---|
| `src/my_agent/run.py` | modify | Adds `BoundExceeded`, the base of the four bound exceptions. `RunTokenBudget.on_llm_end` reads usage through `call_usage`. |
| `src/my_agent/main.py` | modify (one clause) | `except BoundExceeded` replaces the four-name tuple. |
| `src/my_agent/usage.py` | **create** | `call_usage(response)`: the one reading of a model call's usage. |
| `src/my_agent/mirror.py` | modify | `on_llm_end` reads usage through `call_usage`. The backslash check gets its own message. |
| `src/my_agent/contracts.py` | modify | `check_required_parameters` gains a keyword-only `consequence`. |
| `src/my_agent/capabilities.py` | modify | Two inline checks become `check_required_parameters` calls. The `FilesystemMiddleware` signature is read once. One stale constant name in a docstring is fixed. |
| `src/my_agent/model.py` | modify | The duplicated "is unset or empty" message becomes one module-level format string. |
| `tests/test_run.py`, `tests/test_main.py`, `tests/test_mirror.py`, `tests/test_contracts.py`, `tests/test_capabilities.py`, `tests/test_model.py` | modify | Tests for the above. |
| `tests/test_usage.py` | **create** | Tests for `usage.py`. |
| `CLAUDE.md` | modify | Tripped-check counts, test count, layout table (`usage.py`), exception base. |

---

### Task 1: `BoundExceeded`, one base for the four run bounds

**Files:**
- Modify: `src/my_agent/run.py` (the four classes at `:145`, `:215`, `:224`, `:233`, plus `__all__` at `:50`)
- Modify: `src/my_agent/main.py:40-46` (import) and `:670-692` (the `except` clause and its comment)
- Test: `tests/test_run.py`, `tests/test_main.py`

**Interfaces:**
- Produces: `my_agent.run.BoundExceeded(RuntimeError)`. `DeadlineExceeded`, `ResumeLimitExceeded`, `StepLimitExceeded` and `TokenLimitExceeded` all subclass it. It is exported in `run.__all__`.

**Why:** today `main.py` catches the four by name. A fifth bound (a cost ceiling is the obvious one) that is not added to that tuple becomes a traceback on the single-prompt path (`uv run my-agent "..."`). The live-check path is unaffected, because `_attempt` (`main.py:588-591`) catches `Exception`. So the new `test_main.py` test **must** go through `_single_turn`, or it passes with or without this change.

- [ ] **Step 1: Write the failing tests in `tests/test_run.py`**

Add `BoundExceeded` to the existing `from my_agent.run import (...)` block (keep it alphabetical: after `TOKEN_LIMIT`, before `DeadlineExceeded`). Then add these beside `test_the_token_limit_is_an_operating_error_not_a_broken_contract` (around `:1338`):

```python
@pytest.mark.parametrize(
    "bound", [DeadlineExceeded, ResumeLimitExceeded, StepLimitExceeded, TokenLimitExceeded]
)
def test_every_run_bound_fails_as_a_bound_exceeded(bound: type[Exception]) -> None:
    """One base, so the edge catches every bound `RunBounds` rations — including
    the next one — without being edited."""
    assert issubclass(bound, BoundExceeded)


def test_a_spent_budget_is_an_operating_error_not_a_broken_contract() -> None:
    """Still a `RuntimeError`, so anything catching that today is unaffected, and
    never a `CheckFailed`: a programmer error is not a spent budget."""
    assert issubclass(BoundExceeded, RuntimeError)
    assert not issubclass(BoundExceeded, CheckFailed)
```

- [ ] **Step 2: Write the failing tests in `tests/test_main.py`**

Change the import at `:29` to `from my_agent.run import BoundExceeded, DeadlineExceeded, TurnResult`. Then add these after `test_main_reports_a_missed_deadline_instead_of_a_traceback` (which ends around `:206`):

```python
def test_main_reports_a_bound_it_was_never_told_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The property a hand-listed tuple of four could not give: a fifth bound is
    reported, not raised, with no edit to `main.py`.

    Driven through `_single_turn` on purpose. On the live-check path `_attempt`
    catches `Exception` and would report it either way, so a test there passes
    with or without `BoundExceeded`.
    """

    class CostLimitExceeded(BoundExceeded):
        """A budget `main.py` has never heard of."""

    monkeypatch.setattr(main_module, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.setenv("HF_TOKEN", "hf_token_value")
    monkeypatch.setattr(main_module, "available_backends", tuple)
    monkeypatch.setattr(main_module, "run_log_path", lambda: tmp_path / "run.jsonl")
    monkeypatch.setattr("sys.argv", ["my-agent", "ping"])

    def over_budget(config: object, prompt: str, callbacks: list[BaseCallbackHandler]) -> int:
        callbacks[0].on_chain_end({}, run_id=uuid4())
        raise CostLimitExceeded("run spent its whole cost ceiling")

    monkeypatch.setattr(main_module, "_single_turn", over_budget)

    assert main() == EXIT_CHECK_FAILED

    captured = capsys.readouterr()
    assert "whole cost ceiling" in captured.err
    assert "Traceback" not in captured.err


def test_main_still_crashes_on_a_broken_contract_during_a_single_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Narrowing the clause to one base must not widen what it swallows. A
    `CheckFailed` is a bug in our own contracts, and is never reported as a
    spent budget."""
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.setenv("HF_TOKEN", "hf_token_value")
    monkeypatch.setattr(main_module, "available_backends", tuple)
    monkeypatch.setattr(main_module, "run_log_path", lambda: tmp_path / "run.jsonl")
    monkeypatch.setattr("sys.argv", ["my-agent", "ping"])

    def broken(config: object, prompt: str, callbacks: list[BaseCallbackHandler]) -> int:
        callbacks[0].on_chain_end({}, run_id=uuid4())
        raise CheckFailed("a contract of ours was violated mid-turn")

    monkeypatch.setattr(main_module, "_single_turn", broken)

    with pytest.raises(CheckFailed, match="violated mid-turn"):
        main()
```

- [ ] **Step 3: Run the new tests and confirm they fail**

Run: `uv run pytest tests/test_run.py tests/test_main.py -q`
Expected: collection ERROR in both files, `ImportError: cannot import name 'BoundExceeded' from 'my_agent.run'`. Once the import resolves, `test_main_still_crashes_on_a_broken_contract_during_a_single_turn` passes before and after the `main.py` change. It is a guard that the change does not swallow more, not a red-first test.

- [ ] **Step 4: Add `BoundExceeded` to `src/my_agent/run.py`**

Insert this directly **above** `class TokenLimitExceeded` (currently `:145`), because it must be defined before its first subclass:

```python
class BoundExceeded(RuntimeError):
    """A turn ran out of something `RunBounds` rationed. An operating error.

    The one name the edge catches, so a bound added later is reported the way
    the first four are without anyone editing an `except` clause. Still a
    `RuntimeError`, so nothing catching that today changes. `CheckFailed` stays
    outside it: a programmer error is not a spent budget.
    """
```

Then change the four class lines, and nothing else in them (docstrings stay as they are):

```python
class TokenLimitExceeded(BoundExceeded):
class DeadlineExceeded(BoundExceeded):
class ResumeLimitExceeded(BoundExceeded):
class StepLimitExceeded(BoundExceeded):
```

Add `"BoundExceeded",` to `__all__` (`:50`), after `"TOKEN_LIMIT",` and before `"DeadlineExceeded",`.

- [ ] **Step 5: Run the `test_run.py` tests and the `main.py` mutant check**

Run: `uv run pytest tests/test_run.py -q`
Expected: PASS.

Run: `uv run pytest tests/test_main.py -q`
Expected: `test_main_reports_a_bound_it_was_never_told_about` FAILS (`CostLimitExceeded` escapes the four-name tuple) and everything else passes. This is the mutation the spec names ("restore the tuple → the local-subclass test goes red"), observed before the fix rather than after.

- [ ] **Step 6: Change the clause in `src/my_agent/main.py`**

Replace the four names in the `from my_agent.run import (...)` block (`:40-46`) with `BoundExceeded,`, keeping `TurnResult` and any other names that are there. Replace the `except (...) as exc:` clause and its comment (`:676-690`) with:

```python
        except BoundExceeded as exc:
            # Every bound in `RunBounds`, reported the same way, and any bound
            # added later without editing this line. Before `StepLimitExceeded`
            # existed the step limit escaped as langgraph's `GraphRecursionError`
            # and printed a traceback (F24). A hand-listed tuple had the same
            # failure waiting for the next bound. `CheckFailed` is not a
            # `BoundExceeded` and still crashes.
            print(f"error: {exc}", file=sys.stderr)
            exit_code = EXIT_CHECK_FAILED
```

Keep the two statements after the comment exactly as they were. Check with `sed -n 665,700p src/my_agent/main.py` before and after.

- [ ] **Step 7: Run the tests and the gate**

Run: `uv run pytest tests/test_run.py tests/test_main.py -q`
Expected: PASS.

Run: `./scripts/check.sh`
Expected: every step green.

- [ ] **Step 8: Commit**

```bash
git add src/my_agent/run.py src/my_agent/main.py tests/test_run.py tests/test_main.py
git commit -m "Give the four run bounds one base, so the edge catches the next one too

main.py caught DeadlineExceeded, ResumeLimitExceeded, StepLimitExceeded and
TokenLimitExceeded by name. A fifth bound not added to that tuple would print a
traceback on the single-prompt path; the live checks were unaffected because
_attempt catches Exception. BoundExceeded is a RuntimeError, so nothing catching
that changes, and CheckFailed stays outside it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `call_usage`, one reader for per-call token usage

**Files:**
- Create: `src/my_agent/usage.py`
- Create: `tests/test_usage.py`
- Modify: `src/my_agent/run.py:622-645` (`RunTokenBudget.on_llm_end`)
- Modify: `src/my_agent/mirror.py:478-528` (`JsonlMirror.on_llm_end`)
- Test: `tests/test_run.py`, `tests/test_mirror.py`

**Interfaces:**
- Produces: `my_agent.usage.call_usage(response: LLMResult) -> UsageMetadata | None`. It returns the usage of the first choice that carries any, or `None`. It raises `CheckFailed` unless `len(response.generations) == 1`.
- Consumes: nothing from Task 1.

**Why:** `RunTokenBudget` sums usage over every generation, and `JsonlMirror` keeps the last one. langchain-openai's `_create_chat_result` copies the request's whole usage onto *every* choice, so for an `n`-choice response the budget counts n× the real figure and the mirror counts 1×. Measured: one `LLMResult` with two generations reporting 100 and 40 gives `budget 140 mirror 40`. It is latent today (every call returns one choice), and `n` is a one-field change to `ModelConfig`.

**Why the precondition:** `BaseChatModel.generate` and `agenerate` both flatten a batch before calling back. Each prompt's run manager gets `LLMResult(generations=[res.generations], ...)`, so the outer list has length 1 on every `on_llm_end`. That is langchain-core's behaviour, not ours, so it is stated as a check rather than assumed. Both `RunTokenBudget` and `JsonlMirror` set `raise_error = True`, so a trip is loud.

- [ ] **Step 1: Write the failing tests in `tests/test_usage.py`**

```python
"""Tests for `my_agent.usage`: the one reading of what a model call spent.

The budget and the mirror both read this, so "the number this bounds is the
number the log shows" holds by construction rather than by two loops happening
to agree.
"""

from __future__ import annotations

import io
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation, LLMResult

from my_agent.mirror import JsonlMirror
from my_agent.negative_space import CheckFailed
from my_agent.run import RunTokenBudget
from my_agent.usage import call_usage


def _choice(total: int | None) -> ChatGeneration:
    """One choice, with `total` tokens of usage or with none reported."""
    if total is None:
        return ChatGeneration(message=AIMessage("ok"))
    return ChatGeneration(
        message=AIMessage(
            "ok",
            usage_metadata={"input_tokens": total, "output_tokens": 0, "total_tokens": total},
        )
    )


def test_two_choices_of_one_request_count_once() -> None:
    """langchain-openai copies the request's usage onto every choice, so summing
    choices counts an n-choice call n times."""
    usage = call_usage(LLMResult(generations=[[_choice(100), _choice(100)]]))

    assert usage is not None
    assert usage["total_tokens"] == 100


def test_the_first_choice_carrying_usage_is_the_one_read() -> None:
    usage = call_usage(LLMResult(generations=[[_choice(100), _choice(40)]]))

    assert usage is not None
    assert usage["total_tokens"] == 100


def test_usage_on_a_later_choice_only_is_still_read() -> None:
    """A choice with no usage before one with it is not an unmeasured call."""
    usage = call_usage(LLMResult(generations=[[_choice(None), _choice(40)]]))

    assert usage is not None
    assert usage["total_tokens"] == 40


def test_a_call_that_reported_no_usage_reads_as_none() -> None:
    assert call_usage(LLMResult(generations=[[_choice(None)]])) is None


def test_a_call_with_no_choices_reads_as_none() -> None:
    assert call_usage(LLMResult(generations=[[]])) is None


def test_a_streamed_call_is_read_like_any_other() -> None:
    """A streamed turn arrives as chunks. Missing it would make the token bound
    blind to every streamed call."""
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(
            "ok", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
        )
    )

    usage = call_usage(LLMResult(generations=[[chunk]]))

    assert usage is not None
    assert usage["total_tokens"] == 10


def test_a_generation_with_no_message_is_skipped_not_fatal() -> None:
    usage = call_usage(LLMResult(generations=[[Generation(text="ok"), _choice(5)]]))

    assert usage is not None
    assert usage["total_tokens"] == 5


@pytest.mark.parametrize("prompts", [0, 2])
def test_a_result_holding_other_than_one_prompt_is_refused(prompts: int) -> None:
    """langchain-core flattens a batch before calling back, one prompt per
    `on_llm_end`. Any other shape means that stopped being true, and summing or
    picking across prompts would then be a guess."""
    response = LLMResult(generations=[[_choice(10)] for _ in range(prompts)])

    with pytest.raises(CheckFailed, match=f"got {prompts} prompts' generations"):
        call_usage(response)


def test_the_budget_and_the_mirror_count_the_same_tokens_for_one_call() -> None:
    """The claim `RunTokenBudget.on_llm_end`'s docstring makes, tested where it
    is made true."""
    response = LLMResult(generations=[[_choice(100), _choice(100)]])
    budget = RunTokenBudget(10_000)
    stream = io.StringIO()

    budget.on_llm_end(response, run_id=uuid4())
    JsonlMirror(stream).on_llm_end(response, run_id=uuid4())

    logged = json.loads(stream.getvalue())["usage"]["total_tokens"]
    assert budget.tokens == logged == 100
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_usage.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'my_agent.usage'`.

- [ ] **Step 3: Create `src/my_agent/usage.py`**

```python
"""What one model call reported spending, read one way for every consumer.

`RunTokenBudget` bounds a turn on this number and `JsonlMirror` logs it. They
used to read `usage_metadata` with two different loops: the budget summed every
choice and the mirror kept the last. langchain-openai copies the request's
usage onto *every* choice (`BaseChatOpenAI._create_chat_result`), so for an
n-choice response those disagree by a factor of n. One reader makes "the number
this bounds is the number the log shows" true by construction.

Its own module because of the dependency direction: `run.py` is core and
`mirror.py` is observability, neither should import the other, and both import
this.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult

from my_agent.negative_space import require

__all__ = ["call_usage"]


def call_usage(response: LLMResult) -> UsageMetadata | None:
    """The request's usage, or `None` if no choice reported any.

    A chat model's callback receives one prompt per `LLMResult`:
    `BaseChatModel.generate` and `agenerate` flatten a batch before calling
    back, so `response.generations` holds one inner list, that request's
    choices. Every choice carries the same request-level usage, so the first one
    carrying any is the request's figure and the rest are copies of it.

    A streamed call arrives as `ChatGenerationChunk` / `AIMessageChunk`, which
    subclass the types checked here, so it is read the same way. A generation
    with no message (a non-chat model's) is skipped.
    """
    # Precondition: the flattening is langchain-core's behaviour, not ours. Any
    # other shape means it changed, and picking across prompts would be a guess.
    require(
        len(response.generations) == 1,
        f"call_usage expected one prompt per on_llm_end, got "
        f"{len(response.generations)} prompts' generations; langchain-core flattens "
        f"a batch before calling back, so this shape means that changed",
    )
    for generation in response.generations[0]:
        message = generation.message if isinstance(generation, ChatGeneration) else None
        if isinstance(message, AIMessage) and message.usage_metadata:
            return message.usage_metadata
    return None
```

- [ ] **Step 4: Run the `usage.py` tests**

Run: `uv run pytest tests/test_usage.py -q`
Expected: every test PASSES except `test_the_budget_and_the_mirror_count_the_same_tokens_for_one_call`, which FAILS with `200 == 100` (the budget still sums both choices).

- [ ] **Step 5: Add the per-consumer tests**

In `tests/test_run.py`, after `test_the_token_budget_counts_calls_that_reported_no_usage_at_all` (around `:1264`):

```python
def test_the_token_budget_counts_a_multi_choice_call_once() -> None:
    """Every choice carries the whole request's usage, so summing choices would
    charge an n-choice call n times."""
    budget = RunTokenBudget(1000)
    choice = _usage_report(100).generations[0][0]

    budget.on_llm_end(LLMResult(generations=[[choice, choice]]), run_id=uuid4())

    assert budget.tokens == 100
```

In `tests/test_mirror.py`, after `test_llm_end_records_output_and_usage` (around `:105`):

```python
def test_llm_end_records_the_requests_usage_once(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    """The mirror used to keep the *last* choice's usage and the budget summed
    all of them. Both now read the first choice carrying any."""
    first = AIMessage(
        content="a", usage_metadata={"input_tokens": 90, "output_tokens": 10, "total_tokens": 100}
    )
    second = AIMessage(
        content="b", usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}
    )
    result = LLMResult(
        generations=[[ChatGeneration(message=first), ChatGeneration(message=second)]]
    )

    mirror.on_llm_end(result, run_id=RUN_ID)

    assert records(stream)[0]["usage"]["total_tokens"] == 100


def test_llm_end_records_null_usage_when_none_was_reported(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    """Unchanged from before `call_usage`: log readers see `null`, not a missing
    key and not `{}`."""
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="pong"))]])

    mirror.on_llm_end(result, run_id=RUN_ID)

    written = records(stream)[0]
    assert "usage" in written
    assert written["usage"] is None
```

Run: `uv run pytest tests/test_run.py::test_the_token_budget_counts_a_multi_choice_call_once tests/test_mirror.py::test_llm_end_records_the_requests_usage_once tests/test_mirror.py::test_llm_end_records_null_usage_when_none_was_reported -q`
Expected: the first two FAIL (`200 == 100` and `40 == 100`). The null-usage test PASSES already, because it pins today's behaviour.

- [ ] **Step 6: Route `RunTokenBudget.on_llm_end` through `call_usage`**

In `src/my_agent/run.py`, add `from my_agent.usage import call_usage` after the `negative_space` import (`:48`). Replace the body and docstring of `RunTokenBudget.on_llm_end` (`:630-645`) with:

```python
        """Add up what the call reported.

        Read through `usage.call_usage`, which `mirror.py` reads too, so the
        number this bounds is the number the log shows by construction rather
        than by two loops happening to agree. A call that reported nothing is
        counted in `unmeasured_calls` instead.
        """
        usage = call_usage(response)
        if usage is None:
            self._unmeasured_calls += 1
            return
        self._tokens += int(usage.get("total_tokens", 0))
```

- [ ] **Step 7: Route `JsonlMirror.on_llm_end` through `call_usage`**

In `src/my_agent/mirror.py`, add `from my_agent.usage import call_usage` after the `negative_space` import (`:32`). In `on_llm_end`:
- delete the line `usage: dict[str, Any] | None = None` (`:489`);
- delete the three lines that read and assign `token_usage` inside the loop (`:516-518`);
- just before `payload: dict[str, Any] = {`, add:

```python
        reported = call_usage(response)
        usage = dict(reported) if reported is not None else None
```

The `"usage": usage` entry in `payload` stays as it is.

- [ ] **Step 8: Run the tests, then the gate**

Run: `uv run pytest tests/test_usage.py tests/test_run.py tests/test_mirror.py -q`
Expected: PASS.

Mutation check (do it, then undo the edit by hand, since `usage.py` is not committed yet): in `call_usage`, make the loop remember each choice's usage and return the last one instead of the first. Run `uv run pytest tests/test_usage.py -q` and expect `test_the_first_choice_carrying_usage_is_the_one_read` to go red (it gets 40, not 100). Undo it. Then make it **sum** the choices' `total_tokens` into the returned record instead (the budget's old behaviour). Expect `test_two_choices_of_one_request_count_once` and `test_the_budget_and_the_mirror_count_the_same_tokens_for_one_call` to go red. Undo it. The second mutant is the bug this task fixes.

Run: `./scripts/check.sh`
Expected: every step green.

- [ ] **Step 9: Commit**

```bash
git add src/my_agent/usage.py tests/test_usage.py src/my_agent/run.py src/my_agent/mirror.py tests/test_run.py tests/test_mirror.py
git commit -m "Read per-call token usage one way, for the budget and the mirror both

RunTokenBudget summed usage over every choice and JsonlMirror kept the last.
langchain-openai copies the request's usage onto every choice, so for an
n-choice response the budget counted n times the real figure and the mirror
once: one LLMResult with choices of 100 and 40 gave budget 140, mirror 40.
Latent today (every call returns one choice), and n is a one-field change.

call_usage reads the first choice carrying usage, and requires the one-prompt
shape langchain-core's generate/agenerate flatten to before calling back.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `capabilities.py`'s parameter checks through `contracts.py`

**Files:**
- Modify: `src/my_agent/contracts.py:114-130` (`check_required_parameters`)
- Modify: `src/my_agent/capabilities.py:273`, `:289`, `:516-530`
- Test: `tests/test_contracts.py`, `tests/test_capabilities.py:1245` and `:1261`

**Interfaces:**
- Produces: `check_required_parameters(actual: frozenset[str], needed: frozenset[str], callee_name: str, *, consequence: str = "the config must change") -> None`. The message becomes `f"{callee_name} no longer accepts {sorted(missing)}; {consequence}"`, which is byte-identical to today's when `consequence` is omitted.
- Consumes: nothing from Tasks 1-2.

- [ ] **Step 1: Write the failing tests in `tests/test_contracts.py`**

Add after `test_required_parameters_refuses_a_lost_constructor_alias` (which starts at `:183`):

```python
def test_required_parameters_names_the_consequence_it_was_given() -> None:
    """Each caller knows what breaks when a parameter goes, and the message is
    where a reader learns it."""
    with pytest.raises(
        CheckFailed, match=r"^Thing no longer accepts \['b'\]; the widget cannot be set$"
    ):
        check_required_parameters(
            frozenset({"a"}),
            frozenset({"a", "b"}),
            "Thing",
            consequence="the widget cannot be set",
        )


def test_required_parameters_defaults_to_the_config_consequence() -> None:
    """The default keeps `model.py`'s message exactly what it was."""
    with pytest.raises(
        CheckFailed, match=r"^Thing no longer accepts \['b'\]; the config must change$"
    ):
        check_required_parameters(frozenset({"a"}), frozenset({"a", "b"}), "Thing")
```

Run: `uv run pytest tests/test_contracts.py -q`
Expected: `test_required_parameters_names_the_consequence_it_was_given` FAILS with `TypeError: ... unexpected keyword argument 'consequence'`. The default test passes already, because it pins today's message.

- [ ] **Step 2: Add the parameter in `src/my_agent/contracts.py`**

```python
def check_required_parameters(
    actual: frozenset[str],
    needed: frozenset[str],
    callee_name: str,
    *,
    consequence: str = "the config must change",
) -> None:
```

Add one paragraph to the end of its docstring:

```
    `consequence` says what breaks, mirroring `check_known_parameters`'s
    `why_new_matters`. The default is the config-splat case this was written
    for; `capabilities.py` passes its own.
```

Change the second `require` message to `f"{callee_name} no longer accepts {sorted(missing)}; {consequence}"`.

Run: `uv run pytest tests/test_contracts.py -q`
Expected: PASS.

- [ ] **Step 3: Tighten the two `test_capabilities.py` matches (red first)**

At `tests/test_capabilities.py:1245`, change the match so it pins the new message:

```python
    with pytest.raises(
        CheckFailed,
        match=r"FilesystemMiddleware no longer accepts \['_permissions'\]; build_agent must change",
    ):
```

At `:1261` (the `SummarizationMiddleware` test), make the match pin the consequence too, so it proves the consequence arrived rather than merely that some check fired:

```python
    with pytest.raises(
        CheckFailed,
        match=r"SummarizationMiddleware no longer accepts \[.*\]; the compaction bounds "
        r"cannot be set and the agent would run at deepagents' own threshold",
    ):
```

Run: `uv run pytest tests/test_capabilities.py -q -k "permission_channel or cannot_be_bounded"`
Expected: the `_permissions` test FAILS (today's message is `no longer accepts tools/_permissions`). The compaction test PASSES (today's inline message already has that shape).

- [ ] **Step 4: Replace the inline checks in `src/my_agent/capabilities.py`**

Add `from my_agent.contracts import check_required_parameters` after the `negative_space` import (`:29`). `contracts.py` imports only `dataclasses`, `pydantic` and `negative_space`, so there is no import cycle.

Replace `:273`:

```python
_FS_SIGNATURE = inspect.signature(FilesystemMiddleware.__init__).parameters
_FS_MIDDLEWARE_PARAMS = frozenset(_FS_SIGNATURE)
```

Delete the second computation at `:289` (`_FS_SIGNATURE = inspect.signature(FilesystemMiddleware.__init__).parameters`). The loop below it keeps using `_FS_SIGNATURE`, which is now defined above.

Replace the two `require(...)` blocks at `:518-530` (keep the comments above each):

```python
check_required_parameters(
    _SUMMARIZATION_PARAMS,
    _NEEDED_SUMMARIZATION_PARAMS,
    "SummarizationMiddleware",
    consequence="the compaction bounds cannot be set and the agent would run at "
    "deepagents' own threshold",
)

# `_permissions` is private API. Pin it: losing it silently would drop every
# permission rule (see docs/findings.md).
check_required_parameters(
    _FS_MIDDLEWARE_PARAMS,
    frozenset({"tools", "_permissions"}),
    "FilesystemMiddleware",
    consequence="build_agent must change",
)
```

Both calls still run at import, so the existing `tripping_an_import_time_check` tests still drive them. This edit adds no class to `capabilities.py`, so the fixture's refusal to reload class-defining modules is not affected.

- [ ] **Step 5: Run the tests, the mutant and the gate**

Run: `uv run pytest tests/test_capabilities.py tests/test_contracts.py -q`
Expected: PASS.

Mutation check (do it, then revert it): change `frozenset({"tools", "_permissions"})` to `frozenset({"tools"})`. Run `uv run pytest tests/test_capabilities.py -q -k permission_channel` and expect it to go red (no check fires). Revert.

Run: `./scripts/check.sh`
Expected: every step green.

- [ ] **Step 6: Commit**

```bash
git add src/my_agent/contracts.py src/my_agent/capabilities.py tests/test_contracts.py tests/test_capabilities.py
git commit -m "Route capabilities.py's parameter pins through check_required_parameters

Two load-time checks re-implemented check_required_parameters inline, and the
FilesystemMiddleware signature was introspected twice. The function gains a
keyword-only consequence, mirroring check_known_parameters' why_new_matters;
the default keeps model.py's message byte-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Three small drifts

**Files:**
- Modify: `src/my_agent/mirror.py:578-582`
- Modify: `src/my_agent/model.py:252-270`
- Modify: `src/my_agent/capabilities.py:261`
- Test: `tests/test_mirror.py:333-335`, `tests/test_model.py`

**Interfaces:** none new.

- [ ] **Step 1: Tighten the backslash test (red first)**

In `tests/test_mirror.py`, change `test_run_log_path_rejects_a_run_id_with_a_backslash` (`:333`) so it can only match its own site:

```python
def test_run_log_path_rejects_a_run_id_with_a_backslash() -> None:
    """Its own message, not the forward-slash one. Two byte-identical messages
    cannot be told apart by a test, and this one used to read as covered by
    either."""
    with pytest.raises(CheckFailed, match="must not contain a backslash"):
        run_log_path(now=FIXED_NOW, run_id="a\\b")
```

Run: `uv run pytest tests/test_mirror.py -q -k backslash`
Expected: FAIL (today's message says "path separator").

- [ ] **Step 2: Give the backslash check its own message**

In `src/my_agent/mirror.py`, change the second of the two separator `require`s (`:578-582`, the one testing `"\\" not in token`) to:

```python
    require(
        "\\" not in token,
        f"run_id must not contain a backslash (got {token!r}): it is a path "
        "separator on Windows, and this becomes part of a filename, not a subdirectory",
    )
```

Run: `uv run pytest tests/test_mirror.py -q`
Expected: PASS. The `"/"` and traversal tests still match `"separator"` from the first check, which is unchanged.

- [ ] **Step 3: Pin the shared "unset or empty" message (characterisation test)**

This is a refactor, so the test pins today's behaviour and passes before the change. It exists so the hoist cannot quietly make the two messages diverge. In `tests/test_model.py`, after `test_from_env_raises_value_error_when_the_token_is_absent` (`:128-134`):

```python
def test_an_absent_and_a_blank_required_variable_read_the_same() -> None:
    """Both mean "set it", and both used to spell that out separately. One
    message, so an edit to one cannot silently leave the other behind."""
    with pytest.raises(ValueError, match="unset or empty") as absent:
        ModelConfig.from_env({})
    with pytest.raises(ValueError, match="unset or empty") as blank:
        ModelConfig.from_env({API_KEY_ENV_VAR: "   "})

    assert str(absent.value) == str(blank.value)
```

Run: `uv run pytest tests/test_model.py -q -k read_the_same`
Expected: PASS (it pins what is already true).

- [ ] **Step 4: Hoist the message in `src/my_agent/model.py`**

Add this module-level constant directly **above** `class ModelConfig` (`:176`). `from_env` reads it at call time, so any module-level position works, and this one puts it above its only reader:

```python
_UNSET_OR_EMPTY = (
    "{env_var} is unset or empty. Set it in .env (see .env.example) or export it "
    "before starting the agent."
)
"""What a required variable that is missing or blank reports. One string for
both, because both mean the same thing to the person reading it."""
```

Replace both `raise ValueError(f"{spec.env_var} is unset or empty. ...")` blocks in `from_env` (`:254-257` and `:265-268`) with:

```python
                    raise ValueError(_UNSET_OR_EMPTY.format(env_var=spec.env_var))
```

Keep the indentation of each site.

Mutation check (do it, then revert it): change one of the two call sites back to an inline f-string with different wording. Run `uv run pytest tests/test_model.py -q -k read_the_same` and expect red. Revert.

- [ ] **Step 5: Fix the stale constant name**

In `src/my_agent/capabilities.py:261`, change `` `LIBRARY_SUBAGENT_STEP_LIMIT` `` to `` `LIBRARY_STEP_LIMIT` ``. Then confirm the old name is gone:

Run: `grep -rn LIBRARY_SUBAGENT_STEP_LIMIT src tests docs CLAUDE.md`
Expected: no output. If `docs/` or `CLAUDE.md` still has it, fix those too, but not files under `docs/superpowers/`, which record history.

- [ ] **Step 6: Run the gate**

Run: `./scripts/check.sh`
Expected: every step green.

- [ ] **Step 7: Commit**

```bash
git add src/my_agent/mirror.py src/my_agent/model.py src/my_agent/capabilities.py tests/test_mirror.py tests/test_model.py
git commit -m "Fix three drifts from rules CLAUDE.md already states

The two run_id separator checks had byte-identical messages, so a test could not
tell which one fired; the backslash check gets its own. model.py spelled its
'unset or empty' message out twice; it is now one format string, pinned by a
test that both paths read the same. A docstring named LIBRARY_SUBAGENT_STEP_LIMIT,
which is LIBRARY_STEP_LIMIT.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Re-measure the tripped checks and update CLAUDE.md

The spec says: "Re-run the untripped-`require()` measurement after Tasks 1 and 4 ... and update CLAUDE.md's counts ... to whatever it reports. Do not update them by hand." (This plan's Task 3 is the spec's Task 4.) The tool that produced CLAUDE.md's numbers is not in the repo, so this task rebuilds it **outside the repo**.

**What the rebuilt tool reported on the unchanged tree (2026-09-23):** 102 of 112 `require()` sites tripped and 15 of 16 `raise CheckFailed` sites. The 11 untripped sites are all in `main.py`. CLAUDE.md says **113** `require()` sites and 103 tripped. The untripped set agrees, but the site count differs by one, and the old tool is not available to show why. So CLAUDE.md is updated with **this tool's** numbers, and the commit message says the method changed.

**Files:**
- Modify: `CLAUDE.md`
- Scratch (not committed): `$MEASURE/trip_plugin.py`, `$MEASURE/diff_sites.py`

- [ ] **Step 1: Write the measurement tool outside the repo**

```bash
export MEASURE="${TMPDIR:-/tmp}/trip-measure" && mkdir -p "$MEASURE"
```

`$MEASURE/trip_plugin.py`:

```python
"""pytest plugin: record every src/ check site that actually raised CheckFailed.

sys.monitoring RAISE fires where an exception is raised (not on unwind, not on
a bare re-raise). A raise inside negative_space.require is attributed to the
line that called require(); any other CheckFailed is attributed to its raise.
Writes the set to $TRIP_OUT on exit.
"""
import json
import os
import sys
from pathlib import Path

TOOL = 4
tripped: set[tuple[str, int, str]] = set()


def pytest_configure(config):
    from my_agent.negative_space import CheckFailed

    helpers = str(Path(sys.modules["my_agent.negative_space"].__file__).resolve())
    mon = sys.monitoring
    mon.use_tool_id(TOOL, "trip-measure")

    def on_raise(code, offset, exc):
        if not isinstance(exc, CheckFailed):
            return
        frame = sys._getframe(1)
        if str(Path(code.co_filename).resolve()) == helpers:
            if code.co_name == "require" and frame.f_back is not None:
                caller = frame.f_back
                tripped.add(
                    (str(Path(caller.f_code.co_filename).resolve()), caller.f_lineno, "require")
                )
        else:
            tripped.add((str(Path(code.co_filename).resolve()), frame.f_lineno, "raise"))

    mon.register_callback(TOOL, mon.events.RAISE, on_raise)
    mon.set_events(TOOL, mon.events.RAISE)


def pytest_unconfigure(config):
    sys.monitoring.set_events(TOOL, 0)
    sys.monitoring.free_tool_id(TOOL)
    Path(os.environ["TRIP_OUT"]).write_text(json.dumps(sorted(tripped)))
```

`$MEASURE/diff_sites.py`:

```python
"""AST-walk src/ for check sites and diff them against trip_plugin's output.

Usage: python diff_sites.py <src dir> <tripped.json>
"""
import ast
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
tripped = {tuple(t) for t in json.loads(Path(sys.argv[2]).read_text())}
sites = []
for py in sorted(src.rglob("*.py")):
    if py.name == "negative_space.py":
        continue
    for node in ast.walk(ast.parse(py.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require":
            sites.append((str(py.resolve()), node.lineno, "require"))
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "CheckFailed":
                sites.append((str(py.resolve()), node.lineno, "raise"))
for kind in ("require", "raise"):
    every = [s for s in sites if s[2] == kind]
    hit = [s for s in every if s in tripped]
    print(f"{kind}: {len(hit)} of {len(every)} tripped")
for s in sites:
    if s not in tripped:
        print("  untripped:", Path(s[0]).name, s[1], s[2])
```

- [ ] **Step 2: Run it**

```bash
TRIP_OUT="$MEASURE/tripped.json" PYTHONPATH="$MEASURE" uv run pytest -q -p trip_plugin -p no:cacheprovider > "$MEASURE/run.log" 2>&1; tail -1 "$MEASURE/run.log"
uv run python "$MEASURE/diff_sites.py" src/my_agent "$MEASURE/tripped.json"
git status --short   # must show no tripped.json or other stray file in the repo
```

Expected: every untripped site is in `main.py`. Tasks 1-4 add one `require()` (in `usage.py`, tripped by `test_a_result_holding_other_than_one_prompt_is_refused`) and remove two (the inline checks in `capabilities.py`), so the expected result is **101 of 111** `require()` sites and **15 of 16** raises. **If any site outside `main.py` is untripped, stop.** Add the missing test to the task that introduced the site, and re-run.

- [ ] **Step 3: Update CLAUDE.md with the numbers the tool printed**

Edit these and nothing else. Use the tool's printed numbers, not the expected ones above, if they differ.

1. "Testing and evals", the paragraph beginning "Outside one named exclusion": replace the `113 \`require()\` sites, 103 tripped; 16 \`raise CheckFailed\` sites outside the helpers, 15 tripped` figures with the measured ones. Add one sentence after them: "Re-measured with a `sys.monitoring` RAISE plugin and an AST walk; it counts one fewer `require()` site on the tree the earlier figure came from, and the old tool is not in the repo to say why."
2. "Testing and evals", first bullet: replace "502 offline tests and 2 live as of 2026-09-23" with the count from `uv run pytest -q | tail -1` and today's date.
3. The "Repo layout" table: add a row after `mirror.py`:
   `| \`usage.py\` | \`call_usage\` — the one reading of a model call's \`usage_metadata\`, shared by \`RunTokenBudget\` and \`JsonlMirror\` so the bound and the log cannot disagree. |`
4. The `run.py` row of the same table: after "failed (`DeadlineExceeded`, `StepLimitExceeded`, `TokenLimitExceeded`, `ResumeLimitExceeded`)" add ", all subclasses of `BoundExceeded`".
5. "Negative space programming", the bullet beginning "All four bounds fail the same way at the edge": add at its end: "All four subclass `run.BoundExceeded`, which is the one name `main` catches, so a fifth bound is reported without editing it."

- [ ] **Step 4: Run the gate and commit**

Run: `./scripts/check.sh`
Expected: every step green.

```bash
git add CLAUDE.md
git commit -m "Re-measure tripped checks and record usage.py and BoundExceeded in CLAUDE.md

Counts re-measured with a sys.monitoring RAISE plugin diffed against an AST walk,
since the tool behind the previous figure is not in the repo. It counts one
fewer require() site than CLAUDE.md stated on the unchanged tree; the untripped
set (11 sites, all in main.py) is the same.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the last task

- `superpowers:finishing-a-development-branch` decides between a PR and other options. `main` requires the `gate`, `Analyze (python)` and `Analyze (actions)` checks, so a PR is the only route in.
- The spec's Task 1 (subagents) is **not** in this branch. Its open questions 1, 2 and 4 need answers first.
