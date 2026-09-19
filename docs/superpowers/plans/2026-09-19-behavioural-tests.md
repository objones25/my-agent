# Behavioural Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the harness's remaining read-back-only bounds actually fire, add the one deterministic run-fact the harness is missing, and drive the adversarial paths nobody has exercised.

**Architecture:** Every test builds a real compiled agent via `build_agent` with a *fake* `BaseChatModel` that scripts the model's side, then asserts on what the graph produced. No network. Pre-populate the `StateBackend` by passing `files=` on `invoke` — `StateBackend` raises if touched outside a graph run, so calling `tool.func(...)` directly does not work for reads.

**Tech Stack:** pytest 9.1.1, deepagents 0.7.15, langchain 1.4.1, langgraph 1.2.11, Python 3.13. `uv run` for everything.

**Spec:** `docs/superpowers/specs/2026-09-19-behavioural-tests-design.md`

## Global Constraints

- **No live API calls.** Fake models only. `tests/conftest.py::_forbid_network` enforces this and now covers `connect`, `connect_ex`, `create_connection`, `getaddrinfo`, `gethostbyname` and `sendto`.
- **Every test must be mutation-verified before it lands.** Break the code it covers in place, watch it go red, revert. A test that survives its mutant is decorative and does not count as done. Record each mutant in the commit message.
- **Assert the claim and its discriminator.** A "the bound fires" test is paired with a "and it does not fire below the threshold" test, or it is not finished.
- **Use `require()`/`CheckFailed`, never bare `assert`, in `src/`.** In test bodies plain `assert` is correct. A test expecting a tripped `require()` names `CheckFailed`, never `AssertionError`.
- **`filterwarnings = ["error"]`.** Any new warning fails the build. Close sockets and file handles in tests; an unclosed socket becomes a `PytestUnraisableExceptionWarning` and fails.
- **Run `./scripts/check.sh` before every commit.** It is what CI and the pre-commit hook run.
- **One test file per source module.** Fakes stay local to the test file that uses them.
- Measured values, verified 2026-09-19: `GREP_MATCH_LIMIT = 1000`, `TOOL_RESULT_TOKEN_LIMIT = 20000`, `HUMAN_MESSAGE_TOKEN_LIMIT = 50000`, `NUM_CHARS_PER_TOKEN = 4`. `read_file`'s default line limit is **100 lines**.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `tests/test_capabilities.py` | bounds owned by `capabilities.py`: grep, tool-result, human-message eviction | Modify — add one fake + 6 tests |
| `tests/test_run.py` | `TurnResult.answered`, adversarial model behaviour | Modify — add tests |
| `src/my_agent/run.py` | `TurnResult.answered` | Modify — one property |
| `src/my_agent/main.py` | report an unanswered turn | Modify — a few lines |
| `docs/findings.md` | F38: the context-bound gap | Modify — one finding |

A shared `tests/fakes.py` is deliberately **not** created. `AlwaysDispatchesSubagents` and `DispatchesUntilBlocked` already live in `tests/test_agent.py`; follow that.

---

### Task 1: A fake that issues one scripted tool call

**Files:**
- Modify: `tests/test_capabilities.py` (add near `RecordsWhatItWasAsked`, which already exists)

**Interfaces:**
- Consumes: nothing.
- Produces: `CallsOneTool(tool: str, args: dict[str, Any])` — a `BaseChatModel` whose first `_generate` returns an `AIMessage` with a single tool call `{"name": tool, "args": args, "id": "c1"}` and whose later calls return `AIMessage("done")`. Tasks 2–4 use it.
- Produces: `_tool_messages(agent, files, tool, args) -> list[ToolMessage]` — drives the agent once and returns the `ToolMessage`s produced.

- [ ] **Step 1: Add the fake and the helper**

```python
class CallsOneTool(BaseChatModel):
    """Issues one scripted tool call, then stops.

    `StateBackend` refuses to read or write outside a graph run, so a
    filesystem bound can only be exercised by a real dispatch. This is the
    smallest model that produces one.
    """

    tool: str = "read_file"
    args: dict[str, Any] = Field(default_factory=dict)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "calls-one-tool"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            message = AIMessage("", tool_calls=[{"name": self.tool, "args": self.args, "id": "c1"}])
        else:
            message = AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_messages(files: dict[str, Any], tool: str, args: dict[str, Any]) -> list[ToolMessage]:
    """Run one tool call against a pre-populated `StateBackend`.

    Files are passed on `invoke` because `StateBackend` cannot be written from
    outside a graph execution — its own error message says so.
    """
    agent = build_agent(CallsOneTool(tool=tool, args=args), AgentConfig())
    out = agent.invoke({"messages": [("user", "go")], "files": files}, {"recursion_limit": 25})
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]
```

Add `ToolMessage` to the `langchain_core.messages` import line.

- [ ] **Step 2: Verify it collects**

Run: `uv run pytest tests/test_capabilities.py -q`
Expected: PASS, same count as before (nothing new is called yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_capabilities.py
git commit -m "Add a fake that issues one scripted tool call"
```

---

### Task 2: `grep` stops at the match limit

**Files:**
- Modify: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: `CallsOneTool`, `_tool_messages` from Task 1.
- Produces: nothing.

Measured 2026-09-19: 1,200 matching files produce exactly 1,000 references and a trailing message containing `"maximum match count"`.

- [ ] **Step 1: Write both tests**

```python
def test_grep_stops_at_the_match_limit() -> None:
    """`GREP_MATCH_LIMIT` is a bound on a capability we granted. Read back off
    the middleware it is only a number that was passed; here it is the number
    of matches that actually came back."""
    files = {f"/f{i}.txt": {"content": "needle\n"} for i in range(GREP_MATCH_LIMIT + 200)}

    messages = _tool_messages(files, "grep", {"pattern": "needle"})

    result = str(messages[0].content)
    assert result.count("/f") == GREP_MATCH_LIMIT
    assert "maximum match count" in result


def test_grep_under_the_limit_returns_everything_and_says_nothing_about_truncation() -> None:
    """The discriminator. Without it the test above passes just as well if grep
    silently caps every search, which is a different and worse bug."""
    files = {f"/f{i}.txt": {"content": "needle\n"} for i in range(5)}

    messages = _tool_messages(files, "grep", {"pattern": "needle"})

    result = str(messages[0].content)
    assert result.count("/f") == 5
    assert "maximum match count" not in result
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_capabilities.py -q -k grep`
Expected: PASS (these document existing behaviour).

- [ ] **Step 3: Mutate — raise the limit**

```bash
sed -i '' 's/^GREP_MATCH_LIMIT = 1000$/GREP_MATCH_LIMIT = 5000/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k grep
```

Expected: `test_grep_stops_at_the_match_limit` FAILS. Then:

```bash
git checkout -- src/my_agent/capabilities.py
```

- [ ] **Step 4: Mutate — lower the limit**

```bash
sed -i '' 's/^GREP_MATCH_LIMIT = 1000$/GREP_MATCH_LIMIT = 3/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k grep
```

Expected: `test_grep_under_the_limit_...` FAILS. Then `git checkout -- src/my_agent/capabilities.py`.

If either mutant does not produce a failure, the test is decorative — fix it before continuing.

- [ ] **Step 5: Gate and commit**

```bash
./scripts/check.sh
git add tests/test_capabilities.py
git commit -m "Prove grep stops at the match limit

Mutants: limit 1000 -> 5000 kills the fires-test; 1000 -> 3 kills the
discriminator."
```

---

### Task 3: an oversized read is truncated, and the line limit that hides it

**Files:**
- Modify: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: `CallsOneTool`, `_tool_messages` from Task 1.

Measured 2026-09-19, and the reason this task has three tests rather than one:

- `read_file` truncates at `NUM_CHARS_PER_TOKEN * TOOL_RESULT_TOKEN_LIMIT` = **80,000 characters**, appending a message containing `"truncated due to size"` (`deepagents/middleware/filesystem.py:1972-1984`).
- But `read_file` applies a **100-line default limit first**. A 134,890-character file across 4,000 lines came back as 3,233 characters with **no** truncation marker — the line limit had already cut it.
- The character bound is therefore only reachable on files with few, very long lines. 10 lines of 20,000 characters produced 60,386 characters *with* the marker.

- [ ] **Step 1: Write the three tests**

```python
def test_an_oversized_read_is_truncated_with_a_marker() -> None:
    """`TOOL_RESULT_TOKEN_LIMIT` in the only units it is enforced in: characters,
    at `NUM_CHARS_PER_TOKEN` per token. Ten very long lines clear the line limit
    and reach the character bound."""
    fat = "".join("y" * 20_000 + "\n" for _ in range(10))
    assert len(fat) > 4 * TOOL_RESULT_TOKEN_LIMIT  # the fixture must actually be over it

    messages = _tool_messages({"/fat.txt": {"content": fat}}, "read_file", {"file_path": "/fat.txt"})

    result = str(messages[0].content)
    assert len(result) < len(fat)
    assert "truncated due to size" in result


def test_a_small_read_comes_back_whole() -> None:
    """The discriminator: `read_file` does not mark everything truncated."""
    small = "".join(f"line {i}\n" for i in range(50))

    messages = _tool_messages({"/s.txt": {"content": small}}, "read_file", {"file_path": "/s.txt"})

    result = str(messages[0].content)
    assert "truncated due to size" not in result


def test_the_line_limit_cuts_a_long_file_before_the_character_bound_can() -> None:
    """**The reachable surface of `TOOL_RESULT_TOKEN_LIMIT` is narrower than it
    looks.** `read_file` keeps 100 lines by default, so an ordinary long file is
    already small by the time the character bound is consulted and the
    truncation marker never appears. Measured: 4,000 lines and 134,890
    characters came back as ~3,000 characters, unmarked.

    Stated as a test so that a change to either limit has to confront the
    interaction rather than discover it.
    """
    many = "".join(f"line {i} padding padding padding\n" for i in range(4000))
    assert len(many) > 4 * TOOL_RESULT_TOKEN_LIMIT

    messages = _tool_messages({"/many.txt": {"content": many}}, "read_file", {"file_path": "/many.txt"})

    result = str(messages[0].content)
    assert result.count("\n") <= 100
    assert "truncated due to size" not in result
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_capabilities.py -q -k "read"`
Expected: all PASS.

- [ ] **Step 3: Mutate — raise the token limit**

```bash
sed -i '' 's/^TOOL_RESULT_TOKEN_LIMIT = 20000$/TOOL_RESULT_TOKEN_LIMIT = 200000/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k "read"
git checkout -- src/my_agent/capabilities.py
```

Expected: `test_an_oversized_read_is_truncated_with_a_marker` FAILS (the fixture's own `assert` on the fixture size will trip first — that is fine and is the point: the fixture is tied to the constant).

- [ ] **Step 4: Mutate — lower the token limit**

```bash
sed -i '' 's/^TOOL_RESULT_TOKEN_LIMIT = 20000$/TOOL_RESULT_TOKEN_LIMIT = 10/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k "read"
git checkout -- src/my_agent/capabilities.py
```

Expected: `test_a_small_read_comes_back_whole` FAILS.

- [ ] **Step 5: Gate and commit**

```bash
./scripts/check.sh
git add tests/test_capabilities.py
git commit -m "Prove an oversized read is truncated, and pin the line limit that pre-empts it"
```

---

### Task 4: human-message eviction, and the message it never looks at

**Files:**
- Modify: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: `RecordsWhatItWasAsked` (already in the file, added with the compaction tests).

Measured from `deepagents/middleware/filesystem.py:3370-3380`: eviction fires when the **last** message is a `HumanMessage` longer than `NUM_CHARS_PER_TOKEN * HUMAN_MESSAGE_TOKEN_LIMIT` = **200,000 characters**. It tags `additional_kwargs["lc_evicted_to"]` and truncates. `messages[-1]` is the only message examined.

- [ ] **Step 1: Write both tests**

```python
def test_an_oversized_trailing_human_message_is_evicted_to_the_backend() -> None:
    """`HUMAN_MESSAGE_TOKEN_LIMIT` enforced, not merely configured."""
    huge = "z" * (4 * HUMAN_MESSAGE_TOKEN_LIMIT + 1_000)
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())

    out = agent.invoke({"messages": [HumanMessage(huge)]}, {"recursion_limit": 25})

    evicted = [
        m for m in out["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_evicted_to")
    ]
    assert evicted != []
    assert len(evicted[0].text) < len(huge)


def test_a_huge_human_message_that_is_not_last_is_never_evicted() -> None:
    """**The bound examines `messages[-1]` and nothing else.**

    A message just as large, one position from the end, is untouched. Combined
    with compaction -- which cannot reach anything inside `keep` -- a large
    `HumanMessage` in the middle of a conversation escapes both context bounds.
    Neither mechanism is wrong; each does what it documents. This is the test
    that stops "the conversation is bounded" from being read as a claim either
    of them makes about that shape.
    """
    huge = "z" * (4 * HUMAN_MESSAGE_TOKEN_LIMIT + 1_000)
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())

    out = agent.invoke(
        {"messages": [HumanMessage(huge), HumanMessage("now answer")]},
        {"recursion_limit": 25},
    )

    evicted = [
        m for m in out["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_evicted_to")
    ]
    assert evicted == []
    assert any(len(m.text) == len(huge) for m in out["messages"] if isinstance(m, HumanMessage))
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_capabilities.py -q -k "human_message"`
Expected: both PASS. **If the first fails**, the threshold or the tag name has moved — read `filesystem.py:3370-3380` again and fix the test to match the wheel. Do not change the constant.

- [ ] **Step 3: Mutate — raise the limit**

```bash
sed -i '' 's/^HUMAN_MESSAGE_TOKEN_LIMIT = 50000$/HUMAN_MESSAGE_TOKEN_LIMIT = 500000/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k "human_message"
git checkout -- src/my_agent/capabilities.py
```

Expected: `test_an_oversized_trailing_human_message_is_evicted_to_the_backend` FAILS.

- [ ] **Step 4: Mutate — lower the limit**

```bash
sed -i '' 's/^HUMAN_MESSAGE_TOKEN_LIMIT = 50000$/HUMAN_MESSAGE_TOKEN_LIMIT = 10/' src/my_agent/capabilities.py
uv run pytest tests/test_capabilities.py -q -k "human_message"
git checkout -- src/my_agent/capabilities.py
```

Expected: `test_a_huge_human_message_that_is_not_last_is_never_evicted` FAILS — at a 10-token threshold `"now answer"` is itself oversized, so something *is* evicted.

- [ ] **Step 5: Gate and commit**

```bash
./scripts/check.sh
git add tests/test_capabilities.py
git commit -m "Prove human-message eviction fires, and that it only ever looks at the last message"
```

---

### Task 5: F38 — the shape both context bounds miss

**Files:**
- Modify: `docs/findings.md` (add after F37, before the `---` that precedes "Observability API reference")
- Modify: `CLAUDE.md` (the two `F1–F37` occurrences become `F1–F38`)

**Interfaces:**
- Consumes: the measurements proven by Tasks 3 and 4.

- [ ] **Step 1: Write the finding**

Add a section titled `## F38 — a large message in the middle of a conversation escapes every context bound`. It must state, with the measurements from Tasks 3 and 4 and from the compaction tests committed on 2026-09-19:

- Compaction keeps the last `COMPACTION_KEEP_MESSAGES` (6) messages verbatim and can only compact what is older, so a conversation of 6 or fewer messages is uncompactable at any token threshold. Measured: 3 messages worth ~104,000 approximate tokens reached the model whole, all 416,000 characters.
- Human-message eviction examines `messages[-1]` only (`filesystem.py:3376`), so a message of any size in any other position is never evicted.
- `TOOL_RESULT_TOKEN_LIMIT` truncates `read_file` output at 80,000 characters, but `read_file`'s 100-line default cuts most long files first, so the character bound is reachable only on files with few very long lines.
- Therefore a large `HumanMessage` or tool result sitting mid-conversation is bounded by none of the three.
- *What we do:* nothing yet — the shape is now tested, so a change to any of the three has to confront it. Note that `RunTokenBudget` (F35) still bounds the **run**, so this is a context-window risk, not an unbounded-spend risk.
- *Still unverified:* whether a real conversation reaches this shape in practice; it needs a domain to produce one.

- [ ] **Step 2: Add the open item**

In `## Open / unverified`, add a bullet pointing at F38 and noting that the three bounds have never been measured against a real conversation.

- [ ] **Step 3: Gate and commit**

```bash
./scripts/check.sh
git add docs/findings.md CLAUDE.md
git commit -m "F38: a large message mid-conversation escapes every context bound"
```

---

### Task 6: `TurnResult.answered`

**Files:**
- Modify: `src/my_agent/run.py` (add beside `failed_tool_calls`)
- Modify: `tests/test_run.py`
- Modify: `src/my_agent/main.py` (`_single_turn`)

**Interfaces:**
- Consumes: `TurnResult` as it exists today.
- Produces: `TurnResult.answered -> bool` — `False` when the final `AIMessage` has `response_metadata["finish_reason"] == "length"`, empty text, and no tool calls; `True` otherwise.

F36 derived the rule: under harmony, that combination means the model was cut off while still in the analysis channel and the final channel was never opened. The answer did not start — it was not truncated.

- [ ] **Step 1: Write the failing test**

```python
def test_a_turn_cut_off_before_the_answer_began_is_not_answered() -> None:
    """F36: `finish_reason == "length"` with no text and no tool calls means the
    model never opened its final channel. Printing that as an empty reply and
    exiting 0 reports a turn that did not happen."""
    cut_off = AIMessage(
        "",
        response_metadata={"finish_reason": "length"},
        usage_metadata={"input_tokens": 84, "output_tokens": 24, "total_tokens": 108,
                        "output_token_details": {"reasoning": 21}},
    )
    result = TurnResult(messages=[HumanMessage("count to 200"), cut_off])

    assert result.answered is False


def test_an_ordinary_turn_is_answered() -> None:
    """The discriminator. Without it `answered` hardwired to `False` passes."""
    result = TurnResult(messages=[HumanMessage("hi"), AIMessage("hello")])

    assert result.answered is True


def test_a_turn_cut_off_after_calling_a_tool_is_answered() -> None:
    """A length-capped turn that still produced a tool call did real work. Only
    the no-text-and-no-calls combination means nothing started."""
    cut_off = AIMessage(
        "",
        tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
        response_metadata={"finish_reason": "length"},
    )
    result = TurnResult(messages=[HumanMessage("list files"), cut_off])

    assert result.answered is True
```

**Note:** `TurnResult`'s real constructor signature must be read from `src/my_agent/run.py` before writing these — if it takes more required fields than `messages`, pass them. Do not change the constructor to suit the test.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_run.py -q -k answered`
Expected: FAIL with `AttributeError: 'TurnResult' object has no attribute 'answered'`.

- [ ] **Step 3: Implement**

```python
    @property
    def answered(self) -> bool:
        """False when the model was cut off before its answer began.

        F36: under harmony a response opens an analysis channel, reasons, then
        opens a final channel. `finish_reason == "length"` with neither text nor
        tool calls means the cap landed inside the reasoning and the final
        channel was never opened — the answer did not start, so there is nothing
        to have been truncated. A fact about the run, like `failed_tool_calls`,
        not a judgement of the prose.
        """
        last = self.messages[-1] if self.messages else None
        if not isinstance(last, AIMessage):
            return True
        if last.response_metadata.get("finish_reason") != "length":
            return True
        return bool(last.text) or bool(last.tool_calls)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_run.py -q -k answered`
Expected: all three PASS.

- [ ] **Step 5: Mutate**

```bash
# hardwire it
python3 - <<'EOF'
import pathlib, re
p = pathlib.Path("src/my_agent/run.py"); s = p.read_text()
s = s.replace("        last = self.messages[-1] if self.messages else None", "        return True  # MUTANT", 1)
p.write_text(s)
EOF
uv run pytest tests/test_run.py -q -k answered
git checkout -- src/my_agent/run.py
```

Expected: `test_a_turn_cut_off_before_the_answer_began_is_not_answered` FAILS.

- [ ] **Step 6: Surface it in `main`**

In `main._single_turn`, after the existing checks, print a line when `not result.answered` saying the turn was cut off before the answer began and naming the token cap as the likely cause. Read the surrounding code and match its output style. Do not raise — this is a report, not an error.

- [ ] **Step 7: Gate and commit**

```bash
./scripts/check.sh
git add src/my_agent/run.py src/my_agent/main.py tests/test_run.py
git commit -m "Add TurnResult.answered: a turn cut off before its answer began

Mutant: `answered` hardwired to True kills the F36 case."
```

---

### Task 7: adversarial model behaviour

**Files:**
- Modify: `tests/test_run.py`

**Interfaces:**
- Consumes: `run_turn`, `TurnResult`, and the existing fake-graph helpers in `tests/test_run.py`.

These drive paths the harness already claims to handle. Read the existing fakes at the top of `tests/test_run.py` and follow their shape.

- [ ] **Step 1: Write the tests**

Use the existing `FakeGraph` (`tests/test_run.py:92`), which takes the dict its `invoke` returns.
`run_turn`'s message postcondition is **relative** (`> len(sent)`), so the fake must return more
messages than were sent or it trips for the wrong reason.

```python
def test_a_turn_whose_model_returned_nothing_at_all_still_produces_a_result() -> None:
    """An `AIMessage` with neither content nor tool calls is a real provider
    outcome. `run_turn` must return a `TurnResult` rather than trip a
    postcondition — the turn happened, it just said nothing."""
    agent = FakeGraph({"messages": [HumanMessage("say something"), AIMessage("")]})

    result = run_turn(agent, "say something")

    assert result.failed_tool_calls == []
    assert result[-1].text == ""


def test_duplicate_tool_call_ids_let_one_result_answer_two_calls() -> None:
    """**A limitation, asserted so it is known rather than discovered.**

    `_unanswered_tool_calls` (`src/my_agent/run.py:574`) collects requested ids
    into a list and answered ids into a *set*, then filters by membership. Two
    calls sharing an id are therefore both satisfied by a single `ToolMessage`,
    so a genuinely unanswered second call passes the check.

    A provider that reuses ids within one `AIMessage` is not something this
    harness has seen, and counting by multiplicity would be a small change. The
    reason to record it rather than fix it: nothing today produces the shape,
    and an unused branch is a branch nobody tests. If a provider ever does,
    this test names the behaviour to change.
    """
    calls = [
        {"name": "ls", "args": {}, "id": "dup"},
        {"name": "read_file", "args": {"file_path": "/a"}, "id": "dup"},
    ]
    history = [
        HumanMessage("go"),
        AIMessage("", tool_calls=calls),
        ToolMessage("ok", tool_call_id="dup", name="ls"),
    ]
    agent = FakeGraph({"messages": [*history, HumanMessage("next"), AIMessage("done")]})

    result = run_turn(agent, "next", history=history)

    assert result[-1].text == "done"
```

**Before asserting, confirm the second test's premise against the installed code.** If
`_unanswered_tool_calls` has since been changed to count by multiplicity, the call raises
`CheckFailed` instead — in that case assert the raise with `pytest.raises(CheckFailed,
match="unanswered tool call")` and delete the limitation paragraph from the docstring. Assert what
is true, not what this plan predicted.

- [ ] **Step 2: Run and reconcile**

Run: `uv run pytest tests/test_run.py -q -k "returned_nothing or duplicate"`
Expected: the first PASSES. The second either passes or reveals the set/list behaviour above — in which case rewrite it to assert what is true and say so.

- [ ] **Step 3: Gate and commit**

```bash
./scripts/check.sh
git add tests/test_run.py
git commit -m "Drive the adversarial model paths the harness claims to handle"
```

---

## Final verification

- [ ] **Run the whole gate**

```bash
./scripts/check.sh
uv run pytest -q
```

Expected: all pass. Test count should be 360 + the tests added above.

- [ ] **Confirm no live calls were introduced**

```bash
uv run pytest -q -m live --collect-only
```

Expected: still exactly 2 live tests. This plan adds none.
