# Behavioural tests for the harness — design

**Date:** 2026-09-19
**Status:** approved, not yet implemented
**Constraint:** no live API calls. Everything here runs fake models against real compiled graphs.

## Problem

Nearly every bound in this harness is proven by reading a value back off a constructor. That
proves a number was *passed*; it proves nothing about whether the mechanism *runs*. The two are
indistinguishable in a green suite, which is how F31 — compaction configured above the window it
served — survived the project's entire life with 100% line coverage on the module that owned it.

`test_compaction_actually_shrinks_what_the_model_sees` (committed 2026-09-19) closed the largest
case and established the shape: build a real agent with a fake model that records what it was
handed, drive it with input that should trip the mechanism, assert on the result. It immediately
found a limitation nobody had stated — compaction cannot fire while every message fits inside
`keep`, so one enormous `ToolMessage` is uncompactable at any token threshold.

This spec covers what is left of that class, plus the adversarial cases the harness claims to
handle but nobody has driven.

## Audit: what is already behavioural

Checked per bound, not assumed. Recorded here so this work does not redo it.

| Bound | State | Evidence |
|---|---|---|
| `TOOL_CALL_LIMIT` | behavioural | `tests/test_run.py:951` — calls execute, then are blocked |
| `TASK_DISPATCH_LIMIT` | behavioural | `tests/test_agent.py:805` — a model that keeps dispatching |
| `SUBAGENT_STEP_LIMIT` | behavioural | `AlwaysDispatchesSubagents` exists for exactly this |
| Permission deny rules | behavioural | `tests/test_agent.py:520` — calls `write_file` on `/secrets/**` |
| `deadline_s`, `token_limit`, `step_limit`, `resume_limit` | behavioural | fake graphs in `tests/test_run.py` trip each |
| `failed_tool_calls` | behavioural + discriminator | `tests/test_run.py:884`, `:888` |
| `_unanswered_tool_calls` | behavioural | `tests/test_run.py:799`, `:836` |
| Compaction | behavioural | `tests/test_capabilities.py`, 2026-09-19 |
| **`GREP_MATCH_LIMIT`** | **read-back only** | three assertions, all constructor args or private attrs |
| **`TOOL_RESULT_TOKEN_LIMIT`** | **read-back only** | same |
| **`HUMAN_MESSAGE_TOKEN_LIMIT`** | **read-back only** | same |

## Phase 1 — the three remaining mechanism-fires tests

No production change. Values as of today: `GREP_MATCH_LIMIT = 1000`,
`TOOL_RESULT_TOKEN_LIMIT = 20000`, `HUMAN_MESSAGE_TOKEN_LIMIT = 50000`.

All three are applied by `least_privilege_filesystem` and all three are *eviction or truncation*
bounds — the same family as compaction, and the same failure mode: if the mechanism does not run,
nothing anywhere says so.

Tests go in `tests/test_capabilities.py`, in the existing sections for each bound, reusing
`RecordsWhatItWasAsked` (already added for compaction).

The two eviction mechanisms were read off `deepagents/middleware/filesystem.py` rather than guessed,
because they differ from each other and the assertions have to match:

- **`TOOL_RESULT_TOKEN_LIMIT`** truncates `read_file` output **in place**, when
  `len(content) >= NUM_CHARS_PER_TOKEN * limit` — `NUM_CHARS_PER_TOKEN` is 4, so **80,000
  characters** — appending `READ_FILE_TRUNCATION_MSG` (`filesystem.py:1972-1984`).
- **`HUMAN_MESSAGE_TOKEN_LIMIT`** offloads to the backend and tags the message with
  `additional_kwargs["lc_evicted_to"]`, then truncates. Threshold **200,000 characters**
  (`filesystem.py:3370-3380`).

Tests:

1. `test_grep_stops_at_the_match_limit` — seed a `StateBackend` with more matching files than
   `GREP_MATCH_LIMIT`, call `grep` through the compiled graph, assert the result is capped.
2. `test_an_oversized_read_is_truncated_in_place` — write a file over 80,000 characters into the
   backend, `read_file` it, assert the returned content is shorter and carries the truncation
   marker. Discriminator: a file just under the threshold comes back whole.
3. `test_an_oversized_trailing_human_message_is_evicted_to_the_backend` — a final `HumanMessage`
   over 200,000 characters is tagged `lc_evicted_to` and truncated.
4. `test_a_huge_human_message_that_is_not_last_is_never_evicted` — **the limitation this audit
   found.** `filesystem.py:3376` reads `messages[-1]` only, so eviction examines the *trailing*
   message and nothing else.

Item 4 matters more than its size suggests, and it should be written up as a finding. It is the
same shape the compaction test already exposed, in a second bound: a 416,000-character
`HumanMessage` sitting anywhere but last escapes eviction (not last) **and** compaction (inside
`keep`). Two independent context bounds, and the same message slips past both. That is not a bug in
either mechanism — each does what it says — but "the conversation is bounded" is a claim neither
one supports on that shape, and nothing currently says so.

**Mutants, per test:** raise the limit past the fixture (the fires-test must go red), and lower it
far below (the discriminator must go red). A test that survives both is decorative and gets fixed
before it lands — this exact failure happened while writing the compaction discriminator, which
passed under every trigger tried until it was resized.

## Phase 2 — `TurnResult.answered`

The one production change in this spec. TDD: test first.

F36 derived the rule and the arithmetic behind it: `finish_reason == "length"` with empty `outputs`
and no `tool_calls` means the model was cut off while still reasoning and the final channel was
never opened. The answer did not start — it was not truncated.

Today `main._single_turn` checks only that the last message is not a `HumanMessage`, so such a turn
prints an empty reply and exits 0. `answered` reads data the mirror already records and needs no
tokenizer at runtime.

It sits beside `failed_tool_calls` (F32), which is the same kind of thing: a deterministic fact
about the run rather than a judgement of the prose. Per the verification ladder, both are rung 1.

**Mutant:** hardwire `answered` to `True` — the test asserting a truncated turn is unanswered must
go red. This is the shape that caught a `paused` property hardwired to `False` previously.

## Phase 3 — adversarial fakes

Scripted models that misbehave the way real ones do, driving paths the harness already claims to
handle but that nobody has exercised:

- malformed tool arguments
- duplicate tool-call ids in one `AIMessage`
- an empty `AIMessage` carrying neither content nor tool calls

Each asserts the harness's *response*, not the model's output.

**Structure:** fakes stay local to the test file that uses them, following
`AlwaysDispatchesSubagents` and `DispatchesUntilBlocked` in `tests/test_agent.py`. No shared
`tests/fakes.py` until two files genuinely need the same fake — CLAUDE.md's one-test-file-per-module
rule points the same way.

## Explicitly not doing

**Detecting that the model claimed success without calling the tool.** It is the first cause listed
under "why agents report false success", and the harness cannot see it deterministically: knowing a
claim is false requires knowing what was asked, which requires a domain. Any heuristic here would be
a guess wearing a check's clothing, and CLAUDE.md spends a section refusing exactly that.

This belongs in `docs/findings.md` as a stated limit — the honest version of the gap — alongside the
existing "what is deliberately not verified" reasoning. The harness already exposes the full
tool-call record in `TurnResult.messages`, so when a domain lands, the eval that grades this has
what it needs.

## Testing and cost

Zero live calls. Fake models against real compiled graphs throughout. The existing
`_forbid_network` guard (hardened 2026-09-19 to cover DNS and datagram exits) enforces this rather
than assuming it.

Every test in this spec is mutation-verified before it lands: the code it covers is broken in place
and the test watched go red. A test that survives its mutant is decorative and does not count as
done. Mutants get recorded in the `docs/findings.md` ledger, which currently stops at F24/F29 and
was corrected on 2026-09-19 to say so.

## Sequencing

Phases are independent and can land as separate commits. Phase 1 is the largest and has no
production change, so it goes first. Phase 2 touches `run.py` and `main.py`. Phase 3 depends on
neither.
