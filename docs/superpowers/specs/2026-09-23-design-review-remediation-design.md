# Design review remediation: the bounds a new subagent or a new budget would escape

**Status: draft, under discussion. Not approved for implementation.** Task 1's direction was
decided on 2026-09-23 (see Task 1). Open questions 1-4 are still open.

A design review of `src/` on 2026-09-23 (the `design-patterns-expert` pass: smell scanner, then a
full read of every module except `main.py`) found no missing pattern. The protocols, the
composition root, the parameter-object configs and `_BACKEND_SOURCES` already do what a pattern
would add. What it found is **three places where the next ordinary extension silently loses a
guarantee**, and a handful of small drifts from rules this repo already states.

This spec covers five tasks, ordered by risk. Task 1 reverses an earlier decision, and its
direction is now settled. Whether all of it clears the YAGNI bar is still being discussed.

## What this builds

1. **Subagents are bounded on the same terms as the parent.** The `general-purpose` subagent
   that exists today gets the call limits it lacks, while keeping our compaction bound (1a). A
   caller's subagent must be a `CompiledSubAgent` carrying a step limit and the `TOOL_CALL_LIMIT`
   call limit, or the build fails (1b). The helper that would build one for you is deferred until
   a subagent exists (1c).
2. **One exception base for the four run bounds.** `main.py` currently catches four hand-listed
   classes. A fifth bound that is not added to that tuple becomes a traceback rather than a report
   on the single-prompt path (`uv run my-agent "..."`). The live checks are unaffected, because
   `_attempt` catches `Exception` and records a failed check.
3. **One reader for per-call token usage.** `RunTokenBudget` and `JsonlMirror` each parse
   `usage_metadata` with different rules, and for a multi-choice response they disagree by a
   factor of *n*.
4. **`capabilities.py`'s parameter checks go through `contracts.py`.** Two load-time checks
   re-implement `check_required_parameters` inline, and one `inspect.signature` call is made twice.
5. **Three small drifts fixed.** Two `require()`s with byte-identical messages, one duplicated
   `ValueError` text, and a docstring naming a constant that was renamed.

## Out of scope, with the trigger that would bring each in

- **A `TurnBudget` Strategy for `RunDeadline` / `RunTokenBudget`.** They are twins: accumulate in
  `TurnResult`, carry across a pause, resume on the remainder. A third bound of that shape — a cost
  ceiling is the obvious one — touches about ten sites: in `run.py` the constant, the `RunBounds`
  field and its check, the exception, the callback class, the `_run_config` and `_invoke`
  parameters, the `TurnResult` field, construction in `run_turn`, and the remainder check in
  `resume_turn`; in `main.py` the `except` tuple (Task 2 removes that one). Two instances do not
  pay for the indirection, and it would turn `TurnResult.elapsed_s` / `.tokens` into lookups that
  about a dozen tests in `test_run.py` read by name. **Trigger: the third pause-spanning
  budget.** The seam is a Protocol with `remaining(spent: TurnResult) -> ...`, `handler() -> BaseCallbackHandler` and
  `spent_from(handler) -> ...`.
- **Splitting `capabilities.py`.** It is 852 lines and 31 exports doing four jobs: the allowlist,
  the bound constants, the supply-chain pins (plugins, caching probes, harness profile), and the
  graph read-backs (`compiled_tools`, `compiled_tool_names`, `subagent_graphs`,
  `bound_step_limit`, `compiled_output_keys`, `_closure_values`). **Trigger: the next read-back
  of a library internal.** Move the read-back group to `introspection.py` with its own test file.
  That file then becomes the one to audit on every deepagents upgrade.
- **A base class or mixin for `as_kwargs`.** It is one identical line in `ModelConfig` and
  `AgentConfig`, and `check_config_contract` already couples the two by contract.
- **A parameter object for `_invoke`'s eight parameters.** Its `noqa` records why, and folding
  them would not remove any of the ten sites above.
- **`main.py`.** It is scaffolding (see memory, and CLAUDE.md's untripped-check exclusion).
- **Anything in `JsonlMirror`'s callback signatures.** They are LangChain's.

## Corrections to the review as first reported

Two findings from the first review pass did not survive checking against the tests. They are
recorded here so they are not re-raised:

- **"A caller's subagent escapes the step limit" is not an unnoticed bug. It is a decision.**
  `test_a_caller_supplied_subagent_keeps_its_own_step_limit` (`tests/test_agent.py:759`) asserts
  that the limit is `LIBRARY_STEP_LIMIT`, on purpose. The decision is recorded only there and in
  the `AgentConfig.subagents` docstring (`agent.py:187-196`). There is no F-entry. Task 1 argues
  for reversing it.
- **`agent.py:345-348` does not duplicate `capabilities.py:610`.** The callee's postcondition
  compares the middleware against the rules it *received*. The call-site check compares it against
  the rules `create_deep_agent` will get, so it catches a call site that passed the wrong rules.
  `test_agent_kwargs_refuses_middleware_that_lost_the_permission_rules` drives exactly that. It
  stays.

## Verified facts

Checked against this repo and the installed wheels on 2026-09-23, then re-checked the same day in a
second pass that re-ran each measurement below and corrected what did not hold. The scratch probes
used are not in the repo.

**A caller `SubAgent` spec compiles at 9999, and the build passes.** Given
`{"name": "researcher", ..., "middleware": [least_privilege_filesystem(None)]}`, `build_agent`
returns normally, and reading each subagent graph back gives:

```
general-purpose 25
researcher 9999
```

`TASK_DISPATCH_LIMIT`, `RunDeadline` and `RunTokenBudget` still apply, so the turn has an end.
What escapes is the step bound, which is F24 again, arriving through `AgentConfig.subagents`.

**The `general-purpose` subagent has no call limits today.** Read off the graphs `build_agent`
returns now, with the node names that `ToolCallLimitMiddleware`'s `after_model` hook adds:

```
parent          : ['ToolCallLimitMiddleware.after_model', 'ToolCallLimitMiddleware[task].after_model']
general-purpose : []
```

So inside one dispatch, `TOOL_CALL_LIMIT` does not apply. The subagent can fan out any number of
calls per step for its 25 steps, times `TASK_DISPATCH_LIMIT` (3) dispatches. `RunTokenBudget` and
`RunDeadline` still end the turn. This is F30's "reaches the parent only", measured on the
subagent that exists rather than a hypothetical one.

**Only one of the two call limits means anything on a subagent.** The `general-purpose` graph's
tools are `delete edit_file glob grep ls read_file write_file`. There is no `task`, so
`ToolCallLimitMiddleware[task]` (the `TASK_DISPATCH_LIMIT` bound) would be installed there but
could never fire. `TOOL_CALL_LIMIT` is the one that bounds a subagent.

**Call limits *can* be read back off a compiled graph, which corrects CLAUDE.md in part.**
CLAUDE.md (and F41) say there is "no route to read the installed stack back off a compiled
graph". That holds for middleware that only wraps calls. It does not hold for middleware with
node hooks: each becomes a node named `<middleware.name>.<hook>`, visible in `graph.nodes`. The
call limits are of that kind, which is what makes the 1a and 1b read-backs possible. Compaction
is not: `SummarizationMiddleware` adds no node, so it cannot be checked this way, and the next
fact is why that matters.

**An explicit `general-purpose` spec replaces our compaction unless it carries it.** In an explicit
`SubAgent` spec, deepagents builds a base stack of its own `FilesystemMiddleware`,
`create_summarization_middleware(model, backend)` and `PatchToolCallsMiddleware`, and then
replaces entries by `.name` with *the spec's* `middleware` only. The default `general-purpose`
path is different. It inherits every parent middleware whose `.name` matches a default slot, which
is how our `bounded_compaction` reaches it today. So a spec built from `GENERAL_PURPOSE_SUBAGENT`
(keys: `description`, `name`, `system_prompt`) with `middleware=[our filesystem, *call_limits()]`
compiles to a graph whose node set is today's plus exactly the two limit nodes, with the same
tools and `execute` withheld. That node comparison **passes**. But the compaction inside it is
deepagents' own:

```
today's general-purpose     compaction is ours: True   trigger ('tokens', 96000)
spec [fs, *call_limits()]   compaction is ours: False  trigger ('tokens', 170000)
```

170,000 is above the 131,072 window, which is the state F31 exists to prevent. Adding our
compaction middleware object to the spec fixes it. With
`middleware=[our filesystem, our bounded_compaction, *call_limits()]` the stack is today's plus
the two limits, in the same order, and the compaction is ours:

```
today's general-purpose : FilesystemMiddleware, SummarizationMiddleware, PatchToolCallsMiddleware,
                          AnthropicPromptCachingMiddleware
with the fixed spec     : FilesystemMiddleware, SummarizationMiddleware (ours, 96000),
                          PatchToolCallsMiddleware, ToolCallLimitMiddleware,
                          ToolCallLimitMiddleware[task], AnthropicPromptCachingMiddleware
```

The step limit is 9999 until the existing rebind, as today. These stacks were read by wrapping
`deepagents.graph._apply_custom_middleware` in a probe, not off the compiled graph.

**Rebinding a compiled subagent works through the existing route.** (Measured while Task 1 was
still proposing to rebind caller specs. That is no longer the plan, but the fact still backs the
`general-purpose` rebind.) Compile once, read each
graph out with `subagent_graphs`, hand each back as a `CompiledSubAgent` with
`graph.with_config({"recursion_limit": 25})`, compile again. After the second build:

```
general-purpose 25  execute-granted=False
researcher      25  execute-granted=False
```

The tool allowlist survives because the rebound runnable *is* the graph deepagents compiled with
our middleware in it. The route `_bounded_general_purpose_subagent` already relies on generalises
unchanged.

**langchain-openai attaches the whole request's usage to every choice.**
`BaseChatOpenAI._create_chat_result` reads `response_dict["usage"]` once and assigns it to
`message.usage_metadata` inside `for res in choices`. So for an `n`-choice response, every
generation carries the full request's usage. Feeding one `LLMResult` holding two generations
(reporting 100 and 40 tokens) to both callbacks gives:

```
budget 140  mirror 40
```

`RunTokenBudget` sums every generation (and for ChatOpenAI that is n× the real figure).
`JsonlMirror` keeps the last one. **Neither diverges today**, because `ModelConfig` has no `n`
field and every call returns one choice. The bug is latent. `model.py`'s module docstring says
adding a setting is one new field (its examples are `top_p` and `seed`), and `n` would be such a
field.

**A callback's `LLMResult` holds exactly one prompt.** `BaseChatModel.generate` flattens its
results before calling back: each prompt's run manager gets
`LLMResult(generations=[res.generations], ...)`. So the outer list `on_llm_end` receives always has
length 1, and the only multiplicity that can reach either callback is choices within one request.

**Adding the base exception breaks nothing.** The four class names appear on 26 lines of
`test_run.py`, 4 of `test_agent.py` and 2 of `test_main.py`, always by their own name. All four
subclass `RuntimeError` directly today.

**Test messages Task 4 changes.** `tests/test_capabilities.py:1245` matches
`"no longer accepts tools/_permissions"`, which is the inline message being replaced. `:1261`
matches `"no longer accepts"` and survives. `test_contracts.py:189`, `:201` and `:209` call
`check_required_parameters` directly and are unaffected by an added optional parameter.

**The separator checks are distinguishable by input, not by message.**
`test_run_log_path_rejects_a_run_id_with_a_path_separator` and `..._with_a_backslash` both match
`"separator"`. Each input trips only one of the two `require()`s, so both are covered. But the
messages are byte-identical (`mirror.py:573-582`), which CLAUDE.md's
"give every check a message no other check could produce" rule forbids.

## Architecture

### Task 1 — subagents: close today's gap, refuse what is not designed, defer the rest

**Decided (2026-09-23):**

- **Caller subagents are `CompiledSubAgent`s.** The caller compiles the graph, so the caller
  controls it, and there is no hand-rolled rebinding of someone else's spec. The
  compile-read-rebind-recompile route stays only for the subagent deepagents adds itself.
- **Subagents get `call_limits()` on the same terms as the parent.** They are installed by
  `build_agent` where it compiles the graph, and required where the caller does.

**The YAGNI line.** No custom subagent exists yet, so building *for* one is speculative. But two
parts of this task are not about future subagents:

- **The `general-purpose` subagent exists today and runs with no call limits** (measured below).
  That is a gap in the current build, the same kind F30 records for `AgentConfig.middleware`.
- **A refusal is not a feature.** `execute` is withheld by refusing it, not by building a sandbox.
  Refusing a subagent that is not bounded builds nothing for subagents. It stops the first one from
  silently running at 9999 with no call limits, which is what happens today.

So Task 1 splits into three parts, and only the first two are proposed for implementation.

#### 1a — the `general-purpose` subagent gets the call limits (now: the gap exists today)

Today `build_agent` lets deepagents add `general-purpose` from its own spec, then rebinds the step
limit. The call limits never reach it: deepagents inherits caller middleware into that subagent
only when the `.name` shadows a default slot, and a call limit shadows none.

**Shape.** The first compile hands deepagents an explicit `general-purpose` spec built from
deepagents' own `GENERAL_PURPOSE_SUBAGENT` (`name`, `description`, `system_prompt`), with
`middleware=[filesystem, compaction, *call_limits()]`, where `filesystem` and `compaction` are
**the same objects** `_agent_kwargs` assembled for the parent (`least_privilege_filesystem(...)`
and `bounded_compaction(model, backend)`). deepagents still does the compiling. The existing rebind
then applies `SUBAGENT_STEP_LIMIT` and hands it back as a `CompiledSubAgent`, unchanged.

**The compaction entry is not optional.** An explicit spec gets deepagents' own summarization at
170,000 tokens unless the spec carries ours (see "Verified facts"). Leaving it out loosens a bound
that holds today while tightening another.

**Measured, with the compaction entry included:** the stack is today's `general-purpose` stack
plus `ToolCallLimitMiddleware` and `ToolCallLimitMiddleware[task]`, in the same order. The
compaction is ours (96,000), the tools are identical, `execute` is withheld, and the step limit is
9999 before the rebind. This settles the question the "a hand-rolled replacement would silently
drop whichever of those moved next" warning in `_bounded_general_purpose_subagent`'s docstring
raises, for deepagents 0.7.15. It does not settle it for the next release, which is what the
read-back below is for.

**Comparing node sets does not verify this.** A spec that drops our compaction passes a node-set
comparison, because summarization adds no node. That comparison can stay as a check that the call
limits arrived. It cannot be the check that nothing else moved.

**Read-back:** `_require_subagents_bounded` asserts the `TOOL_CALL_LIMIT` node on
`general-purpose`, the same way it asserts the step limit. The node name is the library's to
choose, so derive it from the `call_limits()` entries rather than writing it as a literal.
Compaction cannot be read off the compiled graph (it adds no node), so it is pinned before the
fact instead, as F41 does for the rest of the stack: a postcondition in `build_agent` that the
spec's `middleware` holds the very compaction object `_agent_kwargs` built, found by identity.
Whether some internal route reads it back off the graph is not established. If one is found, add
it.

**Tests:** `test_build_agent_bounds_the_general_purpose_subagents_tool_calls` (a read-back of the
node); a behavioural test where the subagent fans out more than `TOOL_CALL_LIMIT` calls in one
step and gets error `ToolMessage`s back; a discriminator showing a bare `create_deep_agent`
subagent has no limit nodes, so the test cannot pass by deepagents adding them itself; and a test
that the spec's compaction is the parent's object, not deepagents' default.
**Mutations:** drop `*call_limits()` from the spec's middleware → the node and fan-out tests go
red. Drop the compaction entry → the compaction test goes red. Only this test catches that
mutation, because the node set does not change.

#### 1b — caller subagents: `CompiledSubAgent` only, and a tripwire (now: a refusal, not a feature)

| Entry in `AgentConfig.subagents` | Today | After |
|---|---|---|
| none | deepagents' `general-purpose`, step-bounded | same, plus call limits (1a) |
| `SubAgent` dict, any name | compiles at 9999, no call limits, build passes | **refused** in `AgentConfig.__post_init__` |
| `CompiledSubAgent` with a chosen step limit and the `TOOL_CALL_LIMIT` call limit | allowed | allowed, untouched |
| `CompiledSubAgent` unbound / at 9999 / missing a call limit | allowed | **build fails**, naming the subagent and what it lacks |

**Shape:**

- `AgentConfig.subagents` narrows to `Sequence[CompiledSubAgent]`. `__post_init__` requires each
  entry to carry a `runnable`, with a message saying to compile the subagent and bind
  `SUBAGENT_STEP_LIMIT` with `with_config`. A `TypedDict` cannot be checked with `isinstance`, so
  the `runnable` key is the check.
- `_require_subagents_bounded` loops every subagent graph. For each one it requires a step limit
  that is not `None` and not `LIBRARY_STEP_LIMIT`, and the `TOOL_CALL_LIMIT` node. It does not
  require the `task` limit, which is inert on a graph with no `task` tool (open question 4). The caller's
  `CompiledSubAgent` keeps whatever limit they chose. "Control" means the number is theirs, not
  that "no number" is allowed.
- `_supplies_general_purpose_subagent` keeps its role (a caller's compiled `general-purpose`
  replaces ours) and now only ever sees `CompiledSubAgent`s.
- The `AgentConfig.subagents` docstring (`agent.py:187-196`) is rewritten to the table.

**Tests.** Five tests pass subagents as plain specs and must change:

- `test_build_agent_fails_when_a_subagent_re_grants_the_shell_tool` (`test_agent.py:435`) and
  `test_build_agent_refuses_a_vacuous_subagent_allowlist_check` (`:484`) use a dict
  `general-purpose` spec, the first to reach the shell read-back and the second to skip the rebuild
  and reach the vacuity guard. They become compiled specs.
- `test_a_caller_supplied_subagent_keeps_its_own_step_limit` (`:759`) is replaced by
  `test_agent_config_refuses_a_subagent_spec_it_would_have_to_compile` and
  `test_a_caller_compiled_subagent_keeps_the_limit_it_bound` (bind 7, read 7 back).
- `test_a_caller_supplied_subagent_cannot_re_grant_the_shell_tool` (`:777`, a dict spec
  re-granting `execute`) is now refused earlier. The shell read-back keeps its coverage through a
  `CompiledSubAgent` compiled *without* our filesystem middleware.
- `test_agent_config_freezes_the_subagents_it_was_given` (`:793`) switches to a compiled spec.

New tests: `test_build_agent_refuses_a_compiled_subagent_with_no_step_limit_of_its_own` and
`..._without_the_tool_call_limit`, each matched on the subagent's name.
**Mutations:** revert the loop to `general-purpose` only → the refusal tests go red. Drop the
`runnable` check → the dict-spec refusal test goes red.

#### 1c — a `bounded_subagent()` helper (deferred: YAGNI)

Under 1b, a caller building a subagent has to assemble `least_privilege_filesystem`,
`call_limits()`, `bounded_compaction` and the step-limit rebind by hand. The tripwire names a
missing step limit or call limit. It does **not** name a missing `bounded_compaction`, because
compaction cannot be read off a compiled graph, so a caller's subagent compacting at deepagents'
170,000 would pass 1b. A factory in `capabilities.py` returning a ready `CompiledSubAgent` would
remove that. **Trigger: the first custom subagent.** Not built now. A factory with no caller has
nothing to shape it. The tripwire makes a forgotten step limit or call limit fail loudly, but not a
forgotten compaction bound. That gap is an argument for building the factory with the first
custom subagent, not before.

**Docs (1a + 1b):** new **F48** in `docs/findings.md` covering: the 9999 measurement, the missing
call limits on `general-purpose`, the decision and its YAGNI reasoning, the finding that an
explicit subagent spec gets deepagents' 170,000-token compaction unless it carries ours (and that
a node-set comparison cannot see it), and the partial correction to CLAUDE.md's "no route to read
the installed stack back" claim (true for wrap-only middleware such as compaction, false for
middleware with node hooks such as the call limits). Update the CLAUDE.md bullets on
`AgentConfig.subagents` ("a second door") and `AgentConfig.middleware reaches the parent only`.

### Task 2 — `BoundExceeded`

```python
class BoundExceeded(RuntimeError):
    """A turn ran out of something `RunBounds` rationed. An operating error."""

class DeadlineExceeded(BoundExceeded): ...
class StepLimitExceeded(BoundExceeded): ...
class TokenLimitExceeded(BoundExceeded): ...
class ResumeLimitExceeded(BoundExceeded): ...
```

Still a `RuntimeError`, so anything catching that today is unaffected. `CheckFailed` stays
outside it: a programmer error is not a spent budget. Add to `__all__`. `main.py:676-681` becomes
`except BoundExceeded`. That is the one edit to `main.py` in this spec, because the bug it removes
is that `main.py` has to be edited.

**Tests** (`tests/test_run.py`): a parametrised `issubclass` over the four, and one test that none
of them is a `CheckFailed`. `tests/test_main.py`: a local `BoundExceeded` subclass raised by a fake
turn on the **single-prompt path** is reported, not raised. That is the property the tuple could
not give. The test must go through `_single_turn`: on the live-check path, `_attempt`
(`main.py:588-591`) catches `Exception` and the test would pass with or without this change.
**Mutation:** restore the tuple in `main.py` → the local-subclass test goes red.

### Task 3 — one reader for per-call usage

New module `src/my_agent/usage.py` with its own `tests/test_usage.py`, per the one-file-per-module
rule. The dependency direction decides where it lives: `run.py` is core and `mirror.py` is
observability. Neither should import the other, and both import this.

```python
def call_usage(response: LLMResult) -> UsageMetadata | None:
    """The request's usage, or None if none was reported.

    A chat model's callback receives one prompt per `LLMResult`
    (`BaseChatModel.generate` flattens before calling back), so
    `response.generations` has one inner list: that request's choices.
    langchain-openai copies the request's usage onto *every* choice
    (`_create_chat_result`), so the first choice carrying usage is the
    request's figure and the rest are duplicates of it.
    """
```

- Precondition: `len(response.generations) == 1`. The flattening is langchain-core's behaviour,
  not ours, so state it as a check rather than assume it.
- `RunTokenBudget.on_llm_end` adds `total_tokens` from `call_usage(response)`. `None` increments
  `unmeasured_calls` exactly as today.
- `JsonlMirror.on_llm_end` writes `usage` from the same function, so a one-choice call's log line
  is byte-identical to today's.
- The `RunTokenBudget.on_llm_end` docstring's claim, "the number this bounds is the number the log
  shows", becomes true by construction. Say so there.

**Tests:** two choices each carrying 100 → 100, not 200. No usage anywhere → `None` and one
unmeasured call. An `LLMResult` with two prompt lists trips the precondition. Plus a test that the
budget and the mirror report the same total for the same `LLMResult`.
**Mutation:** make `call_usage` sum every choice's usage → the two-choice test goes red.

### Task 4 — `capabilities.py` checks through `contracts.py`

- `contracts.check_required_parameters` gains `consequence: str = "the config must change"`,
  mirroring `check_known_parameters`'s `why_new_matters`. The default keeps the message unchanged
  for its one caller in `src/` (`model.py:163`) and for the tests in `test_contracts.py`.
- Replace `capabilities.py:516-523` and `:525-530` with two calls. Keep each current consequence
  text ("the agent would run at deepagents' own threshold", "build_agent must change").
- Compute the `FilesystemMiddleware.__init__` signature once. `:273` and `:289` both call
  `inspect.signature` on it today.
- Update `tests/test_capabilities.py:1245`'s `match=` to the new wording.
  `tripping_an_import_time_check` still drives both checks, because they still run at import.

**Mutation:** drop `_permissions` from the needed set → the `:1245` test goes red.

### Task 5 — small drifts

- `mirror.py:573-582`: give the backslash check its own message ("…must not contain a backslash
  …"), and tighten the backslash test's `match=` so it can only match that site.
- `model.py:254-257` and `:265-268`: hoist the repeated "is unset or empty" text into one
  module-level format string used by both.
- `capabilities.py:261`: `LIBRARY_SUBAGENT_STEP_LIMIT` → `LIBRARY_STEP_LIMIT`.

## Order and gates

Tasks 2-5 do not touch subagents and do not wait on the open questions, so they can land first,
in that order. Task 1 follows once questions 1, 2 and 4 are settled, with 1a before 1b. Each task is one commit
and lands test-first. `./scripts/check.sh` is green after each. Re-run the untripped-`require()`
measurement after Tasks 1 and 4, because both add or move check sites, and update CLAUDE.md's
counts (currently 113 `require()` / 16 `raise CheckFailed`) to whatever it reports. Do not update
them by hand.

## Open questions

1. **Does 1a clear the YAGNI bar?** The gap is real today, but the turn is still ended by the
   token budget and the deadline. The stack comparison is done. With our compaction in the spec,
   nothing is dropped. Without it, compaction loosens to 170,000. The cost 1a adds is that
   deepagents' auto-added subagent becomes one we specify, so the next deepagents release that
   adds middleware to the default subagent does not reach ours. That is the drift the rebind was
   designed to avoid. The alternative is to leave `general-purpose` as it is and record the gap in
   F48 with its worst case (3 dispatches x about 12 round trips x unbounded fan-out per step,
   capped by `TOKEN_LIMIT`).
2. **Does 1b clear the YAGNI bar?** The argument for doing it now is that it is a refusal (about
   two checks and the rewritten tests) and prevents a silent failure. The argument against is
   that it rewrites five existing tests to guard a path nobody has taken. The smallest version is
   the `runnable` refusal alone, with the call-limit and step-limit read-back added in the same
   change as the first real subagent.
3. **Should a caller's `CompiledSubAgent` be allowed a step limit *above* `SUBAGENT_STEP_LIMIT`?**
   1b as written allows any explicit number. The alternative is capping it, which trades the
   "more control" reason for choosing `CompiledSubAgent` against a ceiling nobody can raise by
   accident.
4. **Should subagents carry the `task` dispatch limit at all?** `call_limits()` returns both
   limits. On a subagent with no `task` tool, `ToolCallLimitMiddleware[task]` is installed and
   never fires. 1a installs `call_limits()` whole, for symmetry with the parent, and 1b requires
   only the `TOOL_CALL_LIMIT` node. The alternative is to install only the all-tools limit on
   subagents. That is cleaner, but `call_limits()` would then need a second entry point.
