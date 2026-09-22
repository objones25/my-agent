# Harness evaluation — 2026-09-21

An outside review of the harness at commit `9b4b9fc`, run against the working tree rather than
against `CLAUDE.md`. Every claim below was reproduced locally; the one item taken from a delegated
inventory is marked as such. Mutations used to prove a test blind spot were reverted and the tree
re-verified clean before this was written.

## Verdict

The harness is at the ceiling `CLAUDE.md` names for itself. Do not add more bounds — the next bound
would be the seventh, and no gap was found that one would close. The two real defects are both in
the parts that check the other parts: **the test suite cannot see a change to any default
constant**, and **two verification artifacts the harness already produces every turn are thrown
away**.

## What holds

`./scripts/check.sh` is green end to end: ruff, mypy strict (22 files), pyright, **376 passed /
2 deselected in 2.31s**, 90% coverage, negative-space gate 0 findings (42 advisory NSP001). No drift
between CI, the pre-commit hook and the script.

Every claim in *Verified API facts* matches the installed wheels exactly — versions,
`create_deep_agent`'s 2 positional + 16 keyword-only parameters, `FsToolName`'s 8 members, both
plugin entry-point groups empty, `ensure_config()["recursion_limit"] == 25`. A bare
`create_deep_agent` on a fake model binds `execute` and does not bind `write_todos`, so F3 and F4
are live facts rather than history. The allowlist holds on the parent and on `general-purpose`; the
subagent carries `recursion_limit: 25`; `step_limit=6` trips after 2 model calls.

That is unusually well-kept for a repo this dense in claims.

## 1. The defaults table is unpinned — no test can see it move

Five constants changed at once, then the suite run:

```
HF_ROUTER_BASE_URL  → "https://api.openai.com/v1"
DEFAULT_TEMPERATURE 0.0 → 1.9      DEFAULT_MAX_RETRIES 2 → 99
RUN_DEADLINE_S 600.0 → 6000.0      TOKEN_LIMIT 500_000 → 5_000_000
→ 376 passed, 2 deselected in 1.70s
```

The harness silently repointed at `api.openai.com`, at temperature 1.9, with a 10x wall clock and a
10x token ceiling, and nothing went red. Cause: every test reads the value back and compares it to
the same symbol it came from. `tests/test_model.py::test_build_model_ignores_ambient_openai_env_vars`
is the sharpest case — it exists specifically to prove an ambient `OPENAI_BASE_URL` cannot redirect
the harness, and it asserts against `HF_ROUTER_BASE_URL`, so when the constant *is* that URL the
test still passes. `RECURSION_LIMIT` is the one constant that is pinned, incidentally, by a
`match="25 steps"` literal in `test_run.py`.

This is F21's own rule — *a pinned value equal to the library default cannot be tested by reading it
back* — except broader, because these are not library defaults. They are the repo's own decisions.

## 2. `exit_behavior="continue"` is unpinned for the same reason

Deleted from both `ToolCallLimitMiddleware` calls in `call_limits()` → **376 passed**.
`ToolCallLimitMiddleware.__init__` defaults `exit_behavior` to `'continue'`, so
`test_call_limits_block_rather_than_abort` reads back a value identical whether it was chosen or
inherited. `test_call_limits_are_per_run_not_per_thread` is worse: it asserts `thread_limit is None`
about a parameter `call_limits()` never passes. The correct pattern is written 80 lines away in the
same file (`test_the_backend_and_context_bounds_are_stated_rather_than_inherited`, which records the
constructor call).

## 3. The parent graph also carries `recursion_limit: 9999`

F24 documents this for subagents. It is true of the parent too, and nothing records it:

```
bound_step_limit(build_agent(m))           → 9999
agent.invoke({...})  # no run_turn         → GraphRecursionError after 3,325 model calls
```

`run.py`'s module docstring says a call-site bound means other callers "inherited langchain-core's
defaults by accident" — that is 25. The actual inheritance is 9999, ~400x worse. Every path the repo
uses goes through `run_turn`, so this is not a live bug; it is the repo's own "a bound belongs to
the thing it bounds" unapplied to the one graph where it is not.

Fix verified: `.with_config({"recursion_limit": RECURSION_LIMIT})` on the returned parent drops a
bare invoke to 6 model calls, and `run_turn`'s per-invocation `step_limit` still wins (tested at 8
→ 2 calls). `with_config` returns `Self`, so the annotated return type survives, and
`compiled_tools`, `subagent_graphs` and the subagent's own bound limit all still read back off the
copy.

## 4. `run_turn` discards the only verification artifact it has

The graph returns `{'files': {...}, 'messages': [...]}`. `_invoke` takes `result["messages"]` and
drops the rest; `TurnResult` has no field for it; `mirror._state_summary` records key names and
message counts only. So this comes back on every turn and is visible nowhere:

```
files = {'/f1-0.txt': {'content': 'x', 'created_at': ..., 'modified_at': ...}, ...}
```

*What is deliberately not verified* argues the verification ladder is blocked on a domain. Rung 1 is
"read-back of the object the agent claims to have created" — that object is already in hand,
domain-free, at zero cost. `failed_tool_calls` and `answered` were the right instinct; this is the
third one, and cheaper than either.

## 5. A middleware door the plugin pin cannot see

`deepagents/graph.py` calls `append_prompt_caching_middleware()` on the parent, on every subagent
spec, and on `general-purpose`. It always appends `AnthropicPromptCachingMiddleware`, and appends
Bedrock and Fireworks caching middleware **if `langchain_aws` / `langchain_fireworks` are
importable** — an `import_module` probe, not an entry point, so `DEEPAGENTS_PLUGIN_GROUPS` cannot
see it.

Blast radius today is nil: `unsupported_model_behavior="ignore"`, and `_should_apply_caching`
returns `False` for a non-`ChatAnthropic` model. Note that `"ignore"` is load-bearing — `"warn"`
plus `filterwarnings = ["error"]` would fail every model call. It appears nowhere in `findings.md`'s
1,651 lines. The pin is a few lines in the same idiom as the entry-point one.

## 6. Check coverage, and one decorative test

*(Inventory delegated; the two specific claims below were reproduced locally.)* 117 check sites —
104 `require()` plus 13 `raise CheckFailed`; `unreachable()` and `bounded()` have zero call sites in
`src/`. **56 tripped by a test (48%)**, 61 not, 14 of those import-time. `CLAUDE.md`'s
self-assessment is accurate: exactly 7 untripped in `run.py`. The untripped set is almost entirely
the read-back *postconditions* the code's own comments call load-bearing — the monkeypatch technique
to force them exists and is used twice elsewhere.

One test is decorative. `test_build_agent_fails_when_the_subagent_reader_finds_nothing` patches
`subagent_graphs → {}` and expects `match="subagent"`. Traced: it raises at `agent.py:375`
(`_bounded_general_purpose_subagent`), not the vacuity guard in `_require_shell_withheld` that its
docstring describes. Both messages contain "subagent".

## 7. Documentation drift

`README.md` says "twenty-three verified findings" and "F1-F23" (twice), `live.yml` says
"twenty-three entries", `findings.md` says "twenty-three verified behaviours". There are **38**.
`CLAUDE.md` is correct. README also promises "One placeholder tool is enough to exercise it" — there
is no such tool; `AgentConfig.tools` defaults to `()` and nothing passes one.

## On evals

`evals/` is empty and `EvalRunner` is a table row. But the eval suite already exists — it is the
eight live checks in `main.py`: real tasks, code-based graders, PASS/FAIL, on a weekly cron. The gap
is not a framework, it is that they run at **n=1**, and three of them (`shell_tool_withheld`,
`filesystem_tools_still_work`, `permissions_are_enforced`) depend on the model choosing to call a
tool. A single non-deterministic failure reads as a regression with no error bars. Repeats and a
pass^k line are cheap; a new eval harness is not, and does not pay before a domain.

`main.py` is deliberately left out of the ranking — its 53% coverage and 11/11 untripped checks
would otherwise look like the largest gap, and it is scaffolding slated for replacement.

## Recommended order

1. Pin the constants — one literal per value, the `match="25 steps"` trick already in the repo.
2. Record the `exit_behavior` call.
3. Bind the parent's step limit.
4. Surface `files` on `TurnResult`.

Roughly a day, and it closes everything found that could bite. Then stop hardening. Findings 5–7 are
worth a findings entry and a README pass, not a fix round.
