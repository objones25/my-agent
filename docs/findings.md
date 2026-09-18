# Findings

Library behaviour verified against the installed wheels, not taken from docs. Each entry records
what was checked, how, and what the code does about it.

Re-verify after any `uv sync` that moves these versions:

```
deepagents 0.7.15   langchain 1.4.1        langchain-core 1.6.3   langgraph 1.2.11
langchain-openai 1.6.2   langsmith 0.12.6   weave 0.53.9          openai 3.14.1
```

Every finding below is pinned by a test or a load-time check, so upstream drift fails loudly rather
than changing behaviour quietly. Verified 2026-09-17.

---

## F1 — `ChatOpenAI` picks the Responses API from the model name, ignoring `base_url`

**Severity: high.** Would break every request against a non-OpenAI endpoint.

`BaseChatOpenAI._use_responses_api(payload)` routes to `/v1/responses` instead of
`/v1/chat/completions` when `use_responses_api` is left at its default `None` and any of:

- `_model_prefers_responses_api(model_name)` — the id starts with a Responses-only prefix, or
  **contains the substring `codex` anywhere**;
- the instance sets `output_version="responses/v1"`, `context_management`, `include`, `reasoning`,
  `truncation`, or `use_previous_response_id`;
- the payload contains `context_management`, `include`, `previous_response_id`, `reasoning`,
  `text`, or `truncation`, or binds any builtin tool.

**None of this consults `base_url`.** The HF router serves Chat Completions only, so an inferred
switch fails at request time. langchain-openai's own `ChatOpenAI` docstring warns about this for
OpenAI-compatible providers.

*What we do:* `USE_RESPONSES_API = False` is a pinned constant, injected by `build_model` and
refused as a config field. A postcondition asserts it survived construction.

## F2 — `max_tokens` is renamed to `max_completion_tokens` on the wire

**Severity: low — the router accepts the renamed key.** Originally logged as a suspected
incompatibility; **resolved by live test**, see below.

HF documents `max_tokens`. `ChatOpenAI` renames it in **two** places —
`ChatOpenAI._default_params` and `ChatOpenAI._get_request_payload` ("max_tokens was deprecated in
favor of max_completion_tokens in September 2024 release") — and the field alias runs the same
direction, so the literal `max_tokens` key cannot be sent through the normal field.

Note the trap in verifying this: reading `BaseChatOpenAI._default_params` alone says it sends
`max_tokens`. The rename lives in the **subclass override**. Confirmed empirically by printing
`ChatOpenAI(...)._default_params`.

Every other parameter matches: `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`,
`frequency_penalty`, `logprobs`, `tools`, `tool_choice`, `response_format`, `stream`.

**Live result (2026-09-17, `openai/gpt-oss-120b` via the router):** the router *does* honour
`max_completion_tokens`. Asking for 200 numbers produced 483 output tokens uncapped, and exactly
24 with `max_tokens=24` bound. `extra_body={"max_tokens": 24}` produced an identical 24. So both
forms work and no workaround is needed for this model.

Caveat: verified for one model on whichever provider the router selected. A provider-pinned id
(`:groq`, `:novita`, …) is a different code path and is not covered.

*What we do:* `ModelConfig` still has no token-cap field (YAGNI). When one is added, the plain
`max_tokens` field is fine. `check_token_cap_reaches_the_router` in `main.py` keeps this honest.

## F3 — deepagents ships no `write_todos`; the docs and skill say otherwise

**Severity: medium.** Causes wasted debugging and wrong prompts.

The `langchain-skills:deep-agents-core` skill lists "TodoListMiddleware — default enabled" and
`write_todos` as a built-in. Neither is true at 0.7.15. There is no `TodoListMiddleware` in
`deepagents.middleware` at all; the class lives in `langchain.agents.middleware` and reaches the
agent through `create_deep_agent(middleware=[...])`, which then adds `write_todos`.

Found because the live agent said so, and the agent was right.

*What we do:* `tests/test_main.py` pins deepagents' raw default tool set and asserts `write_todos`
is absent, with a test recording how to restore planning if it is ever wanted.

## F4 — shell execution is on by default

**Severity: high.** A capability nobody chose.

`create_deep_agent`'s default tools, read off the compiled graph:

```
delete  edit_file  execute  glob  grep  ls  read_file  task  write_file
```

`execute` "executes a shell command in an isolated sandbox". It is enabled with no opt-in.
deepagents classes it as a *filesystem* tool (`FsToolName` includes it), and `FilesystemPermission`
cannot restrict it — `FilesystemOperation` is only `Literal["read", "write"]`.

The only supported lever is `FilesystemMiddleware(tools=[...])` with a narrower allowlist. Note
`read_file` is mandatory: omitting it raises `ValueError`.

*What we do:* `build_agent` installs a `FilesystemMiddleware` carrying `DEFAULT_FILESYSTEM_TOOLS`
— every filesystem tool except `execute`. A load-time check asserts that allowlist equals
`get_args(FsToolName) - {"execute"}`, so a new upstream tool fails the import rather than being
granted silently. Postconditions assert `execute` is absent from both the middleware's tools and
the compiled graph.

## F5 — replacing `FilesystemMiddleware` silently drops every permission rule

**Severity: high.** Silent security failure; follows directly from F4's fix.

deepagents merges middleware by `.name` (`_apply_custom_middleware`): a caller-supplied
`FilesystemMiddleware` **replaces** the default in place. The default is built as
`FilesystemMiddleware(..., _permissions=permissions)`, so `create_deep_agent(permissions=...)`
reaches the tool layer *only* through that instance. Replace it without forwarding and
`_permissions` becomes `[]` — rules gone, no error.

(`_build_interrupt_on_from_permissions` is a separate path and does still fire, which makes the
partial failure harder to spot.)

`_permissions` is private API and stores `list(_permissions or [])`, so `None` normalises to `[]`.

**Live result:** with forwarding in place, `write_file` to a denied path returns
`Error: permission denied for write on /secrets/keys.txt`. With forwarding removed (patched out in
a throwaway script), the same request **succeeds** — the file is written. The bug is real and the
check catches it.

*What we do:* `_least_privilege_filesystem(permissions)` forwards them, with a postcondition
pinning that they landed. A load-time check asserts `tools` and `_permissions` are still accepted
parameters. `build_agent` **refuses** the combination of a caller-supplied `FilesystemMiddleware`
and non-empty `AgentConfig.permissions` rather than half-applying it.

## F6 — `BackendProtocol` is an `abc.ABC`, not a `typing.Protocol`

**Severity: low.** Misleads anyone writing a custom backend.

Despite the name, its MRO is `(BackendProtocol, ABC, object)`. Structural typing will not register;
subclass it explicitly. It requires 9 sync + 9 async methods.

## F7 — `api.endpoints.huggingface.cloud` is not an inference URL

**Severity: low.**

It is the Inference Endpoints **control plane** for creating and managing dedicated deployments. A
dedicated endpoint serves inference at its own
`https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1/`. Both that and
`router.huggingface.co/v1` are OpenAI-compatible, so switching is a `ModelConfig.base_url` change
and nothing else.

## F8 — `load_dotenv()` searches from the calling file, not the cwd

**Severity: low.** Invalidates the obvious way of testing the missing-config path.

Running from another directory with `HF_TOKEN` unset still succeeded, because `find_dotenv()` walks
up from `main.py`'s own directory and found the project `.env`. Convenient at runtime, but it means
"run it from elsewhere" does **not** test the unconfigured path.

This recurs: a verification script written into a scratch directory also failed to find the
project `.env`, for the same reason in reverse. Pass an explicit path from anywhere but the package.

*What we do:* `tests/test_main.py` patches `load_dotenv` to a no-op and clears the env var.

## F9 — `.text()` on messages is deprecated in favour of the property

**Severity: low.** Would fail the suite under `filterwarnings = ["error"]`.

`BaseMessage.text` is a property; calling it emits `LangChainDeprecationWarning`. Use `message.text`.

## F10 — mypy cannot narrow types through `require()`

**Severity: low.** Design constraint, not a bug.

`require(x is not None, ...)` does not narrow `x`. Where a check also narrows, use an explicit
`if ... raise CheckFailed(...)` — same runtime behaviour, survives `python -O`, and mypy follows it.
`check_shape` in `negative_space.py` and `compiled_tool_names` in `capabilities.py` are the worked
examples.

## F11 — LangSmith and Weave coexist over the same deepagents run

**Severity: informational.** Resolves the "not verified end to end" caveat CLAUDE.md carried
since Task 1.

Checked by `tests/test_tracing.py::test_langsmith_and_weave_trace_the_same_run` (`-m live`): both
backends' `.from_env()` configure from the real environment, both `.activate()` without raising,
and `langchain_tracer_names()` reports `{'LangChainTracer', 'WeaveTracer'}` both *before* and
*after* a real `agent.invoke()` turn against the router — neither backend's global install
displaced the other's. Confirmed passing on multiple separate live runs, e.g.
`1 passed, 33 deselected in 1.69s` and `1 passed, 128 deselected in 2.01s` when run as part of the
full `-m live` invocation.

*What the code does:* `available_backends()` activates every configured backend rather than
selecting one (`main._activate_tracing()` loops over all of them), so this policy is safe to
keep — LangSmith's ambient env-var tracing and Weave's global LangChain callback hook do not
conflict.

*Wrinkles found on the way there, not part of the coexistence question itself, all in weave's own
SDK:*

1. `weave.init()` unconditionally calls `ensure_project_exists`, which calls into `gql` using the
   library's old `execute(..., variable_values=..., operation_name=...)` calling convention on
   every invocation. That convention itself emits a `DeprecationWarning` from inside `gql`, and
   under this project's `filterwarnings = ["error"]` it aborted the call before weave ever reported
   whether project access succeeded — masquerading in the traceback as
   `weave.wandb_interface.project_creator`'s "Unable to access `<entity>/<project>`" error path,
   which logs and re-raises whatever exception the call raised (confirmed by reading
   `weave/wandb_interface/project_creator.py`; it fires on every `weave.init()` call, live or not).
2. `weave.init()`'s async HTTP client leaves at least one `ssl.SSLSocket` for GC to close, which
   pytest's unraisable-exception hook promotes to a `PytestUnraisableExceptionWarning` — also fatal
   under `filterwarnings = ["error"]`. When this fired *during* the test's own call phase it made
   the live test itself flaky before it was understood: one early run failed on the `gql` warning
   (1), a run after fixing that passed cleanly, a further repeat run failed on the socket warning
   with the real assertions never having executed, and a subsequent run — with both warnings
   scoped out on the test — passed.

(1) is fixed with a narrowly-targeted `@pytest.mark.filterwarnings(...)` mark on the test itself —
per CLAUDE.md's rule for a new deprecation warning from a fast-moving dependency ("fix it or scope
an ignore, do not widen the setting"). It occurs synchronously, inside the test's own call stack,
so a per-test mark can reach it.

(2) turned out **not** to be fixable per-test, and the investigation into why is itself a finding.
The warning that actually gets raised is not a plain `ResourceWarning` — the socket's
`ResourceWarning` is only the informational `__cause__` on a `pytest.PytestUnraisableExceptionWarning`
that pytest's own `_pytest.unraisableexception.collect_unraisable` constructs and delivers via
`warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))`. That call site collects unraisable
exceptions accumulated via `sys.unraisablehook` and re-emits them **either** from inside
`pytest_runtest_call` (during a specific test) **or** from `pytest_unconfigure` (at the *session's*
teardown, after every test has already reported its result) — whichever GC pass happens to collect
the leaked socket first. An earlier `@pytest.mark.filterwarnings("ignore:unclosed:ResourceWarning")`
mark on the test matched the wrong category entirely (`ResourceWarning`, not
`PytestUnraisableExceptionWarning`) and never actually caught anything; it appeared to fix a flaky
run purely by GC-timing luck, which was only caught by later reproducing the crash with that mark
still in place, running the full `uv run pytest -m live`: the single live test printed `1 passed,
128 deselected`, and the *process* then crashed at `pytest_unconfigure`'s `gc_collect_harder`. No
per-test mark can reach a warning raised at session teardown — that plugin hook runs outside any
single test's marker scope.

Explicitly closing the client (`WeaveClient.finish(use_progress_bar=False)`) in a `finally` block
at the end of the test was tried as a pre-emptive fix and reverted: a follow-up live run with that
change **hung** for >90s with no verdict at all. `finish()` blocks until its send queue drains, and
a queue stuck behind an earlier run's failed writes (`weave.trace_server_bindings` 404s: `"Cannot
end call ...: no start found in project ..."`, observed during this same investigation) apparently
never drains — a silent hang is a strictly worse failure mode than the occasional loud crash it was
meant to fix.

**Resolved** with a project-wide, precisely-scoped `pyproject.toml` `filterwarnings` entry, added
*alongside* `"error"` rather than replacing it:

```toml
'ignore:Exception ignored in.*SSLSocket:pytest.PytestUnraisableExceptionWarning:_pytest\.unraisableexception'
```

Scoped on all three axes `warnings.filterwarnings` supports — message (`"Exception ignored
in.*SSLSocket"`, matching the exact text `_pytest.unraisableexception.unraisable_hook` builds for
an unraisable-exception summary, restricted further to ones naming an `SSLSocket`), category
(`pytest.PytestUnraisableExceptionWarning`, confirmed via `inspect` to be the class actually
raised, not `ResourceWarning`), and originating module (`_pytest.unraisableexception`, confirmed to
be that module's own `__name__`) — so it cannot mask an unrelated
`PytestUnraisableExceptionWarning` (a different unraisable exception, from different code) or any
plain `ResourceWarning` raised anywhere else, at collection or at session teardown alike. Verified
two ways: `uv run pytest -m live` now completes and **exits 0** (`1 passed, 128 deselected in
1.70s`, confirmed via explicit `echo $?`), and a mutation check — temporarily making `require()`
emit a plain `DeprecationWarning` — still makes the *offline* suite fail loudly (`exit 2`, "5 errors
during collection"), proving the entry narrows rather than widens the project's warnings policy.
The now-redundant `@pytest.mark.filterwarnings("ignore:unclosed:ResourceWarning")` mark (the one
that matched the wrong category) was removed from the test; the `gql`-warning mark for (1) was kept,
since it covers a genuinely different warning that the project-wide entry does not touch.

## F12 — LangChain swallows exceptions raised inside a callback handler

**Severity: medium.** A broken handler otherwise fails in silence.

`BaseCallbackHandler.raise_error` and `BaseCallbackHandler.run_inline` both default to `False`
(confirmed via `inspect` against langchain-core 1.6.3). By default, an exception raised inside a
hook is caught by `CallbackManager`'s own dispatch machinery and (depending on `run_inline`) may
also run off the main thread — so a broken handler stops mirroring, or reorders its own output,
without the run itself failing. Seen live, unprompted: the F11 live run logged `WARNING
langchain_core.callbacks.manager:manager.py:341 Error in WeaveTracer.on_llm_end callback:
PydanticDeprecatedSince20(...)` — Weave's own callback raised, LangChain caught it, and the agent
run carried on with that one hook silently degraded for the rest of the run.

*What the code does:* `JsonlMirror` pins `raise_error = True` (a broken mirror must be loud) and
`run_inline = True` (so recorded order is call order), and keeps its own body incapable of raising
on data — `_clip` round-trips every value through `json.dumps(..., default=str, allow_nan=False)`,
catching `TypeError`/`ValueError` and degrading to `repr(value)` when that round trip itself cannot
succeed (a non-str-keyed dict, a circular reference, a non-finite float — see F15). `main()` asserts
`mirror.records > 0` after every run, which is what actually catches "the callbacks were never
attached" — `raise_error=True` alone only catches a handler that ran and threw.

## F13 — `LANGSMITH_TRACING` must be exactly `"true"`; the check is cached and still honours `LANGCHAIN_*`

**Severity: high.** Live in this repo's own `.env` right now (`LANGSMITH_TRACING=True`, capital T).

`langsmith.utils.tracing_is_enabled()` computes
`get_env_var("TRACING_V2", default=get_env_var("TRACING", default="")) == "true"` — a literal
string comparison, no `.strip()`, no `.lower()`, no truthy-word mapping (`True`, `1`, `yes`, `on`
all leave tracing off). Confirmed by reading `tracing_is_enabled`'s source off the installed wheel.

`get_env_var` is `@functools.lru_cache(maxsize=100)`-wrapped and, per its actual default
`namespaces=("LANGSMITH", "LANGCHAIN")`, searches both prefixes — so the legacy
`LANGCHAIN_TRACING`/`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT` names still enable tracing and still
resolve the project. Reproduced live: with only `LANGCHAIN_*` set and no `LANGSMITH_*` variable at
all, `tracing_is_enabled()` → `True`, `get_tracer_project()` → the value from `LANGCHAIN_PROJECT`,
and `CallbackManager.configure().handlers` includes a `LangChainTracer`. This directly disproves
the older claim in CLAUDE.md that the legacy names "no longer work" — re-run 2026-09-17 as part of
this task, same result as originally observed.

The `lru_cache` means a lookup made before `load_dotenv()` runs — anything importing `langsmith`
early, or a prior call in the same process — is remembered for the rest of the process; a stale
`""` default caches as "tracing off" no matter what `.env` says afterward.

*What the code does:* `LangSmithTracing.activate()` calls `get_env_var.cache_clear()` before
checking `tracing_is_enabled()`, and raises `TracingMisconfigured` naming the exact offending
value when langsmith disagrees with what `from_env`'s more permissive `_TRUTHY` set accepted.
Tests that exercise this path clear the cache too (the autouse `_clear_langsmith_cache` fixture),
and the F11 live test explicitly forces `LANGSMITH_TRACING=true` after `load_dotenv()` for exactly
this reason — the repo's own `.env` would otherwise fail it.

## F14 — chat models emit `on_chat_model_start`, never `on_llm_start`

**Severity: low.** Would silently produce an incomplete mirror if assumed otherwise.

A live run's JSONL (`logs/20260917T213525Z-ace27da3.jsonl`, 8 records) contains `chain_start`×3,
`chain_end`×3, `chat_model_start`×1, `llm_end`×1 — no `llm_start` event at all. Confirmed by direct
inspection: `'llm_start' in events` is `False`. `ChatOpenAI`, called through deepagents' chat-model
path, fires `BaseCallbackHandler.on_chat_model_start`, not the legacy plain-LLM `on_llm_start`.

*What the code does:* `JsonlMirror` implements `on_chat_model_start` and deliberately has no
`on_llm_start` handler — this measurement is what justifies the omission rather than it being an
oversight.

## F15 — `json.dumps(default=...)` applies only to values, never to dict keys, and does not cover cycles or non-finite floats

**Severity: important.** With `raise_error = True` pinned on `JsonlMirror`, this let a logging
callback kill an agent run — the exact failure F12's fix was supposed to make impossible.

`_clip`'s original implementation was `json.loads(json.dumps(value, default=str))`, and both this
file's own comment at the time and F12 above asserted that the round trip made the body incapable
of raising on data — "nothing unserializable (a `UUID`, a `BaseMessage`, an arbitrary tool return)
can throw." **That claim was false**, and it was false in the approved spec before it was false in
the implementation: the spec asserted the `default=str` round trip as sufficient, and the code
inherited the claim along with the round trip, unverified against `json.dumps`'s actual contract.

`default=` is a callback `json.dumps` invokes only for values it cannot otherwise serialize — never
for dict keys, which must already be `str`, `int`, `float`, `bool`, or `None`. Reproduced directly
against the installed CPython 3.13 stdlib:

```python
>>> json.dumps({(1, 2): "x"}, default=str)
TypeError: keys must be str, int, float, bool or None, not tuple
```

`on_tool_end(output: Any)` takes an arbitrary tool return, and `on_chain_start`/`on_chain_end` take
arbitrary graph state — a `dict` with a non-`str` key (a tuple, a frozenset, an enum member without
`str` mixed in) is squarely inside what those hooks are documented to receive, not a contrived edge
case.

Two further gaps in the same round trip, found while fixing the first:

- `json.dumps` does not detect circular references through `default=` either — a self-referential
  structure (`d["self"] = d`, which a naive `dict(state)` copy of an agent's own working state can
  produce) raises `ValueError: Circular reference detected`, not a call to `default`.
- `json.dumps`'s default `allow_nan=True` serializes a non-finite float as a bare `NaN`/`Infinity`
  token. Python's own reader accepts that token back, so the original round trip did not fail on
  it — but it is not valid JSON by the interchange-format spec, and `jq`, Go's `encoding/json`, and
  Rust's `serde_json` all reject it. This file's contract is one valid JSON object per line, so a
  silently-accepted bare `NaN` was a correctness bug even though `_clip` itself never raised on it.

*What the code does:* `_clip` wraps the round trip in `try`/`except (TypeError, ValueError)` and
degrades to `repr(value)` on failure, and passes `allow_nan=False` so a non-finite float now raises
`ValueError` into the same `except` instead of round-tripping silently. `raise_error = True` is now
actually safe against data, rather than assumed safe against data. Covering tests exercise all
three shapes: a tuple-keyed dict, a self-referential dict, and a payload containing `float("nan")`
(asserting no bare `NaN` token appears in the written line). F12 above is corrected to describe
this fixed behaviour rather than the original, false claim.

---

## Live verification

`uv run my-agent` runs one check per finding against the real router and prints PASS/FAIL. As of
2026-09-17, 5/5 pass:

| Finding | Check | Evidence |
|---|---|---|
| F1 | chat-completions endpoint reachable | `reply='pong'` |
| F2 | token cap honoured | asked ≤24, produced 24 |
| F4 | shell tool withheld | unbound; no `execute` call in a run that asked for one |
| F4 | remaining filesystem tools usable | `write_file`, `read_file` both succeeded |
| F5 | permission rules survive replacement | `write_file` denied on `/secrets/**` |

The checks assert on tool messages and token counts rather than model prose, so they do not depend
on how the model phrases things. The F5 check was itself verified by breaking the fix and watching
it go red.

## Open / unverified

- Whether a **provider-pinned** model id (`org/model:groq`) behaves identically on the token cap
  (F2). Only the router's default selection is covered.
