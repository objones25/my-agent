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

F20–F22 were found by asking a narrower question than "does the allowlist hold?": *where else does
something run, and what did we not choose?* Both answers were load-bearing.

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

## F16 — mypy and pyright disagree about what satisfies a protocol variable

**Severity: moderate.** The repo's CI type checker is mypy; the editor most likely to open this
code runs pyright (Pylance). A spelling that passes one and fails the other means a clean CI run
and a file full of red squiggles, which is how a protocol ends up quietly re-shaped by whoever's
IDE complained loudest.

`TracingBackend` originally declared `name` as a read-only `@property` and both adapters satisfied
it with a `ClassVar[str]`. mypy accepts that. pyright rejects it:

    "LangSmithTracing" is incompatible with protocol "TracingBackend"
      "name" is not defined as a ClassVar in protocol

*How it was checked.* All four combinations, against mypy 2.3.1 (`--strict`) and pyright (latest,
via `npx pyright`), on 2026-09-17:

| Protocol declares | Implementation uses | mypy | pyright |
|---|---|---|---|
| read-only `@property` | `ClassVar[str]` | pass | **fail** |
| `name: ClassVar[str]` | `ClassVar[str]` | pass | pass |
| read-only `@property` | instance attribute | pass | pass |
| `name: ClassVar[str]` | instance attribute | **fail** | **fail** |

Only two spellings satisfy both checkers, and they are not interchangeable: the `ClassVar` form
requires implementations to use a `ClassVar`, and the `@property` form requires them not to.

*What the code does:* `TracingBackend` declares `name: ClassVar[str]`, and both adapters keep their
`ClassVar`. That form was chosen over the read-only-property one because a `ClassVar` stays out of
a dataclass's generated `__init__`, so no caller can construct a backend with a name of their
choosing — `name` identifies the backend rather than describing an instance. The protocol's
docstring records the constraint so the next implementation does not reach for an instance
attribute and fail pyright in the other direction.

pyright is now a dev dependency and configured in `pyproject.toml`, so `uv run pyright` is part of
the gate rather than something the editor notices later. It runs in `standard` mode, pinned
explicitly because pyright's default has moved between releases. Not `strict`: that reports 46
issues, and the two largest groups argue against this codebase's design rather than finding bugs in
it — `reportUnnecessaryIsInstance` flags the defensive `require(isinstance(...))` checks that
negative-space programming exists to add, and `reportPrivateUsage` flags the deliberate
`_permissions` access in `agent.py`. The remainder are `reportUnknown*` from untyped
langgraph/deepagents surfaces.

*Still unverified:* whether other protocol members in this repo have the same divergence. Nothing
else here declares a protocol variable, so there is nothing else to check yet — but any new
protocol with a non-method member should be run past both checkers before it is relied on.

---

## F17 — reasoning is most of the output, and the only dial is `reasoning_effort`

**Severity: important.** It is a cost lever that was running at the provider's default, and it
silently changes what a token cap means.

`openai/gpt-oss-120b` reasons on essentially every call. From a real `uv run my-agent` run
(`logs/20260918T021743Z-b8cf1012.jsonl`), reasoning tokens as a share of output:

| check | reasoning | output |
|---|---|---|
| F1 "pong" | 36 | 47 |
| F2 token cap | **21** | **24** |
| F4 shell refusal | 74 | 146 |

The F2 row is the one that matters: with `TOKEN_CAP = 24`, reasoning consumed 21 of the 24 tokens
and the visible answer got 3, which is why that record's text is empty. **A token cap on a
reasoning model is mostly a reasoning cap.** The check still verifies what it claims — the cap is
honoured — but it is not evidence that 24 tokens buys 24 tokens of answer.

*How it was checked.* `reasoning_effort` is documented by the router as an optional Chat Completions
body parameter (`none, minimal, low, medium, high, xhigh`). Verified on the wire 2026-09-17, one
prompt, three settings:

```
effort=None   reasoning=50   output=61    text='9'
effort=low    reasoning= 6   output=17    text='9'
effort=high   reasoning=93   output=104   text='9'
```

Identical answers, a ~15x spread in reasoning tokens. Unset is *not* "off" — it is the provider's
default, which sat between `low` and `high`.

**Use `reasoning_effort` (str), not `reasoning` (dict).** Both are `ChatOpenAI` fields.
`reasoning` is the Responses API's parameter, and `BaseChatOpenAI._use_responses_api` returns
`True` whenever `self.reasoning is not None` — so on a default `use_responses_api=None` instance,
setting it would silently reroute every request to `/v1/responses`, which the router does not
serve (F1). Our pinned `USE_RESPONSES_API = False` short-circuits that check before it is reached,
so the pin protects us — but the parameter would then be sent in a Chat Completions body, where it
does not belong.

*What the code does:* `ModelConfig.reasoning_effort` (default `None`, i.e. unchanged behaviour),
validated against `REASONING_EFFORTS`. A bad value from a caller is a `CheckFailed`; a bad value
from `REASONING_EFFORT` in the environment is a `ValueError` at the edge. Verified end to end:
`REASONING_EFFORT=low uv run my-agent ...` put `reasoning_effort: "low"` in the request params and
produced 8/4/0 reasoning tokens where the same three-call shape had produced 16/11/11 at default.

*Still unverified:* whether every provider the router may select honours the parameter, and whether
`none`/`minimal` differ from `low` on this model.

---

## F18 — the mirror recorded the conversation three times and the request not at all

**Severity: important.** The log could not answer either of the two questions it exists for.

Measured on a real 5-check run (52 records, 36KB):

- **`llm_end` dropped the half of a response that matters.** No tool calls, so a tool-calling turn
  logged `outputs: [""]` and was indistinguishable from an empty reply — four of eight records
  looked empty. No `finish_reason`, so "finished" and "hit the cap" were the same record. No
  `model_name` or `model_provider`, though the router picks a provider per request.
- **`chat_model_start` recorded no request at all** — `{type, text}` per message and nothing else.
  No model id, temperature, token cap, tool list, or reasoning effort. Assistant messages lost
  their `tool_calls`, so the history was not replayable.
- **`chain_start`/`chain_end` were 77% of the bytes**, as Python `repr` strings: deepagents passes
  `Command` objects where the signature promises `dict`, and `default=str` stringified them. Not
  queryable by `jq`, not a stable format, and a duplicate of content already recorded elsewhere.
  The largest field was 3745 chars against `MAX_FIELD_CHARS = 4000` — **94% of the truncation
  bound on a smoke test**, so a slightly longer conversation would have truncated a repr blob into
  an unparseable fragment.

*How it was checked.* Every field of every record in a real run, plus a callback probe confirming
the missing data is all reachable: `on_chat_model_start` receives `invocation_params` in `kwargs`,
and `on_llm_end` has `generation.message.tool_calls`, `.response_metadata` and
`.additional_kwargs`. `ChatOpenAI._get_invocation_params()` was checked for credential leakage and
carries none — unlike `serialized`, which is why that one is still never written.

*What the code does:* `llm_end` records `tool_calls`, `metadata` (`finish_reason`, `model_name`,
`model_provider`, `system_fingerprint`, `service_tier`) and, when non-empty, `extra` from
`additional_kwargs` — which is where reasoning *content* would land if a provider ever returned
any. None does today; the token count in `usage.output_token_details.reasoning` is all we get.
`chat_model_start` records `params`, with tool definitions reduced to their names. `chain_*`
records a `_state_summary` — the keys a step carried and how the message list grew — because the
content is already recorded structurally by the model and tool records.

Verified on a live run afterwards: no repr blobs remain, the largest field fell from 3745 to 563
chars, and `chain_*` fell from 77% to 39% of a file less than a quarter the size per turn.

**Two follow-ups from reading the next run's log.**

*One repr blob survived the first pass.* `on_tool_end` receives `ToolMessage` objects, so
`default=str` wrote `content='...' name='write_file' tool_call_id='...'` — and buried `status`,
which is the authoritative success/error signal. A permission denial was findable only by
substring-matching the repr. `_tool_output_summary` now records `content`, `status`, `name` and
`tool_call_id` as fields; a live run shows the denial as `status: "error"` rather than prose inside
a string. `artifact` is reduced to a boolean flag on purpose — tools may attach arbitrary payloads
and a log is not the place to copy them. Plain (non-`ToolMessage`) tool returns pass through
untouched.

*`params` is the langchain-level request, not the HTTP body.* Callbacks only ever receive
`invocation_params`, which is read before `_get_request_payload` renames anything. So a bound token
cap appears in the log as `max_tokens: 24` while the wire carries `max_completion_tokens: 24`
(F2) — confirmed by comparing the two directly. Every other parameter agrees; this is the only
rename langchain performs, and there is no callback hook that sees the final payload. Recorded here
rather than worked around, because reaching the real body would mean monkeypatching a private
method.

Second live run, with both fixes in: 52 records, 19,458 bytes for the same five checks that
originally produced 36,214 — no repr blobs anywhere, `finish_reason` distinguishing `length` (the
capped check) from `tool_calls` and `stop`, and the withheld-`execute` allowlist now visible in the
request record rather than only asserted in tests.

---

## F19 — a hand-maintained mapping drifts; only a check that reads the dataclass stops it

**Severity: important.** `ModelConfig.from_env` read three of seven fields, and one of the four
omissions made this module's own docstring false.

`model.py` states its thesis at the top: adding a setting is "one new field with a default: no
factory signature change, no factory body change, no call site change." That holds for
`build_model` and `as_kwargs()`. It did **not** hold for `from_env`, which needed a constant, a
lookup, a parse, a validation and a new keyword per field — the growing argument list the
parameter-object design exists to eliminate, inside the file that argues against it. Adding
`reasoning_effort` (F17) touched four places in that one method.

The consequence was not merely inelegant. `HF_ROUTER_BASE_URL`'s docstring promises that pointing
at a dedicated Inference Endpoint needs "no code change", while `main.py` builds its config only
through `from_env()` — which never read `base_url`. **Using a dedicated endpoint required editing
source.** Nothing caught it because nothing compared the mapping to the dataclass.

*What the code does:* the mapping is now `_ENV_FIELDS`, a table of
`(field, env var, parser, required)` rows, and three load-time checks assert it against
`dataclasses.fields(ModelConfig)`: no field unreachable, no row naming a non-field, no two rows
sharing a variable. Removing the `base_url` row now fails the import with
`ModelConfig fields unreachable from the environment: ['base_url']`. A table rather than reflection
over annotations, deliberately: inferred variable names would be implicit and inferred parsers
would give generic errors.

The same reasoning produced a second pin, in `agent.py`. `check_config_contract` asserts our
*fields* are real `create_deep_agent` parameters, but it cannot notice a **new** parameter
appearing — and a new parameter is exactly how the shell `execute` tool arrived switched on with
no opt-in (F4). `KNOWN_CREATE_DEEP_AGENT_PARAMS` pins the set at 18, so a deepagents upgrade that
adds one fails the import by name and has to be reviewed for what it enables before being
accepted.

**Taxonomy note.** Everything `from_env` reads is outside input, so every failure is an operating
error: absent, blank, unparseable, *and* out-of-range. The last one used to leak — `MODEL_TEMPERATURE=5`
parses fine and then trips a `require()`, which would have crashed with an `AssertionError`
traceback implying a bug in this code. `from_env` now constructs inside a `try`, catches
`CheckFailed`, and re-raises it as a `ValueError` naming the variable at fault.

*Still unverified:* whether `AgentConfig` wants the same treatment. It deliberately covers 5 of 18
parameters (YAGNI), so completeness is the wrong property for it — but nothing records *which*
omissions were considered and rejected.

---

## F20 — a withheld tool comes back through `task`; the parent graph is not the whole allowlist

**Severity: critical.** `capabilities.py` withholds `execute` and `build_agent` asserted it was
absent. The assertion read the parent graph only, and the parent graph is not the only thing that
runs.

`create_deep_agent` auto-adds a general-purpose subagent, reachable through the `task` tool, and
builds it *its own* `FilesystemMiddleware` — with `backend`, `custom_tool_descriptions` and
`_permissions` forwarded, but **no `tools=` allowlist**, which means `"all"`. Read off the compiled
graphs:

```
create_deep_agent(model=m)   subagent → delete edit_file execute glob grep ls read_file write_file
build_agent(m)               subagent → delete edit_file         glob grep ls read_file write_file
```

So ours is narrowed — but only because deepagents merges middleware by `.name` into the subagent's
list as well as the parent's. That merge is behaviour we do not control and nothing depended on it
deliberately. An upstream change to it would re-grant shell execution through `task` while every
test stayed green, because no test and no postcondition ever looked at a subagent.

*How it was checked:* `agent.get_subgraphs(recurse=True)` returns `[]` for a deep agent and
`get_graph(xray=1)` shows only `model`, `tools` and one middleware node — the subagent graphs are
not reachable through any public API. They live in the `task` tool's closure, under the freevar
`subagent_graphs`.

*What the code does:* `capabilities.subagent_graphs(agent)` reads that closure, and
`_require_shell_withheld` now asserts absence across the parent graph **and** every subagent.
Reaching into a closure is worse than reaching into `nodes["tools"]`, so it is priced the same way
`compiled_tool_names` already was — every way the structure can move raises `CheckFailed` rather
than returning an empty mapping that would make the check vacuous: no closure, neither `func` nor
`coroutine`, a non-mapping value, and an empty mapping are four separate named failures. A `task`
tool that is simply absent returns `{}`, because nothing can then be dispatched.

The test suite states both halves. `test_build_agent_withholds_the_shell_tool_from_every_subagent`
is the claim; `test_a_bare_deep_agent_does_grant_the_shell_tool_to_its_subagent` is what keeps it
honest — without it the first test would keep passing if deepagents stopped granting `execute` to
subagents for its own reasons, and the narrowing would have quietly become a no-op.

**Mitigating, and not a reason to relax.** `StateBackend` does not implement
`SandboxBackendProtocol`, and `create_deep_agent`'s own docstring says `execute` "will return an
error message" for non-sandbox backends. A leak would therefore have been survivable rather than
fatal *at today's backend* — which is exactly the argument least privilege exists to not depend on.

*Also collapsed here:* the same assertion existed in three phrasings, with the message
`"no tools bound at all; the absence check would be vacuous"` duplicated verbatim between
`agent.py` and `main.py`. It is now one `require_withheld(withheld, granted, where)` naming *which*
graph leaked, with `require_granted` as its positive twin (two more verbatim copies in `main.py`).
`main.check_shell_tool_withheld` also carried a dead branch: `build_agent` raises when the tool is
present, so the `CheckResult(..., False, f"bound: ...")` return was unreachable.

*Closed, offline.* Whether a deny rule is enforced *inside* a subagent was the open half, and it
turned out to be provable without a model. The subagent's filesystem tools are not equivalents of
the parent's, they are **the same objects**: `compiled_tools(subagent) is compiled_tools(parent)`
holds name by name, because the middleware deepagents replaced was ours and each tool closes over
that one instance's `self._permissions` and `self.backend` (read off
`FilesystemMiddleware._create_write_file_tool`). Identity is the mechanism; two tests are its
observable consequence, exercised through the subagent's own `write_file` with a hand-built
`ToolRuntime`:

- `/secrets/keys.txt` comes back `status="error"`, `"Error: permission denied for write on
  /secrets/keys.txt"`.
- `/notes/ok.txt` gets *past* the permission gate and fails inside `StateBackend`, which refuses to
  run outside a graph execution. That refusal is the discriminator: it is reachable only by a call
  the rule allowed, so the deny is targeted rather than blanket.

Both were checked against a rule that no longer covers the path and a rule flipped to `allow`; the
enforcement test fails under each, so it is sensitive to the rule's content and not merely to
"some write errored". A mutant that breaks the subagent *alone* does not exist in this code --
which is the finding, not a gap in the tests.

`compiled_tools` is public for this reason: names are enough for an absence check, identity is what
explains why the rules reach a second graph at all.

---

## F21 — `least_privilege_filesystem` inherited five settings from a library default

**Severity: important.** CLAUDE.md's least-privilege rule says a setting is "never something
inherited from a library default." The middleware that exists to enforce that rule was inheriting
five.

`FilesystemMiddleware.__init__` defaults, read off the installed wheel:

```
backend                                  = None   -> StateBackend()
tool_token_limit_before_evict            = 20000
human_message_token_limit_before_evict   = 50000
grep_max_count                           = 1000
max_execute_timeout                      = 3600
```

`least_privilege_filesystem` passed only `tools` and `_permissions`, so the other five came from
deepagents. Two of them *rewrite the conversation*: a tool result over 20,000 tokens, or a user
turn over 50,000, is evicted to the filesystem and replaced with a pointer — a context-engineering
decision this project never made, and one that appears in the JSONL mirror as a state write nobody
asked for.

**The backend is the more serious half, for a reason that is not about defaults.** `backend` was
`None`, so the replacement middleware built a `StateBackend` of its own while `create_deep_agent`
wires the *caller's* backend into `SkillsMiddleware` and the summarisation middleware. Since our
middleware displaces its filesystem middleware by name, a caller who supplied a backend would have
got one agent on two filesystems: skills and summarisation on theirs, every filesystem tool on a
`StateBackend`. This is F5's failure — a setting that reaches `create_deep_agent` but not its
replacement — wearing a different hat, and `agent.py`'s own module docstring advertised adding a
`backend` field as the usual one-line change while that was false.

*What the code does:* the three bounds worth owning are named constants
(`TOOL_RESULT_TOKEN_LIMIT`, `HUMAN_MESSAGE_TOKEN_LIMIT`, `GREP_MATCH_LIMIT`), passed explicitly,
asserted as postconditions on the built instance, and pinned at import against the wheel's current
defaults — so a deepagents change to any of them fails the load with both numbers in the message
instead of quietly resizing how much of a tool result the model sees. `max_execute_timeout` is
deliberately *not* pinned: it bounds `execute`, and `execute` is withheld, so pinning it would
assert something about a tool nobody has. `least_privilege_filesystem` gained a `backend`
parameter, `_agent_kwargs` forwards `kwargs.get("backend")` into it, and a postcondition refuses a
middleware whose backend differs from the one `create_deep_agent` will use. Adding the field really
is one line now.

**A decorative test, caught by mutation.** Every pinned value equals deepagents' current default,
so reading one back off the built middleware cannot distinguish "we chose it" from "we inherited
it": deleting all three arguments left the instance-reading test passing. The discriminating test
records the *constructor call* instead, with the real `FilesystemMiddleware` still building the
object. Both tests are kept, because they make different claims — one that the bound was sent, one
that it landed, and `_permissions` was sent and silently dropped once already (F5).

---

## F22 — mypy and pyright disagree about subscripting a non-required `TypedDict` key

**Severity: minor.** A second instance of the F16 pattern, in a different place.

Every key of `langchain_core.runnables.RunnableConfig` is non-required (`total=False`). mypy
accepts `config["recursion_limit"]`; pyright in `standard` mode reports
`reportTypedDictNotRequiredAccess` — "access may result in runtime exception". pyright is right:
the key genuinely may be absent. `.get(...)` satisfies both.

Surfaced while typing `run.py`'s `Invokable` protocol, where the two checkers *agreed* about
something more interesting: a protocol parameter is contravariant, so declaring
`invoke(self, input: dict[str, Any], ...)` makes a real `CompiledStateGraph` **fail** to satisfy
the protocol, because its own `input` is typed `InputT | Command | None`. Both checkers rejected
it with the same diagnosis. `input` is therefore `Any` and `config` is typed exactly — which is
what earns the check: `run_turn` builds a real `RunnableConfig`, so a misspelled bound is a type
error rather than a silently ignored key at run time.

---

## F23 — weave's exit waits 300s for call starts that were dropped, and nothing configures it

**Severity: important.** A trace server that drops a write does not degrade telemetry; it blocks
process exit for five minutes. Two wrong theories preceded the right one, and both are recorded
because each looked convincing.

*Observed:* a `pytest -m live` run whose tests took **3.11s** while the process took **5m06s** to
exit, and an earlier one killed by `timeout` at 480s. Logs fill with
`httpx.HTTPStatusError: Client error '404 Not Found' for url '.../call/end'` and `Cannot end call
<id>: no start found in project`.

*Root cause, from a `faulthandler.dump_traceback_later` stack dump of every thread.* The main
thread sits in `call_batch_processor.py:188`, inside
`CallBatchProcessor.stop_accepting_new_work_and_flush_queue`, which weave registers with `atexit`:

```python
deadline = time.monotonic() + FLUSH_TIMEOUT_SECONDS      # = 5 * 60
while time.monotonic() < deadline:
    pending_count = len(self._pending_starts) + len(self._pending_ends)
    if pending_count == 0:
        break
    time.sleep(FLUSH_POLL_INTERVAL_SECONDS)              # = 0.1
```

It waits for in-flight calls to *pair* start with end. When a `start` upload was dropped, its
`end` can never pair, `pending_count` never reaches zero, and the handler burns the whole 300s —
which is the 5m06s measured, to the second. The `404` is the symptom of the same drop, not the
cause of the wait.

**`FLUSH_TIMEOUT_SECONDS` is a module constant with no setting and no environment variable.**
Grepping the installed wheel finds it only in `call_batch_processor.py`.

*Two levers that look like fixes and are not:*

- **`WEAVE_RETRY_MAX_INTERVAL` / `WEAVE_RETRY_MAX_ATTEMPTS`** — the wrong mechanism entirely;
  retries are not what blocks. Setting them from `WeaveTracing.activate()` was implemented,
  measured (the 5m06s run), found ineffective and reverted. Worth recording *why* it could never
  have worked either: `weave.utils.retry` applies `@with_retry` as a bare decorator, so
  `stop_after_attempt(retry_max_attempts())` and `wait_exponential_jitter(max=retry_max_interval())`
  are evaluated at **decoration** time. `weave.trace_server_bindings.remote_http_trace_server` is
  in `sys.modules` immediately after `import weave` (checked), and `tracing.py` imports weave at
  module scope, so the numbers are frozen before `activate()` runs. Set before the import they do
  change (`10.0` and `1`); set after, the decorators already hold `300`/`3`. Four tests went with
  the revert: they passed by asserting the variable had been *set* rather than that the wait had
  *shrunk* — a mechanism assertion standing in for an outcome, the same trap as F21.
- **`WEAVE_USE_CALLS_COMPLETE=false`** — would select `AsyncBatchProcessor`, which has no pairing
  flush, and unlike the retry settings it is read during `weave.init()` so `activate()` could set
  it. But `remote_http_trace_server.py:260` logs "Project has been previously written to with
  `use_calls_complete=True` and requires 'calls_complete' mode. Automatically upgrading SDK..." —
  weave puts the project back. It is also a change to the write path, traded for shutdown latency,
  which is not a trade to make silently.

*What the code does:* nothing, deliberately. `WeaveTracing.activate()` carries a comment saying the
wait exists and cannot be bounded from here, so the next reader does not spend the afternoon this
took. `WeaveClient.finish()` remains reverted for hanging worse (F11).

*Not reproducible on demand.* `weave.init()` plus one `@weave.op` call exits in **3.4s**; the long
wait needs the server in the state that drops a start. Insurance to know about, not a permanent
condition.

*Also recorded, about testing this:* `monkeypatch.delenv(name, raising=False)` on a name that is
**absent** records no undo, so a variable the code under test writes afterwards survives into the
next test. Verified with a two-test probe.

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

Re-run 2026-09-18 after the F20/F21 work and the move to `run.run_turn`: still 5/5, exit 0, both
tracers active.

The checks assert on tool messages and token counts rather than model prose, so they do not depend
on how the model phrases things. The F5 check was itself verified by breaking the fix and watching
it go red.

**`uv run pytest -m live` can hang *after* reporting its result — do not pipe it to `tail`.**
Observed 2026-09-18: the weave coexistence test printed `PASSED` (1.61s call, 2.04s total) and the
process then hung in teardown while weave's async batch processor retried against a trace server
answering `404` on `.../call/end`, until `timeout` killed it. This is F11's send-queue hazard, and
it is *post-summary*, so everything after pytest's own output is background-thread noise. The wait
itself is now bounded — see F23 — but the piping lesson stands regardless.

Piping the run through `tail -12` therefore discarded the summary line and kept only that noise —
and `pytest ... | tail; echo $?` reports **`tail`'s** exit status, not pytest's, so the run looked
like a clean exit with no result. Redirect to a file and read pytest's exit code directly:

```bash
uv run pytest -m live -q > live.log 2>&1; echo "exit: $?"   # not `| tail`
```

The tests themselves pass; only process exit is affected. Run a single live test by node id when
one is all you need — it reports in ~2s and the hang costs nothing but the wait.

Every finding added since is pinned offline instead, and each was verified by mutation — the
change was reverted in place and the test watched go red. Nine mutants, all killed: a silent `{}`
from `subagent_graphs`, a dropped subagent loop, a removed vacuity guard, `execute` back in the
allowlist, an unsent step limit, an unattached deadline, a no-op deadline check, `raise_error =
False`, and a fixed rather than relative message postcondition. A tenth survived and is written up
in F21.

## Open / unverified

- Whether a **provider-pinned** model id (`org/model:groq`) behaves identically on the token cap
  (F2). Only the router's default selection is covered.
