# Findings

Library and tooling behaviour verified against the installed wheels, not taken from docs. Each entry
records what was checked, what the code does about it, and what pins it so upstream drift fails
loudly instead of changing behaviour quietly.

Versions are recorded once, in CLAUDE.md's "Verified API facts" — two copies of a version list is
one copy that goes stale. Re-verify this file after any `uv sync` that moves them. Verified
2026-09-17/18.

Read this before debugging anything that looks like a library bug. Add to it when you verify
something new.

---

## F1 — `ChatOpenAI` picks the Responses API from the model name, ignoring `base_url`

**Severity: high.** Would break every request against a non-OpenAI endpoint.

`BaseChatOpenAI._use_responses_api(payload)` routes to `/v1/responses` instead of
`/v1/chat/completions` when `use_responses_api` is left at its default `None` and any of:

- `_model_prefers_responses_api(model_name)` — the id starts with a Responses-only prefix, or
  **contains the substring `codex` anywhere**;
- the instance sets `output_version="responses/v1"`, `context_management`, `include`, `reasoning`,
  `truncation`, or `use_previous_response_id`;
- the payload contains `context_management`, `include`, `previous_response_id`, `reasoning`, `text`,
  or `truncation`, or binds any builtin tool.

**None of this consults `base_url`.** The HF router serves Chat Completions only, so an inferred
switch fails at request time. langchain-openai's own `ChatOpenAI` docstring warns about this for
OpenAI-compatible providers.

*What we do:* `USE_RESPONSES_API = False` is a pinned constant in `model.py`, injected by
`build_model` and refused as a `ModelConfig` field. A postcondition asserts it survived
construction.

## F2 — `max_tokens` is renamed to `max_completion_tokens` on the wire

**Severity: low — the router accepts the renamed key.** Logged as a suspected incompatibility;
resolved by live test.

HF documents `max_tokens`. `ChatOpenAI` renames it in **two** places — `ChatOpenAI._default_params`
and `ChatOpenAI._get_request_payload` ("deprecated in favor of max_completion_tokens") — and the
field alias runs the same direction, so the literal `max_tokens` key cannot be sent through the
normal field. The trap in verifying this: `BaseChatOpenAI._default_params` alone says it sends
`max_tokens`; the rename lives in the **subclass override**. Confirm by printing
`ChatOpenAI(...)._default_params`.

Every other parameter matches the OpenAI/HF name: `temperature`, `top_p`, `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `logprobs`, `tools`, `tool_choice`, `response_format`,
`stream`. No `extra_body` needed for any of them.

**Live result (2026-09-17, `openai/gpt-oss-120b` via the router):** the router honours
`max_completion_tokens`. Asking for 200 numbers produced 483 output tokens uncapped, exactly 24 with
`max_tokens=24` bound, and an identical 24 via `extra_body={"max_tokens": 24}`. Both forms work; no
workaround needed.

*What we do:* `ModelConfig` still has no token-cap field (YAGNI). When one is added, the plain
`max_tokens` field is fine. `main.check_token_cap_reaches_the_router` keeps this honest.

*Still unverified:* a provider-pinned id (`org/model:groq`) is a different code path and is not
covered — only the router's own selection is.

## F3 — deepagents ships no `write_todos`; the docs and skill say otherwise

**Severity: medium.** Causes wasted debugging and wrong prompts.

The `langchain-skills:deep-agents-core` skill lists "TodoListMiddleware — default enabled" and
`write_todos` as built-in. Neither is true at 0.7.15: there is no `TodoListMiddleware` in
`deepagents.middleware` at all. The class lives in `langchain.agents.middleware` and reaches the
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

`execute` "executes a shell command in an isolated sandbox" and is enabled with no opt-in.
deepagents classes it as a *filesystem* tool (`FsToolName` includes it), and `FilesystemPermission`
cannot restrict it — `FilesystemOperation` is only `Literal["read", "write"]`. The only supported
lever is `FilesystemMiddleware(tools=[...])` with a narrower allowlist. `read_file` is mandatory:
omitting it raises `ValueError`.

*What we do:* `build_agent` installs a `FilesystemMiddleware` carrying `DEFAULT_FILESYSTEM_TOOLS` —
every filesystem tool except `execute`. A load-time check asserts that allowlist equals
`get_args(FsToolName) - {"execute"}`, so a new upstream tool fails the import rather than being
granted silently. Postconditions assert `execute` is absent from both the middleware's tools and the
compiled graph. See also F20: the parent graph is not the whole allowlist.

## F5 — replacing `FilesystemMiddleware` silently drops every permission rule

**Severity: high.** Silent security failure; follows directly from F4's fix.

deepagents merges middleware by `.name` (`_apply_custom_middleware`): a caller-supplied
`FilesystemMiddleware` **replaces** the default in place. The default is built as
`FilesystemMiddleware(..., _permissions=permissions)`, so `create_deep_agent(permissions=...)`
reaches the tool layer *only* through that instance. Replace it without forwarding and
`_permissions` becomes `[]` — rules gone, no error. (`_build_interrupt_on_from_permissions` is a
separate path and does still fire, which makes the partial failure harder to spot.) `_permissions`
is private API and stores `list(_permissions or [])`, so `None` normalises to `[]`.

**Live result:** with forwarding, `write_file` to a denied path returns `Error: permission denied
for write on /secrets/keys.txt`. With forwarding patched out, the same request **succeeds** — the
file is written.

*What we do:* `least_privilege_filesystem(permissions, backend)` forwards them, with a postcondition
pinning that they landed, and a load-time check that `tools` and `_permissions` are still accepted
parameters. `build_agent` **refuses** the combination of a caller-supplied `FilesystemMiddleware`
and non-empty `AgentConfig.permissions` rather than half-applying it.

## F6 — `BackendProtocol` is an `abc.ABC`, not a `typing.Protocol`

**Severity: low.** Misleads anyone writing a custom backend.

Despite the name, its MRO is `(BackendProtocol, ABC, object)`. Structural typing will not register;
subclass it explicitly. It requires 9 sync + 9 async methods (`ls/read/write/edit/glob/grep/delete/
upload_files/download_files` and their `a`-prefixed twins). Shipped implementations:
`FilesystemBackend`, `StateBackend`, `StoreBackend`, `CompositeBackend`, `LocalShellBackend`,
`ContextHubBackend`, `LangSmithSandbox`.

## F7 — `api.endpoints.huggingface.cloud` is not an inference URL

**Severity: low.**

It is the Inference Endpoints **control plane** for creating and managing dedicated deployments. A
dedicated endpoint serves inference at its own
`https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1/`. Both that and
`router.huggingface.co/v1` are OpenAI-compatible, so switching is a `MODEL_BASE_URL` change and
nothing else (F19 is why that is true from the environment and not only from code).

## F8 — `load_dotenv()` searches from the calling file, not the cwd

**Severity: low.** Invalidates the obvious way of testing the missing-config path.

Running from another directory with `HF_TOKEN` unset still succeeded, because `find_dotenv()` walks
up from `main.py`'s own directory and finds the project `.env`. So "run it from elsewhere" does
**not** test the unconfigured path. This recurs in reverse: a verification script written into a
scratch directory fails to find the project `.env` for the same reason. Pass an explicit path from
anywhere but the package.

*What we do:* `tests/test_main.py` patches `load_dotenv` to a no-op and clears the env var.

## F9 — `.text()` on messages is deprecated in favour of the property

**Severity: low.** Would fail the suite under `filterwarnings = ["error"]`.

`BaseMessage.text` is a property; calling it emits `LangChainDeprecationWarning`. Use
`message.text`.

## F10 — mypy cannot narrow types through `require()`

**Severity: low.** Design constraint, not a bug.

`require(x is not None, ...)` does not narrow `x`. Where a check also narrows, use an explicit `if
... raise CheckFailed(...)` — same runtime behaviour, survives `python -O`, and mypy follows it.
`compiled_tools` in `capabilities.py` is the worked example.

## F11 — LangSmith and Weave coexist over the same run, but weave's SDK raises warnings this project treats as fatal

**Severity: informational (coexistence) / important (the warnings).**

*Coexistence, verified:* `tests/test_tracing.py::test_langsmith_and_weave_trace_the_same_run` (`-m
live`) has both backends configure from the real environment and activate, and
`langchain_tracer_names()` reports `{'LangChainTracer', 'WeaveTracer'}` both *before* and *after* a
real `agent.invoke()` against the router — neither backend's global install displaces the other's.
This is why `available_backends()` activates every configured backend rather than selecting one.

*Two warnings from inside weave's own SDK, fatal only because of `filterwarnings = ["error"]`:*

1. **`gql` deprecation.** `weave.init()` unconditionally calls `ensure_project_exists`, which uses
   `gql`'s old `execute(..., variable_values=..., operation_name=...)` convention on every call. The
   resulting `DeprecationWarning` aborts the call and *masquerades* in the traceback as
   `weave.wandb_interface.project_creator`'s "Unable to access `<entity>/<project>`" path, which
   logs and re-raises whatever was raised. Fixed with a per-test `@pytest.mark.filterwarnings` mark:
   it fires synchronously inside the test, so a mark reaches it.
2. **An unclosed `ssl.SSLSocket`.** The warning actually raised is **not** a `ResourceWarning` —
   that is only the informational `__cause__` on a `pytest.PytestUnraisableExceptionWarning` that
   `_pytest.unraisableexception.collect_unraisable` constructs. That collector re-emits from
   `pytest_runtest_call` (during a test) **or** from `pytest_unconfigure` (at *session* teardown,
   after every test has reported), whichever GC pass collects the socket first. **No per-test mark
   can reach a warning raised at session teardown.** An earlier
   `@pytest.mark.filterwarnings("ignore:unclosed:ResourceWarning")` mark matched the wrong category
   entirely and never caught anything; it appeared to work by GC-timing luck.

Resolved with a `pyproject.toml` `filterwarnings` entry added *alongside* `"error"`, scoped on all
three axes `warnings.filterwarnings` supports — message, category, and originating module:

   ```toml
   'ignore:Exception ignored in.*SSLSocket:pytest.PytestUnraisableExceptionWarning:_pytest\.unraisableexception'
   ```

Verified two ways: `uv run pytest -m live` exits 0, and a mutation check (making `require()` emit a
plain `DeprecationWarning`) still fails the offline suite loudly — the entry narrows rather than
widens the policy.

*Do not "fix" this with `WeaveClient.finish()`.* Closing the client in a `finally` was tried and
reverted: it blocks until its send queue drains, and a queue stuck behind failed writes never
drains. A live run **hung for >90s with no verdict** — strictly worse than the loud crash it was
meant to fix. See F23 for the related 300s exit wait.

## F12 — LangChain swallows exceptions raised inside a callback handler

**Severity: medium.** A broken handler otherwise fails in silence.

`BaseCallbackHandler.raise_error` and `.run_inline` both default to `False` (confirmed via
`inspect`). By default an exception raised inside a hook is caught by `CallbackManager`'s dispatch
and, unless `run_inline=True`, may run off the main thread — so a broken handler stops mirroring, or
reorders its own output, without the run failing. Seen live, unprompted: an F11 run logged `Error in
WeaveTracer.on_llm_end callback: PydanticDeprecatedSince20(...)` — Weave's callback raised,
LangChain caught it, and the run carried on with that hook silently degraded.

*What we do:* `JsonlMirror` pins `raise_error = True` (a broken mirror must be loud) and `run_inline
= True` (so recorded order is call order), and keeps its body incapable of raising on data (F15).
`main()` asserts `mirror.records > 0` after every run, which is what catches "the callbacks were
never attached" — `raise_error=True` alone only catches a handler that ran and threw.

*Reading back which tracers are installed:* `CallbackManager.configure(...) -> CallbackManager` (all
args optional). `{type(h).__name__ for h in CallbackManager.configure().handlers}` builds and
discards a manager; no network, no side effects. That is `tracing.langchain_tracer_names()`.

## F13 — `LANGSMITH_TRACING` must be exactly `"true"`; the check is cached and still honours `LANGCHAIN_*`

**Severity: high.** Live in this repo's own `.env` right now (`LANGSMITH_TRACING=True`, capital T).

`langsmith.utils.tracing_is_enabled(ctx: dict | None = None) -> bool | Literal["local"]` computes
`get_env_var("TRACING_V2", default=get_env_var("TRACING", default="")) == "true"` — a literal string
comparison, no `.strip()`, no `.lower()`, no truthy-word mapping. `True`, `1`, `yes`, `on` all leave
tracing off.

`get_env_var` is `@functools.lru_cache(maxsize=100)`-wrapped with signature `(name, default=None, *,
namespaces=("LANGSMITH", "LANGCHAIN"))`, so:

- **The legacy prefix is not inert.** With only `LANGCHAIN_*` set and no `LANGSMITH_*` variable at
  all, `tracing_is_enabled()` → `True`, `get_tracer_project()` → `LANGCHAIN_PROJECT`'s value, and
  `CallbackManager.configure().handlers` includes a `LangChainTracer`. Reproduced live, twice.
- **The cache outlives `.env`.** A lookup made before `load_dotenv()` runs — anything importing
  `langsmith` early, or a prior call in the same process — is remembered for the life of the
  process; a stale `""` caches as "tracing off" no matter what `.env` says afterward.

mypy sees `get_env_var` as an `Overload` over the cached wrapper rather than a single
`_lru_cache_wrapper`, so it does not see `.cache_clear` even though it exists at runtime. Every call
site needs `# type: ignore[attr-defined]`.

*What we do:* `LangSmithTracing.activate()` calls `get_env_var.cache_clear()` before checking
`tracing_is_enabled()`, and raises `TracingMisconfigured` naming the offending value when langsmith
disagrees with what `from_env`'s more permissive `_TRUTHY` set accepted. `tests/test_tracing.py`
carries a module-level autouse `_clear_langsmith_cache` fixture, and the F11 live test forces
`LANGSMITH_TRACING=true` after `load_dotenv()` — the repo's own `.env` would otherwise fail it.

## F14 — chat models emit `on_chat_model_start`, never `on_llm_start`

**Severity: low.** Would silently produce an incomplete mirror if assumed otherwise.

A live run's JSONL (8 records) contains `chain_start`×3, `chain_end`×3, `chat_model_start`×1,
`llm_end`×1 — no `llm_start` event at all. `ChatOpenAI`, called through deepagents' chat-model path,
fires `on_chat_model_start`, not the legacy plain-LLM `on_llm_start`.

*What we do:* `JsonlMirror` implements `on_chat_model_start` and deliberately has no `on_llm_start`
handler — this measurement is what justifies the omission rather than it being an oversight.

## F15 — `json.dumps(default=...)` applies only to values, never to dict keys, and does not cover cycles or non-finite floats

**Severity: important.** With `raise_error = True` pinned on `JsonlMirror`, this let a logging
callback kill an agent run — the exact failure F12's fix was supposed to make impossible.

`_clip` was `json.loads(json.dumps(value, default=str))`, and both this file and F12 once asserted
that the round trip made the body incapable of raising on data. **That claim was false**, and it was
false in the approved spec before it was false in the code.

`default=` is a callback `json.dumps` invokes only for values it cannot otherwise serialize — never
for dict keys, which must already be `str`, `int`, `float`, `bool`, or `None`:

```python
>>> json.dumps({(1, 2): "x"}, default=str)
TypeError: keys must be str, int, float, bool or None, not tuple
```

`on_tool_end(output: Any)` takes an arbitrary tool return and `on_chain_start`/`on_chain_end` take
arbitrary graph state, so a dict with a non-`str` key is squarely inside what those hooks receive.
Two further gaps in the same round trip:

- **Cycles.** `json.dumps` does not route circular references through `default=` either — a
  self-referential structure (which a naive `dict(state)` copy can produce) raises `ValueError:
  Circular reference detected`.
- **Non-finite floats.** The default `allow_nan=True` emits a bare `NaN`/`Infinity` token. Python's
  reader accepts it back, so the round trip did not fail — but it is not valid JSON, and `jq`, Go's
  `encoding/json` and `serde_json` all reject it. This file's contract is one valid JSON object per
  line, so a silently-accepted `NaN` was a correctness bug even though `_clip` never raised on it.

*What we do:* `_clip` wraps the round trip in `try`/`except (TypeError, ValueError)`, degrades to
`repr(value)`, and passes `allow_nan=False`. Covering tests exercise all three shapes: a tuple-keyed
dict, a self-referential dict, and a payload containing `float("nan")`.

## F16 — mypy and pyright disagree about what satisfies a protocol variable

**Severity: moderate.** CI's type checker is mypy; the editor most likely to open this code runs
pyright (Pylance). A spelling that passes one and fails the other means a clean CI run and a file
full of red squiggles — which is how a protocol gets quietly re-shaped by whoever's IDE complained
loudest. It shipped once already.

`TracingBackend` originally declared `name` as a read-only `@property` and both adapters satisfied
it with a `ClassVar[str]`. mypy accepts that; pyright rejects it:

    "LangSmithTracing" is incompatible with protocol "TracingBackend"
      "name" is not defined as a ClassVar in protocol

All four combinations, against mypy 2.3.1 (`--strict`) and pyright 1.1.414, on 2026-09-17:

| Protocol declares | Implementation uses | mypy | pyright |
|---|---|---|---|
| read-only `@property` | `ClassVar[str]` | pass | **fail** |
| `name: ClassVar[str]` | `ClassVar[str]` | pass | pass |
| read-only `@property` | instance attribute | pass | pass |
| `name: ClassVar[str]` | instance attribute | **fail** | **fail** |

Only two spellings satisfy both, and they are not interchangeable: one *requires* implementations to
use a `ClassVar`, the other requires them not to.

*What we do:* `TracingBackend` declares `name: ClassVar[str]` and both adapters keep their
`ClassVar` — chosen over the property form because a `ClassVar` stays out of a dataclass's generated
`__init__`, so no caller can construct a backend with a name of their choosing. The protocol's
docstring records the constraint. `pyright` is a dev dependency in the gate, in `standard` mode,
pinned explicitly because pyright's default has moved between releases. Not `strict`: that reports
46 issues whose two largest groups argue against this codebase's design rather than finding bugs in
it — `reportUnnecessaryIsInstance` flags the defensive `require(isinstance(...))` checks that
negative-space programming exists to add, and `reportPrivateUsage` flags the deliberate
`_permissions` access in `agent.py`.

*Still unverified:* nothing else here declares a protocol variable, so there is nothing else to
check — but any new protocol with a non-method member should be run past both checkers first.

## F17 — reasoning is most of the output, and the only dial is `reasoning_effort`

**Severity: important.** A cost lever that was running at the provider's default, and it silently
changes what a token cap means.

`openai/gpt-oss-120b` reasons on essentially every call — essentially, not quite: F36 records a
measured call with `reasoning: 0`. From a real `uv run my-agent` run, reasoning tokens as a share of
output:

| check | reasoning | output |
|---|---|---|
| F1 "pong" | 36 | 47 |
| F2 token cap | **21** | **24** |
| F4 shell refusal | 74 | 146 |

The F2 row is the one that matters: with `TOKEN_CAP = 24`, reasoning consumed 21 of the 24 and the
visible answer got **none**. The remaining 3 are not a squeezed answer — they are the harmony header
that *opens* the reasoning channel, spent before the first reasoning token. The response was cut
mid-reasoning, the final channel was never opened, and that is why the record's text is empty; F36
reconstructs the frame token by token. The floor for a non-empty answer is `reasoning + 11`, so this
call would have needed a cap of 32 to show a single word.

**A token cap on a reasoning model is mostly a reasoning cap** — here it was entirely one. The check
still verifies what it claims — the cap is honoured — but it is not evidence that 24 tokens buys 24
tokens of answer, or any.

`reasoning_effort` is a Chat Completions body parameter. The router *documents* `none, minimal,
low, medium, high, xhigh` — **three of those are rejected on the wire, and `REASONING_EFFORTS` has
since been narrowed to `low, medium, high`; see F26.** Verified on the wire 2026-09-17, one prompt,
three settings:

```
effort=None   reasoning=50   output=61    text='9'
effort=low    reasoning= 6   output=17    text='9'
effort=high   reasoning=93   output=104   text='9'
```

Identical answers, a ~15x spread. **Unset is not "off"** — it is the provider's default, which sat
between `low` and `high`.

**Use `reasoning_effort` (str), not `reasoning` (dict).** Both are `ChatOpenAI` fields. `reasoning`
is the Responses API's, and `_use_responses_api` returns `True` whenever `self.reasoning is not
None` — on a default instance it would silently reroute every request to `/v1/responses`, which the
router does not serve (F1). Our pinned `USE_RESPONSES_API = False` short-circuits that, so the pin
protects us — but the parameter would then be sent in a Chat Completions body, where it does not
belong.

*What we do:* `ModelConfig.reasoning_effort` (default `None`), validated against
`REASONING_EFFORTS`. A bad value from a caller is a `CheckFailed`; a bad value from the
`REASONING_EFFORT` environment variable is a `ValueError` at the edge. Verified end to end:
`REASONING_EFFORT=low uv run my-agent ...` produced 8/4/0 reasoning tokens where the same three-call
shape had produced 16/11/11 at default.

*Answered since, in F26:* `none`, `minimal` and `xhigh` do not differ from `low` — they are
refused with a 400. And unset is not "between low and high": it is `medium`, token for token.

*Still unverified:* whether every provider the router may select honours it.

## F18 — callbacks see the langchain request, not the HTTP body; and `ToolMessage` hides `status` in its repr

**Severity: important.** The log could not answer either of the two questions it exists for.

Measured on a real 5-check run (52 records, 36KB), the mirror recorded the conversation three times
and the request not at all: `llm_end` dropped `tool_calls` and `finish_reason`, `chat_model_start`
recorded no request parameters, and `chain_*` was 77% of the bytes as Python `repr` strings —
because deepagents passes `Command` objects where the signature promises `dict`, and `default=str`
stringified them. The largest field was 3745 chars against `MAX_FIELD_CHARS = 4000`, so a slightly
longer conversation would have truncated a repr blob into an unparseable fragment.

*What the code does now:* `llm_end` records `tool_calls`, `metadata` (`finish_reason`, `model_name`,
`model_provider`, `system_fingerprint`, `service_tier`) and non-empty `extra` from
`additional_kwargs` — where reasoning *content* would land if a provider returned any (none does;
the token count in `usage.output_token_details.reasoning` is all we get). `chat_model_start` records
`params`, with tool definitions reduced to their names. `chain_*` records a `_state_summary` — which
keys a step carried and how the message list grew — because the content is already recorded
structurally. `_tool_output_summary` records `content`, `status`, `name` and `tool_call_id` as
fields, with `artifact` reduced to a boolean flag on purpose (tools may attach arbitrary payloads
and a log is not the place to copy them); plain non-`ToolMessage` returns pass through untouched.
`serialized` is never written: it can carry credentials. `ChatOpenAI._get_invocation_params()` was
checked and carries none.

**Two facts that outlive the redesign:**

- **`ToolMessage.status` is the authoritative success/error signal and it does not survive
  `str()`.** Before the fix a permission denial was findable only by substring-matching a repr; it
  is now `status: "error"`.
- **`params` is the langchain-level request, not the HTTP body.** Callbacks only ever receive
  `invocation_params`, read *before* `_get_request_payload` renames anything. So a bound token cap
  appears in the log as `max_tokens: 24` while the wire carries `max_completion_tokens: 24` (F2).
  There is no callback hook that sees the final payload; reaching it would mean monkeypatching a
  private method. Recorded rather than worked around.

Second live run with both fixes in: 52 records, 19,458 bytes for the same five checks that
originally produced 36,214 — no repr blobs, `finish_reason` distinguishing `length` from
`tool_calls` and `stop`, and the withheld-`execute` allowlist visible in the request record rather
than only asserted in tests.

## F19 — a hand-maintained mapping drifts; only a check that reads the dataclass stops it

**Severity: important.** `ModelConfig.from_env` read three of seven fields, and one of the four
omissions made this module's own docstring false.

`model.py` states its thesis at the top: adding a setting is "one new field with a default: no
factory signature change, no factory body change, no call site change." That held for `build_model`
and `as_kwargs()`. It did **not** hold for `from_env`, which needed a constant, a lookup, a parse, a
validation and a new keyword per field — the growing argument list the parameter-object design
exists to eliminate, inside the file that argues against it.

The consequence was not merely inelegant. `HF_ROUTER_BASE_URL`'s docstring promises that pointing at
a dedicated Inference Endpoint (F7) needs "no code change", while `main.py` builds its config only
through `from_env()` — which never read `base_url`. **Using a dedicated endpoint required editing
source.** Nothing caught it because nothing compared the mapping to the dataclass.

*What we do:* the mapping is `_ENV_FIELDS`, a table of `(field, env var, parser, required)` rows,
and three load-time checks assert it against `dataclasses.fields(ModelConfig)`: no field
unreachable, no row naming a non-field, no two rows sharing a variable. Removing the `base_url` row
now fails the import with `ModelConfig fields unreachable from the environment: ['base_url']`. A
table rather than reflection over annotations, deliberately: inferred names would be implicit and
inferred parsers would give generic errors.

**Why one config object instead of keyword parameters.** `create_deep_agent` has 18 parameters (16
keyword-only). Threading them through `build_agent` one at a time would mean editing its signature
*and* its body every time the agent gains `subagents`, `skills`, `backend` or `interrupt_on` — and
defaults do not help, because that edit still modifies a function that was supposed to be closed.
With a parameter object, adding a setting is one new field with a default: no factory change, no
call-site change. The cost is that `as_kwargs()` splatting trusts field names to be real parameters,
and a typo would otherwise surface as a `TypeError` from inside the library — or vanish into a
`**kwargs` signature. `contracts.check_config_contract` closes that at import time against
`inspect.signature(create_deep_agent)` and `ChatOpenAI.model_fields` (pydantic's `__init__` is
`**data`, so its fields and aliases are the real contract), and refuses fields the factory injects
itself (`model`, `use_responses_api`) because they would collide on splat.

The same reasoning produced a second pin, in `agent.py`. `check_config_contract` asserts our
*fields* are real `create_deep_agent` parameters, but it cannot notice a **new** parameter appearing
— and a new parameter is exactly how the shell `execute` tool arrived switched on with no opt-in
(F4). `KNOWN_CREATE_DEEP_AGENT_PARAMS` pins the set at 18, so a deepagents upgrade that adds one
fails the import by name and has to be reviewed for what it enables before being accepted.

**Taxonomy note.** Everything `from_env` reads is outside input, so every failure is an operating
error: absent, blank, unparseable, *and* out-of-range. The last used to leak — `MODEL_TEMPERATURE=5`
parses fine and then trips a `require()`, which would crash with an `AssertionError` traceback
implying a bug in this code. `from_env` now constructs inside a `try`, catches `CheckFailed`, and
re-raises as a `ValueError` naming the variable at fault.

*Still unverified:* whether `AgentConfig` wants the same treatment. It deliberately covers 5 of 18
parameters (YAGNI), so completeness is the wrong property for it — but nothing records *which*
omissions were considered and rejected.

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

Ours is narrowed — but only because deepagents merges middleware by `.name` into the subagent's list
as well as the parent's. That merge is behaviour we do not control and nothing depended on it
deliberately. An upstream change would re-grant shell execution through `task` while every test
stayed green, because no test and no postcondition ever looked at a subagent.

*How it was checked:* `agent.get_subgraphs(recurse=True)` returns `[]` for a deep agent and
`get_graph(xray=1)` shows only `model`, `tools` and one middleware node — the subagent graphs are
not reachable through any public API. They live in the `task` tool's closure, under the freevar
`subagent_graphs`.

*What we do:* `capabilities.subagent_graphs(agent)` reads that closure, and
`_require_shell_withheld` asserts absence across the parent graph **and** every subagent. Reaching
into a closure is worse than reaching into `nodes["tools"]`, so every way the structure can move
raises `CheckFailed` rather than returning an empty mapping that would make the check vacuous: no
closure, neither `func` nor `coroutine`, a non-mapping value, and an empty mapping are four separate
named failures. A `task` tool that is simply absent returns `{}`, because nothing can then be
dispatched.

The suite states both halves. `test_build_agent_withholds_the_shell_tool_from_every_subagent` is the
claim; `test_a_bare_deep_agent_does_grant_the_shell_tool_to_its_subagent` keeps it honest — without
it the first would keep passing if deepagents stopped granting `execute` for its own reasons.

**Mitigating, and not a reason to relax.** `StateBackend` does not implement
`SandboxBackendProtocol`, and `create_deep_agent`'s docstring says `execute` "will return an error
message" for non-sandbox backends. A leak would have been survivable *at today's backend* — which is
exactly the argument least privilege exists to not depend on.

*Whether a deny rule reaches inside a subagent: closed, offline.* The subagent's filesystem tools
are not equivalents of the parent's, they are **the same objects**: `compiled_tools(subagent) is
compiled_tools(parent)` holds name by name, because the middleware deepagents replaced was ours and
each tool closes over that one instance's `self._permissions` and `self.backend`. Identity is the
mechanism; two tests are its observable consequence, driving the subagent's own `write_file` with a
hand-built `ToolRuntime` and no model involved:

- `/secrets/keys.txt` → `status="error"`, `"Error: permission denied for write on
  /secrets/keys.txt"`.
- `/notes/ok.txt` gets *past* the permission gate and fails inside `StateBackend`, which refuses to
  run outside a graph execution. That refusal is the discriminator: it is reachable only by a call
  the rule allowed, so the deny is targeted rather than blanket.

Both were checked against a rule that no longer covers the path and a rule flipped to `allow`; the
enforcement test fails under each. A mutant that breaks the subagent *alone* does not exist in this
code — which is the finding, not a gap in the tests. `compiled_tools` is public for this reason:
names are enough for an absence check, identity is what explains why the rules reach a second graph
at all.

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

`least_privilege_filesystem` passed only `tools` and `_permissions`. Two of the rest *rewrite the
conversation*: a tool result over 20,000 tokens, or a user turn over 50,000, is evicted to the
filesystem and replaced with a pointer — a context-engineering decision this project never made, and
one that shows up in the JSONL mirror as a state write nobody asked for.

**The backend is the more serious half, for a reason that is not about defaults.** `backend` was
`None`, so the replacement middleware built a `StateBackend` of its own while `create_deep_agent`
wires the *caller's* backend into `SkillsMiddleware` and the summarisation middleware. Since our
middleware displaces its filesystem middleware by name, a caller who supplied a backend would have
got one agent on two filesystems. This is F5's failure — a setting that reaches `create_deep_agent`
but not its replacement — wearing a different hat, and `agent.py`'s own docstring advertised adding
a `backend` field as the usual one-line change while that was false.

*What we do:* the three bounds worth owning are named constants (`TOOL_RESULT_TOKEN_LIMIT`,
`HUMAN_MESSAGE_TOKEN_LIMIT`, `GREP_MATCH_LIMIT`), passed explicitly, asserted as postconditions, and
pinned at import against the wheel's current defaults — so a deepagents change fails the load with
both numbers in the message instead of quietly resizing how much of a tool result the model sees.
`max_execute_timeout` is deliberately **not** pinned: it bounds `execute`, and `execute` is
withheld, so pinning it would assert something about a tool nobody has. `least_privilege_filesystem`
gained a `backend` parameter, `_agent_kwargs` forwards `kwargs.get("backend")` into it, and a
postcondition refuses a middleware whose backend differs from the one `create_deep_agent` will use.

**A decorative test, caught by mutation — the general lesson.** Every pinned value equals
deepagents' current default, so reading one back off the built middleware cannot distinguish "we
chose it" from "we inherited it": deleting all three arguments left the instance-reading test
passing. The discriminating test records the *constructor call* instead, with the real
`FilesystemMiddleware` still building the object. Both are kept, because they make different claims
— one that the bound was sent, one that it landed, and `_permissions` was sent and silently dropped
once already (F5).

## F22 — mypy and pyright disagree about subscripting a non-required `TypedDict` key

**Severity: minor.** A second instance of the F16 pattern, in a different place.

Every key of `langchain_core.runnables.RunnableConfig` is non-required (`total=False`). mypy accepts
`config["recursion_limit"]`; pyright in `standard` mode reports `reportTypedDictNotRequiredAccess`.
pyright is right: the key genuinely may be absent. `.get(...)` satisfies both.

Surfaced while typing `run.py`'s `Invokable`, where the two checkers *agreed* about something more
interesting: a protocol parameter is contravariant, so declaring `invoke(self, input: dict[str,
Any], ...)` makes a real `CompiledStateGraph` **fail** to satisfy the protocol, because its own
`input` is typed `InputT | Command | None`. Both rejected it with the same diagnosis. `input` is
therefore `Any` and `config` is typed exactly — which is what earns the check: `run_turn` builds a
real `RunnableConfig`, so a misspelled bound is a type error rather than a silently ignored key.

## F23 — weave's exit waits 300s for call starts that were dropped, and nothing configures it

**Severity: important.** A trace server that drops a write does not degrade telemetry; it blocks
process exit for five minutes.

*Observed:* a `pytest -m live` run whose tests took **3.11s** while the process took **5m06s** to
exit, and an earlier one killed by `timeout` at 480s. Logs fill with `404 Not Found` for
`.../call/end` and `Cannot end call <id>: no start found in project`.

*Root cause,* from a `faulthandler.dump_traceback_later` stack dump of every thread. The main thread
sits in `CallBatchProcessor.stop_accepting_new_work_and_flush_queue`, which weave registers with
`atexit`:

```python
deadline = time.monotonic() + FLUSH_TIMEOUT_SECONDS      # = 5 * 60
while time.monotonic() < deadline:
    pending_count = len(self._pending_starts) + len(self._pending_ends)
    if pending_count == 0:
        break
    time.sleep(FLUSH_POLL_INTERVAL_SECONDS)              # = 0.1
```

It waits for in-flight calls to *pair* start with end. When a `start` upload was dropped, its `end`
can never pair, `pending_count` never reaches zero, and the handler burns the whole 300s — the 5m06s
measured, to the second. The `404` is the symptom of the same drop, not the cause of the wait.
**`FLUSH_TIMEOUT_SECONDS` is a module constant with no setting and no environment variable**;
grepping the installed wheel finds it only in `call_batch_processor.py`.

*Two levers that look like fixes and are not:*

- **`WEAVE_RETRY_MAX_INTERVAL` / `WEAVE_RETRY_MAX_ATTEMPTS`** — the wrong mechanism entirely;
  retries are not what blocks. Setting them from `WeaveTracing.activate()` was implemented,
  measured, found ineffective and reverted. It could never have worked: `weave.utils.retry` applies
  `@with_retry` as a bare decorator, so `stop_after_attempt(...)` and
  `wait_exponential_jitter(max=...)` are evaluated at **decoration** time.
  `weave.trace_server_bindings.remote_http_trace_server` is in `sys.modules` immediately after
  `import weave`, and `tracing.py` imports weave at module scope, so the numbers are frozen before
  `activate()` runs. Four tests went with the revert: they passed by asserting the variable had been
  *set* rather than that the wait had *shrunk* — a mechanism assertion standing in for an outcome,
  the same trap as F21.
- **`WEAVE_USE_CALLS_COMPLETE=false`** — would select `AsyncBatchProcessor`, which has no pairing
  flush, and unlike the retry settings it is read during `weave.init()`. But
  `remote_http_trace_server.py:260` logs "Project has been previously written to with
  `use_calls_complete=True` ... Automatically upgrading SDK" — weave puts the project back. It is
  also a change to the write path traded for shutdown latency, which is not a trade to make
  silently.

*What we do:* nothing, deliberately. `WeaveTracing.activate()` carries a comment saying the wait
exists and cannot be bounded from here, so the next reader does not spend the afternoon this took.
`WeaveClient.finish()` remains reverted for hanging worse (F11).

*How reliably it fires.* First recorded as "not reproducible on demand", on the evidence that
`weave.init()` plus one `@weave.op` call exits in **3.4s** locally. That reading was too generous.
Measured again 2026-09-18 from a clean GitHub Actions runner — no prior weave state, the scheduled
`-m live` workflow — the job ran **5m51s** for a suite whose tests take about four seconds. So the
minimal one-op case exits fast, but a real agent turn under the LangChain integration hits it
consistently rather than occasionally. **Treat roughly five minutes as the expected shutdown cost of
any process that traces a real run through Weave.**

*Consequence for any timeout around this.* `.github/workflows/live.yml` sets `timeout-minutes: 20`,
which looked generous when written for a four-second suite and was not: 5m51s leaves no useful
headroom under a 10-minute ceiling, and a 5-minute one — an entirely reasonable-looking choice —
would have killed a passing run. Any future timeout wrapping a Weave-traced process needs the same
300s allowance on top of whatever the work itself costs.

---

## F24 — a step limit does not bound an agent that can dispatch a subagent

**Severity: critical.** The bound that exists to stop a runaway did not reach the graph most likely
to run away.

`run.RunBounds.step_limit` is sent as `recursion_limit` on every `invoke`, and it works — on the
parent. It does not reach a `task` subagent. **langchain's `create_agent` binds
`recursion_limit: 9_999` onto every graph it compiles** (`{'recursion_limit': 9999, 'metadata':
{'ls_integration': 'langchain_create_agent', ...}}`), `create_deep_agent` re-binds the same value,
and deepagents invokes a subagent with only `{"configurable": {"ls_agent_type": "subagent"}}` — its
own bound config wins the per-key merge, which deepagents states outright
(`middleware/subagents.py`: *"the subagent's bound config still wins collisions (e.g.
`lc_agent_name`, `recursion_limit`)"*).

Measured 2026-09-18 with a fake model that always calls one tool, under `step_limit=25`:

| what the model does | model calls before the limit tripped |
|---|---|
| always calls `ls` (parent only) | **12** |
| always calls `task` | **5002** — `GraphRecursionError: Recursion limit of 9999 reached` |
| always calls `task`, after the fix | **13** — `StepLimitExceeded` |

So 25 steps buys 12 model/tool round trips, and before the fix a single dispatch bought 5000. Only
`RUN_DEADLINE_S` was bounding a real run.

Two things made this hard to see. The parent's limit *is* honoured, so the bound looked like it
worked. And `create_deep_agent`'s own return carries the same 9999, which a caller's explicit
`invoke` config overrides — so the same number behaves like a decision in one place and an
inheritance in another.

*What we do:* `capabilities.SUBAGENT_STEP_LIMIT` (25), applied by building twice. `build_agent`
compiles the agent, reads deepagents' own general-purpose subagent back out with
`capabilities.subagent_graphs`, rebinds it with `.with_config({"recursion_limit": ...})`, and passes
it back as a `CompiledSubAgent` named `general-purpose` — which is the supported way to replace the
default. Taking deepagents' own graph rather than assembling a replacement means summarisation,
tool-call patching and anything else it adds next keep working; only the number changes. Verified
that deepagents' subsequent `.with_config({"metadata", "run_name"})` does not clear ours.

**The two builds share one `_agent_kwargs` result**, because rebuilding them would call
`least_privilege_filesystem` twice and put the parent and the subagent on two different
`StateBackend`s — F21 arrived at from the other direction.
`test_the_bounded_subagent_is_still_on_the_parents_filesystem` asserts the tool objects are
identical, which is what proves the second build reused the first's middleware.

**A subagent that runs out aborts the whole turn** rather than reporting back, which is why the
measured total is 13 and not 156: the subagent raises through the `task` call. Strict, and loud,
which is the right side to fail on — but it means `StepLimitExceeded` has to name both limits,
since either graph can be the one that ran out.

*Also fixed here:* the step limit had no edge. `main()` caught `DeadlineExceeded` and nothing else,
so a run that exhausted its wall clock printed `error: ...` and a run that exhausted its steps
printed a `GraphRecursionError` traceback — the same `RunBounds`, two vocabularies.
`run.StepLimitExceeded` is the translation, an operating error for the same reason
`DeadlineExceeded` is one.

*Also found here:* `AgentConfig.subagents` is a second door onto the allowlist. A spec that does not
carry our `FilesystemMiddleware` gets one of deepagents' own, with every tool including `execute`.
`_require_shell_withheld` already reads every subagent graph back, so this fails the build rather
than granting a shell — pinned by
`test_a_caller_supplied_subagent_cannot_re_grant_the_shell_tool`.

*Still unverified:* whether a nested subagent (a subagent that itself dispatches) is bounded. Today
none can — the general-purpose subagent has no `task` tool.

## F25 — the router's providers do not all serve the same context window

**Severity: important.** "128k context" is a property of the model; what you get is a property of
whichever provider the router picked.

gpt-oss-120b natively supports 128k (OpenAI, *Introducing gpt-oss*, 5 Aug 2025). Read off
`https://router.huggingface.co/v1/models` 2026-09-18, the live providers for that model do not agree:

| context_length | providers |
|---|---|
| 131072 | groq, novita, cerebras, nscale, together, fireworks-ai, ovhcloud, deepinfra |
| **128072** | baseten |
| *not advertised* | featherless-ai, scaleway |

Nine of eleven state a number, one of those is 3000 tokens shorter than the rest, and two state
nothing. With routing unpinned, the usable window is whichever provider answers — so a prompt sized
against 131072 is a prompt that fails intermittently.

*What we do:* `main.check_every_provider_serves_the_context_we_assume` reads the catalogue (one
HTTP GET, no inference) and fails if any provider that *states* a length is below
`capabilities.CONTEXT_WINDOW_TOKENS` (128,000), reporting the ones that state nothing. Asserting on the unstated
ones would be a check that can never go green; the mitigation for those is pinning `:provider`,
which is a decision, not an assertion. A `require()` guards against the catalogue publishing no
lengths at all, which would otherwise make the check pass by measuring nothing.

**This makes `:provider` a correctness lever, not only a reproducibility one.** CLAUDE.md framed
pinning as something an eval needs; it is also what turns the context window from a range into a
number.

## F26 — `reasoning_effort` has three levels, not the six the router documents

**Severity: important.** A precondition that accepts values the request is guaranteed to fail on.

`REASONING_EFFORTS` listed `none, minimal, low, medium, high, xhigh` — the values the router
documents, recorded in F17. gpt-oss was post-trained on **three**: "the two open-weight models
support three reasoning efforts—low, medium, and high" (OpenAI, *Introducing gpt-oss*), carried in
the system message by the harmony format. Measured on the wire 2026-09-18, one prompt, every value:

```
effort=low       reasoning=10  output=21   text='391'
effort=medium    reasoning=63  output=74   text='391'
effort=high      reasoning=80  output=91   text='391'
effort=None      reasoning=63  output=74   text='391'     <- identical to medium
effort=minimal   400  "reasoning_effort: Input should be 'none', 'low', 'medium' or 'high'"
effort=xhigh     400  same
effort=none      400  "Failed to apply chat template ... Unsupported reasoning effort"
```

Two of those 400s come from the provider's schema and one from the model's own chat template, which
is what "a provider-side mapping onto a three-level model" looks like from outside: `none` passes
validation and the model then refuses it.

This also closes F17's open question. **Unset is `medium`** — token-for-token identical, not merely
"between low and high".

*What we do:* `REASONING_EFFORTS` is `{"low", "medium", "high"}`. It is a property of the *model*,
so a different model means widening it deliberately, the way `DEFAULT_FILESYSTEM_TOOLS` is widened
deliberately — and failing locally is still better than a 400 from inside the provider.
`main.check_reasoning_efforts_are_the_ones_the_router_takes` probes every allowed value live **and**
one forbidden value, because "everything we allow was accepted" also passes on a router that
accepts everything.

## F27 — reasoning comes back as a token count, never as text

**Severity: informational, and worth re-checking.** Three sinks would store it if it ever arrived.

gpt-oss ships a deliberately unsupervised chain of thought: OpenAI "did not put any direct
supervision on the CoT", the model "will often explicitly disobey instructions in its CoT", and
their guidance is that "developers should not directly show CoTs to users… They may contain
hallucinated or harmful content, including language that does not reflect OpenAI's standard safety
policies, and may include information which the model is being explicitly asked to not include in
the final output."

Everything this harness records is a sink for that text if it ever arrives: `logs/*.jsonl` (local
and gitignored), LangSmith, and W&B — the last two not local. Measured 2026-09-18 at
`reasoning_effort="high"`: the reply carried 44 reasoning tokens, `additional_kwargs` held only
`refusal`, and neither `reasoning` nor `reasoning_content` was present. `mirror.py` already noted
this; it is now checked rather than assumed.

*What we do:* `main.check_no_reasoning_content_comes_back` asserts the absence, with a `require()`
that the run actually reasoned — otherwise "no content" would pass for the wrong reason. It also
justifies something already true: the live checks assert on **tool messages and token counts, never
on model prose**, which is now the vendor's own guidance rather than only good practice.

## F28 — an installed package can reconfigure the agent without touching a parameter

**Severity: important.** The door `KNOWN_CREATE_DEEP_AGENT_PARAMS` cannot see.

That pin fails the import when `create_deep_agent` grows a parameter — which is how `execute`
arrived (F4). A profile plugin grows no parameter. `create_deep_agent` resolves a `HarnessProfile`
keyed by the model's provider and id (`graph.py`: `_harness_profile_for_model(model, _model_spec)`)
from a process-global registry, and `deepagents.profiles` populates that registry by executing a
zero-arg callable from **every installed distribution** advertising an entry point in
`deepagents.harness_profiles` or `deepagents.provider_profiles`. A `HarnessProfile` carries
`extra_middleware`, `excluded_tools`, `excluded_middleware`, `tool_description_overrides`,
`base_system_prompt` and `general_purpose_subagent` — every one a capability decision, none of them
visible at a call site.

Verified 2026-09-18: both groups are empty in this environment, and the builtin harness profiles are
keyed per-model (Anthropic models, Nemotron, Codex), so none matches the router model.

**The near miss is the provider side.** deepagents ships
`ProviderProfile("openai", init_kwargs={"use_responses_api": True})`. That is F1's pin, reversed, for
every `openai:*` model. It does not reach us — `init_kwargs` feed `init_chat_model`, so they apply
only when `create_deep_agent` is handed a model **string**, and `build_agent` refuses strings. The
refusal of model strings is therefore load-bearing for correctness, not only for routing hygiene,
and anyone "simplifying" `build_agent` to accept one would silently reroute every request to
`/v1/responses`, which the router does not serve.

*What we do:* `capabilities.DEEPAGENTS_PLUGIN_GROUPS` plus a load-time `require()` that both groups
are empty. Installing such a plugin is then a failed import with the distribution named, rather than
a quiet change of behaviour. Allowing one is a deliberate edit.

## F29 — a paused turn looks exactly like a finished one

**Severity: important.** The failure mode that arrives with human-in-the-loop, found before it
shipped.

`FilesystemPermission.mode` accepts `"interrupt"`, and deepagents turns interrupt-mode rules into
`interrupt_on` entries for the parent *and* every subagent (`_build_interrupt_on_from_permissions`).
So HITL needs no new config field. What it needs is a `run_turn` that can say a turn paused.

Measured 2026-09-18 against a real compiled graph. An interrupted `invoke` returns:

```
keys: ['__interrupt__', 'files', 'messages']
messages: 2      (human, then an AI message carrying the tool call)
```

The messages **already grew**, so the old postcondition (`len(messages) > len(sent)`) passed. A
caller got a partial conversation, the tool never ran, and nothing said so. That is worse than a
tripped assertion.

Worse still, those messages contain a tool call with no `ToolMessage`. Handing them back as
`history` shows the model a call it can see went unanswered — the one shape invariant a conversation
has.

The protocol, all verified rather than read from docs:

- `result["__interrupt__"]` is a `list[Interrupt]`; each has `.id` and `.value`.
- `.value` is `{"action_requests": [{"name", "args", "description"}], "review_configs":
  [{"action_name", "allowed_decisions"}]}`. One interrupt batches a model turn's tool calls, so the
  count a decision list must match is the number of *action requests*, not of interrupts.
- Resuming needs a checkpointer: `RuntimeError: Cannot use Command(resume=...) without checkpointer`.
  Interrupts still *fire* without one — the pause is simply unresumable.
- The resume payload is `Command(resume={"decisions": [...]})`. A bare list raises `TypeError: list
  indices must be integers, not str` from inside `HumanInTheLoopMiddleware.after_model`, which does
  `interrupt(hitl_request)["decisions"]`.
- The middleware zips decisions onto requests in order, so a short list misaligns them silently.

*What we do:* `run.TurnResult` — `messages`, `interrupts`, `thread_id`, and a `paused` property. It
keeps `__iter__`, `__len__` and `__getitem__` so every existing `messages = run_turn(...)` call site
still indexes, slices and measures the way it did; a third outcome is not a reason to break the
first two. The postcondition became "added messages **or** paused". `run.resume_turn` answers the
requests, with preconditions on the decision count, on each decision having a `type`, and on the
turn actually being paused. `run_turn` refuses a `history` carrying an unanswered tool call, which
catches the corruption whether or not the caller kept the `TurnResult`. A pause on a graph with no
checkpointer is a `CheckFailed`: it is a turn nobody can finish, and it is our own misconfiguration.

`AgentConfig` gained `checkpointer` and `subagents`, both one-line fields of the kind its docstring
promised. `thread_id` is carried on the `TurnResult` rather than retyped at the resume call site —
a retyped id does not fail, it starts a second run wearing the first one's name.

Proven end to end offline in `tests/test_run.py`: an interrupt rule pauses a real graph before the
tool runs, approving runs the held tool, **rejecting leaves the write undone**, and an agent with no
interrupt rule never pauses.

## F30 — tool schemas are the request; the conversation is a rounding error

**Severity: important.** ~2,090 input tokens on every turn, paid whether or not a tool is used.

A one-line prompt (`"What defines an AI Agent?"`, system prompt `"You are a helpful assistant."`)
reported **2,086 input tokens** and called no tools. Captured on the wire 2026-09-18 with an httpx
event hook, prompt `"hi"`, `max_tokens=8`:

```
10,508 bytes sent;  router counted input_tokens: 2,092
```

| tool | bytes on the wire |
|---|---|
| `grep` | 2,383 |
| `task` | 1,957 |
| `read_file` | 1,680 |
| `glob` | 1,633 |
| `edit_file` | 1,103 |
| `write_file` | 735 |
| `delete` | 636 |
| `ls` | 466 |
| **total** | **~10.5 KB — effectively the whole request** |

The conversation was 12 tokens. `grep`, `task` and `glob` alone are 57% of it. **This is a fixed
per-turn cost that scales with the tool count, not with the work** — and it is the number to put
against any future "let's add a tool".

### Capturing what is actually sent

Three layers, increasing fidelity:

1. `OPENAI_LOG=debug` — zero code, exists in openai 3.14.1, redacts `authorization`. Goes to stdlib
   logging, not the mirror.
2. `ChatOpenAI._get_request_payload(messages, **bound.kwargs)` — what langchain builds, offline, no
   network. Note the `**bound.kwargs`: calling it on the `RunnableBinding` that `bind_tools`
   returns silently omits the tools and reports a request with none.
3. An httpx event hook — byte truth:
   ```python
   client = httpx.Client(event_hooks={"request": [lambda r: sink(r.read())]})
   ChatOpenAI(**cfg.as_kwargs(), use_responses_api=USE_RESPONSES_API, http_client=client)
   ```
   **It must be passed at construction.** `model_copy(update={"http_client": ...})` is silently
   ignored — the openai client is built during field validation, so a field replaced afterwards
   never reaches it. The hook simply never fires, which looks identical to a request that was never
   made.

*What we do:* `mirror._request_size` records `total_bytes`, `tools_bytes`, `messages_bytes` and
`tool_count` on every `chat_model_start`, plus a per-tool `tool_bytes` breakdown **once per run** —
the schemas are static within a run, while the messages grow. It serializes the langchain-level
request rather than hooking HTTP, and the error is measured rather than assumed: 10,593 bytes
against 10,508 on the wire, **+0.8%**, all of it `json.dumps` whitespace the body omits. Closing
F18's gap properly would mean an HTTP hook that sees the whole conversation; the estimate is enough
to act on and costs nothing.

### Bounds added here

`capabilities.call_limits()` installs two `ToolCallLimitMiddleware` instances on every agent:

- `TOOL_CALL_LIMIT` (24, all tools) — **not covered by `step_limit`.** That bounds graph *steps*,
  and langgraph's tool node executes every call in one `AIMessage`, so a model that fans out ten
  calls a turn does ten times the work per step and the step limit sees one step either way.
- `TASK_DISPATCH_LIMIT` (3, `task` only) — `SUBAGENT_STEP_LIMIT` bounds how far one dispatch runs;
  nothing bounded how many there are. Twelve parent round trips times a 25-step subagent is ~144
  model calls inside a turn bounded at 25 steps. Three caps the worst case near 37.

`exit_behavior="continue"`, not `"error"`: the exceeded call is blocked and the agent answers with
what it has, which is better than crashing a turn that is already bounded twice over by the step
limit and the deadline. The blocked call is visible in the mirror as a tool message. The two
instances take distinct names (`ToolCallLimitMiddleware` and `ToolCallLimitMiddleware[task]`) —
asserted, because deepagents merges middleware by `.name` and a collision would mean one silently
replacing the other, and the names are the library's to choose.

**`AgentConfig.middleware` reaches the parent only.** Measured: a `ModelCallLimitMiddleware` passed
there appears as `before_model`/`after_model` nodes on the parent graph and **not** on the subagent.
deepagents inherits caller middleware into the general-purpose subagent only when its `.name`
shadows one of the default slots — which is why our `FilesystemMiddleware` gets there and a call
limit does not. Anything relied on as a global ceiling has to be checked on both graphs.

### Not done, and why it is written down

The real fix for 2,090 tokens is fewer or leaner tools, not more middleware. Two library levers are
worth revisiting **when the tool count grows**, and neither pays today at eight tools:

- `LLMToolSelectorMiddleware(model=…, max_tools=…, always_include=…)` — picks a subset per turn.
  Spends an extra model call to save input tokens, which are the cheap ones; it trades latency for
  the wrong currency at this scale.
- `ProviderToolSearchMiddleware(searchable_tools=…)` — hides tools behind provider-side search.
  Needs provider support, which the router's selection does not guarantee (F25 is the same
  problem: providers differ).

The cheaper move remains subtraction: `grep`, `glob` and `task` cost ~6 KB per request and none has
a consumer yet.

---

---

## F31 — compaction was configured to fire above the context window it was serving

**Severity: critical.** The one bound on context nobody in this repo owned, and it was set 33%
above the window.

`create_deep_agent` installs a `SummarizationMiddleware` unconditionally — on the parent
(`graph.py:888`) **and** on the general-purpose subagent (`graph.py:803`) — and sizes it from the
model's profile:

```python
# deepagents/middleware/summarization.py, compute_summarization_defaults
has_profile = model.profile is not None and "max_input_tokens" in model.profile
# with a profile:  trigger=("fraction", 0.85)   keep=("fraction", 0.10)
# without one:     trigger=("tokens", 170_000)  keep=("messages", 6)
```

A router model id has no profile at all. Measured 2026-09-18:

```
model: openai/gpt-oss-120b
profile: None
summarization defaults: {'trigger': ('tokens', 170000), 'keep': ('messages', 6),
                         'truncate_args_settings': {'trigger': ('messages', 20), ...}}
```

170,000 against the 128,000-token floor F25 established (shortest stated provider: 128,072).
**Proactive compaction could therefore never fire.** What was left was the middleware's reactive
path: it catches `ContextOverflowError` and retries — but `langchain_openai` raises that only when
the provider's error text matches one of four hardcoded substrings
(`chat_models/base.py:614`: `context_length_exceeded`, `Input tokens exceed the configured limit`,
`prompt is too long`, `ContextWindowExceededError`). Across eleven router providers, whether any of
those strings comes back is unverified — and F25 is the finding that says providers differ. A
substring match against a third party's error prose is not a context strategy.

**The fix.** `capabilities.bounded_compaction` states all four thresholds and replaces deepagents'
own by `.name`, the same route `least_privilege_filesystem` takes:

| bound | ours | deepagents' fallback |
|---|---|---|
| `COMPACTION_TRIGGER_TOKENS` | 96,000 (75% of the floor) | 170,000 |
| `COMPACTION_KEEP_MESSAGES` | 6 | 6 (stated anyway) |
| `COMPACTION_ARG_TRUNCATION_MESSAGES` | 20 | 20, but `None` if you build one by hand |

The remaining quarter of the window is headroom for the reply, the ~2,090 tokens of tool schemas
sent every turn (F30) and the summary itself. Compaction that triggers *at* the window triggers too
late to help.

`truncate_args_settings` is the trap in building one by hand: its constructor default is `None`,
which switches argument clipping off entirely. Passing the middleware yourself and omitting it
silently removes a capability deepagents' own factory switches on.

**What pins it.** `CONTEXT_WINDOW_TOKENS` moved from `main.py` into `capabilities.py` — it is no
longer only a number a live check reports, it is the number the trigger is sized against, and a
bound belongs to the thing it bounds. At import, `0 < COMPACTION_TRIGGER_TOKENS <
CONTEXT_WINDOW_TOKENS` and the four parameter names still exist. At assembly, `_agent_kwargs`
counts the compaction middlewares in the list it hands over: two would mean ours joined the stack
instead of replacing the one sized above the window. `LIBRARY_COMPACTION_TRIGGER_TOKENS` is the
discriminator, so the test cannot pass by agreeing with a library that changed its mind.

Verified on the compiled graph 2026-09-18 — `build_agent` now leaves exactly **one** summarization
middleware alive, carrying `('tokens', 96000)` and `('messages', 6)`, on one backend, reaching both
the parent and the subagent.

---

## F32 — a turn cut short by our own tool-call ceiling looked exactly like a clean one

**Severity: important.** The harness manufactured the false-success case it exists to prevent.

Both call limits run `exit_behavior="continue"` (F30): the exceeded call is replaced by a
`ToolMessage(status="error")` and the agent answers with what it already has. Reproduced
2026-09-18 against a real compiled agent and a fake model fanning out six `write_file` calls a turn:

```
model calls: 6 | paused: False | messages: 37
tool messages: 30 | status=error: 6
blocked message: Tool call limit exceeded. Do not make additional tool calls.
final reply   : All done! I wrote every file you asked for.
TurnResult surface: ['action_requests', 'interrupts', 'messages', 'paused', 'thread_id']
```

Six writes never happened, the agent said they had, and nothing on the result said otherwise.
`main._single_turn` printed the reply and returned 0. The only record was `logs/*.jsonl` — a file
for a human, not a signal for code.

**The fix.** `TurnResult.failed_tool_calls` returns every `ToolMessage` whose `status` is `error`,
which covers all three ways a call comes back unfulfilled: the tool failed, a `FilesystemPermission`
denied it, or a call limit blocked it. `main._single_turn` names them on stderr and exits non-zero —
the reply is still printed, because it is what the agent said, but it is no longer the only thing a
caller reads.

Deliberately **not** a `require()`. A failed tool call is a fact about the run, not a violated
contract of ours, and the caller decides what it means. It is also the one rung of the verification
ladder reachable before the domain lands: "did a tool call error" is a fact about the run, not about
the answer, which is why CLAUDE.md's "deliberately not verified" section does not cover it.

---

## F33 — every bound in `run.py` reset on a resume

**Severity: important.** The human gate was the way around the limits the human gate was added to.

`resume_turn` called `_run_config(bounds, ...)`, which built a fresh `RunDeadline` starting now and
sent `recursion_limit: bounds.step_limit` again in full. A turn paused and approved ten times got
ten complete 600-second budgets and ten complete step allowances. `_invoke`'s own comment said "a
resume path that quietly dropped any of those would make 'pause, then approve' the way around every
limit here" — nothing was dropped, and the property a reader took from it was false anyway.

**What could be fixed exactly.** The wall clock. `TurnResult.elapsed_s` carries what the agent has
spent, `resume_turn` runs on `deadline_s - elapsed_s`, and a remainder of zero or less is a
`DeadlineExceeded` naming the spend rather than a `CheckFailed` from `RunDeadline`'s own
positive-budget precondition. Time a human spends deciding is **not** charged: the clock is read
when an invocation starts and when it returns, so the gap between them belongs to nobody. This
bounds how long the agent may run, not how long a conversation may stay open.

**What could not.** The step limit. `recursion_limit` counts supersteps within one invocation,
langgraph restarts the count on a resume, and the count it reached is not reported back in any form
this module can read — so no remainder can be computed. Saying so is the deliverable; the
compensating bound is `RunBounds.resume_limit` (3), which counts the halves. Worst case is four step
budgets rather than unboundedly many, and a turn needing a fourth round of human approval is a turn
to restart rather than extend. `ResumeLimitExceeded` is an operating error like the other two: a
human who keeps approving is the outside world being persistent, not a caller passing something
impossible.

---

## F34 — the backend postcondition passed by comparing `None` to `None`

**Severity: minor today, and only by luck.** Found while wiring F31.

`_agent_kwargs` asserted:

```python
declared_backend = kwargs.get("backend")
require(declared_backend is None or kwargs["middleware"][0].backend is declared_backend, ...)
```

`AgentConfig` has no `backend` field, so `declared_backend` was always `None` and the check passed
without checking. Meanwhile the default path really did build two: `least_privilege_filesystem`
made a `StateBackend` (`capabilities.py`) and `create_deep_agent` made another
(`graph.py:637`). F21 is the finding about exactly that split; the guard written to catch it bit
only when a backend was passed explicitly, which is the case where it is least needed.

Harmless so far because `StateBackend` is stateless — verified 2026-09-18, its instance `__dict__`
is empty and every operation reads graph state through the config. That is a fact about deepagents,
not about this code, and it is the kind of fact this file exists to stop relying on silently.

**The fix.** `_agent_kwargs` names the fallback once, sets `kwargs["backend"]` to it, and hands the
same object to `create_deep_agent`, `least_privilege_filesystem` and `bounded_compaction`. The
postcondition is now unconditional, and the mutant that reverts it is caught.

---

## F35 — nothing bounded what a turn costs

**Severity: important.** Three bounds on a turn and none of them counted the thing it is billed for.

`step_limit` counts graph steps, `deadline_s` counts seconds, `resume_limit` counts halves. A turn
is charged in tokens, and the numbers are not small: F30 measured ~2,090 input tokens on a one-line
prompt because the tool schemas are resent on every call, and F31 now lets a conversation reach
96,000 tokens before compaction fires.

The arithmetic the ceiling sits against, all of it inside the existing bounds:

```
25 steps                     -> 12 parent model/tool round trips
TASK_DISPATCH_LIMIT = 3      -> 3 dispatches of a 25-step subagent
                             -> ~37 model calls in one turn
37 calls x 96,000 tokens     -> ~3.5M tokens
```

`RunBounds.token_limit` (500,000) is roughly 5% of that worst case and something like seventy times
an ordinary tool-using turn: it never bites on real work and does bite on a runaway. **The number to
revisit first when a domain lands** — it is the one bound here whose right value depends on what a
turn is worth.

**Shape.** `RunTokenBudget` is `RunDeadline` with a different unit: it accumulates on `on_llm_end`
and refuses on `on_chat_model_start`, so the call that crossed the line is paid for and the one
after it is not. That is the only granularity available, because a token count exists only once the
call has returned. `run_inline` and `raise_error` are set for the F12 reason. It reads
`message.usage_metadata`, the same field `mirror.py` records, so the number that bounds a run is the
number the log shows. Subagent calls count: `ensure_config` seeds a subagent's run from the ambient
parent config, which matters more here than for the wall clock because a `task` dispatch is where
the tokens actually go.

**A provider that omits usage makes the bound blind**, so `RunTokenBudget.unmeasured_calls` counts
those separately rather than folding them into a silent zero. Crashing a turn over someone else's
response shape would be worse; the router does report usage, and the live F2 check already refuses
to grade itself without `output_tokens`, so a change would not go unnoticed for long.

Carries across a pause exactly as `elapsed_s` does (F33): `TurnResult.tokens` accumulates and
`resume_turn` runs on `token_limit - tokens`.

## F36 — an empty response text is not a truncated answer; it is an answer that never started

**Severity: low — an observability defect, not a behavioural one.** The log's `""` reads as "the
model said almost nothing"; it means "the model was still reasoning". Different diagnosis, same
record.

`logs/20260918T201908Z-db00e749.jsonl` line 4 — the F2 token-cap check — records `output_tokens:
24`, `reasoning: 21`, `finish_reason: "length"`, `outputs: [""]`. The obvious reading is that
reasoning took 21 and the answer got the leftover 3 and was too short to survive. Wrong on both
counts: the answer got **zero**, and the 3 were spent before the reasoning, not after it.

gpt-oss speaks harmony, so a response is a sequence of channels rather than a string, and the
channel markers are billed as output tokens. The frame, counted with the model's own tokenizer:

```
<|channel|>analysis<|message|>                  3 tokens   the prompt ends at <|start|>assistant
    ...reasoning...                           rsn tokens   output_token_details.reasoning
<|end|>                                         1 token
<|start|>assistant<|channel|>final<|message|>   5 tokens
    ...answer...                          content tokens   the only part that reaches outputs[]
<|return|>                                      1 token
```

A completed reasoning response therefore bills `reasoning + content + 10`, and a response that skips
the analysis channel bills `content + 4`. Every `llm_end` in that run reconciles to the token,
`residual = output_tokens - reasoning - tokens(text)`:

| lines | residual | reading |
|---|---|---|
| 2, 10, 70, 86 | **10** | complete: both channels opened and closed |
| 46 | **4** | `reasoning: 0` — the analysis channel was never opened |
| 4, 78, 82 | **3** | cut mid-reasoning; only the analysis header was emitted |
| 80 | **8** | reasoning finished, cut 4 tokens into the 5-token final header |
| 22, 34, 58 | 30, 34 | not truncated: the commentary header plus JSON arguments, which the mirror records under `tool_calls`, not `outputs` |

Three consequences:

- **The floor for a non-empty answer is `reasoning + 11` tokens.** Line 4 would have needed a cap of
  32 to show one word. This is F17's point sharpened: the cap was not *mostly* a reasoning cap, it
  was entirely one.
- **`""` under `finish_reason: "length"` is a distinguishable state, and the residual says where the
  cut landed** — 3 means still reasoning, 4–9 means reasoning finished and the answer was about to
  begin, ≥10 with empty text means a genuinely empty answer. Line 80 is the middle case and would
  have produced text one token later.
- **An empty `outputs` on a tool-calling record is not truncation at all.** Those tokens are real
  output; they live in `tool_calls` because that is where langchain puts them.

Reproducer — offline, no router call, using the tokenizer `langchain-openai` already pulls in
(tiktoken 0.14.0):

```python
import json, tiktoken
enc = tiktoken.get_encoding("o200k_harmony")          # gpt-oss's own tokenizer
for line in open("logs/<run>.jsonl"):
    d = json.loads(line)
    if d.get("event") != "llm_end" or not d.get("usage"):
        continue
    u = d["usage"]
    text = "".join(d.get("outputs") or [])
    print(u["output_tokens"] - u["output_token_details"].get("reasoning", 0)
          - len(enc.encode(text)), d["metadata"]["finish_reason"])
```

*What we do:* nothing in code. The mirror already records `finish_reason`, `output_tokens` and
`reasoning` (F18), which is everything the reconstruction needs — the ambiguity is in the reader,
not the record. If it becomes worth closing, the cheap version is a derived flag on `llm_end`
(`finish_reason == "length" and not outputs and not tool_calls` → cut before the answer began),
which needs no tokenizer on the hot path.

*Why this is not pinned by an offline test:* `o200k_harmony`'s BPE ranks are downloaded on first use
and cached under `data-gym-cache`, so the `_forbid_network` fixture would fail a cold run. Pinning it
properly means vendoring the ranks, which costs more than the finding is worth.

*Still unverified:* whether every provider frames responses identically. Weak evidence that it is a
model property rather than a provider one: this run carries two distinct `system_fingerprint`s and
the 10-token frame held on both. The routed provider itself is not recorded — unpinned routing plus
a fingerprint is not a provider name (F25 is the reason to pin `:provider`). Also unverified whether
`reasoning: 0` is a model decision or a provider one; it happened once, on the turn that read a
`ToolMessage` back and restated it.

## F37 — streaming never asks for token usage; this router volunteers it anyway

**Severity: low as measured, and the reasoning is the point.** A predicted defeat of `RunTokenBudget`
that does not reproduce live. Recorded because the mechanism is real, the bound survives on
behaviour nobody requested, and the prediction was wrong in an instructive direction.

**The mechanism, read off the installed wheels.** Three steps, each verified:

1. `stream_mode="messages"` attaches langgraph's `StreamMessagesHandler`, which is a
   `langchain_core.tracers._streaming._StreamingCallbackHandler`. `BaseChatModel._should_stream`
   returns `True` whenever one is attached, so a node that calls `model.invoke()` is switched onto
   `_stream` by the *presence of the handler*. This is a property of streaming, not of async: sync
   `graph.stream(stream_mode="messages")` does it too, and `stream_mode="values"` does not.
2. `langchain_openai` asks the provider for usage only when told to
   (`chat_models/base.py:1787-1788` and `:2078-2079`): `if stream_usage: kwargs["stream_options"] =
   {"include_usage": stream_usage}`.
3. The default is off **for us specifically** (`chat_models/base.py:1358-1375`): `stream_usage`
   auto-enables only when `self.openai_api_base is None and "OPENAI_BASE_URL" not in os.environ`.
   `ModelConfig.base_url` always carries `HF_ROUTER_BASE_URL` and `build_model` asserts it, so
   `stream_usage` stays `None` and `_should_stream_usage` falls through to `self.stream_usage or
   False`.

**The prediction:** streamed chunks carry no `usage_metadata`, `RunTokenBudget` counts nothing, and
F35's bound stops bounding while `unmeasured_calls` climbs. Reproduced against a **fake** model:
`tokens=0, unmeasured_calls=1`.

**Live result (2026-09-19, `openai/gpt-oss-120b`, unpinned routing) — the prediction is false
here.** A real `build_agent` graph, the real `RunTokenBudget`, one prompt per mode:

```
stream_mode=values                 chunks=  2  tokens=2116  unmeasured=0
stream_mode=updates                chunks=  4  tokens=2116  unmeasured=0
stream_mode=messages               chunks= 18  tokens=2116  unmeasured=0
stream_mode=['values','messages']  chunks=  8  tokens=2116  unmeasured=0
invoke (baseline)                              tokens=2116  unmeasured=0
```

Identical to the non-streamed baseline in every mode. Separately, at the `ChatOpenAI` layer:
`stream_usage` unset and `stream_usage=True` produced the *same* 8 chunks with exactly one carrying
usage (`total_tokens=114`, matching `invoke`). **The router emits a final usage chunk whether or not
`include_usage` is requested**, so the flag changes nothing on this path.

**Why record a non-bug.** The bound holds because of a provider behaviour this project does not
request, does not control and did not know about. F25 already establishes that the router's eleven
providers disagree about basics; one that omits the final usage chunk blinds the bound, and unpinned
routing means which provider answers is not a decision anyone made. What makes that survivable is
`unmeasured_calls` (F35) — written so a blind bound would not be silent — and it is read by nothing
outside a unit test today.

*What we do:* nothing. No streaming is built, and on the YAGNI grounds in CLAUDE.md none should be
until something consumes it. If it ever is: set `stream_usage=True` explicitly — it costs nothing,
it is a decision rather than a gift, and this measurement is only about the provider that happened
to answer — and assert `unmeasured_calls == 0` across a streamed turn. That assertion is the
deliverable, more than the feature.

**Two adjacent traps, both measured, if streaming ever lands.** `stream_mode="messages"` alone
carries no `__interrupt__` at all, and v3's `GraphRunStream.output` omits it even when
`.interrupted` is `True` — either one recreates F29 exactly. And v3 streaming emits
`LangChainBetaWarning` from `langgraph/pregel/main.py`, which `filterwarnings = ["error"]` turns
into a failed build.

**A caveat about how this was nearly got wrong.** The fake-model reproduction was real and the
mechanism it exercised was real, but a fake that omits chunk usage is evidence about the fake, not
about the router. The live call is what settled it. Same lesson as F21 from the other direction.

*Still unverified:* whether the other ten providers volunteer usage on a streamed response. Only the
one that answered on 2026-09-19 is covered, and it is not recorded which one that was.

## F38 — a large message in the middle of a conversation escapes every context bound

**Severity: important, and precisely scoped.** Three separate mechanisms each bound context in a
different way, and all three share one blind spot: a large message that is not the last one and not
old enough to compact.

**Compaction (F31) only compacts what is older than `keep`.** `COMPACTION_KEEP_MESSAGES` is 6 — a
floor, not a target — so a conversation of 6 or fewer messages has nothing eligible to compact at
any token threshold, no matter how far over `COMPACTION_TRIGGER_TOKENS` it runs. Measured
(`test_compaction_cannot_fire_while_every_message_fits_inside_what_it_keeps`): three messages worth
~104,000 approximate tokens — comfortably over the 96,000 trigger — reached the model whole, all
416,000 characters of the largest one included.

**Human-message eviction only looks at the last message.** deepagents'
`_check_eviction_needed` (`filesystem.py:3376`) reads `messages[-1]` and nothing else:
`if messages and isinstance(messages[-1], HumanMessage): ...`. A `HumanMessage` of any size sitting
one position earlier is never inspected, so `HUMAN_MESSAGE_TOKEN_LIMIT` never sees it. Measured
(`test_a_huge_human_message_that_is_not_last_is_never_evicted`): a 201,000-character `HumanMessage`
followed by one more turn comes back with no `lc_evicted_to` tag at all — the identical fixture size
that *does* get evicted when it is last (the adjacent eviction test) is untouched purely because of
position.

**`TOOL_RESULT_TOKEN_LIMIT` truncates at 80,000 characters (`NUM_CHARS_PER_TOKEN * 20,000`), but
`read_file`'s 100-line default (deepagents' `DEFAULT_READ_LIMIT`,
`deepagents/middleware/filesystem.py:973`) cuts most long files first.** Measured
(`test_the_line_limit_cuts_a_long_file_before_the_character_bound_can`): a 4,000-line, 134,890-
character file — over 1.6 times the character bound — came back as ~3,000 characters with no
truncation marker, because line 100 arrived long before byte 80,000. The character bound is only
reachable on files with few, very long lines.

**Put together: a large `HumanMessage` sitting mid-conversation is bounded by none of the three.**
Each mechanism does exactly what it documents — compaction protects recent context, eviction
protects the next request, the character limit protects one read — and none of the three was
written to cover the others' gap.

*What we do:* nothing yet. The shape is now tested rather than argued, so a change to any of the
three bounds has to confront it instead of discovering it later. Note what this is not: `RunTokenBudget`
(F35) still bounds the **run** in tokens regardless of where in the conversation they sit, so this is
a context-window risk — the model sees less than the full history, or sees more than the window can
hold — not an unbounded-spend risk.

*Still unverified:* whether a real conversation reaches this shape in practice. Producing one needs
a domain, which is deliberately TBD (see "Scope discipline"). Also unverified: whether a
`ToolMessage` that reaches state by a path other than `read_file` (and so skips both the line limit
and the character truncation) would land in the same unbounded gap — nothing here measures that.

---

## F39 — the parent graph runs to 9999 too; only `run_turn` was ever bounded

**Severity: important, and precisely scoped.** F24 recorded that langchain's `create_agent` binds
`recursion_limit: 9999` on every graph it compiles, and fixed it for the `task` subagent. The
"every" is literal: the parent carries it as well, and nothing recorded that.

```python
bound_step_limit(build_agent(model))   # 9999
```

Measured 2026-09-21 with a model that requests one `ls` per turn and never answers — an ordinary
tool, so no subagent limit can abort it early, and both call limits run `exit_behavior="continue"`
so a blocked call is handed back as an error and the model simply asks again:

```
agent.invoke({"messages": [...]})      # no run_turn
→ GraphRecursionError after 3,325 model calls
```

**`run_turn` was never affected.** It sends `RunBounds.step_limit` as `recursion_limit` on the
invocation, and at the top level an explicit invoke config beats the graph's own bound config —
which is the opposite of the subagent case, where deepagents invokes the subagent with *its* bound
config and that wins the per-key merge. So the sanctioned path was bounded all along. What was
wrong is the floor underneath it: `run.py`'s module docstring says a call-site bound means other
callers "inherited langchain-core's defaults by accident", and names 25 as that default. The actual
inheritance is 9999, about four hundred times larger.

*What we do:* `capabilities.PARENT_STEP_LIMIT` (25), bound onto the returned graph by `build_agent`
with `with_config`, asserted as a postcondition because `with_config` returns a copy. `with_config`
returns `Self`, so the annotated return type survives, and `compiled_tools`, `subagent_graphs` and
the subagent's own bound limit all still read back off the copy — checked, not assumed.

Two tests, because the claim and its discriminator are different assertions: a bare invoke now stops
at or under `PARENT_STEP_LIMIT`, and `run_turn` with a smaller `step_limit` still gets the smaller
number. The first was written once against a model that dispatches subagents and **survived the
mutant** — `SUBAGENT_STEP_LIMIT` raises through the tool call and aborts the parent long before the
parent's own limit matters, so the test passed whatever the parent was bound to. That is recorded
here because it is the exact failure mode `CLAUDE.md` warns about, caught only by mutating the fix.

`LIBRARY_SUBAGENT_STEP_LIMIT` is renamed `LIBRARY_STEP_LIMIT`: the number is bound on every graph,
not only on subagents, and a name that says otherwise makes the parent's discriminator read wrong.

---

## F40 — the harness threw away the only read-back it had

**Severity: important.** deepagents' `StateBackend` keeps the agent's filesystem in graph state, so
every invocation returns it:

```
RAW STATE KEYS: ['files', 'messages']
files = {'/f1-0.txt': {'content': 'x', 'encoding': 'utf-8', 'created_at': ..., 'modified_at': ...}}
```

`_invoke` took `result["messages"]` and dropped the rest. `TurnResult` had no field for it.
`mirror._state_summary` deliberately records a state payload's *key names* and message count only,
because writing the whole state was 77% of a real run's bytes (F30). So the object the agent claims
to have created — the deterministic artifact that answers "did the work happen" without reading a
word of prose — was visible in no sink at all: not to the caller, not in the log, not in a trace.

This matters because of what `CLAUDE.md` says next to it. *What is deliberately not verified* argues
that the verification ladder needs a domain before its first rung can be built, and that rung is
"a deterministic rule — an exit code, a schema, a read-back of the object the agent claims to have
created". The read-back was already in hand on every turn. It needed a field, not a domain.

*What we do:* `TurnResult.files`, beside `failed_tool_calls` and `answered` — a fact about the run,
not a judgement of it. Nothing compares it against what the model *said* it wrote, because what the
agent should have produced is the domain's question; what it did produce is this.

Not accumulated across a pause the way `elapsed_s` and `tokens` are: the filesystem is state, so
what the graph reports on a resume is already the whole of it, and adding the earlier half back
would double-count a file the agent edited twice.

*Still unverified:* nothing consumes it yet. `main.py` prints `failed_tool_calls` and would be the
obvious place, and is deliberately left alone as scaffolding. An emptiness assertion on its own is
also vacuous — an empty mapping means "wrote nothing" and "no `StateBackend`" alike, which is why
the test that asserts emptiness is paired with one that writes.

---

## F41 — a second door onto the middleware stack, and it is not an entry point

**Severity: minor today, structural.** F28 pinned the two entry-point groups through which an
installed package can reconfigure the agent. There is a third route that pin cannot see, because it
registers nothing.

`deepagents/middleware/_prompt_caching.py` (0.7.15):

```python
def append_prompt_caching_middleware(middleware):
    middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    bedrock = _create_bedrock_prompt_caching_middleware()       # import_module("langchain_aws...")
    if bedrock is not None: middleware.append(bedrock)
    fireworks = _create_fireworks_prompt_caching_middleware()   # import_module("langchain_fireworks...")
    if fireworks is not None: middleware.append(fireworks)
```

`graph.py` calls it three times: on the parent (`:905`), on every subagent spec (`:709`) and on the
general-purpose subagent (`:812`). So installing `langchain-aws` or `langchain-fireworks` — as
anyone's transitive dependency, never named at a call site — adds middleware to every graph this
harness compiles. It passes through no parameter, so `KNOWN_CREATE_DEEP_AGENT_PARAMS` cannot see it;
it advertises no entry point, so `DEEPAGENTS_PLUGIN_GROUPS` cannot either.

**`AgentConfig.middleware` is therefore not the middleware stack**, and neither is the list
`_agent_kwargs` assembles. Verified 2026-09-21: `_agent_kwargs` returns four middlewares; the
compiled agent runs `AnthropicPromptCachingMiddleware`'s `wrap_model_call` as well, and it appears
in no node list because a `wrap_model_call` middleware is not a graph node. There is no route found
to read the installed stack back off a compiled graph — the closure hunt that works for
`subagent_graphs` finds no middleware list — so unlike the tool allowlist this one cannot be
asserted after the fact, only pinned before.

**Blast radius today is nil, for two reasons that are both someone else's to change.**
`_should_apply_caching` returns `False` for anything that is not a `ChatAnthropic`, and deepagents
constructs the middleware with `unsupported_model_behavior="ignore"`. That argument is load-bearing
in a way worth stating: the same class supports `"warn"`, and `"warn"` plus this repo's
`filterwarnings = ["error"]` would turn every model call into a failed test.

*What we do:* `capabilities.DEEPAGENTS_CACHING_PROBE_MODULES` names the two probed distributions and
`installed_caching_probes()` reports which are importable; neither is, and an import-time `require`
fails the load if that changes. Verified by adding an installed package to the probe list, which
makes `capabilities.py` unimportable — the same failure shape as the plugin pin. The unconditional
Anthropic middleware cannot be pinned away and is recorded by a test instead, including the
`"ignore"` argument.

---

## F42 — a constant compared to itself is not a pinned constant

**Severity: important.** F21 established the rule for values equal to a library default: *record the
call, because reading the value back cannot tell "we chose it" from "we inherited it"*. The wider
version went unnoticed — **a value read back from the same symbol the test imported cannot fail at
all**, default or not.

Measured 2026-09-21. Five constants changed at once:

```
HF_ROUTER_BASE_URL  → "https://api.openai.com/v1"
DEFAULT_TEMPERATURE 0.0 → 1.9        DEFAULT_MAX_RETRIES 2 → 99
RUN_DEADLINE_S 600.0 → 6000.0        TOKEN_LIMIT 500_000 → 5_000_000
→ 376 passed, 2 deselected
```

The harness pointed at another provider, at temperature 1.9, with ten times the wall clock and ten
times the token ceiling, and nothing went red. The sharpest case is
`test_build_model_ignores_ambient_openai_env_vars`, which exists to prove an ambient
`OPENAI_BASE_URL` cannot redirect us: it sets that variable to `api.openai.com` and asserts
`openai_api_base == HF_ROUTER_BASE_URL`, so when the constant *is* that URL the test passes while
making the opposite claim. `RECURSION_LIMIT` was the one constant already pinned, and only
incidentally — by the literal `match="25 steps"` in an unrelated error-message test.

Three more of the same shape:

- **`exit_behavior="continue"`** deleted from both `ToolCallLimitMiddleware` calls → 376 passed.
  Equal to the library default, so F21's own rule applied and had not been.
- **`thread_limit is None`** was asserted about a parameter `call_limits()` never passes — a claim
  about langchain's default dressed as a claim about this repo.
- **`MAX_FIELD_CHARS` 4000 → 200** → all 51 mirror tests passed. The fixture was
  `"x" * (MAX_FIELD_CHARS * 2)` and the assertion `len(output) == MAX_FIELD_CHARS`, so both
  goalposts moved together and no value could turn it red. A 20x loss of log fidelity, invisible.

And two vacuity failures of the kind `require_withheld` refuses in `src/`: `as_kwargs` stubbed to
`return {}` left five contract tests green, because `set(...) <= accepted` and `"model" not in ...`
are both vacuously true of an empty dict; and `on_chat_model_start` stubbed out entirely left the
credential test green, because `"hf_super_secret" not in stream.getvalue()` holds on an empty
stream.

*What we do:* one literal pin per decision — `test_the_router_defaults_are_the_values_that_were_chosen`,
`test_the_run_bounds_are_the_values_that_were_chosen`,
`test_the_capability_bounds_are_the_numbers_that_were_chosen` — plus the default `RunBounds` object,
because pinning the constant does not pin the field that defaults to it. `exit_behavior` is recorded
on the call with the `monkeypatch` pattern `least_privilege_filesystem` already had; the mirror
fixture uses literals behind a guard assertion; the vacuous tests gained the positive half. The
five-constant mutation now fails four tests.

*The general rule, for the next constant:* if the expected value in a test can be changed by editing
`src/`, the test is a tautology. Write the number down.

---

## F43 — `_invoke` read one key and dropped the state

**Severity: important.** F40 gave `TurnResult` a `files` field, which fixed the symptom and left the
shape of the bug intact: `_invoke` still read named keys out of the result and discarded whatever
else came back.

What a compiled deep agent actually declares it may return:

```python
typing.get_type_hints(agent.output_schema)   # ['files', 'messages', 'structured_response']
```

`structured_response` was being dropped. It is empty on every path today, because `AgentConfig` has
no `response_format` field — but adding one is the single-line change `AgentConfig`'s own docstring
promises, and the caller who made it would have found `run_turn` silently discarding the only thing
they added it for. Proven against a graph returning the declared shape: `files` surfaced,
`structured_response` and an unrecognised key both vanished.

Two further facts from the same introspection:

- **The declared output class does not mention `files`.** `OutputAgentState` names `messages` and
  `structured_response` only; `files` is contributed by `FilesystemMiddleware`, which is exactly how
  a key that arrives on every turn stays invisible to anyone reading the type.
- **The graph's internal channels are wider than its output.** `run_tool_call_count` and
  `thread_tool_call_count` — the call-limit counters — are state channels but not output keys, so
  they do not come back from `invoke` and `TurnResult` cannot report how many calls a run made.
  `failed_tool_calls` counts the blocked ones from the messages instead.

*What we do:* `TurnResult.state` carries every key the graph returned except `messages` and
`__interrupt__`, which have fields of their own — excluded because the messages are the bulk of a
turn's memory and two copies can disagree. `files` and `structured_response` are properties over it,
so the keys with a documented meaning keep a name while an unrecognised key is still *there* rather
than destroyed.

Carried whole rather than one field per key on purpose: the keys are not ours to enumerate, and a
middleware a domain adds tomorrow lands in `state` without `run.py` changing. What stops that being
silent is a pin: `agent.KNOWN_OUTPUT_STATE_KEYS` plus `capabilities.compiled_output_keys`, asserted
in `build_agent`, so a newly declared output key fails the build and someone decides whether it
deserves a property. The same argument as `KNOWN_CREATE_DEEP_AGENT_PARAMS`, one layer down — that
pin catches a new parameter, this catches a new output. The reader raises rather than returning an
empty set when the structure moves, because a reader that finds no keys would make every
"is this key known?" check pass by checking nothing.

### The filesystem does not survive the turn

Worth stating beside `files`, because it is easy to assume otherwise. `run_turn` sends
`{"messages": sent}` and nothing else, and with no checkpointer langgraph retains nothing, so the
agent's filesystem is empty at the start of every turn. Measured 2026-09-22 — a file written in turn
one is gone in turn two of the same conversation, `history=` passed:

```
turn 1 files: ['/a.txt']
turn 2 files: ['/b.txt']        # same agent, same conversation
carried? False
```

The conversation continues across turns; the filesystem does not. `CLAUDE.md` says
`run_turn(agent, prompt, history=...)` "returns exactly what the next call wants", which is now
narrower than it reads: `TurnResult` carries `files`, and the next call has no way to accept them.

Two readings, pointing opposite ways. It is *convenient*: because state resets, `files` is always
precisely "what this turn wrote", which is a cleaner verification artifact than a cumulative
filesystem. It is also a *gap*: an agent that writes notes in turn one cannot read them in turn two,
and any multi-turn domain hits it immediately.

*What we do:* nothing. The fix is a `checkpointer` — already an `AgentConfig` field — and it would
also make `files` cumulative, so the read-back stops meaning "this turn". Which of the two readings
matters depends on the domain, and inventing multi-turn filesystem persistence now would be the
YAGNI this repo spends a section refusing. Recorded so the choice is made rather than discovered.

---

## F44 — reloading a module to test its load-time checks rebinds every class it defines

**Severity: important, and the reason two pins moved file.** Fifteen check sites in `src/` run at
*import*: that `execute` is still a filesystem tool, that `create_deep_agent` grew no parameter,
that `ChatOpenAI` still accepts every alias `ModelConfig` splats. They execute once, when the test
session imports the package, and no ordinary test can make one false — by the time it runs the
import has long since succeeded.

`importlib.reload` with the library patched is the only route, and it has a trap that the first
eight tests did not hit and the next three did.

**A reload rebinds every class the module defines.** `importlib.reload` re-executes the module body,
so `class AgentConfig` produces a *new* class object while every other test module still holds the
one imported at collection time. `build_agent`'s own `require(isinstance(agent_config, AgentConfig))`
then compares an instance of the old class against the new one:

```
CheckFailed: expected an AgentConfig, got AgentConfig
```

Measured 2026-09-22: reloading `my_agent.agent` and `my_agent.model` failed **15 unrelated tests**
across `test_capabilities.py`, `test_main.py` and `test_run.py` — the grep bound, the compaction
behaviour, the eviction tests and every HITL test. **None of them fails when run alone**, which is
the worst property a test failure can have. `my_agent.capabilities` reloads safely for one reason
only: it defines no classes, just constants and functions.

*What we do:* two things.

**The fixture refuses the unsafe case.** `tripping_an_import_time_check` in `tests/conftest.py`
asserts the target module defines no classes of its own before reloading it, naming them and saying
what to do instead. A rule that is checked beats a rule that is written down.

**The checks in class-defining modules moved into functions.** `agent.py`'s two
`KNOWN_CREATE_DEEP_AGENT_PARAMS` checks became `contracts.check_known_parameters`, and `model.py`'s
alias pin became `contracts.check_required_parameters`; both are still called at import, so the
load-time guarantee is unchanged, and both are now ordinary functions a test drives with bad
arguments. `contracts.check_config_contract` was already exactly this shape, which is what made it
the obvious precedent rather than an invention.

Two more moved for a related reason: `capabilities.require_compaction_fits_the_window` and
`model.require_env_fields_cover_the_config` compare constants defined *in the file that checks
them*, so no patch a test can apply reaches either. A reload could not drive them and neither could
anything else; as functions they take the numbers as arguments.

**The result, measured rather than counted** (by wrapping `require()` and `CheckFailed` in pytest
plugins that log their call site, then diffing against an AST walk): **110 `require()` sites, 100
tripped; 13 `raise CheckFailed` sites outside the helpers, 12 tripped.** Every remaining untripped
site is in `main.py`. No import-time check is untripped any more.

*Worth knowing for the next one:* a module-level check is a check that can only be driven by
reloading, and reloading is only safe in a module with no classes. **Prefer a function called at
import.** It costs one line and keeps the check testable.

---

## F45 — eight checks at one attempt each cannot tell a flake from a regression

**Severity: important.** The only end-to-end measurement this repo has was
unscheduled and ran once per check.

Two separate gaps, found together on 2026-09-23 while auditing the docs:

**The checks were never scheduled.** `live.yml`'s cron runs `pytest -m live` —
**two tests**. `uv run my-agent` runs the eight checks, and nothing invoked it;
they ran only when a human typed the command. Last recorded green was
2026-09-18, five days and fourteen commits earlier. The 2026-09-21 evaluation
says they run "on a weekly cron", which was wrong.

**One attempt each cannot separate the two failures that matter.** Three of the
eight depend on the model *choosing* to call a tool:

| check | shape | failure mode at n=1 |
|---|---|---|
| `filesystem_tools_still_work` | positive: `write_file` must be called and succeed | flakes **red** — model declines, reads as a regression |
| `permissions_are_enforced` | positive: a denial must appear in a tool message | flakes **red** — same |
| `shell_tool_withheld` | negative: `execute` must be absent | passes **vacuously** — a model that called nothing satisfies it |

The first two turn a non-deterministic model into a red build with no error
bars. The third is the opposite and worse: it is this repo's own
*assert-the-claim-and-its-discriminator* rule unapplied on the live path. The
graph-level `require_withheld` proves the binding; the live half's only job is
"no run can call it", and a turn that did nothing is not evidence for that.

### What we do

`CheckOutcome` aggregates k attempts of one check. **The exit code reads pass^k**
— these are invariants, and a check that held four times in five did not hold.
pass@k is printed alongside it, because it is the entire difference between
"this is broken" and "this is non-deterministic", and that difference does not
exist at one attempt.

`LIVE_CHECK_REPEATS` is 5, uniform across all eight. Uniform is deliberate and
temporary: **which checks can actually vary has not been measured**, and
declaring five of them deterministic would be the same untested assumption the
repeats exist to remove. `check_every_provider_serves_the_context_we_assume` is
a catalogue `GET` with no inference and almost certainly cannot vary — that is a
prediction, and the first run is what settles it. Pin per-check counts once
there is a number.

`_shell_withheld(called, answered)` is the discriminator, extracted as a pure
function so it is testable offline; the check bodies build their own model and
agent, so nothing else in them is.

`live.yml` gained a second step running `uv run my-agent` with
`LIVE_CHECK_REPEATS: "5"` stated rather than inherited. Its `timeout-minutes`
went 20 → 45, and **that number is an estimate, not a measurement** — 40 live
turns of unrecorded duration plus a second hardcoded 300s weave flush (F23,
which this workflow now pays twice because it runs two processes). The 20 was
measured; replace the 45 with the observed figure after the first real run.

### First live run, 2026-09-23 — and it was not the predicted flake

**7/8 pass^5.** `filesystem_tools_still_work` came back **3/5**, and both lost
attempts were the same thing:

```
attempt 4: OpenAIRateLimitError: Error code: 429 - {'message': "We're experiencing
  high traffic right now! Please try again soon.", 'type': 'too_many_requests_error',
  'param': 'queue', 'code': 'queue_exceeded'}
attempt 5: (identical)
```

Three things this run settles, none of them the thing it was built to measure:

1. **The predicted flake did not happen.** The prediction was that the model
   would sometimes decline to call `write_file`. It called it every time it got
   a response. The observed variance is entirely provider-side.
2. **The repeats caused the failure they detected.** Both 429s were on
   attempts 4 and 5 of the *fourth* check — the deepest point of the run's
   request burst. At one attempt per check this never appeared. That is not an
   argument against repeats; it is the first real evidence about what this
   harness does under its own load, which n=1 structurally could not produce.
3. **`shell_tool_withheld` passed 5/5 with `answered=True; tools called: none`** —
   exactly the shape the new discriminator exists to distinguish from a vacuous
   pass. Before this change that line would have read as a pass either way.

`check_every_provider_serves_the_context_we_assume` emitted one weave trace for
five attempts, confirming it makes no inference call and cannot vary. That was a
prediction above; it is now measured.

### Open

- **Retry and backoff on a 429 is unhandled** — `ModelConfig.max_retries` is 2
  and the router still surfaced the error. Whether that is exhausted retries,
  a status code the client does not retry, or backoff too short for a
  `queue_exceeded` queue is not yet established. This is the live gap this run
  found.
- Whether 5 is the right k. It is the smallest count that distinguishes 4/5 from
  5/5 at a cost worth paying weekly, not a figure derived from an observed
  failure rate.
- Whether the remaining six checks vary at all. Five of five on one run is not
  evidence that they cannot.

---

## Observability API reference

Not findings — API surfaces recorded so the next piece of work does not have to re-derive them.
Verified against langsmith 0.12.6 and weave 0.53.9.

**LangSmith is ambient.** Set `LANGSMITH_TRACING=true` (exactly, F13), `LANGSMITH_API_KEY` and
`LANGSMITH_PROJECT` and the whole LangChain/LangGraph/deepagents stack traces with no code change.

- Evals: `langsmith.evaluate(target, /, data=, evaluators=, summary_evaluators=, max_concurrency=,
  num_repetitions=, upload_results=, blocking=)`, plus `aevaluate`. `upload_results=False` runs an
  eval fully locally.
- The `langsmith_plugin` pytest plugin is installed via entry point, so `@pytest.mark.langsmith` and
  `from langsmith import testing as t` (`log_inputs`, `log_outputs`, `log_reference_outputs`,
  `log_feedback`, `trace_feedback`) work with no extra config. `LANGSMITH_TEST_SUITE` names the
  dataset a test file writes to. Nothing in this repo uses these yet.

**Weave.** `weave.init(project_name, *, settings=, autopatch_settings=, postprocess_inputs=, ...)`.

- `weave.trace.context.weave_client_context.get_weave_client() -> WeaveClient | None` reads back the
  installed client without triggering a new `weave.init()`. `WeaveTracing.activate()` uses it to
  make re-activation idempotent.
- Its LangChain integration is gated on the `WEAVE_TRACE_LANGCHAIN` environment variable and
  installed via `register_configure_hook` (`weave/integrations/langchain/langchain.py`). **A Weave
  client can exist with no LangChain hook installed** — which is why `WeaveTracing.activate()`
  checks `langchain_tracer_names()` for `"WeaveTracer"` rather than just checking the client is
  non-`None`.
- Evals: `weave.Evaluation(dataset=, scorers=, trials=, ...)` with `await
  evaluation.evaluate(model)`. Scorers are `@weave.op`-decorated callables taking `output=` as a
  named argument, and the dataset dict keys must match both the scorer args and the model's
  `predict`/`infer`/`forward` parameters.
- **Gotcha:** some `weave.init(settings=...)` values (e.g. `print_call_link`) are evaluated on
  Weave's background thread pool and silently ignore the `settings=` argument. Configure those with
  `WEAVE_*` environment variables instead.

---

## deepagents API surface

Read from the installed wheel, not from docs. CLAUDE.md carries `create_deep_agent`'s signature and
the constraints that bite in practice; this is the rest.

**`create_deep_agent`** takes positional `model` and `tools`; everything else is keyword-only:
`system_prompt, middleware, subagents, skills, memory, permissions, backend, interrupt_on,
response_format, state_schema, context_schema, checkpointer, store, debug, name, cache`. It returns
a `CompiledStateGraph`. `KNOWN_CREATE_DEEP_AGENT_PARAMS` pins that set at 18 (F19).

**`deepagents.backends.BackendProtocol` is an `abc.ABC`, not a `typing.Protocol`** (F6). Its MRO is
`(BackendProtocol, ABC, object)` and it requires 9 sync + 9 async methods. Shipped implementations:
`FilesystemBackend`, `StateBackend`, `StoreBackend`, `CompositeBackend`, `LocalShellBackend`,
`ContextHubBackend`, `LangSmithSandbox`.

**`SubAgent`** is a `TypedDict`; required keys are exactly `{name, description}`. Optional: `tools,
model, middleware, interrupt_on, skills, permissions, response_format, system_prompt, mode` (`mode`
is `Literal["isolated", "fork"]`). **`CompiledSubAgent`** requires `{name, description, runnable}`
and does **not** inherit `state_schema` from the parent.

**`interrupt_on=...` needs a `checkpointer`; `StoreBackend` needs a `store`.** Neither errors
usefully without one.

## Test-infrastructure specifics

**The socket guard was connect-shaped, and three exits are not.** `_forbid_network` patched
`socket.socket.connect`, `connect_ex` and `socket.create_connection` — every TCP path, including
async and TLS ones, bottoms out in the first of those, so anything that opens a *connection* was
caught. What was not: `socket.getaddrinfo` and `socket.gethostbyname` (a name lookup is egress on
its own and never calls connect) and `socket.socket.sendto` (a datagram goes on the wire with no
connection to intercept). All three now deny, and `tests/test_conftest.py` asserts each one —
watched red against the old three-patch guard before the fix, which is the only reason to believe
they can fail. Before that file existed the guard had **no tests at all**, so "offline is enforced,
not assumed" was itself an assumption.

Deliberately *not* covered: an exception raised inside a fire-and-forget `asyncio.create_task`. The
guard fires and the traceback prints, but an unretrieved task exception goes to
`loop.call_exception_handler` → `logger.error`, which is logging rather than a warning, so
`filterwarnings = ["error"]` never converts it and the test still passes. That needs an asyncio
exception handler or an `asyncio.all_tasks()` assertion at teardown, and it is unreachable today
because nothing in `src/` is async. **It is the first thing to fix if async ever lands** — before
the first async test, not after.

**A mutant on a pinned bound fails collection, not the test it targets — and that hides whether the
test itself discriminates.** `_PINNED_FS_BOUNDS` (`capabilities.py:235-264`) asserts
`GREP_MATCH_LIMIT`, `TOOL_RESULT_TOKEN_LIMIT` and `HUMAN_MESSAGE_TOKEN_LIMIT` each equal
`FilesystemMiddleware`'s own default at import, so `sed`-ing any one of them makes the whole file
error out before a single test runs — a stronger failure than the assertion it was meant to trip,
but a different one. To mutation-verify a test built on one of these bounds, the pin has to be
relaxed in the same mutation: `sed -i '' 's/_FS_SIGNATURE\[_name\].default == _pinned/True/'
src/my_agent/capabilities.py`, alongside the constant change — never committed, verification only.
And because all three constants equal the library's own defaults, a behavioural test built on one of
them proves the *mechanism* fires but cannot tell "we chose this value" from "deepagents' default
did" — that claim is carried by the call-recording tests instead (F21), not by driving a real agent
through the bound.

**The `src/` doctests run outside the socket guard.** `--doctest-modules` is set and `src` is a
`testpaths` entry alongside `tests`, but a `conftest.py` is directory-scoped, so those items get
neither `_forbid_network` nor any other `tests/` fixture. Confirmed with `pytest --setup-show`: a
doctest item lists no `_forbid_network`, a `tests/` item does. Harmless today — both doctests are
pure arithmetic over `require()` and `bounded()` and touch no I/O. If a `src/` doctest ever does,
the fix is a small `src/conftest.py` re-exporting the guard, **not** moving `conftest.py` to the
repo root, which would change what applies to every test in the project.

**The doctests are written as `try/except` + `print`, not as `Traceback` blocks**, because the
expected exception line differs between runners: pytest imports the module as
`my_agent.negative_space`, `python -m doctest` as `negative_space`. The `try/except` form also
asserts the message exactly, where `...` elides it.

**`monkeypatch.delenv(name, raising=False)` on a name that is absent records no undo**, so a
variable the code under test writes afterwards survives into the next test. `setenv` then `delenv`
when a test needs the name absent *and* the code under test will set it. Verified with a two-test
probe, not assumed.

---

## Repo gates: what is checked, and why each one is shaped that way

Not library behaviour, but the same kind of fact: verified, load-bearing, and easy to break by
tidying. CLAUDE.md carries the operational summary; this is the reasoning behind it.

**The gate sequence lives in `scripts/check.sh` and nowhere else.** The pre-commit hook and
`.github/workflows/ci.yml` both call it and re-list nothing, because three copies of a command list
is three things to forget. `-m live` and `-m eval` stay out: they need the network and cost money.
`check.sh` runs `uv lock --check` first, because every step below it is `uv run`, and `uv run`
silently re-locks a stale lockfile as a side effect — that would let the gate itself mutate
`uv.lock` and say nothing, and CI (which runs `uv sync --locked`) would be the first to notice.
Coverage is reported, not gated: `[tool.coverage.report]` sets no `fail_under`, deliberately,
because a coverage floor nobody agreed to is a floor someone lowers.

**The pre-commit hook is configured but not installed.** `uv run pre-commit install`; re-run it in a
fresh clone. It adds file-level hooks the gate should not do — `check-toml`, `check-yaml`,
`detect-private-key`, `check-added-large-files`, `check-merge-conflict`, `actionlint`.
`trailing-whitespace` and `end-of-file-fixer` are deliberately absent: they rewrite files
mid-commit. **In a worktree, do not install it** — `git rev-parse --git-common-dir` resolves to the
main checkout's `.git` and hooks live there, so installing from a worktree makes every commit in the
main checkout run a `scripts/check.sh` that may not exist on the branch checked out there.

**Three scans run on this repo and only two are files here.** `ci.yml` runs the gate on every push
and pull request. `.github/workflows/live.yml` runs `-m live` weekly (Mondays 06:00 UTC) and on
demand, because this file is forty-five verified behaviours and nothing else re-checks any of
them; it skips rather than fails when `HF_TOKEN` is absent, so an unconfigured clone does not
produce a weekly red X that means nothing, and it never gates a commit. **CodeQL is the third and it
is not a file here** — it uses GitHub's default setup, so the workflow is generated and managed by
GitHub and appears under Security, not in the repo tree. Reading `.github/workflows/` will not tell
you it exists. It analyses `python` and `actions`; the second earns its place because `ci.yml` and
`live.yml` are hand-written, and workflow-specific defects (script injection through untrusted
interpolation, an over-broad `GITHUB_TOKEN`, an unpinned action) are invisible to every other check
here.

**The required checks are named `gate`, `Analyze (python)` and `Analyze (actions)`.** Those exact
strings are what the ruleset matches, and CodeQL's are *not* "CodeQL" — check names come from the
job, not the workflow. CodeQL's `Adjust Configuration` check is deliberately **not** required: it
reports `skipped`, and a required check that never concludes can never be satisfied, which would
silently make every pull request unmergeable. The same trap applies to renaming the `gate` job in
`ci.yml`: the rule must change in the same commit.

**`main` is protected, so nothing lands without a green gate.** The ruleset requires a pull request,
requires the branch to be up to date with `main` first, and blocks force-pushes and branch deletion.
**A direct `git push` to `main` is rejected** — work goes on a branch and merges through a PR, which
is also the only path that runs the gate *before* the merge rather than after it. Approvals are set
to zero on purpose: a solo author cannot approve their own pull request, so requiring one would mean
bypassing the rule every time instead of satisfying it.

**`dependabot.yml` covers `github-actions` and `uv`, weekly.** The first is not optional
housekeeping: `ci.yml` pins `astral-sh/setup-uv` to a full commit SHA because that action publishes
no major tags, and a SHA pin cannot self-update. A Dependabot PR that moves a dependency is the
prompt to re-verify CLAUDE.md's version block — not a reason to skip that step.

---

## Live verification

`uv run my-agent` runs eight checks against the real router and prints PASS/FAIL. Each is tied to
a finding; seven findings are covered, not all forty-five. Each runs `LIVE_CHECK_REPEATS` times
and the run is scored pass^k (F45). The 8/8 below is from 2026-09-18, exit 0, both tracers
active — **at one attempt each**, before the repeats existed:

| Finding | Check | Evidence |
|---|---|---|
| F1 | chat-completions endpoint reachable | `reply='pong'` |
| F2 | token cap honoured | asked ≤24, produced 24 |
| F4 | shell tool withheld | unbound; no `execute` call in a run that asked for one |
| F4 | remaining filesystem tools usable | `write_file`, `read_file` both succeeded |
| F5 | permission rules survive replacement | `write_file` denied on `/secrets/**` |
| F26 | `reasoning_effort` set matches the router | accepted low/medium/high; `xhigh` rejected |
| F27 | reasoning is a count, never text | 44 reasoning tokens, no content keys |
| F25 | every provider that states a context window meets our floor | shortest 128072 across 9; featherless-ai and scaleway state nothing |

The F25 check found a real divergence on its first run and is the reason that finding exists. It
asserts only on providers that publish a length — the two that publish nothing are an unknown a
check cannot close, and the note in its output says what does (pin `:provider`).

The checks assert on tool messages and token counts rather than model prose, so they do not depend
on how the model phrases things — which F27 turned from good practice into the vendor's own
guidance. The F5 check was itself verified by breaking the fix and watching it go red.

**`uv run pytest -m live` can hang *after* reporting its result — do not pipe it to `tail`.** The
wait is now explained and bounded at 300s (F23), but the piping lesson stands: `pytest ... | tail;
echo $?` reports **`tail`'s** exit status, not pytest's, and the summary line is discarded while
background-thread noise is kept, so the run looks like a clean exit with no result. Redirect
instead:

```bash
uv run pytest -m live -q > live.log 2>&1; echo "exit: $?"   # not `| tail`
```

The tests themselves pass; only process exit is affected. Run a single live test by node id when one
is all you need — it reports in ~2s and the wait costs nothing but the wait.

Every finding added since F11 is pinned offline instead, and each was verified by mutation — the
change reverted in place and the test watched go red. **The ledger immediately below stops at
F24/F29; `feat/behavioural-tests` (below that) extends it to `GREP_MATCH_LIMIT`,
`TOOL_RESULT_TOKEN_LIMIT`, `HUMAN_MESSAGE_TOKEN_LIMIT`, the compaction bound (F31/F38) and
`TurnResult.answered` (F36). Still not through this step**: the rest of F30 (call limits, request
size), `RunTokenBudget` (F35) and `failed_tool_calls` outside what F24/F29 already cover; those
findings' tests exist and several record pre-fix measurements in their own docstrings, but nobody
has reverted the fix and watched them go red.

Fourteen mutants, all killed: a silent `{}`
from `subagent_graphs`, a dropped subagent loop, a removed vacuity guard, `execute` back in the
allowlist, an unsent step limit, an unattached deadline, a no-op deadline check, `raise_error =
False`, a fixed rather than relative message postcondition, and — added with F24 and F29 — a dropped
subagent step-limit rebind, an untranslated `GraphRecursionError`, an unchecked unanswered tool
call, a `paused` property hardwired to `False`, and a dropped checkpointer dead-end check. One
survived and is written up in F21.

**`feat/behavioural-tests` added roughly eight more, recorded only in the branch's own commit
messages** (not restated as source comments, so cited here instead): `GREP_MATCH_LIMIT` raised and
lowered, both caught by the import-time pinned-bound `require()` before either target test could run
(`4aa81fb`); `TOOL_RESULT_TOKEN_LIMIT` raised and lowered, same mechanism (`dba5410`);
`HUMAN_MESSAGE_TOKEN_LIMIT` raised, same mechanism (`77f5a90`); the compaction discriminator's own
fixture, undersized enough to pass under every trigger tried until this branch resized it
(`d3ae394`); and `TurnResult.answered` hardwired to `True` (`eab87ca`) and to `False` (`3fe212a`).

## Open / unverified

Each of these is also noted at the finding it belongs to.

- Whether anything consumes `TurnResult.files` (F40). The field exists and is tested; `main.py`
  prints `failed_tool_calls` and would be the obvious next reader, and is deliberately left alone.
- Whether a `wrap_model_call` middleware installed by deepagents can be read back off a compiled
  graph at all (F41). The closure hunt that finds `subagent_graphs` finds no middleware list, so
  the caching door is pinned before the fact rather than asserted after it — the only door here
  with no read-back.
- ~~Whether the import-time check sites can be tripped by a test.~~ **Closed by F44.** Re-measured
  2026-09-23 by wrapping `require()` and `CheckFailed` in pytest plugins that log a raising call
  site, then diffing against an AST walk: **102 of 112 `require()` sites and 12 of 13
  `raise CheckFailed` sites outside the helpers are tripped**, and all 11 untripped sites are in
  `main.py`, deliberately left. No import-time check is untripped.

- A **provider-pinned** model id (`org/model:groq`) on the token cap (F2), on `reasoning_effort`
  (F17, F26) and on the context window (F25). Only the router's own selection is covered, and F25 is
  the finding that makes pinning worth doing rather than merely tidy.
- What `featherless-ai` and `scaleway` actually serve as a context window (F25). They publish
  nothing; measuring it means sending a prompt and watching it truncate.
- Whether any new protocol with a non-method member diverges between mypy and pyright (F16).
- Which `create_deep_agent` omissions `AgentConfig` considered and rejected (F19).
- Whether a nested subagent would inherit `SUBAGENT_STEP_LIMIT` (F24). None can dispatch one today.
- Whether HITL behaves the same against the live router as against the fake model it is proven with
  (F29). The pause is a middleware decision taken before the model is called again, so it should —
  but "should" is what this file exists to replace.
- Whether every provider frames a harmony response identically, and whether a `reasoning: 0`
  response is a model decision or a provider one (F36). Two fingerprints in one run agreed on the
  frame; the routed provider is not recorded, so that is agreement between unknowns.
- Whether the router's other ten providers volunteer token usage on a streamed response (F37). The
  one that answered on 2026-09-19 does, which is the only reason streaming would not blind
  `RunTokenBudget`; it is not recorded which provider that was.
- Whether a real conversation ever produces a large message that is neither last nor old enough to
  compact (F38). Compaction, human-message eviction and the tool-result character bound have each
  been measured individually; none of the three has been measured against a real conversation, so
  the gap between them is proven in isolation, not in use.
