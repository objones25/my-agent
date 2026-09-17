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
- Whether LangSmith and Weave tracing can run simultaneously over the same deepagents run.
  LangSmith traces via env var and Weave installs a LangChain callback globally, so it *should*
  work — not confirmed end to end.
