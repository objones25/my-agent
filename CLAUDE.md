# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deep-agent built on `deepagents`, driven by TDD, evals, and observability rather than by
feature count. The agent's **domain is deliberately undecided** — see "Scope discipline". The
thing being built right now is the *harness*: contracts, model wiring, observability seams, eval
plumbing, and the test infrastructure that keeps them honest.

Model access goes through the Hugging Face Inference Providers router using `langchain-openai`'s
OpenAI-compatible client. Observability is a mix of LangSmith and W&B Weave, each behind its own
narrow protocol so neither is load-bearing.

## Commands

```bash
uv sync                                   # install/refresh the locked environment
uv run pytest                             # unit tests (live + eval cases deselected)
uv run pytest tests/test_x.py::test_y     # a single test
uv run pytest -m live                     # tests that hit the real HF router / LangSmith / W&B
uv run pytest -m eval                     # the eval suite
uv run pytest --cov                       # coverage report
uv run my-agent                           # live checks: one per finding in docs/findings.md
uv run my-agent "your prompt here"        # one ordinary turn instead
uv run ruff check . --fix                 # lint
uv run mypy                               # type check (strict; src + tests)
uv run python -m doctest src/my_agent/negative_space.py   # contract helpers' doctests

# Negative-space audit. First command is the CI gate, second is advisory.
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007
uv run python scripts/audit_negative_space.py src/ --select NSP001 --min-assertions 2 || true
```

`uv run pytest` exits 5 ("no tests collected") until the first test exists — that is not a pass.

## Non-negotiables

**Never write an API call from memory.** These libraries move faster than any training cutoff, and
this project pins recent majors (`langchain` 1.x, `openai` 3.x, `pytest` 9.x, `mypy` 2.x) where
most published examples are still on the previous one. Before writing code against any library:

1. `npx ctx7@latest library "<Official Name>" "<the actual question>"` → pick the `/org/project` ID.
2. `npx ctx7@latest docs <id> "<the actual question>"`.
3. **Then confirm against the installed code**, because docs lag the wheel:
   ```bash
   uv run python -c "import inspect, X; print(inspect.signature(X.thing))"
   uv run python -c "import X; print(sorted(n for n in dir(X) if not n.startswith('_')))"
   uv run python -c "import X; print(X.Model.model_fields.keys())"   # pydantic models
   ```

ctx7 tells you the intent; `inspect` tells you the truth. When they disagree, `inspect` wins and
the disagreement is worth a note in this file. A signature recorded below without a version tag is
a bug in this file.

## Least privilege

**A capability the agent has not been given cannot be misused.** Every tool, permission, filesystem
path, and network reach is off unless something needs it, and turning one on is a deliberate,
reviewed edit — never something inherited from a library default.

This is not theoretical here: `create_deep_agent` enables a shell `execute` tool with no opt-in
(`docs/findings.md` F4). `capabilities.py` withholds it and `build_agent` installs the
result. Concretely:

- `DEFAULT_FILESYSTEM_TOOLS` is defined by *subtraction* from what deepagents offers, and a
  load-time check asserts it equals `get_args(FsToolName) - {"execute"}`. A new upstream tool fails
  the import rather than being granted silently.
- `AgentConfig.middleware` and `.permissions` default to empty. Empty permissions means "no rules",
  not "deny all" — if the threat model needs denial, write the rules.
- Postconditions assert the withheld capability is actually absent from the compiled graph (and
  that the graph bound *some* tools, so the absence cannot pass vacuously) — an allowlist is only
  a request until it is checked.

When adding a capability, say in the commit message what needs it and what the blast radius is.

## Scope discipline (YAGNI)

The domain is TBD on purpose. Do not invent tools, retrievers, or subagents to "make it useful".
Build a seam, prove it with a test, stop. One trivial placeholder tool is enough to exercise the
harness. When a real domain is chosen, it should slot in behind the existing protocols without
any existing file changing.

`deepagents` already ships filesystem tools, shell execution, and subagent delegation. Do not
reimplement them. The default toolset in 0.7.15, read off the compiled graph, is exactly:

    delete  edit_file  execute  glob  grep  ls  read_file  task  write_file

Two things about that list are easy to get wrong:

- **There is no `write_todos`.** deepagents 0.7.15 ships no planning tool and no `TodoListMiddleware`
  — that class lives in `langchain.agents.middleware`, and reaches the agent through the
  `middleware` parameter. Docs and skills that describe `write_todos` as always-present are stale.
- **`execute` runs shell commands and is on by default**, with no opt-in. Decide sandboxing
  deliberately (`backend`, `permissions`, `interrupt_on`) rather than inheriting it.

`tests/test_main.py` pins this list, so an upstream change fails a test instead of quietly
altering behaviour.

## Architecture: protocol-driven contracts

Every seam is a **narrow** protocol — one consumer's needs, not one implementation's surface. A
component depends on the smallest interface that covers what it actually calls, so swapping an
implementation means writing a new class and changing one line in the composition root.

Planned seams (build them as they are needed, not before):

| Protocol | Method(s) | Why it is separate |
|---|---|---|
| `ChatModelSource` | `build() -> BaseChatModel` | Agent construction must not know about HF, base URLs, or tokens. |
| `TracingBackend` | `activate() -> None` | Tracing install is idempotent and global; callers only need "turn it on". |
| `EvalRunner` | `run(target, dataset, scorers) -> EvalReport` | Eval consumers never emit spans; tracing consumers never score. Keeping these apart is the point. |

`TracingBackend` and `EvalRunner` are deliberately **not** one `Observability` interface. LangSmith
and Weave both activate ambiently (env var / `weave.init`) but score through unrelated APIs, and
most components need exactly one of the two.

### Configs are parameter objects, not argument lists

A factory takes **one config object**, never a growing list of keyword parameters. Each config's
field names are exactly the callee's parameter names, and the factory splats them:

```python
model = ChatOpenAI(**config.as_kwargs(), use_responses_api=USE_RESPONSES_API)
agent = create_deep_agent(model=model, **agent_config.as_kwargs())
```

`create_deep_agent` has 17 keyword parameters. Threading them through `build_agent` one at a time
would mean editing its signature *and* its body every time the agent gains `subagents`, `skills`,
`backend`, `permissions`, or `interrupt_on` — and defaults do not help, because the edit is still a
modification of a function that was supposed to be closed. With a parameter object, adding a
setting is one new field on `AgentConfig` with a default: `build_agent` does not change, and no
call site changes.

The cost is that `as_kwargs()` splatting trusts field names to be real parameters, and a typo would
otherwise surface as a `TypeError` from inside the library — or vanish into a `**kwargs` signature.
So `contracts.check_config_contract` verifies every field against the callee's actual parameters
**at import time**, via `inspect.signature(create_deep_agent)` and `ChatOpenAI.model_fields` (pydantic's
`__init__` is `**data`, so its fields and aliases are the real contract). A misspelled field or an
upstream rename fails on import, by name. It also refuses fields the factory injects itself
(`model`, `use_responses_api`), which would collide on splat.

Values the router constrains — `use_responses_api` — stay constants injected by the factory, not
config fields, so they cannot be overridden by a caller.

Rules for protocols here:

- Use `typing.Protocol` for seams this project defines. Structural typing means an implementation
  need not import the protocol, which is what keeps adapters decoupled.
- `Protocol` enforces nothing at instantiation — a class inheriting one and implementing nothing
  instantiates fine and the missing method returns `None`. Where that silence would be a production
  incident, use `abc.ABC` instead and let `TypeError` fire.
- `@runtime_checkable` checks method *presence* only, never signatures, and `isinstance` against it
  is slow. Do not put one in the agent loop.
- Extend by adding a new protocol, not by widening an existing one. A consumer that needs more
  gets its own narrower contract.

**Composition root.** Concrete classes are chosen in exactly one place — the app/CLI entry point.
Nothing below it constructs a `ChatOpenAI`, reads `os.environ`, or imports `weave`. If a test has
to set an env var to reach the line under test, the wiring is in the wrong place.

## Negative space programming

Bugs live in the states the code was never written to handle. Write those down as executable checks.

- `src/my_agent/negative_space.py` holds `require()`, `unreachable()`, `bounded()`, `check_shape()`,
  `check_finite()`. Use these, not bare `assert` — `python -O` deletes `assert` statements entirely,
  condition and message both, and some container images set `PYTHONOPTIMIZE`.
- Write preconditions before the body and postconditions after it. Assert the positive space *and*
  the negative space: not just "k is in range" but "the two halves do not overlap".
- **`require()` for programmer errors** (a caller you own passed something impossible) — crash.
  **`raise ValueError`/typed exceptions for operating errors** (env var missing, router returned 503,
  malformed model output) — handle at the edge. Same predicate, different category, decided by where
  the value came from. Model output is *always* an operating error: it is untrusted input.
- Every loop, retry, and agent turn gets an explicit bound. An agent that loops forever is the
  worst failure mode here; `bounded()` exists for exactly this.
- Split compound checks: `require(a); require(b)` names the failure, `require(a and b)` does not.
- mypy cannot narrow types through `require()`. Where a check also narrows (`x is not None`), use an
  explicit `if ... raise CheckFailed(...)` — same runtime behaviour, and mypy follows it.
  `check_shape` in `negative_space.py` is the worked example.
- Plain `assert` stays correct in test bodies (pytest rewrites it for readable failures).

## Testing and evals

Two different things; keep them apart.

- **Unit tests** (`tests/`, default selection) are deterministic and offline. One test file per
  source module; a new module gets a new file rather than an extra section in an existing one.
  They test the harness: protocol conformance, wiring, bounds, error paths. For each `require()`,
  a test that trips it — that is what turns a contract into a tested contract.
- "Offline" is enforced, not assumed: an autouse fixture in `tests/conftest.py` fails any test
  that opens a socket, and steps aside only for `live`. Shared setup (`valid_secret`,
  `deny_secrets`) lives there too — as fixtures, so no test can leak a mutation into the next.
- **Evals** (`evals/`, `-m eval`) measure model-dependent behaviour and are allowed to be
  non-deterministic and slow. A failing eval is a signal, not a broken build.
- `-m live` marks anything touching the HF router, LangSmith, or W&B. `addopts` carries
  `-m "not live and not eval"`, so both are deselected by default and the suite stays free and
  fast; a command-line `-m live` overrides it. Registering a marker does *not* deselect it — that
  was a real gap here until it was measured.
- **A passing suite is not a passing state if the tests cannot fail.** Before trusting new tests,
  break the code they cover and watch them go red. The last split was verified with five such
  mutants; a test that survives one is decorative.
- Never run the suite under `python -O`: the assertions inside the tests vanish and everything passes.
- `filterwarnings = ["error"]` is set. A new deprecation warning from these fast-moving libraries
  fails the build on purpose — fix it or scope an ignore, do not widen the setting.

## Verified API facts

Recorded from `inspect` against the installed wheels on 2026-09-17. Re-verify after any `uv sync`
that moves these versions.

**`docs/findings.md` is the full record** — fourteen verified library behaviours (F1–F14), each with
how it was checked, what the code does about it, and what is still unverified. Read it before
debugging anything that looks like a library bug, and add to it when you verify something new. The
summary below covers only what is needed to write code day to day.

```
deepagents 0.7.15   langchain 1.4.1        langchain-core 1.6.3   langgraph 1.2.11
langchain-openai 1.6.2   langsmith 0.12.6   weave 0.53.9          openai 3.14.1
pytest 9.1.1        mypy 2.3.1             ruff 0.16.8            Python 3.13
```

**`langchain_openai.ChatOpenAI`, not `init_chat_model`.** `init_chat_model(model, *,
model_provider, configurable_fields, config_prefix, **kwargs)` infers a provider from the model
string; `org/model:provider` router ids are meaningless to it. Construct `ChatOpenAI` directly.
Its constructor takes *aliases*, not field names: `model` → `model_name`, `base_url` →
`openai_api_base`, `api_key` → `openai_api_key` (coerced to `SecretStr`), `timeout` →
`request_timeout`. `max_retries` and `temperature` have no alias.

**`deepagents.create_deep_agent`** — positional `model`, `tools`; everything else keyword-only:
`system_prompt, middleware, subagents, skills, memory, permissions, backend, interrupt_on,
response_format, state_schema, context_schema, checkpointer, store, debug, name, cache`.
Returns a `CompiledStateGraph`.

**`deepagents.backends.BackendProtocol` is an `abc.ABC`, not a `typing.Protocol`** — despite the
name. Subclass it explicitly; structural typing will not register. Its MRO is
`(BackendProtocol, ABC, object)`. It requires 9 sync + 9 async methods (`ls/read/write/edit/glob/
grep/delete/upload_files/download_files` and their `a`-prefixed twins). Shipped implementations:
`FilesystemBackend`, `StateBackend`, `StoreBackend`, `CompositeBackend`, `LocalShellBackend`,
`ContextHubBackend`, `LangSmithSandbox`.

**`SubAgent`** is a `TypedDict`; required keys are exactly `{name, description}`. Optional:
`tools, model, middleware, interrupt_on, skills, permissions, response_format, system_prompt,
mode` (`mode` is `Literal["isolated", "fork"]`). `CompiledSubAgent` requires
`{name, description, runnable}` and does **not** inherit `state_schema` from the parent.

**Known `deepagents` constraints** (these bite in exactly this order):
- `interrupt_on=...` needs a `checkpointer`.
- `StoreBackend` needs a `store`.
- `skills=[...]` needs a real backend (e.g. `FilesystemBackend`); it silently loads nothing otherwise.
- Skills are **not** inherited by subagents — pass `skills` on each subagent spec.
- A consistent `config={"configurable": {"thread_id": ...}}` is what makes turns share a conversation.

**Callbacks and tracing** (verified against langchain-core 1.6.3, langsmith 0.12.6, weave 0.53.9):

- `BaseCallbackHandler.raise_error` and `.run_inline` both default to `False`. A hook that raises
  is caught by `CallbackManager`'s dispatch and, unless `run_inline=True`, may run off the main
  thread — so a broken handler degrades silently by default. `JsonlMirror` overrides both (F12).
- `CallbackManager.configure(...) -> CallbackManager` (all args optional) is how to read back which
  tracers are actually installed on a run right now: `{type(h).__name__ for h in
  CallbackManager.configure().handlers}`. Builds and discards a manager; no network, no side effects.
- `langsmith.utils.tracing_is_enabled(ctx: dict | None = None) -> bool | Literal["local"]` is the
  literal-`"true"`-comparison function behind LangSmith's ambient activation (F13).
- `langsmith.utils.get_env_var` is `@functools.lru_cache(maxsize=100)`-wrapped with signature
  `(name, default=None, *, namespaces=("LANGSMITH", "LANGCHAIN"))` (F13). mypy sees it as an
  `Overload` over the cached wrapper rather than a single `_lru_cache_wrapper`, so it does not see
  `.cache_clear` even though it exists at runtime — every call site needs
  `# type: ignore[attr-defined]`, confirmed present and working via `inspect` on the installed wheel.
- `weave.trace.context.weave_client_context.get_weave_client() -> WeaveClient | None` reads back
  the installed client without triggering a new `weave.init()`; `WeaveTracing.activate()` uses it
  to make re-activation idempotent.
- Weave's LangChain integration is gated on the `WEAVE_TRACE_LANGCHAIN` environment variable and
  installed via `register_configure_hook` (`weave/integrations/langchain/langchain.py`) — a Weave
  client can exist with no LangChain hook installed, which is why `WeaveTracing.activate()` checks
  `langchain_tracer_names()` for `"WeaveTracer"` rather than just checking the client is non-`None`.

### Hugging Face router via `langchain-openai`

Base URL `https://router.huggingface.co/v1`, auth via `HF_TOKEN`. On `ChatOpenAI`, `base_url` is the
alias of field `openai_api_base` and `api_key` the alias of `openai_api_key` — pass the aliases.

Model IDs are `org/model`, optionally suffixed to steer routing: `:provider`, `:fastest`,
`:cheapest`. Pin a provider when an eval needs to be reproducible; leave it off to let the router pick.

`https://api.endpoints.huggingface.cloud/` is a *different* API — the Inference Endpoints control
plane for creating and managing dedicated deployments, not an inference base URL. A dedicated
endpoint serves at its own `https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1/`. Both
are OpenAI-compatible, so switching is a `ModelConfig.base_url` change and nothing else.

**Request-body names match, with one exception.** The router's documented Chat Completions body
uses the standard OpenAI names, and so does langchain: `temperature`, `top_p`, `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `logprobs`, `tools`, `tool_choice`, `response_format`,
`stream`. No `extra_body` needed for any of those.

The exception is **`max_tokens`**. HF documents `max_tokens`; `ChatOpenAI` renames it to
`max_completion_tokens` in *two* places — `ChatOpenAI._default_params` and
`ChatOpenAI._get_request_payload` ("deprecated in favor of max_completion_tokens"). The field
alias runs the same direction, so there is no way to send `max_tokens` through the normal field.
`ModelConfig` deliberately has no token-cap field. If one is ever needed, send it as
`extra_body={"max_tokens": N}` and leave the `max_tokens` field unset — and verify against the
router with a `-m live` test first, since whether a given provider also accepts
`max_completion_tokens` is unverified here.

- Default: `openai/gpt-oss-120b` — 11 live providers (groq, cerebras, together, fireworks-ai,
  novita, nscale, featherless-ai, scaleway, ovhcloud, deepinfra, baseten). Smaller and faster, and
  a harder test of the harness.
- Alternate: `deepseek-ai/DeepSeek-V4-Flash` — 3 live providers (novita, featherless-ai, deepinfra).

**Pin `use_responses_api=False` explicitly; do not leave it unset.** The router serves
`/v1/chat/completions` only — the Responses API's server-side state params (`store`,
`previous_response_id`) are unsupported there, and replaying reasoning or `mcp_list_tools` items
fails. The default `None` does *not* mean "Chat Completions": `BaseChatOpenAI._use_responses_api`
infers the endpoint from the model name (`_model_prefers_responses_api`, which matches any id
containing `codex`) and from the payload (`reasoning`, `include`, `truncation`, `text`,
`context_management`, or any builtin tool), **independent of `base_url`**. langchain-openai's own
`ChatOpenAI` docstring carries a warning to set this explicitly for OpenAI-compatible providers.
`model.py` pins it via the `USE_RESPONSES_API` constant and asserts it as a postcondition.

### Observability

**LangSmith** is ambient: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
and the whole LangChain/LangGraph/deepagents stack traces with no code change.
`LANGSMITH_TRACING` must be the exact literal string `"true"` — `langsmith.utils
.tracing_is_enabled()` compares it directly, with no stripping or case-folding, so `True`, `1`,
`yes`, `on` all leave tracing off (F13; this is live in this repo's own `.env` right now).
`langsmith.utils.get_env_var` searches `namespaces=("LANGSMITH", "LANGCHAIN")` by default, so the
older `LANGCHAIN_*` names still enable tracing and still resolve the project — prefer
`LANGSMITH_*` for new work, but do not assume the legacy prefix is inert. That function is also
`@functools.lru_cache`d, so a lookup made before `load_dotenv()` runs is remembered for the life
of the process (F13). Eval API: `langsmith.evaluate(target, /, data=, evaluators=,
summary_evaluators=, max_concurrency=, num_repetitions=, upload_results=, blocking=)`, plus
`aevaluate`. `upload_results=False` runs an eval fully locally.

The `langsmith_plugin` pytest plugin is installed via entry point, so `@pytest.mark.langsmith`,
`from langsmith import testing as t` (`log_inputs`, `log_outputs`, `log_reference_outputs`,
`log_feedback`, `trace_feedback`) work with no extra config. Set `LANGSMITH_TEST_SUITE` to name the
dataset a test file writes to.

**Weave**: `weave.init(project_name, *, settings=, autopatch_settings=, postprocess_inputs=, ...)`.
It ships a LangChain autopatch integration (`weave.integrations.langchain`) that installs a
`WeaveTracer` callback globally. Evals are `weave.Evaluation(dataset=, scorers=, trials=, ...)` with
`await evaluation.evaluate(model)`; scorers are `@weave.op`-decorated callables taking `output=` as a
named argument, and the dataset dict keys must match both the scorer args and the model's
`predict`/`infer`/`forward` parameters.

**Weave gotcha:** some `weave.init(settings=...)` values (e.g. `print_call_link`) are evaluated on
Weave's background thread pool and silently ignore the `settings=` argument. Configure those with
`WEAVE_*` environment variables instead.

**Running both at once is verified end to end (F11).** `available_backends()` activates every
configured backend rather than selecting one; a live run confirmed `langchain_tracer_names()`
reports both `LangChainTracer` and `WeaveTracer` before and after a real agent turn, and neither
backend's global install displaced the other's. `uv run pytest -m live` completes and exits `0`.
Getting there needed two fixes for warnings that `weave.init()` raises from inside its own SDK, not
from anything this repo calls directly — an old-style `gql` call (scoped out with a per-test
`@pytest.mark.filterwarnings` mark, since it fires synchronously inside the test) and an
unclosed-socket resource leak that pytest can report either during a test or at the pytest
*session's* teardown (`pytest_unconfigure`, unreachable by any per-test mark — scoped out with a
`pyproject.toml` `filterwarnings` entry instead, added alongside `"error"`, matched on category +
message + originating module so it cannot mask an unrelated warning). Both orthogonal to
coexistence and fatal only because of this project's `filterwarnings = ["error"]`; see F11 for the
full mechanism, including why the more obvious fix (`WeaveClient.finish()`) was tried and reverted
(it can hang on a stuck send queue).

## Repo layout

```
src/my_agent/
  model.py            # ModelConfig, build_model, router/model defaults, USE_RESPONSES_API.
                      # The only module that reads os.environ or knows the router exists.
  agent.py            # AgentConfig, build_agent. Takes a BaseChatModel; imports nothing
                      # from model.py — main.py is the only place the two meet.
  capabilities.py     # DEFAULT_FILESYSTEM_TOOLS, least_privilege_filesystem,
                      # compiled_tool_names. The allowlist and the proof it held.
  contracts.py        # check_config_contract, pydantic_param_names — the import-time
                      # check that makes as_kwargs() splatting safe.
  main.py             # `uv run my-agent` — composition root. Live checks against the
                      # router, one per finding.
  negative_space.py   # contract helpers: require/unreachable/bounded/check_shape/check_finite
  tracing.py          # TracingBackend protocol, LangSmithTracing, WeaveTracing,
                      # available_backends, langchain_tracer_names.
  mirror.py           # JsonlMirror, run_log_path, mirror_to_file — the local,
                      # always-on JSONL mirror of every agent event.
tests/                # deterministic, offline — one file per source module
  conftest.py         # shared fixtures + the autouse guard that blocks sockets
  test_model.py  test_agent.py  test_capabilities.py  test_contracts.py  test_main.py
  test_tracing.py  test_mirror.py
evals/                # model-dependent, -m eval
docs/
  findings.md         # F1-F14: verified library behaviour and what the code does about it
scripts/
  audit_negative_space.py   # VENDORED from the negative-space-programming skill; do not hand-edit.
                            # Refresh by re-copying from the skill; excluded from ruff and mypy.
```
