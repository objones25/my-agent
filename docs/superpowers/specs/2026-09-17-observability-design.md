# Observability: tracing backends and a local JSONL mirror

Date: 2026-09-17
Status: approved, not yet implemented

## What this builds

Two new modules and the wiring that uses them:

- `src/my_agent/tracing.py` — the `TracingBackend` protocol and two adapters, LangSmith
  and W&B Weave. Both activate ambiently and process-wide.
- `src/my_agent/mirror.py` — `JsonlMirror`, a LangChain callback handler that writes every
  graph step, tool call and model call to a JSONL file, one file per process run. Local,
  offline, and independent of whether either SaaS tracer is on.

They do not import each other. One does global, irreversible, process-wide things; the
other is a plain object that writes to a stream. `main.py` is the only module that knows
both exist.

## Out of scope

`EvalRunner`. `evals/` is empty, so any protocol written now would be shaped against an
imagined consumer, and CLAUDE.md's argument for keeping `TracingBackend` and `EvalRunner`
apart applies to the order they are built in too. It gets its own spec when there is a
dataset worth scoring.

Also out of scope: log rotation or retention (`logs/` is gitignored and pruned by hand),
async callback hooks (nothing here runs the agent asynchronously yet), and any second
consumer of the mirror such as replay or diffing.

## Verified API facts

Read off the installed wheels on 2026-09-17 (langchain-core 1.6.3, weave 0.53.9,
langsmith 0.12.6). These are the facts the design rests on; re-verify after any `uv sync`
that moves them.

- `BaseCallbackHandler` carries two class attributes: `raise_error` (default `False`) and
  `run_inline` (default `False`). With `raise_error=False`, LangChain swallows exceptions
  raised inside a handler — a mirror that stopped mirroring would look identical to a quiet
  run.
- Weave installs its LangChain tracer through
  `langchain_core.tracers.context.register_configure_hook`, in
  `weave.integrations.langchain.langchain`. It is ambient and global, exactly like
  LangSmith's env vars — one `activate()` shape fits both.
- `langsmith.utils.tracing_is_enabled(ctx=None) -> bool | Literal["local"]` reports whether
  LangSmith tracing is actually on.
- `weave.trace.context.weave_client_context.get_weave_client() -> WeaveClient | None`
  reports whether `weave.init` has run.
- `weave.init(project_name, *, settings=, autopatch_settings=, postprocess_inputs=,
  postprocess_output=, attributes=, global_postprocess_inputs=, global_postprocess_output=,
  global_attributes=) -> WeaveClient`.
- `RunnableConfig` accepts `callbacks` alongside `recursion_limit`, so a handler reaches a
  compiled graph through the same config dict `_run` already builds.

## Architecture

### `tracing.py`

```python
class TracingBackend(Protocol):
    name: str
    def activate(self) -> None: ...
```

`typing.Protocol`, not `@runtime_checkable`. Conformance is checked by mypy at an
assignment in `tests/test_tracing.py` — strict mypy already covers `tests/` — which checks
signatures, where `runtime_checkable` would only check method presence. `name` is on the
protocol because the one consumer, `main`, prints what turned on.

Each adapter is a frozen slotted dataclass with:

```python
@classmethod
def from_env(cls, env: Mapping[str, str] | None = None) -> Self | None
```

following `ModelConfig.from_env`'s precedent: `env` is injectable so no test touches the
real process environment, and `os.environ` is read only when `main` passes nothing.
Returning `None` means *not configured*, which is a valid and silent answer — not an
operating error. Nothing here raises for absent tracing configuration.

```python
def available_backends(env: Mapping[str, str] | None = None) -> tuple[TracingBackend, ...]
```

collects the non-`None` adapters. `()` is a valid result and means no tracing. This is the
single function `main` calls.

#### `LangSmithTracing`

Available when `LANGSMITH_API_KEY` is present *and* `LANGSMITH_TRACING` is truthy. Carries
the project name from `LANGSMITH_PROJECT`.

`activate()` installs nothing, because there is nothing to install: the LangChain stack
reads the env vars itself. It *verifies* — it asserts `langsmith.utils.tracing_is_enabled()`
agrees with what `from_env` concluded. That is the whole value of the class: without it,
"no trace because there was no run" and "no trace because the variable was misspelled" are
indistinguishable.

Rejected alternative: having `activate()` set `LANGSMITH_TRACING=true` itself when only the
API key is present. Mutating `os.environ` from below the composition root is the coupling
CLAUDE.md bans, and it would start billing traces on a run that never asked to be traced.

#### `WeaveTracing`

Available when `WANDB_API_KEY` is present. Project name from `WEAVE_PROJECT`, defaulting to
`my-agent`.

`activate()` calls `weave.init(project_name)`, guarded for idempotency on
`get_weave_client() is not None`, with `get_weave_client() is not None` as the postcondition.
No `settings=` argument is passed: some of those values are evaluated on Weave's background
thread pool and silently ignore what was passed, so anything configurable there goes through
`WEAVE_*` environment variables instead.

`weave.init` touches the network. Every test of it either monkeypatches `weave.init` or is
marked `live`.

#### Failure policy

Activation failure is an *operating* error — W&B is down, a key is rejected. The agent still
works without a tracer, so `main` catches it, prints `tracing: weave FAILED (…)` loudly, and
continues. Losing a run's output to a telemetry outage would be the worse failure.

A `CheckFailed` from a postcondition is a programmer error and propagates, as everywhere else
in this repo.

### `mirror.py`

```python
class JsonlMirror(BaseCallbackHandler):
    run_inline = True
    raise_error = True

    def __init__(self, stream: TextIO) -> None: ...
```

The stream is injected: a unit test drives the handler with `io.StringIO` and never touches
the filesystem.

```python
@contextmanager
def mirror_to_run_file(
    directory: Path = DEFAULT_LOG_DIR,
    *,
    now: datetime | None = None,      # defaults to datetime.now(UTC)
    run_id: str | None = None,        # defaults to a short uuid4 hex
) -> Iterator[JsonlMirror]:
```

owns the file-per-run lifecycle and the naming, `logs/<utc-timestamp>-<short-run-id>.jsonl`,
and closes the file. `now` and `run_id` are injectable so the path is deterministic under
test. `main` uses the contextmanager; tests use the constructor.

Both class attributes are pinned deliberately:

- `run_inline = True` keeps events off LangChain's thread pool, so the order of records in
  the file is the order of events rather than a race.
- `raise_error = True` because the default swallows handler exceptions, and a mirror that
  silently stopped mirroring is worse than no mirror. This is only safe because the body
  cannot raise *on data* (see serialization); what remains that can raise is a closed stream
  or a bug, and both should crash.

#### Hooks

Implemented: `on_chain_start`, `on_chain_end`, `on_chain_error`, `on_tool_start`,
`on_tool_end`, `on_tool_error`, `on_chat_model_start`, `on_llm_end`, `on_llm_error`.

Not implemented: `on_agent_action`, `on_agent_finish`, `on_llm_new_token`, `on_retriever_*`,
`on_retry`, `on_text`, `on_custom_event`. Nothing in this graph produces them today, and a
hook with no producer is a guess.

#### Record shape

One JSON object per line: `ts` (UTC ISO-8601), `event`, `run_id`, `parent_run_id`, `name`,
and an event-specific payload. Three rules keep it safe:

1. **The `serialized` blob is never written**, only `serialized["name"]`. That dict carries
   the model's constructor kwargs. Rather than depend on LangChain redacting `HF_TOKEN`
   inside it correctly in every version, it is not serialized at all — a field that is never
   written cannot leak.
2. **Every value is bounded** by `MAX_FIELD_CHARS`, and a record that hit the bound carries
   `"truncated": true`. A `read_file` on a large file would otherwise put megabytes on one
   line.
3. **Serialization cannot fail on data**: `json.dumps(record, default=str)`, so `UUID`,
   `BaseMessage` and arbitrary tool output degrade to strings instead of raising.

#### Flush and vacuity

The stream is flushed after every record. The runs most worth having the file for are the
ones that crashed, and buffered output is exactly what is lost there.

The handler counts records written, and `main` asserts the count is non-zero after a turn.
A mirror wired up wrong produces an empty file, and an empty file is indistinguishable from
a quiet run — this is the check that separates them. Same move as
`require(bound != frozenset(), "…the absence check would be vacuous")` in `capabilities.py`.

### `main.py`

```python
backends = available_backends()
for backend in backends:
    ...                      # operating errors reported, not fatal
print(f"tracing: {…}  log: {path}")
with mirror_to_run_file() as mirror:
    ...                      # _run(agent, prompt, callbacks=[mirror])
```

`_run` gains a `callbacks` parameter merged into the `config` dict it already builds
alongside `recursion_limit`. `check_chat_completions_endpoint` calls
`build_model(config).invoke(...)` directly rather than through `_run`, so it takes the same
treatment — otherwise the one check that bypasses `_run` is the one hole in the mirror.

`main.py` remains the only module that knows a tracer and a log file both exist. Nothing
below it reads `os.environ`, constructs a backend, or opens a file.

## Testing

`tests/test_tracing.py` and `tests/test_mirror.py` — one file per source module, offline by
default, per CLAUDE.md.

**Tracing:**

- `from_env` across absent / blank / present environments, per backend.
- `available_backends` returns `()` for an empty mapping and both adapters for a full one.
- LangSmith `activate()` against a `monkeypatch.setenv` environment, and failing loudly when
  `tracing_is_enabled()` disagrees with what `from_env` concluded.
- Weave `activate()` with `weave.init` monkeypatched, called twice, asserting the second call
  is a no-op. Idempotency is claimed in CLAUDE.md and currently tested nowhere.
- A `TracingBackend` conformance assignment that strict mypy checks.

**Mirror:**

- One line per hook; each line is valid JSON with the expected keys.
- Record order matches event order.
- Truncation at `MAX_FIELD_CHARS` sets `truncated`.
- Non-serializable payloads (`UUID`, `BaseMessage`, a bare `object()`) survive.
- Flush happens after every record, verified with a counting fake stream.
- A `serialized` dict containing a secret-shaped value never appears in the output.
- Path naming is deterministic given injected `now` and `run_id`.

**Live (`-m live`, deselected by default):**

- One agent turn with LangSmith and Weave both active. This closes CLAUDE.md's open
  question — "Running both at once should work … but this has not been verified end to end
  in this repo yet". The chosen activation policy, activate every available backend, is only
  worth having if that is true, so it gets measured rather than assumed.

Every new test is mutation-checked before it is trusted: break the code it covers, watch it
go red. A test that survives its mutant is decorative.

## Documentation changes

- `.gitignore` gains `logs/`.
- `.env.example` gains `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` and `WEAVE_PROJECT`. All
  three are read by this design; the first two are already in the working `.env` but missing
  from the template.
- `docs/findings.md` gains two findings in the existing format: the both-tracers-at-once
  result, and LangChain's silently-swallowed callback exceptions with what this code does
  about it.
- `CLAUDE.md` gains the two new modules in the repo layout, and the verified signatures above
  under "Verified API facts".

## Open questions

None. Decisions taken during brainstorming and recorded above: activate every available
backend rather than selecting one; file per run rather than a single append-only log;
per-run callback wiring rather than global registration; tracing activation failure is
non-fatal.
