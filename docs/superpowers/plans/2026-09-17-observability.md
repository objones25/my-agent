# Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the harness two tracing backends that prove they turned on, and a local JSONL mirror of every agent event that works with no network and no API key.

**Architecture:** `tracing.py` holds a one-method `TracingBackend` protocol and two adapters (LangSmith, Weave) that both activate ambiently and process-wide. `mirror.py` holds `JsonlMirror`, a LangChain callback handler writing one JSON object per line to one file per run. The two modules never import each other; `main.py` is the only place they meet.

**Tech Stack:** Python 3.13, langchain-core 1.6.3 callbacks, langsmith 0.12.6, weave 0.53.9, pytest 9.1.1, mypy 2.3.1 (strict), ruff 0.16.8.

**Spec:** `docs/superpowers/specs/2026-09-17-observability-design.md`

## Global Constraints

- **Never write an API call from memory.** Every signature used below was read off the installed wheel on 2026-09-17. Re-verify with `inspect` after any `uv sync` that moves versions.
- **Use `require()` from `my_agent.negative_space`, never bare `assert`** in `src/` — `python -O` deletes `assert`. Plain `assert` stays correct in test bodies.
- **`require()` for programmer errors, typed exceptions for operating errors.** Absent tracing configuration is neither: it is a valid silent "off".
- **Unit tests are offline.** `tests/conftest.py` fails any unmarked test that opens a socket. Anything touching the network is marked `live` and deselected by default.
- **One test file per source module.** A new module gets a new file.
- **`filterwarnings = ["error"]`** is set; a new deprecation warning fails the build.
- **Line length 100**, ruff `select = ["E","F","I","UP","B","SIM","ANN","ASYNC","S","RET","PL","RUF"]`, mypy strict over `src` and `tests`.
- **A passing test is not a passing state until it has failed.** Every task's tests get mutation-checked: break the code they cover, watch them go red, restore.
- Full gate before any commit: `uv run ruff check . --fix && uv run mypy && uv run pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/my_agent/tracing.py` | CREATE — `TracingBackend` protocol, `LangSmithTracing`, `WeaveTracing`, `available_backends`, `langchain_tracer_names` |
| `src/my_agent/mirror.py` | CREATE — `JsonlMirror` callback handler, `run_log_path`, `mirror_to_file` |
| `tests/test_tracing.py` | CREATE — offline tests for both adapters, plus the one `live` coexistence test |
| `tests/test_mirror.py` | CREATE — offline tests for the handler and the file lifecycle |
| `src/my_agent/main.py` | MODIFY — activate tracing, open the run log, thread `callbacks` through every model call |
| `tests/test_main.py` | MODIFY — add wiring tests |
| `.gitignore`, `.env.example`, `CLAUDE.md`, `docs/findings.md` | MODIFY — documentation and ignore rules |

---

## Task 1: `TracingBackend` protocol and `LangSmithTracing`

**Files:**
- Create: `src/my_agent/tracing.py`
- Create: `tests/test_tracing.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `my_agent.negative_space.require`
- Produces:
  - `class TracingBackend(Protocol)` with a read-only `name: str` property and `activate() -> None`
  - `LangSmithTracing(project: str | None = None)`, `ClassVar name = "langsmith"`, `from_env(env: Mapping[str, str] | None = None) -> LangSmithTracing | None`, `activate() -> None`
  - Constants `LANGSMITH_API_KEY_ENV_VAR`, `LANGSMITH_TRACING_ENV_VAR`, `LANGSMITH_PROJECT_ENV_VAR`

**Background the implementer needs:**

LangSmith has no install step. The LangChain stack reads `LANGSMITH_*` environment variables on its own, so `activate()` cannot turn tracing on — it can only *verify* that it is on. That is the entire point of the class: without it, "no trace because there was no run" and "no trace because the variable was misspelled" look identical.

Two verified facts drive the code:

1. `langsmith.utils.get_env_var` is decorated `@functools.lru_cache(maxsize=100)`. A lookup made *before* `load_dotenv()` runs is remembered for the life of the process, so tracing would stay off no matter what `.env` said. `activate()` calls `get_env_var.cache_clear()` first. This also means **tests must clear that cache** after `monkeypatch.setenv`, or they will assert against a stale answer.
2. `get_env_var(name, namespaces=("LANGSMITH", "LANGCHAIN"))` still honours the `LANGCHAIN_*` names. Do not "fix" the fallback away.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracing.py`:

```python
"""Offline tests for the tracing seam.

`weave.init` and LangSmith's env-var cache are both process-global, so every
test here either injects a mapping or monkeypatches, and the cache fixture
guarantees no test leaks its environment into the next one.
"""

from __future__ import annotations

import pytest
from langsmith.utils import get_env_var

from my_agent.tracing import (
    LANGSMITH_API_KEY_ENV_VAR,
    LANGSMITH_PROJECT_ENV_VAR,
    LANGSMITH_TRACING_ENV_VAR,
    LangSmithTracing,
    TracingBackend,
)

LANGSMITH_ENV = {
    LANGSMITH_API_KEY_ENV_VAR: "ls-key-value",
    LANGSMITH_TRACING_ENV_VAR: "true",
    LANGSMITH_PROJECT_ENV_VAR: "my-project",
}


@pytest.fixture(autouse=True)
def _clear_langsmith_cache():
    """`langsmith.utils.get_env_var` is lru_cached, so a stale answer would
    outlive the test that caused it."""
    get_env_var.cache_clear()
    yield
    get_env_var.cache_clear()


def test_langsmith_is_configured_when_key_and_flag_are_present():
    backend = LangSmithTracing.from_env(LANGSMITH_ENV)
    assert backend is not None
    assert backend.project == "my-project"
    assert backend.name == "langsmith"


def test_langsmith_is_absent_without_an_api_key():
    env = LANGSMITH_ENV | {LANGSMITH_API_KEY_ENV_VAR: ""}
    assert LangSmithTracing.from_env(env) is None


def test_langsmith_is_absent_when_tracing_is_not_switched_on():
    """A key alone must not start billing traces on a run that never asked."""
    env = LANGSMITH_ENV | {LANGSMITH_TRACING_ENV_VAR: "false"}
    assert LangSmithTracing.from_env(env) is None


def test_langsmith_is_absent_from_an_empty_environment():
    assert LangSmithTracing.from_env({}) is None


def test_langsmith_project_is_none_when_unset():
    env = {k: v for k, v in LANGSMITH_ENV.items() if k != LANGSMITH_PROJECT_ENV_VAR}
    backend = LangSmithTracing.from_env(env)
    assert backend is not None
    assert backend.project is None


@pytest.mark.parametrize("flag", ["true", "True", "  TRUE  ", "1", "yes", "on"])
def test_langsmith_accepts_the_usual_truthy_spellings(flag):
    assert LangSmithTracing.from_env(LANGSMITH_ENV | {LANGSMITH_TRACING_ENV_VAR: flag}) is not None


def test_langsmith_activate_passes_when_langsmith_agrees(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-value")
    LangSmithTracing(project="my-project").activate()  # must not raise


def test_langsmith_activate_fails_loudly_when_langsmith_disagrees(monkeypatch):
    """The whole value of the class: a misspelled variable is loud, not silent."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    with pytest.raises(AssertionError, match="tracing off"):
        LangSmithTracing(project="my-project").activate()


def test_langsmith_activate_survives_a_poisoned_env_var_cache(monkeypatch):
    """Reading the flag before load_dotenv() caches a False that would otherwise
    disable tracing for the life of the process."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    get_env_var("TRACING")  # poison the cache with "absent"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    LangSmithTracing(project=None).activate()  # must not raise


def test_langsmith_satisfies_the_protocol():
    """Checked by mypy, not at runtime: the assignment is the assertion."""
    backend: TracingBackend = LangSmithTracing(project=None)
    assert backend.name == "langsmith"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'my_agent.tracing'`

- [ ] **Step 3: Write the implementation**

Create `src/my_agent/tracing.py`:

```python
"""Turning observability backends on, and proving they turned on.

Both backends here activate *ambiently* and process-wide: LangSmith reads
environment variables the LangChain stack consults on its own, and Weave
installs a global LangChain callback through `register_configure_hook`. Neither
hands back an object the agent has to be given, which is why the protocol is one
method wide — and why "did it actually turn on?" needs a separate, explicit
answer. Ambient activation that quietly did nothing is the failure this module
exists to make loud.

Nothing here reads `os.environ` unless the composition root asks it to: every
`from_env` takes an injectable mapping, exactly as `ModelConfig.from_env` does.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol

from langsmith.utils import get_env_var, tracing_is_enabled

from my_agent.negative_space import require

__all__ = [
    "LANGSMITH_API_KEY_ENV_VAR",
    "LANGSMITH_PROJECT_ENV_VAR",
    "LANGSMITH_TRACING_ENV_VAR",
    "LangSmithTracing",
    "TracingBackend",
]

LANGSMITH_API_KEY_ENV_VAR = "LANGSMITH_API_KEY"
LANGSMITH_TRACING_ENV_VAR = "LANGSMITH_TRACING"
LANGSMITH_PROJECT_ENV_VAR = "LANGSMITH_PROJECT"

_TRUTHY = frozenset({"true", "1", "yes", "on"})


class TracingBackend(Protocol):
    """One consumer's needs: the composition root turns a backend on and names it.

    A read-only property rather than an attribute, so an implementation is free
    to satisfy it with a `ClassVar`, a field, or a computed property.
    """

    @property
    def name(self) -> str: ...

    def activate(self) -> None: ...


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


@dataclass(frozen=True, slots=True)
class LangSmithTracing:
    """LangSmith tracing, which is already on or already off by the time we look.

    `activate()` installs nothing because there is nothing to install — it
    verifies. See the class docstring of the module for why that is the point.
    """

    project: str | None = None
    name: ClassVar[str] = "langsmith"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LangSmithTracing | None:
        """The backend if it is configured, `None` if it is not.

        Not being configured is a valid, silent answer — not an operating error.
        Both the key and the flag are required: a key alone must not start
        billing traces on a run that never asked to be traced.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        if not source.get(LANGSMITH_API_KEY_ENV_VAR, "").strip():
            return None
        if not _is_truthy(source.get(LANGSMITH_TRACING_ENV_VAR, "")):
            return None
        return cls(project=source.get(LANGSMITH_PROJECT_ENV_VAR, "").strip() or None)

    def activate(self) -> None:
        """Verify LangSmith agrees that tracing is on."""
        # `langsmith.utils.get_env_var` is lru_cached. Any lookup made before
        # load_dotenv() ran is remembered for the life of the process, so a
        # cached "absent" would disable tracing no matter what .env says.
        # Clearing is idempotent and makes the check read the real environment.
        get_env_var.cache_clear()
        require(
            bool(tracing_is_enabled()),
            f"{LANGSMITH_API_KEY_ENV_VAR} and {LANGSMITH_TRACING_ENV_VAR} are set, but "
            f"langsmith reports tracing off; check for a misspelled LANGSMITH_* variable",
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: all PASS

- [ ] **Step 5: Mutation-check the tests**

Make each change, run `uv run pytest tests/test_tracing.py`, confirm RED, then restore:

1. Delete the `get_env_var.cache_clear()` line → `test_langsmith_activate_survives_a_poisoned_env_var_cache` must fail.
2. Change `from_env`'s flag check to `return None` never (i.e. drop the `_is_truthy` guard) → `test_langsmith_is_absent_when_tracing_is_not_switched_on` must fail.
3. Change `require(bool(tracing_is_enabled()), ...)` to `require(True, ...)` → `test_langsmith_activate_fails_loudly_when_langsmith_disagrees` must fail.

A test that survives its mutant is decorative — fix it before moving on.

- [ ] **Step 6: Document the environment variables**

Replace the LangSmith stanza in `.env.example`:

```bash
# LangSmith tracing + evals. Both are required for tracing to switch on:
# an API key alone deliberately does nothing.
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-agent
```

- [ ] **Step 7: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
git add src/my_agent/tracing.py tests/test_tracing.py .env.example
git commit -m "Add the tracing seam and a LangSmith backend that verifies itself"
```

---

## Task 2: `WeaveTracing` and `available_backends`

**Files:**
- Modify: `src/my_agent/tracing.py`
- Modify: `tests/test_tracing.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `TracingBackend`, `LangSmithTracing` from Task 1
- Produces:
  - `WeaveTracing(project: str = DEFAULT_WEAVE_PROJECT)`, `ClassVar name = "weave"`, `from_env(env) -> WeaveTracing | None`, `activate() -> None`
  - `available_backends(env: Mapping[str, str] | None = None) -> tuple[TracingBackend, ...]`
  - `langchain_tracer_names() -> frozenset[str]`
  - Constants `WANDB_API_KEY_ENV_VAR`, `WEAVE_PROJECT_ENV_VAR`, `DEFAULT_WEAVE_PROJECT`

**Background the implementer needs:**

Weave's LangChain integration is *not* installed by `weave.init` alone in an obvious way: `weave.integrations.langchain.langchain` registers a `WeaveTracer` through `register_configure_hook(..., "WEAVE_TRACE_LANGCHAIN")`, applied when Weave autopatches during `init`. So a Weave client can exist while the agent is *not* traced. Asserting only "a client exists" would pass in exactly that broken state, which is why `activate()` also checks that a `WeaveTracer` is among the handlers LangChain would configure.

`langchain_core.callbacks.manager.CallbackManager.configure()` returns a manager whose `.handlers` name the globally installed tracers — verified to list `LangChainTracer` when the LangSmith env vars are set, with no network. That one call is how both postconditions and the coexistence test read back what is installed.

Do **not** pass `settings=` to `weave.init`: some of those values are evaluated on Weave's background thread pool and silently ignore what was passed. Anything configurable there goes through `WEAVE_*` environment variables.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracing.py` (and extend the imports from `my_agent.tracing` to include `DEFAULT_WEAVE_PROJECT`, `WANDB_API_KEY_ENV_VAR`, `WEAVE_PROJECT_ENV_VAR`, `WeaveTracing`, `available_backends`):

```python
import my_agent.tracing as tracing_module

WEAVE_ENV = {WANDB_API_KEY_ENV_VAR: "wandb-key-value", WEAVE_PROJECT_ENV_VAR: "weave-project"}


@pytest.fixture
def weave_spy(monkeypatch):
    """Records weave.init calls and reports a client only once init has run.

    weave.init reaches the network; nothing offline may call the real one.
    """
    calls: list[str] = []
    client: list[object] = []

    def fake_init(project_name, **kwargs):
        assert "settings" not in kwargs, "settings= is silently ignored on weave's thread pool"
        calls.append(project_name)
        client.append(object())
        return client[-1]

    monkeypatch.setattr(tracing_module.weave, "init", fake_init)
    monkeypatch.setattr(tracing_module, "get_weave_client", lambda: client[0] if client else None)
    monkeypatch.setattr(tracing_module, "langchain_tracer_names", lambda: frozenset({"WeaveTracer"}))
    return calls


def test_weave_is_configured_from_a_wandb_key(weave_spy):
    backend = WeaveTracing.from_env(WEAVE_ENV)
    assert backend is not None
    assert backend.project == "weave-project"
    assert backend.name == "weave"


def test_weave_is_absent_without_a_wandb_key():
    assert WeaveTracing.from_env({WEAVE_PROJECT_ENV_VAR: "weave-project"}) is None


def test_weave_falls_back_to_the_default_project():
    backend = WeaveTracing.from_env({WANDB_API_KEY_ENV_VAR: "wandb-key-value"})
    assert backend is not None
    assert backend.project == DEFAULT_WEAVE_PROJECT


def test_weave_activate_initialises_the_client(weave_spy):
    WeaveTracing(project="weave-project").activate()
    assert weave_spy == ["weave-project"]


def test_weave_activate_is_idempotent(weave_spy):
    """Claimed in CLAUDE.md, tested nowhere until now. A second init would
    install global state twice."""
    backend = WeaveTracing(project="weave-project")
    backend.activate()
    backend.activate()
    assert weave_spy == ["weave-project"]


def test_weave_activate_fails_when_the_langchain_tracer_is_missing(monkeypatch, weave_spy):
    """A Weave client with no LangChain hook means Weave is on and the agent is
    still untraced — the exact silent failure this postcondition exists for."""
    monkeypatch.setattr(tracing_module, "langchain_tracer_names", frozenset)
    with pytest.raises(AssertionError, match="WeaveTracer"):
        WeaveTracing(project="weave-project").activate()


def test_weave_rejects_an_empty_project():
    with pytest.raises(AssertionError, match="project"):
        WeaveTracing(project="")


def test_weave_satisfies_the_protocol():
    backend: TracingBackend = WeaveTracing()
    assert backend.name == "weave"


def test_no_backends_are_available_in_an_empty_environment():
    assert available_backends({}) == ()


def test_both_backends_are_available_when_both_are_configured():
    backends = available_backends(LANGSMITH_ENV | WEAVE_ENV)
    assert [b.name for b in backends] == ["langsmith", "weave"]


def test_only_the_configured_backend_is_available():
    backends = available_backends(WEAVE_ENV)
    assert [b.name for b in backends] == ["weave"]


def test_langchain_tracer_names_reports_the_installed_tracers(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-value")
    assert "LangChainTracer" in tracing_module.langchain_tracer_names()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: FAIL with `ImportError: cannot import name 'WeaveTracing'`

- [ ] **Step 3: Write the implementation**

Add to `src/my_agent/tracing.py` — extend the imports and `__all__`, then append the code:

```python
# added imports
from collections.abc import Callable

import weave
from langchain_core.callbacks.manager import CallbackManager
from weave.trace.context.weave_client_context import get_weave_client

# added constants
WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"
WEAVE_PROJECT_ENV_VAR = "WEAVE_PROJECT"
DEFAULT_WEAVE_PROJECT = "my-agent"

WEAVE_LANGCHAIN_TRACER = "WeaveTracer"


def langchain_tracer_names() -> frozenset[str]:
    """Class names of the tracers LangChain would install on a run right now.

    Both backends here are ambient and global, so this is the only way to read
    back what actually got installed rather than what was requested. Builds a
    manager and throws it away; no network, no side effects.
    """
    manager = CallbackManager.configure()
    return frozenset(type(handler).__name__ for handler in manager.handlers)


@dataclass(frozen=True, slots=True)
class WeaveTracing:
    """W&B Weave tracing. Unlike LangSmith, `activate()` does real work."""

    project: str = DEFAULT_WEAVE_PROJECT
    name: ClassVar[str] = "weave"

    def __post_init__(self) -> None:
        require(self.project != "", "project must not be empty")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WeaveTracing | None:
        source: Mapping[str, str] = os.environ if env is None else env
        if not source.get(WANDB_API_KEY_ENV_VAR, "").strip():
            return None
        return cls(project=source.get(WEAVE_PROJECT_ENV_VAR, "").strip() or DEFAULT_WEAVE_PROJECT)

    def activate(self) -> None:
        """`weave.init`, once per process.

        No `settings=`: some of those values are evaluated on Weave's background
        thread pool and silently ignore the argument. Use `WEAVE_*` env vars.
        """
        if get_weave_client() is not None:
            # weave.init installs global state; doing it twice is not harmless.
            return

        weave.init(self.project)

        require(get_weave_client() is not None, "weave.init returned without installing a client")
        # A client without the LangChain hook means Weave is on and the agent is
        # still untraced. Checking only for the client would pass in that state.
        require(
            WEAVE_LANGCHAIN_TRACER in langchain_tracer_names(),
            f"weave.init ran but no {WEAVE_LANGCHAIN_TRACER} is installed; "
            f"the agent would not be traced (is WEAVE_TRACE_LANGCHAIN disabled?)",
        )


_BACKEND_SOURCES: tuple[Callable[[Mapping[str, str] | None], TracingBackend | None], ...] = (
    LangSmithTracing.from_env,
    WeaveTracing.from_env,
)


def available_backends(env: Mapping[str, str] | None = None) -> tuple[TracingBackend, ...]:
    """Every backend the environment configures, in a stable order.

    An empty tuple is a valid answer and means no tracing. Adding a backend is
    one more entry in `_BACKEND_SOURCES`; no caller changes.
    """
    backends = tuple(
        backend for source in _BACKEND_SOURCES if (backend := source(env)) is not None
    )
    # Postcondition: `main` prints these names and activates each once, so a
    # duplicate would mean it reports a lie and double-installs global state.
    require(
        len({backend.name for backend in backends}) == len(backends),
        f"backend names must be unique, got {[b.name for b in backends]}",
    )
    return backends
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: all PASS

- [ ] **Step 5: Mutation-check the tests**

1. Delete the `if get_weave_client() is not None: return` guard → `test_weave_activate_is_idempotent` must fail.
2. Delete the `WEAVE_LANGCHAIN_TRACER in langchain_tracer_names()` check → `test_weave_activate_fails_when_the_langchain_tracer_is_missing` must fail.
3. Make `from_env` return `cls()` regardless of the key → `test_weave_is_absent_without_a_wandb_key` must fail.

- [ ] **Step 6: Document the environment variables**

Replace the Weave stanza in `.env.example`:

```bash
# W&B Weave tracing + evals. WANDB_API_KEY alone switches Weave on;
# WEAVE_PROJECT is optional and defaults to "my-agent".
WANDB_API_KEY=
WEAVE_PROJECT=my-agent
```

- [ ] **Step 7: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
git add src/my_agent/tracing.py tests/test_tracing.py .env.example
git commit -m "Add a Weave backend that proves the LangChain hook installed"
```

---

## Task 3: `JsonlMirror`

**Files:**
- Create: `src/my_agent/mirror.py`
- Create: `tests/test_mirror.py`

**Interfaces:**
- Consumes: `my_agent.negative_space.require`
- Produces: `JsonlMirror(stream: TextIO)` with `.records: int`, and constant `MAX_FIELD_CHARS`. Hooks: `on_chain_start`, `on_chain_end`, `on_chain_error`, `on_tool_start`, `on_tool_end`, `on_tool_error`, `on_chat_model_start`, `on_llm_end`, `on_llm_error`.

**Background the implementer needs:**

`BaseCallbackHandler` defaults `raise_error = False` and `run_inline = False` (verified). Both are pinned to `True` here:

- `run_inline = True` keeps events off LangChain's thread pool, so record order is event order rather than a race.
- `raise_error = True` because the default makes LangChain swallow anything the handler raises, and a mirror that silently stopped mirroring is worse than no mirror. This is only safe because the body cannot raise *on data* — see `_clip`.

Override signatures must match the base exactly or mypy strict will reject them. The base signatures, read off langchain-core 1.6.3, are in the code below.

**The `serialized` dict is never written.** It carries the component's constructor kwargs, which for a chat model include credentials. Rather than depend on LangChain redacting `HF_TOKEN` inside it correctly in every version, only `serialized["name"]` is read. A field that is never written cannot leak.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mirror.py`:

```python
"""Offline tests for the JSONL mirror.

The handler takes a stream, so every test here drives it with StringIO and the
filesystem is never touched.
"""

from __future__ import annotations

import io
import json
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from my_agent.mirror import MAX_FIELD_CHARS, JsonlMirror

RUN_ID = UUID("00000000-0000-4000-8000-000000000001")
PARENT_ID = UUID("00000000-0000-4000-8000-000000000002")


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.fixture
def stream():
    return io.StringIO()


@pytest.fixture
def mirror(stream):
    return JsonlMirror(stream)


def test_chain_start_writes_one_record(mirror, stream):
    mirror.on_chain_start({"name": "agent"}, {"messages": []}, run_id=RUN_ID)
    written = records(stream)
    assert len(written) == 1
    assert written[0]["event"] == "chain_start"
    assert written[0]["name"] == "agent"
    assert written[0]["run_id"] == str(RUN_ID)
    assert written[0]["parent_run_id"] is None


def test_parent_run_id_is_recorded(mirror, stream):
    mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID, parent_run_id=PARENT_ID)
    assert records(stream)[0]["parent_run_id"] == str(PARENT_ID)


def test_tool_calls_are_mirrored(mirror, stream):
    mirror.on_tool_start({"name": "write_file"}, '{"path": "/notes.txt"}', run_id=RUN_ID)
    mirror.on_tool_end("wrote 5 bytes", run_id=RUN_ID)
    written = records(stream)
    assert [r["event"] for r in written] == ["tool_start", "tool_end"]
    assert written[0]["name"] == "write_file"
    assert written[1]["output"] == "wrote 5 bytes"


def test_messages_are_mirrored_with_their_roles(mirror, stream):
    mirror.on_chat_model_start({"name": "ChatOpenAI"}, [[HumanMessage("ping")]], run_id=RUN_ID)
    written = records(stream)[0]
    assert written["event"] == "chat_model_start"
    assert written["messages"] == [{"type": "human", "text": "ping"}]


def test_llm_end_records_output_and_usage(mirror, stream):
    message = AIMessage(
        content="pong",
        usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])
    mirror.on_llm_end(result, run_id=RUN_ID)
    written = records(stream)[0]
    assert written["outputs"] == ["pong"]
    assert written["usage"]["total_tokens"] == 4


def test_errors_are_mirrored(mirror, stream):
    mirror.on_tool_error(ValueError("permission denied"), run_id=RUN_ID)
    written = records(stream)[0]
    assert written["event"] == "tool_error"
    assert written["error_type"] == "ValueError"
    assert "permission denied" in written["error"]


def test_record_order_is_event_order(mirror, stream):
    for index in range(10):
        mirror.on_chain_start({"name": f"step-{index}"}, {}, run_id=RUN_ID)
    assert [r["name"] for r in records(stream)] == [f"step-{i}" for i in range(10)]


def test_long_values_are_truncated_and_marked(mirror, stream):
    mirror.on_tool_end("x" * (MAX_FIELD_CHARS * 2), run_id=RUN_ID)
    written = records(stream)[0]
    assert written["truncated"] is True
    assert len(written["output"]) == MAX_FIELD_CHARS


def test_short_values_are_not_marked_truncated(mirror, stream):
    mirror.on_tool_end("small", run_id=RUN_ID)
    assert "truncated" not in records(stream)[0]


def test_unserialisable_payloads_do_not_raise(mirror, stream):
    mirror.on_tool_end(object(), run_id=RUN_ID)
    assert len(records(stream)) == 1


def test_the_serialized_blob_is_never_written(mirror, stream):
    """It carries the model's constructor kwargs, credentials included."""
    mirror.on_chat_model_start(
        {"name": "ChatOpenAI", "kwargs": {"openai_api_key": "hf_super_secret"}},
        [[HumanMessage("ping")]],
        run_id=RUN_ID,
    )
    assert "hf_super_secret" not in stream.getvalue()


def test_every_record_is_flushed(monkeypatch):
    """A crashed run is the one you most want the file for."""
    flushes = []
    stream = io.StringIO()
    monkeypatch.setattr(stream, "flush", lambda: flushes.append(1))
    mirror = JsonlMirror(stream)
    mirror.on_chain_start({"name": "a"}, {}, run_id=RUN_ID)
    mirror.on_chain_end({}, run_id=RUN_ID)
    assert len(flushes) == 2


def test_records_are_counted(mirror):
    assert mirror.records == 0
    mirror.on_chain_start({"name": "a"}, {}, run_id=uuid4())
    assert mirror.records == 1


def test_handler_flags_are_pinned(mirror):
    """Defaults are False for both; each True is a deliberate decision."""
    assert mirror.run_inline is True
    assert mirror.raise_error is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_mirror.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'my_agent.mirror'`

- [ ] **Step 3: Write the implementation**

Create `src/my_agent/mirror.py`:

```python
"""A local, always-on JSONL mirror of everything the agent does.

LangSmith and Weave answer "what did the model do?" for a vendor that was
reachable and configured. This answers "what happened in this process?" with no
network, no API key, and no account — one file per run, one JSON object per
line, flushed as it goes, because the runs most worth reading back are the ones
that crashed.

It is a plain object that writes to a stream. Nothing here is global, nothing
here is installed, and nothing here knows a tracing backend exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, TextIO
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from my_agent.negative_space import require

__all__ = ["MAX_FIELD_CHARS", "JsonlMirror"]

MAX_FIELD_CHARS = 4000
"""Per-value bound. A `read_file` on something large would otherwise put
megabytes on one line and make the file unreadable exactly when it matters."""


def _clip(value: Any) -> tuple[Any, bool]:
    """A JSON-safe value, bounded, and whether the bound bit.

    The round trip through `default=str` is what makes `raise_error = True` safe:
    a `UUID`, a `BaseMessage`, or a tool returning an arbitrary object degrades
    to a string instead of raising inside a callback.
    """
    safe = json.loads(json.dumps(value, default=str))
    text = safe if isinstance(safe, str) else json.dumps(safe)
    if len(text) <= MAX_FIELD_CHARS:
        return safe, False
    return text[:MAX_FIELD_CHARS], True


def _component_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """The component's name, and *only* the name.

    `serialized` carries the component's constructor kwargs, which for a chat
    model include its credentials. Rather than depend on LangChain redacting
    them correctly in every version, none of that dict is ever written.
    """
    explicit = kwargs.get("name")
    if isinstance(explicit, str) and explicit:
        return explicit
    if serialized:
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        path = serialized.get("id")
        if isinstance(path, list) and path:
            return str(path[-1])
    return "unknown"


class JsonlMirror(BaseCallbackHandler):
    """Writes one JSON object per line for every event LangChain reports."""

    run_inline = True
    """Keep events off the thread pool: record order is then event order."""

    raise_error = True
    """The default (False) makes LangChain swallow exceptions raised in here, and
    a mirror that silently stopped mirroring is worse than no mirror. Safe only
    because `_clip` means the body cannot raise on data."""

    def __init__(self, stream: TextIO) -> None:
        require(hasattr(stream, "write"), f"stream must be writable, got {type(stream).__name__}")
        self._stream = stream
        self._records = 0

    @property
    def records(self) -> int:
        """How many records were written. `main` asserts this is non-zero: an
        empty file and a quiet run are otherwise indistinguishable."""
        return self._records

    def _write(
        self, event: str, run_id: UUID, parent_run_id: UUID | None, **payload: Any
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
        }
        truncated = False
        for key, value in payload.items():
            record[key], hit = _clip(value)
            truncated = truncated or hit
        if truncated:
            record["truncated"] = True

        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
        self._records += 1

    # -- chains (graph steps) ------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            inputs=inputs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write("chain_end", run_id, parent_run_id, outputs=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )

    # -- tools ---------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            input=inputs if inputs is not None else input_str,
        )

    def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self._write("tool_end", run_id, parent_run_id, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )

    # -- model calls ---------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chat_model_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            messages=[
                {"type": message.type, "text": message.text}
                for batch in messages
                for message in batch
            ],
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        outputs: list[str] = []
        usage: dict[str, Any] | None = None
        for batch in response.generations:
            for generation in batch:
                outputs.append(generation.text)
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    usage = dict(metadata)
        self._write("llm_end", run_id, parent_run_id, outputs=outputs, usage=usage)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "llm_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )
```

Note on `message.text`: it is a **property** on langchain-core 1.6.3, not a method (`docs/findings.md` F9). Calling `message.text()` would raise.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mirror.py -v`
Expected: all PASS

- [ ] **Step 5: Mutation-check the tests**

1. Delete `self._stream.flush()` → `test_every_record_is_flushed` must fail.
2. Change `_clip` to return `(safe, False)` always → `test_long_values_are_truncated_and_marked` must fail.
3. Make `_component_name` return `str(serialized)` → `test_the_serialized_blob_is_never_written` must fail.
4. Change `run_inline` to `False` → `test_handler_flags_are_pinned` must fail.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
git add src/my_agent/mirror.py tests/test_mirror.py
git commit -m "Mirror every agent event to a JSONL stream"
```

---

## Task 4: the run-file lifecycle

**Files:**
- Modify: `src/my_agent/mirror.py`
- Modify: `tests/test_mirror.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `JsonlMirror` from Task 3
- Produces:
  - `run_log_path(directory: Path = DEFAULT_LOG_DIR, *, now: datetime | None = None, run_id: str | None = None) -> Path`
  - `mirror_to_file(path: Path) -> Iterator[JsonlMirror]` (a `@contextmanager`)
  - Constant `DEFAULT_LOG_DIR = Path("logs")`

**Design note (a refinement on the spec):** the spec sketched a single `mirror_to_run_file(directory, now, run_id)`. Splitting it into `run_log_path` (names the file) and `mirror_to_file` (opens it) lets `main` print the path *before* the run starts, which is when you want to know where to tail. Naming still lives here, not in `main`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mirror.py` (extend imports with `DEFAULT_LOG_DIR`, `mirror_to_file`, `run_log_path`, plus `from datetime import UTC, datetime` and `from pathlib import Path`):

```python
FIXED_NOW = datetime(2026, 9, 17, 14, 30, 5, tzinfo=UTC)


def test_run_log_path_is_deterministic_when_time_and_id_are_injected():
    path = run_log_path(Path("logs"), now=FIXED_NOW, run_id="abc123")
    assert path == Path("logs/20260917T143005Z-abc123.jsonl")


def test_run_log_path_defaults_to_the_log_directory():
    assert run_log_path(now=FIXED_NOW, run_id="abc123").parent == DEFAULT_LOG_DIR


def test_run_log_path_rejects_a_naive_timestamp():
    """A naive stamp is ambiguous, and these filenames sort by time."""
    with pytest.raises(AssertionError, match="timezone"):
        run_log_path(now=datetime(2026, 9, 17, 14, 30, 5), run_id="abc123")


def test_run_log_path_normalises_to_utc():
    from datetime import timedelta, timezone

    eastern = timezone(timedelta(hours=-4))
    path = run_log_path(now=FIXED_NOW.astimezone(eastern), run_id="abc123")
    assert path.name == "20260917T143005Z-abc123.jsonl"


def test_run_log_paths_differ_between_runs():
    assert run_log_path(now=FIXED_NOW) != run_log_path(now=FIXED_NOW)


def test_mirror_to_file_writes_a_readable_run_log(tmp_path):
    path = tmp_path / "logs" / "run.jsonl"
    with mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
    assert json.loads(path.read_text().strip())["name"] == "agent"


def test_mirror_to_file_creates_the_directory(tmp_path):
    path = tmp_path / "deeper" / "logs" / "run.jsonl"
    with mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
    assert path.exists()


def test_mirror_to_file_closes_the_file_even_when_the_run_raises(tmp_path):
    """The crashed run is the one whose log has to survive."""
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError), mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
        raise RuntimeError("the agent exploded")
    assert json.loads(path.read_text().strip())["name"] == "agent"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_mirror.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_log_path'`

- [ ] **Step 3: Write the implementation**

Add to `src/my_agent/mirror.py` — extend the imports and `__all__`, then append:

```python
# added imports
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# added constant
DEFAULT_LOG_DIR = Path("logs")


def run_log_path(
    directory: Path = DEFAULT_LOG_DIR,
    *,
    now: datetime | None = None,
    run_id: str | None = None,
) -> Path:
    """Where this run's mirror goes: `<dir>/<utc-stamp>-<short-id>.jsonl`.

    `now` and `run_id` are injectable so the name is deterministic under test.
    """
    moment = datetime.now(UTC) if now is None else now
    require(
        moment.tzinfo is not None,
        "now must be timezone-aware; these filenames sort by time and a naive "
        "stamp is ambiguous",
    )
    token = uuid.uuid4().hex[:8] if run_id is None else run_id
    require(token != "", "run_id must not be empty")
    return directory / f"{moment.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{token}.jsonl"


@contextmanager
def mirror_to_file(path: Path) -> Iterator[JsonlMirror]:
    """A mirror writing to `path`, closed on the way out — exception or not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yield JsonlMirror(stream)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mirror.py -v`
Expected: all PASS

- [ ] **Step 5: Mutation-check the tests**

1. Drop `.astimezone(UTC)` → `test_run_log_path_normalises_to_utc` must fail.
2. Drop the `moment.tzinfo is not None` check → `test_run_log_path_rejects_a_naive_timestamp` must fail.
3. Replace `uuid.uuid4().hex[:8]` with a constant → `test_run_log_paths_differ_between_runs` must fail.

- [ ] **Step 6: Ignore the log directory**

Append to `.gitignore`:

```
# Local JSONL run mirrors
logs/
```

- [ ] **Step 7: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
git add src/my_agent/mirror.py tests/test_mirror.py .gitignore
git commit -m "Give the mirror a file per run"
```

---

## Task 5: wire tracing and the mirror into the composition root

**Files:**
- Modify: `src/my_agent/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `available_backends` (Task 2), `JsonlMirror`, `mirror_to_file`, `run_log_path` (Tasks 3-4)
- Produces: `_activate_tracing() -> tuple[str, ...]`; `_run`, `_single_turn`, `_run_checks` and every check function gain a `callbacks: list[BaseCallbackHandler]` parameter

**Background the implementer needs:**

`RunnableConfig` accepts `callbacks` alongside `recursion_limit` (verified), so the handler reaches the compiled graph through the config dict `_run` already builds.

`check_chat_completions_endpoint` calls `build_model(config).invoke(...)` directly rather than through `_run`, so it needs the same treatment — otherwise the one check that bypasses `_run` is the one hole in the mirror.

**A bug to fix while you are here.** `_run_checks` catches `Exception` and its comment claims "A `CheckFailed` is a bug in our own contracts and is left to propagate." That is not true: `CheckFailed` subclasses `AssertionError`, which is an `Exception`, so it is currently caught and reported as a failed check. Add an explicit `except CheckFailed: raise` before the broad handler in both `_run_checks` and the new `_activate_tracing`, so a violated contract crashes as the comment always intended.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py` (extend imports with `io`, `pytest`, `from my_agent.mirror import JsonlMirror`, `from my_agent.negative_space import CheckFailed`):

```python
def test_activate_tracing_reports_each_backend_that_turned_on(monkeypatch):
    activated: list[str] = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def activate(self):
            activated.append(self.name)

    monkeypatch.setattr(
        main_module, "available_backends", lambda: (FakeBackend("a"), FakeBackend("b"))
    )
    assert main_module._activate_tracing() == ("a", "b")
    assert activated == ["a", "b"]


def test_activate_tracing_survives_a_backend_that_cannot_reach_its_service(monkeypatch, capsys):
    """A telemetry outage must not take the run down with it."""

    class Broken:
        name = "weave"

        def activate(self):
            raise ConnectionError("w&b unreachable")

    class Working:
        name = "langsmith"

        def activate(self):
            return None

    monkeypatch.setattr(main_module, "available_backends", lambda: (Broken(), Working()))
    assert main_module._activate_tracing() == ("langsmith",)
    assert "weave" in capsys.readouterr().err


def test_activate_tracing_lets_a_broken_contract_crash(monkeypatch):
    """CheckFailed is a bug in our own code, not an operating error."""

    class Contradictory:
        name = "langsmith"

        def activate(self):
            raise CheckFailed("tracing reported off")

    monkeypatch.setattr(main_module, "available_backends", lambda: (Contradictory(),))
    with pytest.raises(CheckFailed):
        main_module._activate_tracing()


def test_activate_tracing_is_quiet_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(main_module, "available_backends", tuple)
    assert main_module._activate_tracing() == ()


def test_run_passes_callbacks_to_the_graph():
    """The mirror is only worth having if it is actually attached."""
    seen: dict[str, object] = {}

    class FakeAgent:
        def invoke(self, payload, config):
            seen.update(config)
            return {"messages": [{"role": "user", "content": "hi"}, AIMessage("pong")]}

    mirror = JsonlMirror(io.StringIO())
    main_module._run(FakeAgent(), "ping", [mirror])
    assert seen["callbacks"] == [mirror]
    assert seen["recursion_limit"] == main_module.RECURSION_LIMIT
```

`AIMessage` needs importing from `langchain_core.messages` in this test file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — `AttributeError: module 'my_agent.main' has no attribute '_activate_tracing'`

- [ ] **Step 3: Write the implementation**

In `src/my_agent/main.py`:

Add imports:

```python
from langchain_core.callbacks import BaseCallbackHandler

from my_agent.mirror import JsonlMirror, mirror_to_file, run_log_path
from my_agent.negative_space import CheckFailed, require
from my_agent.tracing import available_backends
```

Change `_run` (currently at line 61) to take and forward callbacks:

```python
def _run(agent: object, prompt: str, callbacks: list[BaseCallbackHandler]) -> list[BaseMessage]:
    """One bounded agent turn, mirrored."""
    result = agent.invoke(  # type: ignore[attr-defined]
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": RECURSION_LIMIT, "callbacks": callbacks},
    )
```

Add the tracing activation function:

```python
def _activate_tracing() -> tuple[str, ...]:
    """Turn on every configured backend; return the names that turned on.

    A backend that cannot reach its service is an *operating* error: the agent
    still works without telemetry, and losing a run's output to a W&B outage
    would be the worse failure. A `CheckFailed` is a violated contract of ours
    and still crashes.
    """
    active: list[str] = []
    for backend in available_backends():
        try:
            backend.activate()
        except CheckFailed:
            raise
        except Exception as exc:
            print(f"tracing: {backend.name} FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
        else:
            active.append(backend.name)
    return tuple(active)
```

Give every check and both runners the parameter. Each check signature becomes
`(config: ModelConfig, callbacks: list[BaseCallbackHandler]) -> CheckResult`, and each
`_run(agent, prompt)` call becomes `_run(agent, prompt, callbacks)`. In
`check_chat_completions_endpoint`, the direct invoke becomes:

```python
    reply = build_model(config).invoke(
        "Reply with exactly the word: pong", config={"callbacks": callbacks}
    )
```

Update the `CHECKS` type alias:

```python
CHECKS: tuple[Callable[[ModelConfig, list[BaseCallbackHandler]], CheckResult], ...] = (
```

In `_run_checks`, take `callbacks`, pass it to each check, and stop swallowing contract failures:

```python
def _run_checks(config: ModelConfig, callbacks: list[BaseCallbackHandler]) -> int:
    ...
        try:
            result = check(config, callbacks)
        except CheckFailed:
            raise
        except Exception as exc:
            result = CheckResult("??", check.__name__, False, f"{type(exc).__name__}: {exc}")
```

And `_single_turn(config, prompt, callbacks)` forwards it to `_run`.

Finally, rewrite the body of `main()` after the `ModelConfig.from_env()` block:

```python
    active = _activate_tracing()
    log_path = run_log_path()

    prompt = " ".join(sys.argv[1:]).strip()
    print(f"model:  {config.model}")
    print(f"tracing: {', '.join(active) if active else 'none'}")
    print(f"log:    {log_path}")

    with mirror_to_file(log_path) as mirror:
        callbacks: list[BaseCallbackHandler] = [mirror]
        if prompt:
            print(f"prompt: {prompt}\n")
            exit_code = _single_turn(config, prompt, callbacks)
        else:
            exit_code = _run_checks(config, callbacks)

    # An empty mirror and a quiet run look identical on disk. This is what
    # separates "nothing happened" from "the callbacks were never attached".
    require(mirror.records > 0, f"the mirror wrote nothing to {log_path}; callbacks are not wired")
    return exit_code
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all PASS (the whole suite, since `main.py` signatures changed)

- [ ] **Step 5: Mutation-check the tests**

1. Drop `"callbacks": callbacks` from `_run`'s config → `test_run_passes_callbacks_to_the_graph` must fail.
2. Remove `except CheckFailed: raise` from `_activate_tracing` → `test_activate_tracing_lets_a_broken_contract_crash` must fail.
3. Re-raise instead of printing in the broad handler → `test_activate_tracing_survives_a_backend_that_cannot_reach_its_service` must fail.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
git add src/my_agent/main.py tests/test_main.py
git commit -m "Wire tracing and the run mirror into the composition root"
```

---

## Task 6: prove both tracers coexist, and record what was learned

**Files:**
- Modify: `tests/test_tracing.py`
- Modify: `docs/findings.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-5

**Background the implementer needs:**

CLAUDE.md currently says: "Running both at once should work — LangSmith traces via env var, Weave via its LangChain callback — but **this has not been verified end to end in this repo yet**." The activation policy chosen in the spec (activate every available backend) is only worth having if that is true, so this task measures it and replaces the caveat with a result.

This test hits W&B and needs `HF_TOKEN`, `LANGSMITH_*` and `WANDB_API_KEY` in the environment. It is marked `live`, so it is deselected by default and the socket guard in `conftest.py` steps aside for it.

- [ ] **Step 1: Write the live test**

Append to `tests/test_tracing.py`:

```python
@pytest.mark.live
def test_langsmith_and_weave_trace_the_same_run():
    """The coexistence question CLAUDE.md left open, answered by measurement.

    Asserts on the *installed handlers*, not on either dashboard: what could go
    wrong is one backend's global install displacing the other's, and that is
    visible locally.
    """
    from dotenv import load_dotenv

    from my_agent.agent import build_agent
    from my_agent.model import ModelConfig, build_model
    from my_agent.tracing import available_backends, langchain_tracer_names

    load_dotenv()
    backends = available_backends()
    names = {backend.name for backend in backends}
    assert names == {"langsmith", "weave"}, f"both backends must be configured, got {names}"

    for backend in backends:
        backend.activate()

    installed = langchain_tracer_names()
    assert "LangChainTracer" in installed, f"LangSmith tracer missing: {sorted(installed)}"
    assert "WeaveTracer" in installed, f"Weave tracer missing: {sorted(installed)}"

    agent = build_agent(build_model(ModelConfig.from_env()))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Reply with exactly the word: pong"}]},
        config={"recursion_limit": 25},
    )
    assert result["messages"][-1].type == "ai"

    still_installed = langchain_tracer_names()
    assert "LangChainTracer" in still_installed
    assert "WeaveTracer" in still_installed
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_tracing.py -m live -v`
Expected: PASS. If it fails, that is the finding — record what actually happened in Step 3 rather than making the test agree with the assumption.

- [ ] **Step 3: Re-prove both documentation claims before editing anything**

**Do not edit a doc on the strength of this plan's say-so.** Both claims below were
proved false on 2026-09-17 by running the scripts here. Run them again. If either comes
back the other way — the claim holds on your versions — leave that doc sentence alone
and say so in the task report. A doc is only wrong once you have watched it be wrong.

Proof 1, the `LANGCHAIN_*` claim. Save and run:

```python
"""CLAUDE.md claims: "(Older `LANGCHAIN_*` names no longer work.)" """
import os

for key in list(os.environ):
    if key.startswith(("LANGSMITH_", "LANGCHAIN_")):
        del os.environ[key]

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "fake-key"
os.environ["LANGCHAIN_PROJECT"] = "legacy-project"

from langchain_core.callbacks.manager import CallbackManager
from langsmith.utils import get_tracer_project, tracing_is_enabled

print("tracing_is_enabled():", tracing_is_enabled())
print("get_tracer_project():", get_tracer_project())
print("installed handlers:  ", [type(h).__name__ for h in CallbackManager.configure().handlers])
```

Observed 2026-09-17 (langsmith 0.12.6, langchain-core 1.6.3):

```
tracing_is_enabled(): True
get_tracer_project(): legacy-project
installed handlers:   ['LangChainTracer']
```

With no `LANGSMITH_*` variable set at all, the legacy namespace enables tracing, resolves
the project, and installs the tracer. The claim is false. Cause: `langsmith.utils.get_env_var`
takes `namespaces=("LANGSMITH", "LANGCHAIN")` and searches both.

Proof 2, the `CheckFailed` claim in `src/my_agent/main.py`. Save and run:

```python
"""main.py claims: "A CheckFailed is a bug in our own contracts and is left to propagate." """
from pydantic import SecretStr

from my_agent import main as main_module
from my_agent.model import ModelConfig
from my_agent.negative_space import CheckFailed


def check_that_violates_a_contract(config):
    raise CheckFailed("a contract of ours was violated")


main_module.CHECKS = (check_that_violates_a_contract,)

try:
    exit_code = main_module._run_checks(ModelConfig(api_key=SecretStr("hf_token_value")))
except CheckFailed:
    print("RESULT: propagated -- the comment is correct.")
else:
    print(f"RESULT: swallowed, reported as a failed check (exit {exit_code}) -- comment is wrong.")
```

Observed 2026-09-17:

```
  [FAIL] ??  check_that_violates_a_contract
         CheckFailed: a contract of ours was violated
0/1 checks passed
RESULT: swallowed, reported as a failed check (exit 1) -- comment is wrong.
```

`CheckFailed.__mro__` is `(CheckFailed, AssertionError, Exception, BaseException, object)`,
so `except Exception` catches it. The comment describes an intent the code never had.

- [ ] **Step 4: Record the findings**

Append two findings to `docs/findings.md`, in the existing F-format (heading, how it was checked, what the code does about it):

- **F11 — LangSmith and Weave coexist** (or do not, per Step 2's real result). How checked: `langchain_tracer_names()` after activating both, before and after a live agent turn. What the code does: `available_backends` activates every configured backend rather than selecting one.
- **F12 — LangChain swallows exceptions raised inside a callback handler.** `BaseCallbackHandler.raise_error` defaults to `False`, so a broken handler stops mirroring in silence. What the code does: `JsonlMirror` pins `raise_error = True` and `run_inline = True`, keeps its body incapable of raising on data via `_clip`, and `main` asserts `mirror.records > 0` after every run.

Also record, under the same file's conventions, that `langsmith.utils.get_env_var` is `lru_cache`d — a lookup before `load_dotenv()` disables tracing for the life of the process — and that it still honours the `LANGCHAIN_*` namespace.

- [ ] **Step 5: Update CLAUDE.md**

1. Repo layout: add `tracing.py` (TracingBackend + LangSmith/Weave adapters) and `mirror.py` (JsonlMirror, run-file naming) to the `src/my_agent/` block, and `test_tracing.py` / `test_mirror.py` to `tests/`.
2. Observability section: replace the "this has not been verified end to end in this repo yet" sentence with Step 2's result, linked to F11.
3. **Correct the claim Step 3 disproved**, and only if Step 3 disproved it again on your run. CLAUDE.md line ~300 says "(Older `LANGCHAIN_*` names no longer work.)" Replace it with what the proof showed: `langsmith.utils.get_env_var` searches `namespaces=("LANGSMITH", "LANGCHAIN")`, so the legacy names still enable tracing and still resolve the project; prefer `LANGSMITH_*` for new work. Note the `lru_cache` on the same function while you are there.
4. Verified API facts: add `BaseCallbackHandler.raise_error`/`run_inline` defaults, `CallbackManager.configure()` as the way to read installed tracers, `tracing_is_enabled()`, `get_weave_client()`, and the `WEAVE_TRACE_LANGCHAIN` gate.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run ruff check . --fix && uv run mypy && uv run pytest
uv run pytest -m live
git add tests/test_tracing.py docs/findings.md CLAUDE.md
git commit -m "Verify both tracers coexist and record what the wheels actually do"
```

---

## Verification

After Task 6, the whole thing end to end:

```bash
uv run pytest                     # offline suite, green
uv run pytest -m live             # router + both tracers
uv run my-agent "say pong"        # a real turn
ls logs/                          # one file, named by time
cat logs/*.jsonl | head -5        # readable records, no credentials
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007
```
