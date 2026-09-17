"""Offline tests for the tracing seam.

`weave.init` and LangSmith's env-var cache are both process-global, so every
test here either injects a mapping or monkeypatches, and the cache fixture
guarantees no test leaks its environment into the next one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from langsmith.utils import get_env_var

import my_agent.tracing as tracing_module
from my_agent.tracing import (
    DEFAULT_WEAVE_PROJECT,
    LANGSMITH_API_KEY_ENV_VAR,
    LANGSMITH_PROJECT_ENV_VAR,
    LANGSMITH_TRACING_ENV_VAR,
    WANDB_API_KEY_ENV_VAR,
    WEAVE_PROJECT_ENV_VAR,
    LangSmithTracing,
    TracingBackend,
    TracingMisconfigured,
    WeaveTracing,
    available_backends,
)

LANGSMITH_ENV = {
    LANGSMITH_API_KEY_ENV_VAR: "ls-key-value",
    LANGSMITH_TRACING_ENV_VAR: "true",
    LANGSMITH_PROJECT_ENV_VAR: "my-project",
}

WEAVE_ENV = {WANDB_API_KEY_ENV_VAR: "wandb-key-value", WEAVE_PROJECT_ENV_VAR: "weave-project"}


@pytest.fixture(autouse=True)
def _clear_langsmith_cache() -> Iterator[None]:
    """`langsmith.utils.get_env_var` is lru_cached, so a stale answer would
    outlive the test that caused it."""
    # mypy sees get_env_var's Optional-default signature as an Overload and misses
    # cache_clear; verified present via `inspect` on the installed langsmith wheel.
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    yield
    get_env_var.cache_clear()  # type: ignore[attr-defined]


def test_langsmith_is_configured_when_key_and_flag_are_present() -> None:
    backend = LangSmithTracing.from_env(LANGSMITH_ENV)
    assert backend is not None
    assert backend.project == "my-project"
    assert backend.name == "langsmith"


def test_langsmith_is_absent_without_an_api_key() -> None:
    env = LANGSMITH_ENV | {LANGSMITH_API_KEY_ENV_VAR: ""}
    assert LangSmithTracing.from_env(env) is None


def test_langsmith_is_absent_when_tracing_is_not_switched_on() -> None:
    """A key alone must not start billing traces on a run that never asked."""
    env = LANGSMITH_ENV | {LANGSMITH_TRACING_ENV_VAR: "false"}
    assert LangSmithTracing.from_env(env) is None


def test_langsmith_is_absent_from_an_empty_environment() -> None:
    assert LangSmithTracing.from_env({}) is None


def test_langsmith_project_is_none_when_unset() -> None:
    env = {k: v for k, v in LANGSMITH_ENV.items() if k != LANGSMITH_PROJECT_ENV_VAR}
    backend = LangSmithTracing.from_env(env)
    assert backend is not None
    assert backend.project is None


@pytest.mark.parametrize("flag", ["true", "True", "  TRUE  ", "1", "yes", "on"])
def test_langsmith_accepts_the_usual_truthy_spellings(flag: str) -> None:
    assert LangSmithTracing.from_env(LANGSMITH_ENV | {LANGSMITH_TRACING_ENV_VAR: flag}) is not None


@pytest.mark.parametrize("flag", ["true", "True", "  TRUE  ", "1", "yes", "on"])
def test_langsmith_pipeline_only_the_exact_string_true_actually_traces(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """`from_env` is permissive on purpose (`_TRUTHY` accepts more spellings than
    langsmith's own literal `"true"` comparison), so `from_env` alone configuring a
    backend does not mean the backend will actually trace. This exercises the full
    pipeline, `from_env()` then `.activate()`, for every accepted spelling: only the
    exact string "true" may activate cleanly, and every other spelling must raise
    `TracingMisconfigured` naming the offending value.
    """
    monkeypatch.setenv(LANGSMITH_API_KEY_ENV_VAR, "ls-key-value")
    monkeypatch.setenv(LANGSMITH_TRACING_ENV_VAR, flag)

    backend = LangSmithTracing.from_env()
    assert backend is not None

    if flag == "true":
        backend.activate()  # must not raise
    else:
        with pytest.raises(TracingMisconfigured, match=re.escape(flag)):
            backend.activate()


def test_langsmith_activate_passes_when_langsmith_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-value")
    LangSmithTracing(project="my-project").activate()  # must not raise


def test_langsmith_activate_fails_loudly_when_langsmith_disagrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole value of the class: a misconfigured setup fails loudly, not silently.

    An operating error (everything `activate()` looks at came from the environment,
    not from a caller this code owns), so `TracingMisconfigured` rather than a
    `require()`-raised `CheckFailed`/`AssertionError`.
    """
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    with pytest.raises(TracingMisconfigured, match="exact string"):
        LangSmithTracing(project="my-project").activate()


def test_langsmith_activate_survives_a_poisoned_env_var_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the flag before load_dotenv() caches a False that would otherwise
    disable tracing for the life of the process.

    `get_env_var("TRACING", default="")` (matching the exact call `tracing_is_enabled`
    makes internally) is the call to poison — `get_env_var("TRACING")` alone caches
    under a different key (no `default` kwarg supplied) and never collides with it,
    which made an earlier version of this test pass even without the cache clear.
    """
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    get_env_var("TRACING", default="")  # poison the cache with "absent"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    LangSmithTracing(project=None).activate()  # must not raise


def test_langsmith_satisfies_the_protocol() -> None:
    """Checked by mypy, not at runtime: the assignment is the assertion."""
    backend: TracingBackend = LangSmithTracing(project=None)
    assert backend.name == "langsmith"


@pytest.fixture
def weave_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records weave.init calls and reports a client only once init has run.

    weave.init reaches the network; nothing offline may call the real one.
    """
    calls: list[str] = []
    client: list[object] = []

    def fake_init(project_name: str, **kwargs: object) -> object:
        assert "settings" not in kwargs, "settings= is silently ignored on weave's thread pool"
        calls.append(project_name)
        client.append(object())
        return client[-1]

    monkeypatch.setattr(tracing_module.weave, "init", fake_init)
    monkeypatch.setattr(tracing_module, "get_weave_client", lambda: client[0] if client else None)
    monkeypatch.setattr(
        tracing_module, "langchain_tracer_names", lambda: frozenset({"WeaveTracer"})
    )
    return calls


def test_weave_is_configured_from_a_wandb_key(weave_spy: list[str]) -> None:
    backend = WeaveTracing.from_env(WEAVE_ENV)
    assert backend is not None
    assert backend.project == "weave-project"
    assert backend.name == "weave"


def test_weave_is_absent_without_a_wandb_key() -> None:
    assert WeaveTracing.from_env({WEAVE_PROJECT_ENV_VAR: "weave-project"}) is None


def test_weave_falls_back_to_the_default_project() -> None:
    backend = WeaveTracing.from_env({WANDB_API_KEY_ENV_VAR: "wandb-key-value"})
    assert backend is not None
    assert backend.project == DEFAULT_WEAVE_PROJECT


def test_weave_activate_initialises_the_client(weave_spy: list[str]) -> None:
    WeaveTracing(project="weave-project").activate()
    assert weave_spy == ["weave-project"]


def test_weave_activate_is_idempotent(weave_spy: list[str]) -> None:
    """Claimed in CLAUDE.md, tested nowhere until now. A second init would
    install global state twice."""
    backend = WeaveTracing(project="weave-project")
    backend.activate()
    backend.activate()
    assert weave_spy == ["weave-project"]


def test_weave_activate_fails_when_the_langchain_tracer_is_missing(
    monkeypatch: pytest.MonkeyPatch, weave_spy: list[str]
) -> None:
    """A Weave client with no LangChain hook means Weave is on and the agent is
    still untraced — the exact silent failure this postcondition exists for."""
    monkeypatch.setattr(tracing_module, "langchain_tracer_names", frozenset)
    with pytest.raises(AssertionError, match="WeaveTracer"):
        WeaveTracing(project="weave-project").activate()


def test_weave_rejects_an_empty_project() -> None:
    with pytest.raises(AssertionError, match="project"):
        WeaveTracing(project="")


def test_weave_satisfies_the_protocol() -> None:
    backend: TracingBackend = WeaveTracing()
    assert backend.name == "weave"


def test_no_backends_are_available_in_an_empty_environment() -> None:
    assert available_backends({}) == ()


def test_both_backends_are_available_when_both_are_configured() -> None:
    backends = available_backends(LANGSMITH_ENV | WEAVE_ENV)
    assert [b.name for b in backends] == ["langsmith", "weave"]


def test_only_the_configured_backend_is_available() -> None:
    backends = available_backends(WEAVE_ENV)
    assert [b.name for b in backends] == ["weave"]


def test_langchain_tracer_names_reports_the_installed_tracers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-value")
    assert "LangChainTracer" in tracing_module.langchain_tracer_names()
