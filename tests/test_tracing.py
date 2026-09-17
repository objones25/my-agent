"""Offline tests for the tracing seam.

`weave.init` and LangSmith's env-var cache are both process-global, so every
test here either injects a mapping or monkeypatches, and the cache fixture
guarantees no test leaks its environment into the next one.
"""

from __future__ import annotations

from collections.abc import Iterator

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


def test_langsmith_activate_passes_when_langsmith_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-value")
    LangSmithTracing(project="my-project").activate()  # must not raise


def test_langsmith_activate_fails_loudly_when_langsmith_disagrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole value of the class: a misspelled variable is loud, not silent."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    with pytest.raises(AssertionError, match="tracing off"):
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
