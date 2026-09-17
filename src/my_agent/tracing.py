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
    verifies. See the module docstring for why that is the point.
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
        # mypy infers get_env_var's `default: str | None = None` parameter as an
        # Overload rather than a single _lru_cache_wrapper, so it does not see
        # cache_clear; confirmed present and working via `inspect` on the installed
        # wheel (langsmith 0.12.6).
        get_env_var.cache_clear()  # type: ignore[attr-defined]
        require(
            bool(tracing_is_enabled()),
            f"{LANGSMITH_API_KEY_ENV_VAR} and {LANGSMITH_TRACING_ENV_VAR} are set, but "
            f"langsmith reports tracing off; check for a misspelled LANGSMITH_* variable",
        )
