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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol

import weave
from langchain_core.callbacks.manager import CallbackManager
from langsmith.utils import get_env_var, tracing_is_enabled
from weave.trace.context.weave_client_context import get_weave_client

from my_agent.negative_space import require

__all__ = [
    "DEFAULT_WEAVE_PROJECT",
    "LANGSMITH_API_KEY_ENV_VAR",
    "LANGSMITH_PROJECT_ENV_VAR",
    "LANGSMITH_TRACING_ENV_VAR",
    "WANDB_API_KEY_ENV_VAR",
    "WEAVE_PROJECT_ENV_VAR",
    "LangSmithTracing",
    "TracingBackend",
    "TracingMisconfigured",
    "WeaveTracing",
    "available_backends",
    "langchain_tracer_names",
]

LANGSMITH_API_KEY_ENV_VAR = "LANGSMITH_API_KEY"
LANGSMITH_TRACING_ENV_VAR = "LANGSMITH_TRACING"
LANGSMITH_PROJECT_ENV_VAR = "LANGSMITH_PROJECT"

WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"
WEAVE_PROJECT_ENV_VAR = "WEAVE_PROJECT"
DEFAULT_WEAVE_PROJECT = "my-agent"

WEAVE_LANGCHAIN_TRACER = "WeaveTracer"

# Deliberately more permissive than langsmith's own comparison (see `activate`'s
# docstring): narrowing this to exact "true" would make `LANGSMITH_TRACING=True`
# silently not trace, which is exactly the failure this module exists to prevent.
_TRUTHY = frozenset({"true", "1", "yes", "on"})


class TracingMisconfigured(RuntimeError):
    """Tracing looked configured but langsmith reports it is not active.

    An operating error, not a programmer error: everything `activate()` inspects
    came from the environment, not from a caller we own (CLAUDE.md's rule decides
    the category by where the value came from). A composition root is expected to
    catch this, report it, and keep running with tracing off — a misconfigured
    observability backend should degrade loudly, not take down the agent.
    """


class TracingBackend(Protocol):
    """One consumer's needs: the composition root turns a backend on and names it.

    `name` is a `ClassVar` because mypy and pyright disagree about what satisfies
    a protocol variable, and only two spellings satisfy both (verified 2026-09-17,
    mypy 2.3.1 / pyright latest — see `docs/findings.md` F16):

    - this one: a `ClassVar` member, implemented by a `ClassVar`;
    - a read-only `@property` member, implemented by an instance attribute.

    A read-only `@property` here implemented by a `ClassVar` — the obvious
    spelling, and what this protocol used to say — passes mypy and fails pyright.
    The `ClassVar` form is the one that keeps `name` out of the implementations'
    generated `__init__`, so no caller can construct a backend with a name of
    their choosing. The cost is that an implementation must use a `ClassVar` too:
    an instance attribute no longer conforms.
    """

    name: ClassVar[str]

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
        """Verify LangSmith agrees that tracing is on, and say precisely why when it does not.

        `from_env`'s `_TRUTHY` accepts more spellings ("1", "yes", "True", ...) than
        `langsmith.utils.tracing_is_enabled()` does: it compares the raw
        `LANGSMITH_TRACING` string to the literal `"true"`, with no stripping or
        case-folding. That gap is deliberate — narrowing `_TRUTHY` to match would risk
        silently not tracing on a value the user clearly meant as "on" — which means
        this method, not `from_env`, is where a spelling langsmith will not honour has
        to be caught and named. "Check for a misspelled variable" would be wrong here:
        nothing is misspelled, it just is not the one exact string langsmith accepts.
        """
        # `langsmith.utils.get_env_var` is lru_cached. Any lookup made before
        # load_dotenv() ran is remembered for the life of the process, so a
        # cached "absent" would disable tracing no matter what .env says.
        # Clearing is idempotent and makes the check read the real environment.
        # mypy infers get_env_var's `default: str | None = None` parameter as an
        # Overload rather than a single _lru_cache_wrapper, so it does not see
        # cache_clear; confirmed present and working via `inspect` on the installed
        # wheel (langsmith 0.12.6).
        get_env_var.cache_clear()  # type: ignore[attr-defined]
        if tracing_is_enabled():
            return
        # langsmith resolves the flag across *both* the LANGSMITH and LANGCHAIN
        # namespaces (`get_env_var`'s actual default is
        # `namespaces=("LANGSMITH", "LANGCHAIN")`), preferring LANGSMITH_TRACING
        # but falling back to LANGCHAIN_TRACING. Reading only
        # `os.environ.get(LANGSMITH_TRACING_ENV_VAR, "")` can name the wrong
        # variable: with `LANGCHAIN_TRACING=yes` and no `LANGSMITH_*` set, that
        # read comes back `""` while the value langsmith actually used was
        # `"yes"`. `get_env_var("TRACING", default="")` reads the exact value
        # `tracing_is_enabled()` compared, across both namespaces, and the cache
        # was just cleared above so this read is fresh.
        raw = get_env_var("TRACING", default="")
        raise TracingMisconfigured(
            f"LANGSMITH_TRACING/LANGCHAIN_TRACING resolved to {raw!r}, but langsmith "
            f'enables tracing only for the exact string "true" (it compares literally, '
            f"without stripping or case-folding)"
        )


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

        # A programmer error: this asserts what the library did in direct
        # response to our own `weave.init()` call above, not anything read from
        # the environment.
        require(get_weave_client() is not None, "weave.init returned without installing a client")

        # An operating error, not a programmer error: whether the `WeaveTracer`
        # hook installed depends on the `WEAVE_TRACE_LANGCHAIN` environment
        # variable, not on anything this code controls. Verified against the
        # installed wheels: `weave/integrations/langchain/langchain.py`
        # preserves a user-set `WEAVE_TRACE_LANGCHAIN` and passes it to
        # `register_configure_hook`, and `langchain_core/callbacks/manager.py`
        # creates the handler only when that env var is set. A `require()` here
        # would crash the whole CLI over an environment variable
        # (`main._activate_tracing`'s `except CheckFailed: raise` re-raises it)
        # instead of letting the composition root report "tracing: weave
        # FAILED" and keep running, which is the documented policy. A client
        # without the LangChain hook means Weave is on and the agent is still
        # untraced — checking only for the client would pass in that state.
        if WEAVE_LANGCHAIN_TRACER not in langchain_tracer_names():
            raise TracingMisconfigured(
                f"weave.init ran but no {WEAVE_LANGCHAIN_TRACER} is installed; the agent "
                f"would not be traced (is WEAVE_TRACE_LANGCHAIN set to a falsey value?)"
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
