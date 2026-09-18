"""One bounded agent turn.

`build_agent` compiles a graph; nothing about that graph bounds a *run*. This
module owns the bounds that belong to a turn instead of leaving them to whoever
calls `invoke` — a step limit, a wall clock, and the assertion that the graph
actually answered.

Because it is the only sanctioned way to invoke the agent, it has to be able to
express what callers actually need, or they will go around it and lose the
bounds with it. That is why `run_turn` takes a `history`: a seam that could only
do single turns would be bypassed by the second conversation anyone wrote.

**Why this is not in `agent.py`.** Nothing here needs `create_deep_agent`,
`ChatOpenAI`, or a compiled `CompiledStateGraph`: `run_turn` accepts anything
with an `invoke` method, which is the whole of what it calls. So `tests/
test_run.py` builds no model and compiles no graph, and the bounds are testable
without either. A bound that can only be exercised by a live agent turn is a
bound nobody checks.

**Why the bounds live here rather than at the call site.** They were in
`main.py`, which meant every other caller of `build_agent` inherited
langchain-core's defaults by accident rather than choosing ours. `RECURSION_LIMIT`
happens to equal that default today, which is exactly what makes a call-site
bound dangerous: it looks like a decision and behaves like an inheritance.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from my_agent.negative_space import require

__all__ = [
    "DEFAULT_RUN_BOUNDS",
    "RECURSION_LIMIT",
    "RUN_DEADLINE_S",
    "DeadlineExceeded",
    "Invokable",
    "RunBounds",
    "RunDeadline",
    "run_turn",
]

RECURSION_LIMIT = 25
"""Maximum graph steps in one turn. An agent that loops forever is the worst
failure mode here, so the limit is always sent, never inherited.

Equal to langchain-core's own default today (`ensure_config()["recursion_limit"]`
— verified 2026-09-17). That coincidence is the reason this is stated rather
than omitted: an unsent limit is indistinguishable at runtime and silently
becomes whatever the library decides next.
"""

RUN_DEADLINE_S = 600.0
"""Wall-clock budget for one turn, in seconds.

`ModelConfig.timeout` bounds a single HTTP request; nothing bounded the
sequence. At 120s per request, `max_retries=2` and 25 steps, a single `invoke`
could legitimately run for hours. The two bounds compose: one caps a request,
this caps the run.
"""


@dataclass(frozen=True, slots=True)
class RunBounds:
    """Everything one turn is allowed to consume.

    A parameter object rather than two more keyword arguments, for the reason
    `AgentConfig` is one: a bound added later is one new field with a default,
    and `run_turn`'s signature does not change. It is also the seam a token or
    cost ceiling belongs on when one is needed — the accounting would be a
    callback like `RunDeadline`, and the number would be a field here.

    Frozen, so a bound cannot be widened after it has been validated.
    """

    step_limit: int = RECURSION_LIMIT
    deadline_s: float = RUN_DEADLINE_S

    def __post_init__(self) -> None:
        require(
            self.step_limit >= 1, f"step_limit must permit at least one step, got {self.step_limit}"
        )
        require(self.deadline_s > 0.0, f"deadline_s must be positive, got {self.deadline_s}")


DEFAULT_RUN_BOUNDS = RunBounds()
"""The bounds every turn runs under unless a caller says otherwise.

A module-level singleton rather than a `RunBounds()` call in the signature's
default: the object is frozen, so one shared instance cannot be mutated by a
caller, and a default evaluated once is a default a reader can point at.
"""


class DeadlineExceeded(RuntimeError):
    """A run outlived its wall-clock budget.

    An *operating* error, not a broken contract: a slow provider is the outside
    world misbehaving, so the edge reports it rather than crashing with an
    assertion that implies a bug in this code.
    """


class Invokable(Protocol):
    """The whole of what `run_turn` needs from an agent.

    A `typing.Protocol` with positional-only parameters, so a compiled
    `CompiledStateGraph` satisfies it structurally — as does a ten-line fake —
    without either importing this module. Deliberately not `@runtime_checkable`:
    that checks method presence only and `isinstance` against it is slow, so
    `run_turn` uses a `callable(getattr(...))` check instead.

    `input` is `Any` because langgraph types it as its own state TypedDict, and a
    protocol parameter is contravariant: demanding `dict[str, Any]` here would
    make the real graph *fail* to satisfy this (both mypy and pyright agree on
    that one). `config` is typed exactly, which is what earns the check.
    """

    def invoke(self, input: Any, config: RunnableConfig | None = None, /) -> Any: ...


class RunDeadline(BaseCallbackHandler):
    """Aborts a run once its wall-clock budget is spent.

    Enforced between steps, on the hooks that begin one: a model call or a tool
    call. That is the granularity that matters — the bound inside a single
    blocking request is `ModelConfig.timeout`, and this bounds how many of those
    a turn may string together.

    `on_chat_model_start`, not `on_llm_start`: a chat model never emits the
    latter (F14). Both hooks carry `@override`, so a LangChain rename is a type
    error rather than a deadline that silently stops being checked.
    """

    run_inline = True
    """Check on the main thread, so the deadline is evaluated in step order and
    its exception propagates out of `invoke` rather than a worker."""

    raise_error = True
    """The default (`False`) makes LangChain swallow exceptions raised in here
    (F12). A deadline that degraded silently would be worse than no deadline."""

    def __init__(self, budget_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        require(budget_s > 0.0, f"budget must be positive, got {budget_s}")
        self._budget_s = budget_s
        self._clock = clock
        self._started = clock()

    def _require_time_left(self) -> None:
        elapsed = self._clock() - self._started
        # Negative elapsed time means the clock went backwards, which would make
        # the deadline unreachable — a run with no wall-clock bound at all, which
        # is the one thing this class exists to prevent. The clock is injected by
        # us, so a broken one is a programmer error and crashes.
        require(elapsed >= 0.0, f"clock ran backwards: {elapsed}s elapsed since the run started")
        if elapsed > self._budget_s:
            raise DeadlineExceeded(
                f"run exceeded its {self._budget_s}s deadline after {elapsed}s; "
                f"raise deadline_s or lower step_limit"
            )

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._require_time_left()

    @override
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._require_time_left()


def run_turn(
    agent: Invokable,
    prompt: str,
    *,
    history: Sequence[BaseMessage] = (),
    callbacks: Sequence[BaseCallbackHandler] = (),
    bounds: RunBounds = DEFAULT_RUN_BOUNDS,
) -> list[BaseMessage]:
    """Run one turn against `agent`, bounded, and return the whole conversation.

    `history` is how a conversation continues. The graph `build_agent` compiles
    carries no checkpointer, so langgraph retains nothing between `invoke`
    calls — the prior messages have to be sent again, and what this returns is
    exactly the list the next call wants:

        messages = run_turn(agent, "first")
        messages = run_turn(agent, "second", history=messages)

    That is also why the return value is every message rather than only the new
    ones. (Once a `checkpointer` exists, `config={"configurable": {"thread_id":
    ...}}` becomes the other way to do this, and a better one for long
    conversations. It is not built, because nothing here persists anything yet.)

    The deadline is attached whether or not the caller asked for one: a bound
    that has to be remembered is a bound that will be forgotten. It goes ahead
    of the caller's own handlers so a step that never runs is never recorded as
    though it had.
    """
    require(
        callable(getattr(agent, "invoke", None)),
        f"agent must have an invoke() method, got {type(agent).__name__}; "
        f"build it with build_agent()",
    )
    require(prompt.strip() != "", "prompt must not be blank")
    require(isinstance(bounds, RunBounds), f"bounds must be RunBounds, got {type(bounds).__name__}")
    for entry in history:
        require(
            isinstance(entry, BaseMessage),
            f"history entries must be BaseMessage, got {type(entry).__name__}; "
            f"pass what a previous run_turn returned",
        )

    handlers: list[BaseCallbackHandler] = [RunDeadline(bounds.deadline_s), *callbacks]
    # A `RunnableConfig` rather than a bare dict, so a misspelled key is a type
    # error here instead of a silently ignored bound at run time.
    config: RunnableConfig = {"recursion_limit": bounds.step_limit, "callbacks": handlers}
    # A `HumanMessage`, not `{"role": "user", ...}`: langgraph's `add_messages`
    # reducer turns the dict into exactly this object (verified 2026-09-17), and
    # sending it directly means what `run_turn` sends, returns, and accepts as
    # `history` are all the same type. A dict would leave the round trip
    # depending on that coercion.
    sent: list[BaseMessage] = [*history, HumanMessage(prompt)]
    result = agent.invoke({"messages": sent}, config)

    # Postconditions. The graph is ours, so a shape we did not expect is a
    # broken contract rather than bad input — but note that nothing here trusts
    # the *content* of what the model said, only that a turn happened.
    require(isinstance(result, dict), f"agent returned a {type(result).__name__}, not a mapping")
    require("messages" in result, f"agent returned no messages key: {sorted(result)}")
    messages: list[BaseMessage] = result["messages"]
    # Relative to what was sent, not a fixed floor: with a history of n, a
    # fixed `len > 1` would pass while the agent contributed nothing at all.
    require(
        len(messages) > len(sent),
        f"agent added no messages of its own: sent {len(sent)}, got {len(messages)} back",
    )
    return messages
