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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, overload, override
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# The only langgraph imports in this module, and both are about a bound this
# module owns rather than about a graph it builds: `GraphRecursionError` is what
# `step_limit` raises when it trips, and `Command`/`Interrupt` are how a paused
# turn is reported and resumed. `run_turn` still accepts any `Invokable`, so
# `tests/test_run.py` still compiles nothing.
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, Interrupt

from my_agent.negative_space import CheckFailed, require

__all__ = [
    "DEFAULT_RUN_BOUNDS",
    "RECURSION_LIMIT",
    "RUN_DEADLINE_S",
    "DeadlineExceeded",
    "Decision",
    "Invokable",
    "RunBounds",
    "RunDeadline",
    "StepLimitExceeded",
    "TurnResult",
    "resume_turn",
    "run_turn",
]

Decision = Mapping[str, Any]
"""One human answer to one approval request.

Shaped by langchain's `HumanInTheLoopMiddleware`: a `type` of `approve`,
`reject`, `edit` or `respond`, plus whatever that type carries. Typed as a
mapping rather than a `TypedDict` because the accepted set is the middleware's
to define and is read off `InterruptOnConfig.allowed_decisions` at run time —
pinning it here would be this module asserting something it cannot check.
"""

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


class StepLimitExceeded(RuntimeError):
    """A run used every step `RunBounds.step_limit` allowed it.

    `RunBounds` owns the step limit, so it owns what happens when the limit
    trips. langgraph signals it with `GraphRecursionError`; letting that escape
    would mean the two bounds in one `RunBounds` failed in two different
    vocabularies, and only one of them was something the edge knew to handle.

    An *operating* error, for the same reason `DeadlineExceeded` is one: a model
    that kept calling tools is the outside world being unhelpful, not a broken
    contract in this code.
    """


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What one turn produced: every message, and any pause still outstanding.

    A turn has three outcomes, not two. It can finish, it can fail — and it can
    *pause*, when a human-in-the-loop rule interrupts a tool call before it
    runs. Before this class existed the third outcome was invisible: an
    interrupted `invoke` returns `__interrupt__` **alongside** messages that
    already grew, so "the agent added something" was satisfied by a turn whose
    tool never ran (verified 2026-09-18, F29). The caller got a partial
    conversation and no way to tell.

    Sequence access is preserved on purpose. Every existing caller wrote
    `messages = run_turn(...)` and then indexed, sliced, iterated or measured
    it, and a new outcome is not a reason to break them: `result[-1]`,
    `len(result)` and `for m in result` all still mean the messages.
    `interrupts` is the only thing a caller has to newly look at, and
    `run_turn`'s own postconditions make sure it cannot be *silently* ignored.
    """

    messages: list[BaseMessage]
    interrupts: tuple[Interrupt, ...] = field(default=())
    thread_id: str | None = None
    """The thread this turn ran under, or `None` when it ran without one.

    Carried rather than asked for again: `resume_turn` has to reach the exact
    checkpoint holding the pending interrupt, and a `thread_id` retyped at the
    resume call site is a `thread_id` that can be retyped wrong — which does not
    fail, it silently starts a second run wearing the first one's name.
    """

    @property
    def paused(self) -> bool:
        """Whether the graph stopped waiting for a human decision."""
        return len(self.interrupts) > 0

    @property
    def action_requests(self) -> tuple[Mapping[str, Any], ...]:
        """Every tool call awaiting approval, flattened across interrupts.

        One interrupt can carry several requests — the middleware batches a
        model turn's tool calls into one pause — so the count that a decision
        list has to match is this one, not `len(interrupts)`.
        """
        requests: list[Mapping[str, Any]] = []
        for interrupt in self.interrupts:
            value = interrupt.value
            if isinstance(value, Mapping):
                found = value.get("action_requests", ())
                if isinstance(found, Sequence):
                    requests.extend(r for r in found if isinstance(r, Mapping))
        return tuple(requests)

    def __iter__(self) -> Iterator[BaseMessage]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    @overload
    def __getitem__(self, index: int) -> BaseMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[BaseMessage]: ...

    def __getitem__(self, index: int | slice) -> BaseMessage | list[BaseMessage]:
        return self.messages[index]


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


_UNKNOWN = object()
"""Sentinel for "this object has no `checkpointer` attribute at all".

Distinct from `None`, which means "it has one and it is unset". Only the second
is something to refuse: a fake graph in a test is not making a claim about
resumability either way.
"""


def _unanswered_tool_calls(messages: Sequence[BaseMessage]) -> list[str]:
    """Ids of tool calls in `messages` that no `ToolMessage` ever answered.

    The one shape invariant a conversation has. A paused turn breaks it by
    construction — the assistant's tool call is in the messages and its result
    is not, because the tool is exactly what the pause is waiting on — so a
    caller who hands a paused turn back as `history` sends the model a call it
    can see went unanswered.
    """
    requested: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            requested.extend(
                call_id for call in message.tool_calls if (call_id := call.get("id")) is not None
            )
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [call_id for call_id in requested if call_id not in answered]


def _run_config(
    bounds: RunBounds,
    callbacks: Sequence[BaseCallbackHandler],
    thread_id: str | None,
) -> RunnableConfig:
    """The config every invocation goes out with, bounds attached.

    The deadline goes ahead of the caller's own handlers so a step that never
    runs is never recorded as though it had. A `RunnableConfig` rather than a
    bare dict, so a misspelled key is a type error here instead of a silently
    ignored bound at run time.

    `thread_id` is omitted entirely when there is none, rather than sent as
    `None`: a checkpointer-less graph does not want the key, and a checkpointed
    one rejects a null thread rather than inventing a thread of its own.
    """
    handlers: list[BaseCallbackHandler] = [RunDeadline(bounds.deadline_s), *callbacks]
    config: RunnableConfig = {"recursion_limit": bounds.step_limit, "callbacks": handlers}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def _invoke(
    agent: Invokable,
    payload: Any,
    config: RunnableConfig,
    bounds: RunBounds,
    thread_id: str | None,
) -> TurnResult:
    """Invoke `agent`, translate the step limit, and check what came back.

    Shared by `run_turn` and `resume_turn` because a resumed turn is a turn: it
    gets the same bounds, the same translation of a tripped step limit, and the
    same reading of `__interrupt__`. A resume path that quietly dropped any of
    those would make "pause, then approve" the way around every limit here.
    """
    try:
        result = agent.invoke(payload, config)
    except GraphRecursionError as exc:
        # Either graph can be the one that ran out. A `task` subagent has its
        # own limit (`capabilities.SUBAGENT_STEP_LIMIT`) and raises through the
        # tool call, aborting the parent turn rather than reporting back — so
        # the message names both rather than guessing, and carries langgraph's
        # own text, which states the limit that actually tripped.
        raise StepLimitExceeded(
            f"run used all {bounds.step_limit} steps its step_limit allowed, or a `task` "
            f"subagent used all of its own, without finishing. Raise RunBounds.step_limit "
            f"or the subagent step limit accordingly. langgraph said: {exc}"
        ) from exc

    require(isinstance(result, dict), f"agent returned a {type(result).__name__}, not a mapping")
    require("messages" in result, f"agent returned no messages key: {sorted(result)}")
    messages: list[BaseMessage] = result["messages"]

    raw = result.get("__interrupt__", ())
    interrupts = tuple(i for i in raw if isinstance(i, Interrupt))
    require(
        len(interrupts) == len(tuple(raw)),
        f"graph reported {len(tuple(raw))} interrupts but only {len(interrupts)} are Interrupt "
        f"objects; langgraph changed how a pause is reported and it can no longer be read",
    )

    # A pause that cannot be resumed is a turn nobody can finish. Resuming needs
    # a checkpointer — langgraph raises `Cannot use Command(resume=...) without
    # checkpointer` (verified 2026-09-18) — so interrupts configured without one
    # produce a dead end rather than an approval. That is a caller of ours
    # misconfiguring an agent, so it crashes rather than being reported.
    checkpointer = getattr(agent, "checkpointer", _UNKNOWN)
    if interrupts and checkpointer is None:
        raise CheckFailed(
            "the graph paused for approval but carries no checkpointer, so the pause can "
            "never be resumed; pass a checkpointer to create_deep_agent or drop the "
            "interrupt-mode permission rules"
        )
    return TurnResult(messages=messages, interrupts=interrupts, thread_id=thread_id)


def run_turn(  # noqa: PLR0913 — six is the whole surface: what to run, what to
    # say, what was said before, who is watching, what it may consume, and which
    # thread it belongs to. Folding any pair into an object would hide one of them.
    agent: Invokable,
    prompt: str,
    *,
    history: Sequence[BaseMessage] | TurnResult = (),
    callbacks: Sequence[BaseCallbackHandler] = (),
    bounds: RunBounds = DEFAULT_RUN_BOUNDS,
    thread_id: str | None = None,
) -> TurnResult:
    """Run one turn against `agent`, bounded, and return the whole conversation.

    `history` is how a conversation continues. The graph `build_agent` compiles
    carries no checkpointer, so langgraph retains nothing between `invoke`
    calls — the prior messages have to be sent again, and what this returns is
    exactly what the next call wants:

        result = run_turn(agent, "first")
        result = run_turn(agent, "second", history=result)

    That is also why the return value is every message rather than only the new
    ones. (Once a `checkpointer` exists, `config={"configurable": {"thread_id":
    ...}}` becomes the other way to do this, and a better one for long
    conversations. `resume_turn` already needs one.)

    **A turn can pause.** With an interrupt-mode `FilesystemPermission` the
    graph stops before running the tool and returns `__interrupt__`; the result
    is then `paused`, its messages hold a tool call with no result, and
    `resume_turn` is what finishes it. Handing a paused result back as `history`
    is refused rather than silently corrupting the conversation.

    The deadline is attached whether or not the caller asked for one: a bound
    that has to be remembered is a bound that will be forgotten.
    """
    require(
        callable(getattr(agent, "invoke", None)),
        f"agent must have an invoke() method, got {type(agent).__name__}; "
        f"build it with build_agent()",
    )
    require(prompt.strip() != "", "prompt must not be blank")
    require(isinstance(bounds, RunBounds), f"bounds must be RunBounds, got {type(bounds).__name__}")
    prior: list[BaseMessage] = list(history)
    for entry in prior:
        require(
            isinstance(entry, BaseMessage),
            f"history entries must be BaseMessage, got {type(entry).__name__}; "
            f"pass what a previous run_turn returned",
        )
    orphaned = _unanswered_tool_calls(prior)
    require(
        not orphaned,
        f"history carries an unanswered tool call ({', '.join(orphaned)}); this is what a "
        f"paused turn looks like, and sending it back would show the model a call it can "
        f"see got no result. Finish the turn with resume_turn() first",
    )

    # A `HumanMessage`, not `{"role": "user", ...}`: langgraph's `add_messages`
    # reducer turns the dict into exactly this object (verified 2026-09-17), and
    # sending it directly means what `run_turn` sends, returns, and accepts as
    # `history` are all the same type. A dict would leave the round trip
    # depending on that coercion.
    # A conversation keeps its thread unless the caller names a different one.
    # Losing it between turns is how a checkpointed run silently forks.
    thread = thread_id if thread_id is not None else getattr(history, "thread_id", None)
    require(
        thread is None or thread.strip() != "",
        "thread_id must not be blank; pass None for a graph with no checkpointer",
    )

    sent: list[BaseMessage] = [*prior, HumanMessage(prompt)]
    result = _invoke(
        agent, {"messages": sent}, _run_config(bounds, callbacks, thread), bounds, thread
    )

    # Postcondition, relative to what was sent rather than a fixed floor: with a
    # history of n, a fixed `len > 1` would pass while the agent contributed
    # nothing at all. A paused turn satisfies it by pausing — which is the point
    # of reading `__interrupt__` here instead of leaving it in the raw state for
    # a caller to notice.
    require(
        len(result.messages) > len(sent) or result.paused,
        f"agent neither added messages of its own nor paused: sent {len(sent)}, "
        f"got {len(result.messages)} back",
    )
    return result


def resume_turn(
    agent: Invokable,
    paused: TurnResult,
    decisions: Sequence[Decision],
    *,
    callbacks: Sequence[BaseCallbackHandler] = (),
    bounds: RunBounds = DEFAULT_RUN_BOUNDS,
) -> TurnResult:
    """Answer the approval requests in `paused` and finish the turn.

    The thread comes off `paused` rather than from the caller. The pending
    interrupt lives in that thread's checkpoint, and an id retyped here could be
    retyped wrong — which does not fail, it starts a fresh run under a name that
    makes it look resumed.

    One decision per request, in order. langchain's `HumanInTheLoopMiddleware`
    reads `interrupt(request)["decisions"]` and zips them onto the requests, so
    a short list misaligns them silently — approval meant for one tool landing
    on another — which is why the count is a precondition rather than trust.
    """
    require(
        callable(getattr(agent, "invoke", None)),
        f"agent must have an invoke() method, got {type(agent).__name__}",
    )
    require(
        isinstance(paused, TurnResult),
        f"paused must be the TurnResult run_turn returned, got {type(paused).__name__}",
    )
    require(paused.paused, "the turn is not paused, so there is nothing to resume")
    require(isinstance(bounds, RunBounds), f"bounds must be RunBounds, got {type(bounds).__name__}")
    thread_id = paused.thread_id
    # A pause is only resumable through the checkpoint that holds it, and that
    # checkpoint is addressed by thread. A paused turn with no thread means the
    # graph was invoked without one, which `_invoke` already refuses when there
    # is no checkpointer — so reaching here means a checkpointed graph ran
    # unthreaded and the pause is unreachable.
    if thread_id is None:
        raise CheckFailed(
            "the paused turn carries no thread_id, so the checkpoint holding its pending "
            "approval cannot be addressed; pass thread_id= to run_turn"
        )

    requests = paused.action_requests
    require(
        len(decisions) == len(requests),
        f"the paused turn carries {len(requests)} approval request(s) but "
        f"{len(decisions)} decision(s) were given; the middleware zips them in order, "
        f"so a mismatch approves the wrong tool call",
    )
    for decision in decisions:
        require(
            isinstance(decision, Mapping) and "type" in decision,
            f"each decision needs a type (approve, reject, edit, respond), got {decision!r}",
        )

    config = _run_config(bounds, callbacks, thread_id)
    # `{"decisions": [...]}`, not a bare list: the middleware subscripts the
    # resume value by name, so a list raises `TypeError: list indices must be
    # integers` from inside langchain (verified 2026-09-18).
    return _invoke(agent, Command(resume={"decisions": list(decisions)}), config, bounds, thread_id)
