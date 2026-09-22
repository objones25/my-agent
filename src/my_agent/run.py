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
from langchain_core.outputs import LLMResult
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
    "RESUME_LIMIT",
    "RUN_DEADLINE_S",
    "TOKEN_LIMIT",
    "DeadlineExceeded",
    "Decision",
    "Invokable",
    "ResumeLimitExceeded",
    "RunBounds",
    "RunDeadline",
    "RunTokenBudget",
    "StepLimitExceeded",
    "TokenLimitExceeded",
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

Spent across every invocation one turn takes, pauses included: `TurnResult`
carries what has gone, and `resume_turn` gets the remainder rather than a fresh
budget. Only time the *agent* spends counts — a human deliberating over an
approval is not charged for it, because this bounds how long the agent may run,
not how long a conversation may stay open.
"""

RESUME_LIMIT = 3
"""Times one paused turn may be resumed.

**The bound the other two cannot express.** `deadline_s` now carries across a
pause, but a human who approves promptly never spends it; `step_limit` is
langgraph's `recursion_limit`, which counts supersteps *within one invocation*
and restarts at full on the next, and the count it reached is not reported back
in any form this module can read. So a turn paused and approved repeatedly gets
a complete step budget every time, and nothing counts the halves.

This does. Three is the same reasoning as `capabilities.TASK_DISPATCH_LIMIT`:
the worst case becomes four step budgets rather than unboundedly many, and a
turn that needs a fourth round of human approval is a turn that should be
restarted rather than extended. Zero is a legal value and means a turn may pause but never be
resumed.
"""


TOKEN_LIMIT = 500_000
"""Tokens one turn may spend, summed across every model call it makes.

**The bound the other three cannot express.** `step_limit` counts graph steps,
`deadline_s` counts seconds and `resume_limit` counts halves; a turn is billed
for none of those. The gap is not theoretical: F30 measured ~2,090 input tokens
on a one-line prompt because the tool schemas are resent every call, and with
`capabilities.COMPACTION_TRIGGER_TOKENS` at 96,000 a busy turn can legitimately
carry a conversation forty times that size into each of its calls.

The arithmetic this sits against: 25 steps buys 12 parent round trips, and
`TASK_DISPATCH_LIMIT` (3) dispatches of a 25-step subagent buys ~37 model calls
in one turn. At the compaction ceiling that is ~3.5M tokens — inside every
existing bound. Half a million is roughly 5% of that worst case and something
like seventy times an ordinary tool-using turn, so it never bites on real work
and does bite on a runaway.

A number to revisit with a domain, like the rest: it is the one bound here whose
right value depends on what a turn is worth.
"""


class TokenLimitExceeded(RuntimeError):
    """A run spent every token `RunBounds.token_limit` allowed it.

    An *operating* error, like the other three: a model that kept talking is the
    outside world being expensive, not a caller of ours passing something
    impossible.
    """


@dataclass(frozen=True, slots=True)
class RunBounds:
    """Everything one turn is allowed to consume.

    A parameter object rather than four more keyword arguments, for the reason
    `AgentConfig` is one: a bound added later is one new field with a default,
    and `run_turn`'s signature does not change. `token_limit` arrived exactly
    that way — a number here and a callback beside `RunDeadline`, with no call
    site changed.

    Four bounds because they count four different things and a turn can exhaust
    any one of them while the other three are comfortable: steps, seconds,
    tokens, and the halves a pause splits a turn into. Two of them span a
    pause (`deadline_s`, `token_limit`), one cannot (`step_limit` — see
    `RESUME_LIMIT`), and one exists because of that.

    Frozen, so a bound cannot be widened after it has been validated.
    """

    step_limit: int = RECURSION_LIMIT
    """Graph steps one *invocation* may take, sent as langgraph's
    `recursion_limit`. Per invocation and not per turn, on purpose: langgraph
    restarts the count on a resume and never reports what it reached, so a
    remainder cannot be computed the way `deadline_s`'s is. `resume_limit` is
    what bounds the total instead."""

    deadline_s: float = RUN_DEADLINE_S
    """Wall clock the *turn* may spend, across every invocation it takes."""

    token_limit: int = TOKEN_LIMIT
    """Tokens the *turn* may spend, across every model call and every
    invocation — subagent calls included, because callbacks reach them."""

    resume_limit: int = RESUME_LIMIT
    """Times the turn may be resumed after a pause."""

    def __post_init__(self) -> None:
        require(
            self.step_limit >= 1, f"step_limit must permit at least one step, got {self.step_limit}"
        )
        require(self.deadline_s > 0.0, f"deadline_s must be positive, got {self.deadline_s}")
        require(
            self.token_limit >= 1,
            f"token_limit must permit at least one token, got {self.token_limit}",
        )
        require(
            self.resume_limit >= 0,
            f"resume_limit must not be negative, got {self.resume_limit}; zero means a turn "
            f"may pause but never be resumed",
        )


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


class ResumeLimitExceeded(RuntimeError):
    """One turn was resumed as often as `RunBounds.resume_limit` allowed.

    An *operating* error, for the same reason the other two are: a human who
    keeps approving is the outside world being persistent, not a caller of ours
    passing something impossible.
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

    elapsed_s: float = 0.0
    """Wall clock the *agent* has spent on this turn, across every invocation.

    Carried rather than recomputed, because the thing being bounded spans a
    pause: `resume_turn` subtracts this from `RunBounds.deadline_s` and runs the
    second half on the remainder. Time a human spends deciding is not in here —
    the clock is read when an invocation starts and when it returns, so the gap
    between them is not charged to the agent.
    """

    tokens: int = 0
    """Tokens the turn has spent, across every model call and every invocation.

    Carried for the reason `elapsed_s` is: the bound spans a pause, so
    `resume_turn` subtracts this from `RunBounds.token_limit` and finishes on
    the remainder. Unlike the wall clock there is nothing a human can do to
    make this number grow while they think.
    """

    files: Mapping[str, Any] = field(default_factory=dict)
    """The agent's filesystem as it stood when the turn returned.

    **The read-back this harness already had.** deepagents' `StateBackend` keeps
    the filesystem in graph state, so every `invoke` returns `files` beside
    `messages`; `_invoke` took the messages and dropped the rest, and the mirror
    records the state's key *names* only. So the one deterministic artifact a
    turn produces — the object the agent claims to have created — was visible
    nowhere, while `CLAUDE.md` argued the verification ladder was blocked on a
    domain it does not need (F40).

    A fact about the run, like `failed_tool_calls`, not a judgement: nothing here
    compares it against what the model *said* it wrote, because what the agent
    should have produced is the domain's question. What it did produce is this.

    Empty means the graph reported no filesystem — a turn that wrote nothing, or
    a graph with no `StateBackend` at all. Those are not distinguished, so an
    emptiness assertion on its own proves nothing; pair it with a turn that does
    write, the way `require_withheld` refuses a vacuous absence.
    """

    resumes: int = 0
    """How many times this turn has already been resumed.

    The only bound on the total work one paused turn may do. `step_limit` cannot
    supply one — see `RESUME_LIMIT` — so this is counted here and checked by
    `resume_turn`.
    """

    @property
    def paused(self) -> bool:
        """Whether the graph stopped waiting for a human decision."""
        return len(self.interrupts) > 0

    @property
    def failed_tool_calls(self) -> tuple[ToolMessage, ...]:
        """Every tool call in this turn that came back an error.

        **The outcome that used to be invisible.** A turn has three shapes — it
        finished, it failed, it paused — and none of them says whether the work
        the agent describes actually happened. `status="error"` is langchain's
        authoritative signal and covers all three ways a call comes back
        unfulfilled: the tool itself failed, a `FilesystemPermission` denied it,
        or one of `capabilities.call_limits` blocked it.

        That last one is why this is a property of the *turn* rather than
        something a caller greps the log for. Both call limits run
        `exit_behavior="continue"`: the exceeded call is replaced by an error
        `ToolMessage` and the agent answers with what it already has, usually
        without mentioning the difference. Measured 2026-09-18 against a
        six-wide fan-out — twenty-four calls executed, six blocked, and the
        final message read "All done! I wrote every file." A caller reading
        `paused` and `interrupts` saw a clean turn.

        Deliberately not a `require()`: a failed tool call is not a violated
        contract of ours, it is a fact about the run that the caller decides
        what to do with. `main` prints them; a domain with a success criterion
        is what would eventually fail on them.
        """
        return tuple(m for m in self.messages if isinstance(m, ToolMessage) and m.status == "error")

    @property
    def answered(self) -> bool:
        """False when the model was cut off before its answer began.

        F36: under harmony a response opens an analysis channel, reasons, then
        opens a final channel. `finish_reason == "length"` with neither text nor
        tool calls means the cap landed inside the reasoning and the final
        channel was never opened — the answer did not start, so there is nothing
        to have been truncated. A fact about the run, like `failed_tool_calls`,
        not a judgement of the prose.
        """
        last = self.messages[-1] if self.messages else None
        if not isinstance(last, AIMessage):
            return True
        if last.response_metadata.get("finish_reason") != "length":
            return True
        return bool(last.text) or bool(last.tool_calls)

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

    @property
    def budget_s(self) -> float:
        """What this invocation was given. Read back rather than assumed: on a
        resume it is the *remainder* of the turn's budget, and a resume that
        silently got a fresh one is the defect this property exists to catch."""
        return self._budget_s

    @property
    def elapsed_s(self) -> float:
        """Wall clock since construction. What `TurnResult.elapsed_s` accumulates,
        and therefore what the next resume has subtracted from its budget."""
        elapsed = self._clock() - self._started
        require(elapsed >= 0.0, f"clock ran backwards: {elapsed}s elapsed since the run started")
        return elapsed

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


class RunTokenBudget(BaseCallbackHandler):
    """Stops a run once it has spent the tokens `RunBounds.token_limit` allowed.

    The same shape as `RunDeadline`, deliberately: it accumulates on the hook
    that reports a completed call and refuses on the hook that starts the next
    one, so the spend that crossed the line is paid for and the one after it is
    not. That is the only granularity available — a token count exists once the
    call has returned.

    Counts every model call the run makes, subagents included: langgraph's
    `ensure_config` seeds a subagent's run from the ambient parent config, so
    the handlers reach it. That matters more here than for the wall clock,
    because a `task` dispatch is where the tokens actually go.
    """

    run_inline = True
    """Count on the main thread, so the total is accumulated in call order."""

    raise_error = True
    """The default (`False`) makes LangChain swallow exceptions raised in here
    (F12). A budget that degraded silently would be worse than no budget."""

    def __init__(self, allowance: int) -> None:
        require(allowance >= 1, f"allowance must permit at least one token, got {allowance}")
        self._allowance = allowance
        self._tokens = 0
        self._unmeasured_calls = 0

    @property
    def allowance(self) -> int:
        """What this invocation was given. On a resume it is the turn's
        remainder, and a resume that silently got a fresh allowance is the
        defect this property exists to catch."""
        return self._allowance

    @property
    def tokens(self) -> int:
        """Tokens reported so far. What `TurnResult.tokens` accumulates."""
        return self._tokens

    @property
    def unmeasured_calls(self) -> int:
        """Model calls that reported no usage at all, and are therefore not in
        `tokens`.

        A provider that omits usage makes this bound blind, and a blind bound
        that says nothing is worse than none. The router does report it — the
        live F2 check refuses to grade itself without `output_tokens` — so this
        is the counter that would make a silent change visible rather than a
        cause for crashing a turn over someone else's response shape.
        """
        return self._unmeasured_calls

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
        if self._tokens > self._allowance:
            raise TokenLimitExceeded(
                f"run spent {self._tokens} tokens of the {self._allowance} its token_limit "
                f"allowed, and was about to make another model call; raise "
                f"RunBounds.token_limit or lower step_limit"
            )

    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Add up what the call reported.

        `usage_metadata` on the message rather than `response.llm_output`:
        langchain normalises the former across providers and `mirror.py` reads
        the same field, so the number this bounds is the number the log shows.
        """
        measured = False
        for batch in response.generations:
            for generation in batch:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                measured = True
                self._tokens += int(usage.get("total_tokens", 0))
        if not measured:
            self._unmeasured_calls += 1


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
    deadline: RunDeadline,
    budget: RunTokenBudget,
    bounds: RunBounds,
    callbacks: Sequence[BaseCallbackHandler],
    thread_id: str | None,
) -> RunnableConfig:
    """The config every invocation goes out with, bounds attached.

    The two bounds go ahead of the caller's own handlers so a step that never
    runs is never recorded as though it had, and both are built by the caller
    rather than here: on a resume its budget is the turn's *remainder*, which
    this function has no way to know. A `RunnableConfig` rather than a bare
    dict, so a misspelled key is a type error here instead of a silently ignored
    bound at run time.

    `thread_id` is omitted entirely when there is none, rather than sent as
    `None`: a checkpointer-less graph does not want the key, and a checkpointed
    one rejects a null thread rather than inventing a thread of its own.
    """
    handlers: list[BaseCallbackHandler] = [deadline, budget, *callbacks]
    config: RunnableConfig = {"recursion_limit": bounds.step_limit, "callbacks": handlers}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def _invoke(  # noqa: PLR0913 — one parameter per thing an invocation carries:
    # what runs it, what it is sent, how it is configured, what it may consume,
    # which thread it belongs to, the clock the bound is enforced against, and
    # the counter the token bound is enforced against, and what the turn had
    # already spent before this half of it. Folding any pair into an object
    # would hide one of them, the same reason `run_turn` carries its own six.
    agent: Invokable,
    payload: Any,
    config: RunnableConfig,
    *,
    bounds: RunBounds,
    thread_id: str | None,
    deadline: RunDeadline,
    budget: RunTokenBudget,
    spent: TurnResult | None = None,
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
    # Not a `require`: a fake graph in a test carries no filesystem, and a
    # `StateBackend` that has never been written to reports none either. Absence
    # is a legitimate state, so it is defaulted rather than refused.
    raw_files = result.get("files", {})
    files: Mapping[str, Any] = raw_files if isinstance(raw_files, Mapping) else {}

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
    # Elapsed accumulates across the halves of one turn; the resume count is
    # incremented by `resume_turn`, which is the only thing that knows a resume
    # happened. Read off the deadline rather than measured again here, so the
    # number a caller sees is the same one the bound was enforced against.
    return TurnResult(
        messages=messages,
        interrupts=interrupts,
        thread_id=thread_id,
        elapsed_s=(spent.elapsed_s if spent is not None else 0.0) + deadline.elapsed_s,
        tokens=(spent.tokens if spent is not None else 0) + budget.tokens,
        # Not accumulated across a pause the way the budgets are: the filesystem
        # is state, so what the graph reports on the resume is already the whole
        # of it, and adding the earlier half back would double-count a file the
        # agent edited twice.
        files=files,
        resumes=spent.resumes + 1 if spent is not None else 0,
    )


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
    deadline = RunDeadline(bounds.deadline_s)
    budget = RunTokenBudget(bounds.token_limit)
    result = _invoke(
        agent,
        {"messages": sent},
        _run_config(deadline, budget, bounds, callbacks, thread),
        bounds=bounds,
        thread_id=thread,
        deadline=deadline,
        budget=budget,
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

    # The two bounds a resume used to reset.
    #
    # `resume_limit` first, because it is the one that can refuse outright. The
    # wall clock and the token allowance are then the *remainders* of the turn's
    # budgets rather than fresh ones: a turn paused and approved ten times used
    # to get ten complete 600s budgets, which made "pause, then approve" the way
    # around the bound `_invoke`'s own comment claims a resume cannot drop.
    if paused.resumes >= bounds.resume_limit:
        raise ResumeLimitExceeded(
            f"this turn has already been resumed {paused.resumes} time(s), which is every "
            f"resume RunBounds.resume_limit ({bounds.resume_limit}) allows. Each half gets a "
            f"full step_limit of its own — langgraph restarts the count — so an unbounded "
            f"number of resumes is an unbounded turn. Start a new turn, or raise resume_limit"
        )

    remaining_s = bounds.deadline_s - paused.elapsed_s
    if remaining_s <= 0.0:
        raise DeadlineExceeded(
            f"the turn spent its whole {bounds.deadline_s}s deadline ({paused.elapsed_s}s) "
            f"before it was resumed, so there is no budget left to finish it in. Time a human "
            f"spent deciding is not counted here; this is the agent's own wall clock"
        )

    remaining_tokens = bounds.token_limit - paused.tokens
    if remaining_tokens <= 0:
        raise TokenLimitExceeded(
            f"the turn spent its whole {bounds.token_limit}-token allowance ({paused.tokens}) "
            f"before it was resumed, so there is no budget left to finish it in"
        )

    deadline = RunDeadline(remaining_s)
    budget = RunTokenBudget(remaining_tokens)
    config = _run_config(deadline, budget, bounds, callbacks, thread_id)
    # `{"decisions": [...]}`, not a bare list: the middleware subscripts the
    # resume value by name, so a list raises `TypeError: list indices must be
    # integers` from inside langchain (verified 2026-09-18).
    return _invoke(
        agent,
        Command(resume={"decisions": list(decisions)}),
        config,
        bounds=bounds,
        thread_id=thread_id,
        deadline=deadline,
        budget=budget,
        spent=paused,
    )
