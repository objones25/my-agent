"""Contract tests for `my_agent.run`.

`build_agent` hands back a compiled graph; nothing in it bounds a *turn*. Every
bound that belongs to a run — the step limit, the wall clock, and the assertion
that the graph actually answered — lives here, which is why the offline tests
never compile a deep agent or build a model. A ten-line fake graph is the whole
dependency, and that is the point of the seam.

One `-m live` test at the end is the exception, and it earns it: a fake graph
echoes back whatever it was handed, so only a real one can show that langgraph's
`add_messages` reducer accepts a `history` of `BaseMessage` objects and that
what `run_turn` returns really is feedable to the next call.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from deepagents import FilesystemPermission
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, Interrupt

from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import TOOL_CALL_LIMIT
from my_agent.mirror import JsonlMirror
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed
from my_agent.run import (
    DEFAULT_RUN_BOUNDS,
    RECURSION_LIMIT,
    RESUME_LIMIT,
    RUN_DEADLINE_S,
    TOKEN_LIMIT,
    BoundExceeded,
    DeadlineExceeded,
    ResumeLimitExceeded,
    RunBounds,
    RunDeadline,
    RunTokenBudget,
    StepLimitExceeded,
    TokenLimitExceeded,
    TurnResult,
    resume_turn,
    run_turn,
)

# --------------------------------------------------------------------------
# Fakes. Real enough to drive the code under test, and nothing more.
# --------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that returns each reading once, then repeats the last.

    Injected rather than patched: a deadline measured against the real clock
    could only be tested with a sleep, which is a flake waiting to happen.
    """

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        if len(self._readings) > 1:
            return self._readings.pop(0)
        return self._readings[0]


def _handlers(config: RunnableConfig | None) -> list[BaseCallbackHandler]:
    """The handler list that actually reached the graph, narrowed.

    `RunnableConfig` types `callbacks` as a list *or* a manager *or* `None`, and
    both type checkers are right to insist we say which one we sent. Read with
    `.get` rather than `[...]`: every `RunnableConfig` key is non-required, and
    pyright (standard) rejects direct subscripting of one where mypy allows it.
    """
    assert config is not None
    callbacks = config.get("callbacks")
    assert isinstance(callbacks, list)
    return callbacks


class FakeGraph:
    """The narrowest thing `run_turn` can accept: something with `.invoke`."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        self.result = result if result is not None else _two_messages()
        # A real graph returns its whole state, not only the messages; this is
        # how a test says so without compiling one.
        if extra_state:
            self.result = {**self.result, **extra_state}
        self.payload: dict[str, Any] | None = None
        self.config: RunnableConfig | None = None

    def invoke(self, payload: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        self.payload = payload
        self.config = config
        return self.result


class SteppingGraph:
    """A graph that reports steps to its callbacks, the way a real one does.

    Used to prove the deadline actually stops a run rather than merely being
    attached to it.
    """

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.completed = 0

    def invoke(self, _payload: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        handlers = _handlers(config)
        for _ in range(self.steps):
            for handler in handlers:
                handler.on_chat_model_start({}, [[]], run_id=uuid4())
            self.completed += 1
        return _two_messages()


def _two_messages() -> dict[str, list[BaseMessage]]:
    return {"messages": [HumanMessage("ping"), AIMessage("pong")]}


def _paused_result(sent: int = 1) -> dict[str, Any]:
    """What a real graph returns when a HITL rule pauses it (verified 2026-09-18).

    The shape is the finding: `__interrupt__` arrives *alongside* messages that
    already grew — the assistant's tool call is there, its `ToolMessage` is not.
    A turn that only asserted "the agent added something" would call this a
    success and hand a conversation with an unanswered tool call to the next
    turn.
    """
    messages: list[BaseMessage] = [HumanMessage("write it")]
    messages.extend(HumanMessage(f"filler {i}") for i in range(sent - 1))
    messages.append(
        AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"file_path": "/secrets/k"}, "id": "t1"}],
        )
    )
    return {
        "messages": messages,
        "__interrupt__": [
            Interrupt(
                value={
                    "action_requests": [
                        {"name": "write_file", "args": {"file_path": "/secrets/k"}}
                    ],
                    "review_configs": [
                        {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
                    ],
                },
                id="i1",
            )
        ],
    }


class PausingGraph:
    """Pauses on the first invoke, completes on a `Command(resume=...)`."""

    def __init__(self, *, checkpointer: object = object()) -> None:
        self.checkpointer = checkpointer
        self.resumed_with: Any = None
        self.config: RunnableConfig | None = None

    def invoke(self, payload: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        self.config = config
        if isinstance(payload, Command):
            self.resumed_with = payload.resume
            return {
                "messages": [
                    HumanMessage("write it"),
                    AIMessage(content="", tool_calls=[]),
                    ToolMessage(content="Updated file", tool_call_id="t1", name="write_file"),
                    AIMessage("done"),
                ]
            }
        return _paused_result()


class ExhaustingGraph:
    """A graph that runs out of steps the way langgraph really does."""

    def invoke(self, _payload: Any, _config: RunnableConfig | None = None) -> dict[str, Any]:
        raise GraphRecursionError("Recursion limit of 25 reached without hitting a stop condition")


# --------------------------------------------------------------------------
# run_turn: what reaches the graph
# --------------------------------------------------------------------------


def test_run_turn_sends_the_prompt_as_a_user_message() -> None:
    graph = FakeGraph()

    run_turn(graph, "ping")

    assert graph.payload == {"messages": [HumanMessage("ping")]}


def test_run_bounds_default_to_the_module_constants() -> None:
    """One object, so a new bound — a token budget, say — is one field here and
    no change at any call site."""
    bounds = RunBounds()

    assert bounds.step_limit == RECURSION_LIMIT
    assert bounds.deadline_s == RUN_DEADLINE_S


def test_run_turn_always_states_the_step_limit() -> None:
    """The bound must be sent, not inherited. langchain-core's own default is
    also 25 today, so an unsent limit would look identical at runtime and
    silently become whatever the library decides next."""
    graph = FakeGraph()

    run_turn(graph, "ping")

    assert graph.config is not None
    assert graph.config.get("recursion_limit") == RECURSION_LIMIT


def test_run_turn_honours_a_caller_supplied_step_limit() -> None:
    graph = FakeGraph()

    run_turn(graph, "ping", bounds=RunBounds(step_limit=4))

    assert graph.config is not None
    assert graph.config.get("recursion_limit") == 4


def test_run_turn_passes_the_callers_callbacks_through() -> None:
    """The mirror is only worth having if it is actually attached."""
    graph = FakeGraph()
    mirror = JsonlMirror(io.StringIO())

    run_turn(graph, "ping", callbacks=[mirror])

    assert mirror in _handlers(graph.config)


def test_run_turn_attaches_a_deadline_the_caller_did_not_have_to_ask_for() -> None:
    """A bound nobody has to remember is the only kind that holds."""
    graph = FakeGraph()

    run_turn(graph, "ping")

    assert any(isinstance(h, RunDeadline) for h in _handlers(graph.config))


def test_run_turn_returns_the_messages_the_graph_produced() -> None:
    graph = FakeGraph()

    messages = run_turn(graph, "ping")

    assert [m.text for m in messages] == ["ping", "pong"]


# --------------------------------------------------------------------------
# run_turn: preconditions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [("", "prompt"), ("   ", "prompt"), ("\n\t", "prompt")],
    ids=["empty", "spaces", "whitespace"],
)
def test_run_turn_rejects_a_blank_prompt(prompt: str, expected: str) -> None:
    with pytest.raises(CheckFailed, match=expected):
        run_turn(FakeGraph(), prompt)


@pytest.mark.parametrize("limit", [0, -1], ids=["zero", "negative"])
def test_run_bounds_reject_a_step_limit_that_permits_no_steps(limit: int) -> None:
    with pytest.raises(CheckFailed, match="step_limit"):
        RunBounds(step_limit=limit)


@pytest.mark.parametrize("deadline", [0.0, -1.0], ids=["zero", "negative"])
def test_run_bounds_reject_a_deadline_that_has_already_passed(deadline: float) -> None:
    with pytest.raises(CheckFailed, match="deadline_s"):
        RunBounds(deadline_s=deadline)


def test_run_turn_rejects_something_that_cannot_be_invoked() -> None:
    with pytest.raises(CheckFailed, match="invoke"):
        run_turn(object(), "ping")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# run_turn: postconditions on what the graph answered
# --------------------------------------------------------------------------


def test_run_turn_rejects_a_result_carrying_no_messages() -> None:
    with pytest.raises(CheckFailed, match="messages"):
        run_turn(FakeGraph({"structured_response": "x"}), "ping")


def test_run_turn_rejects_a_graph_that_added_nothing_of_its_own() -> None:
    """One message back is our own prompt echoed. A turn that produced nothing
    is a failure, not an empty success."""
    with pytest.raises(CheckFailed, match="neither added messages"):
        run_turn(FakeGraph({"messages": [HumanMessage("ping")]}), "ping")


# --------------------------------------------------------------------------
# RunDeadline
# --------------------------------------------------------------------------


def test_deadline_allows_a_step_inside_the_budget(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    deadline = RunDeadline(10.0, clock=FakeClock(100.0, 105.0))

    assert_does_not_raise(lambda: deadline.on_chat_model_start({}, [[]], run_id=uuid4()))


def test_deadline_stops_a_model_step_past_the_budget() -> None:
    deadline = RunDeadline(10.0, clock=FakeClock(100.0, 111.0))

    with pytest.raises(DeadlineExceeded, match=re.escape("10.0s deadline")):
        deadline.on_chat_model_start({}, [[]], run_id=uuid4())


def test_deadline_stops_a_tool_step_past_the_budget() -> None:
    """A run can spend its whole life in tools without another model call."""
    deadline = RunDeadline(10.0, clock=FakeClock(100.0, 111.0))

    with pytest.raises(DeadlineExceeded, match=re.escape("10.0s deadline")):
        deadline.on_tool_start({}, "input", run_id=uuid4())


def test_deadline_measures_from_when_it_was_constructed() -> None:
    """Not from the first step: time spent before the first model call is still
    time the run has spent."""
    deadline = RunDeadline(10.0, clock=FakeClock(100.0, 108.0, 112.0))

    deadline.on_chat_model_start({}, [[]], run_id=uuid4())

    with pytest.raises(DeadlineExceeded):
        deadline.on_chat_model_start({}, [[]], run_id=uuid4())


def test_deadline_says_how_long_the_run_had_and_how_long_it_took() -> None:
    deadline = RunDeadline(10.0, clock=FakeClock(0.0, 42.5))

    with pytest.raises(DeadlineExceeded, match=re.escape("after 42.5s")):
        deadline.on_tool_start({}, "input", run_id=uuid4())


def test_deadline_refuses_a_clock_that_runs_backwards() -> None:
    """Negative elapsed time means a deadline that can never be reached — a run
    with no wall-clock bound at all, which is the failure this class exists to
    prevent. The clock is injected by us, so a broken one is a programmer error.
    """
    deadline = RunDeadline(10.0, clock=FakeClock(100.0, 90.0))

    with pytest.raises(CheckFailed, match="backwards"):
        deadline.on_chat_model_start({}, [[]], run_id=uuid4())


def test_deadline_refuses_a_budget_that_permits_nothing() -> None:
    with pytest.raises(CheckFailed, match="budget"):
        RunDeadline(0.0)


def test_deadline_does_not_let_langchain_swallow_its_own_failure() -> None:
    """LangChain catches exceptions raised inside a handler unless `raise_error`
    is set, and runs handlers off the main thread unless `run_inline` is set
    (F12). A deadline that degraded silently would be worse than none."""
    deadline = RunDeadline(10.0)

    assert deadline.raise_error is True
    assert deadline.run_inline is True


def test_deadline_is_an_operating_error_not_a_broken_contract() -> None:
    """A slow router is the outside world misbehaving, so it must be catchable
    at the edge rather than crashing as a violated invariant of ours."""
    assert issubclass(DeadlineExceeded, RuntimeError)
    assert not issubclass(DeadlineExceeded, CheckFailed)


# --------------------------------------------------------------------------
# The two together
# --------------------------------------------------------------------------


def test_run_turn_aborts_a_run_that_outlives_its_deadline() -> None:
    """The claim the seam exists to make, end to end: a graph that keeps taking
    steps is stopped, and stopped part-way rather than after it finishes."""
    graph = SteppingGraph(steps=5)

    with pytest.raises(DeadlineExceeded):
        run_turn(graph, "ping", bounds=RunBounds(deadline_s=1e-9))

    assert graph.completed == 0


def test_run_turn_lets_a_run_inside_its_deadline_finish() -> None:
    """The discriminating half: the same graph, a budget it fits inside."""
    graph = SteppingGraph(steps=5)

    messages = run_turn(graph, "ping", bounds=RunBounds(deadline_s=RUN_DEADLINE_S))

    assert graph.completed == 5
    assert len(messages) == 2


# --------------------------------------------------------------------------
# More than one turn
#
# The graph carries no checkpointer, so langgraph retains nothing between
# `invoke` calls: continuing a conversation means sending the prior messages
# back. `run_turn` returns exactly the list the next call wants as `history`.
# --------------------------------------------------------------------------


class AccumulatingGraph:
    """Appends a reply to whatever history it is given, the way a real turn does."""

    def __init__(self) -> None:
        self.turns = 0

    def invoke(self, payload: Any, _config: RunnableConfig | None = None) -> dict[str, Any]:
        self.turns += 1
        sent = list(payload["messages"])
        return {"messages": [*sent, AIMessage(f"reply {self.turns}")]}


def test_run_turn_sends_prior_messages_ahead_of_the_new_prompt() -> None:
    graph = FakeGraph(
        {"messages": [HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")]}
    )
    history = [HumanMessage("a"), AIMessage("b")]

    run_turn(graph, "c", history=history)

    assert graph.payload is not None
    assert graph.payload["messages"] == [*history, HumanMessage("c")]


def test_run_turn_defaults_to_starting_a_fresh_conversation() -> None:
    graph = FakeGraph()

    run_turn(graph, "ping")

    assert graph.payload is not None
    assert graph.payload["messages"] == [HumanMessage("ping")]


def test_run_turn_returns_a_history_the_next_turn_can_be_handed_straight_back() -> None:
    """The whole multi-turn contract, and the reason the return value is every
    message rather than only the new ones."""
    graph = AccumulatingGraph()

    first = run_turn(graph, "one")
    second = run_turn(graph, "two", history=first)

    assert graph.turns == 2
    assert [m.text for m in second] == ["one", "reply 1", "two", "reply 2"]


def test_run_turn_requires_the_agent_to_add_to_the_history_it_was_given() -> None:
    """The postcondition has to be relative to what was sent. A fixed `len > 1`
    would pass on any non-empty history while the agent contributed nothing."""
    unchanged = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    graph = FakeGraph({"messages": unchanged})

    with pytest.raises(CheckFailed, match="neither added messages"):
        run_turn(graph, "c", history=unchanged[:2])


def test_run_turn_rejects_a_history_that_is_not_messages() -> None:
    """A history is what a previous `run_turn` returned. Anything else would be
    coerced by langgraph or silently dropped, neither of which is visible."""
    with pytest.raises(CheckFailed, match="history"):
        run_turn(FakeGraph(), "ping", history=["a string"])  # type: ignore[list-item]


# --------------------------------------------------------------------------
# HITL against a real compiled graph, offline
# --------------------------------------------------------------------------


class WritesToSecrets(BaseChatModel):
    """Calls `write_file` on a denied path once, then answers.

    A fake graph can show that `run_turn` reads `__interrupt__`; only a real one
    shows that an interrupt-mode `FilesystemPermission` produces it, that the
    pause survives a checkpointer round trip, and that approving actually runs
    the tool that was held.
    """

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "writes-to-secrets"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            call = {
                "name": "write_file",
                "args": {"file_path": "/secrets/k.txt", "content": "x"},
                "id": "t1",
            }
            message = AIMessage(content="", tool_calls=[call])
        else:
            message = AIMessage(content="written")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _interrupting_agent() -> Any:
    rule = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")
    return build_agent(
        WritesToSecrets(), AgentConfig(permissions=[rule], checkpointer=InMemorySaver())
    )


def test_an_interrupt_rule_pauses_a_real_turn_before_the_tool_runs() -> None:
    """`FilesystemPermission(mode="interrupt")` is the whole HITL surface: no
    new config field, and it reaches every subagent too."""
    result = run_turn(_interrupting_agent(), "write it", thread_id="t")

    assert result.paused
    assert [r["name"] for r in result.action_requests] == ["write_file"]
    assert not any(isinstance(m, ToolMessage) for m in result)


def test_approving_a_paused_turn_runs_the_tool_that_was_held() -> None:
    agent = _interrupting_agent()
    paused = run_turn(agent, "write it", thread_id="t")

    resumed = resume_turn(agent, paused, [{"type": "approve"}])

    assert not resumed.paused
    written = [m for m in resumed if isinstance(m, ToolMessage)]
    assert [m.name for m in written] == ["write_file"]


def test_rejecting_a_paused_turn_leaves_the_write_undone() -> None:
    """The half that matters. An approval path that works while rejection
    silently writes anyway is worse than no gate at all."""
    agent = _interrupting_agent()
    paused = run_turn(agent, "write it", thread_id="t")

    resumed = resume_turn(agent, paused, [{"type": "reject"}])

    written = [m for m in resumed if isinstance(m, ToolMessage)]
    assert not any("updated file" in str(m.content).lower() for m in written)


def test_a_turn_with_no_interrupt_rule_never_pauses() -> None:
    """The discriminator: without it, every assertion above would still pass if
    the graph paused on everything, or on nothing and the reader lied."""
    agent = build_agent(WritesToSecrets(), AgentConfig(checkpointer=InMemorySaver()))

    result = run_turn(agent, "write it", thread_id="t")

    assert not result.paused


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


@pytest.mark.live
def test_a_real_agent_continues_a_conversation_across_two_turns() -> None:
    """The multi-turn claim against the real thing.

    The structural assertion is the deterministic one and the one that matters:
    every message from the first turn must still be there, in order, after the
    second. That is what proves langgraph accepted our `history` rather than
    coercing, reordering or dropping it — and it holds whatever the model says.

    The recall assertion is model-dependent, which is allowed here (`live`), and
    is kept to a task no model of this class should fail.
    """
    load_dotenv()
    agent = build_agent(build_model(ModelConfig.from_env()))

    first = run_turn(agent, "Remember the word 'albatross'. Reply with just: ok")
    second = run_turn(agent, "What word did I ask you to remember?", history=first)

    assert [m.text for m in second[: len(first)]] == [m.text for m in first]
    assert len(second) > len(first)
    assert "albatross" in second[-1].text.lower()


# --------------------------------------------------------------------------
# The step limit is an operating error, not a traceback
# --------------------------------------------------------------------------


def test_run_turn_reports_an_exhausted_step_limit_as_an_operating_error() -> None:
    """`RunBounds` owns the step limit, so it owns what happens when it trips.

    langgraph raises `GraphRecursionError`. Letting that escape means the only
    bound in `RunBounds` whose failure the edge handles is the wall clock — the
    same bound reported two different ways depending on which one ran out.
    """
    with pytest.raises(StepLimitExceeded, match="25 steps"):
        run_turn(ExhaustingGraph(), "loop")


def test_step_limit_failure_names_the_limit_that_was_in_force() -> None:
    """The message has to say what to change. A bound that fails anonymously
    sends the reader to the library's docs rather than to `RunBounds`."""
    with pytest.raises(StepLimitExceeded, match="step_limit"):
        run_turn(ExhaustingGraph(), "loop", bounds=RunBounds(step_limit=4))


def test_step_limit_failure_keeps_the_library_error_as_its_cause() -> None:
    """Translating an error must not lose where it came from."""
    with pytest.raises(StepLimitExceeded) as caught:
        run_turn(ExhaustingGraph(), "loop")

    assert isinstance(caught.value.__cause__, GraphRecursionError)


def test_step_limit_is_an_operating_error_not_a_broken_contract() -> None:
    """The same split `DeadlineExceeded` makes: a run that hit a real ceiling is
    the outside world being slow or the model looping, not a bug in this code."""
    assert not issubclass(StepLimitExceeded, CheckFailed)
    assert issubclass(StepLimitExceeded, RuntimeError)


# --------------------------------------------------------------------------
# A paused turn is an outcome, not a completed turn
# --------------------------------------------------------------------------


def test_run_turn_reports_a_turn_the_graph_paused() -> None:
    """The defect this closes: an interrupted invoke returns `__interrupt__`
    *and* more messages than were sent, so "the agent added something" is
    satisfied by a turn whose tool never ran."""
    result = run_turn(PausingGraph(), "write it")

    assert result.paused


def test_run_turn_carries_the_approval_requests_back_to_the_caller() -> None:
    """A pause the caller cannot act on is the same as a hang."""
    result = run_turn(PausingGraph(), "write it")

    assert [r["name"] for r in result.interrupts[0].value["action_requests"]] == ["write_file"]


def test_a_completed_turn_is_not_paused_and_carries_no_interrupts() -> None:
    result = run_turn(FakeGraph(), "ping")

    assert not result.paused
    assert result.interrupts == ()


def test_run_turn_still_reads_as_the_message_list_every_caller_already_uses() -> None:
    """`TurnResult` is a new outcome, not a new calling convention: indexing,
    slicing, iteration and `len` all still mean the messages."""
    result = run_turn(FakeGraph(), "ping")

    assert len(result) == 2
    assert result[-1].text == "pong"
    assert [m.text for m in result[:1]] == ["ping"]
    assert result.messages == list(result)


def test_run_turn_refuses_a_pause_on_a_graph_that_could_never_resume() -> None:
    """Resuming needs a checkpointer (langgraph raises without one), so a graph
    that pauses without one has produced a turn nobody can finish. That is a
    misconfiguration by a caller we own, not the outside world misbehaving."""
    with pytest.raises(CheckFailed, match="checkpointer"):
        run_turn(PausingGraph(checkpointer=None), "write it")


# --------------------------------------------------------------------------
# resume_turn
# --------------------------------------------------------------------------


def test_resume_turn_sends_the_decisions_in_the_shape_the_middleware_reads() -> None:
    """Verified against langchain's `HumanInTheLoopMiddleware` 2026-09-18: it
    does `interrupt(request)["decisions"]`, so a bare list raises `TypeError:
    list indices must be integers` from inside the library."""
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    resume_turn(graph, paused, [{"type": "approve"}])

    assert graph.resumed_with == {"decisions": [{"type": "approve"}]}


def test_resume_turn_completes_the_turn_the_pause_left_unfinished() -> None:
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    resumed = resume_turn(graph, paused, [{"type": "approve"}])

    assert not resumed.paused
    assert [m.type for m in resumed] == ["human", "ai", "tool", "ai"]


def test_resume_turn_reaches_the_thread_the_pause_was_recorded_under() -> None:
    """Read off the paused turn, never retyped. A `thread_id` the caller
    supplies again can be supplied wrong, and a wrong one does not fail — it
    starts a second run under a name that makes it look resumed."""
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="thread-9")

    resume_turn(graph, paused, [{"type": "approve"}])

    assert graph.config is not None
    assert graph.config.get("configurable", {}).get("thread_id") == "thread-9"


def test_resume_turn_still_bounds_the_run_it_is_finishing() -> None:
    """A resumed turn is a turn. Dropping the bounds here would make "pause,
    approve" the way around every limit `run_turn` enforces."""
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    resume_turn(graph, paused, [{"type": "approve"}], bounds=RunBounds(step_limit=7))

    assert graph.config is not None
    assert graph.config.get("recursion_limit") == 7
    assert any(isinstance(h, RunDeadline) for h in _handlers(graph.config))


def test_resume_turn_refuses_a_turn_that_was_never_paused() -> None:
    """Resuming a finished turn would replay it against a checkpoint that holds
    no pending interrupt — a second run wearing the first one's thread id."""
    with pytest.raises(CheckFailed, match="not paused"):
        resume_turn(PausingGraph(), run_turn(FakeGraph(), "ping"), [{"type": "approve"}])


def test_resume_turn_refuses_a_pause_that_ran_without_a_thread() -> None:
    """The checkpoint holding the approval is addressed by thread. Without one
    there is nothing to address, and resuming would start a fresh run."""
    graph = PausingGraph()
    paused = run_turn(graph, "write it")

    with pytest.raises(CheckFailed, match="no thread_id"):
        resume_turn(graph, paused, [{"type": "approve"}])


def test_resume_turn_refuses_a_decision_count_that_does_not_match_the_requests() -> None:
    """The middleware zips decisions onto action requests. A short list
    misaligns them silently: approval meant for one tool lands on another."""
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    with pytest.raises(CheckFailed, match="1 approval request"):
        resume_turn(graph, paused, [{"type": "approve"}, {"type": "reject"}])


def test_resume_turn_refuses_a_decision_with_no_type() -> None:
    graph = PausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    with pytest.raises(CheckFailed, match="type"):
        resume_turn(graph, paused, [{"approve": True}])


def test_run_turn_keeps_a_conversation_on_the_thread_it_started_on() -> None:
    """A thread lost between turns is a checkpointed conversation silently
    forking into two."""
    graph = AccumulatingGraph()

    first = run_turn(graph, "one", thread_id="t-1")
    second = run_turn(graph, "two", history=first)

    assert second.thread_id == "t-1"


def test_run_turn_rejects_a_blank_thread_id() -> None:
    with pytest.raises(CheckFailed, match="thread_id"):
        run_turn(FakeGraph(), "ping", thread_id="  ")


# --------------------------------------------------------------------------
# History hygiene: what a paused turn must not be allowed to become
# --------------------------------------------------------------------------


def test_run_turn_refuses_a_history_with_an_unanswered_tool_call() -> None:
    """The corruption path a paused turn opens. `run_turn` returns the messages
    of a paused turn, and the assistant's tool call in them has no
    `ToolMessage`. Feeding that back as `history` sends the model a tool call it
    can see it never got a result for — the one message-shape invariant a
    conversation has."""
    paused = run_turn(PausingGraph(), "write it")

    with pytest.raises(CheckFailed, match="unanswered tool call"):
        run_turn(FakeGraph(), "next", history=paused)


def test_run_turn_accepts_a_history_whose_tool_calls_were_all_answered() -> None:
    graph = FakeGraph(
        {
            "messages": [
                HumanMessage("a"),
                AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "t1"}]),
                ToolMessage(content="[]", tool_call_id="t1", name="ls"),
                AIMessage("done"),
                HumanMessage("b"),
                AIMessage("ok"),
            ]
        }
    )
    answered: list[BaseMessage] = [
        HumanMessage("a"),
        AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "t1"}]),
        ToolMessage(content="[]", tool_call_id="t1", name="ls"),
        AIMessage("done"),
    ]

    result = run_turn(graph, "b", history=answered)

    assert result[-1].text == "ok"


def test_run_turn_names_the_tool_call_that_was_left_unanswered() -> None:
    """A rejection nobody can act on is a worse failure than the corruption."""
    orphaned: list[BaseMessage] = [
        HumanMessage("a"),
        AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "t9"}]),
    ]

    with pytest.raises(CheckFailed, match="t9"):
        run_turn(FakeGraph(), "b", history=orphaned)


# --------------------------------------------------------------------------
# A tool call that failed is an outcome too
#
# `exit_behavior="continue"` on the call limits means an exceeded call is
# blocked and the agent answers with what it already has. Before these, the
# only record of that was the mirror — a file for a human — so a turn cut short
# by the harness's own ceiling returned exactly like a clean one.
# --------------------------------------------------------------------------


class ToolFailingGraph:
    """A graph whose tool call came back as an error, the way a real one does."""

    def invoke(self, _payload: Any, _config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage("write it"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {}, "id": "t1"}],
                ),
                ToolMessage(
                    content="Error: permission denied",
                    tool_call_id="t1",
                    name="write_file",
                    status="error",
                ),
                AIMessage("All done!"),
            ]
        }


def test_a_turn_reports_the_tool_calls_that_came_back_as_errors() -> None:
    """The claim: a caller can tell that the agent's confident summary is not
    backed by the work it describes, without reading the log file."""
    result = run_turn(ToolFailingGraph(), "write it")

    assert [m.name for m in result.failed_tool_calls] == ["write_file"]


def test_a_turn_whose_tools_all_succeeded_reports_no_failures() -> None:
    """The discriminator. Without it, `failed_tool_calls` returning everything
    would satisfy the test above."""
    graph = FakeGraph(
        {
            "messages": [
                HumanMessage("a"),
                AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "t1"}]),
                ToolMessage(content="[]", tool_call_id="t1", name="ls"),
                AIMessage("done"),
            ]
        }
    )

    result = run_turn(graph, "a")

    assert result.failed_tool_calls == ()


def test_a_completed_turn_with_no_tools_at_all_reports_no_failures() -> None:
    assert run_turn(FakeGraph(), "ping").failed_tool_calls == ()


def test_a_turn_cut_off_before_the_answer_began_is_not_answered() -> None:
    """F36: `finish_reason == "length"` with no text and no tool calls means the
    model never opened its final channel. Printing that as an empty reply and
    exiting 0 reports a turn that did not happen."""
    cut_off = AIMessage(
        "",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "input_tokens": 84,
            "output_tokens": 24,
            "total_tokens": 108,
            "output_token_details": {"reasoning": 21},
        },
    )
    result = TurnResult(messages=[HumanMessage("count to 200"), cut_off])

    assert result.answered is False


def test_an_ordinary_turn_is_answered() -> None:
    """The discriminator. Without it `answered` hardwired to `False` passes."""
    result = TurnResult(messages=[HumanMessage("hi"), AIMessage("hello")])

    assert result.answered is True


def test_a_turn_cut_off_after_calling_a_tool_is_answered() -> None:
    """A length-capped turn that still produced a tool call did real work. Only
    the no-text-and-no-calls combination means nothing started."""
    cut_off = AIMessage(
        "",
        tool_calls=[{"name": "ls", "args": {}, "id": "c1"}],
        response_metadata={"finish_reason": "length"},
    )
    result = TurnResult(messages=[HumanMessage("list files"), cut_off])

    assert result.answered is True


def test_a_turn_with_no_messages_at_all_is_answered() -> None:
    """Pins the empty-list edge: `answered` must not index `messages[-1]`
    unguarded. There is no cut-off answer to report when there is no turn."""
    result = TurnResult(messages=[])

    assert result.answered is True


def test_a_turn_that_ended_on_a_tool_result_is_answered() -> None:
    """Pins the trailing-non-`AIMessage` edge: a turn that ended on a
    `ToolMessage` (mid-conversation, not yet the model's turn to reply) is not
    the same shape as a model cut off before its answer began, so it reads as
    answered rather than as the F36 case."""
    result = TurnResult(
        messages=[
            HumanMessage("hi"),
            AIMessage("x"),
            ToolMessage("ok", tool_call_id="1", name="ls"),
        ]
    )

    assert result.answered is True


class FanningOutModel(BaseChatModel):
    """A model that asks for `width` tools a turn until it is cut off.

    Exists because the call limits are middleware: nothing about them is
    reachable through a fake *graph*, and the failure they produce — a blocked
    call the agent then talks over — only appears on a real compiled agent.
    """

    width: int = 6
    turns_before_answering: int = 5
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fanning-out"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        if self.calls > self.turns_before_answering:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage("All done! I wrote every file."))]
            )
        tool_calls = [
            {
                "name": "write_file",
                "args": {"file_path": f"/f{self.calls}-{i}.txt", "content": "x"},
                "id": f"c{self.calls}-{i}",
            }
            for i in range(self.width)
        ]
        answer = AIMessage("", tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=answer)])


def test_a_turn_stopped_by_the_tool_call_limit_says_so() -> None:
    """The regression this property exists for, end to end on a real graph.

    Measured 2026-09-18 before it was added: thirty tool calls requested,
    `TOOL_CALL_LIMIT` executed, the rest blocked, and the agent signing off with
    "All done!" — while `run_turn` returned a result whose only outcome fields
    were `paused` and `interrupts`, both of them clean.
    """
    result = run_turn(build_agent(FanningOutModel()), "write me thirty files")

    executed = [m for m in result.messages if isinstance(m, ToolMessage) and m.status != "error"]
    assert len(executed) == TOOL_CALL_LIMIT
    assert len(result.failed_tool_calls) == 6
    assert not result.paused


def test_a_turn_inside_the_tool_call_limit_blocks_nothing() -> None:
    """The discriminator: the same graph and the same model, fanning out
    narrowly enough to stay inside the ceiling."""
    model = FanningOutModel(width=2, turns_before_answering=3)

    result = run_turn(build_agent(model), "write me six files")

    assert result.failed_tool_calls == ()


# --------------------------------------------------------------------------
# A pause must not be a way around the bounds
#
# Every bound `run_turn` applies is per *invocation*: a fresh `RunDeadline`
# starting now, and `recursion_limit` sent again in full. So a turn paused and
# approved ten times used to get ten complete budgets, and "a turn is bounded"
# quietly stopped being true the moment the human gate was used.
# --------------------------------------------------------------------------


def _budget_of(config: RunnableConfig | None) -> RunTokenBudget:
    """The token budget that actually reached the graph, so its allowance can be
    read back — on a resume it is the turn's remainder, not a fresh one."""
    return next(h for h in _handlers(config) if isinstance(h, RunTokenBudget))


def _deadline_of(config: RunnableConfig | None) -> RunDeadline:
    """The deadline that actually reached the graph, so its budget can be read
    back. A resume that silently got a fresh one is the defect being tested."""
    return next(h for h in _handlers(config) if isinstance(h, RunDeadline))


class AlwaysPausingGraph:
    """Pauses on every invoke, including the resumes. Real enough to exhaust a
    resume bound with, which one pause followed by one completion cannot."""

    def __init__(self) -> None:
        self.checkpointer = object()
        self.config: RunnableConfig | None = None
        self.invocations = 0

    def invoke(self, _payload: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        self.config = config
        self.invocations += 1
        return _paused_result()


def test_a_finished_turn_reports_the_wall_clock_it_consumed() -> None:
    """Without this the turn's own spend is unreadable, and `resume_turn` has
    nothing to subtract a remaining budget from."""
    result = run_turn(SteppingGraph(steps=3), "ping")

    assert result.elapsed_s > 0.0


def test_a_resumed_turn_gets_only_the_wall_clock_the_pause_left_over() -> None:
    """The bound made whole. 600s spent before the pause plus a fresh 600s after
    it is a 1200s turn that every docstring here calls 600s."""
    graph = PausingGraph()
    paused = TurnResult(messages=run_turn(graph, "write it", thread_id="t").messages,
                        interrupts=run_turn(graph, "write it", thread_id="t").interrupts,
                        thread_id="t",
                        elapsed_s=550.0)

    resume_turn(graph, paused, [{"type": "approve"}], bounds=RunBounds(deadline_s=600.0))

    assert _deadline_of(graph.config).budget_s == pytest.approx(50.0)


def test_a_resume_is_refused_once_the_turn_has_spent_its_whole_budget() -> None:
    """A remaining budget of zero is not a very short deadline, it is a turn
    that is already over — and `RunDeadline` refuses a non-positive budget, so
    the failure has to be named here or it arrives as a broken contract."""
    graph = PausingGraph()
    paused = TurnResult(
        messages=[HumanMessage("write it")],
        interrupts=run_turn(graph, "write it", thread_id="t").interrupts,
        thread_id="t",
        elapsed_s=600.0,
    )

    with pytest.raises(DeadlineExceeded, match="before it was resumed"):
        resume_turn(graph, paused, [{"type": "approve"}], bounds=RunBounds(deadline_s=600.0))


def test_a_resumed_turn_adds_its_own_time_to_the_turns_running_total() -> None:
    """Elapsed accumulates across the halves of one turn, which is what makes
    the subtraction above correct on the second resume as well as the first."""
    graph = PausingGraph()
    paused = TurnResult(
        messages=[HumanMessage("write it")],
        interrupts=run_turn(graph, "write it", thread_id="t").interrupts,
        thread_id="t",
        elapsed_s=42.0,
    )

    resumed = resume_turn(graph, paused, [{"type": "approve"}])

    assert resumed.elapsed_s > 42.0


def test_each_resume_counts_towards_the_bound_on_how_many_a_turn_may_have() -> None:
    graph = AlwaysPausingGraph()
    first = run_turn(graph, "write it", thread_id="t")

    second = resume_turn(graph, first, [{"type": "approve"}])
    third = resume_turn(graph, second, [{"type": "approve"}])

    assert (first.resumes, second.resumes, third.resumes) == (0, 1, 2)


def test_a_turn_may_not_be_resumed_more_often_than_its_bound_allows() -> None:
    """The bound the wall clock cannot express: a human who approves promptly
    every time never spends the deadline, and the step limit restarts at full
    on each half. Something has to count the halves."""
    graph = AlwaysPausingGraph()
    result = run_turn(graph, "write it", thread_id="t")

    for _ in range(RESUME_LIMIT):
        result = resume_turn(graph, result, [{"type": "approve"}])

    with pytest.raises(ResumeLimitExceeded, match=str(RESUME_LIMIT)):
        resume_turn(graph, result, [{"type": "approve"}])


def test_a_turn_inside_the_resume_bound_is_still_resumable() -> None:
    """The discriminator: a bound that refused the first resume would satisfy
    the test above just as well."""
    graph = AlwaysPausingGraph()
    paused = run_turn(graph, "write it", thread_id="t")

    resumed = resume_turn(graph, paused, [{"type": "approve"}])

    assert resumed.paused
    assert graph.invocations == 2


def test_run_bounds_default_the_resume_limit_to_the_module_constant() -> None:
    assert RunBounds().resume_limit == RESUME_LIMIT


@pytest.mark.parametrize("limit", [-1, -25])
def test_run_bounds_reject_a_resume_limit_below_zero(limit: int) -> None:
    """Zero is meaningful — a turn that may pause but never be resumed — so the
    floor is zero rather than one."""
    with pytest.raises(CheckFailed, match="resume_limit"):
        RunBounds(resume_limit=limit)


def test_the_resume_limit_is_an_operating_error_not_a_broken_contract() -> None:
    """A human who keeps approving is the outside world being persistent, not a
    caller of ours passing something impossible. Reported at the edge, like the
    other two bounds in `RunBounds`."""
    assert issubclass(ResumeLimitExceeded, RuntimeError)
    assert not issubclass(ResumeLimitExceeded, CheckFailed)


# --------------------------------------------------------------------------
# The bound the other three cannot express: what the turn costs
#
# `step_limit` counts graph steps, `deadline_s` counts seconds and
# `resume_limit` counts halves. None of them counts tokens, and tokens are what
# a turn is billed for — a fan-out of cheap steps against a 96,000-token
# conversation is a large bill inside every existing bound.
# --------------------------------------------------------------------------


def _usage_report(total: int) -> LLMResult:
    """What `on_llm_end` carries back from a real call, narrowed to the field
    the budget reads."""
    message = AIMessage(
        "ok",
        usage_metadata={"input_tokens": total, "output_tokens": 0, "total_tokens": total},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class SpendingGraph:
    """A graph that reports token usage to its callbacks, the way a real one
    does. Used to prove the budget stops a run rather than merely counting it."""

    def __init__(self, calls: int, tokens_each: int) -> None:
        self.calls = calls
        self.tokens_each = tokens_each
        self.completed = 0

    def invoke(self, _payload: Any, config: RunnableConfig | None = None) -> dict[str, Any]:
        handlers = _handlers(config)
        for _ in range(self.calls):
            for handler in handlers:
                handler.on_chat_model_start({}, [[]], run_id=uuid4())
            for handler in handlers:
                handler.on_llm_end(_usage_report(self.tokens_each), run_id=uuid4())
            self.completed += 1
        return _two_messages()


def test_the_token_budget_adds_up_what_each_model_call_reported() -> None:
    budget = RunTokenBudget(1000)

    budget.on_llm_end(_usage_report(120), run_id=uuid4())
    budget.on_llm_end(_usage_report(80), run_id=uuid4())

    assert budget.tokens == 200


def test_the_token_budget_stops_the_next_call_once_the_allowance_is_gone() -> None:
    """Checked when a call is about to start, like the deadline: the spend that
    crossed the line is already paid for, the one after it is not."""
    budget = RunTokenBudget(100)
    budget.on_llm_end(_usage_report(101), run_id=uuid4())

    with pytest.raises(TokenLimitExceeded, match="101"):
        budget.on_chat_model_start({}, [[]], run_id=uuid4())


def test_the_token_budget_counts_calls_that_reported_no_usage_at_all() -> None:
    """A provider that omits usage makes this bound blind, and a blind bound
    that says nothing is worse than none. The router does report it (F2
    asserts so live), which is exactly why a silent change would go unnoticed."""
    budget = RunTokenBudget(100)

    silent = LLMResult(generations=[[ChatGeneration(message=AIMessage("ok"))]])

    budget.on_llm_end(silent, run_id=uuid4())

    assert budget.tokens == 0
    assert budget.unmeasured_calls == 1


def test_the_token_budget_does_not_let_langchain_swallow_its_own_failure() -> None:
    budget = RunTokenBudget(100)

    assert budget.raise_error is True
    assert budget.run_inline is True


def test_the_token_budget_refuses_an_allowance_that_permits_nothing() -> None:
    with pytest.raises(CheckFailed, match="allowance"):
        RunTokenBudget(0)


def test_run_turn_aborts_a_run_that_spends_past_its_token_limit() -> None:
    """End to end on the seam: a graph that keeps spending is stopped part-way,
    not after it finishes."""
    graph = SpendingGraph(calls=5, tokens_each=1000)

    with pytest.raises(TokenLimitExceeded):
        run_turn(graph, "ping", bounds=RunBounds(token_limit=1500))

    assert graph.completed < 5


def test_run_turn_lets_a_run_inside_its_token_limit_finish() -> None:
    """The discriminating half: the same graph, an allowance it fits inside."""
    graph = SpendingGraph(calls=5, tokens_each=1000)

    result = run_turn(graph, "ping", bounds=RunBounds(token_limit=TOKEN_LIMIT))

    assert graph.completed == 5
    assert result.tokens == 5000


def test_a_resumed_turn_gets_only_the_tokens_the_pause_left_over() -> None:
    """The same carry-across as the wall clock, for the same reason: a turn
    approved ten times would otherwise be billed ten full allowances."""
    graph = PausingGraph()
    paused = TurnResult(
        messages=[HumanMessage("write it")],
        interrupts=run_turn(graph, "write it", thread_id="t").interrupts,
        thread_id="t",
        tokens=900,
    )

    resume_turn(graph, paused, [{"type": "approve"}], bounds=RunBounds(token_limit=1000))

    assert _budget_of(graph.config).allowance == 100


def test_a_resume_is_refused_once_the_turn_has_spent_every_token() -> None:
    graph = PausingGraph()
    paused = TurnResult(
        messages=[HumanMessage("write it")],
        interrupts=run_turn(graph, "write it", thread_id="t").interrupts,
        thread_id="t",
        tokens=1000,
    )

    with pytest.raises(TokenLimitExceeded, match="before it was resumed"):
        resume_turn(graph, paused, [{"type": "approve"}], bounds=RunBounds(token_limit=1000))


def test_run_bounds_default_the_token_limit_to_the_module_constant() -> None:
    assert RunBounds().token_limit == TOKEN_LIMIT


@pytest.mark.parametrize("limit", [0, -1])
def test_run_bounds_reject_a_token_limit_that_permits_nothing(limit: int) -> None:
    with pytest.raises(CheckFailed, match="token_limit"):
        RunBounds(token_limit=limit)


def test_the_token_limit_is_an_operating_error_not_a_broken_contract() -> None:
    """A model that kept talking is the outside world being expensive, not a
    caller of ours passing something impossible."""
    assert issubclass(TokenLimitExceeded, RuntimeError)
    assert not issubclass(TokenLimitExceeded, CheckFailed)


@pytest.mark.parametrize(
    "bound", [DeadlineExceeded, ResumeLimitExceeded, StepLimitExceeded, TokenLimitExceeded]
)
def test_every_run_bound_fails_as_a_bound_exceeded(bound: type[Exception]) -> None:
    """One base, so the edge catches every bound `RunBounds` rations — including
    the next one — without being edited."""
    assert issubclass(bound, BoundExceeded)


def test_a_spent_budget_is_an_operating_error_not_a_broken_contract() -> None:
    """Still a `RuntimeError`, so anything catching that today is unaffected, and
    never a `CheckFailed`: a programmer error is not a spent budget."""
    assert issubclass(BoundExceeded, RuntimeError)
    assert not issubclass(BoundExceeded, CheckFailed)


# --------------------------------------------------------------------------
# Adversarial model behaviour: paths the harness already claims to handle.
# --------------------------------------------------------------------------


def test_a_turn_whose_model_returned_nothing_at_all_still_produces_a_result() -> None:
    """An `AIMessage` with neither content nor tool calls is a real provider
    outcome. `run_turn` must return a `TurnResult` rather than trip a
    postcondition — the turn happened, it just said nothing.

    `result.failed_tool_calls == ()` and `result[-1].text == ""` only restate
    what `FakeGraph` was constructed with. The actual claim under test is that
    `run_turn`'s own postconditions — in particular the relative message check
    (`len(messages) > len(sent)`) — do not raise on this shape; `len(result)
    == 2` is what exercises that, since the one-message prompt plus the empty
    `AIMessage` is the smallest input that satisfies it.
    """
    agent = FakeGraph({"messages": [HumanMessage("say something"), AIMessage("")]})

    result = run_turn(agent, "say something")

    assert len(result) == 2
    assert result.failed_tool_calls == ()
    assert result[-1].text == ""


def test_duplicate_tool_call_ids_let_one_result_answer_two_calls() -> None:
    """**A limitation, asserted so it is known rather than discovered.**

    `_unanswered_tool_calls` (`src/my_agent/run.py:592`) collects requested ids
    into a list and answered ids into a *set*, then filters by membership. Two
    calls sharing an id are therefore both satisfied by a single `ToolMessage`,
    so a genuinely unanswered second call passes the check.

    A provider that reuses ids within one `AIMessage` is not something this
    harness has seen, and counting by multiplicity would be a small change. The
    reason to record it rather than fix it: nothing today produces the shape,
    and an unused branch is a branch nobody tests. If a provider ever does,
    this test names the behaviour to change.
    """
    calls = [
        {"name": "ls", "args": {}, "id": "dup"},
        {"name": "read_file", "args": {"file_path": "/a"}, "id": "dup"},
    ]
    history = [
        HumanMessage("go"),
        AIMessage("", tool_calls=calls),
        ToolMessage("ok", tool_call_id="dup", name="ls"),
    ]
    agent = FakeGraph({"messages": [*history, HumanMessage("next"), AIMessage("done")]})

    result = run_turn(agent, "next", history=history)

    assert result[-1].text == "done"


# --------------------------------------------------------------------------
# What the agent actually wrote (F40)
# --------------------------------------------------------------------------


def test_a_turn_surfaces_the_files_the_agent_wrote() -> None:
    """The read-back the harness already had and threw away.

    deepagents' `StateBackend` puts the agent's filesystem in graph state, so
    every `invoke` returns `files` alongside `messages`. `_invoke` took the
    messages and dropped the rest, `TurnResult` had nowhere to put it and the
    mirror records the state's *key names* only — so the one deterministic
    artifact a turn produces, the object the agent claims to have created, was
    visible nowhere.
    """
    model = FanningOutModel(width=2, turns_before_answering=1)

    result = run_turn(build_agent(model), "write two files")

    assert sorted(result.files) == ["/f1-0.txt", "/f1-1.txt"]
    assert result.files["/f1-0.txt"]["content"] == "x"


def test_a_turn_that_wrote_nothing_reports_no_files() -> None:
    """The discriminator. Without it the assertion above is satisfied by any
    non-empty mapping, and a `files` that simply echoed state would read as
    proof the agent did something."""
    result = run_turn(FakeGraph(), "say hello")

    assert result.files == {}


def test_the_run_bounds_are_the_values_that_were_chosen() -> None:
    """Literals, for the reason `test_model.py` now pins the router defaults.

    Measured 2026-09-21: `RUN_DEADLINE_S` 600.0 -> 6000.0 and `TOKEN_LIMIT`
    500_000 -> 5_000_000 left all 376 tests green, because every test that reads
    a bound compares it to the same module constant it came from — including the
    two that pass the constant in as the budget, so the goalposts move together.
    `RECURSION_LIMIT` was the one already pinned, incidentally, by the
    `match="25 steps"` literal further up this file.
    """
    assert RECURSION_LIMIT == 25
    assert RUN_DEADLINE_S == 600.0
    assert TOKEN_LIMIT == 500_000
    assert RESUME_LIMIT == 3


def test_the_default_bounds_object_carries_those_same_values() -> None:
    """The pin above is on the constants; this is on the object every turn
    actually runs under. A `RunBounds` field that stopped defaulting to its
    constant would leave the first test green and every run unbounded by the
    number it names."""
    assert DEFAULT_RUN_BOUNDS.step_limit == 25
    assert DEFAULT_RUN_BOUNDS.deadline_s == 600.0
    assert DEFAULT_RUN_BOUNDS.token_limit == 500_000
    assert DEFAULT_RUN_BOUNDS.resume_limit == 3


def test_a_turn_carries_every_state_key_the_graph_returned() -> None:
    """`_invoke` read `messages` and dropped the rest of the state.

    `files` was the key that mattered enough to notice (F40), but reading one
    named key is the same mistake at a smaller scale: the compiled output schema
    declares `['files', 'messages', 'structured_response']`, and a middleware may
    add more. Anything the graph reports and this class cannot hold is
    information the harness destroys at the only point it is available.
    """
    graph = FakeGraph(
        extra_state={
            "files": {"/a.txt": {"content": "x"}},
            "structured_response": {"verdict": "pass"},
            "some_future_key": 42,
        }
    )

    result = run_turn(graph, "go")

    assert result.state["some_future_key"] == 42
    assert result.structured_response == {"verdict": "pass"}
    assert dict(result.files) == {"/a.txt": {"content": "x"}}


def test_the_state_a_turn_carries_excludes_what_has_its_own_field() -> None:
    """The discriminator, and the reason this is not just `result.raw`.

    `messages` and `__interrupt__` are already `TurnResult` fields. Repeating
    them in `state` would double the memory a turn holds — the messages are the
    bulk of it — and leave two copies that a later edit can disagree about.
    """
    graph = FakeGraph(extra_state={"files": {}})

    result = run_turn(graph, "go")

    assert "messages" not in result.state
    assert "__interrupt__" not in result.state
    assert result.messages


def test_a_turn_against_a_graph_with_no_extra_state_carries_none() -> None:
    """Absence is a legitimate state: a fake graph carries no filesystem, and a
    `StateBackend` never written to reports none either."""
    result = run_turn(FakeGraph(), "go")

    assert result.state == {}
    assert result.structured_response is None
    assert result.files == {}


# --------------------------------------------------------------------------
# The last seven checks in this module with no test that trips them
#
# Measured by instrumenting `require()` to log its call site whenever it raises
# and running the suite. These were what remained: the two shape checks on what
# a graph hands back, the three type preconditions nothing had passed a wrong
# type to, and the backwards-clock check on the accounting path.
# --------------------------------------------------------------------------


def test_the_deadline_refuses_a_backwards_clock_while_accounting() -> None:
    """The *other* backwards-clock check. `_require_time_left`'s has a test;
    this one feeds `TurnResult.elapsed_s`, so a negative value hands the next
    resume a budget larger than the turn had left — the opposite of a bound.

    Its message used to be byte-identical to the enforcement path's, so no
    `match=` could tell the two apart and the pair read as covered.
    """
    deadline = RunDeadline(10.0, clock=FakeClock(0.0, -5.0))

    with pytest.raises(CheckFailed, match="accounting for the turn"):
        _ = deadline.elapsed_s


def test_run_turn_rejects_a_result_that_is_not_a_mapping() -> None:
    """`_invoke` subscripts what comes back. A graph returning a list would
    otherwise fail with a `TypeError` from inside this module rather than
    naming the library that changed."""
    graph = FakeGraph(result=cast(Any, [HumanMessage("hi"), AIMessage("there")]))

    with pytest.raises(CheckFailed, match="not a mapping"):
        run_turn(graph, "go")


def test_run_turn_rejects_interrupts_it_cannot_read() -> None:
    """The shape check on a pause. `interrupts` is built by filtering for
    `Interrupt` objects, so a langgraph that reported pauses some other way
    would yield an empty tuple — a paused turn indistinguishable from a
    finished one, which is the exact failure F29 exists for."""
    # Typed loosely on purpose: the whole point is a payload shaped the way
    # langgraph does *not* currently shape one.
    result: dict[str, Any] = {
        **_two_messages(),
        "__interrupt__": [{"action_requests": [{"name": "write_file"}]}],
    }
    graph = FakeGraph(result=result)

    with pytest.raises(CheckFailed, match="are Interrupt"):
        run_turn(graph, "go")


def test_run_turn_rejects_bounds_that_are_not_run_bounds() -> None:
    """A duck-typed stand-in with the right attribute names would reach
    `_run_config` and be sent as a `recursion_limit`, so the turn would run
    under numbers that never passed `RunBounds.__post_init__`."""
    loose = cast(Any, SimpleNamespace(step_limit=1, deadline_s=1.0, token_limit=1, resume_limit=0))

    with pytest.raises(CheckFailed, match="bounds must be RunBounds"):
        run_turn(FakeGraph(), "go", bounds=loose)


def test_resume_turn_rejects_something_that_cannot_be_invoked() -> None:
    """The resume half of `run_turn`'s own precondition. A resume path that
    skipped it would fail with an `AttributeError` after the bounds had already
    been computed and the resume counted."""
    paused = run_turn(PausingGraph(), "write it", thread_id="t1")

    with pytest.raises(CheckFailed, match="invoke"):
        resume_turn(cast(Any, object()), paused, [{"type": "approve"}])


def test_resume_turn_rejects_something_that_is_not_a_turn_result() -> None:
    """`paused` carries the thread the pending interrupt lives in and the
    budgets already spent. A look-alike would resume the wrong checkpoint with a
    fresh budget — the two defects F33 closed, reached from a new direction."""
    impostor = cast(Any, SimpleNamespace(paused=True, thread_id="t1", elapsed_s=0.0, tokens=0))

    with pytest.raises(CheckFailed, match="must be the TurnResult"):
        resume_turn(FakeGraph(), impostor, [{"type": "approve"}])


def test_resume_turn_rejects_bounds_that_are_not_run_bounds() -> None:
    """The same precondition as `run_turn`'s, on the path that subtracts the
    spent budget from them. Unvalidated bounds here mean a resume computing its
    remainder from numbers nothing checked."""
    paused = run_turn(PausingGraph(), "write it", thread_id="t1")
    loose = cast(Any, SimpleNamespace(step_limit=1, deadline_s=1.0, token_limit=1, resume_limit=3))

    with pytest.raises(CheckFailed, match="bounds must be RunBounds"):
        resume_turn(FakeGraph(), paused, [{"type": "approve"}], bounds=loose)
