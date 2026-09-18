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
from typing import Any
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from my_agent.agent import build_agent
from my_agent.mirror import JsonlMirror
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed
from my_agent.run import (
    RECURSION_LIMIT,
    RUN_DEADLINE_S,
    DeadlineExceeded,
    RunBounds,
    RunDeadline,
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

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result if result is not None else _two_messages()
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
    with pytest.raises(CheckFailed, match="added no messages"):
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

    with pytest.raises(CheckFailed, match="added no messages"):
        run_turn(graph, "c", history=unchanged[:2])


def test_run_turn_rejects_a_history_that_is_not_messages() -> None:
    """A history is what a previous `run_turn` returned. Anything else would be
    coerced by langgraph or silently dropped, neither of which is visible."""
    with pytest.raises(CheckFailed, match="history"):
        run_turn(FakeGraph(), "ping", history=["a string"])  # type: ignore[list-item]


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
