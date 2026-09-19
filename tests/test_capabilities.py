"""Contract tests for `my_agent.capabilities`.

The allowlist is defined by subtraction and the permission rules travel through
deepagents' private `_permissions`. Both are assumptions about a library we do
not control, so both are stated here as well as at import time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_args

import pytest
from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    FsToolName,
    create_deep_agent,
)
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.summarization import (
    compute_summarization_defaults,
    create_summarization_middleware,
)
from deepagents.profiles import _builtin_profiles
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field, SecretStr

from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import (
    COMPACTION_ARG_TRUNCATION_MESSAGES,
    COMPACTION_KEEP_MESSAGES,
    COMPACTION_TRIGGER_TOKENS,
    CONTEXT_WINDOW_TOKENS,
    DEEPAGENTS_PLUGIN_GROUPS,
    DEFAULT_FILESYSTEM_TOOLS,
    GREP_MATCH_LIMIT,
    HUMAN_MESSAGE_TOKEN_LIMIT,
    LIBRARY_COMPACTION_TRIGGER_TOKENS,
    SHELL_TOOL_NAME,
    SUBAGENT_TASK_TOOL_NAME,
    TASK_DISPATCH_LIMIT,
    TOOL_CALL_LIMIT,
    TOOL_RESULT_TOKEN_LIMIT,
    bounded_compaction,
    call_limits,
    compiled_tool_names,
    least_privilege_filesystem,
    require_granted,
    require_withheld,
    subagent_graphs,
)
from my_agent.model import DEFAULT_MODEL, ModelConfig, build_model
from my_agent.negative_space import CheckFailed


def test_allowlist_is_every_filesystem_tool_except_the_shell() -> None:
    """If deepagents adds a filesystem tool, the import-time check forces a
    deliberate decision instead of granting it by default."""
    assert set(DEFAULT_FILESYSTEM_TOOLS) == set(get_args(FsToolName)) - {SHELL_TOOL_NAME}
    assert SHELL_TOOL_NAME not in DEFAULT_FILESYSTEM_TOOLS


def test_least_privilege_middleware_withholds_the_shell_tool() -> None:
    """The allowlist is only a request until the built middleware is read back."""
    middleware = least_privilege_filesystem(None)

    granted = {getattr(t, "name", None) for t in middleware.tools}

    assert granted != set()
    assert SHELL_TOOL_NAME not in granted


def test_permissions_reach_the_filesystem_middleware(deny_secrets: FilesystemPermission) -> None:
    """`permissions` only takes effect through FilesystemMiddleware's private
    `_permissions`. build_agent installs that middleware, so it must forward
    them or every rule is silently lost."""
    middleware = least_privilege_filesystem([deny_secrets])

    assert middleware._permissions == [deny_secrets]


def test_no_permissions_still_produces_a_usable_middleware() -> None:
    """`None` normalises to an empty rule list, not a missing attribute."""
    assert least_privilege_filesystem(None)._permissions == []


# --------------------------------------------------------------------------
# Vacuity-guarded capability assertions
# --------------------------------------------------------------------------


def test_require_withheld_accepts_a_real_absence(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    """The ordinary case: a non-empty grant that does not include the capability."""
    assert_does_not_raise(
        lambda: require_withheld(SHELL_TOOL_NAME, frozenset({"ls", "read_file"}), "a grant")
    )


def test_require_withheld_rejects_a_granted_capability() -> None:
    with pytest.raises(CheckFailed, match=SHELL_TOOL_NAME):
        require_withheld(SHELL_TOOL_NAME, frozenset({"ls", SHELL_TOOL_NAME}), "a test grant")


def test_require_withheld_rejects_an_empty_grant_as_vacuous() -> None:
    """Nothing granted means the absence proves nothing. This is the check that
    stops every capability assertion in the repo from passing by accident."""
    with pytest.raises(CheckFailed, match="vacuous"):
        require_withheld(SHELL_TOOL_NAME, frozenset(), "a test grant")


def test_require_withheld_names_where_it_looked() -> None:
    """A bare 'execute leaked' says nothing about which graph leaked it."""
    with pytest.raises(CheckFailed, match="the general-purpose subagent"):
        require_withheld(
            SHELL_TOOL_NAME, frozenset({SHELL_TOOL_NAME}), "the general-purpose subagent"
        )


def test_require_granted_accepts_a_present_tool(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    assert_does_not_raise(
        lambda: require_granted("write_file", frozenset({"write_file", "ls"}), "a grant")
    )


def test_require_granted_rejects_a_missing_tool() -> None:
    with pytest.raises(CheckFailed, match="write_file"):
        require_granted("write_file", frozenset({"ls"}), "a test grant")


# --------------------------------------------------------------------------
# Reading the subagent graphs back out of the `task` tool
#
# `agent.get_subgraphs()` returns nothing for a deep agent: the subagent graphs
# are held in the `task` tool's closure and reachable nowhere else. These tests
# pin that reach against the real library, and pin every way it can move.
# --------------------------------------------------------------------------


def _graph_with_tools(tools_by_name: dict[str, Any]) -> CompiledStateGraph[Any, Any, Any, Any]:
    """A stand-in shaped like the only part of a compiled graph we read.

    Hand-built rather than compiled, because the cases below are structures
    deepagents would never produce and the point is that we notice them.
    """
    node = SimpleNamespace(bound=SimpleNamespace(tools_by_name=tools_by_name))
    return cast(
        "CompiledStateGraph[Any, Any, Any, Any]", SimpleNamespace(nodes={"tools": node})
    )


def test_subagent_graphs_finds_the_general_purpose_subagent() -> None:
    """Pins the closure reach against the real library. A deepagents change to
    how `task` holds its subagents fails here rather than silently returning
    nothing and making every subagent assertion vacuous."""
    agent = create_deep_agent(model=ParrotFakeChatModel())

    assert set(subagent_graphs(agent)) == {"general-purpose"}


def test_subagent_graphs_returns_real_graphs_we_can_read_tools_from() -> None:
    """A name-to-graph mapping is only useful if the values are inspectable."""
    agent = create_deep_agent(model=ParrotFakeChatModel())

    graphs = subagent_graphs(agent)

    assert compiled_tool_names(graphs["general-purpose"]) != frozenset()


def test_subagent_graphs_is_empty_when_no_task_tool_is_bound() -> None:
    """No `task` tool means no subagent can be dispatched, so there is nothing
    to prove — an empty mapping, not an error."""
    assert subagent_graphs(_graph_with_tools({"ls": object()})) == {}


def test_subagent_graphs_rejects_a_task_tool_that_carries_no_closure() -> None:
    """The failure mode this guards: returning {} because the structure moved
    would report 'no subagents' and pass every absence check vacuously."""

    def no_closure() -> None:
        return None

    task = SimpleNamespace(func=no_closure, coroutine=None)

    with pytest.raises(CheckFailed, match="subagent"):
        subagent_graphs(_graph_with_tools({"task": task}))


def test_subagent_graphs_rejects_a_task_tool_with_neither_func_nor_coroutine() -> None:
    task = SimpleNamespace(func=None, coroutine=None)

    with pytest.raises(CheckFailed, match="task"):
        subagent_graphs(_graph_with_tools({"task": task}))


def test_subagent_graphs_rejects_a_tools_node_with_no_tool_mapping() -> None:
    """The other half of the structural pin. Returning nothing here would make
    both `compiled_tool_names` and every absence check silently vacuous."""
    moved = cast(
        "CompiledStateGraph[Any, Any, Any, Any]",
        SimpleNamespace(nodes={"tools": SimpleNamespace(bound=object())}),
    )

    with pytest.raises(CheckFailed, match="tools_by_name"):
        subagent_graphs(moved)


def test_subagent_graphs_rejects_a_graph_with_no_tools_node() -> None:
    empty = cast("CompiledStateGraph[Any, Any, Any, Any]", SimpleNamespace(nodes={}))

    with pytest.raises(CheckFailed, match="tools"):
        subagent_graphs(empty)


def _task_tool_holding(graphs: object) -> SimpleNamespace:
    """A `task` stand-in whose closure really carries `subagent_graphs`.

    The freevar name is the contract, so the local below must keep that exact
    spelling — it is what the reader looks for.
    """
    subagent_graphs = graphs  # read via the closure, not by name

    def dispatch() -> object:
        return subagent_graphs

    return SimpleNamespace(func=dispatch, coroutine=None)


def test_subagent_graphs_rejects_a_task_tool_that_holds_no_subagents() -> None:
    """A `task` tool with an empty mapping would report 'no subagents' and pass
    every absence check without checking anything."""
    with pytest.raises(CheckFailed, match="vacuous"):
        subagent_graphs(_graph_with_tools({"task": _task_tool_holding({})}))


def test_subagent_graphs_rejects_a_closure_value_that_is_not_a_mapping() -> None:
    with pytest.raises(CheckFailed, match="subagent"):
        subagent_graphs(_graph_with_tools({"task": _task_tool_holding(["general-purpose"])}))


# --------------------------------------------------------------------------
# The backend, and the context bounds that came with it
#
# Least privilege says a capability is never inherited from a library default.
# `least_privilege_filesystem` was inheriting five: the backend and four bounds
# on how much context the filesystem tools may consume (F21).
# --------------------------------------------------------------------------


def test_least_privilege_middleware_runs_on_the_state_backend_by_default() -> None:
    """The whole prompt-injection blast radius is this one line. A `StateBackend`
    filesystem lives in graph state: there is no `.env` to read, no repo to walk
    and no path out. Swapping it is meant to turn this test red."""
    assert isinstance(least_privilege_filesystem(None).backend, StateBackend)


def test_the_state_backend_cannot_run_shell_commands_even_if_asked() -> None:
    """Defence in depth behind the allowlist: deepagents only wires `execute` to
    a backend implementing `SandboxBackendProtocol`. Recorded as a test because
    it is the reason a leaked `execute` would be survivable rather than fatal —
    not a reason to grant it."""
    assert not isinstance(StateBackend(), SandboxBackendProtocol)


def test_least_privilege_middleware_uses_the_backend_it_was_given(tmp_path: Path) -> None:
    """The parameter exists so that a `backend` reaching `create_deep_agent` can
    also reach the middleware that replaces its filesystem tools. Without it the
    two would look at different filesystems — F5's failure, one field away."""
    backend = FilesystemBackend(root_dir=tmp_path)

    middleware = least_privilege_filesystem(None, backend=backend)

    assert middleware.backend is backend


def test_context_bounds_are_pinned_to_what_deepagents_defaults_to_today() -> None:
    """The pin, stated where a reader will look. These are the values the
    load-time check compares against the installed wheel, so a deepagents
    change to any of them fails the import rather than quietly resizing how
    much of a tool result reaches the model."""
    defaults = inspect.signature(FilesystemMiddleware.__init__).parameters

    assert defaults["tool_token_limit_before_evict"].default == TOOL_RESULT_TOKEN_LIMIT
    assert defaults["human_message_token_limit_before_evict"].default == HUMAN_MESSAGE_TOKEN_LIMIT
    assert defaults["grep_max_count"].default == GREP_MATCH_LIMIT


def test_the_backend_and_context_bounds_are_stated_rather_than_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discriminating on purpose.

    Every pinned value equals deepagents' current default, so reading one back
    off the built middleware cannot tell "we chose it" from "we inherited it" —
    a mutant that deletes the arguments passes such a test untouched, which is
    what makes it decorative. The *call* is the only place the difference shows,
    so this records it, with the real constructor still doing the work.
    """
    seen: dict[str, Any] = {}

    def recording(**kwargs: Any) -> FilesystemMiddleware:
        seen.update(kwargs)
        return FilesystemMiddleware(**kwargs)

    # Patched where it is used, not where it is defined, and the real
    # constructor still builds the object so nothing passes on a stub.
    monkeypatch.setattr("my_agent.capabilities.FilesystemMiddleware", recording)

    least_privilege_filesystem(None)

    assert isinstance(seen["backend"], StateBackend)
    assert seen["tool_token_limit_before_evict"] == TOOL_RESULT_TOKEN_LIMIT
    assert seen["human_message_token_limit_before_evict"] == HUMAN_MESSAGE_TOKEN_LIMIT
    assert seen["grep_max_count"] == GREP_MATCH_LIMIT


def test_least_privilege_middleware_states_every_context_bound_it_runs_under() -> None:
    """The other half of the claim, and a different failure: `_permissions` was
    passed and silently dropped once already (F5), so a bound that was sent is
    not yet a bound that landed."""
    middleware = least_privilege_filesystem(None)

    assert middleware._tool_token_limit_before_evict == TOOL_RESULT_TOKEN_LIMIT
    assert middleware._human_message_token_limit_before_evict == HUMAN_MESSAGE_TOKEN_LIMIT
    assert middleware._grep_max_count == GREP_MATCH_LIMIT


# --------------------------------------------------------------------------
# The plugin door (F28)
# --------------------------------------------------------------------------


def test_no_package_is_registering_deepagents_profile_plugins() -> None:
    """The escape the parameter pin cannot cover.

    `KNOWN_CREATE_DEEP_AGENT_PARAMS` fails the import when deepagents grows a
    parameter. A harness-profile plugin grows no parameter: it registers into a
    process-global registry from an entry point, and can add middleware, drop
    tools or rewrite the system prompt from there.
    """
    installed = {
        group: sorted(ep.name for ep in entry_points(group=group))
        for group in DEEPAGENTS_PLUGIN_GROUPS
    }

    assert installed == {group: [] for group in DEEPAGENTS_PLUGIN_GROUPS}


def test_the_plugin_groups_are_the_names_deepagents_actually_reads() -> None:
    """A pin on the wrong group name is a pin on nothing — it would pass
    forever while the real groups filled up."""
    source = inspect.getsource(_builtin_profiles)

    for group in DEEPAGENTS_PLUGIN_GROUPS:
        assert f'"{group}"' in source


# --------------------------------------------------------------------------
# Call limits (F30)
# --------------------------------------------------------------------------


def test_call_limits_bound_both_all_tools_and_task_specifically() -> None:
    """Two bounds, because they answer different questions: how much work a run
    may do at all, and how much of it may be delegated."""
    limits = call_limits()

    assert [m.name for m in limits] == [
        "ToolCallLimitMiddleware",
        f"ToolCallLimitMiddleware[{SUBAGENT_TASK_TOOL_NAME}]",
    ]


def test_the_two_call_limits_have_distinct_names() -> None:
    """deepagents merges middleware by `.name`. Two entries sharing one would
    mean the second silently replacing the first, and a bound nobody applied."""
    names = [m.name for m in call_limits()]

    assert len(set(names)) == len(names)


def test_the_task_limit_is_tighter_than_the_overall_tool_limit() -> None:
    """A `task` cap at or above the overall cap could never bind first, which is
    the one thing it exists to do — a dispatch costs a whole subagent run."""
    assert TASK_DISPATCH_LIMIT < TOOL_CALL_LIMIT


def test_call_limits_are_per_run_not_per_thread() -> None:
    """A thread limit needs a checkpointer to mean anything, and the graph
    carries none by default — it would be a bound that never counts."""
    for middleware in call_limits():
        assert middleware.thread_limit is None
        assert middleware.run_limit is not None


def test_call_limits_block_rather_than_abort() -> None:
    """`continue` blocks the exceeded call and lets the agent answer with what
    it has. `error` would turn a model that asked for too much into a crashed
    turn, and the run is already bounded by the step limit and the deadline."""
    for middleware in call_limits():
        assert middleware.exit_behavior == "continue"


# --------------------------------------------------------------------------
# Compaction: the bound on how much conversation the model ever sees
# --------------------------------------------------------------------------


def test_compaction_trigger_leaves_room_below_the_window_we_assume() -> None:
    """The whole point of owning the number. A trigger at or above the window
    can only fire after the provider has already rejected the request."""
    assert COMPACTION_TRIGGER_TOKENS < CONTEXT_WINDOW_TOKENS


def test_deepagents_would_compact_above_the_window_it_is_serving() -> None:
    """The discriminator, and the reason this bound is stated at all.

    Without it, `test_compaction_trigger_leaves_room_below_the_window_we_assume`
    keeps passing on the day deepagents picks a sane number for its own reasons,
    and a bound we merely agree with reads as a bound we set.
    """
    assert LIBRARY_COMPACTION_TRIGGER_TOKENS > CONTEXT_WINDOW_TOKENS
    assert compute_summarization_defaults(ParrotFakeChatModel())["trigger"] == (
        "tokens",
        LIBRARY_COMPACTION_TRIGGER_TOKENS,
    )


def test_a_router_model_has_no_profile_for_deepagents_to_size_itself_from() -> None:
    """Why the library lands on its fallback: `compute_summarization_defaults`
    picks fraction-of-window thresholds only when the model exposes
    `max_input_tokens`, and a `org/model` router id resolves to no profile at
    all. Verified against langchain-openai 1.6.2, 2026-09-18."""
    model = build_model(ModelConfig(api_key=SecretStr("hf_token_value"), model=DEFAULT_MODEL))

    assert model.profile is None
    assert compute_summarization_defaults(model)["trigger"] == (
        "tokens",
        LIBRARY_COMPACTION_TRIGGER_TOKENS,
    )


def test_bounded_compaction_states_the_thresholds_it_runs_under() -> None:
    """Passing a bound and having it land are different claims. Read back off
    the langchain middleware deepagents wraps, which is where they end up."""
    middleware = bounded_compaction(ParrotFakeChatModel(), StateBackend())

    assert middleware._lc_helper.trigger == ("tokens", COMPACTION_TRIGGER_TOKENS)
    assert middleware._lc_helper.keep == ("messages", COMPACTION_KEEP_MESSAGES)


def test_bounded_compaction_keeps_tool_argument_truncation_switched_on() -> None:
    """`truncate_args_settings` defaults to `None`, which disables clipping
    oversized `write_file` arguments entirely — a capability deepagents' own
    factory switches on and a hand-built replacement silently drops."""
    middleware = bounded_compaction(ParrotFakeChatModel(), StateBackend())

    assert middleware._truncate_args_trigger == ("messages", COMPACTION_ARG_TRUNCATION_MESSAGES)
    assert middleware._truncate_args_keep == ("messages", COMPACTION_ARG_TRUNCATION_MESSAGES)


def test_bounded_compaction_runs_on_the_backend_it_was_given() -> None:
    """Compaction offloads the messages it replaces. On a different backend
    from the filesystem tools, the agent would read files it cannot see."""
    backend = StateBackend()

    middleware = bounded_compaction(ParrotFakeChatModel(), backend)

    assert middleware._backend is backend


def test_bounded_compaction_replaces_deepagents_own_rather_than_joining_it() -> None:
    """deepagents merges middleware by `.name`. A different name would leave
    both installed and the library's 170k trigger still in the stack."""
    ours = bounded_compaction(ParrotFakeChatModel(), StateBackend())
    theirs = create_summarization_middleware(ParrotFakeChatModel(), StateBackend())

    assert ours.name == theirs.name


# --------------------------------------------------------------------------
# Compaction: that it fires, not merely that it was configured to
#
# Every test above reads a threshold back off a constructor. That proves the
# number was passed and nothing about whether the mechanism runs -- which is
# exactly how F31 survived the project's whole life. These drive a real
# compiled agent with a real oversized history and assert on what the model
# was actually handed.
# --------------------------------------------------------------------------


class CallsOneTool(BaseChatModel):
    """Issues one scripted tool call, then stops.

    `StateBackend` refuses to read or write outside a graph run, so a
    filesystem bound can only be exercised by a real dispatch. This is the
    smallest model that produces one.
    """

    tool: str = "read_file"
    args: dict[str, Any] = Field(default_factory=dict)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "calls-one-tool"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            message = AIMessage("", tool_calls=[{"name": self.tool, "args": self.args, "id": "c1"}])
        else:
            message = AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_messages(files: dict[str, Any], tool: str, args: dict[str, Any]) -> list[ToolMessage]:
    """Run one tool call against a pre-populated `StateBackend`.

    Files are passed on `invoke` because `StateBackend` cannot be written from
    outside a graph execution -- its own error message says so.
    """
    agent = build_agent(CallsOneTool(tool=tool, args=args), AgentConfig())
    input_state = cast(Any, {"messages": [("user", "go")], "files": files})
    out = agent.invoke(input_state, {"recursion_limit": 25})
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def test_grep_stops_at_the_match_limit() -> None:
    """`GREP_MATCH_LIMIT` is a bound on a capability we granted. Read back off
    the middleware it is only a number that was passed; here it is the number
    of matches that actually came back.

    Fixture size and expected count are literals (1,200 and 1,000), not
    derived from `GREP_MATCH_LIMIT`. A fixture sized as `GREP_MATCH_LIMIT +
    200` and an expectation of `GREP_MATCH_LIMIT` both move in lockstep with
    the constant under test, so a mutant that changes it moves the test's own
    goalposts and the test cannot fail. The guard below still ties the
    literals to the constant, so a deliberate change to the bound is caught
    here rather than discovered later.
    """
    assert GREP_MATCH_LIMIT < 1200  # the fixture must actually exceed today's limit
    files = {f"/f{i}.txt": {"content": "needle\n"} for i in range(1200)}

    messages = _tool_messages(files, "grep", {"pattern": "needle"})

    result = str(messages[0].content)
    assert result.count("/f") == 1000
    assert "maximum match count" in result


def test_grep_under_the_limit_returns_everything_and_says_nothing_about_truncation() -> None:
    """The discriminator. Without it the test above passes just as well if grep
    silently caps every search, which is a different and worse bug."""
    files = {f"/f{i}.txt": {"content": "needle\n"} for i in range(5)}

    messages = _tool_messages(files, "grep", {"pattern": "needle"})

    result = str(messages[0].content)
    assert result.count("/f") == 5
    assert "maximum match count" not in result


def test_an_oversized_read_is_truncated_with_a_marker() -> None:
    """`TOOL_RESULT_TOKEN_LIMIT` in the only units it is enforced in: characters,
    at `NUM_CHARS_PER_TOKEN` per token. Ten very long lines clear the line limit
    and reach the character bound."""
    fat = "".join("y" * 20_000 + "\n" for _ in range(10))
    assert len(fat) > 4 * TOOL_RESULT_TOKEN_LIMIT  # the fixture must actually be over it

    messages = _tool_messages(
        {"/fat.txt": {"content": fat}}, "read_file", {"file_path": "/fat.txt"}
    )

    result = str(messages[0].content)
    assert len(result) < len(fat)
    assert "truncated due to size" in result


def test_a_small_read_comes_back_whole() -> None:
    """The discriminator: `read_file` does not mark everything truncated."""
    small = "".join(f"line {i}\n" for i in range(50))

    messages = _tool_messages({"/s.txt": {"content": small}}, "read_file", {"file_path": "/s.txt"})

    result = str(messages[0].content)
    assert "truncated due to size" not in result


def test_the_line_limit_cuts_a_long_file_before_the_character_bound_can() -> None:
    """**The reachable surface of `TOOL_RESULT_TOKEN_LIMIT` is narrower than it
    looks.** `read_file` keeps 100 lines by default, so an ordinary long file is
    already small by the time the character bound is consulted and the
    truncation marker never appears. Measured: 4,000 lines and 134,890
    characters came back as ~3,000 characters, unmarked.

    Stated as a test so that a change to either limit has to confront the
    interaction rather than discover it.
    """
    many = "".join(f"line {i} padding padding padding\n" for i in range(4000))
    assert len(many) > 4 * TOOL_RESULT_TOKEN_LIMIT

    messages = _tool_messages(
        {"/many.txt": {"content": many}}, "read_file", {"file_path": "/many.txt"}
    )

    result = str(messages[0].content)
    assert result.count("\n") <= 100
    assert "truncated due to size" not in result


class RecordsWhatItWasAsked(BaseChatModel):
    """A model that answers nothing and remembers everything it was sent.

    Compaction rewrites the request on its way to the model, so the only place
    its effect is observable is the argument list of the call it precedes.
    """

    seen: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "records-what-it-was-asked"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def _history(messages: int, chars_each: int) -> list[Any]:
    """A human/ai conversation of a stated size. `chars_each // 4` is what
    `count_tokens_approximately` will score each human turn at."""
    out: list[Any] = []
    for i in range(messages // 2):
        out.append(HumanMessage("filler text. " * (chars_each // 13)))
        out.append(AIMessage(f"noted {i}"))
    out.append(HumanMessage("now answer"))
    return out


def test_compaction_actually_shrinks_what_the_model_sees() -> None:
    """The behavioural claim every other compaction test in this file assumes.

    A 21-message history worth ~104,000 approximate tokens is over
    `COMPACTION_TRIGGER_TOKENS`, so the model must be handed fewer messages
    than were sent, with a summary standing in for the ones that were dropped.
    """
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())
    history = _history(20, 41_600)

    agent.invoke({"messages": history}, {"recursion_limit": 25})

    sent_to_model = model.seen[0]
    assert len(sent_to_model) < len(history)
    assert any("has been summarized" in (m.text or "") for m in sent_to_model)


def test_a_conversation_under_the_trigger_reaches_the_model_intact() -> None:
    """The discriminator. Without it the test above keeps passing on the day
    something truncates every conversation for an unrelated reason.

    Sized at ~40,000 approximate tokens: comfortably under the trigger, and far
    enough above zero that a trigger set much *lower* than ours compacts it and
    turns this red. A history of a few hundred tokens would pass under almost
    any threshold and so would discriminate nothing -- measured, as a surviving
    mutant, before it was resized.
    """
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())
    history = _history(20, 16_000)

    agent.invoke({"messages": history}, {"recursion_limit": 25})

    sent_to_model = model.seen[0]
    assert len(sent_to_model) == len(history) + 1  # + the system prompt
    assert not any("has been summarized" in (m.text or "") for m in sent_to_model)


def test_compaction_cannot_fire_while_every_message_fits_inside_what_it_keeps() -> None:
    """**A bound that does not bind on the shape this agent actually produces.**

    `keep=("messages", 6)` is a floor, not a target: compaction only has
    something to compact once there are more messages than it keeps. Three
    messages worth ~104,000 approximate tokens -- comfortably over the trigger
    -- reach the model whole.

    That shape is not hypothetical. A filesystem agent's expensive turn is one
    `read_file` returning one enormous `ToolMessage`, and no token threshold
    reaches it. Measured, not reasoned: the model was handed all 416,000
    characters.
    """
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())
    history: list[Any] = [
        HumanMessage("filler text. " * 32_000),
        AIMessage("noted"),
        HumanMessage("go"),
    ]

    agent.invoke({"messages": history}, {"recursion_limit": 25})

    sent_to_model = model.seen[0]
    assert len(sent_to_model) == len(history) + 1
    assert not any("has been summarized" in (m.text or "") for m in sent_to_model)
    assert max(len(m.text) for m in sent_to_model) > 400_000


def test_an_oversized_trailing_human_message_is_evicted_to_the_backend() -> None:
    """`HUMAN_MESSAGE_TOKEN_LIMIT` enforced, not merely configured.

    Measured against the installed wheel (`filesystem.py`
    `_apply_eviction_and_truncate` / `_build_truncated_human_message`): the
    tagged `HumanMessage` kept in graph *state* carries the full original text
    -- only `additional_kwargs["lc_evicted_to"]` changes. Truncation is
    computed fresh from that full text and applied solely to the message list
    handed to the model on each request, which is why the assertion on length
    reads from `model.seen`, not from `out["messages"]`.

    The fixture size (201,000 characters) is a literal, not
    `4 * HUMAN_MESSAGE_TOKEN_LIMIT + 1_000`: a size derived from the constant
    under test grows with it, so a mutant that raises the constant keeps this
    message oversized under the new threshold too and the test cannot fail.
    The guard ties the literal to today's threshold (200,000 characters) so a
    deliberate change to the bound is caught here.
    """
    assert 4 * HUMAN_MESSAGE_TOKEN_LIMIT < 201_000  # the fixture must actually exceed today's limit
    huge = "z" * 201_000
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())

    out = agent.invoke({"messages": [HumanMessage(huge)]}, {"recursion_limit": 25})

    evicted = [
        m
        for m in out["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_evicted_to")
    ]
    assert evicted != []

    sent_to_model = model.seen[0]
    truncated = [
        m
        for m in sent_to_model
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_evicted_to")
    ]
    assert truncated != []
    assert len(truncated[0].text) < len(huge)


def test_a_huge_human_message_that_is_not_last_is_never_evicted() -> None:
    """**The bound examines `messages[-1]` and nothing else.**

    A message just as large, one position from the end, is untouched. Combined
    with compaction -- which cannot reach anything inside `keep` -- a large
    `HumanMessage` in the middle of a conversation escapes both context bounds.
    Neither mechanism is wrong; each does what it documents. This is the test
    that stops "the conversation is bounded" from being read as a claim either
    of them makes about that shape.

    Same literal fixture size as the eviction test above, for the same
    reason: a size derived from `HUMAN_MESSAGE_TOKEN_LIMIT` would still
    demonstrate nothing useful if it moved with the constant, since this
    test's claim is about position, not size, and a literal keeps "huge"
    meaning something concrete rather than "whatever the constant is now".
    """
    assert 4 * HUMAN_MESSAGE_TOKEN_LIMIT < 201_000  # oversized under today's limit, if it mattered
    huge = "z" * 201_000
    model = RecordsWhatItWasAsked()
    agent = build_agent(model, AgentConfig())

    out = agent.invoke(
        {"messages": [HumanMessage(huge), HumanMessage("now answer")]},
        {"recursion_limit": 25},
    )

    evicted = [
        m
        for m in out["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_evicted_to")
    ]
    assert evicted == []
    assert any(len(m.text) == len(huge) for m in out["messages"] if isinstance(m, HumanMessage))
