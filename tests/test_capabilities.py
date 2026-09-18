"""Contract tests for `my_agent.capabilities`.

The allowlist is defined by subtraction and the permission rules travel through
deepagents' private `_permissions`. Both are assumptions about a library we do
not control, so both are stated here as well as at import time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
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
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langgraph.graph.state import CompiledStateGraph

from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    GREP_MATCH_LIMIT,
    HUMAN_MESSAGE_TOKEN_LIMIT,
    SHELL_TOOL_NAME,
    TOOL_RESULT_TOKEN_LIMIT,
    compiled_tool_names,
    least_privilege_filesystem,
    require_granted,
    require_withheld,
    subagent_graphs,
)
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
