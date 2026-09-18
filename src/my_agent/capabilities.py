"""What the agent is allowed to do, and proof that it got no more.

A capability the agent has not been given cannot be misused. `create_deep_agent`
enables a shell `execute` tool with no opt-in (`docs/findings.md` F4), so the
allowlist here is defined by *subtraction* from what deepagents offers, the
subtraction is checked at import, and the result is checked again against the
compiled graph. An allowlist is only a request until something reads back what
was actually granted — that is what `compiled_tool_names` is for.

Nothing here knows about models, the router, or the environment.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection
from typing import Any, get_args

from deepagents import FilesystemMiddleware, FilesystemPermission, FsToolName
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langgraph.graph.state import CompiledStateGraph

from my_agent.negative_space import CheckFailed, require

__all__ = [
    "DEFAULT_FILESYSTEM_TOOLS",
    "GREP_MATCH_LIMIT",
    "HUMAN_MESSAGE_TOKEN_LIMIT",
    "SHELL_TOOL_NAME",
    "SUBAGENT_TASK_TOOL_NAME",
    "TOOL_RESULT_TOKEN_LIMIT",
    "compiled_tool_names",
    "least_privilege_filesystem",
    "require_granted",
    "require_withheld",
    "subagent_graphs",
]

SHELL_TOOL_NAME = "execute"
"""deepagents classes shell execution as a filesystem tool."""

SUBAGENT_TASK_TOOL_NAME = "task"
"""The tool deepagents binds to dispatch a subagent.

Its presence is what makes `subagent_graphs` worth reading: a capability
withheld from the parent graph is only withheld if no subagent re-grants it.
"""

_SUBAGENT_GRAPHS_FREEVAR = "subagent_graphs"
"""Where the `task` tool keeps the graphs it can dispatch to (F20).

`CompiledStateGraph.get_subgraphs()` reports nothing for a deep agent, so the
closure is the only route. Pinned by `subagent_graphs`, which fails loudly
rather than reporting an empty mapping when the structure moves.
"""

DEFAULT_FILESYSTEM_TOOLS: tuple[FsToolName, ...] = (
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
)
"""Every filesystem tool deepagents offers **except** `execute`.

Principle of least privilege. `create_deep_agent` enables shell execution with no
opt-in; an agent that only needs to read and write files has no business running
arbitrary commands. Re-enable it deliberately by passing a `FilesystemMiddleware`
that includes `execute` in `AgentConfig.middleware` — see `build_agent`.
"""

TOOL_RESULT_TOKEN_LIMIT = 20000
"""Tokens a tool result may add before deepagents evicts it to the filesystem.

Equal to deepagents' own default, and stated anyway. Least privilege says a
setting is never inherited from a library default — and this one *rewrites the
conversation*, moving a tool result out of the messages and leaving a pointer,
which would appear in the mirror as a state write nobody asked for. Owning the
number means an upstream change to it fails the load-time check below instead
of silently resizing how much of a tool result the model ever sees.
"""

HUMAN_MESSAGE_TOKEN_LIMIT = 50000
"""The same, for a user turn large enough that deepagents would evict it."""

GREP_MATCH_LIMIT = 1000
"""Total matches `grep` may return across all files.

A bound on a capability we *do* grant, which is why it is pinned while
`max_execute_timeout` is not: that one bounds `execute`, and `execute` is
withheld, so pinning it would assert something about a tool nobody has.
"""

_PINNED_FS_BOUNDS = {
    "tool_token_limit_before_evict": TOOL_RESULT_TOKEN_LIMIT,
    "human_message_token_limit_before_evict": HUMAN_MESSAGE_TOKEN_LIMIT,
    "grep_max_count": GREP_MATCH_LIMIT,
}

_ALL_FILESYSTEM_TOOLS = frozenset(get_args(FsToolName))
_FS_MIDDLEWARE_PARAMS = frozenset(inspect.signature(FilesystemMiddleware.__init__).parameters)

# The least-privilege allowlist is defined by subtraction, and the subtraction is
# checked here rather than assumed. If deepagents adds a filesystem tool, this
# fails at import instead of silently granting the agent a capability nobody
# chose. Deciding to include a new tool is then a deliberate edit.
require(SHELL_TOOL_NAME in _ALL_FILESYSTEM_TOOLS, f"{SHELL_TOOL_NAME} is no longer an fs tool")
require(
    set(DEFAULT_FILESYSTEM_TOOLS) == _ALL_FILESYSTEM_TOOLS - {SHELL_TOOL_NAME},
    "deepagents changed its filesystem tool set; review DEFAULT_FILESYSTEM_TOOLS "
    f"deliberately. Now offered: {sorted(_ALL_FILESYSTEM_TOOLS)}",
)
# Every pinned bound must still be a real parameter, and must still agree with
# what deepagents defaults to. Agreement is what makes these safe to state: if
# an upstream default moves, a human decides whether ours moves with it rather
# than finding out from a context window that behaves differently.
_FS_SIGNATURE = inspect.signature(FilesystemMiddleware.__init__).parameters
for _name, _pinned in _PINNED_FS_BOUNDS.items():
    require(_name in _FS_SIGNATURE, f"FilesystemMiddleware no longer accepts {_name}")
    require(
        _FS_SIGNATURE[_name].default == _pinned,
        f"deepagents changed its default for {_name}: ours is {_pinned}, theirs is now "
        f"{_FS_SIGNATURE[_name].default}. Review the bound deliberately before updating.",
    )

# `_permissions` is private API. Pin it: losing it silently would drop every
# permission rule (see docs/findings.md).
require(
    {"tools", "_permissions"} <= _FS_MIDDLEWARE_PARAMS,
    "FilesystemMiddleware no longer accepts tools/_permissions; build_agent must change",
)


def require_withheld(withheld: str, granted: Collection[str | None], where: str) -> None:
    """Assert `withheld` is absent from `granted`, and that the absence means something.

    Two failures, named separately, because they are different bugs: an empty
    grant means the check could not have failed, and a present capability means
    the allowlist did not hold. `where` names the graph or middleware inspected,
    so a leak says which one leaked.
    """
    require(
        len(granted) > 0,
        f"nothing at all is granted in {where}, so the absence of {withheld} is vacuous",
    )
    require(
        withheld not in granted,
        f"{withheld} leaked into {where}: granted {sorted(n for n in granted if n)}",
    )


def require_granted(needed: str, granted: Collection[str | None], where: str) -> None:
    """Assert `needed` is present — the positive twin, for checks that would
    otherwise pass by never exercising the tool they are about."""
    require(
        needed in granted,
        f"{needed} is not granted in {where}, so any check that uses it is vacuous: "
        f"granted {sorted(n for n in granted if n)}",
    )


def least_privilege_filesystem(
    permissions: list[FilesystemPermission] | None,
    backend: BackendProtocol | None = None,
) -> FilesystemMiddleware:
    """The filesystem middleware `build_agent` installs in place of the default.

    Carries the narrowed tool allowlist, the caller's permission rules, the
    backend the tools act on, and the bounds on how much context they may
    consume — because this instance replaces the one `create_deep_agent` would
    have built with all of that already attached.

    `backend` defaults to a `StateBackend`, which is also what deepagents would
    have chosen; it is passed explicitly because "inherited a library default"
    and "chose the safest option" are different claims about the same object.
    **A `backend` given to `create_deep_agent` must also be given here**, or the
    filesystem tools and everything else deepagents wires that backend into
    (skills, summarisation) end up on different filesystems.
    """
    middleware = FilesystemMiddleware(
        backend=backend if backend is not None else StateBackend(),
        tools=list(DEFAULT_FILESYSTEM_TOOLS),
        _permissions=permissions,
        tool_token_limit_before_evict=TOOL_RESULT_TOKEN_LIMIT,
        human_message_token_limit_before_evict=HUMAN_MESSAGE_TOKEN_LIMIT,
        grep_max_count=GREP_MATCH_LIMIT,
    )

    # Postcondition: passing a bound and having it land are different claims,
    # and all four attributes below are private API.
    require(
        middleware.backend is not None,
        "FilesystemMiddleware kept no backend; the filesystem tools have nothing to act on",
    )
    require(
        middleware._tool_token_limit_before_evict == TOOL_RESULT_TOKEN_LIMIT,
        "FilesystemMiddleware did not retain the tool-result token bound",
    )
    require(
        middleware._human_message_token_limit_before_evict == HUMAN_MESSAGE_TOKEN_LIMIT,
        "FilesystemMiddleware did not retain the human-message token bound",
    )
    require(
        middleware._grep_max_count == GREP_MATCH_LIMIT,
        "FilesystemMiddleware did not retain the grep match bound",
    )

    # Postcondition: `_permissions` is private API, so pin that it is really
    # where the rules land. Losing this silently disables every rule. The
    # middleware stores `list(_permissions or [])`, so None normalises to [].
    require(
        middleware._permissions == list(permissions or []),
        "FilesystemMiddleware did not retain the permission rules",
    )

    # The capability we withheld must actually be absent from the tools this
    # middleware contributes — the allowlist is only a request until checked.
    granted = {getattr(t, "name", None) for t in middleware.tools}
    require_withheld(SHELL_TOOL_NAME, granted, "the least-privilege middleware")
    return middleware



def _tools_by_name(agent: CompiledStateGraph[Any, Any, Any, Any]) -> dict[str, Any]:
    """The compiled graph's tool mapping.

    Reaches through langgraph internals, so the structure is pinned here: if it
    moves, this fails loudly rather than returning nothing and making every
    capability assertion vacuous.
    """
    # Explicit raises rather than require(): mypy narrows `if ... raise`, but
    # cannot narrow through a helper call.
    node = agent.nodes.get("tools")
    if node is None:
        raise CheckFailed("compiled agent has no 'tools' node; graph structure changed")
    by_name: dict[str, Any] | None = getattr(node.bound, "tools_by_name", None)
    if by_name is None:
        raise CheckFailed("ToolNode has no tools_by_name; graph structure changed")
    return by_name


def compiled_tool_names(agent: CompiledStateGraph[Any, Any, Any, Any]) -> frozenset[str]:
    """Tool names actually bound in a compiled agent."""
    return frozenset(_tools_by_name(agent))


def _closure_values(fn: Callable[..., Any] | None) -> dict[str, Any]:
    """A function's closure as a name-to-value mapping, empty if it has none.

    `strict=True` is the assertion here, not decoration: `co_freevars` and
    `__closure__` are parallel by construction, so a length mismatch would mean
    CPython's own invariant broke and every name read afterwards would be wrong.
    Zipping strictly turns that into a `ValueError` instead of a silent misread.
    """
    if fn is None or fn.__closure__ is None:
        return {}
    return dict(
        zip(fn.__code__.co_freevars, [cell.cell_contents for cell in fn.__closure__], strict=True)
    )


def subagent_graphs(
    agent: CompiledStateGraph[Any, Any, Any, Any],
) -> dict[str, CompiledStateGraph[Any, Any, Any, Any]]:
    """Every subagent graph the `task` tool can dispatch to, by name.

    An allowlist on the parent graph proves nothing on its own: deepagents gives
    its general-purpose subagent a `FilesystemMiddleware` of its own, with no
    tool allowlist at all (F20). What withholds `execute` there is deepagents
    merging middleware by `.name` into the subagent's list too — behaviour we do
    not control and therefore have to read back.

    Returns `{}` when no `task` tool is bound: nothing can be dispatched, so
    there is nothing to prove. Every *other* way of finding nothing raises,
    because reporting "no subagents" for a structure that moved would pass every
    absence check without checking anything.
    """
    task = _tools_by_name(agent).get(SUBAGENT_TASK_TOOL_NAME)
    if task is None:
        return {}

    graphs: Any = None
    for attribute in ("func", "coroutine"):
        graphs = _closure_values(getattr(task, attribute, None)).get(_SUBAGENT_GRAPHS_FREEVAR)
        if graphs is not None:
            break

    if graphs is None:
        raise CheckFailed(
            f"the {SUBAGENT_TASK_TOOL_NAME} tool no longer carries {_SUBAGENT_GRAPHS_FREEVAR} "
            f"in its closure; deepagents changed how subagents are held and the "
            f"subagent capability checks can no longer see them"
        )
    if not isinstance(graphs, dict):
        raise CheckFailed(
            f"{_SUBAGENT_GRAPHS_FREEVAR} is a {type(graphs).__name__}, not a mapping of "
            f"name to graph; deepagents changed how subagents are held"
        )
    if not graphs:
        raise CheckFailed(
            f"a {SUBAGENT_TASK_TOOL_NAME} tool is bound but carries no subagents, so every "
            f"subagent capability check would be vacuous"
        )
    return dict(graphs)
