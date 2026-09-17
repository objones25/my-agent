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
from typing import Any, get_args

from deepagents import FilesystemMiddleware, FilesystemPermission, FsToolName
from langgraph.graph.state import CompiledStateGraph

from my_agent.negative_space import CheckFailed, require

__all__ = [
    "DEFAULT_FILESYSTEM_TOOLS",
    "SHELL_TOOL_NAME",
    "compiled_tool_names",
    "least_privilege_filesystem",
]

SHELL_TOOL_NAME = "execute"
"""deepagents classes shell execution as a filesystem tool."""

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
# `_permissions` is private API. Pin it: losing it silently would drop every
# permission rule (see docs/findings.md).
require(
    {"tools", "_permissions"} <= _FS_MIDDLEWARE_PARAMS,
    "FilesystemMiddleware no longer accepts tools/_permissions; build_agent must change",
)


def least_privilege_filesystem(
    permissions: list[FilesystemPermission] | None,
) -> FilesystemMiddleware:
    """The filesystem middleware `build_agent` installs in place of the default.

    Carries both the narrowed tool allowlist and the caller's permission rules,
    because this instance replaces the one `create_deep_agent` would have built
    with those rules already attached.
    """
    middleware = FilesystemMiddleware(
        tools=list(DEFAULT_FILESYSTEM_TOOLS), _permissions=permissions
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
    require(
        SHELL_TOOL_NAME not in granted,
        f"{SHELL_TOOL_NAME} survived the allowlist; granted {sorted(n for n in granted if n)}",
    )
    return middleware


def compiled_tool_names(agent: CompiledStateGraph[Any, Any, Any, Any]) -> frozenset[str]:
    """Tool names actually bound in a compiled agent.

    Reaches through langgraph internals, so it is pinned by a load-time-style
    check on first use: if the structure moves, this fails loudly rather than
    returning an empty set that would make every capability assertion vacuous.
    """
    # Explicit raises rather than require(): mypy narrows `if ... raise`, but
    # cannot narrow through a helper call.
    node = agent.nodes.get("tools")
    if node is None:
        raise CheckFailed("compiled agent has no 'tools' node; graph structure changed")
    by_name: dict[str, Any] | None = getattr(node.bound, "tools_by_name", None)
    if by_name is None:
        raise CheckFailed("ToolNode has no tools_by_name; graph structure changed")
    return frozenset(by_name)
