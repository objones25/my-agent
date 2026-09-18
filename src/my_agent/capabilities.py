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
from importlib.metadata import entry_points
from typing import Any, get_args

from deepagents import FilesystemMiddleware, FilesystemPermission, FsToolName
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.graph.state import CompiledStateGraph

from my_agent.negative_space import CheckFailed, require

__all__ = [
    "DEEPAGENTS_PLUGIN_GROUPS",
    "DEFAULT_FILESYSTEM_TOOLS",
    "GENERAL_PURPOSE_SUBAGENT_NAME",
    "GREP_MATCH_LIMIT",
    "HUMAN_MESSAGE_TOKEN_LIMIT",
    "LIBRARY_SUBAGENT_STEP_LIMIT",
    "SHELL_TOOL_NAME",
    "SUBAGENT_STEP_LIMIT",
    "SUBAGENT_TASK_TOOL_NAME",
    "TASK_DISPATCH_LIMIT",
    "TOOL_CALL_LIMIT",
    "TOOL_RESULT_TOKEN_LIMIT",
    "bound_step_limit",
    "call_limits",
    "compiled_tool_names",
    "compiled_tools",
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

GENERAL_PURPOSE_SUBAGENT_NAME = "general-purpose"
"""The subagent deepagents adds to every deep agent unless a caller supplies one.

An explicit spec of this name is the supported way to replace it (`graph.py`
0.7.15), which is how `build_agent` gets a step limit onto it.
"""

SUBAGENT_STEP_LIMIT = 25
"""Graph steps one `task` dispatch may run.

**The bound `RunBounds.step_limit` cannot reach.** langchain's `create_agent`
binds `recursion_limit: 9999` on every graph it compiles, and deepagents invokes
a subagent with that graph's own bound config — which wins the per-key merge
against the ambient parent config by design (`middleware/subagents.py`, 0.7.15).
So a caller's step limit stops at the parent: measured 2026-09-18, a
`step_limit` of 25 allowed 12 parent model calls on its own and **5002** once
each step dispatched a subagent (F24).

Equal to `run.RECURSION_LIMIT` today and deliberately not defined as it: the two
bound different graphs, and a subagent asked to do real research is the first
thing that would need a different number. Two graph steps buy one model/tool
round trip, so this is twelve of them — below the depth a tool-using model
reaches on its own (OpenAI's own gpt-oss write-up shows it chaining 28 browsing
calls in one turn), which is the trade being made until a domain says otherwise.
"""

TOOL_CALL_LIMIT = 24
"""Tool calls one run may make, across every tool.

**Not covered by `RunBounds.step_limit`.** That bounds graph *steps*, and one
step can execute any number of tool calls: langgraph's tool node runs every
call in a single `AIMessage`, so a model that fans out ten calls a turn does ten
times the work per step. The step limit sees one step either way.

Twenty-four is two calls per round trip at the default step limit (25 steps buys
12 model/tool round trips). Ordinary work never approaches it; a fan-out does.
Stated as its own number rather than derived from `run.RECURSION_LIMIT`, because
the two bound different things and the first domain that needs wide parallel
tool use will move this one alone.
"""

TASK_DISPATCH_LIMIT = 3
"""`task` dispatches one run may make.

`SUBAGENT_STEP_LIMIT` bounds how far *one* dispatch runs; nothing bounded how
many there are. With 12 parent round trips and a 25-step subagent, twelve
dispatches is ~144 model calls inside a turn the caller asked to bound at 25
steps. Three caps the worst case at roughly 37 — and a delegating agent that
needs a fourth subagent to answer one turn is a agent that has lost the thread,
not one that needs a wider budget.
"""

LIBRARY_SUBAGENT_STEP_LIMIT = 9999
"""What a subagent runs to when nobody sets a limit.

Stated so the test that asserts ours is applied has a discriminator: without
this, that test keeps passing if the library starts choosing a small number for
its own reasons, and a bound we merely agree with reads as a bound we set.
"""

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

DEEPAGENTS_PLUGIN_GROUPS = (
    "deepagents.harness_profiles",
    "deepagents.provider_profiles",
)
"""Entry-point groups through which an installed package reconfigures the agent.

**The door `KNOWN_CREATE_DEEP_AGENT_PARAMS` cannot see.** That pin catches a new
*parameter*; this is not a parameter. `create_deep_agent` resolves a
`HarnessProfile` keyed by the model's provider and id from a process-global
registry, and `deepagents.profiles` populates that registry by executing a
zero-arg callable from every installed distribution advertising one of these
groups (`profiles/_builtin_profiles.py`, 0.7.15). A profile can add
`extra_middleware`, exclude tools, override tool descriptions and rewrite the
system prompt — every one of them a capability decision, none of them visible at
a call site.

Verified 2026-09-18: both groups are empty in this environment, and the builtin
harness profiles are keyed per-model (Anthropic models, Nemotron, Codex) so none
matches the router model. A pin, not a claim of safety: adding the plugin is how
it would stop being true, and this is what makes that an import failure rather
than a quiet change of behaviour (F28).
"""

_INSTALLED_PLUGINS = {
    group: sorted(ep.name for ep in entry_points(group=group)) for group in DEEPAGENTS_PLUGIN_GROUPS
}
require(
    not any(_INSTALLED_PLUGINS.values()),
    f"a package is registering deepagents profile plugins: {_INSTALLED_PLUGINS}. Each one runs "
    f"at import and may add middleware, drop tools or rewrite the system prompt without "
    f"passing through create_deep_agent. Review what it grants, then allow it here.",
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


def compiled_tools(agent: CompiledStateGraph[Any, Any, Any, Any]) -> dict[str, Any]:
    """The tool objects actually bound in a compiled agent, by name.

    Public because identity matters, not just names: a subagent's filesystem
    tools are *the same objects* as the parent's, closing over the same
    middleware — which is why the permission rules and the allowlist reach it at
    all (F20). A caller that only needs the names wants `compiled_tool_names`.

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
    return frozenset(compiled_tools(agent))


def call_limits() -> list[ToolCallLimitMiddleware]:
    """The two per-run call bounds `build_agent` installs on every agent.

    Build-time rather than a `RunBounds` field: these are middleware, and
    middleware is fixed when the graph compiles, while `RunBounds` is chosen per
    invocation. Both numbers therefore belong here, beside the other bounds on
    capabilities we *do* grant.

    Typed as the concrete class rather than `AgentMiddleware`, so a test can
    read `run_limit` and `exit_behavior` back off it. A bound stated in a
    constructor call and never read again is a bound nothing checks.

    `run_limit`, never `thread_limit`: a thread limit counts across a
    checkpointed conversation, and the graph carries no checkpointer by default,
    so it would be a bound that never counts.

    `exit_behavior="continue"` blocks the exceeded call and lets the agent
    answer with what it already has. `"error"` would turn a model that asked for
    too much into a crashed turn, and the run is bounded twice over already by
    the step limit and the deadline; this bound exists to stop *spend*, not to
    stop the turn. The blocked call is visible in the mirror as a tool message,
    which is where a reader looks for what a run actually did.
    """
    limits = [
        ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT, exit_behavior="continue"),
        ToolCallLimitMiddleware(
            tool_name=SUBAGENT_TASK_TOOL_NAME,
            run_limit=TASK_DISPATCH_LIMIT,
            exit_behavior="continue",
        ),
    ]
    # Postcondition: deepagents merges middleware by `.name`, so two entries
    # sharing one would mean the second silently replacing the first. The names
    # are the library's to choose (`ToolCallLimitMiddleware[task]` today), which
    # is exactly why this is read back rather than assumed.
    names = [m.name for m in limits]
    require(
        len(set(names)) == len(names),
        f"the call limits share a middleware name ({names}); one would replace the other",
    )
    return limits


def bound_step_limit(graph: CompiledStateGraph[Any, Any, Any, Any]) -> int | None:
    """The `recursion_limit` bound onto `graph` itself, if any.

    `with_config` is the only lever over a subagent's step limit, because the
    subagent's own bound config beats the ambient parent one. Reading it back is
    what turns "we passed a limit" into "the limit is on the graph that runs".
    """
    config: Any = graph.config or {}
    limit: object = config.get("recursion_limit")
    if limit is None:
        return None
    # An explicit narrowing rather than `require()`: mypy cannot narrow through
    # a helper call (F10), and a limit read back as the wrong type would make
    # every comparison against it quietly false.
    if not isinstance(limit, int):
        raise CheckFailed(f"recursion_limit on the graph is a {type(limit).__name__}, not an int")
    return limit


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
    task = compiled_tools(agent).get(SUBAGENT_TASK_TOOL_NAME)
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
