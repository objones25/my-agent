"""Compiling the deep agent: `AgentConfig` -> `create_deep_agent`.

Takes a `BaseChatModel` and knows nothing about where it came from — no router,
no tokens, no environment. That is `model.py`'s job, and the two modules import
nothing from each other.

**`AgentConfig` is a parameter object, not an argument list.** Its field names
are exactly `create_deep_agent` parameter names, and `build_agent` splats them.
Giving the agent `subagents`, `skills`, a `backend` or `interrupt_on` is one new
field with a default: no factory signature change, no factory body change, no
call site change. That coupling is checked at import rather than assumed, so a
typo or an upstream rename fails when this module loads, by name.

Two fields `build_agent` does not simply pass through:

- `middleware`, because the shell tool can only be withheld by replacing
  deepagents' own filesystem middleware — see `capabilities.py`.
- `backend`, because replacing that middleware means the backend has to be
  handed to *both* `create_deep_agent` and the replacement. `_agent_kwargs`
  already forwards it, so the field really is one line; it did not used to be,
  and the two would have ended up on different filesystems (F21).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    create_deep_agent,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from my_agent.capabilities import (
    SHELL_TOOL_NAME,
    compiled_tool_names,
    least_privilege_filesystem,
    require_withheld,
    subagent_graphs,
)
from my_agent.contracts import check_config_contract
from my_agent.negative_space import require

__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "AgentConfig",
    "build_agent",
]

# --------------------------------------------------------------------------
# Defaults. Change them here, nowhere else.
# --------------------------------------------------------------------------

DEFAULT_AGENT_NAME = "my-agent"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# --------------------------------------------------------------------------
# Load-time contract between our configs and the libraries they configure
# --------------------------------------------------------------------------


_CREATE_DEEP_AGENT_PARAMS = frozenset(inspect.signature(create_deep_agent).parameters)

KNOWN_CREATE_DEEP_AGENT_PARAMS = frozenset(
    {
        "model",
        "tools",
        "system_prompt",
        "middleware",
        "subagents",
        "skills",
        "memory",
        "permissions",
        "backend",
        "interrupt_on",
        "response_format",
        "state_schema",
        "context_schema",
        "checkpointer",
        "store",
        "debug",
        "name",
        "cache",
    }
)
"""Every parameter `create_deep_agent` had when this code was reviewed (0.7.15).

`check_config_contract` asserts our *fields* are real parameters. It cannot
notice a **new** parameter appearing — and a new parameter is exactly how the
shell `execute` tool arrived switched on with no opt-in (F4). Least privilege
says a capability is off unless something turns it on deliberately, so an
upstream addition has to be reviewed rather than inherited.

Pinning the set makes that review mandatory: a deepagents upgrade that adds a
parameter fails this import, by name, instead of changing behaviour quietly.
Adding the name here is the deliberate acceptance.
"""

_NEW_PARAMS = _CREATE_DEEP_AGENT_PARAMS - KNOWN_CREATE_DEEP_AGENT_PARAMS
require(
    not _NEW_PARAMS,
    f"create_deep_agent gained parameters {sorted(_NEW_PARAMS)}. Review each for what it "
    f"enables by default before adding it to KNOWN_CREATE_DEEP_AGENT_PARAMS — this is the "
    f"same door the shell `execute` tool came through (F4).",
)

_REMOVED_PARAMS = KNOWN_CREATE_DEEP_AGENT_PARAMS - _CREATE_DEEP_AGENT_PARAMS
require(
    not _REMOVED_PARAMS,
    f"create_deep_agent no longer accepts {sorted(_REMOVED_PARAMS)}; "
    f"AgentConfig and this pin must change together",
)

_AGENT_INJECTED_PARAMS = frozenset({"model"})

# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Everything `create_deep_agent` needs except the model.

    Field names are `create_deep_agent` parameter names. To give the agent
    subagents, skills, a backend, permissions, or interrupts, add the field —
    `build_agent` does not change.
    """

    name: str = DEFAULT_AGENT_NAME
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] = ()
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] = ()
    """Extra middleware. Empty by default — nothing is added speculatively.

    A `FilesystemMiddleware` here *replaces* the least-privilege one `build_agent`
    installs, because deepagents merges middleware by `.name`. That is the
    supported way to re-enable `execute`, and it means you own the tool allowlist
    and the permission rules from then on.
    """
    permissions: Sequence[FilesystemPermission] = ()
    """Filesystem access rules. Empty by default — no rules, not "deny all"."""

    def __post_init__(self) -> None:
        # Coerce before validating. `frozen=True` stops a field being rebound but
        # not the sequence behind it being mutated, so validating the caller's own
        # list leaves every check below defeatable by a later append -- and for
        # `permissions` and `middleware` that means a capability decision that
        # can be changed after it was reviewed. `object.__setattr__` is how a
        # frozen dataclass assigns; it works with `slots=True`.
        #
        # The boundary, stated exactly, because overstating it is worse than not
        # claiming it: `tuple()` is shallow, so what is frozen is the *sequence*
        # -- its length and which objects are in it. The objects themselves are
        # deepagents' own mutable classes and are not frozen by this:
        # `FilesystemPermission` is a dataclass declared `frozen=False`, and
        # `AgentMiddleware` is an ordinary class. Verified against the installed
        # wheel: on a rule already inside a validated AgentConfig, both
        # `rule.paths.append("/")` and `rule.mode = "allow"` succeed, so a
        # reviewed *deny* rule can still be flipped to *allow* after
        # construction. Deepcopying instead was considered and rejected --
        # it would break the identity assertion in
        # `test_agent_config_freezes_the_rules_it_was_given` and buy little,
        # since the caller who can reach inside a rule can also build a
        # different config. What this guarantees is that no rule can be *added*
        # or *removed* behind the validation; not that each rule's own fields
        # are immutable.
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "middleware", tuple(self.middleware))
        object.__setattr__(self, "permissions", tuple(self.permissions))

        require(self.name != "", "name must not be empty")
        require(self.system_prompt.strip() != "", "system_prompt must not be blank")
        for entry in self.middleware:
            require(
                isinstance(entry, AgentMiddleware),
                f"middleware entries must be AgentMiddleware, got {type(entry).__name__}",
            )
        for rule in self.permissions:
            require(
                isinstance(rule, FilesystemPermission),
                f"permissions entries must be FilesystemPermission, got {type(rule).__name__}",
            )

    def as_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `create_deep_agent`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


check_config_contract(
    AgentConfig, _CREATE_DEEP_AGENT_PARAMS, "create_deep_agent", _AGENT_INJECTED_PARAMS
)


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------


def _replaces_filesystem_middleware(config: AgentConfig) -> bool:
    """Whether the caller supplied their own `FilesystemMiddleware`.

    deepagents merges middleware by `.name`, so a caller-supplied one displaces
    the least-privilege middleware entirely — allowlist, permission rules and
    all. Two decisions hang on this: `_agent_kwargs` refuses to pair it with
    `permissions`, and `build_agent` stops asserting a tool allowlist it no
    longer owns.
    """
    return any(isinstance(m, FilesystemMiddleware) for m in config.middleware)


def _agent_kwargs(config: AgentConfig) -> dict[str, Any]:
    """`create_deep_agent` keywords, with the middleware list assembled.

    Everything splats straight from the config except `middleware`, which gains
    the least-privilege `FilesystemMiddleware` in front of the caller's own.
    Extracted from `build_agent` because this is where the subtle failure lives
    and it is worth testing without building a model or compiling a graph.
    """
    # `permissions` reaches the tool layer only through FilesystemMiddleware's
    # `_permissions`. Replacing that middleware without forwarding them drops
    # every rule silently, so refuse the combination rather than half-apply it.
    require(
        not (_replaces_filesystem_middleware(config) and config.permissions),
        "a FilesystemMiddleware in AgentConfig.middleware replaces the one build_agent "
        "installs and would silently drop AgentConfig.permissions; pass the rules to "
        "that middleware's own _permissions instead",
    )

    # Empty means "no rules", and deepagents spells that `None`, not `[]`.
    permissions = list(config.permissions) or None
    kwargs = config.as_kwargs()
    kwargs["permissions"] = permissions

    # `backend` is read the same way, and for the same reason: deepagents wires
    # it into skills and summarisation while our middleware owns the filesystem
    # tools, so a backend that reached only one of them would put one agent on
    # two filesystems (F21). `.get` rather than `[...]` because `AgentConfig`
    # has no such field today — this is what makes adding it the one-line change
    # the class docstring promises.
    kwargs["middleware"] = [
        least_privilege_filesystem(permissions, kwargs.get("backend")),
        *config.middleware,
    ]

    # Postconditions: every place a setting has to land must agree. They are set
    # a few lines apart today, which is exactly how they drift later.
    require(
        kwargs["middleware"][0]._permissions == list(permissions or []),
        "assembled middleware does not carry the permissions passed to create_deep_agent",
    )
    declared_backend = kwargs.get("backend")
    require(
        declared_backend is None or kwargs["middleware"][0].backend is declared_backend,
        "assembled middleware is on a different backend than create_deep_agent will use",
    )
    return kwargs


def _require_shell_withheld(agent: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    """Assert the withheld capability is absent from every graph that can run.

    The allowlist is only a request until something reads back what was bound —
    and the parent graph is not the only thing that runs. deepagents builds its
    general-purpose subagent a `FilesystemMiddleware` of its own with *no* tool
    allowlist, so a subagent is a second place `execute` can appear. Ours
    reaches it only because deepagents merges middleware by `.name` into the
    subagent's list too, which is behaviour we do not control (F20) — hence a
    read-back rather than an assumption.
    """
    require_withheld(SHELL_TOOL_NAME, compiled_tool_names(agent), "the compiled graph")

    graphs = subagent_graphs(agent)
    # deepagents adds a general-purpose subagent unless a caller supplies its
    # own spec, and `AgentConfig` cannot express one. So an empty mapping here
    # means the reader lost them, not that none exist.
    require(
        graphs != {},
        "no subagent graphs found, so the subagent capability check is vacuous; "
        "deepagents binds a `task` tool by default and its subagents must be readable",
    )
    for name, graph in graphs.items():
        require_withheld(SHELL_TOOL_NAME, compiled_tool_names(graph), f"subagent {name!r}")


def build_agent(
    model: BaseChatModel, config: AgentConfig | None = None
) -> CompiledStateGraph[AgentState[Any], Any, InputAgentState, OutputAgentState[Any]]:
    """Compile the deep agent.

    Takes an already-built `BaseChatModel` rather than a `ModelConfig`: that is
    the seam. Swapping the model source — a different provider, a fake in tests —
    changes the caller, never this function.

    `create_deep_agent` also accepts a model *string*, which we refuse: a string
    would be resolved by langchain's own provider inference and silently bypass
    the router configuration in `build_model`.

    **`middleware` is the one field this function does not simply pass through.**
    `create_deep_agent` enables the shell `execute` tool with no opt-in, and the
    only supported way to withhold it is to supply a `FilesystemMiddleware` with
    a narrower `tools` allowlist — deepagents merges middleware by `.name`, so
    ours replaces the default. Everything else still splats untouched, so adding
    a config field remains a one-line change.
    """
    require(
        isinstance(model, BaseChatModel),
        f"expected a BaseChatModel, got {type(model).__name__}; build it with build_model()",
    )
    agent_config = AgentConfig() if config is None else config
    require(
        isinstance(agent_config, AgentConfig),
        f"expected an AgentConfig, got {type(agent_config).__name__}",
    )

    agent = create_deep_agent(model=model, **_agent_kwargs(agent_config))

    require(agent is not None, "create_deep_agent returned None")

    # A caller-supplied FilesystemMiddleware owns the allowlist from then on, so
    # there is no allowlist of ours left to assert.
    if not _replaces_filesystem_middleware(agent_config):
        _require_shell_withheld(agent)
    return agent
