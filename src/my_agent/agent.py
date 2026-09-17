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

The one field `build_agent` does not pass straight through is `middleware` —
see `capabilities.py` for why the shell tool has to be withheld by replacing
deepagents' own filesystem middleware.
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
    kwargs["middleware"] = [least_privilege_filesystem(permissions), *config.middleware]

    # Postcondition: the two places the rules have to land must agree. They are
    # set three lines apart today, which is exactly how they drift later.
    require(
        kwargs["middleware"][0]._permissions == list(permissions or []),
        "assembled middleware does not carry the permissions passed to create_deep_agent",
    )
    return kwargs


def _require_shell_withheld(agent: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    """Assert the withheld capability is really absent from the compiled graph.

    The allowlist is only a request until something reads back what was bound.
    """
    bound = compiled_tool_names(agent)
    require(bound != frozenset(), "no tools bound at all; the absence check would be vacuous")
    require(
        SHELL_TOOL_NAME not in bound,
        f"{SHELL_TOOL_NAME} leaked into the agent despite the allowlist: {sorted(bound)}",
    )


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
