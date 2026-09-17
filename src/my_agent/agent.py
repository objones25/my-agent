"""Composition root: config -> model -> agent.

This is the only module that knows about Hugging Face, base URLs, tokens, or the
environment. Everything downstream accepts a `BaseChatModel` and a compiled graph
and stays ignorant of where they came from.

**Configs are parameter objects, not argument lists.** Each config's field names
are exactly the callee's parameter names, and the factories splat them
(`ChatOpenAI(**config.as_kwargs())`). Adding a setting later — `subagents`,
`backend`, `permissions`, `top_p` — is one new field with a default. No factory
signature changes, no factory body changes, no call site changes.

The coupling that buys is checked at import time rather than assumed: every
config field is verified against the callee's real signature when this module
loads, so a typo or an upstream rename fails immediately and by name.
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
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    compiled_tool_names,
    least_privilege_filesystem,
)
from my_agent.contracts import check_config_contract
from my_agent.model import (
    API_KEY_ENV_VAR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    HF_ROUTER_BASE_URL,
    MODEL_ENV_VAR,
    USE_RESPONSES_API,
    ModelConfig,
    build_model,
)
from my_agent.negative_space import require

__all__ = [
    "API_KEY_ENV_VAR",
    "DEFAULT_AGENT_NAME",
    "DEFAULT_FILESYSTEM_TOOLS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "HF_ROUTER_BASE_URL",
    "MODEL_ENV_VAR",
    "SHELL_TOOL_NAME",
    "USE_RESPONSES_API",
    "AgentConfig",
    "ModelConfig",
    "build_agent",
    "build_model",
    "compiled_tool_names",
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

    # `permissions` reaches the tool layer only through FilesystemMiddleware's
    # `_permissions`. Replacing that middleware without forwarding them drops
    # every rule silently, so refuse the combination rather than half-apply it.
    caller_filesystem = [m for m in agent_config.middleware if isinstance(m, FilesystemMiddleware)]
    require(
        not (caller_filesystem and agent_config.permissions),
        "a FilesystemMiddleware in AgentConfig.middleware replaces the one build_agent "
        "installs and would silently drop AgentConfig.permissions; pass the rules to "
        "that middleware's own _permissions instead",
    )

    permissions = list(agent_config.permissions) or None
    kwargs = agent_config.as_kwargs()
    kwargs["permissions"] = permissions
    kwargs["middleware"] = [
        least_privilege_filesystem(permissions),
        *agent_config.middleware,
    ]

    agent = create_deep_agent(model=model, **kwargs)

    require(agent is not None, "create_deep_agent returned None")

    # Postcondition: the capability we withheld must actually be absent. This is
    # the guarantee that matters, and it is cheap to state once per build.
    if not caller_filesystem:
        bound = compiled_tool_names(agent)
        require(
            SHELL_TOOL_NAME not in bound,
            f"{SHELL_TOOL_NAME} leaked into the agent despite the allowlist: {sorted(bound)}",
        )
    return agent
