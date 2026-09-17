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
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, get_args

from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    FsToolName,
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
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from my_agent.negative_space import CheckFailed, require

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

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
"""Hugging Face Inference Providers router (serverless), OpenAI-compatible.

Not to be confused with `https://api.endpoints.huggingface.cloud/`, which is the
Inference Endpoints *control plane* for creating and managing dedicated
deployments. A dedicated endpoint serves inference at its own
`https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1/` URL — point
`ModelConfig.base_url` there to use one; no code change is needed.
"""

DEFAULT_MODEL = "openai/gpt-oss-120b"
"""`org/model`, optionally suffixed to steer routing: `:provider`, `:fastest`,
`:cheapest`."""

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2

API_KEY_ENV_VAR = "HF_TOKEN"
MODEL_ENV_VAR = "MODEL_ID"

DEFAULT_AGENT_NAME = "my-agent"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

USE_RESPONSES_API = False
"""Pinned, never inferred, and never configurable.

With `use_responses_api=None` (the library default) `ChatOpenAI` picks the
endpoint from the model name and the request payload, independent of `base_url`
— `langchain_openai.chat_models.base.BaseChatOpenAI._use_responses_api`. The HF
router serves `/v1/chat/completions` only, so an inferred switch would fail at
request time rather than here.
"""

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

_URL_SCHEMES = ("http://", "https://")
_MAX_TEMPERATURE = 2.0


# --------------------------------------------------------------------------
# Load-time contract between our configs and the libraries they configure
# --------------------------------------------------------------------------


def _chat_openai_param_names() -> frozenset[str]:
    """Constructor keywords `ChatOpenAI` accepts: field names and their aliases.

    `ChatOpenAI.__init__` is pydantic-generated (`**data`), so `inspect.signature`
    reveals nothing usable; the model fields are the real contract.
    """
    names: set[str] = set()
    for name, field in ChatOpenAI.model_fields.items():
        names.add(name)
        if field.alias is not None:
            names.add(field.alias)

    # Postconditions. This set is what `_check_config_contract` validates against,
    # so a silently degraded result would make that check vacuous. Aliases are the
    # part most likely to move: every keyword below is one ModelConfig relies on,
    # and three of the four are aliases rather than field names.
    require(names != set(), "ChatOpenAI exposes no model_fields; introspection broke")
    missing = {"model", "base_url", "api_key", "timeout"} - names
    require(not missing, f"ChatOpenAI no longer accepts {sorted(missing)}; ModelConfig must change")
    return frozenset(names)


_CHAT_OPENAI_PARAMS = _chat_openai_param_names()
_CREATE_DEEP_AGENT_PARAMS = frozenset(inspect.signature(create_deep_agent).parameters)


def _check_config_contract(
    config_cls: type,
    callee_params: frozenset[str],
    callee_name: str,
    injected: frozenset[str],
) -> None:
    """Fail at import if a config field is not a real parameter of its callee.

    This is the check that makes `as_kwargs()` splatting safe. Without it, a
    misspelled field or an upstream rename would surface as a `TypeError` from
    deep inside the library on the first call — or worse, be silently swallowed
    by a `**kwargs` signature. Here it names the offending field at load time.
    """
    require(callee_params != frozenset(), f"could not introspect {callee_name} parameters")

    field_names = {f.name for f in fields(config_cls)}
    require(field_names != set(), f"{config_cls.__name__} has no fields")

    unknown = field_names - callee_params
    require(
        not unknown,
        f"{config_cls.__name__} fields are not {callee_name} parameters: {sorted(unknown)}",
    )

    conflicting = field_names & injected
    require(
        not conflicting,
        f"{config_cls.__name__} must not configure {sorted(conflicting)}; "
        f"those are supplied by the factory and would collide when splatted",
    )


_MODEL_INJECTED_PARAMS = frozenset({"use_responses_api"})
_AGENT_INJECTED_PARAMS = frozenset({"model"})

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


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Everything needed to reach a chat model.

    Field names are `ChatOpenAI` constructor keywords. Frozen, so it cannot be
    invalidated after construction and `build_model` needs no defensive re-checks.

    Construct directly in tests; use `from_env` at the process edge.
    """

    api_key: SecretStr
    model: str = DEFAULT_MODEL
    base_url: str = HF_ROUTER_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT_S
    """Seconds."""
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        # Programmer errors: every one of these is fixed in code, not at runtime.
        # SecretStr keeps the token out of repr(), str() and dataclasses.asdict().
        raw_key = self.api_key.get_secret_value()
        require(raw_key != "", "api_key must not be empty")
        require(
            raw_key == raw_key.strip(),
            "api_key has leading/trailing whitespace; the router answers 401 for this",
        )
        require(self.model != "", "model must not be empty")
        require(
            self.base_url.startswith(_URL_SCHEMES),
            f"base_url must be an http(s) URL, got {self.base_url!r}",
        )
        require(
            0.0 <= self.temperature <= _MAX_TEMPERATURE,
            f"temperature must be in [0.0, {_MAX_TEMPERATURE}], got {self.temperature}",
        )
        require(self.timeout > 0.0, f"timeout must be positive, got {self.timeout}")
        require(self.max_retries >= 0, f"max_retries must be non-negative, got {self.max_retries}")

    def as_kwargs(self) -> dict[str, Any]:
        """Constructor keywords for `ChatOpenAI`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        """Read the config from the environment.

        Absent or malformed environment is an *operating* error — the outside
        world failed to supply something, which is not a bug in this code — so it
        raises `ValueError` to be handled at the edge rather than tripping a check.

        `env` is injectable so tests never touch the real process environment.
        """
        source: Mapping[str, str] = os.environ if env is None else env

        api_key = source.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(
                f"{API_KEY_ENV_VAR} is unset or empty. Set it in .env "
                f"(see .env.example) or export it before starting the agent."
            )

        model = source.get(MODEL_ENV_VAR, DEFAULT_MODEL).strip()
        if not model:
            raise ValueError(f"{MODEL_ENV_VAR} is set but empty; unset it to use the default.")

        return cls(api_key=SecretStr(api_key), model=model)


_check_config_contract(ModelConfig, _CHAT_OPENAI_PARAMS, "ChatOpenAI", _MODEL_INJECTED_PARAMS)


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


_check_config_contract(
    AgentConfig, _CREATE_DEEP_AGENT_PARAMS, "create_deep_agent", _AGENT_INJECTED_PARAMS
)


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------


def build_model(config: ModelConfig) -> ChatOpenAI:
    """Build the chat model for the HF router.

    `ChatOpenAI` rather than `init_chat_model`: the latter resolves a provider
    from the model string, and `org/model:provider` router ids are meaningless
    to it.

    Returns the concrete type while `build_agent` accepts the abstract one —
    specific in what we return, liberal in what we accept.
    """
    require(isinstance(config, ModelConfig), f"expected a ModelConfig, got {type(config).__name__}")

    model = ChatOpenAI(**config.as_kwargs(), use_responses_api=USE_RESPONSES_API)

    # Postconditions. ChatOpenAI falls back to OPENAI_API_BASE / OPENAI_BASE_URL
    # and rewrites `temperature` for some model families, so what we asked for is
    # not necessarily what we got. A silent redirect to api.openai.com would
    # otherwise surface as a confusing auth failure much later.
    require(
        model.openai_api_base == config.base_url,
        f"base_url was overridden: asked {config.base_url!r}, got {model.openai_api_base!r}",
    )
    require(
        model.model_name == config.model,
        f"model was overridden: asked {config.model!r}, got {model.model_name!r}",
    )
    require(
        model.use_responses_api is False,
        "use_responses_api must stay False; the HF router is Chat Completions only",
    )
    return model


def _least_privilege_filesystem(
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
        _least_privilege_filesystem(permissions),
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
