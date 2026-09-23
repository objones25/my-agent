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
    CompiledSubAgent,
    FilesystemMiddleware,
    FilesystemPermission,
    SubAgent,
    create_deep_agent,
)
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware import SummarizationMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
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
    GENERAL_PURPOSE_SUBAGENT_NAME,
    PARENT_STEP_LIMIT,
    SHELL_TOOL_NAME,
    SUBAGENT_STEP_LIMIT,
    bound_step_limit,
    bounded_compaction,
    call_limits,
    compiled_output_keys,
    compiled_tool_names,
    least_privilege_filesystem,
    require_withheld,
    subagent_graphs,
)
from my_agent.contracts import check_config_contract, check_known_parameters
from my_agent.negative_space import CheckFailed, require

__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "KNOWN_OUTPUT_STATE_KEYS",
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

check_known_parameters(
    _CREATE_DEEP_AGENT_PARAMS,
    KNOWN_CREATE_DEEP_AGENT_PARAMS,
    "create_deep_agent",
    "Review each for what it enables by default before adding it to "
    "KNOWN_CREATE_DEEP_AGENT_PARAMS — this is the same door the shell `execute` tool "
    "came through (F4).",
)

KNOWN_OUTPUT_STATE_KEYS = frozenset({"files", "messages", "structured_response"})
"""Every state key a compiled agent declared it may return, when this was reviewed.

`TurnResult.state` carries whatever comes back, so a new key is not lost — but
nothing would *say* it had appeared, and a key nobody reviewed is a key nobody
decided to surface. That is not hypothetical: `files` arrived this way, declared
by `FilesystemMiddleware`, returned on every turn and read by nothing until F40.

The same argument as `KNOWN_CREATE_DEEP_AGENT_PARAMS`, one layer down — that pin
catches a new *parameter*, this catches a new *output*. Adding the name here is
the deliberate acceptance, and the place to ask whether `TurnResult` should give
it a property of its own the way `files` has one.
"""


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
    """Filesystem access rules. Empty by default — no rules, not "deny all".

    A rule with `mode="interrupt"` is how a tool call is routed through a human,
    and deepagents turns those into `interrupt_on` entries for the parent *and*
    every subagent. Pausing needs `checkpointer` to also be set, or the pause
    can never be resumed — `run.run_turn` refuses that combination rather than
    handing back a turn nobody can finish.
    """
    subagents: Sequence[SubAgent | CompiledSubAgent] = ()
    """Subagent specs. Empty by default, which is *not* the same as no subagents.

    deepagents adds a general-purpose one behind `task` unless a caller supplies
    a spec of that name, and it compiles at langchain's `recursion_limit` of
    9999 (F24). `build_agent` therefore supplies one: deepagents' own subagent,
    rebound to `SUBAGENT_STEP_LIMIT`. A spec named `general-purpose` here
    replaces that, limit included — it is then the caller's graph and the
    caller's bound.
    """
    checkpointer: Any = None
    """langgraph checkpointer, or `None` for a graph that persists nothing.

    Only needed to resume an interrupted turn today. Typed `Any` because
    deepagents accepts `None | bool | BaseCheckpointSaver` and narrowing it here
    would make this field a different contract from the parameter it splats to.
    """

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
        object.__setattr__(self, "subagents", tuple(self.subagents))

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


def _supplies_general_purpose_subagent(config: AgentConfig) -> bool:
    """Whether the caller replaced deepagents' default subagent themselves.

    An explicit spec of that name is how deepagents lets a caller override it,
    so one here means the subagent — and its step limit — is the caller's.
    """
    return any(
        spec.get("name") == GENERAL_PURPOSE_SUBAGENT_NAME
        for spec in config.subagents
        if isinstance(spec, dict)
    )


def _agent_kwargs(config: AgentConfig, model: BaseChatModel) -> dict[str, Any]:
    """`create_deep_agent` keywords, with the middleware list assembled.

    Everything splats straight from the config except `middleware`, which gains
    the least-privilege `FilesystemMiddleware`, the bounded compaction
    middleware and the two call limits in front of the caller's own. Extracted
    from `build_agent` because this is where the subtle failures live and it is
    worth testing without compiling a graph.

    Takes the `model` only because compaction needs one: the summary that
    replaces a conversation is written by the same model the agent runs on.
    Nothing else here looks at it.
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
    # deepagents spells "no subagents" `None` too, and an empty list would stop
    # it adding the general-purpose one — which is the graph `build_agent` needs
    # to exist before it can bind a step limit onto it.
    kwargs["subagents"] = list(config.subagents) or None

    # One backend object for every consumer of one. deepagents wires its own
    # into skills and compaction while our middleware owns the filesystem tools,
    # so a backend that reached only some of them would put one agent on two
    # filesystems (F21). `.get` rather than `[...]` because `AgentConfig` has no
    # such field today — that is what makes adding it the one-line change the
    # class docstring promises — and naming the fallback here rather than
    # letting each consumer default separately is what makes the postcondition
    # below mean something. It used to read `declared_backend is None or ...`,
    # which on the default path compared `None` against `None` and passed while
    # `least_privilege_filesystem` and `create_deep_agent` each quietly built a
    # `StateBackend` of their own.
    backend: BackendProtocol = kwargs.get("backend") or StateBackend()
    kwargs["backend"] = backend

    # Order matters only in that ours go first: deepagents merges a caller's
    # middleware by `.name`, so a caller who wants different bounds supplies
    # middleware of the same name deliberately rather than by accident.
    kwargs["middleware"] = [
        least_privilege_filesystem(permissions, backend),
        bounded_compaction(model, backend),
        *call_limits(),
        *config.middleware,
    ]

    # Postconditions: every place a setting has to land must agree. They are set
    # a few lines apart today, which is exactly how they drift later.
    require(
        kwargs["middleware"][0]._permissions == list(permissions or []),
        "assembled middleware does not carry the permissions passed to create_deep_agent",
    )
    require(
        kwargs["middleware"][0].backend is backend,
        "assembled middleware is on a different backend than create_deep_agent will use",
    )

    # deepagents installs a compaction middleware of its own and merges by
    # `.name`, so two here would mean ours joined the stack rather than
    # replacing the one sized above the window (`bounded_compaction`). Counted
    # rather than assumed: a caller may legitimately supply their own, and that
    # is a decision to surface, not to silently take second place behind.
    compaction = [m for m in kwargs["middleware"] if isinstance(m, SummarizationMiddleware)]
    require(
        len(compaction) == 1,
        f"the assembled middleware carries {len(compaction)} compaction middlewares; "
        f"deepagents merges by name, so more than one means the agent may still run at "
        f"the library's own threshold",
    )
    return kwargs


def _bounded_general_purpose_subagent(
    agent: CompiledStateGraph[Any, Any, Any, Any],
) -> CompiledSubAgent:
    """deepagents' own general-purpose subagent, rebound to our step limit.

    Its `runnable` is the graph deepagents just compiled, not one assembled
    here. That matters: the default subagent carries summarisation, tool-call
    patching and whatever else deepagents decides it needs, and a hand-rolled
    replacement would silently drop whichever of those moved next. Taking the
    real graph and putting one config key on it keeps the behaviour and changes
    only the number.

    Two `with_config` calls then apply to the same graph — ours here, and
    deepagents' own `{metadata, run_name}` when it accepts the spec. Verified
    2026-09-18 that the later call does not clear `recursion_limit`, which is
    what makes this route work at all.
    """
    graph = subagent_graphs(agent).get(GENERAL_PURPOSE_SUBAGENT_NAME)
    if graph is None:
        raise CheckFailed(
            f"deepagents did not add a {GENERAL_PURPOSE_SUBAGENT_NAME!r} subagent to bind a "
            f"step limit onto; it is added by default, so this means the name or the "
            f"default changed and `task` is running unbounded"
        )
    bounded = graph.with_config({"recursion_limit": SUBAGENT_STEP_LIMIT})
    # Postcondition: `with_config` returns a copy, so an assertion on `graph`
    # would pass while the thing actually handed over kept the old limit.
    require(
        bound_step_limit(bounded) == SUBAGENT_STEP_LIMIT,
        f"rebinding the subagent step limit did not take: wanted {SUBAGENT_STEP_LIMIT}, "
        f"graph carries {bound_step_limit(bounded)}",
    )
    return {
        "name": GENERAL_PURPOSE_SUBAGENT_NAME,
        "description": GENERAL_PURPOSE_SUBAGENT["description"],
        "runnable": bounded,
    }


def _require_subagents_bounded(agent: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    """Assert no subagent runs to the library's limit instead of ours.

    Read back rather than assumed, for the same reason the tool allowlist is:
    the limit is applied by handing deepagents a spec, and whether it survives
    the round trip is deepagents' behaviour, not ours.
    """
    limit = bound_step_limit(subagent_graphs(agent)[GENERAL_PURPOSE_SUBAGENT_NAME])
    require(
        limit == SUBAGENT_STEP_LIMIT,
        f"the {GENERAL_PURPOSE_SUBAGENT_NAME!r} subagent runs to {limit} steps, not "
        f"{SUBAGENT_STEP_LIMIT}; one `task` dispatch would outlive every bound run_turn sends",
    )


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

    # Computed once and reused by both builds. Rebuilding the kwargs would call
    # `least_privilege_filesystem` a second time and put the two graphs on two
    # different `StateBackend`s — the same split F21 was about, arrived at from
    # the other direction.
    kwargs = _agent_kwargs(agent_config, model)
    agent = create_deep_agent(model=model, **kwargs)
    require(agent is not None, "create_deep_agent returned None")

    # The second build is what bounds `task`. deepagents' general-purpose
    # subagent only exists once it has compiled one, and the only lever over its
    # step limit is the config bound to the graph itself — so the graph has to
    # be built before it can be handed back as a spec. A caller who supplied
    # their own spec of that name owns it, limit included.
    if not _supplies_general_purpose_subagent(agent_config):
        bounded = _bounded_general_purpose_subagent(agent)
        agent = create_deep_agent(
            model=model, **{**kwargs, "subagents": [bounded, *agent_config.subagents]}
        )
        require(agent is not None, "create_deep_agent returned None on the bounded rebuild")
        _require_subagents_bounded(agent)

    # A caller-supplied FilesystemMiddleware owns the allowlist from then on, so
    # there is no allowlist of ours left to assert.
    if not _replaces_filesystem_middleware(agent_config):
        _require_shell_withheld(agent)

    # A key the graph returns and nobody reviewed. `TurnResult.state` will carry
    # it either way; this is what makes its arrival a decision.
    unknown = compiled_output_keys(agent) - KNOWN_OUTPUT_STATE_KEYS
    require(
        not unknown,
        f"the compiled agent declares output state keys nobody has reviewed: {sorted(unknown)}. "
        f"TurnResult.state carries them, but decide whether each deserves a property of its own "
        f"the way `files` has one, then add it to KNOWN_OUTPUT_STATE_KEYS.",
    )

    # The floor under `RunBounds.step_limit`, and the other half of F24.
    # `create_agent` binds `recursion_limit: 9999` onto every graph it compiles,
    # the parent included — so a caller who invokes this graph without going
    # through `run_turn` inherited 9999, not langchain-core's 25. `run_turn`
    # itself is unaffected either way: it sends the limit on the invocation, and
    # a top-level invoke config beats the graph's own bound config. This binds
    # the fallback for the caller who does not take that path.
    bounded_agent = agent.with_config({"recursion_limit": PARENT_STEP_LIMIT})
    # Postcondition: `with_config` returns a copy, so asserting on `agent` would
    # pass while the object actually handed back kept the library's limit.
    require(
        bound_step_limit(bounded_agent) == PARENT_STEP_LIMIT,
        f"binding the parent step limit did not take: wanted {PARENT_STEP_LIMIT}, "
        f"graph carries {bound_step_limit(bounded_agent)}",
    )
    return bounded_agent
