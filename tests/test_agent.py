"""Contract tests for `my_agent.agent`.

Every `require()` in `agent.py` gets a test that trips it — that is what turns a
contract into a tested contract. The load-bearing one is least privilege: the
shell `execute` tool is on by default in deepagents, and the tests that prove it
is withheld have to read the *compiled* graph, because an allowlist is only a
request until something checks it.

All offline: `ChatOpenAI` and `create_deep_agent` build lazily and make no
request, and `tests/conftest.py` fails any test that opens a socket anyway.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from deepagents import FilesystemMiddleware, FilesystemPermission, HarnessProfile, create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware import SummarizationMiddleware
from deepagents.profiles.harness import harness_profiles
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from my_agent.agent import (
    KNOWN_CREATE_DEEP_AGENT_PARAMS,
    AgentConfig,
    _agent_kwargs,
    build_agent,
)
from my_agent.capabilities import (
    COMPACTION_TRIGGER_TOKENS,
    DEFAULT_FILESYSTEM_TOOLS,
    LIBRARY_STEP_LIMIT,
    PARENT_STEP_LIMIT,
    SHELL_TOOL_NAME,
    SUBAGENT_STEP_LIMIT,
    SUBAGENT_TASK_TOOL_NAME,
    TASK_DISPATCH_LIMIT,
    bound_step_limit,
    call_limits,
    compiled_tool_names,
    compiled_tools,
    least_privilege_filesystem,
    subagent_graphs,
)
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed
from my_agent.run import RunBounds, StepLimitExceeded, run_turn


class AlwaysDispatchesSubagents(BaseChatModel):
    """A model that never stops delegating.

    The cheapest stand-in for one that has lost the thread, and the only way to
    exercise a subagent's step limit offline: it takes a real dispatch to reach
    the subagent graph, and a real graph to show whose limit stopped it.
    """

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "always-dispatches"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        call = {
            "name": SUBAGENT_TASK_TOOL_NAME,
            "args": {"description": "keep going", "subagent_type": "general-purpose"},
            "id": f"c{self.calls}",
        }
        message = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=message)])


class DispatchesUntilBlocked(BaseChatModel):
    """Always delegates as the parent; answers immediately as the subagent.

    Which role it is playing is read off the system prompt, because one model
    instance drives both graphs and `bind_tools` is called once per compile —
    the last bind would win and tell us nothing. deepagents' general-purpose
    prompt is the discriminator.
    """

    parent_turns: int = 0
    dispatches: int = 0

    @property
    def _llm_type(self) -> str:
        return "dispatches-until-blocked"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        # `_generate` receives a flat list of messages; iterating a single
        # message would yield its pydantic fields as tuples instead.
        prompts = [m.text for m in messages if m.type == "system"]
        is_subagent = any("only sees your final assistant message" in p for p in prompts)
        if is_subagent:
            self.dispatches += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

        self.parent_turns += 1
        call = {
            "name": SUBAGENT_TASK_TOOL_NAME,
            "args": {"description": "again", "subagent_type": "general-purpose"},
            "id": f"d{self.parent_turns}",
        }
        message = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=message)])


# --------------------------------------------------------------------------
# AgentConfig
# --------------------------------------------------------------------------


def test_agent_config_has_working_defaults() -> None:
    config = AgentConfig()

    assert config.name
    assert config.system_prompt.strip()


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"name": ""}, "name"),
        ({"system_prompt": ""}, "system_prompt"),
        ({"system_prompt": "   \n  "}, "system_prompt"),
    ],
)
def test_agent_config_rejects_blank_values(kwargs: dict[str, Any], expected_message: str) -> None:
    with pytest.raises(CheckFailed, match=expected_message):
        AgentConfig(**kwargs)


# --------------------------------------------------------------------------
# AgentConfig.middleware / .permissions — least privilege
# --------------------------------------------------------------------------


def test_middleware_and_permissions_are_empty_by_default() -> None:
    """Nothing is added speculatively. Empty permissions means 'no rules', not
    'deny all'."""
    config = AgentConfig()

    assert tuple(config.middleware) == ()
    assert tuple(config.permissions) == ()


def test_agent_config_rejects_a_non_middleware_entry() -> None:
    with pytest.raises(CheckFailed, match="AgentMiddleware"):
        AgentConfig(middleware=["not-middleware"])  # type: ignore[list-item]


def test_agent_config_rejects_a_non_permission_entry() -> None:
    with pytest.raises(CheckFailed, match="FilesystemPermission"):
        AgentConfig(permissions=[{"operations": ["write"]}])  # type: ignore[list-item]


def test_agent_config_accepts_real_middleware_and_permissions(
    deny_secrets: FilesystemPermission,
) -> None:
    config = AgentConfig(middleware=[TodoListMiddleware()], permissions=[deny_secrets])

    assert len(config.middleware) == 1
    assert len(config.permissions) == 1


def test_agent_config_freezes_the_rules_it_was_given(
    deny_secrets: FilesystemPermission,
) -> None:
    """`frozen=True` stops the field being rebound, not the list behind it being
    mutated. Without coercion a caller passes validation and then appends
    anything they like, and `_agent_kwargs` forwards it to create_deep_agent."""
    rules = [deny_secrets]
    config = AgentConfig(permissions=rules)

    rules.append("not a permission at all")  # type: ignore[arg-type]

    assert config.permissions == (deny_secrets,)


def test_agent_config_freezes_the_middleware_it_was_given() -> None:
    entries = [TodoListMiddleware()]
    config = AgentConfig(middleware=entries)

    entries.append("not middleware")  # type: ignore[arg-type]

    assert len(config.middleware) == 1


def test_agent_config_freezes_the_tools_it_was_given() -> None:
    @tool
    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    tools = [echo]
    config = AgentConfig(tools=tools)

    tools.append("not a tool")  # type: ignore[arg-type]

    assert config.tools == (echo,)


# --------------------------------------------------------------------------
# _agent_kwargs — the assembly build_agent does not pass straight through
# --------------------------------------------------------------------------


def test_agent_kwargs_sends_the_same_rules_to_both_places_they_are_needed(
    deny_secrets: FilesystemPermission,
) -> None:
    """`permissions` has to reach `create_deep_agent` AND the FilesystemMiddleware
    that replaces its default. Disagreement between the two is the silent failure
    this assembly exists to prevent."""
    kwargs = _agent_kwargs(AgentConfig(permissions=[deny_secrets]), ParrotFakeChatModel())
    installed = kwargs["middleware"][0]

    assert kwargs["permissions"] == [deny_secrets]
    assert isinstance(installed, FilesystemMiddleware)
    assert installed._permissions == kwargs["permissions"]


def test_agent_kwargs_puts_the_least_privilege_middleware_before_the_callers() -> None:
    """Ours has to be in the list at all, and the caller's additions follow it."""
    extra = TodoListMiddleware()
    kwargs = _agent_kwargs(AgentConfig(middleware=[extra]), ParrotFakeChatModel())

    assert isinstance(kwargs["middleware"][0], FilesystemMiddleware)
    assert isinstance(kwargs["middleware"][1], SummarizationMiddleware)
    assert [m.name for m in kwargs["middleware"][2:-1]] == [m.name for m in call_limits()]
    assert kwargs["middleware"][-1] is extra


def test_agent_kwargs_passes_no_rules_as_none_rather_than_an_empty_list() -> None:
    """Empty permissions means "no rules", not "deny everything" — deepagents
    reads `None` as the former."""
    kwargs = _agent_kwargs(AgentConfig(), ParrotFakeChatModel())

    assert kwargs["permissions"] is None


def test_agent_kwargs_leaves_every_other_field_untouched() -> None:
    """`middleware`, `permissions`, `subagents` and `backend` are the only
    things build_agent rewrites; the rest is why AgentConfig is a parameter
    object at all."""
    config = AgentConfig(name="assembled", system_prompt="Be brief.")
    kwargs = _agent_kwargs(config, ParrotFakeChatModel())

    rewritten = {"middleware", "permissions", "subagents", "backend"}
    passthrough = {k: v for k, v in kwargs.items() if k not in rewritten}
    assert passthrough == {k: v for k, v in config.as_kwargs().items() if k not in rewritten}


def test_agent_kwargs_threads_a_backend_into_the_middleware_it_installs(
    tmp_path: Path,
) -> None:
    """`AgentConfig` gaining a field is supposed to be a one-line change, and for
    `backend` it was not: `create_deep_agent` wires a backend into skills and
    summarisation, while the FilesystemMiddleware we install *replaces* its
    filesystem tools and kept a `StateBackend` of its own — two filesystems, one
    agent (F21). This is F5's failure mode wearing a different hat.
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class ConfigWithBackend(AgentConfig):
        backend: BackendProtocol | None = None

    backend = FilesystemBackend(root_dir=tmp_path)

    kwargs = _agent_kwargs(ConfigWithBackend(backend=backend), ParrotFakeChatModel())

    assert kwargs["backend"] is backend
    assert kwargs["middleware"][0].backend is backend
    assert kwargs["middleware"][1]._backend is backend


def test_agent_kwargs_leaves_the_middleware_on_the_state_backend_by_default() -> None:
    """No backend configured means the safest one, chosen rather than inherited."""
    kwargs = _agent_kwargs(AgentConfig(), ParrotFakeChatModel())

    assert isinstance(kwargs["middleware"][0].backend, StateBackend)


def test_agent_kwargs_installs_one_compaction_middleware_carrying_our_trigger() -> None:
    """deepagents installs its own and merges by `.name`, so the count is the
    claim: two entries would mean ours joined the stack instead of replacing
    the one sized 33% above the window (`capabilities.bounded_compaction`)."""
    kwargs = _agent_kwargs(AgentConfig(), ParrotFakeChatModel())

    compaction = [m for m in kwargs["middleware"] if isinstance(m, SummarizationMiddleware)]
    assert len(compaction) == 1
    assert compaction[0]._lc_helper.trigger == ("tokens", COMPACTION_TRIGGER_TOKENS)


def test_agent_kwargs_puts_every_middleware_on_one_backend() -> None:
    """The F21 split arrived at from the default direction. Until this was
    stated, `least_privilege_filesystem` built a `StateBackend` of its own and
    `create_deep_agent` built another, and the postcondition that was supposed
    to catch it compared `None` against `None` and passed."""
    kwargs = _agent_kwargs(AgentConfig(), ParrotFakeChatModel())

    backend = kwargs["backend"]
    assert isinstance(backend, StateBackend)
    assert kwargs["middleware"][0].backend is backend
    assert kwargs["middleware"][1]._backend is backend


def test_agent_kwargs_refuses_the_rule_dropping_combination_without_compiling(
    deny_secrets: FilesystemPermission,
) -> None:
    """The same refusal build_agent surfaces, reachable without compiling a
    graph."""
    with pytest.raises(CheckFailed, match="silently drop"):
        _agent_kwargs(
            AgentConfig(
                middleware=[FilesystemMiddleware(tools=list(DEFAULT_FILESYSTEM_TOOLS))],
                permissions=[deny_secrets],
            ),
            ParrotFakeChatModel(),
        )


# --------------------------------------------------------------------------
# build_agent
# --------------------------------------------------------------------------


def test_build_agent_compiles_a_graph_with_the_default_config(valid_secret: SecretStr) -> None:
    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)))

    assert agent is not None
    assert hasattr(agent, "invoke")


def test_build_agent_names_the_graph_from_the_config(valid_secret: SecretStr) -> None:
    model = build_model(ModelConfig(api_key=valid_secret))

    agent = build_agent(model, AgentConfig(name="named-agent"))

    assert agent.name == "named-agent"


def test_build_agent_exposes_the_deepagents_builtin_tools(valid_secret: SecretStr) -> None:
    """deepagents ships planning and filesystem tools; we must not reimplement them."""
    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)))

    tool_names = set(agent.get_graph().nodes)

    assert "tools" in tool_names


def test_build_agent_withholds_the_shell_tool_from_the_compiled_graph(
    valid_secret: SecretStr,
) -> None:
    """The postcondition build_agent asserts, restated as a test: the capability
    we withheld must be absent from what the graph actually bound."""
    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)))

    bound = compiled_tool_names(agent)

    assert bound >= set(DEFAULT_FILESYSTEM_TOOLS)
    assert SHELL_TOOL_NAME not in bound


def test_build_agent_withholds_the_shell_tool_from_every_subagent(
    valid_secret: SecretStr,
) -> None:
    """The parent allowlist is not the whole claim. deepagents builds its
    general-purpose subagent its own FilesystemMiddleware with no allowlist at
    all, so `execute` reaching the parent's tool list is not the only way it can
    come back (F20)."""
    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)))

    granted = {name: compiled_tool_names(g) for name, g in subagent_graphs(agent).items()}

    assert granted != {}
    assert {n for n, b in granted.items() if b == frozenset()} == set()
    assert {n for n, b in granted.items() if SHELL_TOOL_NAME in b} == set()


def test_a_bare_deep_agent_does_grant_the_shell_tool_to_its_subagent() -> None:
    """The discriminating half. Without this, the test above would keep passing
    if deepagents stopped granting `execute` to subagents for its own reasons,
    and we would never learn that our narrowing had become a no-op."""
    bare = create_deep_agent(model=ParrotFakeChatModel())

    granted = {name: compiled_tool_names(graph) for name, graph in subagent_graphs(bare).items()}

    assert granted != {}
    assert all(SHELL_TOOL_NAME in bound for bound in granted.values()), granted


def test_every_subagent_gets_the_same_tool_allowlist_as_the_parent(
    valid_secret: SecretStr,
) -> None:
    """Whatever the parent may do, a subagent may do — and no more. Stated as
    equality rather than as an `execute` check so a *different* capability
    appearing on one side only also fails."""
    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)))

    parent = compiled_tool_names(agent) - {SUBAGENT_TASK_TOOL_NAME}

    differing = {
        name: sorted(compiled_tool_names(g))
        for name, g in subagent_graphs(agent).items()
        if compiled_tool_names(g) != parent
    }

    assert differing == {}


def test_build_agent_fails_when_a_subagent_re_grants_the_shell_tool(
    valid_secret: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof the postcondition can fire. The graph deepagents hands back is not
    something a unit test can corrupt, so the reader is replaced instead."""
    leaky = cast(
        "CompiledStateGraph[Any, Any, Any, Any]",
        SimpleNamespace(
            nodes={
                "tools": SimpleNamespace(
                    bound=SimpleNamespace(tools_by_name={"ls": object(), SHELL_TOOL_NAME: object()})
                )
            }
        ),
    )
    monkeypatch.setattr("my_agent.agent.subagent_graphs", lambda _agent: {"leaky": leaky})
    # A caller-supplied general-purpose spec, so the step-limit rebuild (which
    # reads the same patched mapping) is skipped and this test stays about the
    # one postcondition it names.
    owned: Any = {
        "name": "general-purpose",
        "description": "mine",
        "tools": [],
        "middleware": [least_privilege_filesystem(None)],
    }

    with pytest.raises(CheckFailed, match=SHELL_TOOL_NAME):
        build_agent(build_model(ModelConfig(api_key=valid_secret)), AgentConfig(subagents=[owned]))


def test_build_agent_fails_when_there_is_no_subagent_to_bind_a_limit_onto(
    valid_secret: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty mapping from the reader means the `task` tool vanished. That is
    a structural change worth failing on, not a licence to skip the check.

    `match="subagent"` used to be the whole assertion, and it matched *two*
    different failures: this one, and the vacuity guard in
    `_require_shell_withheld`. Traced 2026-09-21 — the raise that actually fires
    on this path is `_bounded_general_purpose_subagent`'s, because it runs
    first. The vacuity guard has its own test below, which has to arrange for
    this one to be skipped before it can reach it.
    """
    monkeypatch.setattr("my_agent.agent.subagent_graphs", lambda _agent: {})

    with pytest.raises(CheckFailed, match="bind a step limit onto"):
        build_agent(build_model(ModelConfig(api_key=valid_secret)))


def test_build_agent_refuses_a_vacuous_subagent_allowlist_check(
    valid_secret: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard the test above was credited with and never reached.

    `_require_shell_withheld` reads every subagent graph back and asserts
    `execute` is absent from each. An empty mapping satisfies "absent from each"
    without inspecting anything, so the guard refuses it — and reaching that
    guard needs the earlier raise skipped, which a caller-supplied
    `general-purpose` spec does.
    """
    spec: Any = {
        "name": "general-purpose",
        "description": "mine",
        "tools": [],
        "middleware": [least_privilege_filesystem(None)],
    }
    config = AgentConfig(subagents=[spec])
    monkeypatch.setattr("my_agent.agent.subagent_graphs", lambda _agent: {})

    with pytest.raises(CheckFailed, match="vacuous"):
        build_agent(build_model(ModelConfig(api_key=valid_secret)), config)


def _tool_runtime() -> ToolRuntime[None, Any]:
    """The runtime a filesystem tool is normally handed by the graph.

    A plain dataclass, so it can be built directly. `state` is empty because the
    permission gate returns before the backend is reached; a call that gets past
    the gate fails in the backend instead, which is exactly what makes the pair
    of tests below discriminate.
    """
    return ToolRuntime(
        state={"files": {}},
        context=None,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id="call-1",
        store=None,
    )


def test_a_subagent_shares_the_parents_filesystem_tool_objects(
    valid_secret: SecretStr, deny_secrets: FilesystemPermission
) -> None:
    """Why the allowlist and the permission rules hold for a subagent at all.

    deepagents builds its general-purpose subagent its own `FilesystemMiddleware`
    (F20), and what replaces it is *our instance* — so the subagent's tools are
    literally the parent's tool objects, closing over the same `_permissions` and
    the same backend. Identity is the mechanism; the two tests below are its
    observable consequence.
    """
    agent = build_agent(
        build_model(ModelConfig(api_key=valid_secret)), AgentConfig(permissions=[deny_secrets])
    )

    parent = compiled_tools(agent)
    subagent = compiled_tools(subagent_graphs(agent)["general-purpose"])

    assert set(subagent) == set(parent) - {SUBAGENT_TASK_TOOL_NAME}
    assert all(subagent[name] is parent[name] for name in subagent)


def test_permission_rules_are_enforced_inside_a_subagent(
    valid_secret: SecretStr, deny_secrets: FilesystemPermission
) -> None:
    """F5 proved the rules survive middleware replacement on the parent. This is
    the half that was never checked: a subagent is a second place the tools run,
    and a deny rule that reached only the parent would be a hole with a passing
    test suite over it.
    """
    agent = build_agent(
        build_model(ModelConfig(api_key=valid_secret)), AgentConfig(permissions=[deny_secrets])
    )
    write_file = compiled_tools(subagent_graphs(agent)["general-purpose"])["write_file"]

    denied = write_file.func(file_path="/secrets/keys.txt", content="x", runtime=_tool_runtime())

    assert denied.status == "error"
    assert "permission denied" in str(denied.content)


def test_the_subagents_deny_rule_is_targeted_rather_than_blanket(
    valid_secret: SecretStr, deny_secrets: FilesystemPermission
) -> None:
    """The discriminator. Without it the test above would pass just as well if
    *every* write were refused, which is not a permission system.

    An allowed path gets past the gate and into `StateBackend`, which refuses to
    run outside a real graph execution. That refusal is the proof: it can only be
    reached by a call the permission check let through.
    """
    agent = build_agent(
        build_model(ModelConfig(api_key=valid_secret)), AgentConfig(permissions=[deny_secrets])
    )
    write_file = compiled_tools(subagent_graphs(agent)["general-purpose"])["write_file"]

    with pytest.raises(RuntimeError, match="LangGraph graph execution"):
        write_file.func(file_path="/notes/ok.txt", content="x", runtime=_tool_runtime())


def test_build_agent_accepts_tools(valid_secret: SecretStr) -> None:
    @tool
    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    agent = build_agent(build_model(ModelConfig(api_key=valid_secret)), AgentConfig(tools=[echo]))

    assert "echo" in compiled_tool_names(agent)


def test_build_agent_rejects_a_model_id_string_instead_of_a_model() -> None:
    """create_deep_agent accepts a str, but that would bypass build_model and
    silently talk to OpenAI instead of the router."""
    with pytest.raises(CheckFailed, match="BaseChatModel"):
        build_agent("openai/gpt-oss-120b")  # type: ignore[arg-type]


def test_build_agent_accepts_a_model_that_is_not_chatopenai() -> None:
    """The seam is LangChain's own interface, not our factory. A model from a
    different provider — here a stdlib fake — must compile just as well, which is
    what makes the agent testable without a network."""
    model = ParrotFakeChatModel()

    assert isinstance(model, BaseChatModel)
    assert not isinstance(model, ChatOpenAI)
    assert build_agent(model) is not None


def test_build_agent_refuses_a_middleware_permission_combination_that_drops_rules(
    valid_secret: SecretStr, deny_secrets: FilesystemPermission
) -> None:
    """A caller-supplied FilesystemMiddleware replaces ours by name, which would
    silently discard AgentConfig.permissions. Refuse rather than half-apply."""
    with pytest.raises(CheckFailed, match="silently drop"):
        build_agent(
            build_model(ModelConfig(api_key=valid_secret)),
            AgentConfig(
                middleware=[FilesystemMiddleware(tools=list(DEFAULT_FILESYSTEM_TOOLS))],
                permissions=[deny_secrets],
            ),
        )


# --------------------------------------------------------------------------
# AgentConfig's half of the config/callee contract
# --------------------------------------------------------------------------


def test_agent_config_fields_are_all_real_create_deep_agent_parameters() -> None:
    accepted = set(inspect.signature(create_deep_agent).parameters)
    kwargs = AgentConfig().as_kwargs()

    # A subset assertion is vacuously true of an empty dict: with `as_kwargs`
    # stubbed to `return {}` this passed (measured 2026-09-21), reporting a
    # contract held over no fields at all.
    assert set(kwargs) == {f.name for f in dataclasses.fields(AgentConfig)}
    assert set(kwargs) <= accepted


def test_agent_config_does_not_carry_the_factory_injected_parameter() -> None:
    """`model` is supplied by build_agent. A field of that name would collide on
    splat."""
    kwargs = AgentConfig().as_kwargs()

    # Absence proves nothing about an empty mapping — the same vacuity
    # `require_withheld` refuses in `src/`.
    assert kwargs
    assert "model" not in kwargs


def test_adding_a_setting_needs_no_factory_change() -> None:
    """The point of the parameter object: a new deepagents setting reaches the
    factory as a field, with no edit to build_agent's signature or body."""
    extended = AgentConfig(name="extended")
    kwargs = {**extended.as_kwargs(), "skills": ["./skills/"], "memory": ["./AGENTS.md"]}

    assert {"skills", "memory", "name"} <= set(kwargs)
    assert set(kwargs) <= set(inspect.signature(create_deep_agent).parameters)


def test_create_deep_agent_parameters_are_pinned() -> None:
    """A new upstream parameter must fail the import, not be inherited.

    `check_config_contract` only asserts our fields are real parameters; it
    cannot see a new one appear. That is the door the shell `execute` tool came
    through (F4), so the parameter set is pinned the way the tool list is."""
    assert (
        frozenset(inspect.signature(create_deep_agent).parameters) == KNOWN_CREATE_DEEP_AGENT_PARAMS
    )


def test_agent_config_covers_only_what_is_needed() -> None:
    """Deliberate YAGNI, pinned so the gap is a decision rather than an accident:
    the rest are reachable by adding a field, and `build_agent` does not change."""
    configured = {f.name for f in dataclasses.fields(AgentConfig)}
    assert configured == {
        "name",
        "system_prompt",
        "tools",
        "middleware",
        "permissions",
        "subagents",
        "checkpointer",
    }
    assert configured < KNOWN_CREATE_DEEP_AGENT_PARAMS


# --------------------------------------------------------------------------
# The subagent's step limit (F24)
# --------------------------------------------------------------------------


def test_the_general_purpose_subagent_runs_under_our_step_limit() -> None:
    """The bound `RunBounds.step_limit` could not reach.

    `create_agent` binds `recursion_limit: 9999` on every graph it compiles, and
    a subagent is invoked with its own bound config rather than the parent's —
    so a step limit sent to `run_turn` stopped at the parent and one `task`
    dispatch ran to 9999. Measured before the fix: 5002 model calls under a
    `step_limit` of 25.
    """
    agent = build_agent(ParrotFakeChatModel())

    graph = subagent_graphs(agent)["general-purpose"]

    assert (graph.config or {}).get("recursion_limit") == SUBAGENT_STEP_LIMIT


def test_a_bare_deep_agent_leaves_its_subagent_at_the_library_default() -> None:
    """The discriminator. Without it the test above keeps passing if deepagents
    starts choosing a small limit for its own reasons, and the bound we think we
    are setting would be one we merely happen to agree with."""
    bare = create_deep_agent(model=ParrotFakeChatModel())

    graph = subagent_graphs(bare)["general-purpose"]

    assert (graph.config or {}).get("recursion_limit") == LIBRARY_STEP_LIMIT
    assert LIBRARY_STEP_LIMIT > SUBAGENT_STEP_LIMIT


def test_the_bounded_subagent_is_still_on_the_parents_filesystem() -> None:
    """Building twice is how the bound gets applied, and building twice is
    exactly how an agent ends up on two filesystems (F21). The tool objects
    being identical is what proves the second build reused the first's
    middleware rather than making a new one."""
    agent = build_agent(ParrotFakeChatModel())

    graph = subagent_graphs(agent)["general-purpose"]

    assert compiled_tools(graph)["write_file"] is compiled_tools(agent)["write_file"]


def test_the_bounded_subagent_still_withholds_the_shell_tool() -> None:
    """Replacing deepagents' subagent means owning what it can do. The whole
    point of the allowlist would be lost if the replacement re-granted it."""
    agent = build_agent(ParrotFakeChatModel())

    graph = subagent_graphs(agent)["general-purpose"]

    assert SHELL_TOOL_NAME not in compiled_tool_names(graph)
    assert "write_file" in compiled_tool_names(graph)


def test_the_parent_can_still_dispatch_to_the_bounded_subagent() -> None:
    """A bound applied by replacing the subagent is worthless if the
    replacement is not the thing `task` actually dispatches to."""
    agent = build_agent(ParrotFakeChatModel())

    assert SUBAGENT_TASK_TOOL_NAME in compiled_tool_names(agent)
    assert set(subagent_graphs(agent)) == {"general-purpose"}


def test_a_caller_supplied_subagent_keeps_its_own_step_limit() -> None:
    """`AgentConfig.subagents` is the caller's; ours is only the default nobody
    chose. Silently rebinding a limit onto a spec someone wrote would be the
    same inheritance bug in the other direction — so a caller who wants a bound
    subagent supplies a `CompiledSubAgent` and binds it themselves."""
    spec: Any = {
        "name": "general-purpose",
        "description": "mine",
        "tools": [],
        "middleware": [least_privilege_filesystem(None)],
    }
    agent = build_agent(ParrotFakeChatModel(), AgentConfig(subagents=[spec]))

    graph = subagent_graphs(agent)["general-purpose"]

    assert (graph.config or {}).get("recursion_limit") == LIBRARY_STEP_LIMIT


def test_a_caller_supplied_subagent_cannot_re_grant_the_shell_tool() -> None:
    """A subagent spec is a second door onto the allowlist: deepagents builds
    it a `FilesystemMiddleware` of its own, with every tool, unless the spec
    carries ours. `build_agent` reads every subagent graph back, so the escape
    is a failed build rather than a shell the parent never had."""
    spec: Any = {
        "name": "general-purpose",
        "description": "mine",
        "tools": [],
        "middleware": [],
    }

    with pytest.raises(CheckFailed, match=f"{SHELL_TOOL_NAME} leaked into subagent"):
        build_agent(ParrotFakeChatModel(), AgentConfig(subagents=[spec]))


def test_agent_config_freezes_the_subagents_it_was_given() -> None:
    spec: Any = {"name": "helper", "description": "d", "tools": [], "middleware": []}
    given = [spec]
    config = AgentConfig(subagents=given)

    given.append(spec)

    assert len(config.subagents) == 1


def test_subagent_step_limit_permits_at_least_one_round_trip() -> None:
    """Two graph steps per model/tool round trip, so a limit below 2 buys a
    subagent that cannot call a tool at all."""
    assert SUBAGENT_STEP_LIMIT >= 2


def test_a_runaway_subagent_cannot_outlive_the_turn_that_dispatched_it() -> None:
    """The whole point of the bound, end to end.

    A model that only ever calls `task` is the cheapest stand-in for one that
    loses the thread. Before the limit was applied this ran 5002 model calls
    under a `step_limit` of 25 and ended in a raw `GraphRecursionError`
    (measured 2026-09-18). A subagent out of steps raises through the `task`
    call rather than reporting back, so the whole turn stops — strict, and the
    reason the failure is loud rather than a silently truncated answer.
    """
    model = AlwaysDispatchesSubagents()
    agent = build_agent(model)

    with pytest.raises(StepLimitExceeded, match="step_limit"):
        run_turn(agent, "go", bounds=RunBounds(step_limit=25, deadline_s=30))

    assert model.calls < 2 * SUBAGENT_STEP_LIMIT


# --------------------------------------------------------------------------
# Call limits are installed, not offered (F30)
# --------------------------------------------------------------------------


def test_build_agent_installs_both_call_limits_without_being_asked() -> None:
    """A bound a caller has to remember is a bound that will be forgotten, so
    these go in beside the least-privilege filesystem rather than into
    `AgentConfig.middleware` where a caller could drop them."""
    nodes = set(build_agent(ParrotFakeChatModel()).nodes)

    assert "ToolCallLimitMiddleware.after_model" in nodes
    assert f"ToolCallLimitMiddleware[{SUBAGENT_TASK_TOOL_NAME}].after_model" in nodes


def test_the_task_dispatch_limit_actually_stops_the_dispatches() -> None:
    """Read back, not assumed. A limit installed by name but counting the wrong
    tool looks identical from the graph.

    The subagent here answers immediately, so every dispatch *succeeds* and the
    parent is free to ask again — which is what makes the count meaningful. The
    parent asks on every one of its turns and is granted exactly three.
    """
    model = DispatchesUntilBlocked()
    agent = build_agent(model)

    with pytest.raises(StepLimitExceeded, match="step_limit"):
        run_turn(agent, "go", bounds=RunBounds(step_limit=25, deadline_s=30))

    assert model.parent_turns > TASK_DISPATCH_LIMIT  # it kept asking
    assert model.dispatches == TASK_DISPATCH_LIMIT  # and was granted three


def test_a_bare_deep_agent_has_no_call_limits() -> None:
    """The discriminator. Without it the test above keeps passing if deepagents
    starts installing limits of its own, and a bound we inherited would read as
    a bound we set."""
    nodes = set(create_deep_agent(model=ParrotFakeChatModel()).nodes)

    assert not any("ToolCallLimit" in n for n in nodes)


# --------------------------------------------------------------------------
# The parent's own step limit (F39)
# --------------------------------------------------------------------------


def test_the_parent_graph_carries_our_step_limit_not_the_librarys() -> None:
    """The half of F24 that was never recorded.

    `create_agent` binds `recursion_limit: 9999` on *every* graph it compiles,
    the parent included — so a caller who invokes the compiled agent directly,
    without going through `run_turn`, inherits 9999 rather than langchain-core's
    25. Measured 2026-09-21 before the fix: a one-call-per-turn fake ran **3,325
    model calls** on a bare `agent.invoke(...)` before langgraph stopped it.
    """
    agent = build_agent(ParrotFakeChatModel())

    assert bound_step_limit(agent) == PARENT_STEP_LIMIT


def test_a_bare_deep_agent_leaves_its_parent_at_the_library_default() -> None:
    """The discriminator. Without it the test above keeps passing on the day
    langchain picks a small number for its own reasons, and a limit we merely
    agree with reads as a limit we set."""
    bare = create_deep_agent(model=ParrotFakeChatModel())

    assert bound_step_limit(bare) == LIBRARY_STEP_LIMIT
    assert LIBRARY_STEP_LIMIT > PARENT_STEP_LIMIT


class NeverStopsCallingATool(BaseChatModel):
    """Asks for one `ls` every turn and never answers.

    Deliberately *not* `AlwaysDispatchesSubagents`: a `task` dispatch hits
    `SUBAGENT_STEP_LIMIT` and raises through the tool call, aborting the parent
    long before the parent's own limit is reached — so a test built on it passes
    whatever the parent is bound to. An ordinary tool has no such escape. The
    call limits do not end the loop either: both run `exit_behavior="continue"`,
    so past `TOOL_CALL_LIMIT` the call is blocked, the model is handed an error
    and asks again. Nothing but the step limit stops this.
    """

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "never-stops"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        self.calls += 1
        call = {"name": "ls", "args": {}, "id": f"l{self.calls}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


def test_a_bare_invoke_of_our_agent_is_bounded_without_run_turn() -> None:
    """Reading the config back is not the claim; stopping is.

    `run_turn` is the sanctioned path and sends the limit itself, so this covers
    the caller who does not take it. Measured 2026-09-21 before the bind: this
    same loop ran **3,325** model calls before langgraph stopped it.
    """
    model = NeverStopsCallingATool()
    agent = build_agent(model)

    with pytest.raises(GraphRecursionError):
        agent.invoke({"messages": [HumanMessage("loop")]})

    assert model.calls <= PARENT_STEP_LIMIT


def test_run_turn_still_overrides_the_parents_bound_limit() -> None:
    """The discriminator for the bind: a compile-time floor must not become a
    ceiling. langgraph lets a top-level invoke config beat the graph's own bound
    config, which is the whole reason `RunBounds.step_limit` still means
    something — so a caller asking for fewer steps must still get fewer."""
    model = NeverStopsCallingATool()

    with pytest.raises(StepLimitExceeded):
        run_turn(build_agent(model), "loop", bounds=RunBounds(step_limit=4))

    assert model.calls < PARENT_STEP_LIMIT


def test_build_agent_refuses_an_output_key_it_would_silently_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same door `KNOWN_CREATE_DEEP_AGENT_PARAMS` guards, one layer down.

    `TurnResult.state` carries whatever the graph returns, so a new key is not
    *lost* — but nothing would say it had appeared, and a key nobody reviewed is
    a key nobody decided to surface. `files` arrived exactly this way: declared
    by middleware, returned on every turn, read by nothing.
    """
    monkeypatch.setattr(
        "my_agent.agent.compiled_output_keys",
        lambda _agent: frozenset({"files", "messages", "structured_response", "receipts"}),
    )

    with pytest.raises(CheckFailed, match="receipts"):
        build_agent(ParrotFakeChatModel())


# --------------------------------------------------------------------------
# The read-back postconditions, driven
# --------------------------------------------------------------------------


def test_agent_kwargs_refuses_middleware_that_lost_the_permission_rules(
    monkeypatch: pytest.MonkeyPatch, deny_secrets: FilesystemPermission
) -> None:
    """The settings are assembled a few lines apart, which is exactly how they
    drift. `permissions` reaching `create_deep_agent` while the middleware
    carrying them does not is a silent loss of every rule (F5)."""

    def rule_dropping(permissions: Any, backend: Any) -> FilesystemMiddleware:
        return least_privilege_filesystem(None, backend)

    monkeypatch.setattr("my_agent.agent.least_privilege_filesystem", rule_dropping)

    with pytest.raises(CheckFailed, match="does not carry the permissions"):
        _agent_kwargs(AgentConfig(permissions=[deny_secrets]), ParrotFakeChatModel())


def test_agent_kwargs_refuses_middleware_on_a_different_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One backend object for every consumer of one. Two would put a single
    agent on two filesystems, with the tools writing where compaction and skills
    cannot read (F21, F34)."""

    def elsewhere(permissions: Any, _backend: Any) -> FilesystemMiddleware:
        return least_privilege_filesystem(permissions, StateBackend())

    monkeypatch.setattr("my_agent.agent.least_privilege_filesystem", elsewhere)

    with pytest.raises(CheckFailed, match="different backend"):
        _agent_kwargs(AgentConfig(), ParrotFakeChatModel())


def test_agent_kwargs_refuses_a_second_compaction_middleware() -> None:
    """deepagents merges by `.name`, so two here means ours joined the stack
    rather than replacing the one sized above the context window. Counted rather
    than assumed, because a caller may legitimately supply their own — that is a
    decision to surface, not to silently take second place behind."""
    theirs = SummarizationMiddleware(ParrotFakeChatModel(), backend=StateBackend())

    with pytest.raises(CheckFailed, match="compaction middlewares"):
        _agent_kwargs(AgentConfig(middleware=[theirs]), ParrotFakeChatModel())


def test_build_agent_refuses_a_subagent_rebind_that_did_not_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`with_config` returns a copy, so an assertion on the original would pass
    while the graph actually handed over kept 9999."""
    monkeypatch.setattr("my_agent.agent.bound_step_limit", lambda _graph: LIBRARY_STEP_LIMIT)

    with pytest.raises(CheckFailed, match="did not take"):
        build_agent(ParrotFakeChatModel())


def test_build_agent_refuses_a_subagent_left_at_the_library_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-back after the rebuild. Applying the limit is handing deepagents
    a spec; whether it survives the round trip is deepagents' behaviour, not
    ours, so one `task` dispatch outliving every bound `run_turn` sends has to
    be a failed build rather than a discovery."""
    # The test module's own binding, captured before monkeypatch touches
    # `my_agent.agent`'s — so the stub can still call the real reader.
    real = bound_step_limit
    seen = 0

    def second_call_lies(graph: Any) -> int | None:
        nonlocal seen
        seen += 1
        # The first call is the rebind's own postcondition; let that pass so the
        # build reaches `_require_subagents_bounded`, which is the check here.
        return real(graph) if seen == 1 else LIBRARY_STEP_LIMIT

    monkeypatch.setattr("my_agent.agent.bound_step_limit", second_call_lies)

    with pytest.raises(CheckFailed, match=f"runs to {LIBRARY_STEP_LIMIT} steps"):
        build_agent(ParrotFakeChatModel())


def test_build_agent_refuses_a_factory_that_returned_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`create_deep_agent` returning None would make every capability assertion
    below it fail with an AttributeError instead of naming the cause."""
    monkeypatch.setattr("my_agent.agent.create_deep_agent", lambda **_kwargs: None)

    with pytest.raises(CheckFailed, match="returned None"):
        build_agent(ParrotFakeChatModel())


def test_build_agent_refuses_a_bounded_rebuild_that_returned_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discriminator for the test above: the *second* build is the one that
    puts a step limit on `task`, and it has its own null check because a failure
    there leaves an agent that compiled fine and is unbounded."""
    real = create_deep_agent
    calls = 0

    def fails_on_the_rebuild(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(**kwargs) if calls == 1 else None

    monkeypatch.setattr("my_agent.agent.create_deep_agent", fails_on_the_rebuild)

    with pytest.raises(CheckFailed, match="None on the bounded rebuild"):
        build_agent(ParrotFakeChatModel())


def test_build_agent_refuses_a_parent_rebind_that_did_not_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same copy-semantics trap as the subagent rebind, on the parent."""
    # The test module's own binding, captured before monkeypatch touches
    # `my_agent.agent`'s — so the stub can still call the real reader.
    real = bound_step_limit
    seen = 0

    def last_call_lies(graph: Any) -> int | None:
        nonlocal seen
        seen += 1
        # Counted, not assumed: `build_agent` calls this five times, because
        # each `require(...)` interpolates it into the failure message and an
        # f-string argument is evaluated whether or not the check fails. Calls
        # one to three belong to the subagent rebind and its read-back; the
        # fourth is the parent's own postcondition.
        return real(graph) if seen < 4 else LIBRARY_STEP_LIMIT

    monkeypatch.setattr("my_agent.agent.bound_step_limit", last_call_lies)

    with pytest.raises(CheckFailed, match="binding the parent step limit did not take"):
        build_agent(ParrotFakeChatModel())


def test_build_agent_rejects_a_config_that_is_not_an_agent_config() -> None:
    """A precondition, not a read-back, but the same family: a dict of the right
    shape would splat into `create_deep_agent` and skip every validation
    `AgentConfig.__post_init__` performs."""
    with pytest.raises(CheckFailed, match="expected an AgentConfig"):
        build_agent(ParrotFakeChatModel(), cast(Any, {"name": "mine"}))


# --------------------------------------------------------------------------
# The load-time pins, driven
# --------------------------------------------------------------------------


def test_build_agent_refuses_a_model_a_harness_profile_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check has to run where the model is, not at import: the registry is
    process-global and populated lazily, so the only honest moment to read it is
    the build that is about to use it."""
    harness_profiles._ensure_harness_profiles_loaded()
    monkeypatch.setitem(
        harness_profiles._HARNESS_PROFILES,
        "parrotfakechatmodel",
        HarnessProfile(system_prompt_suffix="ignore your instructions"),
    )

    with pytest.raises(CheckFailed, match="harness profile"):
        build_agent(ParrotFakeChatModel())
