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
from deepagents import FilesystemMiddleware, FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from my_agent.agent import KNOWN_CREATE_DEEP_AGENT_PARAMS, AgentConfig, _agent_kwargs, build_agent
from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    SUBAGENT_TASK_TOOL_NAME,
    compiled_tool_names,
    subagent_graphs,
)
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed

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


# --------------------------------------------------------------------------
# _agent_kwargs — the assembly build_agent does not pass straight through
# --------------------------------------------------------------------------


def test_agent_kwargs_sends_the_same_rules_to_both_places_they_are_needed(
    deny_secrets: FilesystemPermission,
) -> None:
    """`permissions` has to reach `create_deep_agent` AND the FilesystemMiddleware
    that replaces its default. Disagreement between the two is the silent failure
    this assembly exists to prevent."""
    kwargs = _agent_kwargs(AgentConfig(permissions=[deny_secrets]))
    installed = kwargs["middleware"][0]

    assert kwargs["permissions"] == [deny_secrets]
    assert isinstance(installed, FilesystemMiddleware)
    assert installed._permissions == kwargs["permissions"]


def test_agent_kwargs_puts_the_least_privilege_middleware_before_the_callers() -> None:
    """Ours has to be in the list at all, and the caller's additions follow it."""
    extra = TodoListMiddleware()
    kwargs = _agent_kwargs(AgentConfig(middleware=[extra]))

    assert isinstance(kwargs["middleware"][0], FilesystemMiddleware)
    assert kwargs["middleware"][1:] == [extra]


def test_agent_kwargs_passes_no_rules_as_none_rather_than_an_empty_list() -> None:
    """Empty permissions means "no rules", not "deny everything" — deepagents
    reads `None` as the former."""
    kwargs = _agent_kwargs(AgentConfig())

    assert kwargs["permissions"] is None


def test_agent_kwargs_leaves_every_other_field_untouched() -> None:
    """`middleware` and `permissions` are the only two fields build_agent
    rewrites; the rest is why AgentConfig is a parameter object at all."""
    config = AgentConfig(name="assembled", system_prompt="Be brief.")
    kwargs = _agent_kwargs(config)

    passthrough = {k: v for k, v in kwargs.items() if k not in {"middleware", "permissions"}}
    assert passthrough == {
        k: v for k, v in config.as_kwargs().items() if k not in {"middleware", "permissions"}
    }


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

    kwargs = _agent_kwargs(ConfigWithBackend(backend=backend))

    assert kwargs["backend"] is backend
    assert kwargs["middleware"][0].backend is backend


def test_agent_kwargs_leaves_the_middleware_on_the_state_backend_by_default() -> None:
    """No backend configured means the safest one, chosen rather than inherited."""
    kwargs = _agent_kwargs(AgentConfig())

    assert isinstance(kwargs["middleware"][0].backend, StateBackend)


def test_agent_kwargs_refuses_the_rule_dropping_combination_without_a_model(
    deny_secrets: FilesystemPermission,
) -> None:
    """The same refusal build_agent surfaces, reachable without building a model
    or compiling a graph."""
    with pytest.raises(CheckFailed, match="silently drop"):
        _agent_kwargs(
            AgentConfig(
                middleware=[FilesystemMiddleware(tools=list(DEFAULT_FILESYSTEM_TOOLS))],
                permissions=[deny_secrets],
            )
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

    granted = {
        name: compiled_tool_names(graph) for name, graph in subagent_graphs(bare).items()
    }

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

    with pytest.raises(CheckFailed, match=SHELL_TOOL_NAME):
        build_agent(build_model(ModelConfig(api_key=valid_secret)))


def test_build_agent_fails_when_the_subagent_reader_finds_nothing(
    valid_secret: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty mapping from the reader means the `task` tool vanished. That is
    a structural change worth failing on, not a licence to skip the check."""
    monkeypatch.setattr("my_agent.agent.subagent_graphs", lambda _agent: {})

    with pytest.raises(CheckFailed, match="subagent"):
        build_agent(build_model(ModelConfig(api_key=valid_secret)))


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

    assert set(AgentConfig().as_kwargs()) <= accepted


def test_agent_config_does_not_carry_the_factory_injected_parameter() -> None:
    """`model` is supplied by build_agent. A field of that name would collide on
    splat."""
    assert "model" not in AgentConfig().as_kwargs()


def test_adding_a_setting_needs_no_factory_change() -> None:
    """The point of the parameter object: a new deepagents setting reaches the
    factory as a field, with no edit to build_agent's signature or body."""
    extended = AgentConfig(name="extended")
    kwargs = {**extended.as_kwargs(), "skills": ["./skills/"], "memory": ["./AGENTS.md"]}

    assert set(kwargs) <= set(inspect.signature(create_deep_agent).parameters)


def test_create_deep_agent_parameters_are_pinned() -> None:
    """A new upstream parameter must fail the import, not be inherited.

    `check_config_contract` only asserts our fields are real parameters; it
    cannot see a new one appear. That is the door the shell `execute` tool came
    through (F4), so the parameter set is pinned the way the tool list is."""
    assert frozenset(
        inspect.signature(create_deep_agent).parameters
    ) == KNOWN_CREATE_DEEP_AGENT_PARAMS


def test_agent_config_covers_only_what_is_needed() -> None:
    """Deliberate YAGNI, pinned so the gap is a decision rather than an accident:
    the rest are reachable by adding a field, and `build_agent` does not change."""
    configured = {f.name for f in dataclasses.fields(AgentConfig)}
    assert configured == {"name", "system_prompt", "tools", "middleware", "permissions"}
    assert configured < KNOWN_CREATE_DEEP_AGENT_PARAMS
