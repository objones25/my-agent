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

import inspect
from typing import Any

import pytest
from deepagents import FilesystemMiddleware, FilesystemPermission, create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from my_agent.agent import AgentConfig, _agent_kwargs, build_agent
from my_agent.capabilities import DEFAULT_FILESYSTEM_TOOLS, SHELL_TOOL_NAME, compiled_tool_names
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
