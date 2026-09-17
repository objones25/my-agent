"""Contract tests for the agent composition root.

Every `require()` in `agent.py` gets a test that trips it — that is what turns a
contract into a tested contract. All of these run offline: `ChatOpenAI` and
`create_deep_agent` build lazily and make no network call.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, get_args

import pytest
from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    FsToolName,
    create_deep_agent,
)
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from my_agent.agent import (
    API_KEY_ENV_VAR,
    DEFAULT_FILESYSTEM_TOOLS,
    DEFAULT_MODEL,
    HF_ROUTER_BASE_URL,
    MODEL_ENV_VAR,
    SHELL_TOOL_NAME,
    USE_RESPONSES_API,
    AgentConfig,
    ModelConfig,
    _least_privilege_filesystem,
    build_agent,
    build_model,
)
from my_agent.contracts import check_config_contract, pydantic_param_names
from my_agent.negative_space import CheckFailed

VALID_KEY = "hf_token_value"
VALID_SECRET = SecretStr(VALID_KEY)
DENY_SECRETS = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")


# --------------------------------------------------------------------------
# ModelConfig: positive space
# --------------------------------------------------------------------------


def test_model_config_defaults_target_the_hf_router() -> None:
    config = ModelConfig(api_key=VALID_SECRET)

    assert config.base_url == HF_ROUTER_BASE_URL
    assert config.model == DEFAULT_MODEL


def test_model_config_is_frozen() -> None:
    config = ModelConfig(api_key=VALID_SECRET)

    with pytest.raises(AttributeError):
        config.model = "other"  # type: ignore[misc]


def test_model_config_repr_does_not_leak_the_api_key() -> None:
    config = ModelConfig(api_key=SecretStr("super-secret-token"))

    assert "super-secret-token" not in repr(config)


# --------------------------------------------------------------------------
# ModelConfig: negative space — programmer errors, so CheckFailed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"api_key": SecretStr("")}, "api_key"),
        ({"api_key": VALID_SECRET, "model": ""}, "model"),
        ({"api_key": VALID_SECRET, "base_url": "router.huggingface.co/v1"}, "base_url"),
        ({"api_key": VALID_SECRET, "temperature": -0.1}, "temperature"),
        ({"api_key": VALID_SECRET, "temperature": 2.1}, "temperature"),
        ({"api_key": VALID_SECRET, "timeout": 0.0}, "timeout"),
        ({"api_key": VALID_SECRET, "max_retries": -1}, "max_retries"),
    ],
)
def test_model_config_rejects_impossible_values(
    kwargs: dict[str, Any], expected_message: str
) -> None:
    with pytest.raises(CheckFailed, match=expected_message):
        ModelConfig(**kwargs)


def test_model_config_rejects_api_key_with_surrounding_whitespace() -> None:
    """A token with a trailing newline is the classic .env bug: it produces a
    confusing 401 from the router rather than a clear local failure."""
    with pytest.raises(CheckFailed, match="whitespace"):
        ModelConfig(api_key=SecretStr(f"{VALID_KEY}\n"))


# --------------------------------------------------------------------------
# ModelConfig.from_env: operating errors, so ValueError (never CheckFailed)
# --------------------------------------------------------------------------


def test_from_env_reads_the_token_and_defaults_the_model() -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: VALID_KEY})

    assert config.api_key.get_secret_value() == VALID_KEY
    assert config.model == DEFAULT_MODEL


def test_from_env_honours_an_explicit_model_id() -> None:
    config = ModelConfig.from_env(
        {API_KEY_ENV_VAR: VALID_KEY, MODEL_ENV_VAR: "org/other:novita"}
    )

    assert config.model == "org/other:novita"


def test_from_env_strips_whitespace_so_a_trailing_newline_is_survivable() -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: f"  {VALID_KEY}\n"})

    assert config.api_key.get_secret_value() == VALID_KEY


@pytest.mark.parametrize("env", [{}, {API_KEY_ENV_VAR: ""}, {API_KEY_ENV_VAR: "   "}])
def test_from_env_raises_value_error_when_the_token_is_absent(env: dict[str, str]) -> None:
    """Missing configuration is an operating error, not a programmer error."""
    with pytest.raises(ValueError, match=API_KEY_ENV_VAR):
        ModelConfig.from_env(env)

    assert not issubclass(ValueError, CheckFailed)


def test_from_env_raises_value_error_on_a_blank_model_id() -> None:
    with pytest.raises(ValueError, match=MODEL_ENV_VAR):
        ModelConfig.from_env({API_KEY_ENV_VAR: VALID_KEY, MODEL_ENV_VAR: "   "})


# --------------------------------------------------------------------------
# build_model
# --------------------------------------------------------------------------


def test_build_model_applies_every_config_field() -> None:
    config = ModelConfig(
        api_key=VALID_SECRET,
        model="org/model:provider",
        temperature=0.7,
        timeout=30.0,
        max_retries=5,
    )

    model = build_model(config)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "org/model:provider"
    assert model.openai_api_base == HF_ROUTER_BASE_URL
    assert model.temperature == 0.7
    assert model.request_timeout == 30.0
    assert model.max_retries == 5


def test_build_model_pins_the_chat_completions_api() -> None:
    """`use_responses_api=None` lets langchain infer the endpoint from the model
    name and payload. The HF router is Chat Completions only, so the choice must
    be explicit rather than inferred."""
    model = build_model(ModelConfig(api_key=VALID_SECRET))

    assert model.use_responses_api is USE_RESPONSES_API is False


def test_build_model_ignores_ambient_openai_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """ChatOpenAI falls back to OPENAI_BASE_URL / OPENAI_API_KEY when they are
    unset on the instance. An inherited value must not silently redirect us."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    model = build_model(ModelConfig(api_key=VALID_SECRET))

    assert model.openai_api_base == HF_ROUTER_BASE_URL


def test_build_model_rejects_a_non_config_argument() -> None:
    with pytest.raises(CheckFailed, match="ModelConfig"):
        build_model("openai/gpt-oss-120b")  # type: ignore[arg-type]


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
# build_agent
# --------------------------------------------------------------------------


def test_build_agent_compiles_a_graph_with_the_default_config() -> None:
    agent = build_agent(build_model(ModelConfig(api_key=VALID_SECRET)))

    assert agent is not None
    assert hasattr(agent, "invoke")


def test_build_agent_names_the_graph_from_the_config() -> None:
    model = build_model(ModelConfig(api_key=VALID_SECRET))

    agent = build_agent(model, AgentConfig(name="named-agent"))

    assert agent.name == "named-agent"


def test_build_agent_exposes_the_deepagents_builtin_tools() -> None:
    """deepagents ships planning and filesystem tools; we must not reimplement them."""
    agent = build_agent(build_model(ModelConfig(api_key=VALID_SECRET)))

    tool_names = set(agent.get_graph().nodes)

    assert "tools" in tool_names


def test_build_agent_accepts_tools() -> None:
    @tool
    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    agent = build_agent(
        build_model(ModelConfig(api_key=VALID_SECRET)), AgentConfig(tools=[echo])
    )

    assert agent is not None


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


# --------------------------------------------------------------------------
# The config/callee contract — what makes as_kwargs() splatting safe
# --------------------------------------------------------------------------


def test_model_config_fields_are_all_real_chatopenai_keywords() -> None:
    accepted = set(ChatOpenAI.model_fields) | {
        f.alias for f in ChatOpenAI.model_fields.values() if f.alias is not None
    }

    assert set(ModelConfig(api_key=VALID_SECRET).as_kwargs()) <= accepted


def test_agent_config_fields_are_all_real_create_deep_agent_parameters() -> None:
    accepted = set(inspect.signature(create_deep_agent).parameters)

    assert set(AgentConfig().as_kwargs()) <= accepted


def test_configs_do_not_carry_factory_injected_parameters() -> None:
    """`model` and `use_responses_api` are supplied by the factories. A config
    field of either name would collide on splat."""
    assert "use_responses_api" not in ModelConfig(api_key=VALID_SECRET).as_kwargs()
    assert "model" not in AgentConfig().as_kwargs()


def test_pydantic_param_names_includes_aliases_not_just_field_names() -> None:
    """The aliases are the whole point: `ChatOpenAI` takes `base_url`, while the
    field behind it is named `openai_api_base`. A field-names-only reading would
    reject every keyword ModelConfig actually uses."""
    names = pydantic_param_names(ChatOpenAI)

    assert {"base_url", "api_key", "timeout"} <= names
    assert {"openai_api_base", "openai_api_key", "request_timeout"} <= names


def test_pydantic_param_names_rejects_a_model_with_no_fields() -> None:
    """An empty result would make every contract check vacuously pass."""

    class Fieldless(BaseModel):
        pass

    with pytest.raises(CheckFailed, match="model_fields"):
        pydantic_param_names(Fieldless)


def test_contract_check_rejects_a_field_the_callee_does_not_accept() -> None:
    @dataclass(frozen=True)
    class BadConfig:
        systemprompt: str = "typo"

    with pytest.raises(CheckFailed, match="systemprompt"):
        check_config_contract(
            BadConfig, frozenset({"system_prompt"}), "create_deep_agent", frozenset()
        )


def test_contract_check_rejects_a_field_the_factory_injects() -> None:
    @dataclass(frozen=True)
    class ClashingConfig:
        model: str = "x"

    with pytest.raises(CheckFailed, match="model"):
        check_config_contract(
            ClashingConfig, frozenset({"model"}), "create_deep_agent", frozenset({"model"})
        )


def test_contract_check_rejects_an_uninspectable_callee() -> None:
    """An empty parameter set means introspection failed, not that anything goes."""

    @dataclass(frozen=True)
    class AnyConfig:
        whatever: str = "x"

    with pytest.raises(CheckFailed, match="introspect"):
        check_config_contract(AnyConfig, frozenset(), "mystery", frozenset())


def test_adding_a_setting_needs_no_factory_change() -> None:
    """The point of the parameter object: a new deepagents setting reaches the
    factory as a field, with no edit to build_agent's signature or body."""
    extended = AgentConfig(name="extended")
    kwargs = {**extended.as_kwargs(), "skills": ["./skills/"], "memory": ["./AGENTS.md"]}

    assert set(kwargs) <= set(inspect.signature(create_deep_agent).parameters)


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


def test_agent_config_accepts_real_middleware_and_permissions() -> None:
    config = AgentConfig(
        middleware=[TodoListMiddleware()],
        permissions=[DENY_SECRETS],
    )

    assert len(config.middleware) == 1
    assert len(config.permissions) == 1


def test_permissions_reach_the_filesystem_middleware() -> None:
    """`permissions` only takes effect through FilesystemMiddleware's private
    `_permissions`. build_agent installs that middleware, so it must forward
    them or every rule is silently lost."""
    middleware = _least_privilege_filesystem([DENY_SECRETS])

    assert middleware._permissions == [DENY_SECRETS]


def test_no_permissions_still_produces_a_usable_middleware() -> None:
    """`None` normalises to an empty rule list, not a missing attribute."""
    assert _least_privilege_filesystem(None)._permissions == []


def test_build_agent_refuses_a_middleware_permission_combination_that_drops_rules() -> None:
    """A caller-supplied FilesystemMiddleware replaces ours by name, which would
    silently discard AgentConfig.permissions. Refuse rather than half-apply."""
    with pytest.raises(CheckFailed, match="silently drop"):
        build_agent(
            build_model(ModelConfig(api_key=VALID_SECRET)),
            AgentConfig(
                middleware=[FilesystemMiddleware(tools=list(DEFAULT_FILESYSTEM_TOOLS))],
                permissions=[DENY_SECRETS],
            ),
        )


def test_allowlist_is_every_filesystem_tool_except_the_shell() -> None:
    """If deepagents adds a filesystem tool, the import-time check forces a
    deliberate decision instead of granting it by default."""
    assert set(DEFAULT_FILESYSTEM_TOOLS) == set(get_args(FsToolName)) - {SHELL_TOOL_NAME}
    assert SHELL_TOOL_NAME not in DEFAULT_FILESYSTEM_TOOLS
