"""Contract tests for `my_agent.model`.

Every `require()` in `model.py` gets a test that trips it — that is what turns a
contract into a tested contract. The split that matters here is the one between
error categories: `ModelConfig(...)` raises `CheckFailed` because a caller we own
passed something impossible, while `from_env` raises `ValueError` because the
outside world failed to supply something. Same predicate, different category.

All offline: `ChatOpenAI` builds lazily and makes no request.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from my_agent.model import (
    API_KEY_ENV_VAR,
    DEFAULT_MODEL,
    HF_ROUTER_BASE_URL,
    MODEL_ENV_VAR,
    REASONING_EFFORT_ENV_VAR,
    REASONING_EFFORTS,
    USE_RESPONSES_API,
    ModelConfig,
    build_model,
)
from my_agent.negative_space import CheckFailed

# --------------------------------------------------------------------------
# ModelConfig: positive space
# --------------------------------------------------------------------------


def test_model_config_defaults_target_the_hf_router(valid_secret: SecretStr) -> None:
    config = ModelConfig(api_key=valid_secret)

    assert config.base_url == HF_ROUTER_BASE_URL
    assert config.model == DEFAULT_MODEL


def test_model_config_is_frozen(valid_secret: SecretStr) -> None:
    config = ModelConfig(api_key=valid_secret)

    with pytest.raises(AttributeError):
        config.model = "other"  # type: ignore[misc]


def test_model_config_repr_does_not_leak_the_api_key() -> None:
    config = ModelConfig(api_key=SecretStr("super-secret-token"))

    assert "super-secret-token" not in repr(config)


# --------------------------------------------------------------------------
# ModelConfig: negative space — programmer errors, so CheckFailed
# --------------------------------------------------------------------------


def test_model_config_rejects_an_empty_api_key() -> None:
    with pytest.raises(CheckFailed, match="api_key"):
        ModelConfig(api_key=SecretStr(""))


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"model": ""}, "model"),
        ({"base_url": "router.huggingface.co/v1"}, "base_url"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"timeout": 0.0}, "timeout"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_model_config_rejects_impossible_values(
    valid_secret: SecretStr, kwargs: dict[str, Any], expected_message: str
) -> None:
    with pytest.raises(CheckFailed, match=expected_message):
        ModelConfig(api_key=valid_secret, **kwargs)


def test_model_config_rejects_api_key_with_surrounding_whitespace(valid_key: str) -> None:
    """A token with a trailing newline is the classic .env bug: it produces a
    confusing 401 from the router rather than a clear local failure."""
    with pytest.raises(CheckFailed, match="whitespace"):
        ModelConfig(api_key=SecretStr(f"{valid_key}\n"))


# --------------------------------------------------------------------------
# ModelConfig.from_env: operating errors, so ValueError (never CheckFailed)
# --------------------------------------------------------------------------


def test_from_env_reads_the_token_and_defaults_the_model(valid_key: str) -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: valid_key})

    assert config.api_key.get_secret_value() == valid_key
    assert config.model == DEFAULT_MODEL


def test_from_env_honours_an_explicit_model_id(valid_key: str) -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, MODEL_ENV_VAR: "org/other:novita"})

    assert config.model == "org/other:novita"


def test_from_env_strips_whitespace_so_a_trailing_newline_is_survivable(valid_key: str) -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: f"  {valid_key}\n"})

    assert config.api_key.get_secret_value() == valid_key


@pytest.mark.parametrize("env", [{}, {API_KEY_ENV_VAR: ""}, {API_KEY_ENV_VAR: "   "}])
def test_from_env_raises_value_error_when_the_token_is_absent(env: dict[str, str]) -> None:
    """Missing configuration is an operating error, not a programmer error."""
    with pytest.raises(ValueError, match=API_KEY_ENV_VAR):
        ModelConfig.from_env(env)

    assert not issubclass(ValueError, CheckFailed)


def test_from_env_raises_value_error_on_a_blank_model_id(valid_key: str) -> None:
    with pytest.raises(ValueError, match=MODEL_ENV_VAR):
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, MODEL_ENV_VAR: "   "})


# --------------------------------------------------------------------------
# build_model
# --------------------------------------------------------------------------


def test_build_model_applies_every_config_field(valid_secret: SecretStr) -> None:
    config = ModelConfig(
        api_key=valid_secret,
        model="org/model:provider",
        temperature=0.7,
        timeout=30.0,
        max_retries=5,
    )

    model = build_model(config)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "org/model:provider"
    assert model.openai_api_base == HF_ROUTER_BASE_URL
    assert model.temperature == pytest.approx(0.7)
    assert model.request_timeout == pytest.approx(30.0)
    assert model.max_retries == 5


def test_build_model_pins_the_chat_completions_api(valid_secret: SecretStr) -> None:
    """`use_responses_api=None` lets langchain infer the endpoint from the model
    name and payload. The HF router is Chat Completions only, so the choice must
    be explicit rather than inferred."""
    model = build_model(ModelConfig(api_key=valid_secret))

    assert model.use_responses_api is USE_RESPONSES_API is False


def test_build_model_ignores_ambient_openai_env_vars(
    valid_secret: SecretStr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ChatOpenAI falls back to OPENAI_BASE_URL / OPENAI_API_KEY when they are
    unset on the instance. An inherited value must not silently redirect us."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    model = build_model(ModelConfig(api_key=valid_secret))

    assert model.openai_api_base == HF_ROUTER_BASE_URL


def test_build_model_rejects_a_non_config_argument() -> None:
    with pytest.raises(CheckFailed, match="ModelConfig"):
        build_model("openai/gpt-oss-120b")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# ModelConfig's half of the config/callee contract
# --------------------------------------------------------------------------


def test_model_config_fields_are_all_real_chatopenai_keywords(valid_secret: SecretStr) -> None:
    accepted = set(ChatOpenAI.model_fields) | {
        f.alias for f in ChatOpenAI.model_fields.values() if f.alias is not None
    }

    assert set(ModelConfig(api_key=valid_secret).as_kwargs()) <= accepted


def test_model_config_does_not_carry_the_factory_injected_parameter(
    valid_secret: SecretStr,
) -> None:
    """`use_responses_api` is supplied by build_model. A field of that name would
    collide on splat."""
    assert "use_responses_api" not in ModelConfig(api_key=valid_secret).as_kwargs()


# --------------------------------------------------------------------------
# reasoning_effort — the dial (F17)
# --------------------------------------------------------------------------


def test_reasoning_effort_is_unset_by_default(valid_secret: SecretStr) -> None:
    """Unset means "provider's choice", which is what every run did before this
    field existed. Adding the dial must not silently change behaviour."""
    assert ModelConfig(api_key=valid_secret).reasoning_effort is None


@pytest.mark.parametrize("effort", sorted(REASONING_EFFORTS))
def test_reasoning_effort_accepts_every_documented_value(
    valid_secret: SecretStr, effort: str
) -> None:
    assert ModelConfig(api_key=valid_secret, reasoning_effort=effort).reasoning_effort == effort


def test_reasoning_effort_rejects_an_undocumented_value(valid_secret: SecretStr) -> None:
    """A caller we own passed it, so this is a programmer error and crashes.
    The router answers 400 for an unknown effort; failing here names the field."""
    with pytest.raises(CheckFailed, match="reasoning_effort"):
        ModelConfig(api_key=valid_secret, reasoning_effort="maximum")


def test_reasoning_effort_reaches_the_model(valid_secret: SecretStr) -> None:
    """The whole point: it has to survive the splat into ChatOpenAI."""
    model = build_model(ModelConfig(api_key=valid_secret, reasoning_effort="low"))
    assert model.reasoning_effort == "low"


def test_reasoning_effort_is_absent_from_the_payload_when_unset(valid_secret: SecretStr) -> None:
    """None must not be sent as a literal null — the router would reject it."""
    model = build_model(ModelConfig(api_key=valid_secret))
    assert "reasoning_effort" not in model._default_params


def test_from_env_reads_reasoning_effort(valid_key: str) -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, REASONING_EFFORT_ENV_VAR: "high"})
    assert config.reasoning_effort == "high"


def test_from_env_rejects_an_invalid_reasoning_effort(valid_key: str) -> None:
    """From the environment this is an *operating* error, not a programmer error:
    the outside world supplied it, so it is reported, not crashed on."""
    with pytest.raises(ValueError, match="REASONING_EFFORT"):
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, REASONING_EFFORT_ENV_VAR: "maximum"})


def test_from_env_leaves_reasoning_effort_unset_when_absent(valid_key: str) -> None:
    assert ModelConfig.from_env({API_KEY_ENV_VAR: valid_key}).reasoning_effort is None
