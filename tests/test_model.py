"""Contract tests for `my_agent.model`.

Every `require()` in `model.py` gets a test that trips it — that is what turns a
contract into a tested contract. The split that matters here is the one between
error categories: `ModelConfig(...)` raises `CheckFailed` because a caller we own
passed something impossible, while `from_env` raises `ValueError` because the
outside world failed to supply something. Same predicate, different category.

All offline: `ChatOpenAI` builds lazily and makes no request.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from my_agent.model import (
    _ENV_FIELDS,
    API_KEY_ENV_VAR,
    BASE_URL_ENV_VAR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    HF_ROUTER_BASE_URL,
    MAX_RETRIES_ENV_VAR,
    MODEL_ENV_VAR,
    REASONING_EFFORT_ENV_VAR,
    REASONING_EFFORTS,
    TEMPERATURE_ENV_VAR,
    TIMEOUT_ENV_VAR,
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

    kwargs = ModelConfig(api_key=valid_secret).as_kwargs()

    # A subset assertion is vacuously true of an empty dict: with `as_kwargs`
    # stubbed to `return {}` this test passed (measured 2026-09-21), reporting a
    # contract held over no fields at all.
    assert set(kwargs) == {f.name for f in dataclasses.fields(ModelConfig)}
    assert set(kwargs) <= accepted


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


def test_reasoning_effort_offers_exactly_the_three_levels_the_model_has() -> None:
    """gpt-oss was post-trained on three efforts, and the provider rejects the
    rest with a 400 (measured 2026-09-18, F26). A set wider than that is a
    precondition that passes values the request is guaranteed to fail on."""
    assert {"low", "medium", "high"} == REASONING_EFFORTS


@pytest.mark.parametrize(
    "effort",
    ["none", "minimal", "xhigh", "maximum"],
    ids=["chat-template-refuses", "schema-refuses", "schema-refuses-too", "never-existed"],
)
def test_reasoning_effort_rejects_a_value_the_router_would_400_on(
    valid_secret: SecretStr, effort: str
) -> None:
    """A caller we own passed it, so this is a programmer error and crashes.
    `none`, `minimal` and `xhigh` are the ones that matter: the router documents
    them, this set used to list them, and every one of them fails on the wire."""
    with pytest.raises(CheckFailed, match="reasoning_effort"):
        ModelConfig(api_key=valid_secret, reasoning_effort=effort)


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


# --------------------------------------------------------------------------
# from_env reaches every field (F19)
# --------------------------------------------------------------------------


def test_every_field_is_reachable_from_the_environment() -> None:
    """The check that `from_env` cannot drift again. It previously read three of
    seven fields, leaving `base_url` unreachable while this module's docstring
    promised a dedicated endpoint needed no code change."""
    assert {f.name for f in dataclasses.fields(ModelConfig)} == {
        spec.name for spec in _ENV_FIELDS
    }


def test_env_var_names_are_unique() -> None:
    """Two rows sharing a variable would mean one silently shadows the other."""
    names = [spec.env_var for spec in _ENV_FIELDS]
    assert len(set(names)) == len(names)


def test_from_env_reads_the_base_url(valid_key: str) -> None:
    """The bug this fixes: pointing at a dedicated Inference Endpoint used to
    require editing source, despite the docstring promising otherwise."""
    endpoint = "https://abc123.us-east-1.aws.endpoints.huggingface.cloud/v1"
    config = ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, BASE_URL_ENV_VAR: endpoint})
    assert config.base_url == endpoint


def test_from_env_reads_the_numeric_fields(valid_key: str) -> None:
    config = ModelConfig.from_env(
        {
            API_KEY_ENV_VAR: valid_key,
            TEMPERATURE_ENV_VAR: "0.7",
            TIMEOUT_ENV_VAR: "30.5",
            MAX_RETRIES_ENV_VAR: "5",
        }
    )
    assert (config.temperature, config.timeout, config.max_retries) == (0.7, 30.5, 5)


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        (TEMPERATURE_ENV_VAR, "warm"),
        (TIMEOUT_ENV_VAR, "soon"),
        (MAX_RETRIES_ENV_VAR, "2.5"),
    ],
)
def test_from_env_rejects_an_unparseable_value(valid_key: str, env_var: str, value: str) -> None:
    """Naming the variable matters: the user has to know which line to edit."""
    with pytest.raises(ValueError, match=env_var):
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, env_var: value})


def test_from_env_reports_an_out_of_range_value_as_an_operating_error(valid_key: str) -> None:
    """The taxonomy fix. `MODEL_TEMPERATURE=5` parses fine and then trips a
    `require()` — but the value came from the environment, so it must surface as
    a ValueError for the edge to report, not an AssertionError traceback."""
    with pytest.raises(ValueError, match="temperature") as caught:
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, TEMPERATURE_ENV_VAR: "5"})
    assert not isinstance(caught.value, CheckFailed)


def test_an_out_of_range_value_names_the_variable_at_fault(valid_key: str) -> None:
    with pytest.raises(ValueError, match=TEMPERATURE_ENV_VAR):
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, TEMPERATURE_ENV_VAR: "5"})


@pytest.mark.parametrize(
    "env_var", [MODEL_ENV_VAR, BASE_URL_ENV_VAR, TEMPERATURE_ENV_VAR, REASONING_EFFORT_ENV_VAR]
)
def test_from_env_rejects_a_blank_optional_variable(valid_key: str, env_var: str) -> None:
    """A blank line in .env is a broken line, not a request for the default."""
    with pytest.raises(ValueError, match="set but empty"):
        ModelConfig.from_env({API_KEY_ENV_VAR: valid_key, env_var: ""})


def test_from_env_still_falls_back_to_every_default(valid_key: str) -> None:
    config = ModelConfig.from_env({API_KEY_ENV_VAR: valid_key})
    assert (config.model, config.base_url, config.temperature) == (
        DEFAULT_MODEL,
        HF_ROUTER_BASE_URL,
        DEFAULT_TEMPERATURE,
    )
    assert (config.timeout, config.max_retries, config.reasoning_effort) == (
        DEFAULT_TIMEOUT_S,
        DEFAULT_MAX_RETRIES,
        None,
    )


def test_the_router_defaults_are_the_values_that_were_chosen() -> None:
    """Literals, because every other test compares a constant to itself.

    Measured 2026-09-21: `HF_ROUTER_BASE_URL` was changed to
    `"https://api.openai.com/v1"`, `DEFAULT_TEMPERATURE` to 1.9 and
    `DEFAULT_MAX_RETRIES` to 99, and all 376 tests stayed green — including
    `test_build_model_ignores_ambient_openai_env_vars`, which exists to prove an
    ambient `OPENAI_BASE_URL` cannot redirect us and asserts against the
    constant, so it passes when the constant *is* that URL. A pin is only a pin
    when the expected value is written somewhere the mutation cannot reach.
    """
    assert HF_ROUTER_BASE_URL == "https://router.huggingface.co/v1"
    assert DEFAULT_MODEL == "openai/gpt-oss-120b"
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_TIMEOUT_S == 120.0
    assert DEFAULT_MAX_RETRIES == 2
    assert USE_RESPONSES_API is False
    assert frozenset({"low", "medium", "high"}) == REASONING_EFFORTS


def test_the_router_base_url_is_not_a_url_any_other_provider_answers() -> None:
    """The discriminator for the pin above, and the claim
    `test_build_model_ignores_ambient_openai_env_vars` was trying to make: a
    redirect is only detectable if the thing redirected *to* is known to be
    somewhere else."""
    assert "huggingface" in HF_ROUTER_BASE_URL
    assert "openai.com" not in HF_ROUTER_BASE_URL
