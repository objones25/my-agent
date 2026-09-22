"""Contract tests for `my_agent.contracts`.

These test the checker itself, not any config that uses it. If this machinery is
wrong, every `as_kwargs()` splat in the project is unguarded and the failures
surface as `TypeError`s from inside langchain instead of by name at import.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from my_agent.contracts import check_config_contract, pydantic_param_names
from my_agent.negative_space import CheckFailed

# --------------------------------------------------------------------------
# pydantic_param_names
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# check_config_contract
# --------------------------------------------------------------------------


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


def test_contract_check_rejects_a_config_class_with_no_fields() -> None:
    """The vacuity guard. Every check below it compares the config's field names
    against the callee's parameters, and an empty set is a subset of anything —
    so a dataclass that lost its fields would pass the contract by having
    nothing to contradict it."""

    @dataclass(frozen=True)
    class Empty:
        pass

    with pytest.raises(CheckFailed, match="has no fields"):
        check_config_contract(Empty, frozenset({"a"}), "callee", frozenset())
