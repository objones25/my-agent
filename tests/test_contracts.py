"""Contract tests for `my_agent.contracts`.

These test the checker itself, not any config that uses it. If this machinery is
wrong, every `as_kwargs()` splat in the project is unguarded and the failures
surface as `TypeError`s from inside langchain instead of by name at import.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from my_agent.agent import KNOWN_CREATE_DEEP_AGENT_PARAMS
from my_agent.contracts import (
    check_config_contract,
    check_known_parameters,
    check_required_parameters,
    pydantic_param_names,
)
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


# --------------------------------------------------------------------------
# The load-time parameter pins
#
# These live here, as functions, rather than as bare checks in `agent.py` and
# `model.py`, because a check at module level can only be driven by reloading
# the module that holds it — and reloading a module rebinds every class it
# defines, so `AgentConfig` and `ModelConfig` become new objects and every
# `isinstance` against them fails for instances other test modules are holding
# (F44). Expressed as functions, they are ordinary tests.
# --------------------------------------------------------------------------


def test_known_parameters_refuses_a_callee_that_gained_one() -> None:
    """The door `check_config_contract` cannot see: it asserts our fields are
    real parameters and cannot notice a *new* one appearing, which is exactly
    how deepagents' shell `execute` tool arrived switched on (F4)."""
    with pytest.raises(CheckFailed, match="enable_remote_code_execution"):
        check_known_parameters(
            frozenset({"model", "tools", "enable_remote_code_execution"}),
            frozenset({"model", "tools"}),
            "create_deep_agent",
            "Review each for what it enables by default.",
        )


def test_known_parameters_carries_the_reason_a_new_one_matters() -> None:
    """The message is the whole value of this check — it is read by whoever hits
    a failed import after an upgrade, and 'gained a parameter' without 'here is
    why you must look' is an instruction to add the name and move on."""
    with pytest.raises(CheckFailed, match="same door the shell"):
        check_known_parameters(
            frozenset({"a", "b"}),
            frozenset({"a"}),
            "callee",
            "This is the same door the shell `execute` tool came through.",
        )


def test_known_parameters_refuses_a_callee_that_lost_one() -> None:
    """The other direction, and the discriminator. A removed parameter is a
    config field that no longer splats anywhere — silently, because `as_kwargs()`
    still produces it and the factory is still called."""
    with pytest.raises(CheckFailed, match="no longer accepts"):
        check_known_parameters(
            frozenset({"model"}), frozenset({"model", "permissions"}), "create_deep_agent", "why"
        )


def test_known_parameters_accepts_a_callee_that_has_not_moved(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    """Both halves must fail; neither may fail on the set that actually ships."""
    assert_does_not_raise(
        lambda: check_known_parameters(
            KNOWN_CREATE_DEEP_AGENT_PARAMS,
            KNOWN_CREATE_DEEP_AGENT_PARAMS,
            "create_deep_agent",
            "why",
        )
    )


def test_the_shipped_create_deep_agent_still_matches_its_pin(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    """Against the installed wheel, not a fixture. This is the assertion
    `agent.py` makes at import; stating it here is what makes the import-time
    call a tested contract rather than one that has only ever passed."""
    assert_does_not_raise(
        lambda: check_known_parameters(
            frozenset(inspect.signature(create_deep_agent).parameters),
            KNOWN_CREATE_DEEP_AGENT_PARAMS,
            "create_deep_agent",
            "why",
        )
    )


def test_required_parameters_refuses_a_lost_constructor_alias() -> None:
    """`ModelConfig` field names are `ChatOpenAI` *constructor* names, and three
    of the four this relies on are aliases. An alias that disappeared would make
    `build_model`'s splat raise from inside pydantic, naming a field rather than
    the rename that caused it."""
    with pytest.raises(CheckFailed, match="no longer accepts"):
        check_required_parameters(
            frozenset({"model", "api_key", "timeout"}),
            frozenset({"model", "base_url", "api_key", "timeout"}),
            "ChatOpenAI",
        )


def test_required_parameters_refuses_a_callee_it_could_not_introspect() -> None:
    """The vacuity guard: an empty parameter set makes every keyword look
    missing, or — with an empty `needed` — makes the check pass by asking
    nothing."""
    with pytest.raises(CheckFailed, match="could not introspect"):
        check_required_parameters(frozenset(), frozenset({"model"}), "ChatOpenAI")


def test_the_shipped_chat_openai_still_accepts_every_alias_model_config_uses(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    """The discriminator, against the installed wheel."""
    assert_does_not_raise(
        lambda: check_required_parameters(
            pydantic_param_names(ChatOpenAI),
            frozenset({"model", "base_url", "api_key", "timeout"}),
            "ChatOpenAI",
        )
    )


def test_required_parameters_names_the_consequence_it_was_given() -> None:
    """Each caller knows what breaks when a parameter goes, and the message is
    where a reader learns it."""
    with pytest.raises(
        CheckFailed, match=r"^Thing no longer accepts \['b'\]; the widget cannot be set$"
    ):
        check_required_parameters(
            frozenset({"a"}),
            frozenset({"a", "b"}),
            "Thing",
            consequence="the widget cannot be set",
        )


def test_required_parameters_defaults_to_the_config_consequence() -> None:
    """The default keeps `model.py`'s message exactly what it was."""
    with pytest.raises(
        CheckFailed, match=r"^Thing no longer accepts \['b'\]; the config must change$"
    ):
        check_required_parameters(frozenset({"a"}), frozenset({"a", "b"}), "Thing")
