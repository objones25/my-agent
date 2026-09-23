"""Load-time checks that our configs really match the libraries they configure.

`ModelConfig` and `AgentConfig` are parameter objects: their field names are the
callee's parameter names, and the factories splat them (`ChatOpenAI(**cfg)`).
That buys a factory signature which never changes, at the cost of trusting field
names to be real parameters — a typo would surface as a `TypeError` from deep
inside the library, or vanish into a `**kwargs` signature.

`check_config_contract` is what converts that trust into a check. Both config
modules call it at import, so a misspelled field or an upstream rename fails when
the package loads, naming the offending field.

Nothing here knows about a specific library; the callers supply the callee's
parameters and its name.
"""

from __future__ import annotations

from dataclasses import fields

from pydantic import BaseModel

from my_agent.negative_space import require

__all__ = [
    "check_config_contract",
    "check_known_parameters",
    "check_required_parameters",
    "pydantic_param_names",
]


def pydantic_param_names(model_cls: type[BaseModel]) -> frozenset[str]:
    """Constructor keywords a pydantic model accepts: field names and aliases.

    A pydantic-generated `__init__` is `(**data)`, so `inspect.signature` reveals
    nothing usable and the model fields are the real contract. Aliases count:
    `ChatOpenAI` takes `base_url`, not its field name `openai_api_base`.
    """
    names: set[str] = set()
    for name, field in model_cls.model_fields.items():
        names.add(name)
        if field.alias is not None:
            names.add(field.alias)

    # Postcondition. Callers validate their fields against this set, so an empty
    # result would make every one of those checks vacuously pass.
    require(names != set(), f"{model_cls.__name__} exposes no model_fields; introspection broke")
    return frozenset(names)


def check_config_contract(
    config_cls: type,
    callee_params: frozenset[str],
    callee_name: str,
    injected: frozenset[str],
) -> None:
    """Fail at import if a config field is not a real parameter of its callee.

    `injected` names parameters the factory supplies itself. A config field of
    the same name would collide on splat, so those are refused rather than
    silently shadowed.
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


def check_known_parameters(
    actual: frozenset[str], known: frozenset[str], callee_name: str, why_new_matters: str
) -> None:
    """Fail if a callee's parameter set has drifted from the reviewed one.

    `check_config_contract` asserts our fields are real parameters. It cannot
    notice a **new** parameter appearing — and a new parameter is exactly how
    deepagents' shell `execute` tool arrived switched on with no opt-in.

    A function rather than two checks written at module level, for the reason
    `check_config_contract` is one: a load-time check nothing can call is a
    check nothing can test, and the only way to drive it otherwise is to reload
    the defining module — which rebinds every class it defines and breaks
    `isinstance` for every instance another module is holding (F44).

    Two `require()`s, not one: gaining and losing a parameter are different
    events needing different responses, and a compound check names neither.
    """
    gained = actual - known
    require(
        not gained,
        f"{callee_name} gained parameters {sorted(gained)}. {why_new_matters}",
    )
    removed = known - actual
    require(
        not removed,
        f"{callee_name} no longer accepts {sorted(removed)}; the config and this pin must "
        f"change together",
    )


def check_required_parameters(
    actual: frozenset[str], needed: frozenset[str], callee_name: str
) -> None:
    """Fail if a callee stopped accepting a keyword a config relies on.

    Narrower than `check_known_parameters`: this says nothing about parameters
    appearing, only that the ones being counted on are still there. Aliases are
    the usual casualty — `ChatOpenAI` takes `base_url`, not its field name
    `openai_api_base`, and an alias that disappears turns a splat into a
    pydantic error naming a field rather than the rename behind it.
    """
    require(actual != frozenset(), f"could not introspect {callee_name} parameters")
    missing = needed - actual
    require(
        not missing,
        f"{callee_name} no longer accepts {sorted(missing)}; the config must change",
    )
