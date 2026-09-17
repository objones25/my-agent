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

__all__ = ["check_config_contract", "pydantic_param_names"]


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
