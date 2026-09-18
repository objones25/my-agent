"""Reaching a chat model through the Hugging Face router.

The only module that knows about Hugging Face, base URLs, tokens, or the
environment. Everything downstream accepts a `BaseChatModel` and stays ignorant
of where it came from — `build_agent` never sees a `ModelConfig`.

**`ModelConfig` is a parameter object, not an argument list.** Its field names
are exactly `ChatOpenAI` constructor keywords, and `build_model` splats them.
Adding a setting — `top_p`, `seed` — is one new field with a default: no factory
signature change, no factory body change, no call site change. That coupling is
checked at import rather than assumed, so a typo or an upstream rename fails
when this module loads, by name.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from my_agent.contracts import check_config_contract, pydantic_param_names
from my_agent.negative_space import CheckFailed, require

__all__ = [
    "API_KEY_ENV_VAR",
    "BASE_URL_ENV_VAR",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "HF_ROUTER_BASE_URL",
    "MAX_RETRIES_ENV_VAR",
    "MODEL_ENV_VAR",
    "REASONING_EFFORTS",
    "REASONING_EFFORT_ENV_VAR",
    "TEMPERATURE_ENV_VAR",
    "TIMEOUT_ENV_VAR",
    "USE_RESPONSES_API",
    "ModelConfig",
    "build_model",
]

# --------------------------------------------------------------------------
# Defaults. Change them here, nowhere else.
# --------------------------------------------------------------------------

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
"""Hugging Face Inference Providers router (serverless), OpenAI-compatible.

Not to be confused with `https://api.endpoints.huggingface.cloud/`, which is the
Inference Endpoints *control plane* for creating and managing dedicated
deployments. A dedicated endpoint serves inference at its own
`https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1/` URL — point
`ModelConfig.base_url` there to use one; no code change is needed.
"""

DEFAULT_MODEL = "openai/gpt-oss-120b"
"""`org/model`, optionally suffixed to steer routing: `:provider`, `:fastest`,
`:cheapest`."""

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2

API_KEY_ENV_VAR = "HF_TOKEN"
MODEL_ENV_VAR = "MODEL_ID"
REASONING_EFFORT_ENV_VAR = "REASONING_EFFORT"
BASE_URL_ENV_VAR = "MODEL_BASE_URL"
TEMPERATURE_ENV_VAR = "MODEL_TEMPERATURE"
TIMEOUT_ENV_VAR = "MODEL_TIMEOUT_S"
MAX_RETRIES_ENV_VAR = "MODEL_MAX_RETRIES"

REASONING_EFFORTS = frozenset({"low", "medium", "high"})
"""Values `reasoning_effort` may take (Chat Completions body).

**Three, not six.** gpt-oss was post-trained on exactly three reasoning efforts —
low, medium and high — carried in the system message by the harmony format
(OpenAI, *Introducing gpt-oss*, 5 Aug 2025). The router documents more, and this
set used to list them; measured on the wire 2026-09-18 against
`openai/gpt-oss-120b`, the extras are rejected rather than mapped:

    effort=low       reasoning=10  output=21
    effort=medium    reasoning=63  output=74
    effort=high      reasoning=80  output=91
    effort=None      reasoning=63  output=74     <- identical to medium
    effort=minimal   400  "Input should be 'none', 'low', 'medium' or 'high'"
    effort=xhigh     400  same
    effort=none      400  "Failed to apply chat template ... Unsupported reasoning effort"

`none` is the interesting one: the provider's schema accepts it and the model's
own chat template then refuses it, which is what "provider-side mapping onto a
three-level model" looks like from the outside. Listing it here would mean this
precondition passing a value the request is guaranteed to fail on — the one
thing a precondition exists to prevent (F26).

Unset is *not* zero effort. It is the provider's default, and that default
measured token-for-token identical to `medium`.

This set is a property of the *model*, not of `ModelConfig`: a model with more
levels means widening it deliberately, the way `DEFAULT_FILESYSTEM_TOOLS` is
widened deliberately. Failing here is still better than the alternative, which
is discovering it as a 400 from inside the provider.

`reasoning_effort` is the Chat Completions knob. `ChatOpenAI` also has a
`reasoning` *dict* field, which is the Responses API's; do not use it here.
"""

USE_RESPONSES_API = False
"""Pinned, never inferred, and never configurable.

With `use_responses_api=None` (the library default) `ChatOpenAI` picks the
endpoint from the model name and the request payload, independent of `base_url`
— `langchain_openai.chat_models.base.BaseChatOpenAI._use_responses_api`. The HF
router serves `/v1/chat/completions` only, so an inferred switch would fail at
request time rather than here.
"""

_URL_SCHEMES = ("http://", "https://")
_MAX_TEMPERATURE = 2.0


# --------------------------------------------------------------------------
# Load-time contract between ModelConfig and ChatOpenAI
# --------------------------------------------------------------------------

_CHAT_OPENAI_PARAMS = pydantic_param_names(ChatOpenAI)

# Aliases are the part of the ChatOpenAI contract most likely to move: every
# keyword below is one ModelConfig relies on, and three of the four are aliases
# rather than field names.
_MISSING_CHAT_OPENAI_PARAMS = {"model", "base_url", "api_key", "timeout"} - _CHAT_OPENAI_PARAMS
require(
    not _MISSING_CHAT_OPENAI_PARAMS,
    f"ChatOpenAI no longer accepts {sorted(_MISSING_CHAT_OPENAI_PARAMS)}; ModelConfig must change",
)

_MODEL_INJECTED_PARAMS = frozenset({"use_responses_api"})


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Everything needed to reach a chat model.

    Field names are `ChatOpenAI` constructor keywords. Frozen, so it cannot be
    invalidated after construction and `build_model` needs no defensive re-checks.

    Construct directly in tests; use `from_env` at the process edge.
    """

    api_key: SecretStr
    model: str = DEFAULT_MODEL
    base_url: str = HF_ROUTER_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT_S
    """Seconds."""
    max_retries: int = DEFAULT_MAX_RETRIES
    reasoning_effort: str | None = None
    """How hard the model thinks before answering, or `None` for the provider's
    default. On a reasoning model this is a cost dial, not a quality knob: the
    same answer cost 6 tokens at `low` and 93 at `high` (F17). It is also what a
    token cap actually constrains — reasoning is spent first."""

    def __post_init__(self) -> None:
        # Programmer errors: every one of these is fixed in code, not at runtime.
        # SecretStr keeps the token out of repr(), str() and dataclasses.asdict().
        raw_key = self.api_key.get_secret_value()
        require(raw_key != "", "api_key must not be empty")
        require(
            raw_key == raw_key.strip(),
            "api_key has leading/trailing whitespace; the router answers 401 for this",
        )
        require(self.model != "", "model must not be empty")
        require(
            self.base_url.startswith(_URL_SCHEMES),
            f"base_url must be an http(s) URL, got {self.base_url!r}",
        )
        require(
            0.0 <= self.temperature <= _MAX_TEMPERATURE,
            f"temperature must be in [0.0, {_MAX_TEMPERATURE}], got {self.temperature}",
        )
        require(self.timeout > 0.0, f"timeout must be positive, got {self.timeout}")
        require(self.max_retries >= 0, f"max_retries must be non-negative, got {self.max_retries}")
        require(
            self.reasoning_effort is None or self.reasoning_effort in REASONING_EFFORTS,
            f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)} or None, "
            f"got {self.reasoning_effort!r}; the router answers 400 for an unknown effort",
        )

    def as_kwargs(self) -> dict[str, Any]:
        """Constructor keywords for `ChatOpenAI`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        """Read the config from the environment.

        Every field is reachable — `_ENV_FIELDS` is checked against the dataclass
        at import, so a new field that nobody wired up fails the load rather than
        being quietly unreachable.

        **Everything here is an operating error.** Absent, blank, unparseable, or
        out-of-range values all came from the outside world, so all of them raise
        `ValueError` for the edge to report. That includes contract violations:
        constructing the config can trip a `require()`, and this catches that
        `CheckFailed` and re-raises it in the right category. Without that,
        `MODEL_TEMPERATURE=5` would crash with an `AssertionError` traceback
        implying a bug in this code, rather than telling the user to fix their
        environment.

        `env` is injectable so tests never touch the real process environment.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        kwargs: dict[str, Any] = {}

        for spec in _ENV_FIELDS:
            raw = source.get(spec.env_var)
            if raw is None:
                if spec.required:
                    raise ValueError(
                        f"{spec.env_var} is unset or empty. Set it in .env "
                        f"(see .env.example) or export it before starting the agent."
                    )
                continue

            value = raw.strip()
            if not value:
                # Set-but-blank is a broken .env line, not a request for the
                # default. Saying so beats silently ignoring what someone wrote.
                if spec.required:
                    raise ValueError(
                        f"{spec.env_var} is unset or empty. Set it in .env "
                        f"(see .env.example) or export it before starting the agent."
                    )
                raise ValueError(
                    f"{spec.env_var} is set but empty; unset it to use the default."
                )

            try:
                kwargs[spec.name] = spec.parse(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{spec.env_var}={value!r} is not a valid {spec.name}: {exc}"
                ) from exc

        try:
            return cls(**kwargs)
        except CheckFailed as exc:
            raise ValueError(f"{_blame(exc, kwargs)}{exc}") from exc


check_config_contract(ModelConfig, _CHAT_OPENAI_PARAMS, "ChatOpenAI", _MODEL_INJECTED_PARAMS)


# --------------------------------------------------------------------------
# The environment -> ModelConfig table, and the check that keeps it complete
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EnvField:
    """One field, the variable that sets it, and how to read it."""

    name: str
    """A `ModelConfig` field name. Checked against the dataclass at import."""
    env_var: str
    parse: Callable[[str], Any]
    required: bool = False


_ENV_FIELDS: tuple[_EnvField, ...] = (
    _EnvField("api_key", API_KEY_ENV_VAR, SecretStr, required=True),
    _EnvField("model", MODEL_ENV_VAR, str),
    _EnvField("base_url", BASE_URL_ENV_VAR, str),
    _EnvField("temperature", TEMPERATURE_ENV_VAR, float),
    _EnvField("timeout", TIMEOUT_ENV_VAR, float),
    _EnvField("max_retries", MAX_RETRIES_ENV_VAR, int),
    _EnvField("reasoning_effort", REASONING_EFFORT_ENV_VAR, str),
)
"""Data, not code, so adding a setting stays one new field plus one row here.

A table rather than reflection over annotations on purpose: inferred variable
names would be implicit and inferred parsers would produce generic errors, and
this module pays for explicitness everywhere else.
"""

# The point of the table. `from_env` was hand-maintained and drifted: it read
# three of seven fields, so `base_url` was unreachable from the environment while
# this module's own docstring promised a dedicated endpoint needed "no code
# change". Defining the mapping as data lets a check assert it covers the
# dataclass, the same way DEFAULT_FILESYSTEM_TOOLS is defined by subtraction and
# then asserted. A new field now fails the import instead of being forgotten.
_MODEL_CONFIG_FIELDS = {f.name for f in fields(ModelConfig)}
_ENV_FIELD_NAMES = {spec.name for spec in _ENV_FIELDS}

require(
    not (_MODEL_CONFIG_FIELDS - _ENV_FIELD_NAMES),
    f"ModelConfig fields unreachable from the environment: "
    f"{sorted(_MODEL_CONFIG_FIELDS - _ENV_FIELD_NAMES)}; add a row to _ENV_FIELDS",
)
require(
    not (_ENV_FIELD_NAMES - _MODEL_CONFIG_FIELDS),
    f"_ENV_FIELDS names that are not ModelConfig fields: "
    f"{sorted(_ENV_FIELD_NAMES - _MODEL_CONFIG_FIELDS)}",
)
require(
    len({spec.env_var for spec in _ENV_FIELDS}) == len(_ENV_FIELDS),
    "two _ENV_FIELDS rows share an environment variable; one would shadow the other",
)


def _blame(exc: CheckFailed, kwargs: dict[str, Any]) -> str:
    """Name the variable at fault, when the failed check names its field.

    Every `require()` message in `__post_init__` starts with the field name, so
    this maps the failure back to the variable the user actually has to edit.
    A heuristic for the message only — correctness does not depend on it, and it
    degrades to no prefix when it cannot tell.
    """
    for spec in _ENV_FIELDS:
        if spec.name in kwargs and str(exc).startswith(spec.name):
            return f"{spec.env_var}: "
    return ""


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_model(config: ModelConfig) -> ChatOpenAI:
    """Build the chat model for the HF router.

    `ChatOpenAI` rather than `init_chat_model`: the latter resolves a provider
    from the model string, and `org/model:provider` router ids are meaningless
    to it.

    Returns the concrete type while `build_agent` accepts the abstract one —
    specific in what we return, liberal in what we accept.
    """
    require(isinstance(config, ModelConfig), f"expected a ModelConfig, got {type(config).__name__}")

    model = ChatOpenAI(**config.as_kwargs(), use_responses_api=USE_RESPONSES_API)

    # Postconditions. ChatOpenAI falls back to OPENAI_API_BASE / OPENAI_BASE_URL
    # and rewrites `temperature` for some model families, so what we asked for is
    # not necessarily what we got. A silent redirect to api.openai.com would
    # otherwise surface as a confusing auth failure much later.
    require(
        model.openai_api_base == config.base_url,
        f"base_url was overridden: asked {config.base_url!r}, got {model.openai_api_base!r}",
    )
    require(
        model.model_name == config.model,
        f"model was overridden: asked {config.model!r}, got {model.model_name!r}",
    )
    require(
        model.use_responses_api is False,
        "use_responses_api must stay False; the HF router is Chat Completions only",
    )
    return model
