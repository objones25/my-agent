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
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from my_agent.contracts import check_config_contract, pydantic_param_names
from my_agent.negative_space import require

__all__ = [
    "API_KEY_ENV_VAR",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "HF_ROUTER_BASE_URL",
    "MODEL_ENV_VAR",
    "REASONING_EFFORTS",
    "REASONING_EFFORT_ENV_VAR",
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

REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
"""Values the router documents for `reasoning_effort` (Chat Completions body).

Verified on the wire 2026-09-17 with `openai/gpt-oss-120b`: one prompt answered
identically at every setting, while reasoning tokens went 6 (`low`), 50 (unset),
93 (`high`). See `docs/findings.md` F17. Unset is *not* zero effort — it is the
provider's default, which sat between `low` and `high`.

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

        Absent or malformed environment is an *operating* error — the outside
        world failed to supply something, which is not a bug in this code — so it
        raises `ValueError` to be handled at the edge rather than tripping a check.

        `env` is injectable so tests never touch the real process environment.
        """
        source: Mapping[str, str] = os.environ if env is None else env

        api_key = source.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(
                f"{API_KEY_ENV_VAR} is unset or empty. Set it in .env "
                f"(see .env.example) or export it before starting the agent."
            )

        model = source.get(MODEL_ENV_VAR, DEFAULT_MODEL).strip()
        if not model:
            raise ValueError(f"{MODEL_ENV_VAR} is set but empty; unset it to use the default.")

        # Absent means "provider's default", which is the behaviour every run had
        # before this field existed. A *wrong* value came from the outside world,
        # so it is an operating error reported at the edge, not a CheckFailed.
        effort = source.get(REASONING_EFFORT_ENV_VAR, "").strip() or None
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(
                f"{REASONING_EFFORT_ENV_VAR}={effort!r} is not one of "
                f"{sorted(REASONING_EFFORTS)}; unset it to use the provider's default."
            )

        return cls(api_key=SecretStr(api_key), model=model, reasoning_effort=effort)


check_config_contract(ModelConfig, _CHAT_OPENAI_PARAMS, "ChatOpenAI", _MODEL_INJECTED_PARAMS)


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
