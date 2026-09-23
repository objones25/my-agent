# deepagents profiles, and the door the entry-point pin cannot see — 2026-09-23

An investigation into `ProviderProfile`, `HarnessProfile`, and whether either is usable — or
dangerous — for this harness. Prompted by a question about `with_config` in `build_agent`, which
turned out to be a separate matter; §1 disposes of it.

Everything below was read off the installed wheel (deepagents 0.7.15) and confirmed by running it,
not from documentation alone.

## Verdict

**Provider profiles cannot reach us. Harness profiles can, and nothing was checking.**

`DEEPAGENTS_PLUGIN_GROUPS` pins the two *entry-point* groups to empty. deepagents registers its own
builtin profiles by explicit module import instead, which that pin cannot see — the library says so
in its own bootstrap docstring. Harness-profile resolution reaches a **pre-built model instance**
and falls back to a **bare provider key**, and our model reports provider `openai`. Today no bare
`openai` key is registered, so nothing matches. That was a comment in `capabilities.py`, not a
check. It is now a check.

## 1. `with_config` is not on the model

`create_deep_agent` takes `model: str | BaseChatModel`, and a `RunnableConfig` would indeed be
wrong there. `build_agent` never does that. Both calls apply to a **compiled graph**:

- `agent.py:521` — `PARENT_STEP_LIMIT` on the returned parent (F39)
- `agent.py:392` — `SUBAGENT_STEP_LIMIT` on deepagents' own subagent graph (F24)

Worth verifying anyway, because base `Runnable.with_config` returns a `RunnableBinding` — which
would make `build_agent`'s return annotation false and break every read-back in `capabilities.py`.
langgraph overrides it:

```python
# langgraph.pregel.Pregel
def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
    """Create a copy of the Pregel object with an updated config."""
    return self.copy({"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))})
```

`-> Self`, via `.copy()` with `merge_configs`. The annotation is honest, and the *merge* is why
F24's note holds: deepagents' later `with_config({metadata, run_name})` on the same graph does not
clear our `recursion_limit`.

### Should `build_agent` accept a `RunnableConfig`?

**No, and the signature does not need to change either way.** Config keys divide in two:

- **Bounds** (`recursion_limit`, `max_concurrency`) already have an owner. A caller-supplied graph
  config would be a *third* place they can be set — graph default, caller config, per-invocation
  `run_turn` — with `merge_configs` deciding. That is exactly what "a bound belongs to the thing it
  bounds, not to the call site" exists to prevent.
- **Non-bounds** (`tags`, `metadata`, `run_name`) are harmless but unused: tracing is ambient and
  `run_turn` already carries callbacks.

If it is ever needed, it belongs as an `AgentConfig` field — the repo's stated pattern is one config
object, not a growing parameter list — and it should reject the bound keys explicitly.

## 2. Two mechanisms, and only one reaches us

| | `ProviderProfile` | `HarnessProfile` |
|---|---|---|
| registered by | `register_provider_profile(key, profile)` | `register_harness_profile(key, profile)` |
| key shape | `provider` or `provider:model` | same |
| payload | `init_kwargs`, `init_kwargs_factory`, `pre_init` | prompt, tools, middleware, subagent |
| applies to a model **string** | yes | yes |
| applies to a pre-built **instance** | **no** | **yes** |

### Provider profiles are inert here

Their entire payload is forwarded to `init_chat_model`, which only happens when deepagents
*constructs* the model — that is, when given a `provider:model` string. `build_agent` refuses
strings, so `register_provider_profile` would do nothing for us.

This is F28 restated, and it still holds: the builtin `openai` provider profile sets
`use_responses_api=True`, which is F1's pin reversed, and the string refusal is what keeps it away.

### Harness profiles reach a pre-built instance

`_harness_profile_for_model(model, spec)` has an explicit branch for `spec=None`, documented as
"pre-built instances". With no spec it derives the key from the object and tries, in order:

1. `provider:identifier`, exact
2. `identifier` alone, exact, when the identifier contains a colon
3. **the bare provider key**, as provider-wide defaults

Measured for our model:

```
identifier: openai/gpt-oss-120b
provider  : openai
resolved  : HarnessProfile()        # empty — nothing matches
```

Registered keys in 0.7.15:

```
NVIDIA:nvidia/nemotron-3-ultra-550b-a55b        anthropic:claude-haiku-4-5
anthropic:claude-opus-4-7                       anthropic:claude-sonnet-4-6
baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B
fireworks:accounts/fireworks/models/nemotron-3-ultra-bf16
fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4
nebius:nvidia/Nemotron-3-Ultra-550b-a55b        nvidia:nvidia/nemotron-3-ultra-550b-a55b
openai:gpt-5.1-codex                            openai:gpt-5.2-codex
openai:gpt-5.3-codex                            openrouter:nvidia/nemotron-3-ultra-550b-a55b
together:nvidia/nemotron-3-ultra-550b-a55b
```

All exact `provider:model`. No bare `openai`. **The `openai` provider namespace is already in
active use**, and one upstream release registering provider-wide defaults under it would apply to
us automatically.

A matching `HarnessProfile` may set `base_system_prompt`, `system_prompt_suffix`,
`tool_description_overrides`, `excluded_tools`, `excluded_middleware`, `extra_middleware`, and
`general_purpose_subagent`. `_require_shell_withheld` would still catch the tool half by reading
the compiled graph back. **Nothing can read a prompt or middleware change back off a compiled
graph** (F41), so those two are invisible after the fact.

## 3. Why `DEEPAGENTS_PLUGIN_GROUPS` does not cover this

From `deepagents/profiles/_builtin_profiles.py`, the library's own words:

> "Built-in provider and harness profiles are registered via explicit module imports — **not entry
> points** — so a malformed or missing `dist-info` in the environment cannot silently disable the
> SDK's own defaults."

Our pin asserts the two entry-point groups are empty, which covers the *third-party* door and
nothing else. The builtin registry is a second door with the same blast radius.

The repo already knew. `capabilities.DEEPAGENTS_PLUGIN_GROUPS`' docstring says *"the builtin harness
profiles are keyed per-model (Anthropic models, Nemotron, Codex) so none matches the router model"*
— verified on a date, in prose. That is the repo's own failure mode: a claim nothing executes. An
upstream release makes the comment false and no test notices.

## 4. What was added

`capabilities.require_no_harness_profile(model)`, called from `build_agent` **before** the build,
because the prompt and middleware halves cannot be asserted afterwards. It resolves through
deepagents' own `_harness_profile_for_model` rather than scanning the registry by hand, so a change
to key semantics keeps the check correct instead of quietly ceasing to match. When a profile does
resolve, the error names the fields it would change.

A postcondition on the model rather than an import-time key check, for two reasons: the registry is
populated lazily, and `build_agent` already has the real object — which is the same preference for
reading back a real thing that the rest of `capabilities.py` follows.

Four mutants killed: check never raises, check always raises, changed-field list dropped from the
message, and `build_agent` no longer calling it.

### A checker disagreement, and why the fields ended up pinned

The first version built its diagnostic with `dataclasses.fields(profile)`. `pyright` 1.1.414
(standard), which `scripts/check.sh` runs, accepts that. **Pylance's bundled build rejects it:**

```
Argument of type "HarnessProfile" cannot be assigned to parameter "class_or_instance"
of type "DataclassInstance | type[DataclassInstance]"
  "__dataclass_fields__" is not present
```

`HarnessProfile` *is* a dataclass at runtime — `dataclasses.is_dataclass()` is `True` and
`__dataclass_fields__` is present — so this is the two checkers disagreeing about a third-party
type, which F16 says to resolve rather than silence.

Silencing it with a `cast(Any, ...)` would have been papering over. Instead `HARNESS_PROFILE_FIELDS`
pins the seven field names as literals, and `require_known_harness_profile_fields` asserts at import
that the installed class still has exactly those. That buys three things at once: both checkers
agree on a literal tuple; the check no longer depends on upstream remaining a dataclass; and **a new
field upstream becomes an import failure** instead of a new reconfiguration knob quietly missing
from the diagnostic. The one remaining read uses
`getattr(HarnessProfile, "__dataclass_fields__", {})`, which yields `()` and fails the pin loudly if
upstream ever stops being a dataclass.

Pinned to literals rather than read off the class, because a tuple compared to
`dataclasses.fields()` of the same object is the tautology F42 exists to stop.

## 5. `ModelFallbackMiddleware` — the string form is a silent redirect

The documented example passes model strings. For this harness that is not merely wrong:

```
init_chat_model("openai:openai/gpt-oss-120b")
→ OpenAIError: Missing credentials ... set the OPENAI_API_KEY env var
```

It is building a client against **api.openai.com**, not the router. It failed only because
`OPENAI_API_KEY` is unset here; with that variable present it would have succeeded and sent traffic
to OpenAI proper. `build_model` has a postcondition guarding exactly this redirect
(`test_build_model_ignores_ambient_openai_env_vars`), but a fallback string never passes through
`build_model`, so the guard never runs.

**If fallback is ever added: instances only, each built by `build_model`.** And note it reaches the
parent graph only (F30), so it is not a global ceiling.

## 6. Should we *use* profiles deliberately?

Considered and declined, for now.

`register_harness_profile("openai:openai/gpt-oss-120b", HarnessProfile(excluded_tools={"execute"}))`
is a deepagents-native way to express the allowlist. It is weaker than what `capabilities.py`
already does: withholding by subtraction with a read-back assertion against the compiled graph and
every subagent graph. Registration is also a process-global mutation with no unregister — action at
a distance, which is what the rest of this harness is built to avoid.

The one thing a profile offers that we lack is `general_purpose_subagent`, a supported hook for
configuring the subagent that `agent.py` currently reaches by rebinding a `CompiledSubAgent` (F24).
Worth revisiting only if that route breaks.

## 7. Open

- Whether a future deepagents release registers bare-provider harness profiles. The new check turns
  that into a build failure rather than a silent change, which is the whole point of it.
- Whether `excluded_middleware` in a matching profile could strip our `FilesystemMiddleware` and
  thereby the allowlist. `_require_shell_withheld` should catch it; untested, because no profile
  matches to try it with.
- `GeneralPurposeSubagentProfile`'s fields were not examined. If the F24 rebinding route ever
  breaks, that is where to look first.
