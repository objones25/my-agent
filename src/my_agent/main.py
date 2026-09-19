"""Live smoke checks: do the fixes in `docs/findings.md` actually hold against a
real model?

    uv run my-agent                  # run every check
    uv run my-agent "your prompt"    # one ordinary turn instead

Each check maps to a finding and costs one live round trip. They assert on *tool
messages and token counts*, not on model prose, so they stay deterministic even
though the model does not. Unit tests cover the same fixes offline; these prove
the other half — that the router and the provider actually behave as assumed.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deepagents import FilesystemPermission
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import (
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    compiled_tool_names,
    require_granted,
    require_withheld,
)
from my_agent.mirror import mirror_to_file, run_log_path
from my_agent.model import REASONING_EFFORTS, ModelConfig, build_model
from my_agent.negative_space import CheckFailed, require
from my_agent.run import (
    DeadlineExceeded,
    ResumeLimitExceeded,
    StepLimitExceeded,
    TokenLimitExceeded,
    TurnResult,
    run_turn,
)
from my_agent.tracing import available_backends

EXIT_MISCONFIGURED = 2
EXIT_CHECK_FAILED = 1

TOKEN_CAP = 24
"""Small enough that an ignored cap is unmistakable against an uncapped reply."""

REASONING_PROBE_TOKENS = 16
"""Enough for a one-digit answer. The probe is about acceptance, not output."""

REJECTED_REASONING_EFFORT = "xhigh"
"""A value the router documents and this model refuses (measured 2026-09-18).

The discriminator for the effort check: without a value that must fail, "every
effort we allow was accepted" also passes on a router that accepts everything.
"""

COT_CONTENT_KEYS = ("reasoning", "reasoning_content")
"""Where a provider would put chain-of-thought *text* if it returned any."""

DENIED_PREFIX = "/secrets"
ALLOWED_PATH = "/notes/smoke.txt"
DENY_SECRETS = FilesystemPermission(
    operations=["write"], paths=[f"{DENIED_PREFIX}/**"], mode="deny"
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    finding: str
    name: str
    passed: bool
    detail: str


def _tool_messages(result: TurnResult) -> list[BaseMessage]:
    """The tool record a check reads, and the assertion that the turn finished.

    A paused turn has a tool call in its messages and no result for it, so a
    check reading tool messages off one would report "the tool was not called"
    for a tool that is waiting on a human. None of these checks uses an
    interrupt rule, so a pause here means the harness grew one somewhere else.
    """
    require(
        not result.paused,
        f"turn paused on {len(result.action_requests)} approval request(s); "
        f"these checks configure no interrupt rules",
    )
    return [m for m in result.messages if m.type == "tool"]


def _router_models(config: ModelConfig) -> list[dict[str, Any]]:
    """The router's `/v1/models` listing.

    Read with `urllib` rather than through `ChatOpenAI`: this is a catalogue
    lookup, not a completion, and routing it through the chat client would make
    a check about *what providers offer* depend on a provider answering.
    """
    request = urllib.request.Request(  # noqa: S310 — the URL is our own pinned base
        f"{config.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {config.api_key.get_secret_value()}"},
    )
    with urllib.request.urlopen(request, timeout=config.timeout) as response:  # noqa: S310
        body = json.load(response)
    listing = body.get("data", []) if isinstance(body, dict) else body
    require(
        isinstance(listing, list),
        f"the router returned a {type(listing).__name__}, not a list of models",
    )
    return [m for m in listing if isinstance(m, dict)]


def _activate_tracing() -> tuple[str, ...]:
    """Turn on every configured backend; return the names that turned on.

    A backend that cannot reach its service is an *operating* error: the agent
    still works without telemetry, and losing a run's output to a W&B outage
    would be the worse failure. A `CheckFailed` is a violated contract of ours
    and still crashes.
    """
    active: list[str] = []
    for backend in available_backends():
        try:
            backend.activate()
        except CheckFailed:
            raise
        except Exception as exc:
            print(f"tracing: {backend.name} FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
        else:
            active.append(backend.name)
    return tuple(active)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_chat_completions_endpoint(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F1 — `use_responses_api=False` is pinned, so the router gets
    /v1/chat/completions. A reply at all proves the endpoint is right."""
    reply = build_model(config).invoke(
        "Reply with exactly the word: pong", config={"callbacks": callbacks}
    )
    require(reply.type == "ai", f"expected an AI reply, got {reply.type}")
    text = reply.text.strip().lower()
    return CheckResult(
        "F1",
        "chat-completions endpoint reachable",
        "pong" in text,
        f"reply={text[:60]!r}",
    )


def check_token_cap_reaches_the_router(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F2 — langchain sends the cap as `max_completion_tokens`, while HF documents
    `max_tokens`. Does the router honour what we actually send?"""
    prompt = "Count from 1 to 200, separated by spaces. Output only the numbers."
    capped = (
        build_model(config)
        .bind(max_tokens=TOKEN_CAP)
        .invoke(prompt, config={"callbacks": callbacks})
    )
    usage: dict[str, Any] = dict(capped.usage_metadata or {})
    # Without usage metadata this check cannot distinguish "cap honoured" from
    # "cannot tell", and a PASS would be vacuous. Crash instead of reporting.
    require("output_tokens" in usage, "router returned no usage metadata; cap is unmeasurable")
    produced = int(usage["output_tokens"])
    return CheckResult(
        "F2",
        "token cap honoured as max_completion_tokens",
        0 < produced <= TOKEN_CAP,
        f"asked <={TOKEN_CAP}, produced {produced}",
    )


def check_shell_tool_withheld(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F4 — `execute` is off. Assert both that it is unbound and that no run can
    call it, which holds regardless of how the model phrases its refusal."""
    # build_agent already asserts this over the parent graph *and* every subagent
    # (F20), so it cannot be false here — restated because a live check that only
    # exercises the model would not say whether the tool was ever bound.
    agent = build_agent(build_model(config))
    require_withheld(SHELL_TOOL_NAME, compiled_tool_names(agent), "the compiled graph")

    result = run_turn(
        agent, "Run the shell command `echo hello` and show me the output.", callbacks=callbacks
    )
    called = {m.name for m in _tool_messages(result) if m.name is not None}
    return CheckResult(
        "F4",
        "shell tool withheld",
        SHELL_TOOL_NAME not in called,
        f"unbound; tools called: {sorted(called) or 'none'}",
    )


def check_filesystem_tools_still_work(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F4 corollary — narrowing the allowlist must not break what remains. Uses
    the same permission rules as the denial check, so a pass here proves the
    rules are targeted rather than blanket."""
    agent = build_agent(build_model(config), AgentConfig(permissions=[DENY_SECRETS]))
    # Separates "the model did not try" from "the tool was never there" — without
    # this, a missing tool reports as a model failure.
    require_granted("write_file", compiled_tool_names(agent), "the compiled graph")
    result = run_turn(
        agent,
        f"Use write_file to write the text 'pong' to {ALLOWED_PATH}, then read it back.",
        callbacks=callbacks,
    )
    tools = _tool_messages(result)
    wrote = any(
        m.name == "write_file" and "permission denied" not in str(m.content).lower() for m in tools
    )
    return CheckResult(
        "F4",
        "remaining filesystem tools still usable",
        wrote,
        f"tool calls: {[m.name for m in tools] or 'none'}",
    )


def check_permissions_are_enforced(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F5 — the big one. Our FilesystemMiddleware replaces the default, so it has
    to forward `_permissions`; if it does not, every rule vanishes silently."""
    agent = build_agent(build_model(config), AgentConfig(permissions=[DENY_SECRETS]))
    require_granted("write_file", compiled_tool_names(agent), "the compiled graph")
    result = run_turn(
        agent,
        f"Use write_file to write the text 'hello' to {DENIED_PREFIX}/keys.txt. "
        f"Then tell me whether it succeeded.",
        callbacks=callbacks,
    )
    tools = _tool_messages(result)
    denied = any("permission denied" in str(m.content).lower() for m in tools)
    attempted = any(m.name == "write_file" for m in tools)
    return CheckResult(
        "F5",
        "permission rules survive middleware replacement",
        denied,
        "write_file denied" if denied else f"NOT denied (attempted={attempted})",
    )


def check_reasoning_efforts_are_the_ones_the_router_takes(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F26 — `REASONING_EFFORTS` used to list six values the router documents.
    gpt-oss has three. A precondition that accepts a value the request is
    guaranteed to 400 on is worse than no precondition, so both halves are
    checked: every value we allow works, and a value we forbid really fails."""
    accepted: list[str] = []
    for effort in sorted(REASONING_EFFORTS):
        model = build_model(
            ModelConfig(api_key=config.api_key, model=config.model, reasoning_effort=effort)
        )
        model.bind(max_tokens=REASONING_PROBE_TOKENS).invoke(
            "Reply with the digit 1.", config={"callbacks": callbacks}
        )
        accepted.append(effort)

    # The discriminator. Without it this check passes on a router that accepts
    # anything, and the narrowing it exists to defend would be unfalsifiable.
    rejected = False
    try:
        build_model(ModelConfig(api_key=config.api_key, model=config.model)).bind(
            reasoning_effort=REJECTED_REASONING_EFFORT, max_tokens=REASONING_PROBE_TOKENS
        ).invoke("Reply with the digit 1.", config={"callbacks": callbacks})
    # Any refusal is the evidence; which exception the provider raises is its own
    # business and pinning it would make this check about the SDK instead.
    except Exception:
        rejected = True

    return CheckResult(
        "F26",
        "reasoning_effort set matches what the router accepts",
        sorted(accepted) == sorted(REASONING_EFFORTS) and rejected,
        f"accepted {accepted}; {REJECTED_REASONING_EFFORT!r} rejected={rejected}",
    )


def check_no_reasoning_content_comes_back(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F27 — gpt-oss ships an unsupervised chain of thought, and OpenAI's own
    guidance is that it may hold content the final answer was told to leave out.

    Everything this harness records is a sink: `logs/*.jsonl`, LangSmith, W&B.
    Today the router returns reasoning as a *token count* and nothing else, so
    there is no CoT to leak — a fact worth a check rather than an assumption,
    because the day a provider starts returning the text, three sinks start
    storing it and nothing else would say so.
    """
    reply = build_model(
        ModelConfig(api_key=config.api_key, model=config.model, reasoning_effort="high")
    ).invoke("Think it through, then answer in one word: 2 + 2?", config={"callbacks": callbacks})

    extra = reply.additional_kwargs or {}
    leaked = sorted(k for k in COT_CONTENT_KEYS if extra.get(k))
    usage: dict[str, Any] = dict(reply.usage_metadata or {})
    counted = (usage.get("output_token_details") or {}).get("reasoning")
    # A run that did no reasoning at all would find no content either, and pass
    # for the wrong reason.
    require(counted, "the model reported no reasoning tokens; absence of content proves nothing")

    return CheckResult(
        "F27",
        "reasoning arrives as a count, never as text",
        not leaked,
        f"{counted} reasoning tokens, content keys present: {leaked or 'none'}",
    )


def check_every_provider_serves_the_context_we_assume(
    config: ModelConfig, _callbacks: list[BaseCallbackHandler]
) -> CheckResult:
    """F25 — gpt-oss natively supports 128k, but the router picks among
    providers and a provider serves what it serves. One `/v1/models` read, no
    inference: a prompt sized for the largest advertised window is a prompt that
    fails on whichever provider advertises less.

    The floor is `capabilities.CONTEXT_WINDOW_TOKENS`, which is no longer only a
    reporting number: `COMPACTION_TRIGGER_TOKENS` is sized against it, so a
    provider dropping below it moves a bound rather than a printed figure.

    Asserted on the providers that state a length, and reported for those that
    do not. An unstated window is a real unknown — the mitigation is pinning
    `:provider`, not a check that can never go green — while a *stated* window
    dropping below the floor is the thing that would actually truncate a run.
    """
    entry = next((m for m in _router_models(config) if m.get("id") == config.model), None)
    # An explicit raise rather than `require()`: this also narrows, and neither
    # type checker can follow a narrowing through a helper call (F10).
    if entry is None:
        raise CheckFailed(f"the router does not list {config.model}; the check has no subject")

    lengths = {
        str(p.get("provider")): p.get("context_length")
        for p in entry.get("providers", [])
        if isinstance(p, dict)
    }
    require(lengths, f"the router lists no providers for {config.model}")
    stated = {name: n for name, n in lengths.items() if isinstance(n, int)}
    # Without this the check passes by measuring nothing on the day the router
    # stops publishing context lengths at all.
    require(stated, f"no provider states a context length for {config.model}: {sorted(lengths)}")
    shortest = min(stated.values())
    short = sorted(name for name, n in stated.items() if n < CONTEXT_WINDOW_TOKENS)
    unstated = sorted(name for name in lengths if name not in stated)

    return CheckResult(
        "F25",
        "every provider that states a context window meets our floor",
        not short,
        f"shortest {shortest} across {len(stated)} providers (floor {CONTEXT_WINDOW_TOKENS}); "
        f"below floor: {short or 'none'}; unstated (pin :provider to remove the unknown): "
        f"{unstated or 'none'}",
    )


CHECKS: tuple[Callable[[ModelConfig, list[BaseCallbackHandler]], CheckResult], ...] = (
    check_chat_completions_endpoint,
    check_token_cap_reaches_the_router,
    check_shell_tool_withheld,
    check_filesystem_tools_still_work,
    check_permissions_are_enforced,
    check_reasoning_efforts_are_the_ones_the_router_takes,
    check_no_reasoning_content_comes_back,
    check_every_provider_serves_the_context_we_assume,
)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _single_turn(config: ModelConfig, prompt: str, callbacks: list[BaseCallbackHandler]) -> int:
    agent = build_agent(build_model(config))
    result = run_turn(agent, prompt, callbacks=callbacks)
    if result.paused:
        # Not reachable with today's configuration — nothing here passes an
        # interrupt-mode rule — but a pause printed as a reply would be a
        # half-finished turn reported as a whole one.
        for request in result.action_requests:
            print(f"paused: {request.get('name')} awaiting approval {request.get('args')}")
        print(
            "error: turn paused for approval; resume_turn() is not wired to a CLI",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED
    reply = result[-1]
    require(reply.type != "human", f"last message is still our own turn: {reply.type}")
    print(f"reply:  {reply.text}")

    # A turn can finish and still not have done what it says. Both call limits
    # run `exit_behavior="continue"`, so an exceeded call comes back as an error
    # `ToolMessage` and the model answers over the top of it — measured
    # 2026-09-18 as twenty-four writes executed, six blocked, and a reply
    # reading "All done! I wrote every file." The reply is still printed,
    # because it is what the agent said; the exit code stops that from being
    # the only thing a caller reads.
    failed = result.failed_tool_calls
    if failed:
        for message in failed:
            print(f"failed: {message.name} — {str(message.content)[:120]}", file=sys.stderr)
        print(
            f"error: {len(failed)} tool call(s) did not run, so the reply above is not "
            f"backed by the work it describes",
            file=sys.stderr,
        )
        return EXIT_CHECK_FAILED

    # F36: an empty reply is not always a short one. `finish_reason == "length"`
    # with no text and no tool calls means the cap landed inside gpt-oss's
    # reasoning channel and the final channel never opened — the answer did not
    # start, so nothing above was truncated. A report, not an error: the turn
    # still finished and spent no other bound, so the exit code stays 0.
    if not result.answered:
        print(
            "note: the reply above is empty because the turn was cut off before its answer "
            "began, not because the model had nothing to say; raise the token cap",
            file=sys.stderr,
        )
    return 0


def _run_checks(config: ModelConfig, callbacks: list[BaseCallbackHandler]) -> int:
    print(f"tools:  {sorted(DEFAULT_FILESYSTEM_TOOLS)} (+ task)\n")

    results: list[CheckResult] = []
    for check in CHECKS:
        # An operating error — the router is down, a provider rejects the
        # request — is a failed check, not a crashed program. A CheckFailed is
        # a bug in our own contracts and must still propagate.
        try:
            result = check(config, callbacks)
        except CheckFailed:
            raise
        except Exception as exc:
            result = CheckResult("??", check.__name__, False, f"{type(exc).__name__}: {exc}")
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{mark}] {result.finding}  {result.name}\n         {result.detail}")

    require(len(results) == len(CHECKS), "a check produced no result")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return EXIT_CHECK_FAILED if failed else 0


def main() -> int:
    """Run the live checks, or a single turn when given a prompt."""
    load_dotenv()

    # Missing configuration is an operating error: report it and exit, rather
    # than letting a traceback imply the code is broken.
    try:
        config = ModelConfig.from_env()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISCONFIGURED

    active = _activate_tracing()
    log_path = run_log_path()

    prompt = " ".join(sys.argv[1:]).strip()
    print(f"model:  {config.model}")
    print(f"tracing: {', '.join(active) if active else 'none'}")
    print(f"log:    {log_path}")

    with mirror_to_file(log_path) as mirror:
        callbacks: list[BaseCallbackHandler] = [mirror]
        # A run that outlived its wall clock is an operating error: the router
        # or a provider was slow. Report it like a missing token rather than
        # letting a traceback imply the code is broken. `_run_checks` already
        # converts one into a failed check, so this covers the prompt path.
        try:
            if prompt:
                print(f"prompt: {prompt}\n")
                exit_code = _single_turn(config, prompt, callbacks)
            else:
                exit_code = _run_checks(config, callbacks)
        except (
            DeadlineExceeded,
            ResumeLimitExceeded,
            StepLimitExceeded,
            TokenLimitExceeded,
        ) as exc:
            # Every bound in `RunBounds`, reported the same way. Before
            # `StepLimitExceeded` existed the step limit escaped as langgraph's
            # `GraphRecursionError` and printed a traceback, so the wall clock
            # was a handled ceiling and the step count was a crash (F24).
            # `ResumeLimitExceeded` joined them with the bound on how many times
            # one paused turn may be resumed, and `TokenLimitExceeded` with the
            # bound on what the turn costs.
            print(f"error: {exc}", file=sys.stderr)
            exit_code = EXIT_CHECK_FAILED

    # An empty mirror and a quiet run look identical on disk. This is what
    # separates "nothing happened" from "the callbacks were never attached".
    require(mirror.records > 0, f"the mirror wrote nothing to {log_path}; callbacks are not wired")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
