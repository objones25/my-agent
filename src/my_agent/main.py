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

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deepagents import FilesystemPermission
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    compiled_tool_names,
)
from my_agent.mirror import mirror_to_file, run_log_path
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed, require
from my_agent.tracing import available_backends

EXIT_MISCONFIGURED = 2
EXIT_CHECK_FAILED = 1

RECURSION_LIMIT = 25
"""Bound every agent run. An agent that loops forever is the worst failure here."""

TOKEN_CAP = 24
"""Small enough that an ignored cap is unmistakable against an uncapped reply."""

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


def _tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if m.type == "tool"]


def _run(
    agent: object, prompt: str, callbacks: list[BaseCallbackHandler]
) -> list[BaseMessage]:
    """One bounded agent turn, mirrored."""
    result = agent.invoke(  # type: ignore[attr-defined]
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": RECURSION_LIMIT, "callbacks": callbacks},
    )
    require("messages" in result, f"agent returned no messages key: {sorted(result)}")
    messages: list[BaseMessage] = result["messages"]
    require(len(messages) > 1, "agent added no messages of its own")
    return messages


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
    agent = build_agent(build_model(config))
    bound = compiled_tool_names(agent)
    require(bound != frozenset(), "no tools bound at all; the absence check would be vacuous")
    if SHELL_TOOL_NAME in bound:
        return CheckResult("F4", "shell tool withheld", False, f"bound: {sorted(bound)}")

    messages = _run(
        agent, "Run the shell command `echo hello` and show me the output.", callbacks
    )
    called = {m.name for m in _tool_messages(messages) if m.name is not None}
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
    require("write_file" in compiled_tool_names(agent), "write_file is not bound; check is vacuous")
    messages = _run(
        agent,
        f"Use write_file to write the text 'pong' to {ALLOWED_PATH}, then read it back.",
        callbacks,
    )
    tools = _tool_messages(messages)
    wrote = any(
        m.name == "write_file" and "permission denied" not in str(m.content).lower()
        for m in tools
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
    require("write_file" in compiled_tool_names(agent), "write_file is not bound; check is vacuous")
    messages = _run(
        agent,
        f"Use write_file to write the text 'hello' to {DENIED_PREFIX}/keys.txt. "
        f"Then tell me whether it succeeded.",
        callbacks,
    )
    tools = _tool_messages(messages)
    denied = any("permission denied" in str(m.content).lower() for m in tools)
    attempted = any(m.name == "write_file" for m in tools)
    return CheckResult(
        "F5",
        "permission rules survive middleware replacement",
        denied,
        "write_file denied" if denied else f"NOT denied (attempted={attempted})",
    )


CHECKS: tuple[Callable[[ModelConfig, list[BaseCallbackHandler]], CheckResult], ...] = (
    check_chat_completions_endpoint,
    check_token_cap_reaches_the_router,
    check_shell_tool_withheld,
    check_filesystem_tools_still_work,
    check_permissions_are_enforced,
)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _single_turn(config: ModelConfig, prompt: str, callbacks: list[BaseCallbackHandler]) -> int:
    agent = build_agent(build_model(config))
    messages = _run(agent, prompt, callbacks)
    reply = messages[-1]
    require(reply.type != "human", f"last message is still our own turn: {reply.type}")
    print(f"reply:  {reply.text}")
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
        if prompt:
            print(f"prompt: {prompt}\n")
            exit_code = _single_turn(config, prompt, callbacks)
        else:
            exit_code = _run_checks(config, callbacks)

    # An empty mirror and a quiet run look identical on disk. This is what
    # separates "nothing happened" from "the callbacks were never attached".
    require(mirror.records > 0, f"the mirror wrote nothing to {log_path}; callbacks are not wired")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
