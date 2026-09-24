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
import os
import sys
import urllib.request
from collections.abc import Callable, Mapping
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
    BoundExceeded,
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

LIVE_CHECK_REPEATS = 5
"""Attempts per check.

Three of the eight depend on the model *choosing* to call a tool, so at one
attempt each a flake and a regression are the same observation. Five is the
smallest count that tells them apart without making a weekly run expensive;
`LIVE_CHECK_REPEATS` in the environment overrides it for a developer debugging
one check. Uniform on purpose for now: which checks can actually vary is a
measurement this has not made yet, and assuming five of them are deterministic
would be the same untested assumption the repeats exist to remove.
"""

DENIED_PREFIX = "/secrets"
ALLOWED_PATH = "/notes/smoke.txt"
DENY_SECRETS = FilesystemPermission(
    operations=["write"], paths=[f"{DENIED_PREFIX}/**"], mode="deny"
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One attempt at one check."""

    finding: str
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """One check's verdict across every attempt.

    Two numbers, because they answer different questions. `pass_all` (pass^k)
    is what the exit code reads: these are invariants, and "the shell tool is
    withheld" is not a thing that may hold four times in five. `pass_any`
    (pass@k) is what tells a reader *which* failure they have — a check that
    never passed is broken, one that passed four times in five is
    non-deterministic, and at one attempt those two are indistinguishable.
    """

    finding: str
    name: str
    results: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        # Without this an empty tuple makes `pass_all` vacuously true: a check
        # that ran zero times would report as holding.
        require(
            self.results,
            f"{self.name}: an outcome with no attempts measured nothing",
        )

    @property
    def attempts(self) -> int:
        return len(self.results)

    @property
    def passes(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_any(self) -> bool:
        """pass@k — at least one attempt held."""
        return self.passes > 0

    @property
    def pass_all(self) -> bool:
        """pass^k — every attempt held. This is what the exit code reads."""
        return self.passes == self.attempts


def live_check_repeats(env: Mapping[str, str] | None = None) -> int:
    """How many times to run each check.

    Reads the environment as its own documented default, the way an injectable
    default argument reads anything else — the caller can always pass a mapping
    instead, which is what keeps this testable without setting a real variable.
    A bad value is an *operating* error: it came from outside.
    """
    source = os.environ if env is None else env
    raw = source.get("LIVE_CHECK_REPEATS")
    if raw is None:
        return LIVE_CHECK_REPEATS
    try:
        repeats = int(raw)
    except ValueError:
        raise ValueError(f"LIVE_CHECK_REPEATS must be an integer, got {raw!r}") from None
    if repeats < 1:
        raise ValueError(f"LIVE_CHECK_REPEATS must be at least 1, got {repeats}")
    return repeats


def _shell_withheld(*, called: frozenset[str], answered: bool) -> bool:
    """The live half of F4's check, with its discriminator.

    `execute` not appearing is only evidence if the turn actually ran to an
    answer. A turn cut off before its answer began called nothing *and* proves
    nothing, so the absence there is vacuous — the same failure mode the
    allowlist tests guard against offline, which this check had on the live
    path until it was given the second half.
    """
    return answered and SHELL_TOOL_NAME not in called


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
    called = frozenset(m.name for m in _tool_messages(result) if m.name is not None)
    return CheckResult(
        "F4",
        "shell tool withheld",
        _shell_withheld(called=called, answered=result.answered),
        f"unbound; answered={result.answered}; tools called: {sorted(called) or 'none'}",
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


ROUTING_POLICY_SUFFIXES = frozenset({"fastest", "cheapest", "preferred"})
"""Model-id suffixes that *select among* providers rather than naming one.

`openai/gpt-oss-120b:groq` pins a provider; `:fastest` picks one at request
time. Omitting the suffix is equivalent to `:fastest`, which is a routing
decision inherited rather than made -- and measured 2026-09-23 to be the one
that concentrates traffic on the lowest-latency providers, where the queues
fill.
"""


def _model_route(model_id: str) -> tuple[str, str | None]:
    """Split a router model id into `(catalogue id, pinned provider or None)`.

    **The catalogue lists the bare repo id.** A suffixed `config.model` matches
    nothing in `/v1/models`, so reading the catalogue with the configured id
    raised "the router does not list ..." the moment a provider was pinned --
    following F25's own recommendation broke the check that records F25.
    """
    repo, _, suffix = model_id.partition(":")
    if not suffix or suffix in ROUTING_POLICY_SUFFIXES:
        return repo, None
    return repo, suffix


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
    repo, pinned = _model_route(config.model)
    entry = next((m for m in _router_models(config) if m.get("id") == repo), None)
    # An explicit raise rather than `require()`: this also narrows, and neither
    # type checker can follow a narrowing through a helper call (F10).
    if entry is None:
        raise CheckFailed(f"the router does not list {repo}; the check has no subject")

    lengths = {
        str(p.get("provider")): p.get("context_length")
        for p in entry.get("providers", [])
        if isinstance(p, dict)
    }
    require(lengths, f"the router lists no providers for {repo}")
    if pinned is not None:
        # Narrowed, because a pinned run cannot be served by anyone else -- and
        # refused rather than narrowed to nothing, since a typo in the pin would
        # otherwise leave the check passing on an empty set.
        if pinned not in lengths:
            raise CheckFailed(
                f"the router does not serve {repo} via {pinned!r}; "
                f"available: {sorted(lengths)}"
            )
        lengths = {pinned: lengths[pinned]}
    stated = {name: n for name, n in lengths.items() if isinstance(n, int)}
    # Without this the check passes by measuring nothing on the day the router
    # stops publishing context lengths at all.
    require(stated, f"no provider states a context length for {repo}: {sorted(lengths)}")
    shortest = min(stated.values())
    short = sorted(name for name, n in stated.items() if n < CONTEXT_WINDOW_TOKENS)
    unstated = sorted(name for name in lengths if name not in stated)

    scope = f"pinned provider {pinned!r}" if pinned else f"{len(stated)} providers"
    return CheckResult(
        "F25",
        "every provider that states a context window meets our floor",
        not short,
        f"shortest {shortest} across {scope} (floor {CONTEXT_WINDOW_TOKENS}); "
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
            "began, not because the model had nothing to say; set REASONING_EFFORT=low or "
            "raise the provider's output cap",
            file=sys.stderr,
        )
    return 0


def _attempt(
    check: Callable[[ModelConfig, list[BaseCallbackHandler]], CheckResult],
    config: ModelConfig,
    callbacks: list[BaseCallbackHandler],
) -> CheckResult:
    """One attempt, with operating errors folded into the verdict.

    The router being down or a provider rejecting the request is a failed
    attempt, not a crashed program — and now that a check runs k times, it must
    cost *that attempt* rather than the run, or one slow response would hide
    every later attempt's evidence. A `CheckFailed` is a bug in our own
    contracts and must still propagate.
    """
    try:
        return check(config, callbacks)
    except CheckFailed:
        raise
    except Exception as exc:
        return CheckResult("??", check.__name__, False, f"{type(exc).__name__}: {exc}")


def _print_outcome(outcome: CheckOutcome) -> None:
    """One block per check. A run where everything held stays as short as it
    was at one attempt, because only failing attempts print their detail —
    eight checks times five attempts of detail is a wall nobody reads."""
    print(f"  [{outcome.passes}/{outcome.attempts}] {outcome.finding}  {outcome.name}")
    if outcome.pass_all:
        print(f"         {outcome.results[0].detail}")
        return
    if outcome.pass_any:
        print(f"         pass@{outcome.attempts} yes, pass^{outcome.attempts} NO — flaky")
    else:
        print(f"         failed every one of {outcome.attempts} attempts")
    for number, result in enumerate(outcome.results, start=1):
        if not result.passed:
            print(f"         attempt {number}: {result.detail}")


def _run_checks(
    config: ModelConfig, callbacks: list[BaseCallbackHandler], repeats: int
) -> int:
    """Every check, `repeats` times each, scored pass^k.

    pass^k rather than pass@k because these are invariants: a check that held
    four times in five did not hold. pass@k is printed anyway, because it is
    the difference between "this is broken" and "this is non-deterministic",
    and that difference is invisible at one attempt.
    """
    require(repeats >= 1, f"a check run needs at least one attempt, got repeats={repeats}")
    print(f"tools:  {sorted(DEFAULT_FILESYSTEM_TOOLS)} (+ task)")
    print(f"repeats: {repeats} per check (exit code reads pass^{repeats})\n")

    outcomes: list[CheckOutcome] = []
    for check in CHECKS:
        results = tuple(_attempt(check, config, callbacks) for _ in range(repeats))
        outcome = CheckOutcome(results[0].finding, results[0].name, results)
        outcomes.append(outcome)
        _print_outcome(outcome)

    require(len(outcomes) == len(CHECKS), "a check produced no outcome")
    held = [o for o in outcomes if o.pass_all]
    flaky = [o for o in outcomes if o.pass_any and not o.pass_all]
    print(f"\n{len(held)}/{len(outcomes)} checks passed (pass^{repeats})")
    if flaky:
        # Named rather than merely counted: a flaky check and a broken one both
        # exit non-zero, and the next reader needs to know which they have.
        print(f"flaky (passed at least once, not every time): {[o.finding for o in flaky]}")
    return EXIT_CHECK_FAILED if len(held) != len(outcomes) else 0


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
                exit_code = _run_checks(config, callbacks, live_check_repeats())
        except BoundExceeded as exc:
            # Every bound in `RunBounds`, reported the same way, and any bound
            # added later without editing this line. Before `StepLimitExceeded`
            # existed the step limit escaped as langgraph's `GraphRecursionError`
            # and printed a traceback (F24). A hand-listed tuple had the same
            # failure waiting for the next bound. `CheckFailed` is not a
            # `BoundExceeded` and still crashes.
            print(f"error: {exc}", file=sys.stderr)
            exit_code = EXIT_CHECK_FAILED

    # An empty mirror and a quiet run look identical on disk. This is what
    # separates "nothing happened" from "the callbacks were never attached".
    require(mirror.records > 0, f"the mirror wrote nothing to {log_path}; callbacks are not wired")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
