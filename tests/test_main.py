"""Offline tests for the smoke-test entry point and the deepagents defaults.

The live round trip is not tested here — that is what `uv run my-agent` is for.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from deepagents import FilesystemMiddleware, create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from my_agent import main as main_module
from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    compiled_tool_names,
)
from my_agent.main import EXIT_CHECK_FAILED, EXIT_MISCONFIGURED, main
from my_agent.model import ModelConfig, build_model
from my_agent.negative_space import CheckFailed
from my_agent.run import DeadlineExceeded, TurnResult

VALID_SECRET = SecretStr("hf_token_value")

# Verified against deepagents 0.7.15 by inspecting the compiled graph.
DEEPAGENTS_RAW_TOOLS = frozenset(
    {"delete", "edit_file", "execute", "glob", "grep", "ls", "read_file", "task", "write_file"}
)
OUR_TOOLS = DEEPAGENTS_RAW_TOOLS - {"execute"}


def _tool_names(agent: object) -> frozenset[str]:
    return compiled_tool_names(agent)  # type: ignore[arg-type]


def test_deepagents_raw_default_still_includes_shell_execution() -> None:
    """Pins what deepagents gives you with no opt-out, so an upstream change to
    the built-ins is a failing test rather than a silent shift. This is the
    behaviour `build_agent` deliberately overrides."""
    agent = create_deep_agent(model=build_model(ModelConfig(api_key=VALID_SECRET)))

    assert _tool_names(agent) == DEEPAGENTS_RAW_TOOLS


def test_our_agent_withholds_shell_execution() -> None:
    """Principle of least privilege: `execute` runs arbitrary shell commands and
    nothing here needs it, so it is off unless a caller opts back in."""
    agent = build_agent(build_model(ModelConfig(api_key=VALID_SECRET)))

    assert _tool_names(agent) == OUR_TOOLS
    assert SHELL_TOOL_NAME not in _tool_names(agent)


def test_shell_execution_can_be_restored_deliberately() -> None:
    """The opt-out is not a dead end: a caller-supplied FilesystemMiddleware
    replaces ours by name and takes ownership of the allowlist."""
    agent = build_agent(
        build_model(ModelConfig(api_key=VALID_SECRET)),
        AgentConfig(
            middleware=[FilesystemMiddleware(tools=[*DEFAULT_FILESYSTEM_TOOLS, "execute"])]
        ),
    )

    assert SHELL_TOOL_NAME in _tool_names(agent)


def test_write_todos_is_not_a_default_tool() -> None:
    """deepagents 0.7.15 ships no planning tool. `TodoListMiddleware` lives in
    langchain, not deepagents, and must be passed in explicitly."""
    agent = build_agent(build_model(ModelConfig(api_key=VALID_SECRET)))

    assert "write_todos" not in _tool_names(agent)


def test_todo_middleware_is_what_restores_planning() -> None:
    """Records how to get planning back if it is ever wanted: `TodoListMiddleware`
    from langchain, passed as `middleware`.

    `AgentConfig` has no `middleware` field yet, because nothing needs one —
    adding it would be one field with a default and no change to `build_agent`.
    Until then this asserts against `create_deep_agent` directly rather than
    inventing the field speculatively.
    """
    agent = create_deep_agent(
        model=build_model(ModelConfig(api_key=VALID_SECRET)),
        middleware=[TodoListMiddleware()],
    )

    assert "write_todos" in _tool_names(agent)


def test_main_exits_cleanly_when_the_token_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing token must produce a readable message and a non-zero exit, not
    a traceback that implies the code is broken."""
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["my-agent"])

    assert main() == EXIT_MISCONFIGURED

    captured = capsys.readouterr()
    assert "HF_TOKEN" in captured.err
    assert "Traceback" not in captured.err


def test_activate_tracing_reports_each_backend_that_turned_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activated: list[str] = []

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def activate(self) -> None:
            activated.append(self.name)

    monkeypatch.setattr(
        main_module, "available_backends", lambda: (FakeBackend("a"), FakeBackend("b"))
    )
    assert main_module._activate_tracing() == ("a", "b")
    assert activated == ["a", "b"]


def test_activate_tracing_survives_a_backend_that_cannot_reach_its_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A telemetry outage must not take the run down with it."""

    class Broken:
        name = "weave"

        def activate(self) -> None:
            raise ConnectionError("w&b unreachable")

    class Working:
        name = "langsmith"

        def activate(self) -> None:
            return None

    monkeypatch.setattr(main_module, "available_backends", lambda: (Broken(), Working()))
    assert main_module._activate_tracing() == ("langsmith",)
    assert "weave" in capsys.readouterr().err


def test_activate_tracing_lets_a_broken_contract_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """CheckFailed is a bug in our own code, not an operating error."""

    class Contradictory:
        name = "langsmith"

        def activate(self) -> None:
            raise CheckFailed("tracing reported off")

    monkeypatch.setattr(main_module, "available_backends", lambda: (Contradictory(),))
    with pytest.raises(CheckFailed):
        main_module._activate_tracing()


def test_activate_tracing_is_quiet_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "available_backends", tuple)
    assert main_module._activate_tracing() == ()


def test_main_reports_a_missed_deadline_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that outlives its wall clock is the outside world being slow, not a
    broken contract — so the CLI reports it the way it reports a missing token.

    The fake records one event before raising, because that is what a real
    timeout looks like: steps happened, then the budget ran out. It also keeps
    the mirror postcondition honest rather than sidestepping it.
    """
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.setenv("HF_TOKEN", "hf_token_value")
    monkeypatch.setattr(main_module, "available_backends", tuple)
    monkeypatch.setattr(main_module, "run_log_path", lambda: tmp_path / "run.jsonl")
    monkeypatch.setattr("sys.argv", ["my-agent", "ping"])

    def timed_out(
        config: object, prompt: str, callbacks: list[BaseCallbackHandler]
    ) -> int:
        callbacks[0].on_chain_end({}, run_id=uuid4())
        raise DeadlineExceeded("run exceeded its 600.0s deadline after 601.0s")

    monkeypatch.setattr(main_module, "_single_turn", timed_out)

    assert main() == EXIT_CHECK_FAILED

    captured = capsys.readouterr()
    assert "600.0s deadline" in captured.err
    assert "Traceback" not in captured.err


# --------------------------------------------------------------------------
# What the CLI says about a turn whose tools did not all run
# --------------------------------------------------------------------------


def _turn_with_a_blocked_tool_call() -> TurnResult:
    """A finished turn whose agent claims work a blocked tool call never did.

    The exact shape `capabilities.call_limits` produces: `exit_behavior` is
    `"continue"`, so the blocked call becomes an error `ToolMessage` and the
    model answers over the top of it.
    """
    return TurnResult(
        messages=[
            HumanMessage("write me thirty files"),
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "c1"}]),
            ToolMessage(
                content="Tool call limit exceeded.",
                tool_call_id="c1",
                name="write_file",
                status="error",
            ),
            AIMessage("All done! I wrote every file."),
        ]
    )


def _stub_turn(monkeypatch: pytest.MonkeyPatch, result: TurnResult) -> ModelConfig:
    """Replace everything `_single_turn` needs a network for, and hand back a
    config it can be called with. The turn is the subject; the model is not."""
    monkeypatch.setattr(main_module, "build_model", lambda _config: object())
    monkeypatch.setattr(main_module, "build_agent", lambda _model: object())
    monkeypatch.setattr(main_module, "run_turn", lambda *_a, **_k: result)
    return ModelConfig(api_key=SecretStr("hf_token_value"))


def test_a_single_turn_names_the_tool_calls_that_did_not_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gap this closes: the reply was printed and the exit code was 0 while
    the work it described had been blocked, and the only record was the mirror."""
    config = _stub_turn(monkeypatch, _turn_with_a_blocked_tool_call())

    exit_code = main_module._single_turn(config, "write me thirty files", [])

    captured = capsys.readouterr()
    assert exit_code == EXIT_CHECK_FAILED
    assert "write_file" in captured.err
    assert "All done!" in captured.out


def test_a_single_turn_whose_tools_all_ran_says_nothing_about_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The discriminator: an ordinary turn must not grow a warning."""
    config = _stub_turn(monkeypatch, TurnResult(messages=[HumanMessage("ping"), AIMessage("pong")]))

    exit_code = main_module._single_turn(config, "ping", [])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# What the CLI says about a turn cut off before its answer began (F36)
# --------------------------------------------------------------------------


def _turn_cut_off_before_the_answer_began() -> TurnResult:
    """The exact shape F36 records: `finish_reason == "length"` with no text
    and no tool calls, meaning the cap landed inside the reasoning channel and
    the final channel never opened."""
    return TurnResult(
        messages=[
            HumanMessage("count to 200"),
            AIMessage("", response_metadata={"finish_reason": "length"}),
        ]
    )


def test_a_single_turn_notes_when_the_reply_was_cut_off_before_it_began(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gap this closes: an empty reply printed with exit code 0 reads as
    "the model had nothing to say" when it actually means "the token cap never
    let the model start". Still exit 0 — no other bound was spent."""
    config = _stub_turn(monkeypatch, _turn_cut_off_before_the_answer_began())

    exit_code = main_module._single_turn(config, "count to 200", [])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cut off before its answer began" in captured.err
