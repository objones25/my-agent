"""Offline tests for the smoke-test entry point and the deepagents defaults.

The live round trip is not tested here — that is what `uv run my-agent` is for.
"""

from __future__ import annotations

import pytest
from deepagents import FilesystemMiddleware, create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from pydantic import SecretStr

from my_agent import main as main_module
from my_agent.agent import AgentConfig, build_agent
from my_agent.capabilities import (
    DEFAULT_FILESYSTEM_TOOLS,
    SHELL_TOOL_NAME,
    compiled_tool_names,
)
from my_agent.main import EXIT_MISCONFIGURED, main
from my_agent.model import ModelConfig, build_model

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
