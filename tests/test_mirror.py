"""Offline tests for the JSONL mirror.

The handler takes a stream, so every test here drives it with StringIO and the
filesystem is never touched.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from my_agent.mirror import (
    DEFAULT_LOG_DIR,
    MAX_FIELD_CHARS,
    JsonlMirror,
    mirror_to_file,
    run_log_path,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000000001")
PARENT_ID = UUID("00000000-0000-4000-8000-000000000002")


def records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def mirror(stream: io.StringIO) -> JsonlMirror:
    return JsonlMirror(stream)


def test_chain_start_writes_one_record(mirror: JsonlMirror, stream: io.StringIO) -> None:
    mirror.on_chain_start({"name": "agent"}, {"messages": []}, run_id=RUN_ID)
    written = records(stream)
    assert len(written) == 1
    assert written[0]["event"] == "chain_start"
    assert written[0]["name"] == "agent"
    assert written[0]["run_id"] == str(RUN_ID)
    assert written[0]["parent_run_id"] is None


def test_parent_run_id_is_recorded(mirror: JsonlMirror, stream: io.StringIO) -> None:
    mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID, parent_run_id=PARENT_ID)
    assert records(stream)[0]["parent_run_id"] == str(PARENT_ID)


def test_tool_calls_are_mirrored(mirror: JsonlMirror, stream: io.StringIO) -> None:
    mirror.on_tool_start({"name": "write_file"}, '{"path": "/notes.txt"}', run_id=RUN_ID)
    mirror.on_tool_end("wrote 5 bytes", run_id=RUN_ID)
    written = records(stream)
    assert [r["event"] for r in written] == ["tool_start", "tool_end"]
    assert written[0]["name"] == "write_file"
    assert written[1]["output"] == "wrote 5 bytes"


def test_messages_are_mirrored_with_their_roles(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    mirror.on_chat_model_start({"name": "ChatOpenAI"}, [[HumanMessage("ping")]], run_id=RUN_ID)
    written = records(stream)[0]
    assert written["event"] == "chat_model_start"
    assert written["messages"] == [{"type": "human", "text": "ping"}]


def test_llm_end_records_output_and_usage(mirror: JsonlMirror, stream: io.StringIO) -> None:
    message = AIMessage(
        content="pong",
        usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])
    mirror.on_llm_end(result, run_id=RUN_ID)
    written = records(stream)[0]
    assert written["outputs"] == ["pong"]
    assert written["usage"]["total_tokens"] == 4


def test_errors_are_mirrored(mirror: JsonlMirror, stream: io.StringIO) -> None:
    mirror.on_tool_error(ValueError("permission denied"), run_id=RUN_ID)
    written = records(stream)[0]
    assert written["event"] == "tool_error"
    assert written["error_type"] == "ValueError"
    assert "permission denied" in written["error"]


def test_record_order_is_event_order(mirror: JsonlMirror, stream: io.StringIO) -> None:
    for index in range(10):
        mirror.on_chain_start({"name": f"step-{index}"}, {}, run_id=RUN_ID)
    assert [r["name"] for r in records(stream)] == [f"step-{i}" for i in range(10)]


def test_long_values_are_truncated_and_marked(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    mirror.on_tool_end("x" * (MAX_FIELD_CHARS * 2), run_id=RUN_ID)
    written = records(stream)[0]
    assert written["truncated"] is True
    assert len(written["output"]) == MAX_FIELD_CHARS


def test_short_values_are_not_marked_truncated(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    mirror.on_tool_end("small", run_id=RUN_ID)
    assert "truncated" not in records(stream)[0]


def test_unserialisable_payloads_do_not_raise(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    mirror.on_tool_end(object(), run_id=RUN_ID)
    assert len(records(stream)) == 1


def test_the_serialized_blob_is_never_written(
    mirror: JsonlMirror, stream: io.StringIO
) -> None:
    """It carries the model's constructor kwargs, credentials included."""
    mirror.on_chat_model_start(
        {"name": "ChatOpenAI", "kwargs": {"openai_api_key": "hf_super_secret"}},
        [[HumanMessage("ping")]],
        run_id=RUN_ID,
    )
    assert "hf_super_secret" not in stream.getvalue()


def test_every_record_is_flushed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed run is the one you most want the file for."""
    flushes: list[int] = []
    stream = io.StringIO()
    monkeypatch.setattr(stream, "flush", lambda: flushes.append(1))
    mirror = JsonlMirror(stream)
    mirror.on_chain_start({"name": "a"}, {}, run_id=RUN_ID)
    mirror.on_chain_end({}, run_id=RUN_ID)
    assert len(flushes) == 2


def test_records_are_counted(mirror: JsonlMirror) -> None:
    assert mirror.records == 0
    mirror.on_chain_start({"name": "a"}, {}, run_id=uuid4())
    assert mirror.records == 1


def test_handler_flags_are_pinned(mirror: JsonlMirror) -> None:
    """Defaults are False for both; each True is a deliberate decision."""
    assert mirror.run_inline is True
    assert mirror.raise_error is True


FIXED_NOW = datetime(2026, 9, 17, 14, 30, 5, tzinfo=UTC)


def test_run_log_path_is_deterministic_when_time_and_id_are_injected() -> None:
    path = run_log_path(Path("logs"), now=FIXED_NOW, run_id="abc123")
    assert path == Path("logs/20260917T143005Z-abc123.jsonl")


def test_run_log_path_defaults_to_the_log_directory() -> None:
    assert run_log_path(now=FIXED_NOW, run_id="abc123").parent == DEFAULT_LOG_DIR


def test_run_log_path_rejects_a_naive_timestamp() -> None:
    """A naive stamp is ambiguous, and these filenames sort by time."""
    with pytest.raises(AssertionError, match="timezone"):
        run_log_path(now=datetime(2026, 9, 17, 14, 30, 5), run_id="abc123")


def test_run_log_path_normalises_to_utc() -> None:
    eastern = timezone(timedelta(hours=-4))
    path = run_log_path(now=FIXED_NOW.astimezone(eastern), run_id="abc123")
    assert path.name == "20260917T143005Z-abc123.jsonl"


def test_run_log_paths_differ_between_runs() -> None:
    assert run_log_path(now=FIXED_NOW) != run_log_path(now=FIXED_NOW)


def test_run_log_path_rejects_a_traversal_run_id() -> None:
    """The path is built by string interpolation: an unvalidated run_id can escape
    the log directory entirely (`../../etc/passwd`)."""
    with pytest.raises(AssertionError, match="separator"):
        run_log_path(now=FIXED_NOW, run_id="../../etc/passwd")


def test_run_log_path_rejects_a_run_id_with_a_path_separator() -> None:
    with pytest.raises(AssertionError, match="separator"):
        run_log_path(now=FIXED_NOW, run_id="a/b")


def test_run_log_path_rejects_a_run_id_with_a_backslash() -> None:
    with pytest.raises(AssertionError, match="separator"):
        run_log_path(now=FIXED_NOW, run_id="a\\b")


def test_run_log_path_rejects_a_bare_traversal_token() -> None:
    with pytest.raises(AssertionError, match="traversal"):
        run_log_path(now=FIXED_NOW, run_id="..")


def test_run_log_path_accepts_a_normal_run_id() -> None:
    """Guard against over-tightening: an ordinary hex token still works."""
    path = run_log_path(now=FIXED_NOW, run_id="deadbeef")
    assert path == Path("logs/20260917T143005Z-deadbeef.jsonl")


def test_mirror_to_file_writes_a_readable_run_log(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    with mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
    assert json.loads(path.read_text().strip())["name"] == "agent"


def test_mirror_to_file_creates_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "deeper" / "logs" / "run.jsonl"
    with mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
    assert path.exists()


def test_mirror_to_file_closes_the_file_even_when_the_run_raises(tmp_path: Path) -> None:
    """The crashed run is the one whose log has to survive."""
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError), mirror_to_file(path) as mirror:
        mirror.on_chain_start({"name": "agent"}, {}, run_id=RUN_ID)
        raise RuntimeError("the agent exploded")
    assert json.loads(path.read_text().strip())["name"] == "agent"
