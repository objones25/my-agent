"""A local, always-on JSONL mirror of everything the agent does.

LangSmith and Weave answer "what did the model do?" for a vendor that was
reachable and configured. This answers "what happened in this process?" with no
network, no API key, and no account — one file per run, one JSON object per
line, flushed as it goes, because the runs most worth reading back are the ones
that crashed.

It is a plain object that writes to a stream. Nothing here is global, nothing
here is installed, and nothing here knows a tracing backend exists.

`*_end` and `*_error` records carry no `name` field — only `*_start` records do
— so reading a component's name back requires joining an end or error record to
its start record by `run_id`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, override
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from my_agent.negative_space import require

__all__ = [
    "DEFAULT_LOG_DIR",
    "MAX_FIELD_CHARS",
    "JsonlMirror",
    "mirror_to_file",
    "run_log_path",
]

MAX_FIELD_CHARS = 4000
"""Per-value bound. A `read_file` on something large would otherwise put
megabytes on one line and make the file unreadable exactly when it matters."""

DEFAULT_LOG_DIR = Path("logs")


def _clip(value: Any) -> tuple[Any, bool]:
    """A JSON-safe value, bounded, and whether the bound bit.

    The round trip through `default=str` is what makes `raise_error = True` safe
    for most payloads: a `UUID`, a `BaseMessage`, or a tool returning an
    arbitrary object degrades to a string instead of raising inside a callback.

    `default=str` alone is not enough, though: `json.dumps` applies `default=`
    to *values* only, never to dict keys, and it does not handle circular
    references. A non-str-keyed dict (`{(1, 2): "x"}`) still raises `TypeError`,
    and a self-referential structure still raises `ValueError`, straight out of
    this function and into a `raise_error = True` handler — see docs/findings.md
    F15. `allow_nan=False` is folded into the same try: `json.dumps` otherwise
    emits a bare `NaN`/`Infinity` token that Python reads back but `jq`, Go and
    Rust do not, and this file's contract is one valid JSON object per line; a
    non-finite float now raises `ValueError` too and is caught by the same
    `except`. Whatever cannot round-trip degrades to `repr(value)` instead of
    propagating, so the claim in `raise_error`'s docstring and F12 stays true.
    """
    try:
        safe = json.loads(json.dumps(value, default=str, allow_nan=False))
    except (TypeError, ValueError):
        safe = repr(value)
    text = safe if isinstance(safe, str) else json.dumps(safe)
    if len(text) <= MAX_FIELD_CHARS:
        return safe, False
    return text[:MAX_FIELD_CHARS], True


def _component_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """The component's name, and *only* the name.

    `serialized` carries the component's constructor kwargs, which for a chat
    model include its credentials. Rather than depend on LangChain redacting
    them correctly in every version, none of that dict is ever written.
    """
    explicit = kwargs.get("name")
    if isinstance(explicit, str) and explicit:
        return explicit
    if serialized:
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        path = serialized.get("id")
        if isinstance(path, list) and path:
            return str(path[-1])
    return "unknown"


class JsonlMirror(BaseCallbackHandler):
    """Writes one JSON object per line for every event LangChain reports."""

    run_inline = True
    """Keep events off the thread pool: record order is then event order."""

    raise_error = True
    """The default (False) makes LangChain swallow exceptions raised in here, and
    a mirror that silently stopped mirroring is worse than no mirror. Safe only
    because `_clip` means the body cannot raise on data."""

    def __init__(self, stream: TextIO) -> None:
        require(hasattr(stream, "write"), f"stream must be writable, got {type(stream).__name__}")
        self._stream = stream
        self._records = 0

    @property
    def records(self) -> int:
        """How many records were written. `main` asserts this is non-zero: an
        empty file and a quiet run are otherwise indistinguishable."""
        return self._records

    def _write(
        self, event: str, run_id: UUID, parent_run_id: UUID | None, **payload: Any
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
        }
        truncated = False
        for key, value in payload.items():
            record[key], hit = _clip(value)
            truncated = truncated or hit
        if truncated:
            record["truncated"] = True

        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
        self._records += 1

    # -- chains (graph steps) ------------------------------------------------

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            inputs=inputs,
        )

    @override
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write("chain_end", run_id, parent_run_id, outputs=outputs)

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )

    # -- tools ---------------------------------------------------------------

    @override
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            input=inputs if inputs is not None else input_str,
        )

    @override
    def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self._write("tool_end", run_id, parent_run_id, output=output)

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )

    # -- model calls ---------------------------------------------------------

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chat_model_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            messages=[
                {"type": message.type, "text": message.text}
                for batch in messages
                for message in batch
            ],
        )

    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        outputs: list[str] = []
        usage: dict[str, Any] | None = None
        for batch in response.generations:
            for generation in batch:
                outputs.append(generation.text)
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    usage = dict(metadata)
        self._write("llm_end", run_id, parent_run_id, outputs=outputs, usage=usage)

    @override
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "llm_error",
            run_id,
            parent_run_id,
            error_type=type(error).__name__,
            error=str(error),
        )


# -- run-file lifecycle -------------------------------------------------------


def run_log_path(
    directory: Path = DEFAULT_LOG_DIR,
    *,
    now: datetime | None = None,
    run_id: str | None = None,
) -> Path:
    """Where this run's mirror goes: `<dir>/<utc-stamp>-<short-id>.jsonl`.

    `now` and `run_id` are injectable so the name is deterministic under test.
    """
    moment = datetime.now(UTC) if now is None else now
    require(
        moment.tzinfo is not None,
        "now must be timezone-aware; these filenames sort by time and a naive "
        "stamp is ambiguous",
    )
    token = uuid.uuid4().hex[:8] if run_id is None else run_id
    require(token != "", "run_id must not be empty")
    require(
        "/" not in token,
        f"run_id must not contain a path separator (got {token!r}): "
        "it becomes part of a filename, not a subdirectory",
    )
    require(
        "\\" not in token,
        f"run_id must not contain a path separator (got {token!r}): "
        "it becomes part of a filename, not a subdirectory",
    )
    require(
        token not in (".", ".."),
        f"run_id must not be a directory-traversal token (got {token!r})",
    )
    return directory / f"{moment.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{token}.jsonl"


@contextmanager
def mirror_to_file(path: Path) -> Iterator[JsonlMirror]:
    """A mirror writing to `path`, closed on the way out — exception or not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yield JsonlMirror(stream)
