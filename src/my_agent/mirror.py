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
from collections.abc import Iterator, Sequence
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


_RESPONSE_METADATA_KEYS = (
    "finish_reason",
    "model_name",
    "model_provider",
    "system_fingerprint",
    "service_tier",
)
"""Response fields worth keeping, named rather than copied wholesale.

`finish_reason` separates a complete answer from one that hit a token cap, and
the router picks a provider per request — so without `model_provider` a log
cannot say who actually served a call.
"""


def _tool_call_summary(call: Any) -> dict[str, Any]:
    """A tool call as name, arguments and id.

    Tool calls are the half of an assistant turn that `.text` does not carry: a
    turn that requested a tool has empty text, and without this a tool-calling
    turn and an empty reply look identical in the log.
    """
    if not isinstance(call, dict):
        return {"raw": call}
    return {"name": call.get("name"), "args": call.get("args"), "id": call.get("id")}


def _message_summary(message: BaseMessage) -> dict[str, Any]:
    """Role, text, and any tool calls the message asked for."""
    summary: dict[str, Any] = {"type": message.type, "text": message.text}
    calls = getattr(message, "tool_calls", None)
    if calls:
        summary["tool_calls"] = [_tool_call_summary(c) for c in calls]
    return summary


def _invocation_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """What was actually sent: model, temperature, caps, reasoning effort, tools.

    Unlike `serialized` — which carries constructor kwargs and is never written —
    `ChatOpenAI._get_invocation_params()` was verified on 2026-09-17 to contain no
    credential: `model`, `model_name`, `temperature`, `stream`, `stop`, `_type`.

    Tool *definitions* are reduced to their names. The schemas are static, and
    repeating them on every model call would dominate the file.

    **This is the langchain-level request, not the HTTP body.** Callbacks only
    ever receive `invocation_params`, which is read before
    `_get_request_payload` renames anything — so a bound token cap is recorded
    as `max_tokens: 24` while the wire actually carries
    `max_completion_tokens: 24` (F2). Verified 2026-09-17. For every other
    parameter the two agree; this is the one rename langchain performs.
    """
    params = kwargs.get("invocation_params")
    if not isinstance(params, dict):
        return {}

    summary = {k: v for k, v in params.items() if k != "tools"}
    tools = params.get("tools")
    if isinstance(tools, list):
        summary["tools"] = [_tool_definition_name(t) for t in tools]
    return summary


def _json_bytes(value: Any) -> int:
    """Serialized size of `value`, or 0 for anything that will not serialize.

    A size is a diagnostic, never a reason to fail a run, so an exotic object in
    a tool schema costs its own bytes rather than the record.
    """
    try:
        return len(json.dumps(value, default=str).encode())
    except (TypeError, ValueError):  # pragma: no cover - default=str covers the known cases
        return 0


def _request_size(params: dict[str, Any], messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Roughly how many bytes this call is about to send, and where they go.

    **The number that explains the bill.** A two-line prompt against this agent
    costs ~2,090 input tokens: the conversation is twelve of them and the rest
    is tool schemas, sent on every turn whether or not a tool is used (F30).
    Nothing in the log said so, and a per-call `usage` figure cannot — it
    reports a total, not a breakdown.

    An estimate, and the error is known: this serializes the langchain-level
    request, which measured 10,593 bytes against 10,508 on the wire for the same
    call — **+0.8%**, all of it `json.dumps` whitespace the HTTP body omits.
    Close enough to act on, and cheaper than an HTTP hook that would have to see
    the whole conversation to count it.
    """
    tools = params.get("tools")
    tool_bytes: dict[str, int] = {}
    if isinstance(tools, list):
        for tool in tools:
            name = _tool_definition_name(tool)
            tool_bytes[str(name)] = _json_bytes(tool)
    messages_bytes = sum(len(message.text.encode()) for message in messages)
    tools_bytes = sum(tool_bytes.values())
    return {
        "total_bytes": tools_bytes + messages_bytes,
        "tools_bytes": tools_bytes,
        "messages_bytes": messages_bytes,
        "tool_count": len(tool_bytes),
        "tool_bytes": tool_bytes,
    }


def _tool_definition_name(tool: Any) -> Any:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict) and "name" in function:
            return function["name"]
        if "name" in tool:
            return tool["name"]
    return tool


def _tool_output_summary(output: Any) -> Any:
    """A tool result as fields rather than a repr.

    deepagents hands `ToolMessage` objects to `on_tool_end`, so `default=str`
    wrote `content='...' name='write_file' tool_call_id='...'` — the same
    unqueryable blob that chain records used to be, and it buried `status`, which
    is the authoritative success/error signal. A permission denial was findable
    only by substring-matching the repr.

    `artifact` is reduced to a flag on purpose: tools may attach arbitrary
    payloads there, and a log is not the place to copy them.
    """
    content = getattr(output, "content", None)
    if content is None:
        return output

    summary: dict[str, Any] = {"content": content}
    for key in ("status", "name", "tool_call_id"):
        value = getattr(output, key, None)
        if value is not None:
            summary[key] = value
    if getattr(output, "artifact", None) is not None:
        summary["has_artifact"] = True
    return summary


def _state_summary(value: Any) -> Any:
    """What a graph step carried, without repeating the whole conversation.

    `chain_*` payloads are the agent's entire state, every time. Written in full
    they were 77% of a real run's bytes and within 6% of `MAX_FIELD_CHARS` on a
    five-check smoke run — and because deepagents passes `Command` objects here
    rather than the `dict` the signature promises, `default=str` turned them into
    Python reprs that no JSON reader can query.

    The content is already recorded structurally by `chat_model_start`, `llm_end`
    and the tool records. What is worth keeping here is the *shape*: which keys a
    step passed and how the message list grew.
    """
    if isinstance(value, dict):
        summary: dict[str, Any] = {"keys": sorted(str(k) for k in value)}
        messages = value.get("messages")
        if isinstance(messages, list):
            summary["messages"] = len(messages)
            summary["message_types"] = [
                getattr(m, "type", type(m).__name__) for m in messages
            ]
        return summary
    if isinstance(value, list):
        return {"items": len(value), "types": sorted({type(v).__name__ for v in value})}
    return {"type": type(value).__name__}


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


_RETRY_ADVICE_HEADERS = ("x-should-retry", "retry-after", "retry-after-ms")
"""The headers the openai client reads *before* it looks at the status code.

`BaseClient._should_retry` returns `False` outright on `x-should-retry: false`
or a `Retry-After` above its 120s cap — so a 429 can arrive unretried while
`max_retries` is set and working. Measured offline against a local server: a
plain 429 produced three requests over 1.49s, and either of those headers
produced one request in 0.00s (F46).
"""


def _retry_advice(error: BaseException) -> dict[str, Any] | None:
    """What the server told the client about retrying, or `None`.

    `None` means the request never reached a server — a timeout or a connection
    failure — which is a different diagnosis from a server that answered and
    declined a retry. An empty-but-present mapping of headers means the server
    answered and said nothing, so the retry was exhausted rather than vetoed.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    advice: dict[str, Any] = {}
    status = getattr(response, "status_code", None)
    if status is not None:
        advice["status_code"] = status
    for header in _RETRY_ADVICE_HEADERS:
        value = headers.get(header)
        if value is not None:
            advice[header] = value
    return advice


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
        self._wrote_tool_sizes = False
        """The per-tool breakdown is written once. The schemas do not change
        within a run, so repeating them on every model call would be the same
        bytes logged forever for no new information — while the aggregate stays
        per call, because the messages do grow."""

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
            inputs=_state_summary(inputs),
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
        self._write("chain_end", run_id, parent_run_id, outputs=_state_summary(outputs))

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
        self._write("tool_end", run_id, parent_run_id, output=_tool_output_summary(output))

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
        flat = [message for batch in messages for message in batch]
        params = kwargs.get("invocation_params")
        size = _request_size(params if isinstance(params, dict) else {}, flat)
        if self._wrote_tool_sizes:
            del size["tool_bytes"]
        else:
            self._wrote_tool_sizes = True
        self._write(
            "chat_model_start",
            run_id,
            parent_run_id,
            name=_component_name(serialized, kwargs),
            params=_invocation_params(kwargs),
            size=size,
            messages=[_message_summary(message) for message in flat],
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
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] | None = None
        metadata: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for batch in response.generations:
            for generation in batch:
                outputs.append(generation.text)
                message = getattr(generation, "message", None)
                if message is None:
                    continue

                calls = getattr(message, "tool_calls", None)
                if calls:
                    tool_calls.extend(_tool_call_summary(call) for call in calls)

                response_metadata = getattr(message, "response_metadata", None) or {}
                for key in _RESPONSE_METADATA_KEYS:
                    if key in response_metadata:
                        metadata[key] = response_metadata[key]

                # Where a provider would put reasoning content if it returned any.
                # None does today — the token count is all we get — so this is
                # empty in practice and omitted rather than written as `{}`.
                for key, value in (getattr(message, "additional_kwargs", None) or {}).items():
                    if value:
                        extra[key] = value

                token_usage = getattr(message, "usage_metadata", None)
                if token_usage:
                    usage = dict(token_usage)

        payload: dict[str, Any] = {
            "outputs": outputs,
            "tool_calls": tool_calls,
            "usage": usage,
            "metadata": metadata,
        }
        if extra:
            payload["extra"] = extra
        self._write("llm_end", run_id, parent_run_id, **payload)

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
        payload: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": str(error),
        }
        # Only on errors that reached a server, so the key's presence is itself
        # the signal (F46).
        advice = _retry_advice(error)
        if advice is not None:
            payload["retry_advice"] = advice
        self._write("llm_error", run_id, parent_run_id, **payload)


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
