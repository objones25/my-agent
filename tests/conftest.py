"""Shared fixtures, and the offline guarantee the unit suite only claimed.

CLAUDE.md says unit tests are deterministic and offline: no network, no real
model. `_forbid_network` turns that from a convention into a check. `ChatOpenAI`
and `create_deep_agent` both build lazily and make no request, so a socket
opening during the default suite means a test started reaching the router — a
defect in the test, not a slow test.

Anything that really needs the network is marked `live`, deselected by default
in `addopts`, and stepped over here.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any, NoReturn

import pytest
from deepagents import FilesystemPermission
from pydantic import SecretStr

VALID_KEY = "hf_token_value"
"""Shaped like an HF token and obviously not one. Never a real credential."""


@pytest.fixture
def valid_key() -> str:
    """The raw token string, for tests asserting it does not leak."""
    return VALID_KEY


@pytest.fixture
def valid_secret() -> SecretStr:
    """A `ModelConfig.api_key` that passes every precondition."""
    return SecretStr(VALID_KEY)


@pytest.fixture
def deny_secrets() -> FilesystemPermission:
    """One targeted deny rule. Fresh per test, so no test can leak a mutation
    into the next one."""
    return FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny")


@pytest.fixture
def assert_does_not_raise() -> Callable[[Callable[[], object]], None]:
    """State "this must not raise" as an assertion instead of as an absence.

    A test whose whole body is a call to a contract helper asserts nothing: it
    passes if the helper is deleted, and it reports a crash rather than a
    failure when the helper rejects something it should accept. Wrapping the
    call names the expectation and gives a readable failure either way.
    """

    def check(call: Callable[[], object]) -> None:
        try:
            call()
        except Exception as exc:
            pytest.fail(f"expected no exception, got {type(exc).__name__}: {exc}")

    return check


@pytest.fixture(autouse=True)
def _forbid_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any unmarked test that opens a socket."""
    if request.node.get_closest_marker("live") is not None:
        return

    def deny(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            "a unit test opened a socket. Unit tests are offline by contract; "
            "mark it `live` if it genuinely needs the network."
        )

    # Connect-shaped exits.
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    # Exits that never call connect, and so were open while the suite claimed
    # to be offline: a name lookup is egress on its own, and a datagram send
    # puts a packet on the wire with no connection to intercept. `sendmsg` is
    # the same hole as `sendto` under a different name — it also takes a
    # destination address and needs no prior `connect`.
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "gethostbyname", deny)
    monkeypatch.setattr(socket.socket, "sendto", deny)
    monkeypatch.setattr(socket.socket, "sendmsg", deny)
