"""Shared fixtures, and the offline guarantee the unit suite only claimed.

CLAUDE.md says unit tests are deterministic and offline: no network, no real
model. `_forbid_network` turns that from a convention into a check. `ChatOpenAI`
and `create_deep_agent` both build lazily and make no request, so a socket
opening during the default suite means a test started reaching the router — a
defect in the test, not a slow test.

Anything that really needs the network is marked `live`, deselected by default
in `addopts`, and stepped over here.

`_forbid_leaked_threads` closes the hole the socket guard cannot see on its own:
the guard is only patched in *during* a test, so a thread a test starts and
leaves running reaches the network between tests, where nothing is patched.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from collections.abc import Callable, Collection, Iterator
from typing import Any, NoReturn

import pytest
from deepagents import FilesystemPermission
from pydantic import SecretStr

# Only `pytester` — nothing else here needs a non-default plugin. It backs
# `tests/test_conftest.py`'s wiring test for `_forbid_leaked_threads`, which
# has to run the fixture as pytest actually runs it (autouse, around a real
# test) rather than call its helper directly.
pytest_plugins = ["pytester"]

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


THREAD_EXIT_GRACE_S = 1.0
"""How long, in total, a test's new threads get to finish on their own.

Only paid when a test leaves a thread behind; a test that starts none waits for
nothing. A thread that ends inside this window was finishing, not leaked.
"""


def threads_left_running(
    before: Collection[threading.Thread], grace_s: float
) -> list[threading.Thread]:
    """Threads not in `before` that are still alive once `grace_s` has run out.

    One shared deadline rather than `grace_s` per thread, so ten leaked threads
    cost one grace period, not ten.

    `thread.join()` cannot be trusted here: a `threading._DummyThread` — the
    handle Python fabricates for a thread that entered the interpreter without
    going through `threading.Thread` (a worker spun up with the low-level
    `_thread.start_new_thread`, as some C-backed libraries do) — raises
    `RuntimeError("cannot join a dummy thread")` unconditionally, timeout or
    not. Falling back to polling `is_alive()` against the same shared deadline
    keeps a dummy thread from crashing the check that is supposed to report
    it.
    """
    deadline = time.monotonic() + grace_s
    left: list[threading.Thread] = []
    for thread in threading.enumerate():
        if thread in before:
            continue
        try:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except RuntimeError:
            while thread.is_alive() and time.monotonic() < deadline:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if thread.is_alive():
            left.append(thread)
    return left


@pytest.fixture(autouse=True)
def _forbid_leaked_threads(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail any unmarked test that leaves a thread running after it ends.

    `_forbid_network` is a monkeypatch, so it is undone at every teardown and
    re-applied at the next setup. A thread outliving its test therefore runs
    partly *unguarded*: measured, langsmith's tracing thread resolved
    `api.smith.langchain.com` with the real `getaddrinfo` in the gap between
    two tests, then hit the re-applied `connect` guard inside the next test.
    The guard raises `RuntimeError`, which urllib3's `create_connection` does
    not catch (it closes the socket on `OSError` only), so the socket leaked
    and its finaliser failed whichever test the garbage collector happened to
    run in. A leaked thread is the defect; the socket was only its symptom.
    """
    if request.node.get_closest_marker("live") is not None:
        yield
        return
    before = frozenset(threading.enumerate())
    yield
    leaked = threads_left_running(before, THREAD_EXIT_GRACE_S)
    if leaked:
        pytest.fail(
            f"a unit test left {len(leaked)} thread(s) running: "
            f"{sorted(t.name for t in leaked)}. A thread outliving its test runs "
            "outside the socket guard; stop it, or stop the code under test from "
            "starting it."
        )


@pytest.fixture
def tripping_an_import_time_check() -> Callable[..., None]:
    """Re-import a module with a library patched, so its load-time checks run again.

    **The one group of checks a test cannot otherwise reach.** `capabilities.py`,
    `agent.py` and `model.py` assert things about the installed wheels *at
    import* — that `execute` is still a filesystem tool, that
    `create_deep_agent` grew no parameter, that `ChatOpenAI` still accepts every
    `ModelConfig` field. Those run once, when the test session imports the
    package, and a normal test has no way to make one false: by the time it runs
    the import has long since succeeded.

    `importlib.reload` is the way, and the restore is the delicate half. A
    reload executes the module body in the *existing* module's namespace, so a
    reload that raises part way leaves the module half re-executed and every
    later test importing wreckage. The `finally` therefore does two things in
    order: put the library back, then reload again so the module is left in the
    state the rest of the session expects.

    Patches are applied to the *library* module rather than to the module under
    test, because the checks read what `from deepagents import ...` brings in —
    patching the copy would leave the import re-reading the real one.
    """

    def trip(module: Any, target: Any, attribute: str, value: Any) -> None:
        # **A module that defines a class must never be reloaded here.** Reload
        # rebinds the class to a *new* object, while every other test module
        # still holds the old one — so `isinstance(cfg, AgentConfig)` inside the
        # reloaded module fails against an instance built anywhere else, and the
        # error reads `expected an AgentConfig, got AgentConfig`. Measured: 15
        # unrelated tests failed this way, none of them when run alone (F44).
        # A load-time check in a module that defines classes belongs in a
        # function instead, the way `contracts.check_known_parameters` is.
        defined_here = sorted(
            name
            for name, value_ in vars(module).items()
            if isinstance(value_, type) and value_.__module__ == module.__name__
        )
        assert not defined_here, (
            f"{module.__name__} defines {defined_here}; reloading it would rebind "
            f"those classes and break isinstance across the suite. Express the "
            f"load-time check as a function and test it directly."
        )
        original = getattr(target, attribute)
        setattr(target, attribute, value)
        try:
            importlib.reload(module)
        finally:
            setattr(target, attribute, original)
            # Not inside a `try`: if *this* reload fails the session is already
            # unrecoverable and a swallowed error would present as a confusing
            # failure in some unrelated test much later.
            importlib.reload(module)

    return trip
