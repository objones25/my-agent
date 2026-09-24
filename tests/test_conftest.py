"""Tests for the socket guard and the thread guard in `conftest.py`.

`_forbid_network` is the enforcement behind CLAUDE.md's "'Offline' is enforced,
not assumed". It had no tests, which made that claim exactly the kind of
unverified assertion the rest of this suite exists to refuse: a guard nobody has
watched fire is a guard nobody knows fires.

Each test here names one way out of the process and asserts the guard closes it.
They run *under* the autouse fixture, so they assert on the real patched state
rather than on a reconstruction of it.
"""

from __future__ import annotations

import _thread
import socket
import threading
import time
from pathlib import Path

import pytest

from tests.conftest import threads_left_running


def test_the_guard_blocks_a_tcp_connect() -> None:
    """The hole the guard was written for. Characterisation: this passed before
    the DNS and datagram cases below existed, and it must keep passing."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match="opened a socket"),
    ):
        sock.connect(("127.0.0.1", 9))


def test_the_guard_blocks_create_connection() -> None:
    """`socket.create_connection` is a module-level function, so patching
    `socket.socket.connect` alone would not reach it."""
    with pytest.raises(RuntimeError, match="opened a socket"):
        socket.create_connection(("127.0.0.1", 9))


def test_the_guard_blocks_a_dns_lookup() -> None:
    """A name lookup is network egress that never calls `connect`.

    `localhost` resolves from `/etc/hosts` without leaving the machine, so this
    test proves the guard *intercepts* `getaddrinfo` without itself depending on
    a network. A real hostname would have left the machine — which is the
    defect.
    """
    with pytest.raises(RuntimeError, match="opened a socket"):
        socket.getaddrinfo("localhost", 80)


def test_the_guard_blocks_gethostbyname() -> None:
    """The older resolver entry point, reachable independently of
    `getaddrinfo`."""
    with pytest.raises(RuntimeError, match="opened a socket"):
        socket.gethostbyname("localhost")


def test_the_guard_blocks_an_unconnected_datagram_send() -> None:
    """`sendto` puts a packet on the wire with no `connect` first, so every
    connect-shaped patch misses it."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(RuntimeError, match="opened a socket"),
    ):
        sock.sendto(b"x", ("127.0.0.1", 9))


def test_the_guard_blocks_sendmsg() -> None:
    """`sendmsg` is the same connectionless exit as `sendto` under a different
    name: it also carries its own destination address and needs no prior
    `connect`, so a guard that only patches `sendto` misses it."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(RuntimeError, match="opened a socket"),
    ):
        sock.sendmsg([b"x"], [], 0, ("127.0.0.1", 9))


def test_a_thread_still_running_after_the_grace_period_is_reported() -> None:
    """The shape of the leak `_forbid_leaked_threads` exists for: a thread that
    outlives its test and would run outside the socket guard."""
    before = frozenset(threading.enumerate())
    release = threading.Event()
    lingering = threading.Thread(target=release.wait, name="lingering", daemon=True)
    lingering.start()
    try:
        assert threads_left_running(before, grace_s=0.05) == [lingering]
    finally:
        release.set()
        lingering.join(timeout=5)


def test_a_thread_that_finishes_inside_the_grace_period_is_not_reported() -> None:
    """The discriminator: without it, a guard that reported *every* new thread
    would pass the test above, and fail any test that briefly uses one."""
    before = frozenset(threading.enumerate())
    # Still running when the check starts, so only the wait can clear it.
    finishing = threading.Thread(target=time.sleep, args=(0.05,), name="finishing")
    finishing.start()

    assert threads_left_running(before, grace_s=5.0) == []


def test_threads_that_existed_before_the_test_are_not_reported() -> None:
    """Only threads the test started are its responsibility."""
    release = threading.Event()
    earlier = threading.Thread(target=release.wait, name="earlier", daemon=True)
    earlier.start()
    try:
        before = frozenset(threading.enumerate())
        assert threads_left_running(before, grace_s=0.05) == []
    finally:
        release.set()
        earlier.join(timeout=5)


def test_a_dummy_thread_does_not_crash_the_check() -> None:
    """`threading._DummyThread.join()` raises unconditionally on 3.13 — see the
    docstring on `threads_left_running`. A real one only exists for a thread
    that entered the interpreter without going through `threading.Thread`, so
    it is made the same way: `_thread.start_new_thread` (the module
    `threading` itself is built on) starts a raw OS thread, and calling
    `threading.current_thread()` from inside it is what makes Python fabricate
    the `_DummyThread` and register it in `threading.enumerate()`.

    This still leaks the underlying OS thread until `done` is set — daemon by
    construction (`_thread.start_new_thread` threads always are), so it costs
    nothing beyond this test.
    """
    done = threading.Event()
    registered = threading.Event()

    def body() -> None:
        thread = threading.current_thread()
        assert isinstance(thread, threading._DummyThread)
        registered.set()
        done.wait(timeout=5)

    before = frozenset(threading.enumerate())
    _thread.start_new_thread(body, ())
    try:
        assert registered.wait(timeout=5), "the dummy thread never registered itself"
        # Must not raise despite the dummy thread among the new ones, and must
        # still report it as left running rather than silently dropping it.
        left = threads_left_running(before, grace_s=0.05)
        assert len(left) == 1
        assert isinstance(left[0], threading._DummyThread)
    finally:
        done.set()


def test_the_leaked_thread_fixture_fails_an_unmarked_test_and_spares_a_live_one(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the *wiring*, not just the helper.

    Every other test in this file calls `threads_left_running` directly; none
    of them exercise `_forbid_leaked_threads` the way pytest actually runs
    it — autouse, wrapped around a real test, with the `live` bypass read off
    a real marker. This spins up a real, separate pytest run with two tests
    that each leak a thread, one plain and one `@pytest.mark.live`, and checks
    that only the plain one is reported.

    Runs as a **subprocess** (`runpytest_subprocess`, not `runpytest`/
    in-process), on purpose: an in-process `pytest.main()` executes inside
    *this* test's own call phase, where this project's `filterwarnings =
    ["error"]` is still the active filter on Python's process-global warnings
    module — an unregistered `live` marker in the inner run would warn, that
    warning would be caught by the outer filter, and it would fail this test
    for a reason that has nothing to do with the thread guard. A subprocess
    gets its own interpreter and its own warnings state, so it cannot trip the
    outer filter. It also means the inner run has no `pyproject.toml` of its
    own, so `--doctest-modules` collects nothing and `_forbid_network` is
    irrelevant (nothing in the inner test touches a socket).

    The inner run loads `tests.conftest` itself as a plugin (`-p
    tests.conftest`, via `pytester.plugins`), so `_forbid_leaked_threads` is
    the *real* fixture, not a reimplementation. `PYTHONPATH` is set to the
    repo root so that import resolves; `THREAD_EXIT_GRACE_S` is lowered on the
    imported module so the inner run does not pay the full 1s grace period —
    the fixture's *reporting*, not the length of the wait, is what this test
    is about.
    """
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(repo_root))
    pytester.plugins.append("tests.conftest")
    pytester.makepyfile(
        test_leaked_thread="""
        import threading

        import pytest
        import tests.conftest as _guard

        _guard.THREAD_EXIT_GRACE_S = 0.05


        def _leak_a_thread():
            threading.Thread(
                target=threading.Event().wait, name="leaked", daemon=True
            ).start()


        def test_unmarked_leak():
            _leak_a_thread()


        @pytest.mark.live
        def test_live_leak():
            _leak_a_thread()
        """
    )

    result = pytester.runpytest_subprocess(timeout=60)

    # A leaked thread is reported at *teardown*, so the test's own call phase
    # still passes; pytest counts the teardown failure separately, as an
    # error, on top of that pass. `test_live_leak`'s call also passes, and the
    # bypass means its teardown reports nothing.
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["*ERROR at teardown of test_unmarked_leak*"])
    assert "test_live_leak" not in "\n".join(
        line for line in result.stdout.lines if "ERROR" in line
    )
