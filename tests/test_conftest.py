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

import socket
import threading
import time

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
