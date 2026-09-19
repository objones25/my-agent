"""Tests for the socket guard in `conftest.py`.

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

import pytest


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
