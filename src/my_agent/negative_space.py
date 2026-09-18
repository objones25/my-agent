"""Runtime checks that survive ``python -O``.

Adapted from the negative-space-programming skill and owned here: this copy has
diverged deliberately (PEP 695 generics, a sorted ``__all__``, and only the
helpers this project actually uses). Standard library only, no dependencies.

Why not ``assert``: ``python -O`` / ``PYTHONOPTIMIZE=1`` removes every ``assert``
statement from the bytecode, condition and message included. Checks that encode
your contracts must not disappear when someone flips an interpreter flag.

Use for programmer errors (a state your own code should have made impossible).
For operating errors -- bad user input, missing file, network failure -- raise a
typed exception instead and handle it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NoReturn

__all__ = [
    "CheckFailed",
    "bounded",
    "require",
    "unreachable",
]



class CheckFailed(AssertionError):
    """A contract was violated. Subclasses AssertionError so existing handlers
    and ``pytest.raises(AssertionError)`` keep working."""


def require(condition: object, message: str = "") -> None:
    """Fail unless ``condition`` is truthy.

    >>> require(1 < 2)
    >>> try:
    ...     require(2 < 1, "ordering broken")
    ... except CheckFailed as exc:
    ...     print(exc)
    ordering broken

    Keep one predicate per call: ``require(a); require(b)`` reports which half
    failed, ``require(a and b)`` does not.
    """
    if not condition:
        raise CheckFailed(message or "requirement failed")


def unreachable(message: str = "") -> NoReturn:
    """Mark a branch that cannot execute -- an exhausted ``match``, an ``else``
    that exists only for symmetry. Reaching it means the code is wrong."""
    raise CheckFailed(message or "reached unreachable code")


def bounded[T](iterable: Iterable[T], limit: int, name: str = "loop") -> Iterator[T]:
    """Yield from ``iterable``, failing if it produces more than ``limit`` items.

    Wraps any loop whose length you did not compute yourself, so an unbounded
    producer becomes a crash instead of a hang.

    >>> list(bounded(range(3), 5))
    [0, 1, 2]
    >>> try:
    ...     list(bounded(range(10), 3, name="retries"))
    ... except CheckFailed as exc:
    ...     print(exc)
    retries exceeded its bound of 3 iterations
    """
    require(limit >= 1, f"{name}: bound must be at least 1, got {limit}")
    for count, item in enumerate(iterable, start=1):
        if count > limit:
            raise CheckFailed(f"{name} exceeded its bound of {limit} iterations")
        yield item
