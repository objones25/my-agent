"""Runtime checks that survive ``python -O``.

Copy this module into your project. Standard library only, no dependencies.

Why not ``assert``: ``python -O`` / ``PYTHONOPTIMIZE=1`` removes every ``assert``
statement from the bytecode, condition and message included. Checks that encode
your contracts must not disappear when someone flips an interpreter flag.

Use for programmer errors (a state your own code should have made impossible).
For operating errors -- bad user input, missing file, network failure -- raise a
typed exception instead and handle it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NoReturn

__all__ = [
    "CheckFailed",
    "bounded",
    "check_finite",
    "check_shape",
    "require",
    "unreachable",
]



class CheckFailed(AssertionError):
    """A contract was violated. Subclasses AssertionError so existing handlers
    and ``pytest.raises(AssertionError)`` keep working."""


def require(condition: object, message: str = "") -> None:
    """Fail unless ``condition`` is truthy.

    >>> require(1 < 2)
    >>> require(2 < 1, "ordering broken")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: ordering broken

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
    >>> list(bounded(range(10), 3, name="retries"))
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: retries exceeded its bound of 3 iterations
    """
    require(limit >= 1, f"{name}: bound must be at least 1, got {limit}")
    for count, item in enumerate(iterable, start=1):
        if count > limit:
            raise CheckFailed(f"{name} exceeded its bound of {limit} iterations")
        yield item


def check_shape(
    array: Any,
    spec: Sequence[int | str | None],
    name: str = "array",
) -> dict[str, int]:
    """Check a NumPy/PyTorch-style ``.shape`` against ``spec`` and return the
    bindings for any named dimensions.

    ``spec`` entries: an ``int`` for a fixed size, a ``str`` for a named
    dimension that must be consistent across its uses, or ``None``/``-1`` for
    "any size".

    >>> class Fake:  # any object with a .shape tuple
    ...     shape = (8, 3, 32, 32)
    >>> check_shape(Fake(), ("B", 3, "H", "H"), name="images")
    {'B': 8, 'H': 32}
    """
    raw_shape = getattr(array, "shape", None)
    # Explicit raise rather than require(): mypy narrows `if ... raise`, but cannot
    # narrow through a helper call. Same runtime behaviour, and survives `python -O`.
    if raw_shape is None:
        raise CheckFailed(f"{name} has no .shape attribute")
    shape = tuple(int(d) for d in raw_shape)
    require(
        len(shape) == len(spec),
        f"{name}: expected {len(spec)} dims {tuple(spec)}, got {len(shape)} {shape}",
    )

    bindings: dict[str, int] = {}
    for axis, (actual, expected) in enumerate(zip(shape, spec, strict=True)):
        if expected is None or expected == -1:
            continue
        if isinstance(expected, str):
            if expected in bindings:
                require(
                    bindings[expected] == actual,
                    f"{name}: dim '{expected}' is {bindings[expected]} elsewhere "
                    f"but {actual} at axis {axis}; full shape {shape}",
                )
            else:
                bindings[expected] = actual
        else:
            require(
                actual == expected,
                f"{name}: axis {axis} expected {expected}, got {actual}; "
                f"full shape {shape}",
            )
    return bindings


def check_finite(value: Any, name: str = "value") -> None:
    """Fail if ``value`` contains NaN or infinity. Accepts a Python float or any
    array exposing ``.isfinite()`` / working with ``math.isfinite`` after
    ``float()``.

    >>> check_finite(0.5)
    >>> check_finite(float("nan"), name="loss")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: loss is not finite: nan
    """
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        allf = getattr(result, "all", None)
        ok = bool(allf()) if callable(allf) else bool(result)
        require(ok, f"{name} is not finite: {value}")
        return
    require(math.isfinite(float(value)), f"{name} is not finite: {value}")


if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)
    raise SystemExit(1 if failures else 0)
