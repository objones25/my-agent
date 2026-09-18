"""Unit tests for the contract helpers.

The doctests in `negative_space.py` are documentation that happens to execute,
and they now run in this suite. What they read badly for lives here: the failure
paths, the boundaries, and the category discipline every other module's
`require()` calls depend on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from my_agent.negative_space import CheckFailed, bounded, require, unreachable


def test_require_passes_a_truthy_condition(
    assert_does_not_raise: Callable[[Callable[[], object]], None],
) -> None:
    assert_does_not_raise(lambda: require(True, "should not fire"))


def test_require_raises_check_failed_with_its_message() -> None:
    with pytest.raises(CheckFailed, match="ordering broken"):
        require(False, "ordering broken")


def test_require_has_a_message_even_when_none_is_given() -> None:
    """A bare `require(x)` that fires must still say something."""
    with pytest.raises(CheckFailed, match="requirement failed"):
        require(False)


def test_check_failed_is_an_assertion_error() -> None:
    """Existing handlers and `pytest.raises(AssertionError)` must keep working —
    which is also why a test expecting a tripped require() must name
    CheckFailed, the narrower type."""
    assert issubclass(CheckFailed, AssertionError)


def test_unreachable_always_raises() -> None:
    with pytest.raises(CheckFailed, match="reached unreachable code"):
        unreachable()


def test_unreachable_carries_its_message() -> None:
    with pytest.raises(CheckFailed, match="match was exhausted"):
        unreachable("match was exhausted")


def test_bounded_yields_everything_inside_the_bound() -> None:
    assert list(bounded(range(3), 5)) == [0, 1, 2]


def test_bounded_allows_exactly_the_limit() -> None:
    """The boundary, stated on its own: off-by-one here would either reject a
    legal loop or let one extra iteration through."""
    assert list(bounded(range(3), 3)) == [0, 1, 2]


def test_bounded_rejects_one_item_past_the_limit() -> None:
    with pytest.raises(CheckFailed, match="exceeded its bound of 3"):
        list(bounded(range(4), 3))


def test_bounded_names_the_loop_in_its_failure() -> None:
    with pytest.raises(CheckFailed, match="retries exceeded"):
        list(bounded(range(10), 3, name="retries"))


@pytest.mark.parametrize("limit", [0, -1], ids=["zero", "negative"])
def test_bounded_rejects_a_bound_that_permits_no_iterations(limit: int) -> None:
    with pytest.raises(CheckFailed, match="bound must be at least 1"):
        list(bounded(range(3), limit))


def test_bounded_is_lazy() -> None:
    """It wraps producers whose length you did not compute, so it must not
    consume the iterable to check the bound — that would hang on exactly the
    infinite producer it exists to catch."""
    consumed: list[int] = []

    def producer() -> Iterator[int]:
        for i in range(100):
            consumed.append(i)
            yield i

    first = next(bounded(producer(), 5))

    assert first == 0
    assert consumed == [0]
