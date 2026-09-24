"""Tests for `my_agent.usage`: the one reading of what a model call spent.

The budget and the mirror both read this, so "the number this bounds is the
number the log shows" holds by construction rather than by two loops happening
to agree.
"""

from __future__ import annotations

import io
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation, LLMResult

from my_agent.mirror import JsonlMirror
from my_agent.negative_space import CheckFailed
from my_agent.run import RunTokenBudget
from my_agent.usage import call_usage


def _choice(total: int | None) -> ChatGeneration:
    """One choice, with `total` tokens of usage or with none reported."""
    if total is None:
        return ChatGeneration(message=AIMessage("ok"))
    return ChatGeneration(
        message=AIMessage(
            "ok",
            usage_metadata={"input_tokens": total, "output_tokens": 0, "total_tokens": total},
        )
    )


def test_two_choices_of_one_request_count_once() -> None:
    """langchain-openai copies the request's usage onto every choice, so summing
    choices counts an n-choice call n times."""
    usage = call_usage(LLMResult(generations=[[_choice(100), _choice(100)]]))

    assert usage is not None
    assert usage["total_tokens"] == 100


def test_the_first_choice_carrying_usage_is_the_one_read() -> None:
    usage = call_usage(LLMResult(generations=[[_choice(100), _choice(40)]]))

    assert usage is not None
    assert usage["total_tokens"] == 100


def test_usage_on_a_later_choice_only_is_still_read() -> None:
    """A choice with no usage before one with it is not an unmeasured call."""
    usage = call_usage(LLMResult(generations=[[_choice(None), _choice(40)]]))

    assert usage is not None
    assert usage["total_tokens"] == 40


def test_a_call_that_reported_no_usage_reads_as_none() -> None:
    assert call_usage(LLMResult(generations=[[_choice(None)]])) is None


def test_a_call_with_no_choices_reads_as_none() -> None:
    assert call_usage(LLMResult(generations=[[]])) is None


def test_a_streamed_call_is_read_like_any_other() -> None:
    """A streamed turn arrives as chunks. Missing it would make the token bound
    blind to every streamed call."""
    chunk = ChatGenerationChunk(
        message=AIMessageChunk(
            content="ok",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
    )

    usage = call_usage(LLMResult(generations=[[chunk]]))

    assert usage is not None
    assert usage["total_tokens"] == 10


def test_a_generation_with_no_message_is_skipped_not_fatal() -> None:
    usage = call_usage(LLMResult(generations=[[Generation(text="ok"), _choice(5)]]))

    assert usage is not None
    assert usage["total_tokens"] == 5


@pytest.mark.parametrize("prompts", [0, 2])
def test_a_result_holding_other_than_one_prompt_is_refused(prompts: int) -> None:
    """langchain-core flattens a batch before calling back, one prompt per
    `on_llm_end`. Any other shape means that stopped being true, and summing or
    picking across prompts would then be a guess."""
    response = LLMResult(generations=[[_choice(10)] for _ in range(prompts)])

    with pytest.raises(CheckFailed, match=f"got {prompts} prompts' generations"):
        call_usage(response)


def test_the_budget_and_the_mirror_count_the_same_tokens_for_one_call() -> None:
    """The claim `RunTokenBudget.on_llm_end`'s docstring makes, tested where it
    is made true."""
    response = LLMResult(generations=[[_choice(100), _choice(100)]])
    budget = RunTokenBudget(10_000)
    stream = io.StringIO()

    budget.on_llm_end(response, run_id=uuid4())
    JsonlMirror(stream).on_llm_end(response, run_id=uuid4())

    logged = json.loads(stream.getvalue())["usage"]["total_tokens"]
    assert budget.tokens == logged == 100
