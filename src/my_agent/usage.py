"""What one model call reported spending, read one way for every consumer.

`RunTokenBudget` bounds a turn on this number and `JsonlMirror` logs it. They
used to read `usage_metadata` with two different loops: the budget summed every
choice and the mirror kept the last. langchain-openai copies the request's
usage onto *every* choice (`BaseChatOpenAI._create_chat_result`), so for an
n-choice response those disagree by a factor of n. One reader makes "the number
this bounds is the number the log shows" true by construction.

Its own module because of the dependency direction: `run.py` is core and
`mirror.py` is observability, neither should import the other, and both import
this.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult

from my_agent.negative_space import require

__all__ = ["call_usage"]


def call_usage(response: LLMResult) -> UsageMetadata | None:
    """The request's usage, or `None` if no choice reported any.

    A chat model's callback receives one prompt per `LLMResult`:
    `BaseChatModel.generate` and `agenerate` flatten a batch before calling
    back, so `response.generations` holds one inner list, that request's
    choices. Every choice carries the same request-level usage, so the first one
    carrying any is the request's figure and the rest are copies of it.

    A streamed call arrives as `ChatGenerationChunk` / `AIMessageChunk`, which
    subclass the types checked here, so it is read the same way. A generation
    with no message (a non-chat model's) is skipped.
    """
    # Precondition: the flattening is langchain-core's behaviour, not ours. Any
    # other shape means it changed, and picking across prompts would be a guess.
    require(
        len(response.generations) == 1,
        f"call_usage expected one prompt per on_llm_end, got "
        f"{len(response.generations)} prompts' generations; langchain-core flattens "
        f"a batch before calling back, so this shape means that changed",
    )
    for generation in response.generations[0]:
        message = generation.message if isinstance(generation, ChatGeneration) else None
        if isinstance(message, AIMessage) and message.usage_metadata:
            return message.usage_metadata
    return None
