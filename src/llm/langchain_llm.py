from llama_index.core.llms import LLM
from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional, Iterable, AsyncIterable
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
import asyncio

class LangchainLLM(LLM):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self._llm = llm

    def _complete(self, prompt: str, **kwargs: Any) -> str:
        with llm_completion_callback():
            response = self._llm.invoke(prompt)
            return response.content

    def _stream_complete(self, prompt: str, **kwargs: Any):
        # Not implemented
        yield self._complete(prompt, **kwargs)

    # Required abstract methods (sync)
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        text = self._complete(prompt, **kwargs)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        for chunk in self._stream_complete(prompt, **kwargs):
            yield CompletionResponse(text=str(chunk))

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Flatten messages into a single prompt for underlying LangChain LLM
        prompt_parts = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            prompt_parts.append(f"{role}: {m.content}")
        prompt = "\n".join(prompt_parts)
        text = self._complete(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        for c in self.chat(messages, **kwargs).message.content.splitlines():
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=c))

    # Async variants
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._complete, prompt)
        return CompletionResponse(text=text)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncIterable[CompletionResponse]:
        # Simple non-streaming async fallback
        resp = await self.acomplete(prompt, **kwargs)
        async def agen():
            yield resp
        return agen()

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        loop = asyncio.get_running_loop()
        prompt_parts = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            prompt_parts.append(f"{role}: {m.content}")
        prompt = "\n".join(prompt_parts)
        text = await loop.run_in_executor(None, self._complete, prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncIterable[ChatResponse]:
        resp = await self.achat(messages, **kwargs)
        async def agen():
            for c in resp.message.content.splitlines():
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=c))
        return agen()

    @property
    def metadata(self):
        return {}
