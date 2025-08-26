try:
    from llama_index.core.llms import LLM
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.llms.types import (
        CompletionResponse,
        CompletionResponseGen,
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        MessageRole,
    )
except ImportError:
    # Fallback imports for different LlamaIndex versions
    try:
        from llama_index.llms import LLM
        from llama_index.llms.callbacks import llm_completion_callback
        from llama_index.llms.types import (
            CompletionResponse,
            CompletionResponseGen,
            ChatMessage,
            ChatResponse,
            ChatResponseGen,
            MessageRole,
        )
    except ImportError:
        # Ultimate fallback - create minimal implementations
        from abc import ABC, abstractmethod
        
        class LLM(ABC):
            def __init__(self):
                pass
        
        class CompletionResponse:
            def __init__(self, text: str):
                self.text = text
        
        class ChatMessage:
            def __init__(self, role, content: str):
                self.role = role
                self.content = content
        
        class ChatResponse:
            def __init__(self, message):
                self.message = message
        
        class MessageRole:
            ASSISTANT = "assistant"
        
        def llm_completion_callback():
            from contextlib import nullcontext
            return nullcontext()
        
        CompletionResponseGen = None
        ChatResponseGen = None

from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional, Iterable, AsyncIterable
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

    def stream_complete(self, prompt: str, **kwargs: Any):
        if CompletionResponseGen is None:
            # Fallback for missing types
            yield CompletionResponse(text=self._complete(prompt, **kwargs))
        else:
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

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        if ChatResponseGen is None:
            # Fallback for missing types
            yield self.chat(messages, **kwargs)
        else:
            for c in self.chat(messages, **kwargs).message.content.splitlines():
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=c))

    # Async variants
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._complete, prompt)
        return CompletionResponse(text=text)

    async def astream_complete(self, prompt: str, **kwargs: Any):
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

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        resp = await self.achat(messages, **kwargs)
        async def agen():
            for c in resp.message.content.splitlines():
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=c))
        return agen()

    @property
    def metadata(self):
        return {}
