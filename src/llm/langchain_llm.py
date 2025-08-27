try:
    from llama_index.core.llms import LLM
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseGen,
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        MessageRole,
    )
except ImportError:
    # Fallback for older LlamaIndex package layout
    from llama_index.llms import LLM
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseGen,
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        MessageRole,
    )

from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional, Iterable, AsyncIterable
import asyncio

class LLMMetadata:
    """Metadata class for LlamaIndex LLM compatibility."""
    def __init__(self, is_chat_model: bool = True, **kwargs):
        self.is_chat_model = is_chat_model
        self.model_name = kwargs.get('model_name', 'langchain_wrapped_model')
        self.context_window = kwargs.get('context_window', 4096)
        self.num_output = kwargs.get('num_output', 512)
        # Add other metadata attributes as needed
        for key, value in kwargs.items():
            setattr(self, key, value)

class LangchainLLM(LLM):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self._llm = llm
        # Create proper metadata object for LlamaIndex compatibility
        self._metadata = LLMMetadata(
            is_chat_model=True,  # Most modern LLMs are chat models
            model_name=getattr(llm, 'model_name', 'langchain_wrapped_model'),
            context_window=getattr(llm, 'context_window', 4096),
            num_output=getattr(llm, 'num_output', 512)
        )

    def _complete(self, prompt: str, **kwargs: Any) -> str:
        # Remove the context manager entirely
        response = self._llm.invoke(prompt)
        # Handle different response types from LangChain LLMs
        if hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return (response)

    def _stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion chunks from the underlying LangChain LLM if supported.

        Falls back to yielding the full completion once if streaming is not available.
        """
        with llm_completion_callback():
            # Prefer native streaming if the underlying LLM supports it
            if hasattr(self._llm, "stream") and callable(getattr(self._llm, "stream")):
                try:
                    for chunk in self._llm.stream(prompt):
                        if chunk is None:
                            continue
                        # LangChain message chunks commonly expose `content` or `text`
                        if hasattr(chunk, "content") and chunk.content:
                            yield chunk.content
                        elif hasattr(chunk, "text") and chunk.text:
                            yield chunk.text
                        else:
                            # Last resort, stringify the chunk
                            yield str(chunk)
                    return
                except Exception:
                    # If native streaming fails, fall back to non-streaming once
                    pass

            # Fallback: non-streaming single chunk
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
            # Flatten messages and stream using the same mechanism as completion
            prompt_parts = []
            for m in messages:
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                prompt_parts.append(f"{role}: {m.content}")
            prompt = "\n".join(prompt_parts)
            for chunk in self._stream_complete(prompt, **kwargs):
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=str(chunk)))

    # Async variants
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._complete, prompt)
        return CompletionResponse(text=text)

    async def astream_complete(self, prompt: str, **kwargs: Any):
        # Prefer native async streaming if supported by the underlying LLM
        if hasattr(self._llm, "astream") and callable(getattr(self._llm, "astream")):
            async def agen():
                try:
                    async for chunk in self._llm.astream(prompt):
                        if chunk is None:
                            continue
                        if hasattr(chunk, "content") and chunk.content:
                            yield CompletionResponse(text=str(chunk.content))
                        elif hasattr(chunk, "text") and chunk.text:
                            yield CompletionResponse(text=str(chunk.text))
                        else:
                            yield CompletionResponse(text=str(chunk))
                except Exception:
                    # Fall back to single completion on error
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(None, self._complete, prompt)
                    yield CompletionResponse(text=text)
            return agen()
        else:
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
        # Stream chat by flattening messages into a prompt and reusing astream_complete
        prompt_parts = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            prompt_parts.append(f"{role}: {m.content}")
        prompt = "\n".join(prompt_parts)

        async def agen():
            async for resp in await self.astream_complete(prompt, **kwargs):
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=resp.text))
        return agen()
    
    # Additional LlamaIndex compatibility methods
    @property
    def system_prompt(self):
        """System prompt for the LLM."""
        return getattr(self._llm, 'system_prompt', None)
    
    def set_system_prompt(self, system_prompt: str):
        """Set system prompt for the LLM."""
        if hasattr(self._llm, 'set_system_prompt'):
            self._llm.set_system_prompt(system_prompt)
    
    @property 
    def class_name(self) -> str:
        """Get class name for LlamaIndex."""
        return self.__class__.__name__

    @property
    def metadata(self):
        return self._metadata
