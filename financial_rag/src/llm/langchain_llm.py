from llama_index.core.llms import LLM
from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional
from llama_index.core.llms.callbacks import llm_completion_callback

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
        pass

    @property
    def metadata(self):
        return {}
