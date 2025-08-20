from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class DummyLLM(BaseLanguageModel):
    """Dummy LLM for testing."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "This is a dummy response from the LLM."

    def _llm_type(self) -> str:
        return "dummy"
