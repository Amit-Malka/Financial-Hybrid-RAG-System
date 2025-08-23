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

    # Provide an invoke method returning an object with a content attribute
    def invoke(self, input: Any, **kwargs: Any):
        class Reply:
            def __init__(self, content: str):
                self.content = content
        if isinstance(input, str):
            return Reply(f"This is a dummy response from the LLM for: {input}")
        return Reply("This is a dummy response from the LLM.")

    # Implement required abstract API methods for BaseLanguageModel
    def predict(self, text: str, **kwargs: Any) -> str:
        return f"This is a dummy response from the LLM for: {text}"

    def predict_messages(self, messages: Any, **kwargs: Any) -> str:
        return "This is a dummy response from the LLM."

    def generate_prompt(self, *args: Any, **kwargs: Any) -> str:
        return "Generated prompt"

    async def apredict(self, text: str, **kwargs: Any) -> str:
        return self.predict(text, **kwargs)

    async def apredict_messages(self, messages: Any, **kwargs: Any) -> str:
        return self.predict_messages(messages, **kwargs)

    async def agenerate_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self.generate_prompt(*args, **kwargs)