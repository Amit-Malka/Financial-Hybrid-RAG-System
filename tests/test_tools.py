from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from src.tools.base import SimpleTool


class EchoRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"context for: {query}")]


from src.llm.dummy_llm import DummyLLM as EchoLLM


def test_simple_tool_executes_with_llm():
    tool = SimpleTool(EchoRetriever(), EchoLLM())
    out = tool.execute("what is revenue?")
    assert "what is revenue?" in out.lower()
