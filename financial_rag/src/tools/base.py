from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

class SimpleTool:
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm
    
    def execute(self, query: str) -> str:
        # Simple: retrieve → generate → return
        context = self.retriever.get_relevant_documents(query)
        # The following line is commented out because we don't have an LLM yet.
        # return self.llm.invoke(f"Context: {context}\n\nQuestion: {query}")
        return f"Context: {context}\n\nQuestion: {query}"