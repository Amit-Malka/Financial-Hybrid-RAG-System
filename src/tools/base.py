from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
import logging
import time

class SimpleTool:
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, query: str) -> str:
        # Simple: retrieve → generate → return
        self.logger.info("Executing tool")
        context = self.retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in context])
        prompt = f"Context: {context_text}\n\nQuestion: {query}"
        start = time.perf_counter()
        try:
            response = self.llm.invoke(prompt)
        except Exception:
            self.logger.exception("LLM invocation failed")
            raise
        finally:
            self.logger.debug(f"LLM invocation took {time.perf_counter() - start:.2f}s")
        try:
            # Handle Chat-like results returning an object with 'content'
            return getattr(response, "content", str(response))
        except Exception:
            return str(response)