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
        self.logger.info(f"Executing tool for query: {query}")
        context = self.retriever.get_relevant_documents(query)
        
        # ENHANCED LOGGING: Log all retrieved chunks with detailed metadata
        self.logger.info(f"Retrieved {len(context)} chunks for query: '{query}'")
        for i, doc in enumerate(context):
            chunk_preview = doc.page_content[:200].replace('\n', ' ')  # First 200 chars
            metadata_info = {k: v for k, v in doc.metadata.items()}  # All metadata
            self.logger.info(f"CHUNK {i+1}/{len(context)}: {metadata_info}")
            self.logger.info(f"CONTENT_PREVIEW: {chunk_preview}...")
            self.logger.debug(f"FULL_CONTENT_{i+1}: {doc.page_content}")
        
        context_text = "\n\n".join([doc.page_content for doc in context])
        prompt = f"Context: {context_text}\n\nQuestion: {query}"
        start = time.perf_counter()
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            self.logger.exception(f"LLM invocation failed with error: {e}")
            raise RuntimeError(f"LLM invocation failed: {str(e)}") from e
        finally:
            self.logger.debug(f"LLM invocation took {time.perf_counter() - start:.2f}s")
        
        # Handle different response types from various LLM implementations
        try:
            if hasattr(response, "content"):
                return response.content
            elif hasattr(response, "text"):
                return response.text
            else:
                return str(response)
        except Exception as e:
            self.logger.warning(f"Failed to extract response content: {e}")
            return str(response)