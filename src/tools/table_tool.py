from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from sec_parser.semantic_elements.table_element.table_element import TableElement
import logging

class TableAnswer(BaseModel):
    """Data model for a table answer."""
    answer: str = Field(..., description="The answer to the question about the table.")

class TableTool(SimpleTool):
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel, elements: list):
        logger = logging.getLogger("tools.table")

        # If retriever passed, use it directly (current behavior)
        if retriever:
            super().__init__(retriever, llm)
        else:
            # Create specialized financial table retriever
            table_elements = self._filter_by_financial_keywords(elements)
            from ..processing.chunker import chunk_document
            from ..retrieval.dense_retriever import get_dense_retriever
            from ..retrieval.tfidf_retriever import Financial10QRetriever
            from ..retrieval.ensemble_setup import create_ensemble_retriever

            table_chunks = chunk_document(table_elements)
            logger.info(f"TableTool initialized with {len(table_chunks)} chunks")

            dense_retriever = get_dense_retriever(table_chunks)
            sparse_retriever = Financial10QRetriever(table_chunks)
            specialized_retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
            super().__init__(specialized_retriever, llm)

        self.elements = elements
        self.program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=TableAnswer),
            prompt_template_str=(
                "Please answer the following question based on the financial data below.\n"
                "Focus on extracting specific numbers, percentages, and financial metrics.\n"
                "If the data contains tabular information, parse it carefully.\n"
                "---------------------\n"
                "{table_str}\n"
                "---------------------\n"
                "Question: {query_str}\n"
                "\n"
                "Provide a clear, specific answer with exact numbers and percentages when available."
            ),
            llm=llm,
            verbose=True,
        )

    def execute(self, query: str) -> str:
        # ENHANCED: Use retrieval to find relevant financial/tabular content
        self.logger.info(f"Executing table tool for query: {query}")
        
        # First try to find specific TableElement instances
        table_element = next((el for el in self.elements if isinstance(el, TableElement)), None)
        
        if table_element is not None:
            # Found actual table element - use it directly
            self.logger.info("Found TableElement, using direct table processing")
            table_str = str(table_element)
            response = self.program(table_str=table_str, query_str=query)
            return response.answer
        
        # No TableElement found - use retrieval to find financial data
        self.logger.info("No TableElement found, using retrieval for financial/tabular data")
        context_docs = self.retriever.get_relevant_documents(query)
        
        if not context_docs:
            return "No relevant financial data found for this query."
        
        # Combine retrieved context focusing on numerical/tabular data
        financial_context = []
        for doc in context_docs:
            content = doc.page_content
            # Prioritize content with numbers, percentages, financial terms
            if any(indicator in content.lower() for indicator in 
                  ['$', '%', 'million', 'billion', 'increase', 'decrease', 'quarter', 'year-over-year', 
                   'revenue', 'cost-per-click', 'paid clicks', 'table', 'financial']):
                financial_context.append(content)
        
        if not financial_context:
            # Fallback to all retrieved content
            financial_context = [doc.page_content for doc in context_docs]
        
        combined_context = "\n\n".join(financial_context)
        
        # Use LlamaIndex program with financial context instead of table
        response = self.program(table_str=combined_context, query_str=query)
        return response.answer

    def _filter_by_financial_keywords(self, elements):
        """Filter elements that likely contain financial data and tables."""
        financial_keywords = [
            "revenue", "income", "earnings", "balance sheet", "cash flow",
            "assets", "liabilities", "equity", "financial statements",
            "consolidated", "thousands", "millions", "$", "net income",
            "total revenue", "operating income", "comprehensive income"
        ]

        filtered_elements = []
        for element in elements:
            text_content = str(element).lower()
            if any(keyword in text_content for keyword in financial_keywords):
                filtered_elements.append(element)

        return filtered_elements