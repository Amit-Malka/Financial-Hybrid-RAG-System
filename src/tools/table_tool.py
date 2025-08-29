from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from sec_parser.semantic_elements.table_element.table_element import TableElement

class TableAnswer(BaseModel):
    """Data model for a table answer."""
    answer: str = Field(..., description="The answer to the question about the table.")

class TableTool(SimpleTool):
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel, elements: list):
        super().__init__(retriever, llm)
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