from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import PandasQueryEngine
from sec_parser.semantic_elements.table_element.table_element import TableElement
import pandas as pd
import re
from typing import Optional

class TableAnswer(BaseModel):
    """Data model for a table answer."""
    answer: str = Field(..., description="The answer to the question about the table.")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    data_source: str = Field(..., description="Source of the data used")

class AdvancedTableTool(SimpleTool):
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel, elements: list):
        super().__init__(retriever, llm)
        self.elements = elements
        self.program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=TableAnswer),
            prompt_template_str=(
                "You are a financial analyst expert at extracting precise information from financial documents.\n"
                "Based on the financial data below, answer the question with exact numbers and context.\n"
                "Focus on:\n"
                "- Exact numerical values (revenue, costs, percentages, rates)\n"
                "- Time periods (quarters, years)\n"
                "- Comparisons and trends\n"
                "- Financial ratios and metrics\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Question: {query_str}\n"
                "\n"
                "Requirements:\n"
                "1. Provide exact numbers with units (millions, billions, percentages)\n"
                "2. Include time period context (Q1 2025, FY 2024, etc.)\n"
                "3. If data shows trends, mention direction and magnitude\n"
                "4. If information is not available, clearly state this\n"
                "5. Cite specific sections when possible\n"
            ),
            llm=llm,
            verbose=True,
        )

    def execute(self, query: str) -> str:
        self.logger.info(f"Executing advanced table tool for query: {query}")

        # Step 1: Check for actual TableElement instances
        table_elements = [el for el in self.elements if isinstance(el, TableElement)]

        if table_elements:
            self.logger.info(f"Found {len(table_elements)} TableElement instances")
            return self._process_table_elements(table_elements, query)

        # Step 2: Use enhanced retrieval for financial/tabular content
        self.logger.info("No TableElement found, using enhanced retrieval for financial data")
        return self._process_with_enhanced_retrieval(query)

    def _process_table_elements(self, table_elements: list, query: str) -> str:
        """Process actual table elements using pandas and LlamaIndex."""
        try:
            # Convert table elements to pandas DataFrames
            dfs = []
            for table in table_elements:
                df = self._table_element_to_dataframe(table)
                if df is not None and not df.empty:
                    dfs.append(df)

            if dfs:
                # Use LlamaIndex PandasQueryEngine for advanced querying
                combined_df = pd.concat(dfs, ignore_index=True)
                query_engine = PandasQueryEngine(df=combined_df, verbose=True)
                response = query_engine.query(query)
                return str(response)

        except Exception as e:
            self.logger.error(f"Table element processing failed: {e}")

        # Fallback to text-based processing
        table_text = "\n\n".join([str(table) for table in table_elements])
        response = self.program(context_str=table_text, query_str=query)
        return f"{response.answer}\n\nConfidence: {response.confidence}\nSource: {response.data_source}"

    def _process_with_enhanced_retrieval(self, query: str) -> str:
        """Enhanced retrieval focusing on numerical and tabular content."""
        context_docs = self.retriever.get_relevant_documents(query)

        if not context_docs:
            return "No relevant financial data found for this query."

        # Enhanced filtering for financial/tabular content
        financial_context = self._filter_financial_content(context_docs)

        if not financial_context:
            # Fallback to all retrieved content
            financial_context = [doc.page_content for doc in context_docs]

        # Try to extract structured data
        structured_data = self._extract_structured_data(financial_context, query)

        if structured_data:
            # Use pandas processing if we found structured data
            try:
                df = pd.DataFrame(structured_data)
                query_engine = PandasQueryEngine(df=df, verbose=True)
                response = query_engine.query(query)
                return str(response)
            except Exception as e:
                self.logger.debug(f"Pandas processing failed: {e}")

        # Standard LLM processing with enhanced context
        combined_context = "\n\n".join(financial_context)
        response = self.program(context_str=combined_context, query_str=query)
        return f"{response.answer}\n\nConfidence: {response.confidence}\nSource: {response.data_source}"

    def _filter_financial_content(self, docs) -> list[str]:
        """Filter documents for financial/numerical content."""
        from ..processing.chunker import extract_tabular_patterns

        financial_content = []
        for doc in docs:
            content = doc.page_content

            # Priority 1: Content with tabular patterns
            if extract_tabular_patterns(content):
                financial_content.append(content)
                continue

            # Priority 2: Content with financial keywords and numbers
            financial_indicators = [
                '$', '%', 'million', 'billion', 'revenue', 'cost', 'income',
                'quarter', 'Q1', 'Q2', 'Q3', 'Q4', 'year-over-year',
                'TAC', 'traffic acquisition cost', 'cost-per-click',
                'paid clicks', 'advertising', 'search', 'youtube',
                'cloud', 'other bets', 'capex', 'operating margin'
            ]

            content_lower = content.lower()
            if (any(indicator in content_lower for indicator in financial_indicators) and
                re.search(r'\d+', content)):
                financial_content.append(content)

        return financial_content

    def _extract_structured_data(self, contexts: list[str], query: str) -> Optional[list[dict]]:
        """Attempt to extract structured data for pandas processing."""
        structured_data = []

        for context in contexts:
            # Extract financial metrics patterns
            patterns = {
                'revenue': r'revenue[:\s]*\$?([\d,]+(?:\.\d+)?)\s*(million|billion)?',
                'tac_rate': r'TAC rate[:\s]*([\d.]+)%',
                'cost_per_click': r'cost-per-click[:\s]*\$?([\d.]+)',
                'quarter': r'Q([1-4])\s+(\d{4})',
                'percentage': r'([\d.]+)%',
            }

            row_data = {}
            for key, pattern in patterns.items():
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    row_data[key] = matches

            if row_data:
                structured_data.append(row_data)

        return structured_data if structured_data else None

    def _table_element_to_dataframe(self, table_element) -> Optional[pd.DataFrame]:
        """Convert TableElement to pandas DataFrame."""
        try:
            # This depends on the specific TableElement implementation
            # May need adjustment based on actual sec-parser TableElement structure
            if hasattr(table_element, 'to_dataframe'):
                return table_element.to_dataframe()
            elif hasattr(table_element, 'rows'):
                # Manual conversion from rows
                data = []
                headers = None
                for i, row in enumerate(table_element.rows):
                    if i == 0 and self._looks_like_header(row):
                        headers = [str(cell) for cell in row]
                    else:
                        data.append([str(cell) for cell in row])

                if headers and data:
                    return pd.DataFrame(data, columns=headers)

        except Exception as e:
            self.logger.error(f"Failed to convert table element to DataFrame: {e}")

        return None

    def _looks_like_header(self, row) -> bool:
        """Check if a row looks like table headers."""
        row_text = ' '.join([str(cell) for cell in row]).lower()
        header_indicators = ['quarter', 'year', 'revenue', 'cost', 'income', 'total', 'percent']
        return any(indicator in row_text for indicator in header_indicators)

# Alias for backward compatibility
TableTool = AdvancedTableTool