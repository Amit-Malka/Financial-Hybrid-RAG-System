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
                "Please answer the following question based on the table below:\n"
                "---------------------\n"
                "{table_str}\n"
                "---------------------\n"
                "Question: {query_str}\n"
            ),
            llm=llm,
            verbose=True,
        )

    def execute(self, query: str) -> str:
        # Find the first table in the document
        table_element = next((el for el in self.elements if isinstance(el, TableElement)), None)

        if table_element is None:
            return "No table found in the document."

        table_str = str(table_element)
        response = self.program(table_str=table_str, query_str=query)
        return response.answer