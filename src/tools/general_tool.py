from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

class GeneralTool(SimpleTool):
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel):
        super().__init__(retriever, llm)