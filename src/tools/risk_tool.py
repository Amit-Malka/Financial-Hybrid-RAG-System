from .base import SimpleTool
from langchain_core.language_models import BaseLanguageModel
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever
from ..processing.chunker import chunk_document, get_elements_in_section

class RiskTool(SimpleTool):
    def __init__(self, llm: BaseLanguageModel, elements: list):
        # Use TopSectionTitle identifiers for Risk Factors in 10-Q: 'part2item1a'
        risk_elements = get_elements_in_section(elements, section_identifier="part2item1a")
        risk_chunks = chunk_document(risk_elements)
        dense_retriever = get_dense_retriever(risk_chunks)
        sparse_retriever = Financial10QRetriever(risk_chunks)
        retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        super().__init__(retriever, llm)
