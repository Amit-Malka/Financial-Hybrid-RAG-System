from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever
from ..processing.chunker import get_section_chunks
from sec_parser.semantic_elements.management_discussion_and_analysis_element import MDNAElement

class MDATool(SimpleTool):
    def __init__(self, llm: BaseLanguageModel, elements: list):
        mda_chunks = get_section_chunks(elements, MDNAElement)
        dense_retriever = get_dense_retriever(mda_chunks)
        sparse_retriever = Financial10QRetriever(mda_chunks)
        retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        super().__init__(retriever, llm)