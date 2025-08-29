from .base import SimpleTool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever
from ..processing.chunker import chunk_document, get_elements_in_section
import logging

class MDATool(SimpleTool):
    def __init__(self, llm: BaseLanguageModel, elements: list):
        logger = logging.getLogger("tools.mda")

        # Try structured approach first
        mda_elements = get_elements_in_section(elements, section_identifier="part1item2")

        # Fallback: if no structured sections found, filter by content keywords
        if not mda_elements:
            logger.info("No part1item2 section found, using keyword-based filtering")
            mda_elements = self._filter_by_mda_keywords(elements)

        mda_chunks = chunk_document(mda_elements)
        logger.info(f"MDATool initialized with {len(mda_chunks)} chunks")

        dense_retriever = get_dense_retriever(mda_chunks)
        sparse_retriever = Financial10QRetriever(mda_chunks)
        retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        super().__init__(retriever, llm)

    def _filter_by_mda_keywords(self, elements):
        """Filter elements that likely contain MD&A content using keywords."""
        mda_keywords = [
            "management", "discussion", "analysis", "md&a", "results of operations",
            "financial condition", "liquidity", "capital resources", "outlook",
            "business environment", "market conditions", "financial performance"
        ]

        filtered_elements = []
        for element in elements:
            text_content = str(element).lower()
            if any(keyword in text_content for keyword in mda_keywords):
                filtered_elements.append(element)

        return filtered_elements