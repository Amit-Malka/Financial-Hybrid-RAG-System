import os
# Disable ChromaDB telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False" 
os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"

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

        # Check if we got meaningful content
        if mda_elements:
            mda_chunks = chunk_document(mda_elements)
            total_content_length = sum(len(chunk.page_content.strip()) for chunk in mda_chunks)
            logger.info(f"MDATool structured approach: {len(mda_chunks)} chunks, {total_content_length} total characters")

            # If content is too minimal, fall back to keyword filtering
            if total_content_length < 100:
                logger.info("Structured content too minimal, falling back to keyword-based filtering")
                mda_elements = self._filter_by_mda_keywords(elements)
        else:
            logger.info("No part1item2 section found, using keyword-based filtering")
            mda_elements = self._filter_by_mda_keywords(elements)

        # If still no good content, use the full document approach like Summary
        if not mda_elements:
            logger.warning("No MD&A content found, using full document retrieval like Summary tool")
            mda_chunks = chunk_document(elements)  # Use all elements
        else:
            mda_chunks = chunk_document(mda_elements)

        logger.info(f"MDATool final initialization: {len(mda_chunks)} chunks")

        dense_retriever = get_dense_retriever(mda_chunks)
        sparse_retriever = Financial10QRetriever(mda_chunks)
        retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        super().__init__(retriever, llm)

    def _filter_by_mda_keywords(self, elements):
        """Filter elements that likely contain MD&A content using keywords."""
        mda_keywords = [
            "management", "discussion", "analysis", "md&a", "results of operations",
            "financial condition", "liquidity", "capital resources", "outlook",
            "business environment", "market conditions", "financial performance",
            "executive overview", "consolidated revenues", "operating income",
            "three months ended", "revenue", "expenses", "profitability"
        ]

        filtered_elements = []
        for element in elements:
            text_content = str(element).lower()
            # Require meaningful content length and keyword match
            if len(text_content.strip()) > 50 and any(keyword in text_content for keyword in mda_keywords):
                filtered_elements.append(element)

        return filtered_elements