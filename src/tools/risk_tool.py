from .base import SimpleTool
from langchain_core.language_models import BaseLanguageModel
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever
from ..processing.chunker import chunk_document, get_elements_in_section
import logging

class RiskTool(SimpleTool):
    def __init__(self, llm: BaseLanguageModel, elements: list):
        logger = logging.getLogger("tools.risk")

        # Try structured approach first
        risk_elements = get_elements_in_section(elements, section_identifier="part2item1a")

        # Check if we got meaningful content
        if risk_elements:
            risk_chunks = chunk_document(risk_elements)
            total_content_length = sum(len(chunk.page_content.strip()) for chunk in risk_chunks)
            logger.info(f"RiskTool structured approach: {len(risk_chunks)} chunks, {total_content_length} total characters")

            # If content is too minimal, fall back to keyword filtering
            if total_content_length < 100:
                logger.info("Structured content too minimal, falling back to keyword-based filtering")
                risk_elements = self._filter_by_risk_keywords(elements)
        else:
            logger.info("No part2item1a section found, using keyword-based filtering")
            risk_elements = self._filter_by_risk_keywords(elements)

        # If still no good content, use the full document approach like Summary
        if not risk_elements:
            logger.warning("No risk content found, using full document retrieval like Summary tool")
            risk_chunks = chunk_document(elements)  # Use all elements
        else:
            risk_chunks = chunk_document(risk_elements)

        logger.info(f"RiskTool final initialization: {len(risk_chunks)} chunks")

        dense_retriever = get_dense_retriever(risk_chunks)
        sparse_retriever = Financial10QRetriever(risk_chunks)
        retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        super().__init__(retriever, llm)

    def _filter_by_risk_keywords(self, elements):
        """Filter elements that likely contain risk factor content using keywords."""
        risk_keywords = [
            "risk", "risks", "uncertainty", "uncertainties", "may adversely",
            "could adversely", "risk factors", "forward-looking", "cautionary",
            "material adverse", "significant risk", "potential impact",
            "contractual obligations", "commitments", "acquisition", "regulatory",
            "market risk", "competitive", "economic conditions"
        ]

        filtered_elements = []
        for element in elements:
            text_content = str(element).lower()
            # Require meaningful content length and keyword match
            if len(text_content.strip()) > 50 and any(keyword in text_content for keyword in risk_keywords):
                filtered_elements.append(element)

        return filtered_elements
