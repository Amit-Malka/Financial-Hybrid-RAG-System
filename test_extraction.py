#!/usr/bin/env python3
"""
Test script for the enhanced text extraction and table processing functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.processing.pdf_parser import load_html
from src.processing.chunker import chunk_document, extract_element_text, extract_tabular_patterns
from src.tools.table_tool import AdvancedTableTool
from src.retrieval.ensemble_setup import create_ensemble_retriever
from src.llm.langchain_llm import LangchainLLM
from src.config import Config
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")

def test_basic_text_extraction():
    """Test basic text extraction from a document."""
    logger.info("Testing basic text extraction...")

    # Load HTML document
    html_file = "data/uploads/goog-10-q-q2-2025.pdf.html"
    if not os.path.exists(html_file):
        logger.error(f"HTML file not found: {html_file}")
        return

    logger.info(f"Loading document: {html_file}")
    elements = load_html(html_file)

    if not elements:
        logger.error("No elements loaded from HTML")
        return

    logger.info(f"Loaded {len(elements)} elements")

    # Test enhanced text extraction
    company_found = False
    for i, element in enumerate(elements[:20]):  # Test first 20 elements
        text = extract_element_text(element)
        logger.info(f"Element {i}: {text[:100]}...")

        # Check for company name
        if "alphabet" in text.lower() or "google" in text.lower():
            logger.info(f"✅ Found company reference in element {i}: {text}")
            company_found = True
            break

    if not company_found:
        logger.warning("❌ Company name not found in first 20 elements")

    return elements

def test_tabular_pattern_detection():
    """Test tabular pattern detection."""
    logger.info("Testing tabular pattern detection...")

    test_texts = [
        "Revenue increased by $10 million in Q2 2025",
        "Q1 2024 vs Q1 2025 revenue comparison",
        "TAC rate decreased to 2.5% from 3.1%",
        "Cost-per-click reduced by 15% year-over-year",
        "Normal text without financial data"
    ]

    for text in test_texts:
        has_pattern = extract_tabular_patterns(text)
        logger.info(f"Text: '{text[:50]}...' -> Tabular pattern: {has_pattern}")

def test_table_tool():
    """Test the advanced table tool functionality."""
    logger.info("Testing advanced table tool...")

    # Load HTML document
    html_file = "data/uploads/goog-10-q-q2-2025.pdf.html"
    if not os.path.exists(html_file):
        logger.error(f"HTML file not found: {html_file}")
        return

    elements = load_html(html_file)

    if not elements:
        logger.error("No elements loaded")
        return

    # Create a mock LLM for testing
    class MockLLM:
        def invoke(self, prompt):
            class MockResponse:
                content = f"Mock response to: {prompt[:50]}..."
            return MockResponse()

    mock_llm = MockLLM()

    # Create dummy retriever
    try:
        from src.retrieval.dense_retriever import get_dense_retriever
        retriever = get_dense_retriever()
        logger.info("Retriever created successfully")
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        # Create a simple mock retriever
        class MockRetriever:
            def get_relevant_documents(self, query):
                return []
        retriever = MockRetriever()

    # Test table tool
    try:
        table_tool = AdvancedTableTool(retriever, mock_llm, elements)
        logger.info("AdvancedTableTool created successfully")

        # Test with a simple query
        result = table_tool.execute("What is the company name?")
        logger.info(f"Table tool result: {result}")

        # Test with a numerical query
        result2 = table_tool.execute("What was the TAC rate in Q2 2025?")
        logger.info(f"TAC query result: {result2}")

    except Exception as e:
        logger.error(f"Table tool test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting extraction tests...")

    # Test 1: Basic text extraction
    elements = test_basic_text_extraction()

    print("\n" + "="*50 + "\n")

    # Test 2: Tabular pattern detection
    test_tabular_pattern_detection()

    print("\n" + "="*50 + "\n")

    # Test 3: Advanced table tool
    if elements:
        test_table_tool()

    logger.info("Tests completed!")
