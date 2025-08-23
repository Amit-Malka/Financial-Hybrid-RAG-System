from sec_parser import Edgar10QParser
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
import logging

def load_html(file_path: str) -> list[AbstractSemanticElement]:
    """Parses HTML with sec-parser and returns semantic elements.

    We intentionally return the raw semantic elements to preserve structure
    (e.g., MD&A, Risk Factors, Tables). Downstream modules are responsible
    for converting these elements into the appropriate `langchain` Documents
    when constructing retrievers.
    """
    logger = logging.getLogger("processing.pdf_parser")
    logger.info(f"Parsing HTML file: {file_path}")
    parser = Edgar10QParser()
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    elements = parser.parse(html)
    logger.info(f"Parsed {len(elements)} semantic elements")
    return elements
