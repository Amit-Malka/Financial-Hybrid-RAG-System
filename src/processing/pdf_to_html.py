import fitz
import logging

def convert_pdf_to_html(pdf_path: str) -> str:
    """Converts a PDF file to HTML."""
    logger = logging.getLogger("processing.pdf_to_html")
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    html = ""
    for page in doc:
        html += page.get_text("html")
    logger.info(f"Extracted HTML from {doc.page_count} pages")
    return html
