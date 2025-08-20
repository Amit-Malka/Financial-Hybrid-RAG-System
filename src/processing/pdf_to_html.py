import fitz

def convert_pdf_to_html(pdf_path: str) -> str:
    """Converts a PDF file to HTML."""
    doc = fitz.open(pdf_path)
    html = ""
    for page in doc:
        html += page.get_text("html")
    return html
