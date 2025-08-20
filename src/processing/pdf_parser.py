from sec_parser import SecParser
from llama_index.core.schema import Document

def load_html(file_path: str) -> list[Document]:
    """Loads an HTML document and parses it using sec-parser."""
    parser = SecParser()
    with open(file_path, "r") as f:
        html = f.read()
    
    elements = parser.parse(html)
    # For now, we are just returning the text of each element as a document.
    # We will need to process the elements to extract the structure.
    documents = [Document(page_content=str(element)) for element in elements]
    return documents
