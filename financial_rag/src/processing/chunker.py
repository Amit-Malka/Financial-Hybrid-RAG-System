from llama_index.core.schema import Document
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
from sec_parser.semantic_elements.management_discussion_and_analysis_element import MDNAElement
from sec_parser.semantic_elements.risk_factors_element import RiskFactorsElement

def chunk_document(elements: list[AbstractSemanticElement]) -> list[Document]:
    """Chunks a list of semantic elements into a list of documents."""
    chunks = []
    for element in elements:
        # We can add more metadata here later
        metadata = {"element_type": element.__class__.__name__}
        chunks.append(Document(page_content=str(element), metadata=metadata))
    return chunks

def get_section_chunks(elements: list[AbstractSemanticElement], section_type: type) -> list[Document]:
    """Gets the chunks for a specific section."""
    section_elements = [el for el in elements if isinstance(el, section_type)]
    return chunk_document(section_elements)