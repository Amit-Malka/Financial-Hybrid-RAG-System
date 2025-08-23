from langchain_core.documents import Document
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
from sec_parser.semantic_elements.top_section_title import TopSectionTitle
import logging
from ..config import Config

def chunk_document(elements: list[AbstractSemanticElement]) -> list[Document]:
    """Converts SEC semantic elements into LangChain Documents with metadata."""
    logger = logging.getLogger("processing.chunker")
    chunks = []
    for i, element in enumerate(elements):
        metadata = {"element_type": element.__class__.__name__}
        # Add unique chunk identifier (5th metadata field per specification)
        metadata["chunk_id"] = f"{Config.CHUNK_ID_PREFIX}{i}"
        # Safely enrich metadata if attributes exist on the element
        page_number = getattr(element, "page_number", None)
        if page_number is not None:
            metadata["page_number"] = page_number
        section_path = getattr(element, "section_path", None)
        if section_path is not None:
            metadata["section_path"] = section_path
        content_type = getattr(element, "content_type", None)
        if content_type is not None:
            metadata["content_type"] = content_type
        chunks.append(Document(page_content=str(element), metadata=metadata))
    logger.info(f"Chunked {len(elements)} elements -> {len(chunks)} documents with 5-field metadata")
    return chunks

def get_section_chunks(elements: list[AbstractSemanticElement], section_type: type) -> list[Document]:
    """Gets chunked Documents for a specific section type."""
    section_elements = [el for el in elements if isinstance(el, section_type)]
    return chunk_document(section_elements)

def get_elements_in_section(
    elements: list[AbstractSemanticElement],
    *,
    section_identifier: str,
) -> list[AbstractSemanticElement]:
    """Return all elements belonging to the top-level section identified by identifier.

    Uses TopSectionTitle markers from sec_parser to find the start of the section with
    the given identifier and collects all following elements until the next TopSectionTitle
    of the same or higher level.
    """
    logger = logging.getLogger("processing.chunker")
    start_idx = None
    current_level = None
    for i, el in enumerate(elements):
        if isinstance(el, TopSectionTitle):
            sect = getattr(el, "section_type", None)
            ident = getattr(sect, "identifier", None)
            if ident == section_identifier:
                start_idx = i + 1
                current_level = getattr(sect, "level", None)
                break
    if start_idx is None:
        logger.warning(f"Section identifier not found: {section_identifier}")
        return []

    collected: list[AbstractSemanticElement] = []
    for el in elements[start_idx:]:
        if isinstance(el, TopSectionTitle):
            sect = getattr(el, "section_type", None)
            next_level = getattr(sect, "level", None)
            if current_level is not None and next_level is not None and next_level <= current_level:
                break
        collected.append(el)
    logger.info(f"Section {section_identifier}: collected {len(collected)} elements")
    return collected