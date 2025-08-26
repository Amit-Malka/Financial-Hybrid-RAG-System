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
        
        # ENHANCED CONTENT EXTRACTION: Try multiple methods to get actual text content
        text_content = extract_element_text(element)
        
        # Log content extraction for debugging
        if len(text_content.strip()) < 10:
            logger.warning(f"Element {i} ({element.__class__.__name__}) has minimal content: '{text_content[:50]}...'")
        
        chunks.append(Document(page_content=text_content, metadata=metadata))
    logger.info(f"Chunked {len(elements)} elements -> {len(chunks)} documents with 5-field metadata")
    return chunks

def extract_element_text(element: AbstractSemanticElement) -> str:
    """Extract actual text content from SEC parser elements using multiple fallback methods."""
    
    # Method 1: Try .text attribute
    if hasattr(element, 'text') and element.text:
        text = str(element.text).strip()
        if text:
            return text
    
    # Method 2: Try .content attribute
    if hasattr(element, 'content') and element.content:
        text = str(element.content).strip()
        if text:
            return text
    
    # Method 3: Try .inner_text attribute
    if hasattr(element, 'inner_text') and element.inner_text:
        text = str(element.inner_text).strip()
        if text:
            return text
    
    # Method 4: Try .get_text() method
    if hasattr(element, 'get_text'):
        try:
            text = str(element.get_text()).strip()
            if text:
                return text
        except Exception:
            pass
    
    # Method 5: Try to access underlying HTML and extract text
    if hasattr(element, 'html_tag') and element.html_tag:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(str(element.html_tag), 'html.parser')
            text = soup.get_text().strip()
            if text:
                return text
        except Exception:
            pass
    
    # Method 6: Check for html attribute and extract text
    if hasattr(element, 'html') and element.html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(str(element.html), 'html.parser')
            text = soup.get_text().strip()
            if text:
                return text
        except Exception:
            pass
    
    # Method 7: Try accessing the raw HTML content if available
    for attr_name in ['_html', 'raw_html', 'source_html']:
        if hasattr(element, attr_name):
            try:
                html_content = getattr(element, attr_name)
                if html_content:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(str(html_content), 'html.parser')
                    text = soup.get_text().strip()
                    if text:
                        return text
            except Exception:
                pass
    
    # Method 8: Fallback to string representation
    fallback_text = str(element).strip()
    
    # If the string representation is just the class name, try to find any text attributes
    if len(fallback_text) < 50 or fallback_text.startswith(element.__class__.__name__):
        # Look for any attribute that might contain text
        for attr in dir(element):
            if not attr.startswith('_') and attr not in ['html_tag', 'html']:
                try:
                    value = getattr(element, attr)
                    if isinstance(value, str) and len(value.strip()) > 10:
                        return value.strip()
                except Exception:
                    pass
    
    return fallback_text

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