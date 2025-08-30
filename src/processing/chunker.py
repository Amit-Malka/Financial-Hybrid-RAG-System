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
    """Extract actual text content from SEC parser elements using enhanced HTML parsing."""

    # Method 1: Try existing text attributes first
    for attr in ['text', 'content', 'inner_text']:
        if hasattr(element, attr):
            value = getattr(element, attr)
            if value and len(str(value).strip()) > 5:
                return str(value).strip()

    # Method 2: Enhanced HTML parsing with BeautifulSoup
    html_content = None

    # Try to get HTML content from various attributes
    for attr in ['html_tag', 'html', '_html', 'raw_html', 'source_html']:
        if hasattr(element, attr):
            html_content = getattr(element, attr)
            if html_content:
                break

    if html_content:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(str(html_content), 'html.parser')

            # Enhanced text extraction for CSS-positioned content
            text_parts = []

            # Get text from all p, span, b, strong, em elements
            for tag in soup.find_all(['p', 'span', 'b', 'strong', 'em', 'div']):
                tag_text = tag.get_text(strip=True)
                if tag_text and len(tag_text) > 1:
                    text_parts.append(tag_text)

            # Join with spaces and clean up
            full_text = ' '.join(text_parts)

            # Clean up multiple spaces and normalize
            import re
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            if len(full_text) > 5:
                return full_text

        except Exception as e:
            # Log parsing error but continue
            import logging
            logger = logging.getLogger("processing.chunker")
            logger.debug(f"HTML parsing failed for element: {e}")

    # Method 3: Try get_text() method
    if hasattr(element, 'get_text'):
        try:
            text = str(element.get_text()).strip()
            if text and len(text) > 5:
                return text
        except Exception:
            pass

    # Method 4: Fallback to string representation
    fallback_text = str(element).strip()

    # If still minimal, try to find any meaningful text attributes
    if len(fallback_text) < 20:
        for attr in dir(element):
            if not attr.startswith('_') and not callable(getattr(element, attr)):
                try:
                    value = getattr(element, attr)
                    if isinstance(value, str) and len(value.strip()) > 10:
                        return value.strip()
                except Exception:
                    continue

    return fallback_text if fallback_text else f"{element.__class__.__name__}<{getattr(element, 'tag_name', 'unknown')}>"


def extract_tabular_patterns(text: str) -> bool:
    """Detect if text contains tabular financial data patterns."""
    import re

    # Financial table patterns
    patterns = [
        r'\$[\d,]+\s+\$[\d,]+',  # Multiple dollar amounts in sequence
        r'Q[1-4]\s+\d{4}.*Q[1-4]\s+\d{4}',  # Multiple quarters
        r'\d+\.?\d*%.*\d+\.?\d*%',  # Multiple percentages
        r'(Revenue|Cost|Income|Expense).*\$[\d,]+',  # Financial line items
        r'(millions?|billions?|thousands?).*\$[\d,]+',  # Scale indicators
        r'(increase|decrease).*\d+\.?\d*%',  # Change indicators
        r'(TAC|Traffic|Acquisition|Cost).*\$?[\d,]+',  # TAC specific
        r'\d{4}\s+\d{4}\s+\d{4}',  # Multiple years in sequence
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


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