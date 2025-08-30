from langchain_core.documents import Document
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
from sec_parser.semantic_elements.top_section_title import TopSectionTitle
import logging
from collections import Counter
from ..config import Config

def chunk_document(elements: list[AbstractSemanticElement]) -> list[Document]:
    """Builds chunks from SEC semantic elements.

    - Default: preserves existing 1:1 element→chunk behavior to maintain backward compatibility.
    - If Config.USE_SECTION_AWARE_CHUNKING is True: creates section-aware semantic chunks
      that never cross top-level section boundaries, target Config.CHUNK_SIZE, and apply
      Config.CHUNK_OVERLAP within the same section.
    """
    logger = logging.getLogger("processing.chunker")

    if getattr(Config, "USE_SECTION_AWARE_CHUNKING", False):
        return _build_section_aware_chunks(elements, logger)

    # Fallback to 1:1 element wrapping (legacy behavior)
    chunks: list[Document] = []
    for i, element in enumerate(elements):
        metadata = {"element_type": element.__class__.__name__}
        metadata["chunk_id"] = f"{Config.CHUNK_ID_PREFIX}{i}"
        page_number = getattr(element, "page_number", None)
        if page_number is not None:
            metadata["page_number"] = page_number
        section_path = getattr(element, "section_path", None)
        if section_path is not None:
            metadata["section_path"] = section_path
        content_type = getattr(element, "content_type", None)
        if content_type is not None:
            metadata["content_type"] = content_type

        text_content = extract_element_text(element)
        if len(text_content.strip()) < 10:
            logger.warning(
                f"Element {i} ({element.__class__.__name__}) has minimal content: '{text_content[:50]}...'"
            )

        chunks.append(Document(page_content=text_content, metadata=metadata))

    logger.info(
        f"Chunked {len(elements)} elements -> {len(chunks)} documents (legacy 1:1) with 5-field metadata"
    )
    return chunks

def _build_section_aware_chunks(elements: list[AbstractSemanticElement], logger: logging.Logger) -> list[Document]:
    """Create section-aware semantic chunks using size and overlap from Config.

    Rules:
      - Do not cross TopSectionTitle boundaries (treat each sequence after a TopSectionTitle
        as a section until the next TopSectionTitle).
      - Concatenate adjacent element texts until reaching CHUNK_SIZE; when starting a new
        chunk, carry CHUNK_OVERLAP characters from the tail of previous chunk within the
        same section.
      - Preserve 5-field metadata; for merged chunks:
          element_type: "Composite" if >1 source types else that type
          page_number: lowest page; include all pages in metadata["pages"]
          chunk_id: sequential based on final chunk index
          section_path: dominant/first non-empty within section
          content_type: "mixed" if multiple else that type or "unknown"
    """
    chunk_size = int(Config.CHUNK_SIZE)
    overlap = int(Config.CHUNK_OVERLAP)
    assert overlap < chunk_size

    # Identify section boundaries based on TopSectionTitle occurrences
    sections: list[list[int]] = []  # list of lists of indices into elements
    current_indices: list[int] = []
    for idx, el in enumerate(elements):
        if isinstance(el, TopSectionTitle):
            # Start a new section; flush prior if exists
            if current_indices:
                sections.append(current_indices)
                current_indices = []
            continue
        current_indices.append(idx)
    if current_indices:
        sections.append(current_indices)

    documents: list[Document] = []
    running_chunk_index = 0

    for sec_indices in sections:
        if not sec_indices:
            continue

        # Collect section metadata candidates once
        sec_section_paths = []
        for i in sec_indices:
            sp = getattr(elements[i], "section_path", None)
            if sp:
                sec_section_paths.append(str(sp))
        section_path_value = (
            Counter(sec_section_paths).most_common(1)[0][0] if sec_section_paths else ""
        )

        # Build chunks within this section
        cursor = 0
        current_text = ""
        current_elements: list[int] = []

        def flush_current_chunk():
            nonlocal running_chunk_index, current_text, current_elements
            if not current_text:
                return

            # Aggregate metadata from current_elements
            element_types = [elements[i].__class__.__name__ for i in current_elements]
            content_types = [getattr(elements[i], "content_type", None) for i in current_elements]
            pages = [getattr(elements[i], "page_number", None) for i in current_elements if getattr(elements[i], "page_number", None) is not None]
            page_min = min(pages) if pages else None
            element_type_value = (
                element_types[0] if len(set(element_types)) == 1 else "Composite"
            )
            # Resolve content_type
            ct_candidates = [ct for ct in content_types if ct]
            content_type_value = (
                ct_candidates[0] if len(set(ct_candidates)) <= 1 and ct_candidates else ("mixed" if ct_candidates else "unknown")
            )

            metadata = {
                "element_type": element_type_value,
                "chunk_id": f"{Config.CHUNK_ID_PREFIX}{running_chunk_index}",
                "section_path": section_path_value,
                "content_type": content_type_value,
            }
            if page_min is not None:
                metadata["page_number"] = page_min
            if pages:
                metadata["pages"] = sorted(set(pages))

            documents.append(Document(page_content=current_text, metadata=metadata))
            running_chunk_index += 1

        # Iterate through elements and build text windows
        while cursor < len(sec_indices):
            idx = sec_indices[cursor]
            el = elements[idx]
            el_text = extract_element_text(el)
            if not el_text:
                cursor += 1
                continue

            # If adding this element exceeds target chunk size, flush current and start new with overlap
            if current_text and len(current_text) + 1 + len(el_text) > chunk_size:
                flush_current_chunk()
                # Start new chunk with overlap from previous chunk tail within the section
                if documents and overlap > 0:
                    tail = documents[-1].page_content[-overlap:]
                    current_text = tail
                    current_elements = []  # overlap is from text, not from a single element
                else:
                    current_text = ""
                    current_elements = []

            # Append element text (with a space) and record source element index
            current_text = (current_text + (" " if current_text else "") + el_text).strip()
            current_elements.append(idx)
            cursor += 1

        # Flush any remainder for this section
        flush_current_chunk()

    logger.info(
        f"Built {len(documents)} section-aware chunks from {len(elements)} elements (size={chunk_size}, overlap={overlap})"
    )
    return documents

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

            # Get text from common textual and tabular/list elements
            for tag in soup.find_all(['p', 'span', 'b', 'strong', 'em', 'div', 'li', 'td', 'th']):
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