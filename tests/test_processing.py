from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
from src.processing.chunker import chunk_document
from langchain_core.documents import Document


class DummyElement(AbstractSemanticElement):
    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        return self._text


def test_chunk_document_creates_langchain_documents():
    elements = [DummyElement("hello"), DummyElement("world")]
    chunks = chunk_document(elements)
    assert all(isinstance(d, Document) for d in chunks)
    assert chunks[0].page_content == "hello"
