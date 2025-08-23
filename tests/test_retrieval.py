from langchain_core.documents import Document
from src.retrieval.tfidf_retriever import Financial10QRetriever


def test_tfidf_retriever_boosts_financial_terms():
    docs = [
        Document(page_content="The company reported strong revenue growth this quarter."),
        Document(page_content="The organization experienced operational improvements."),
        Document(page_content="Random text unrelated to finance."),
    ]
    retriever = Financial10QRetriever(docs)
    results = retriever.get_relevant_documents("revenue growth")
    assert len(results) > 0
    # Expect the top result to be the document containing 'revenue'
    assert "revenue" in results[0].page_content.lower()
