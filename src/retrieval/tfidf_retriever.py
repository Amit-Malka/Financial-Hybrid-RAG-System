from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..config import Config

class Financial10QRetriever(BaseRetriever):
    """A custom TF-IDF retriever for 10-Q financial documents."""

    def __init__(self, documents: List[Document]):
        super().__init__()
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform([doc.page_content for doc in self.documents])

        # Boost financial terms
        financial_terms_indices = [self.vectorizer.vocabulary_.get(term) for term in Config.FINANCIAL_10Q_TERMS if term in self.vectorizer.vocabulary_]
        for idx in financial_terms_indices:
            self.tfidf_matrix[:, idx] *= Config.FINANCIAL_BOOST

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        relevant_doc_indices = np.argsort(similarity_scores)[::-1][:Config.DEFAULT_TOP_K]
        return [self.documents[i] for i in relevant_doc_indices]
