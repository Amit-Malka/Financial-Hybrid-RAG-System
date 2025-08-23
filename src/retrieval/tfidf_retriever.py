from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import PrivateAttr, Field
import logging
from ..config import Config

class Financial10QRetriever(BaseRetriever):
    """A custom TF-IDF retriever for 10-Q financial documents."""

    # Pydantic fields
    documents: List[Document] = Field(default_factory=list)

    # Private runtime attributes
    _vectorizer: TfidfVectorizer = PrivateAttr()
    _tfidf_matrix: Any = PrivateAttr()

    def __init__(self, documents: List[Document]):
        super().__init__(documents=documents)
        logger = logging.getLogger("retrieval.tfidf")
        self._vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english'
        )
        # Fit TF-IDF
        self._tfidf_matrix = self._vectorizer.fit_transform([doc.page_content for doc in self.documents])
        logger.info(f"TFIDF fit on {len(self.documents)} docs with {self._tfidf_matrix.shape[1]} features")

        # Build a feature weight vector to boost financial terms
        vocab = self._vectorizer.vocabulary_ or {}
        num_features = len(vocab)
        feature_weights = np.ones(num_features, dtype=np.float32)
        for term in Config.FINANCIAL_10Q_TERMS:
            idx = vocab.get(term)
            if idx is not None:
                feature_weights[idx] = Config.FINANCIAL_BOOST
        # Apply weights to TF-IDF matrix effectively: scale columns by weights
        self._tfidf_matrix = self._tfidf_matrix.tocsc(copy=True)
        for j in range(self._tfidf_matrix.shape[1]):
            if feature_weights[j] != 1.0:
                start_ptr, end_ptr = self._tfidf_matrix.indptr[j], self._tfidf_matrix.indptr[j + 1]
                self._tfidf_matrix.data[start_ptr:end_ptr] *= feature_weights[j]
        self._tfidf_matrix = self._tfidf_matrix.tocsr(copy=True)
        logger.debug("Applied financial term boosting to TFIDF matrix")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        query_vector = self._vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self._tfidf_matrix).flatten()
        relevant_doc_indices = np.argsort(similarity_scores)[::-1][:Config.DEFAULT_TOP_K]
        logging.getLogger("retrieval.tfidf").debug(f"Top indices: {relevant_doc_indices}")
        return [self.documents[i] for i in relevant_doc_indices]
