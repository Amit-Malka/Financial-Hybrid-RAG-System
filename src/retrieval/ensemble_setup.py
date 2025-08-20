from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from ..config import Config

def create_ensemble_retriever(dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever) -> EnsembleRetriever:
    """Creates an EnsembleRetriever with the given dense and sparse retrievers."""
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[Config.DENSE_WEIGHT, Config.TFIDF_WEIGHT]
    )
