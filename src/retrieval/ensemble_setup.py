from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from ..config import Config
from .graph_retriever import GraphEnhancedRetriever


def create_ensemble_retriever(dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever) -> EnsembleRetriever:
    """Creates an EnsembleRetriever with the given dense and sparse retrievers."""
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[Config.DENSE_WEIGHT, Config.TFIDF_WEIGHT]
    )


def create_graph_enhanced_retriever(
    dense_retriever: BaseRetriever, 
    sparse_retriever: BaseRetriever, 
    neo4j_graph=None
) -> GraphEnhancedRetriever:
    """Creates a graph-enhanced retriever per specification architecture."""
    # First create the ensemble retriever (Dense 70% + TF-IDF 30%)
    ensemble_retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
    
    # Then wrap with graph enhancement (15% boost)
    return GraphEnhancedRetriever(
        base_retriever=ensemble_retriever,
        neo4j_graph=neo4j_graph,
        enhancement_weight=Config.GRAPH_ENHANCEMENT_WEIGHT
    )
