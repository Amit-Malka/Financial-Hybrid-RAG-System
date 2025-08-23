from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
from pydantic import PrivateAttr, Field
import logging
from ..config import Config
from ..graph.neo4j_graph import Neo4jGraph


class GraphEnhancedRetriever(BaseRetriever):
    """A retriever that enhances base retrieval with graph database relationships."""

    # Pydantic fields
    base_retriever: BaseRetriever = Field(...)
    neo4j_graph: Any = Field(default=None)  # Optional Neo4j graph instance
    enhancement_weight: float = Field(default=0.15)

    # Private attributes
    _logger: Any = PrivateAttr()

    def __init__(self, base_retriever: BaseRetriever, neo4j_graph=None, enhancement_weight: float = 0.15):
        super().__init__(
            base_retriever=base_retriever,
            neo4j_graph=neo4j_graph,
            enhancement_weight=enhancement_weight
        )
        self._logger = logging.getLogger("retrieval.graph_enhanced")
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents enhanced with graph relationships."""
        
        # Get base retrieval results
        base_documents = self.base_retriever.get_relevant_documents(query)
        self._logger.debug(f"Base retrieval returned {len(base_documents)} documents")
        
        # If no graph or graph enhancement disabled, return base results
        if not self.neo4j_graph or not Config.ENABLE_GRAPH_ENHANCEMENT:
            self._logger.debug("Graph enhancement disabled, returning base results")
            return base_documents
            
        try:
            # Enhance with graph relationships
            enhanced_documents = self._enhance_with_graph(base_documents, query)
            self._logger.debug(f"Graph enhancement returned {len(enhanced_documents)} documents")
            return enhanced_documents
            
        except Exception as e:
            self._logger.warning(f"Graph enhancement failed: {e}, falling back to base results")
            return base_documents
    
    def _enhance_with_graph(self, base_documents: List[Document], query: str) -> List[Document]:
        """Enhance retrieval results using graph database relationships."""
        
        if not base_documents:
            return base_documents
            
        enhanced_docs = list(base_documents)  # Start with base results
        
        try:
            # For each base document, find related documents through graph relationships
            for doc in base_documents:
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    continue
                    
                # Get sequential neighbors (NEXT relationships)
                neighbors = self._get_sequential_neighbors(chunk_id)
                
                # Get section-related documents 
                section_docs = self._get_section_documents(doc.metadata.get("section_path"))
                
                # Add unique neighbors to enhanced results
                for neighbor in neighbors + section_docs:
                    if neighbor not in enhanced_docs:
                        enhanced_docs.append(neighbor)
                        
            # Apply enhancement weight by limiting additional documents
            max_additional = int(len(base_documents) * self.enhancement_weight)
            if len(enhanced_docs) > len(base_documents) + max_additional:
                enhanced_docs = base_documents + enhanced_docs[len(base_documents):len(base_documents) + max_additional]
                
            return enhanced_docs
            
        except Exception as e:
            self._logger.error(f"Graph enhancement processing failed: {e}")
            return base_documents
    
    def _get_sequential_neighbors(self, chunk_id: str) -> List[Document]:
        """Get sequential neighbor chunks using NEXT relationships."""
        try:
            # This is a simplified implementation since we have basic Neo4j integration
            # In a full implementation, this would query Neo4j for NEXT relationships
            return []
        except Exception:
            return []
    
    def _get_section_documents(self, section_path: str) -> List[Document]:
        """Get related documents from the same section."""
        try:
            # This is a simplified implementation
            # In a full implementation, this would query Neo4j for section-related chunks
            return []
        except Exception:
            return []
