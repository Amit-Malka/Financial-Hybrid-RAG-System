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
                # Optionally get similar documents
                similar_docs = self._get_similar_documents(chunk_id)
                
                # Add unique neighbors to enhanced results
                for neighbor in neighbors + section_docs + similar_docs:
                    if neighbor and isinstance(neighbor.metadata, dict):
                        if neighbor in neighbors:
                            neighbor.metadata["graph_source"] = "NEXT"
                        elif neighbor in section_docs:
                            neighbor.metadata["graph_source"] = "SECTION"
                        else:
                            neighbor.metadata["graph_source"] = "SIMILAR_TO"
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
            if not self.neo4j_graph or not getattr(self.neo4j_graph, "driver", None):
                return []
            with self.neo4j_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c1:Chunk {chunk_id: $chunk_id})-[:NEXT]->(c2:Chunk)
                    RETURN c2.chunk_id as chunk_id, c2.text as content, c2.page_number as page_number,
                           c2.section_path as section_path, c2.element_type as element_type, c2.content_type as content_type
                    """,
                    chunk_id=chunk_id,
                )
                neighbors: List[Document] = []
                for record in result:
                    metadata = {
                        "chunk_id": record.get("chunk_id"),
                        "page_number": record.get("page_number"),
                        "section_path": record.get("section_path"),
                        "content_type": record.get("content_type"),
                        "element_type": record.get("element_type"),
                    }
                    neighbors.append(Document(page_content=record.get("content") or "", metadata={k: v for k, v in metadata.items() if v is not None}))
                return neighbors
        except Exception as e:
            self._logger.warning(f"Neo4j NEXT query failed: {e}")
            return []
    
    def _get_section_documents(self, section_path: str) -> List[Document]:
        """Get related documents from the same section."""
        try:
            if not section_path:
                return []
            if not self.neo4j_graph or not getattr(self.neo4j_graph, "driver", None):
                return []
            with self.neo4j_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk {section_path: $section_path})
                    RETURN c.chunk_id as chunk_id, c.text as content, c.page_number as page_number,
                           c.section_path as section_path, c.element_type as element_type, c.content_type as content_type
                    LIMIT 25
                    """,
                    section_path=section_path,
                )
                docs: List[Document] = []
                for record in result:
                    metadata = {
                        "chunk_id": record.get("chunk_id"),
                        "page_number": record.get("page_number"),
                        "section_path": record.get("section_path"),
                        "content_type": record.get("content_type"),
                        "element_type": record.get("element_type"),
                    }
                    docs.append(Document(page_content=record.get("content") or "", metadata={k: v for k, v in metadata.items() if v is not None}))
                return docs
        except Exception as e:
            self._logger.warning(f"Neo4j section query failed: {e}")
            return []

    def _get_similar_documents(self, chunk_id: str) -> List[Document]:
        """Get embedding-similar chunks via SIMILAR_TO edges if enabled."""
        try:
            from ..config import Config
            if not getattr(Config, "ENABLE_SIMILAR_TO", False):
                return []
            if not self.neo4j_graph or not getattr(self.neo4j_graph, "driver", None):
                return []
            with self.neo4j_graph.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c1:Chunk {chunk_id: $chunk_id})-[:SIMILAR_TO]->(c2:Chunk)
                    RETURN c2.chunk_id as chunk_id, c2.text as content, c2.page_number as page_number,
                           c2.section_path as section_path, c2.element_type as element_type, c2.content_type as content_type
                    ORDER BY coalesce(c2.page_number, 1e9)
                    LIMIT 25
                    """,
                    chunk_id=chunk_id,
                )
                docs: List[Document] = []
                for record in result:
                    metadata = {
                        "chunk_id": record.get("chunk_id"),
                        "page_number": record.get("page_number"),
                        "section_path": record.get("section_path"),
                        "content_type": record.get("content_type"),
                        "element_type": record.get("element_type"),
                    }
                    docs.append(Document(page_content=record.get("content") or "", metadata={k: v for k, v in metadata.items() if v is not None}))
                return docs
        except Exception as e:
            self._logger.warning(f"Neo4j SIMILAR_TO query failed: {e}")
            return []
