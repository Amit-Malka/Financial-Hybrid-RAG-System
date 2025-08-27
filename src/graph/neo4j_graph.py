from neo4j import GraphDatabase
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement
import logging
from ..config import Config

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.getLogger("graph").info("Connected to Neo4j")

    def close(self):
        self.driver.close()
        logging.getLogger("graph").info("Closed Neo4j connection")

    def add_document_structure(self, elements: list[AbstractSemanticElement]):
        with self.driver.session() as session:
            # Clear the graph first
            session.run("MATCH (n) DETACH DELETE n")
            logging.getLogger("graph").warning("Graph cleared (DETACH DELETE n)")

            # Create nodes for each element, persisting 5-field metadata
            for i, element in enumerate(elements):
                # Derive chunk_id consistent with processing.chunker
                chunk_id = f"{Config.CHUNK_ID_PREFIX}{i}"
                element_type = element.__class__.__name__
                text_content = str(element)
                page_number = getattr(element, "page_number", None)
                section_path = getattr(element, "section_path", None)
                content_type = getattr(element, "content_type", None)

                session.run(
                    """
                    CREATE (e:Element {
                        id: $id,
                        chunk_id: $chunk_id,
                        type: $type,
                        text: $text,
                        page_number: $page_number,
                        section_path: $section_path,
                        content_type: $content_type
                    })
                    """,
                    id=i,
                    chunk_id=chunk_id,
                    type=element_type,
                    text=text_content,
                    page_number=page_number,
                    section_path=section_path,
                    content_type=content_type,
                )
            logging.getLogger("graph").info(f"Inserted {len(elements)} nodes with metadata")

            # Create relationships between consecutive elements
            for i in range(len(elements) - 1):
                session.run(
                    "MATCH (e1:Element {id: $id1}), (e2:Element {id: $id2}) "
                    "CREATE (e1)-[:NEXT]->(e2)",
                    id1=i, id2=i+1
                )
            logging.getLogger("graph").info("Created NEXT relationships between consecutive elements")