from neo4j import GraphDatabase
from sec_parser.semantic_elements.abstract_semantic_element import AbstractSemanticElement

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_document_structure(self, elements: list[AbstractSemanticElement]):
        with self.driver.session() as session:
            # Clear the graph first
            session.run("MATCH (n) DETACH DELETE n")

            # Create nodes for each element
            for i, element in enumerate(elements):
                session.run(
                    "CREATE (e:Element {id: $id, type: $type, text: $text})",
                    id=i,
                    type=element.__class__.__name__,
                    text=str(element)
                )

            # Create relationships between consecutive elements
            for i in range(len(elements) - 1):
                session.run(
                    "MATCH (e1:Element {id: $id1}), (e2:Element {id: $id2}) "
                    "CREATE (e1)-[:NEXT]->(e2)",
                    id1=i, id2=i+1
                )