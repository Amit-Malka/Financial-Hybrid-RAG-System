from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import logging


class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.getLogger("graph").info("Connected to Neo4j")

    def close(self):
        self.driver.close()
        logging.getLogger("graph").info("Closed Neo4j connection")

    def add_document_structure(self, chunks: list[Document], *, doc_title: str | None = None):
        """Create Document/Section/Chunk graph from already built chunks.

        Deletes only the subgraph for the current document title if provided.
        """
        logger = logging.getLogger("graph")
        title = doc_title or "Untitled"
        with self.driver.session() as session:
            # For single-document mode: clear ALL previous documents to prevent mixing
            session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section)
                OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c, s, d
                """
            )
            logger.info("Cleared ALL existing documents for single-document processing")
            
            # Alternative scoped delete (if needed for multi-document mode later):
            # session.run(
            #     """
            #     MATCH (d:Document {title: $title})
            #     OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section)
            #     OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
            #     DETACH DELETE c, s, d
            #     """,
            #     title=title,
            # )

            # Ensure document node
            session.run(
                "CREATE (d:Document {title: $title})",
                title=title,
            )

            # Prepare rows for Sections and Chunks
            section_rows = []
            chunk_rows = []
            for ch in chunks:
                md = ch.metadata or {}
                section_path = md.get("section_path", "") or ""
                section_rows.append({"title": title, "section_path": section_path})
                chunk_rows.append(
                    {
                        "title": title,
                        "section_path": section_path,
                        "chunk_id": md.get("chunk_id"),
                        "text": ch.page_content,
                        "page_number": md.get("page_number"),
                        "element_type": md.get("element_type"),
                        "content_type": md.get("content_type"),
                    }
                )

            # Create sections (distinct by section_path)
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Document {title: row.title})
                MERGE (s:Section {title: row.title, path: row.section_path})
                MERGE (d)-[:CONTAINS]->(s)
                """,
                rows=section_rows,
            )

            # Create chunks with properties, attach to sections
            session.run(
                """
                UNWIND $rows AS row
                MERGE (s:Section {title: row.title, path: row.section_path})
                CREATE (c:Chunk {
                  chunk_id: row.chunk_id,
                  text: row.text,
                  page_number: row.page_number,
                  section_path: row.section_path,
                  element_type: row.element_type,
                  content_type: row.content_type
                })
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                rows=chunk_rows,
            )

            # Create NEXT relationships between neighboring chunks (by numeric order)
            session.run(
                """
                MATCH (d:Document {title: $title})-[:CONTAINS]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, c ORDER BY toInteger(replace(c.chunk_id, 'chunk_', ''))
                WITH collect(c) AS chunks
                UNWIND range(0, size(chunks)-2) AS i
                WITH chunks[i] AS c1, chunks[i+1] AS c2
                MERGE (c1)-[:NEXT]->(c2)
                """,
                title=title,
            )
            logger.info(
                f"Inserted document graph for '{title}': {len(set([r['section_path'] for r in section_rows]))} sections, {len(chunk_rows)} chunks"
            )

            # Optional: create SIMILAR_TO edges using embeddings
            from ..config import Config  # local import to avoid cycles at import time
            if getattr(Config, "ENABLE_SIMILAR_TO", False) and chunk_rows:
                logger.info("Building SIMILAR_TO edges using embeddings")
                # Compute embeddings for chunks' text
                texts = [row["text"] or "" for row in chunk_rows]
                ids = [row["chunk_id"] for row in chunk_rows]
                hf = HuggingFaceEmbeddings()
                vectors = hf.embed_documents(texts)
                vectors = np.array(vectors, dtype=float)
                # Normalize
                norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
                nv = vectors / norms
                # Cosine similarity matrix
                sims = nv @ nv.T
                top_n = int(getattr(Config, "SIMILAR_TOP_N", 5))
                threshold = float(getattr(Config, "SIMILARITY_THRESHOLD", 0.7))

                rel_rows = []
                for i, src in enumerate(ids):
                    # Exclude self
                    sims[i, i] = -1.0
                    # Top-N indices above threshold
                    nn_idx = np.argsort(-sims[i])[: top_n * 2]  # take a few extra then filter
                    added = 0
                    for j in nn_idx:
                        if sims[i, j] < threshold:
                            continue
                        rel_rows.append({
                            "src": src,
                            "dst": ids[j],
                            "score": float(sims[i, j]),
                        })
                        added += 1
                        if added >= top_n:
                            break

                if rel_rows:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (c1:Chunk {chunk_id: row.src})
                        MATCH (c2:Chunk {chunk_id: row.dst})
                        MERGE (c1)-[r:SIMILAR_TO]->(c2)
                        SET r.score = row.score
                        """,
                        rows=rel_rows,
                    )
                    logger.info(f"Created {len(rel_rows)} SIMILAR_TO relations")