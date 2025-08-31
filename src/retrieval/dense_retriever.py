import os
# Fix ChromaDB telemetry errors completely - multiple environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False" 
os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"

# Import config first to ensure telemetry is disabled
from ..config import Config

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import logging

# Try to disable ChromaDB telemetry programmatically
try:
    import chromadb
    # Disable telemetry at the chromadb level if possible
    chromadb.telemetry.disable()
except (ImportError, AttributeError):
    # ChromaDB version doesn't support this method, environment variables should work
    pass

def get_dense_retriever(documents: List[Document]) -> BaseRetriever:
    """Creates a dense retriever using ChromaDB and HuggingFace embeddings."""
    logger = logging.getLogger("retrieval.dense")
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": getattr(Config, "DEFAULT_TOP_K", 5)})
    logger.info(f"Dense retriever built with {len(documents)} docs; top_k={getattr(Config, 'DEFAULT_TOP_K', 5)}")
    return retriever