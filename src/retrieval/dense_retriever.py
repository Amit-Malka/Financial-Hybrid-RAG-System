from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import logging
from ..config import Config

def get_dense_retriever(documents: List[Document]) -> BaseRetriever:
    """Creates a dense retriever using ChromaDB and HuggingFace embeddings."""
    logger = logging.getLogger("retrieval.dense")
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": getattr(Config, "DEFAULT_TOP_K", 5)})
    logger.info(f"Dense retriever built with {len(documents)} docs; top_k={getattr(Config, 'DEFAULT_TOP_K', 5)}")
    return retriever