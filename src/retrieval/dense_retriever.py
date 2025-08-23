from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import logging

def get_dense_retriever(documents: List[Document]) -> BaseRetriever:
    """Creates a dense retriever using ChromaDB and HuggingFace embeddings."""
    logger = logging.getLogger("retrieval.dense")
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    logger.info(f"Dense retriever built with {len(documents)} docs")
    return vectorstore.as_retriever()