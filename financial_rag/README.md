# Financial RAG System

This project is a metadata-driven hybrid RAG agent for SEC 10-Q quarterly reports that provides answers from documents containing text, tables, and charts.

## Project Overview

**Goal**: Develop a metadata-driven hybrid RAG agent for SEC 10-Q quarterly reports that provides answers from documents containing text, tables, and charts.

**Test Case**: SEC 10-Q quarterly reports (up to 50 pages, single document focus)

**Key Requirements**:
- Hybrid retrieval (Dense + Graph) with metadata
- Optional reranker (user-controlled toggle)
- 5 metadata fields per chunk optimized for single documents
- Graph database integration for document structure relationships
- LlamaIndex-based table processing for financial data
- Complete modularity with no hardcoded responses

## Architecture

The system uses a hybrid retrieval architecture with a combination of dense and sparse retrieval methods, enhanced with a graph database for understanding document structure. The core technologies include Python, ChromaDB, LlamaIndex, Gradio, Gemini Flash 2.5, Neo4j, and LangChain.

## Project Structure

The project is organized into the following directories:
- `src`: Contains the source code for the project, divided into modules for processing, retrieval, tools, and UI.
- `data`: Contains the data used in the project, including uploaded 10-Q documents and processed data.
- `tests`: Contains the tests for the project.
