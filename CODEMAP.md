### Codebase Map

This document inventories all Python modules, their imports, top-level exports (classes/functions), and function/method signatures with inputs/outputs (where explicit). It reflects the current repository state.

#### src/config.py
- Imports: none
- Exports: `Config`
- Members:
  - Constants: `CHUNK_SIZE:int`, `CHUNK_OVERLAP:int`, `DENSE_WEIGHT:float`, `TFIDF_WEIGHT:float`, `FINANCIAL_10Q_TERMS:set[str]`, `MAX_FEATURES:int`, `FINANCIAL_BOOST:float`, `TABLE_KEYWORDS:list[str]`, `RISK_KEYWORDS:list[str]`, `MDA_KEYWORDS:list[str]`, `DEFAULT_TOP_K:int`, `GOOGLE_API_KEY:str`, `METADATA_SCHEMA:dict`

#### src/processing/pdf_to_html.py
- Imports: `fitz`
- Exports: `convert_pdf_to_html(pdf_path:str) -> str`

#### src/processing/pdf_parser.py
- Imports: `sec_parser.SecParser`, `AbstractSemanticElement`
- Exports: `load_html(file_path:str) -> list[AbstractSemanticElement]`

#### src/processing/chunker.py
- Imports: `langchain_core.documents.Document`, `AbstractSemanticElement`, `TopSectionTitle`
- Exports:
  - `chunk_document(elements:list[AbstractSemanticElement]) -> list[Document]`
  - `get_section_chunks(elements:list[AbstractSemanticElement], section_type:type) -> list[Document]`
  - `get_elements_in_section(elements:list[AbstractSemanticElement], *, section_identifier:str) -> list[AbstractSemanticElement]`

#### src/retrieval/dense_retriever.py
- Imports: `Chroma`, `HuggingFaceEmbeddings`, `BaseRetriever`, `Document`, `typing.List`
- Exports: `get_dense_retriever(documents:List[Document]) -> BaseRetriever`

#### src/retrieval/tfidf_retriever.py
- Imports: `BaseRetriever`, `CallbackManagerForRetrieverRun`, `Document`, `typing.List`, `TfidfVectorizer`, `cosine_similarity`, `numpy as np`, `Config`
- Exports: `class Financial10QRetriever(BaseRetriever)`
  - `__init__(self, documents: List[Document]) -> None`
    - builds TF-IDF matrix; applies feature weights from `Config.FINANCIAL_10Q_TERMS`
  - `_get_relevant_documents(self, query:str, *, run_manager:CallbackManagerForRetrieverRun) -> List[Document]`

#### src/retrieval/ensemble_setup.py
- Imports: `langchain.retrievers.EnsembleRetriever`, `BaseRetriever`, `Config`
- Exports: `create_ensemble_retriever(dense:BaseRetriever, sparse:BaseRetriever) -> EnsembleRetriever`

#### src/tools/base.py
- Imports: `BaseRetriever`, `BaseLanguageModel`
- Exports: `class SimpleTool`
  - `__init__(self, retriever:BaseRetriever, llm:BaseLanguageModel) -> None`
  - `execute(self, query:str) -> str` uses retriever, builds prompt, invokes LLM; returns text

#### src/tools/general_tool.py
- Imports: `.base.SimpleTool`, `BaseRetriever`, `BaseLanguageModel`
- Exports: `class GeneralTool(SimpleTool)`
  - `__init__(self, retriever:BaseRetriever, llm:BaseLanguageModel) -> None`

#### src/tools/mda_tool.py
- Imports: `.base.SimpleTool`, `BaseRetriever`, `BaseLanguageModel`, `get_dense_retriever`, `Financial10QRetriever`, `create_ensemble_retriever`, `chunk_document`, `get_elements_in_section`
- Exports: `class MDATool(SimpleTool)`
  - `__init__(self, llm:BaseLanguageModel, elements:list) -> None`
    - selects elements in 10-Q section `part1item2`, chunks, builds dense+sparse retrievers, ensembles

#### src/tools/risk_tool.py
- Imports: `.base.SimpleTool`, `BaseLanguageModel`, `get_dense_retriever`, `Financial10QRetriever`, `create_ensemble_retriever`, `chunk_document`, `get_elements_in_section`
- Exports: `class RiskTool(SimpleTool)`
  - `__init__(self, llm:BaseLanguageModel, elements:list) -> None`
    - selects elements in 10-Q section `part2item1a`, chunks, builds retrievers, ensembles

#### src/tools/table_tool.py
- Imports: `.base.SimpleTool`, `BaseRetriever`, `BaseLanguageModel`, `LLMTextCompletionProgram`, `Pydantic BaseModel/Field`, `PydanticOutputParser`, `sec_parser.TableElement`
- Exports:
  - `class TableAnswer(BaseModel)` with field `answer:str`
  - `class TableTool(SimpleTool)`
    - `__init__(self, retriever:BaseRetriever, llm:BaseLanguageModel, elements:list) -> None`
      - builds a LlamaIndex program for table QA
    - `execute(self, query:str) -> str` finds first `TableElement`, runs program, returns `answer`

#### src/tools/router.py
- Imports: `Config`
- Exports: `route_query(query:str) -> str` chooses tool by keywords

#### src/ui/gradio_app.py
- Imports: `gradio as gr`, `os`, `shutil`, `route_query`, tools (`GeneralTool`, `TableTool`, `MDATool`, `RiskTool`), retrievers, `LangchainLLM`, processing (`convert_pdf_to_html`, `load_html`, `chunk_document`), `Neo4jGraph`, `evaluate_ragas`, `ChatGoogleGenerativeAI`, `Document`, `ContextualCompressionRetriever`, `CohereRerank`
- Exports: Gradio app (module entrypoint)
- Functions:
  - `process_file(file, api_key) -> str`
    - writes PDF to `data/uploads`, converts to HTML, parses to elements, chunks, builds ensemble retriever
  - `add_to_graph(neo4j_uri, neo4j_user, neo4j_password) -> str`
    - writes elements as nodes with `:NEXT` links
  - `answer_question_for_app(question, api_key, cohere_api_key, use_reranker) -> str`
  - `answer_question_and_context(question, api_key, cohere_api_key, use_reranker) -> tuple[str, list[Document]]`
    - sets API keys, builds LLMs, optional reranker, instantiates tools, routes and runs, returns answer and context
  - `run_evaluation(question, ground_truth, api_key, cohere_api_key, use_reranker) -> Any`
    - calls RAGAS evaluation
  - Module main: launches Gradio interface

#### src/llm/langchain_llm.py
- Imports: `llama_index.core.llms.LLM`, `BaseLanguageModel`, `llm_completion_callback`
- Exports: `class LangchainLLM(LLM)`
  - `__init__(self, llm:BaseLanguageModel) -> None`
  - `_complete(self, prompt:str, **kwargs) -> str`
  - `_stream_complete(self, prompt:str, **kwargs)` (not implemented)
  - `metadata` property -> `{}`

#### src/llm/dummy_llm.py
- Imports: `BaseLanguageModel`, typing, `CallbackManagerForLLMRun`
- Exports: `class DummyLLM(BaseLanguageModel)`
  - `_generate(self, prompts:List[str], ...) -> str`
  - `_llm_type(self) -> str`

#### src/graph/neo4j_graph.py
- Imports: `GraphDatabase`, `AbstractSemanticElement`
- Exports: `class Neo4jGraph`
  - `__init__(self, uri, user, password) -> None`
  - `close(self) -> None`
  - `add_document_structure(self, elements:list[AbstractSemanticElement]) -> None`

#### src/evaluation/ragas_evaluation.py
- Imports: `ragas.evaluate`, metrics, `datasets.Dataset`
- Exports: `evaluate_ragas(question:str, answer:str, context:list[Document], ground_truth:str)` -> result

#### tests/*.py
- `tests/test_processing.py`: tests `chunk_document` returns LangChain `Document`s
- `tests/test_retrieval.py`: tests TF-IDF retriever ranks a revenue doc first
- `tests/test_tools.py`: tests `SimpleTool` executes with injected Echo retriever/LLM
- `tests/test_router.py`: tests routing for table/risk/mda/general

### Notes on I/O expectations vs. provided
- Processing
  - `load_html` expects HTML path; returns `list[AbstractSemanticElement]`
  - `chunk_document` expects semantic elements; returns `list[langchain_core.documents.Document]`
  - `get_elements_in_section` expects `TopSectionTitle` markers present in elements; returns subset
- Retrieval
  - Dense and TF-IDF expect LangChain `Document`s; ensemble returns via LangChain BaseRetriever API
  - TF-IDF boosts financial terms using column-wise scaling
- Tools
  - All tools expect a working `BaseRetriever` and `BaseLanguageModel` compatible with `.invoke()`
  - `TableTool` expects presence of `TableElement` among `elements`
- UI
  - `process_file` expects a PDF file; writes HTML; builds retrievers from chunked elements
  - Reranker requires `COHERE_API_KEY` when enabled
  - Answer path returns string answer and LangChain `Document` context


