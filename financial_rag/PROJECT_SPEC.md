# Financial RAG System - Project Specification

## 🎯 Project Overview

**Goal**: Develop a metadata-driven hybrid RAG agent for SEC 10-Q quarterly reports that provides answers from documents containing text, tables, and charts.

**Test Case**: SEC 10-Q quarterly reports (up to 50 pages, single document focus)

**Key Requirements**:
- Hybrid retrieval (Dense + Graph) with metadata
- Optional reranker (user-controlled toggle)
- 5 metadata fields per chunk optimized for single documents
- Graph database integration for document structure relationships
- LlamaIndex-based table processing for financial data
- Complete modularity with no hardcoded responses

## 🏗️ Architecture Components

### Core Technologies
- **Python** - Main development language
- **ChromaDB** - Vector database
- **LlamaIndex** - Document processing, table analysis, and orchestration
- **Gradio** - User interface with configurable controls
- **Gemini Flash 2.5** - LLM via Google API
- **Neo4j** - Graph database for document structure
- **GitHub** - Version control and collaboration
- **LangChain** - EnsembleRetriever for hybrid search coordination

### Hybrid Retrieval Architecture
```
Query → EnsembleRetriever → [Dense (70%) + TF-IDF Sparse (30%)] → Graph Enhancement (15%) → Results
              ↓                    ↓                   ↓
          ChromaDB           Financial TF-IDF        Neo4j
         (Semantic)        (Keyword + Domain)    (Structure)
```

### System Architecture
```
UI Layer (Gradio) → Modular Tools → Hybrid Retriever → Storage (Chroma + Neo4j)
                         ↓              ↓
                   LlamaIndex      EnsembleRetriever
                   Components      (Dense + TF-IDF)
```

## 📋 Implementation Phases (Simplified)

### Phase 1: Core System (Week 1)
- [x] Basic LlamaIndex PDF processing for 10-Q structure
- [x] Simple chunking respecting Part/Item boundaries  
- [x] ChromaDB storage with 3-field metadata
- [x] Basic Gradio interface

### Phase 2: Hybrid Retrieval (Week 2)
- [x] Custom TF-IDF retriever with 10-Q financial terms
- [x] LangChain EnsembleRetriever (Dense 80% + TF-IDF 20%)
- [x] Simple tool router for different query types
- [x] Working Q&A functionality

### Phase 3: Specialized Tools (Week 3)
- [x] LlamaIndex table analysis for financial statements
- [x] MD&A text analysis tool
- [x] Risk factor finder tool
- [x] Complete Gradio interface with tool selection

### Phase 4: Polish & Evaluation (Week 4)
- [x] RAGAS evaluation on 10-Q specific test cases
- [x] Optional Neo4j graph integration (if time permits)
- [ ] Performance optimization and user testing
- [ ] Documentation and final demo preparation

## 🗂️ Metadata Schema (3 Fields - 10-Q Optimized)

```python
metadata_schema = {
    "section_path": [
        "part_i/item_1/financial_statements",
        "part_i/item_2/md_a", 
        "part_i/item_3/market_risk",
        "part_i/item_4/controls",
        "part_ii/item_1a/risk_factors",
        "part_ii/other"
    ],
    "content_type": [
        "balance_sheet", "income_statement", "cash_flow", 
        "notes", "md_a_narrative", "risk_disclosure", "legal"
    ],
    "page_number": int  # For citations and anchoring
}
```

### 10-Q Structure Understanding
```
PART I (Financial Information)
├── Item 1: Financial Statements (Tables + Notes)
├── Item 2: MD&A (Management Discussion & Analysis)  
├── Item 3: Market Risk Disclosures
└── Item 4: Controls and Procedures

PART II (Other Information)
├── Item 1A: Risk Factors (Updates only)
├── Item 1: Legal Proceedings
└── Item 5: Other Information
```

## 🛠️ Gradio Interface Tabs

1. **Q&A Chat** - Classic chatbot for finding paragraphs/sections
2. **Summary** - Full document summarization
3. **Table Queries** - Numeric/tabular data questions
4. **Financial Analysis** - Health assessment and risk factor analysis
5. **System Info** - Metadata display, evaluation scores, system metrics
6. **Configuration** - Google API key setup

## 🔧 Simplified Tools Design

### Core Tool Architecture
```python
class SimpleTool:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def execute(self, query: str) -> str:
        # Simple: retrieve → generate → return
        context = self.retriever.get_relevant_documents(query)
        return self.llm.invoke(f"Context: {context}\n\nQuestion: {query}")

# Tool router based on query type
def route_query(query: str) -> str:
    if "table" in query.lower() or "revenue" in query.lower():
        return "table_tool"
    elif "risk" in query.lower():
        return "risk_tool"  
    elif "management" in query.lower() or "md&a" in query.lower():
        return "mda_tool"
    else:
        return "general_tool"
```

### 10-Q Specific Tools
- **TableTool**: LlamaIndex table analysis for Item 1 (Financial Statements)
- **MDANavigator**: Text analysis for Item 2 (Management Discussion)  
- **RiskFinder**: Specialized search for Item 1A (Risk Factors)
- **GeneralSearch**: Default tool for everything else

### Simple EnsembleRetriever Setup
```python
# Dense retriever (ChromaDB)
dense_retriever = vector_store.as_retriever()

# TF-IDF with 10-Q financial terms  
tfidf_retriever = Financial10QRetriever(documents)

# Simple ensemble (no complex weighting)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, tfidf_retriever],
    weights=[0.8, 0.2]  # Dense-heavy for semantic understanding
)
```

## 📊 Graph Database Schema (Neo4j - Simplified)

### Node Types (Document Structure Focus)
- `Document` - Root quarterly report node
- `Section` - Major document sections (MD&A, Financial Statements, etc.)
- `Chunk` - Text chunks with embeddings for retrieval

### Relationships (Simple Hierarchy)
```cypher
(:Document {title, fiscal_quarter, upload_date})
(:Section {name, type, page_start, page_end})
(:Chunk {text, embedding, page_number, chunk_id})

# Simple document structure relationships
(:Document)-[:CONTAINS]->(:Section)
(:Section)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:NEXT]->(:Chunk)           # Sequential document flow
(:Chunk)-[:SIMILAR_TO]->(:Chunk)     # Semantic similarity connections
```

### Graph Usage
- **Document Navigation**: Follow sequential chunk relationships
- **Section Discovery**: Find all chunks within specific sections
- **Contextual Retrieval**: Use similarity relationships for expansion
- **Citation Support**: Track page numbers and section hierarchies

## 📈 Evaluation Metrics

### Required Thresholds
- **Context Precision** ≥ 0.75
- **Context Recall** ≥ 0.70
- **Faithfulness** ≥ 0.85
- **Table-QA Accuracy** ≥ 0.90

### Evaluation Strategy
- Ground Truth: Not context-sensitive (simulates chunking division)
- RAGAS framework implementation
- Real-time metrics display in UI
- Continuous performance monitoring

## 📁 Project Structure (Simplified)

```
financial_rag/
├── src/
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py          # LlamaIndex 10-Q processing
│   │   └── chunker.py             # Simple section-aware chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── tfidf_retriever.py     # Custom 10-Q TF-IDF
│   │   └── ensemble_setup.py      # Simple EnsembleRetriever
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── table_tool.py          # Financial statements (LlamaIndex)
│   │   ├── mda_tool.py            # MD&A analysis
│   │   ├── risk_tool.py           # Risk factors finder
│   │   └── general_tool.py        # Default search
│   ├── ui/
│   │   ├── __init__.py
│   │   └── gradio_app.py          # Main interface
│   └── config.py                  # Simple configuration
├── data/
│   ├── uploads/                   # 10-Q document uploads
│   └── processed/                 # Chunks and embeddings
├── tests/
│   ├── test_processing.py
│   ├── test_retrieval.py
│   └── test_tools.py
├── requirements.txt
├── README.md
├── PROJECT_SPEC.md               
└── .gitignore
```

## 🚀 Simple Development Guidelines

### Core Principles (Streamlined)
- **10-Q First**: Design everything around actual SEC 10-Q structure
- **Simple Stack**: LlamaIndex + ChromaDB + LangChain EnsembleRetriever
- **No Hardcoding**: Configuration-driven behavior
- **Tool Specialization**: Each tool handles specific 10-Q sections
- **Dense-Heavy**: 80% semantic search, 20% keyword search

### Development Rules
1. **Respect 10-Q Structure**: Never split across Parts or Items when chunking
2. **LlamaIndex for Tables**: Use their proven financial table processing
3. **Simple Tools**: Basic retrieve → generate → return pattern
4. **Query Routing**: Route based on keywords, not complex logic
5. **Optional Graph**: Add Neo4j only if Phase 4 has time

### Technology Stack (Minimal)
- **Document Processing**: LlamaIndex (handles 10-Q parsing)
- **Vector Storage**: ChromaDB (simple setup)
- **Hybrid Search**: LangChain EnsembleRetriever
- **Table Analysis**: LlamaIndex table tools
- **UI**: Gradio (3 simple tabs)
- **LLM**: Gemini Flash 2.5

### Success Criteria (Simplified)
- Upload 10-Q → get accurate answers about financial data
- Tables work with LlamaIndex processing
- MD&A and Risk Factor sections searchable
- Simple but effective hybrid retrieval
- Clean, working Gradio interface

## ⚙️ Simple Configuration

```python
# config.py
class Config:
    # Core settings
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    
    # Hybrid retrieval (simplified)
    DENSE_WEIGHT = 0.8
    TFIDF_WEIGHT = 0.2
    
    # 10-Q specific TF-IDF terms
    FINANCIAL_10Q_TERMS = {
        # Core financial
        'revenue', 'assets', 'liabilities', 'equity', 'cash_flow',
        # 10-Q specific
        'quarterly', 'interim', 'unaudited', 'condensed',
        'yoy', 'quarter_over_quarter', 'guidance', 'outlook',
        # SEC specific
        'md_a', 'risk_factors', 'forward_looking', 'material'
    }
    
    # Simple TF-IDF settings
    MAX_FEATURES = 5000
    FINANCIAL_BOOST = 2.0
    
    # Tools routing keywords
    TABLE_KEYWORDS = ['revenue', 'income', 'balance', 'cash_flow', 'financial_statement']
    RISK_KEYWORDS = ['risk', 'uncertainty', 'factor', 'may_adversely']
    MDA_KEYWORDS = ['management', 'discussion', 'analysis', 'outlook', 'results']
    
    # UI settings
    DEFAULT_TOP_K = 5
    GOOGLE_API_KEY = ""  # User provided
```

### Gradio Interface (Simplified)
```python
# Simple 3-tab interface
tabs = {
    "Q&A": "Ask questions about the 10-Q",
    "Tables": "Query financial data and tables", 
    "Settings": "Configure search weights and API key"
}
```

## ⚠️ Critical Success Factors

1. **LangChain EnsembleRetriever**: Use proven LangChain coordination for dense + TF-IDF hybrid search rather than custom combination logic.

2. **Financial TF-IDF Implementation**: Custom retriever must inherit from BaseRetriever, include domain-specific term boosting, and integrate seamlessly with EnsembleRetriever.

3. **User-Controlled Reranker**: Implement reranker as optional toggle. Never apply by default. Users decide when to enable based on their needs.

4. **No Hardcoded Responses**: All tool outputs must come from actual retrieval and LLM generation. Zero mocked or stub responses.

5. **Configuration-Driven Architecture**: Every system behavior controlled via settings.py. TF-IDF parameters, weights, and financial terms all configurable.

6. **True Modularity**: Each component (TF-IDF retriever, dense retriever, graph store, tools) must be independently swappable.

7. **Single Document Optimization**: Focus on 50-page quarterly reports. Don't over-engineer for multi-document scenarios initially.

8. **LlamaIndex Table Processing**: Use their proven table analysis tools rather than custom table parsing implementations.

9. **Financial Domain Knowledge**: TF-IDF retriever must boost financial terms and phrases relevant to SEC 10-Q documents.

10. **Graph as Enhancement**: Neo4j provides document structure navigation, not primary retrieval. Keep graph schema simple.

## 🎯 Success Metrics

- All RAGAS evaluation thresholds met (Context Precision ≥ 0.75, Context Recall ≥ 0.70, Faithfulness ≥ 0.85, Table-QA Accuracy ≥ 0.90)
- Functional LangChain EnsembleRetriever with custom TF-IDF integration
- Working financial term boosting with measurable relevance improvements
- Intuitive Gradio interface with working user controls for all retrieval weights
- Complete modularity demonstrated through component swapping
- Zero hardcoded responses in any tool
- Successful reranker toggle functionality
- LlamaIndex table processing integration working
- TF-IDF retriever properly inheriting from LangChain BaseRetriever
- Simple, maintainable codebase under 2500 lines
- Demonstration of future scalability potential

## 📝 Implementation Notes

- **Ground Truth**: Generate evaluation datasets that simulate chunking divisions for RAGAS
- **Summary Strategy**: Full document summarization using LlamaIndex, not chunk-based approaches  
- **Financial Analysis**: Combine health assessment with risk factor identification via hybrid retrieval
- **Graph Relationships**: Focus on document structure (sections, chunks, pages) not financial entities
- **TF-IDF Integration**: Must work seamlessly with LangChain's EnsembleRetriever for clean coordination
- **Domain Expertise**: Financial term dictionary should be expandable and configurable
- **User Experience**: Provide clear feedback on retrieval results, term boosting effects, and system configuration state