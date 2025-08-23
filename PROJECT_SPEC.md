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

## 📋 Current Implementation Status

### ✅ **COMPLETED COMPONENTS (Phase 1-3)**
- [x] **PDF Processing Pipeline**: PyMuPDF → HTML → SEC-parser conversion
- [x] **Semantic Chunking**: SEC-aware chunking respecting Part/Item boundaries
- [x] **ChromaDB Integration**: Vector storage with HuggingFace embeddings
- [x] **Financial TF-IDF**: Custom retriever with domain-specific term boosting
- [x] **LangChain EnsembleRetriever**: Hybrid search coordination (⚠️ weights need adjustment)
- [x] **Specialized Tools**: TableTool, MDATool, RiskTool, GeneralTool
- [x] **Query Router**: Keyword-based tool routing system
- [x] **LLM Integration**: Gemini Flash 2.5 via Google API with LangChain bridge
- [x] **Optional Reranker**: User-controlled Cohere reranker toggle
- [x] **RAGAS Evaluation**: Complete evaluation framework with all 4 metrics
- [x] **Neo4j Integration**: Basic graph database with document structure
- [x] **Configuration System**: Centralized, modular configuration
- [x] **Logging Infrastructure**: Comprehensive logging and debugging
- [x] **Testing Suite**: Unit tests for core components

### 🔶 **PARTIALLY IMPLEMENTED**
- [🔶] **Metadata Schema**: 4/5 fields implemented (missing `chunk_id`)
- [🔶] **Hybrid Retrieval**: Dense + TF-IDF working, but missing graph enhancement
- [🔶] **Gradio UI**: 2/6 tabs implemented (Q&A and Evaluation only)
- [🔶] **Graph Database**: Storage working, but not integrated into retrieval pipeline

### ❌ **NOT IMPLEMENTED (Critical Gaps)**
- [ ] **Graph-Enhanced Retrieval**: Graph expansion not integrated into main retrieval flow
- [ ] **Complete UI Interface**: Missing 4 specialized tabs (Summary, Table Queries, Financial Analysis, System Info, Configuration)
- [ ] **Correct Retrieval Weights**: Currently 80/20, should be 70/30 per specification
- [ ] **Real-time Metrics Display**: System performance monitoring in UI
- [ ] **User-Configurable Weights**: Dynamic weight adjustment interface

## 🗂️ Metadata Schema (5 Fields - 10-Q Optimized)

### **CURRENT IMPLEMENTATION (4/5 Fields)**
```python
# Currently implemented in src/processing/chunker.py
current_metadata = {
    "element_type": str,     # ✅ SEC element class name
    "page_number": int,      # ✅ PDF page number for citations
    "section_path": str,     # ✅ SEC section identifier (e.g., "part1item2")
    "content_type": str,     # ✅ Content classification
    # "chunk_id": str        # ❌ MISSING - unique chunk identifier
}
```

### **REQUIRED SPECIFICATION (5 Fields)**
```python
# Target metadata schema per specification
required_metadata_schema = {
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
    "page_number": int,      # For citations and anchoring
    "element_type": str,     # SEC semantic element type
    "chunk_id": str          # Unique identifier for each chunk
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

### **CURRENT IMPLEMENTATION (2/6 Tabs)**
1. ✅ **Q&A** - Classic chatbot with file upload, reranker toggle, and Neo4j integration
2. ✅ **Evaluation** - RAGAS evaluation with ground truth comparison

### **MISSING TABS (Required for 100% Compliance)**
3. ❌ **Summary** - Full document summarization using LlamaIndex
4. ❌ **Table Queries** - Specialized interface for numeric/tabular data questions
5. ❌ **Financial Analysis** - Health assessment and risk factor analysis
6. ❌ **System Info** - Metadata display, evaluation scores, system metrics
7. ❌ **Configuration** - User-configurable weights, API key management

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

### Current EnsembleRetriever Setup (⚠️ NEEDS FIXING)
```python
# CURRENT IMPLEMENTATION (INCORRECT WEIGHTS)
dense_retriever = vector_store.as_retriever()
tfidf_retriever = Financial10QRetriever(documents)

# ❌ WRONG: Current weights don't match specification
current_ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, tfidf_retriever],
    weights=[0.8, 0.2]  # Should be [0.7, 0.3] per spec
)

# ❌ MISSING: Graph enhancement layer
# No graph-enhanced retrieval integrated
```

### Required EnsembleRetriever Setup (Per Specification)
```python
# REQUIRED IMPLEMENTATION
dense_retriever = vector_store.as_retriever()
tfidf_retriever = Financial10QRetriever(documents)
graph_retriever = GraphEnhancedRetriever(neo4j_graph)  # ❌ NOT IMPLEMENTED

# ✅ CORRECT: Specification weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, tfidf_retriever],
    weights=[0.7, 0.3]  # Dense 70%, TF-IDF 30%
)

# ✅ REQUIRED: Graph enhancement (15% boost)
final_retriever = GraphEnhancedRetriever(
    base_retriever=hybrid_retriever,
    graph_db=neo4j_graph,
    enhancement_weight=0.15
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

## ⚙️ Configuration Status

### **CURRENT CONFIGURATION (src/config.py)**
```python
# ✅ IMPLEMENTED: Core functionality working
class Config:
    # Core settings
    CHUNK_SIZE = 400                    # ✅ Correct
    CHUNK_OVERLAP = 50                  # ✅ Correct
    
    # ❌ WRONG: Hybrid retrieval weights don't match specification
    DENSE_WEIGHT = 0.8                  # Should be 0.7
    TFIDF_WEIGHT = 0.2                  # Should be 0.3
    # GRAPH_WEIGHT = 0.15               # ❌ MISSING
    
    # ✅ CORRECT: Financial terms properly implemented
    FINANCIAL_10Q_TERMS = {
        'revenue', 'assets', 'liabilities', 'equity', 'cash_flow',
        'quarterly', 'interim', 'unaudited', 'condensed',
        'yoy', 'quarter_over_quarter', 'guidance', 'outlook',
        'md_a', 'risk_factors', 'forward_looking', 'material'
    }
    
    # ✅ CORRECT: TF-IDF settings working
    MAX_FEATURES = 5000
    FINANCIAL_BOOST = 2.0
    
    # ✅ CORRECT: Tool routing implemented
    TABLE_KEYWORDS = ['revenue', 'income', 'balance', 'cash_flow', 'financial_statement']
    RISK_KEYWORDS = ['risk', 'uncertainty', 'factor', 'may_adversely'] 
    MDA_KEYWORDS = ['management', 'discussion', 'analysis', 'outlook', 'results']
    
    # ✅ CORRECT: UI settings
    DEFAULT_TOP_K = 5
    GOOGLE_API_KEY = ""
```

### **REQUIRED CONFIGURATION FIXES**
```python
# FIXED: Correct weights per specification
class Config:
    # Hybrid retrieval (CORRECTED)
    DENSE_WEIGHT = 0.7                  # ✅ 70% dense retrieval
    TFIDF_WEIGHT = 0.3                  # ✅ 30% sparse retrieval  
    GRAPH_ENHANCEMENT_WEIGHT = 0.15     # ✅ 15% graph boost
    
    # ADDED: Missing configuration options
    ENABLE_GRAPH_ENHANCEMENT = True     # ✅ Graph toggle
    CHUNK_ID_PREFIX = "chunk_"          # ✅ Chunk ID format
    REAL_TIME_METRICS = True            # ✅ Performance monitoring
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

---

## 🚀 **TODO: CRITICAL FIXES FOR 100% SPECIFICATION COMPLIANCE**

### **🔴 HIGH PRIORITY (Required for Spec Compliance)**

#### **1. METADATA SCHEMA FIX (CRITICAL)**
- **File**: `src/processing/chunker.py`
- **Issue**: Missing 5th metadata field (`chunk_id`)
- **Current**: 4/5 fields implemented
- **Required**: Add unique `chunk_id` field to metadata
```python
# FIX NEEDED in chunk_document():
metadata["chunk_id"] = f"chunk_{i}"  # Add this line
```

#### **2. HYBRID RETRIEVAL WEIGHTS FIX (CRITICAL)**
- **File**: `src/config.py`
- **Issue**: Weights are 80/20, specification requires 70/30
- **Current**: `DENSE_WEIGHT = 0.8, TFIDF_WEIGHT = 0.2`
- **Required**: `DENSE_WEIGHT = 0.7, TFIDF_WEIGHT = 0.3`
```python
# FIX NEEDED in Config class:
DENSE_WEIGHT = 0.7      # Change from 0.8
TFIDF_WEIGHT = 0.3      # Change from 0.2
GRAPH_ENHANCEMENT_WEIGHT = 0.15  # Add this
```

#### **3. GRAPH ENHANCEMENT INTEGRATION (CRITICAL)**
- **Files**: New file `src/retrieval/graph_retriever.py` + modify `src/ui/gradio_app.py`
- **Issue**: Graph database exists but not integrated into retrieval pipeline
- **Current**: Neo4j used only for storage
- **Required**: Implement GraphEnhancedRetriever class
```python
# NEW FILE NEEDED: src/retrieval/graph_retriever.py
class GraphEnhancedRetriever(BaseRetriever):
    def __init__(self, base_retriever, neo4j_graph, enhancement_weight=0.15):
        # Implement graph-enhanced retrieval
```

#### **4. COMPLETE GRADIO UI TABS (CRITICAL)**
- **File**: `src/ui/gradio_app.py`
- **Issue**: Missing 4 out of 6 required tabs
- **Current**: Only Q&A and Evaluation tabs
- **Required**: Add Summary, Table Queries, Financial Analysis, System Info, Configuration tabs

### **🟡 MEDIUM PRIORITY (Enhancement & Polish)**

#### **5. REAL-TIME METRICS DISPLAY**
- **File**: `src/ui/gradio_app.py` (System Info tab)
- **Required**: Show live RAGAS scores, retrieval performance, system metrics

#### **6. USER-CONFIGURABLE WEIGHTS**
- **File**: `src/ui/gradio_app.py` (Configuration tab)
- **Required**: Allow users to adjust Dense/TF-IDF/Graph weights dynamically

#### **7. ENHANCED ERROR HANDLING**
- **Files**: All retrieval and tool modules
- **Required**: Graceful fallbacks when components fail

### **🟢 LOW PRIORITY (Future Enhancements)**

#### **8. PERFORMANCE OPTIMIZATION**
- **Files**: All modules
- **Target**: Caching, batch processing, memory optimization

#### **9. ADVANCED GRAPH RELATIONSHIPS**
- **File**: `src/graph/neo4j_graph.py`
- **Target**: Semantic similarity connections, advanced navigation

---

## 📊 **COMPLIANCE SCORECARD & TARGETS**

| **Component** | **Current Status** | **Target** | **Priority** | **Files to Modify** |
|---------------|-------------------|------------|--------------|-------------------|
| **Metadata Fields** | 🔴 4/5 (80%) | 5/5 (100%) | 🔴 HIGH | `src/processing/chunker.py` |
| **Retrieval Weights** | 🔴 Wrong (80/20) | Correct (70/30) | 🔴 HIGH | `src/config.py` |
| **Graph Integration** | 🔴 Storage Only | Full Integration | 🔴 HIGH | `src/retrieval/graph_retriever.py` |
| **UI Tabs** | 🔴 2/6 (33%) | 6/6 (100%) | 🔴 HIGH | `src/ui/gradio_app.py` |
| **Financial TF-IDF** | 🟢 100% | 100% | ✅ DONE | - |
| **Reranker Toggle** | 🟢 100% | 100% | ✅ DONE | - |
| **RAGAS Evaluation** | 🟢 100% | 100% | ✅ DONE | - |
| **LlamaIndex Tables** | 🟢 100% | 100% | ✅ DONE | - |
| **Configuration** | 🟡 85% | 100% | 🟡 MEDIUM | `src/config.py` |
| **Testing** | 🟢 95% | 100% | 🟢 LOW | `tests/` |

### **TARGET: 100% SPECIFICATION COMPLIANCE**
**Current Overall Score**: 🔴 **75%**  
**Target Score**: 🟢 **100%**  
**Critical Path**: Fix 4 HIGH PRIORITY items above

---

## ⏰ **IMPLEMENTATION TIMELINE**

### **Phase 4A: Critical Fixes (Estimated: 2-3 days)**
1. **Day 1**: Metadata field fix + retrieval weights fix
2. **Day 2**: Graph enhancement integration
3. **Day 3**: Complete UI tabs implementation

### **Phase 4B: Enhancement & Polish (Estimated: 1-2 days)**
4. **Day 4**: Real-time metrics + configurable weights
5. **Day 5**: Final testing and documentation

### **SUCCESS CRITERIA**
- ✅ All RAGAS thresholds met (≥0.75, ≥0.70, ≥0.85, ≥0.90)
- ✅ 5/5 metadata fields implemented
- ✅ Correct retrieval weights (70/30/15)
- ✅ Graph enhancement working in retrieval pipeline
- ✅ 6/6 UI tabs fully functional
- ✅ User-configurable system parameters
- ✅ Real-time performance monitoring