# 📘 Master Development Checklist (MDC) — Ground Truth

Single source of truth for building a metadata‑driven Hybrid RAG system that fully satisfies the original mission while remaining simple, modular, and scalable. No mocks, no stubs, no hardcoded fallbacks.

Key principles:
- Keep it simple. Prefer deterministic preprocessing over LLM complexity.
- No mocks/stubs/hardcoded fallbacks in any layer.
- Fully modular components with clean interfaces.
- Scalability-first: local-first, multi-document ready, domain-agnostic design.
- Reranker implemented but OFF by default, user-toggle at runtime.

***

## 0. Project Targets

- Domain: Documents with text, tables, charts (initially SEC 10-Q; structure flexible enough for insurance incident reports).
- Hybrid retrieval: Dense ∪ Sparse with metadata filters.
- Chunk budget: ≈5% of document size or up to 10 chunks.
- Metadata: 3–5 fields per chunk chosen from a canonical list.
- Reranker: Implemented; configurable ON/OFF at runtime; never applied by default.
- Graph DB: Integrated for document structure and navigation.
- Table‑QA: Deterministic numeric lookups with anchor return (table_id,row[,col]).
- Evaluation: Context Precision≥0.75, Context Recall≥0.70, Faithfulness≥0.85, Table‑QA Accuracy≥0.90.

***

## 1. Repository Structure

```
hybrid_rag/
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py           # text, tables, figures; anchors
│   │   ├── chunker.py              # section/table aware; budgeted distillation
│   │   └── metadata.py             # extraction + validation (3–5 fields)
│   ├── indexing/
│   │   ├── vector_store.py         # Chroma/Qdrant driver (adapter pattern)
│   │   ├── sparse_index.py         # BM25/keyword index
│   │   ├── embeddings.py           # model adapters; batch/embed functions
│   │   └── hybrid_retriever.py     # dense ∪ sparse + metadata filters + rerank toggle
│   ├── graph/
│   │   ├── schema.py               # Neo4j schema (Document/Section/Table/Figure/Page)
│   │   └── ingest.py               # node/rel ingestion; id mapping
│   ├── tools/
│   │   ├── base_tool.py            # abstract interface
│   │   ├── needle_tool.py          # precise section/paragraph finder
│   │   ├── summary_tool.py         # section-aware summarization
│   │   ├── table_qa_tool.py        # numeric/tabular QA + anchors
│   │   ├── policy_check_tool.py    # policy vs incident matcher (domain-agnostic hooks)
│   │   └── graph_tool.py           # structure-aware navigation
│   ├── evaluation/
│   │   ├── ragas_eval.py           # evaluation runner
│   │   └── metrics.py              # context precision/recall, faithfulness, table accuracy
│   ├── ui/
│   │   └── gradio_app.py           # tabs, config, toggles (reranker, filters)
│   ├── config/
│   │   ├── settings.py             # env/config loader; no secrets in repo
│   │   └── schema.py               # Pydantic config models
│   └── orchestrator/
│       └── pipeline.py             # end-to-end ingest/query API
├── data/
│   ├── uploads/
│   ├── processed/
│   └── evaluation/
├── tests/                          # real functional tests; no mocks/stubs
├── requirements.txt
├── README.md
└── MDC.md                          # this file
```

***

## 2. Canonical Metadata Schema

Allowed fields (choose 3–5 per chunk):
- PageNumber (int)
- ChunkSummary (short sentence)
- Keywords (list[str])
- CriticalEntities (list[str])  // e.g., entities, KPIs, dates, parties
- IncidentType (str)            // domain-adaptable; optional for generic docs
- IncidentDate (date/str)       // when applicable
- SectionType (enum)            // e.g., md_a, financial_statements, controls, legal
- AmountRange (str)             // e.g., 0–1M, 1M–10M
- TableId / FigureId (str)

Rules:
- Minimum 3, maximum 5 metadata fields per chunk.
- Always include PageNumber.
- Include ChunkSummary OR Keywords (at least one, not required both).
- Include SectionType if section anchor is detected.
- Store metadata alongside vectors for filtering.

***

## 3. Anchors and IDs

- SectionAnchor: human-readable anchor (e.g., "Item 2 – MD&A" or "Section 5.2 – Claims").
- TableId/FigureId: stable deterministic IDs per document: table_{page}_{index}, figure_{page}_{index}.
- Table Cell Anchor: (table_id, row_index[, col_index]) zero-based indices.
- Paragraph Anchor: (page, block_index) for precise Needle results.

***

## 4. End-to-End Workflow (Per Document)

1) Upload PDF (≤50 pages).  
2) Parse:
   - Extract text blocks with page numbers.
   - Detect sections/headers (regex + heuristics).
   - Extract tables as structured frames; assign TableId; keep original cell positions when available.
   - Extract figures with FigureId (store caption/alt-text if available).
3) Chunk:
   - Section-aware segmentation; keep tables atomic.
   - Distill to budget: select the most important chunks up to ≈5% of doc or max 10 chunks.
   - Importance heuristics: section type priority, presence of KPIs, frequency of critical entities/keywords, summary salience.
4) Metadata:
   - Extract 3–5 fields per chunk from canonical schema.
   - Validate schema, types, and limits.
5) Index:
   - Dense vectors + metadata persisted in vector store.
   - Sparse index (BM25/keyword) built from normalized text.
6) Graph Ingestion:
   - Create nodes: Document, Section, Subsection, Table, Figure, Page.
   - Create relations (CONTAINS, HAS_SUBSECTION, REFERENCES, LOCATED_ON, CROSS_REFERENCES).
   - Map anchors/IDs to nodes for later navigation.
7) Evaluation (optional per doc or batch):
   - Generate or load synthetic Q/A set aligned with chunks and tables.
   - Run RAGAS/metrics and store results.
8) Expose in UI:
   - Tabs for Q&A, Summary, Table Queries, Policy Check, System Info, Config.
   - Reranker toggle OFF by default; user can enable per-query or session.

***

## 5. Retrieval & Reranking

Query processing:
- Normalize text; extract keywords, entities, and temporal hints deterministically.
- Build metadata filters based on query entities/sections if present.

Hybrid retrieval:
- Dense topK_d (e.g., 30) ∪ Sparse topK_s (e.g., 30) → union unique candidates.
- Apply metadata filters first to reduce token waste.

Reranker:
- Implement cross-encoder or lightweight scoring model.
- Candidate narrowing: from up to K≈40 to 6–8.
- OFF by default; controlled via runtime UI toggle and API parameter.
- Log incremental lift; if average lift  dict

Tools:
- NeedleTool:
  - Input: free text + optional anchors/filters.
  - Output: best paragraph/section + exact paragraph anchor and section anchor.
- SummaryTool:
  - Section-aware synthesis; configurable granularity.
  - Use short prompts + compression; always cite anchors used.
- TableQATool:
  - Deterministic first: resolve table, select cells/rows, compute aggregates if asked.
  - Output includes numeric answer + exact (table_id,row[,col]) anchor(s) + source excerpt.
  - Only call LLM if disambiguation or reasoning across multiple tables is needed.
- PolicyCheckTool:
  - Domain-agnostic: given policy clauses and incident facts, compute coverage match matrix.
  - Deterministic matching on entities/dates/amounts; LLM used for borderline clauses.
  - Output: pass/fail per clause + rationales + anchors to clauses and incident facts.
- GraphQueryTool:
  - Navigate document structures; return node IDs/paths and associated text/table anchors.

All tools must:
- Be stateless; accept dependencies via constructor.
- Return machine-readable payloads (dict) with anchors, ids, and confidence.

***

## 7. UI (Gradio)

Tabs:
- Q&A Chat:
  - Hybrid retrieval; citations with anchors.
  - Controls: reranker toggle, topK sliders, metadata filter chips.
- Summary:
  - Section-level or full-document; shows used anchors.
- Table Queries:
  - Numeric lookups; displays table rendering + highlighted rows/cols + returned anchors.
- Policy Check:
  - Upload policy text + incident facts; show clause matches with anchors.
- System Info:
  - Sample metadata, graph node counts, latest eval metrics, latency stats.
- Configuration:
  - API keys (e.g., Google/Gemini), model selection, reranker ON/OFF default, thresholds.

UX rules:
- No hidden fallbacks; any LLM usage must be explicit and visible.
- Always show source anchors with each answer.
- Provide export of Q/A + anchors as JSON for auditing.

***

## 8. Evaluation & Metrics

Targets:
- Context Precision ≥0.75
- Context Recall ≥0.70
- Faithfulness ≥0.85
- Table‑QA Accuracy ≥0.90

Strategy:
- Ground truth generated to mirror chunk boundaries (not context-sensitive).
- Include text, tabular, and graph-navigation questions.
- Track with RAGAS + custom metrics.
- Per-feature dashboards: hybrid vs dense-only, reranker OFF vs ON.

Policy:
- CI must fail if metrics regress beyond allowed deltas (configurable).
- Store per-run JSON of metrics and candidate lists.

***

## 9. Development Phases

Phase 1: Parsing + Chunking + Metadata
- Implement pdf_parser.py with tables/figures and anchors.
- Implement chunker.py with section/table awareness and budgeted distillation.
- Implement metadata.py with selection rules and validators.

Phase 2: Storage & Retrieval
- Implement embeddings.py (adapter), vector_store.py (Chroma or Qdrant via adapter), sparse_index.py (BM25).
- Implement hybrid_retriever.py with metadata filters and reranker interface (toggle).

Phase 3: Graph
- Implement graph schema and ingestion.
- Ensure anchors↔nodes mappings.

Phase 4: Tools
- Implement NeedleTool, SummaryTool, TableQATool (deterministic-first), PolicyCheckTool, GraphQueryTool.

Phase 5: UI
- Implement Gradio tabs; expose reranker toggle and config.
- Add system info and metrics display.

Phase 6: Evaluation
- Implement ragas_eval.py and metrics.py.
- Add synthetic dataset generation respecting chunk rules.

***

## 10. Interfaces and Contracts

Ingestion API:
- ingest(pdf_path: str) -> DocumentHandle {doc_id, pages, sections, tables, figures}

Query API:
- query(question: str, options: {filters, use_reranker: bool, topk_dense, topk_sparse}) ->
  {answer, used_chunks:[anchors], confidence, diagnostics}

Table QA Output Schema:
- {
  "answer": number|string,
  "units": string|null,
  "anchors": [{"table_id": str, "row": int, "col": int|null}],
  "source_excerpt": str,
  "confidence": float
}

Policy Check Output Schema:
- {
  "clause_results": [
    {"clause_id": str, "covered": bool, "reason": str, "anchors": {"policy": anchor, "incident": anchor}}
  ],
  "overall": {"covered": bool, "confidence": float}
}

All outputs must be JSON-serializable and logged.

***

## 11. Reranker Implementation Rules

- Provide Reranker interface:
  - score(query, candidates) -> ranked candidates with scores
- Backend: cross-encoder or lightweight bi-encoder with late interaction.
- Default OFF in both API and UI.
- User can toggle per request/session.
- Capture and store metrics deltas when ON; persist cumulative stats.
- If avg lift in Context Precision ≤3pts over last N runs, keep OFF by default.

***

## 12. Chunk Budgeting Rules

- Budget: min(total_chunks, max_allowed)
  - max_allowed = min(ceil(0.05 * estimated_doc_chunks), 10)
- If natural chunking exceeds max_allowed:
  - Distill via importance scoring:
    - SectionType priority (e.g., financial_statements, MD&A, controls)
    - Presence of CriticalEntities/KPIs/dates/amounts
    - Table presence for numeric-heavy questions
- Always preserve atomic tables selected into the budget.

***

## 13. Deterministic First Policy

- Before calling LLM:
  - Normalize numbers, units, dates.
  - Resolve entities and sections deterministically.
  - For table QA, compute results from DataFrame first.
- LLM used only for:
  - Disambiguation
  - Cross-chunk synthesis
  - Policy clause natural language reasoning
- Every LLM call records prompt, truncation budget, and source anchors.

***

## 14. Configuration & Security

- settings.py loads from environment (.env allowed locally).
- No secrets in repo; never hardcode keys.
- Provide model adapters for embeddings and LLMs.
- Offline-capable by default for retrieval; LLM optional.

***

## 15. Testing Policy

- Functional tests only with real sample documents (anonymized where necessary).
- No mocks/stubs. Use small real PDFs in data/uploads for CI.
- Golden-files tests for:
  - Anchors stability
  - Chunk budget enforcement
  - Table QA numeric correctness and anchor returns
  - Reranker toggle behavior
- Regression tests for metrics thresholds.

***

## 16. Coding Standards

- Style: black, ruff, mypy (basic).
- Functions small, pure where possible; side effects documented.
- Adapter pattern for swappable backends (Chroma/Qdrant, BM25 variants, LLMs).
- Clear docstrings and type hints; raise explicit exceptions.
- Logging: structured JSON logs with doc_id, query_id, timings, toggle states.

***

## 17. Performance & Token Economy

- Pre-filter with metadata and graph before LLM.
- Compress context deterministically (dedupe, normalize, section title kept).
- Limit context to top 6–8 chunks even without reranker.
- Cache embeddings; batch operations.

***

## 18. Graph Schema (Neo4j)

Nodes:
- Document, Section, Subsection, Table, Figure, Page

Relations:
- (Document)-[:CONTAINS]->(Section)
- (Section)-[:HAS_SUBSECTION]->(Subsection)
- (Section)-[:REFERENCES]->(Table|Figure)
- (Table)-[:LOCATED_ON]->(Page)
- (Section)-[:CROSS_REFERENCES]->(Section)

Node properties include anchors and ids; indexes on (doc_id, section_anchor), (table_id), (page).

***

## 19. UI Controls and Telemetry

- Toggles: Reranker ON/OFF, Dense/Sparse K, Metadata filter chips, SectionType filter.
- Display: Answer + anchors, table highlights, metrics snapshot, latency.
- Export: JSON of query, settings, results, anchors.
- Telemetry: anonymous local stats (no PII), persisted in data/processed.

***

## 20. Quick Start Checklist

- [ ] Setup repo, CI (lint/type/test).
- [ ] Implement pdf_parser with anchors for text/tables/figures.
- [ ] Implement chunker with budget enforcement.
- [ ] Implement metadata extraction with 3–5 fields rule.
- [ ] Embed + index into vector store; build BM25.
- [ ] Implement hybrid retrieval with metadata filters.
- [ ] Implement reranker interface (default OFF) and UI toggle.
- [ ] Build Neo4j schema and ingest mappings.
- [ ] Implement Needle, Summary, Table‑QA, Policy‑Check, Graph tools.
- [ ] Build Gradio tabs with controls and citations.
- [ ] Implement evaluation pipeline with thresholds and CI guardrails.
- [ ] Run end-to-end on at least 2 real documents and validate metrics.

***

## 21. Deviation Log (if needed)

Any deviation from this MDC must be recorded with:
- What changed
- Why (data/metrics evidence)
- Impact on metrics and complexity
- Rollback plan

***

This MDC is the authoritative blueprint. It fully satisfies the original mission’s functional scope while enforcing simplicity, modularity, scalability, and strict operational discipline with a user-controlled reranker.