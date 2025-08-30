import gradio as gr
import os
import shutil
import logging
import time
from ..logging_setup import initialize_logging
from ..tools.router import route_query
from ..tools.general_tool import GeneralTool
from ..tools.table_tool import TableTool
from ..tools.mda_tool import MDATool
from ..tools.risk_tool import RiskTool
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever, create_graph_enhanced_retriever
from ..llm.langchain_llm import LangchainLLM
from ..processing.pdf_to_html import convert_pdf_to_html
from ..processing.pdf_parser import load_html
from ..processing.chunker import chunk_document
from ..graph.neo4j_graph import Neo4jGraph
from ..evaluation.ragas_evaluation import evaluate_ragas
from ..evaluation.deepeval_evaluation import evaluate_deepeval
from ..config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Initialize logging for UI component
initialize_logging(component_name="ui")
logger = logging.getLogger("ui")

# Global variables
elements = []
chunks = []
ensemble_retriever = None
neo4j_graph_instance = None
last_doc_title = None
global_google_api_key = ""
global_cohere_api_key = ""

def clear_global_state():
    """Clear global state and close any active resources."""
    global elements, chunks, ensemble_retriever, neo4j_graph_instance, last_doc_title, global_google_api_key, global_cohere_api_key
    elements = []
    chunks = []
    ensemble_retriever = None
    last_doc_title = None
    global_google_api_key = ""
    global_cohere_api_key = ""
    if neo4j_graph_instance:
        try:
            neo4j_graph_instance.close()
        except Exception:
            logger.warning("Failed to close Neo4j graph instance during cleanup")
        neo4j_graph_instance = None
    return "✅ Global state cleared successfully!"

def process_file_with_progress(file, progress=gr.Progress()):
    """Enhanced file processing with progress tracking."""
    global elements, chunks, ensemble_retriever, last_doc_title
    logger.info("process_file called")
    
    if file is not None:
        progress(0.1, desc="🔄 Starting file upload...")
        time.sleep(0.5)
        
        # Determine source path (supports gradio file object or direct filepath)
        src_path = file if isinstance(file, str) else getattr(file, "name", None)
        if not src_path:
            logger.error("Invalid file input: missing path/name")
            return "❌ Invalid file input."

        progress(0.2, desc="📁 Saving uploaded file...")
        # Save/copy the uploaded file to project uploads dir
        os.makedirs(os.path.join("data", "uploads"), exist_ok=True)
        upload_path = os.path.join("data", "uploads", os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(upload_path):
            shutil.copyfile(src_path, upload_path)
        time.sleep(0.5)

        logger.info(f"Copy complete to {upload_path}")

        progress(0.4, desc="🔄 Converting PDF to HTML...")
        # Convert PDF to HTML
        html_content = convert_pdf_to_html(upload_path)
        html_path = upload_path + ".html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        time.sleep(1)

        logger.info(f"PDF converted to HTML at {html_path}")

        progress(0.6, desc="📄 Parsing document elements...")
        # Parse the document
        elements = load_html(html_path)
        time.sleep(0.5)

        logger.info(f"Parsed {len(elements)} elements")

        progress(0.8, desc="✂️ Creating document chunks...")
        # Chunk the document
        chunks = chunk_document(elements)
        time.sleep(0.5)

        logger.info(f"Created {len(chunks)} chunks")

        # Stash title for graph scoping
        last_doc_title = os.path.basename(html_path)

        progress(0.95, desc="🔗 Building ensemble retrievers...")
        # Create retrievers with graph enhancement per specification
        dense_retriever = get_dense_retriever(chunks)
        sparse_retriever = Financial10QRetriever(chunks)
        # Use graph-enhanced retriever (Dense 70% + TF-IDF 30% + Graph 15%)
        ensemble_retriever = create_graph_enhanced_retriever(dense_retriever, sparse_retriever)
        logger.info("Graph-enhanced ensemble retriever created per specification")
        time.sleep(0.5)

        progress(1.0, desc="✅ File processed successfully!")
        return f"✅ **File processed successfully!**\n\n📊 **Statistics:**\n- Elements parsed: {len(elements)}\n- Chunks created: {len(chunks)}\n- Retriever: Graph-enhanced ensemble ready"
    
    return "⚠️ Please upload a file first."

def add_to_graph_with_progress(neo4j_uri, neo4j_user, neo4j_password, progress=gr.Progress()):
    """Enhanced graph connection with progress tracking."""
    global neo4j_graph_instance, ensemble_retriever, elements, chunks, last_doc_title
    logger.info("add_to_graph called")
    
    if not elements:
        return "⚠️ Please process a file first."
    
    try:
        progress(0.2, desc="🔗 Connecting to Neo4j...")
        time.sleep(0.5)
        
        # Ensure chunks exist (recompute if needed)
        if not chunks:
            progress(0.4, desc="✂️ Recomputing chunks...")
            chunks = chunk_document(elements)
            time.sleep(0.5)

        progress(0.6, desc="📊 Creating graph structure...")
        # Create and populate graph from CHUNKS to guarantee chunk_id parity
        neo4j_graph_instance = Neo4jGraph(neo4j_uri, neo4j_user, neo4j_password)
        doc_title = last_doc_title or 'ProcessedDocument'
        neo4j_graph_instance.add_document_structure(chunks, doc_title=doc_title)
        time.sleep(1)
        
        progress(0.9, desc="🔄 Updating retrievers...")
        # Recreate ensemble retriever with graph integration
        if ensemble_retriever:
            # Get base retrievers and recreate with graph
            dense_retriever = get_dense_retriever(chunks)
            sparse_retriever = Financial10QRetriever(chunks)
            ensemble_retriever = create_graph_enhanced_retriever(
                dense_retriever, sparse_retriever, neo4j_graph_instance
            )
            logger.info("Retriever updated with graph integration")
        time.sleep(0.5)
        
        progress(1.0, desc="✅ Graph integration complete!")
        return "✅ **Document structure added to graph and retriever enhanced!**\n\n🔗 Graph database is now connected and integrated with the retrieval system."
        
    except Exception as e:
        logger.error(f"Graph integration failed: {e}")
        return f"❌ **Failed to add to graph:** {e}"

def answer_question_with_progress(question, use_reranker, progress=gr.Progress()):
    """Enhanced question answering with progress tracking."""
    try:
        if not elements:
            return "⚠️ Please process a file first."

        progress(0.1, desc="🔧 Initializing models...")
        time.sleep(0.3)
        
        # Set keys in environment from global once
        if global_google_api_key:
            os.environ["GOOGLE_API_KEY"] = global_google_api_key

        logger.debug("Initializing ChatGoogleGenerativeAI model")
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)

        progress(0.3, desc="🔍 Setting up retrievers...")
        retriever = ensemble_retriever
        if use_reranker:
            logger.debug("Enabling Cohere reranker")
            if global_cohere_api_key:
                os.environ["COHERE_API_KEY"] = global_cohere_api_key
            reranker = CohereRerank(model="rerank-english-v3.0")
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=ensemble_retriever
            )
        time.sleep(0.3)

        progress(0.5, desc="🛠️ Initializing tools...")
        # Initialize tools
        logger.debug("Initializing tools")
        general_tool = GeneralTool(retriever, langchain_llm)
        table_tool = TableTool(retriever, llama_llm, elements)
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)

        tools = {
            "general_tool": general_tool,
            "table_tool": table_tool,
            "mda_tool": mda_tool,
            "risk_tool": risk_tool,
        }
        time.sleep(0.3)

        progress(0.7, desc="🎯 Routing question...")
        tool_name = route_query(question)
        tool = tools[tool_name]
        logger.info(f"ROUTING: Question '{question}' routed to tool: {tool_name}")
        time.sleep(0.2)
        
        progress(0.9, desc="💭 Generating answer...")
        # Execute tool and get answer
        answer = tool.execute(question)
        
        progress(1.0, desc="✅ Answer ready!")
        
        return f"**🎯 Routed to:** {tool_name.replace('_', ' ').title()}\n\n**📝 Answer:**\n\n{answer}"
        
    except Exception as e:
        logger.exception("answer_question failed")
        return f"❌ **Error:** {e}"

def generate_summary_with_progress(progress=gr.Progress()):
    """Generate comprehensive 10-Q summarization with progress tracking."""
    logger.info("generate_summary called")
    if not elements:
        return "⚠️ Please process a file first."

    try:
        progress(0.1, desc="🔧 Initializing analysis tools...")
        if global_google_api_key:
            os.environ["GOOGLE_API_KEY"] = global_google_api_key

        # Use correct model version
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)

        # Use specialized tools instead of raw text processing
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)
        table_tool = TableTool(ensemble_retriever, llama_llm, elements)
        general_tool = GeneralTool(ensemble_retriever, langchain_llm)

        logger.info("Starting comprehensive 10-Q summarization with specialized tools")
        time.sleep(0.5)

        # Generate targeted summaries for each section
        sections_summary = {}

        progress(0.25, desc="💰 Analyzing financial performance...")
        # Financial Performance Summary (using TableTool)
        try:
            financial_summary = table_tool.execute(
                "Summarize the key financial performance metrics, revenue trends, and profitability indicators from the financial statements"
            )
            sections_summary["financial_performance"] = financial_summary
            logger.info("Financial performance summary completed")
        except Exception as e:
            logger.error(f"Financial summary failed: {e}")
            sections_summary["financial_performance"] = "Financial summary not available due to processing error."
        time.sleep(0.5)

        progress(0.50, desc="📈 Processing MD&A section...")
        # Management Discussion & Analysis Summary (using MDATool)
        try:
            mda_summary = mda_tool.execute(
                "Provide a comprehensive summary of management's discussion and analysis, including business outlook, operational highlights, and forward-looking statements"
            )
            sections_summary["mda_analysis"] = mda_summary
            logger.info("MD&A summary completed")
        except Exception as e:
            logger.error(f"MD&A summary failed: {e}")
            sections_summary["mda_analysis"] = "MD&A summary not available due to processing error."
        time.sleep(0.5)

        progress(0.75, desc="⚠️ Extracting risk factors...")
        # Risk Factors Summary (using RiskTool)
        try:
            risk_summary = risk_tool.execute(
                "Summarize the primary risk factors, uncertainties, and potential challenges facing the company"
            )
            sections_summary["risk_factors"] = risk_summary
            logger.info("Risk factors summary completed")
        except Exception as e:
            logger.error(f"Risk summary failed: {e}")
            sections_summary["risk_factors"] = "Risk factors summary not available due to processing error."
        time.sleep(0.5)

        progress(0.90, desc="🏢 Analyzing business operations...")
        # Business Operations Summary (using GeneralTool)
        try:
            business_summary = general_tool.execute(
                "Summarize key business developments, operational changes, market conditions, and strategic initiatives mentioned in the quarterly report"
            )
            sections_summary["business_operations"] = business_summary
            logger.info("Business operations summary completed")
        except Exception as e:
            logger.error(f"Business summary failed: {e}")
            sections_summary["business_operations"] = "Business operations summary not available due to processing error."
        time.sleep(0.5)

        progress(1.0, desc="✅ Summary complete!")

        # Create structured comprehensive summary
        comprehensive_summary = f"""# 📊 10-Q Quarterly Report - Comprehensive Summary

## 💰 Financial Performance Highlights
{sections_summary["financial_performance"]}

## 📈 Management Discussion & Analysis
{sections_summary["mda_analysis"]}

## ⚠️ Risk Factors & Uncertainties
{sections_summary["risk_factors"]}

## 🏢 Business Operations & Developments
{sections_summary["business_operations"]}

## 🎯 Summary Assessment
This quarterly report analysis leverages hybrid retrieval with financial domain expertise to provide comprehensive insights across all major sections of the 10-Q filing. Each section has been analyzed using specialized tools optimized for financial document understanding.

*✨ Generated using metadata-driven hybrid RAG with specialized financial tools*
"""

        logger.info("Comprehensive 10-Q summary generated successfully")
        return comprehensive_summary

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"❌ **Summary generation failed:** {e}\n\n🔧 **Please check:**\n1. Google API key is valid\n2. Document is properly processed\n3. Internet connection is stable"

def query_tables_with_progress(question, progress=gr.Progress()):
    """Enhanced table queries with progress tracking."""
    logger.info("query_tables called")
    if not elements:
        return "⚠️ Please process a file first."
    
    try:
        progress(0.2, desc="🔧 Initializing table analysis...")
        if global_google_api_key:
            os.environ["GOOGLE_API_KEY"] = global_google_api_key
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)
        time.sleep(0.5)
        
        progress(0.6, desc="📊 Analyzing financial tables...")
        # Force routing to table tool
        table_tool = TableTool(ensemble_retriever, llama_llm, elements)
        time.sleep(0.8)
        
        progress(0.9, desc="💭 Generating insights...")
        answer = table_tool.execute(question)
        time.sleep(0.3)
        
        progress(1.0, desc="✅ Table analysis complete!")
        return f"**📊 Table Analysis Results:**\n\n{answer}"
        
    except Exception as e:
        logger.error(f"Table query failed: {e}")
        return f"❌ **Table query failed:** {e}"

def financial_analysis_with_progress(progress=gr.Progress()):
    """Enhanced financial analysis with progress tracking."""
    logger.info("financial_analysis called")
    if not elements:
        return "⚠️ Please process a file first."

    try:
        progress(0.1, desc="🔧 Initializing analysis tools...")
        if global_google_api_key:
            os.environ["GOOGLE_API_KEY"] = global_google_api_key
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

        # Use specialized tools for analysis with improved fallback
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)

        logger.info("Starting specialized financial analysis with improved tools")
        time.sleep(0.5)

        progress(0.4, desc="📈 Analyzing MD&A section...")
        # Generate comprehensive analysis with specific queries
        try:
            mda_analysis = mda_tool.execute("What are the key financial performance trends and management outlook? Include revenue growth, profitability metrics, operating margins, and forward-looking statements from management.")
            logger.info("MD&A analysis completed successfully")
        except Exception as e:
            logger.error(f"MD&A analysis failed: {e}")
            mda_analysis = f"MD&A analysis unavailable: {str(e)}"
        time.sleep(1)

        progress(0.8, desc="⚠️ Evaluating risk factors...")
        try:
            risk_analysis = risk_tool.execute("What are the primary risk factors and uncertainties facing the company? Include contractual obligations, pending acquisitions, regulatory risks, and market challenges.")
            logger.info("Risk analysis completed successfully")
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            risk_analysis = f"Risk analysis unavailable: {str(e)}"
        time.sleep(1)

        progress(1.0, desc="✅ Financial analysis complete!")

        # Create comprehensive analysis
        analysis = f"""# 🏦 Financial Health Assessment

## 📈 Management Discussion & Analysis
{mda_analysis}

## ⚠️ Risk Factor Analysis
{risk_analysis}

## 🎯 Overall Assessment
This analysis combines management's discussion of financial performance with identified risk factors to provide a comprehensive view of the company's financial health and forward-looking challenges. The analysis uses specialized retrieval tools optimized for financial document understanding.

*✨ Generated using enhanced financial analysis tools with intelligent fallback mechanisms*
"""

        logger.info("Financial analysis completed successfully")
        return analysis

    except Exception as e:
        logger.error(f"Financial analysis failed: {e}")
        return f"❌ **Financial analysis failed:** {e}\n\n🔧 **Please check:**\n1. Google API key is valid\n2. Document is properly processed\n3. Internet connection is stable"

def run_evaluation_with_progress(question, ground_truth, use_reranker, progress=gr.Progress()):
    """Enhanced evaluation with progress tracking."""
    progress(0.2, desc="🔧 Setting up evaluation...")
    # Ensure keys available in env
    if global_google_api_key:
        os.environ["GOOGLE_API_KEY"] = global_google_api_key
    if global_cohere_api_key:
        os.environ["COHERE_API_KEY"] = global_cohere_api_key
    time.sleep(0.5)
    
    progress(0.5, desc="💭 Generating answer...")
    answer, context = answer_question_and_context(question, use_reranker)
    logger.info("run_evaluation called")
    time.sleep(1)
    
    progress(0.8, desc="📊 Running evaluation metrics...")
    # Use DeepEval only; if it fails, return just the answer and an empty metrics dict
    result = evaluate_deepeval(question, answer, context, ground_truth, model_name="gemini-1.5-pro-002", provider="google", api_key=global_google_api_key)
    time.sleep(0.5)
    
    progress(1.0, desc="✅ Evaluation complete!")
    
    if result and not result.get("error"):
        return f"**📝 System Answer:**\n\n{answer}", result
    logger.error(f"DeepEval evaluation failed: {result.get('error') if isinstance(result, dict) else result}")
    return f"**📝 System Answer:**\n\n{answer}", {}

def update_weights_enhanced(dense_weight, tfidf_weight, graph_weight):
    """Enhanced weight updating with validation."""
    logger.info("update_weights called")
    
    try:
        # Validate weights
        total = dense_weight + tfidf_weight
        if abs(total - 1.0) > 0.01:
            return f"⚠️ **Validation Error:** Dense + TF-IDF weights must sum to 1.0 (current: {total:.2f})"
        
        if graph_weight < 0 or graph_weight > 0.5:
            return "⚠️ **Validation Error:** Graph weight must be between 0 and 0.5"
        
        # Update configuration (Note: this updates the runtime instance only)
        Config.DENSE_WEIGHT = dense_weight
        Config.TFIDF_WEIGHT = tfidf_weight  
        Config.GRAPH_ENHANCEMENT_WEIGHT = graph_weight
        
        logger.info(f"Weights updated: Dense={dense_weight}, TF-IDF={tfidf_weight}, Graph={graph_weight}")
        
        return f"""✅ **Configuration Updated Successfully!**

📊 **New Weights:**
- Dense Retrieval: {dense_weight:.2f} ({dense_weight*100:.0f}%)
- TF-IDF Retrieval: {tfidf_weight:.2f} ({tfidf_weight*100:.0f}%)
- Graph Enhancement: {graph_weight:.2f} ({graph_weight*100:.0f}%)

⚠️ **Note:** Restart required for changes to take full effect in production."""
        
    except Exception as e:
        logger.error(f"Weight update failed: {e}")
        return f"❌ **Weight update failed:** {e}"

def set_api_keys_enhanced(google_api_key: str, cohere_api_key: str):
    """Enhanced API key setting with better feedback."""
    global global_google_api_key, global_cohere_api_key
    global_google_api_key = google_api_key or ""
    global_cohere_api_key = cohere_api_key or ""
    
    status_parts = []
    if global_google_api_key:
        os.environ["GOOGLE_API_KEY"] = global_google_api_key
        status_parts.append("✅ Google API key")
    if global_cohere_api_key:
        os.environ["COHERE_API_KEY"] = global_cohere_api_key
        status_parts.append("✅ Cohere API key")
    
    if status_parts:
        return f"**🔑 API Keys Configured:**\n\n{chr(10).join(status_parts)}\n\n🌟 Ready for enhanced processing!"
    else:
        return "⚠️ **No API keys provided.** Please enter at least one API key."

def get_enhanced_system_info():
    """Enhanced system information display."""
    logger.info("get_system_info called")
    
    try:
        import psutil
        import platform
        from datetime import datetime
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Configuration info
        config_info = f"""# 🖥️ System Information & Status

## ⚙️ Current Configuration
| Setting | Value | Percentage |
|---------|-------|------------|
| **Dense Weight** | {Config.DENSE_WEIGHT} | {Config.DENSE_WEIGHT*100:.0f}% |
| **TF-IDF Weight** | {Config.TFIDF_WEIGHT} | {Config.TFIDF_WEIGHT*100:.0f}% |
| **Graph Enhancement** | {Config.GRAPH_ENHANCEMENT_WEIGHT} | {Config.GRAPH_ENHANCEMENT_WEIGHT*100:.0f}% |
| **Financial Boost** | {Config.FINANCIAL_BOOST} | - |
| **Max Features** | {Config.MAX_FEATURES:,} | - |
| **Default Top-K** | {Config.DEFAULT_TOP_K} | - |

## 📊 System Performance
| Metric | Value | Status |
|--------|-------|--------|
| **CPU Usage** | {cpu_percent}% | {'🟢 Good' if cpu_percent < 70 else '🟡 High' if cpu_percent < 90 else '🔴 Critical'} |
| **Memory Usage** | {memory.percent}% | {'🟢 Good' if memory.percent < 70 else '🟡 High' if memory.percent < 90 else '🔴 Critical'} |
| **Available Memory** | {memory.available / (1024**3):.1f} GB | - |
| **Platform** | {platform.system()} {platform.release()} | - |
| **Timestamp** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | - |

## 🔧 Components Status
| Component | Status | Count/Info |
|-----------|--------|------------|
| **Elements Loaded** | {'✅ Active' if elements else '❌ Not Loaded'} | {len(elements) if elements else 0} |
| **Document Chunks** | {'✅ Ready' if chunks else '❌ Not Created'} | {len(chunks) if chunks else 0} |
| **Ensemble Retriever** | {'✅ Active' if ensemble_retriever else '❌ Not Created'} | {'Graph-Enhanced' if ensemble_retriever else '-'} |
| **Graph Integration** | {'✅ Connected' if neo4j_graph_instance else '❌ Not Connected'} | {'Neo4j Active' if neo4j_graph_instance else '-'} |
| **Graph Enhancement** | {'✅ Enabled' if Config.ENABLE_GRAPH_ENHANCEMENT else '❌ Disabled'} | - |

## 🏗️ Metadata Schema (5 Fields)
1. ✅ **element_type**: SEC semantic element class
2. ✅ **chunk_id**: Unique chunk identifier  
3. ✅ **page_number**: PDF page for citations
4. ✅ **section_path**: SEC section identifier
5. ✅ **content_type**: Content classification

---
*🚀 System optimized for financial document analysis*
"""
        
        return config_info
        
    except Exception as e:
        logger.error(f"System info failed: {e}")
        return f"❌ **System info failed:** {e}"

# Original functions (keeping compatibility)
def answer_question_for_app(question, use_reranker):
    try:
        answer, _ = answer_question_and_context(question, use_reranker)
        return answer
    except Exception as e:
        logger.exception("answer_question_for_app failed")
        return f"❌ **Error answering question:** {e}"

def answer_question_and_context(question, use_reranker):
    logger.info("answer_question called")
    try:
        if not elements:
            return "⚠️ Please process a file first.", []

        logger.debug(f"use_reranker={use_reranker}")
        # Set keys in environment from global once
        if global_google_api_key:
            os.environ["GOOGLE_API_KEY"] = global_google_api_key

        logger.debug("Initializing ChatGoogleGenerativeAI model")
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)

        retriever = ensemble_retriever
        if use_reranker:
            logger.debug("Enabling Cohere reranker")
            if global_cohere_api_key:
                os.environ["COHERE_API_KEY"] = global_cohere_api_key
            reranker = CohereRerank(model="rerank-english-v3.0")
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=ensemble_retriever
            )

        # Initialize tools
        logger.debug("Initializing tools")
        general_tool = GeneralTool(retriever, langchain_llm)
        table_tool = TableTool(retriever, llama_llm, elements)
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)

        tools = {
            "general_tool": general_tool,
            "table_tool": table_tool,
            "mda_tool": mda_tool,
            "risk_tool": risk_tool,
        }

        tool_name = route_query(question)
        tool = tools[tool_name]
        logger.info(f"ROUTING: Question '{question}' routed to tool: {tool_name}")
        
        # Execute tool and get answer
        answer = tool.execute(question)
        
        # Get context documents for additional logging
        context = retriever.get_relevant_documents(question)
        logger.info(f"RETRIEVAL_SUMMARY: Retrieved {len(context)} context documents for UI logging")
        
        # ENHANCED CONTEXT LOGGING: Log all context chunks with metadata
        for i, doc in enumerate(context):
            chunk_preview = doc.page_content[:150].replace('\n', ' ')  # First 150 chars
            metadata_summary = {k: v for k, v in doc.metadata.items()}
            logger.info(f"UI_CONTEXT_CHUNK {i+1}/{len(context)}: {metadata_summary}")
            logger.info(f"UI_CONTENT_PREVIEW: {chunk_preview}...")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"UI_FULL_CONTENT_{i+1}: {doc.page_content}")
        return answer, context
    except Exception:
        logger.exception("answer_question_and_context failed")
        return "❌ **An error occurred while answering the question.** Check logs for details.", []

# Dark CSS theme without blue accents
custom_css = """
/* Dark Theme Base */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent-primary: #f0f6fc;
    --accent-secondary: #30363d;
    --border-color: #30363d;
    --shadow-color: rgba(0, 0, 0, 0.4);
}

/* Force dark background for entire app */
.gradio-container, body, html {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    padding: 20px !important;
}

/* Simple container - no complex layouts */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Basic styling for clean dark appearance */
.gr-button {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.gr-button:hover {
    background: var(--accent-secondary) !important;
    color: var(--text-primary) !important;
}

.gr-button[variant="primary"] {
    background: var(--text-primary) !important;
    color: var(--bg-primary) !important;
    border: 1px solid var(--text-primary) !important;
}

.gr-button[variant="primary"]:hover {
    background: var(--text-secondary) !important;
    color: var(--bg-primary) !important;
}

.gr-button[variant="stop"] {
    background: #da3633 !important;
    color: white !important;
    border: 1px solid #da3633 !important;
}

.gr-button[variant="secondary"] {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
}

/* Input styling */
.gr-textbox, .gr-textarea {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Markdown styling */
.gr-markdown {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    color: var(--text-primary) !important;
    max-height: 60vh !important;
    overflow-y: auto !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-primary) !important;
}

/* Ensure vertical spacing */
.gr-column {
    gap: 16px !important;
}

/* Simple scrollbars */
*::-webkit-scrollbar {
    width: 8px !important;
}

*::-webkit-scrollbar-track {
    background: var(--bg-tertiary) !important;
}

*::-webkit-scrollbar-thumb {
    background: var(--text-secondary) !important;
    border-radius: 4px !important;
}

*::-webkit-scrollbar-thumb:hover {
    background: var(--text-primary) !important;
}

/* Tab styling */
.gr-tab {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    border: none !important;
}

.gr-tab.selected {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--text-primary) !important;
}

/* Form elements */
.gr-form {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

/* Labels */
label {
    color: var(--text-primary) !important;
}

/* Checkboxes and radios */
.gr-checkbox input, .gr-radio input {
    accent-color: var(--text-primary) !important;
}

/* File upload */
.gr-file {
    background: var(--bg-secondary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 8px !important;
}

.gr-file:hover {
    border-color: var(--text-secondary) !important;
}

/* JSON display */
.gr-json {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}
"""

# Create the enhanced Gradio interface with simple layout
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ).set(
        # Dark theme colors
        background_fill_primary="#0d1117",
        background_fill_secondary="#161b22", 
        background_fill_primary_dark="#0d1117",
        background_fill_secondary_dark="#161b22",
        color_accent_soft="#f0f6fc",
        color_accent="#f0f6fc",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        border_color_primary="#30363d",
        border_color_primary_dark="#30363d"
    ),
    css=custom_css,
    title="🏦 Financial RAG System",
    analytics_enabled=False
) as iface:
    
    # API Configuration Section - ALWAYS VISIBLE AT TOP
    gr.HTML("<h2>🔑 API Configuration</h2>")
    with gr.Row():
        main_api_key_box = gr.Textbox(
            label="Google API Key", 
            type="password",
            placeholder="Enter your Google API key...",
            scale=2
        )
        main_cohere_api_key_box = gr.Textbox(
            label="Cohere API Key", 
            type="password", 
            placeholder="Enter your Cohere API key...",
            scale=2
        )
        set_keys_button = gr.Button("Set Keys", variant="primary", scale=1)
    
    keys_status = gr.Markdown()
    
    # Navigation Tabs - Simple and Clean
    with gr.Tabs():
        # Q&A Tab
        with gr.TabItem("💬 Q&A"):
            gr.HTML("<h3>📄 Document Processing</h3>")
            file_upload = gr.File(
                file_types=[".pdf"], 
                file_count="single", 
                type="filepath",
                label="Upload PDF Document"
            )
            process_button = gr.Button("🚀 Process File", variant="primary")
            process_status = gr.Markdown()
            
            gr.HTML("<h3>🔗 Graph Database (Optional)</h3>")
            with gr.Row():
                neo4j_uri_box = gr.Textbox(label="Neo4j URI", placeholder="bolt://localhost:7687", scale=2)
                neo4j_user_box = gr.Textbox(label="Username", placeholder="neo4j", scale=1)
                neo4j_password_box = gr.Textbox(label="Password", type="password", scale=1)
            add_to_graph_button = gr.Button("Connect to Graph", variant="secondary")
            graph_status = gr.Markdown()
            
            gr.HTML("<h3>💬 Ask Questions</h3>")
            use_reranker_checkbox = gr.Checkbox(label="🎯 Use Reranker", value=False)
            question_box = gr.Textbox(
                label="Your Question", 
                placeholder="Ask anything about your 10-Q document...",
                lines=3
            )
            answer_button = gr.Button("💭 Get Answer", variant="primary")
            answer_box = gr.Markdown()

        # Summary Tab
        with gr.TabItem("📋 Summary"):
            gr.HTML("<h3>📊 Comprehensive Document Summarization</h3>")
            gr.Markdown("Generate an AI-powered summary of your 10-Q document using specialized financial analysis tools.")
            generate_summary_button = gr.Button("✨ Generate Summary", variant="primary")
            summary_output = gr.Markdown()
            
        # Table Queries Tab
        with gr.TabItem("📊 Table Queries"):
            gr.HTML("<h3>📊 Financial Table Analysis</h3>")
            gr.Markdown("Query financial statements, balance sheets, and other tabular data with AI precision.")
            table_question_box = gr.Textbox(
                label="Table Analysis Question", 
                placeholder="e.g., What was the revenue growth compared to last quarter?",
                lines=2
            )
            query_tables_button = gr.Button("🔍 Analyze Tables", variant="primary")
            table_output = gr.Markdown()
            
        # Financial Analysis Tab
        with gr.TabItem("📈 Financial Analysis"):
            gr.HTML("<h3>🏦 Financial Health Assessment</h3>")
            gr.Markdown("Comprehensive financial health assessment combining MD&A and risk factor analysis.")
            run_analysis_button = gr.Button("🚀 Run Financial Analysis", variant="primary")
            analysis_output = gr.Markdown()
            
        # System Info Tab
        with gr.TabItem("🖥️ System Info"):
            gr.HTML("<h3>🖥️ System Information</h3>")
            gr.Markdown("Real-time system status, configuration, and component health monitoring.")
            with gr.Row():
                refresh_info_button = gr.Button("🔄 Refresh Info", variant="primary")
                cleanup_button = gr.Button("🧹 Clear State", variant="stop")
            system_info_output = gr.Markdown()
            cleanup_status = gr.Markdown()

        # Configuration Tab
        with gr.TabItem("⚙️ Configuration"):
            gr.HTML("<h3>⚙️ System Configuration</h3>")
            gr.Markdown("Fine-tune retrieval weights and system parameters for optimal performance.")
            
            gr.HTML("<h4>⚖️ Retrieval Weight Configuration</h4>")
            dense_weight_slider = gr.Slider(
                0.1, 0.9, value=Config.DENSE_WEIGHT, step=0.05, 
                label="🔍 Dense Retrieval Weight", info="Semantic similarity weight"
            )
            tfidf_weight_slider = gr.Slider(
                0.1, 0.9, value=Config.TFIDF_WEIGHT, step=0.05,
                label="📝 TF-IDF Weight", info="Keyword matching weight" 
            )
            graph_weight_slider = gr.Slider(
                0.0, 0.5, value=Config.GRAPH_ENHANCEMENT_WEIGHT, step=0.05,
                label="🕸️ Graph Enhancement Weight", info="Knowledge graph boost"
            )
            
            update_config_button = gr.Button("💾 Update Configuration", variant="primary")
            config_status = gr.Markdown()

        # Evaluation Tab
        with gr.TabItem("🧪 Evaluation"):
            gr.HTML("<h3>🧪 Model Performance Evaluation</h3>")
            gr.Markdown("Evaluate system performance using DeepEval metrics with ground truth comparisons.")
            
            eval_use_reranker = gr.Checkbox(label="🎯 Use Reranker", value=False)
            eval_question_box = gr.Textbox(
                label="Evaluation Question", 
                placeholder="Enter a question to evaluate...",
                lines=2
            )
            ground_truth_box = gr.Textbox(
                label="Ground Truth Answer", 
                placeholder="Enter the expected correct answer...",
                lines=3
            )
            eval_button = gr.Button("🧪 Run Evaluation", variant="primary")
            
            system_answer_box = gr.Markdown()
            eval_results_box = gr.JSON()

    # Wire up all the functionality
    set_keys_button.click(set_api_keys_enhanced, inputs=[main_api_key_box, main_cohere_api_key_box], outputs=[keys_status])
    
    # Q&A Tab
    process_button.click(process_file_with_progress, inputs=[file_upload], outputs=[process_status])
    add_to_graph_button.click(add_to_graph_with_progress, inputs=[neo4j_uri_box, neo4j_user_box, neo4j_password_box], outputs=[graph_status])
    answer_button.click(answer_question_with_progress, inputs=[question_box, use_reranker_checkbox], outputs=[answer_box])
    
    # Other tabs
    generate_summary_button.click(generate_summary_with_progress, outputs=[summary_output])
    query_tables_button.click(query_tables_with_progress, inputs=[table_question_box], outputs=[table_output])
    run_analysis_button.click(financial_analysis_with_progress, outputs=[analysis_output])
    refresh_info_button.click(get_enhanced_system_info, outputs=[system_info_output])
    cleanup_button.click(clear_global_state, outputs=[cleanup_status])
    update_config_button.click(update_weights_enhanced, inputs=[dense_weight_slider, tfidf_weight_slider, graph_weight_slider], outputs=[config_status])
    eval_button.click(run_evaluation_with_progress, inputs=[eval_question_box, ground_truth_box, eval_use_reranker], outputs=[system_answer_box, eval_results_box])

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path=None,
        share=False
    )