import gradio as gr
import os
import shutil
import logging
from ..logging_setup import initialize_logging
from ..config import Config

# Initialize logging for UI component
initialize_logging(component_name="ui")
logger = logging.getLogger("ui")

# Global variables
elements = []
ensemble_retriever = None
neo4j_graph_instance = None

def clear_global_state():
    """Clear global state and close any active resources."""
    global elements, ensemble_retriever, neo4j_graph_instance
    elements = []
    ensemble_retriever = None
    if neo4j_graph_instance:
        try:
            neo4j_graph_instance.close()
        except Exception:
            logger.warning("Failed to close Neo4j graph instance during cleanup")
        neo4j_graph_instance = None

def process_file(file, api_key):
    global elements, ensemble_retriever
    logger.info("process_file called")

    # Import heavy libraries only when needed
    from ..processing.pdf_to_html import convert_pdf_to_html
    from ..processing.pdf_parser import load_html
    from ..processing.chunker import chunk_document
    from ..retrieval.dense_retriever import get_dense_retriever
    from ..retrieval.tfidf_retriever import Financial10QRetriever
    from ..retrieval.ensemble_setup import create_graph_enhanced_retriever

    if file is not None:
        # Determine source path (supports gradio file object or direct filepath)
        src_path = file if isinstance(file, str) else getattr(file, "name", None)
        if not src_path:
            logger.error("Invalid file input: missing path/name")
            return "Invalid file input."

        # Save/copy the uploaded file to project uploads dir
        os.makedirs(os.path.join("data", "uploads"), exist_ok=True)
        upload_path = os.path.join("data", "uploads", os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(upload_path):
            shutil.copyfile(src_path, upload_path)

        logger.info(f"Copy complete to {upload_path}")

        # Convert PDF to HTML
        html_content = convert_pdf_to_html(upload_path)
        html_path = upload_path + ".html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"PDF converted to HTML at {html_path}")

        # Parse the document
        elements = load_html(html_path)

        logger.info(f"Parsed {len(elements)} elements")

        # Chunk the document
        chunks = chunk_document(elements)

        logger.info(f"Created {len(chunks)} chunks")

        # Create retrievers with graph enhancement per specification
        dense_retriever = get_dense_retriever(chunks)
        sparse_retriever = Financial10QRetriever(chunks)
        # Use graph-enhanced retriever (Dense 70% + TF-IDF 30% + Graph 15%)
        ensemble_retriever = create_graph_enhanced_retriever(dense_retriever, sparse_retriever)
        logger.info("Graph-enhanced ensemble retriever created per specification")

        return "File processed successfully!"
    return "Please upload a file first."

def add_to_graph(neo4j_uri, neo4j_user, neo4j_password):
    global neo4j_graph_instance, ensemble_retriever, elements
    logger.info("add_to_graph called")
    if not elements:
        return "Please process a file first."

    try:
        # Import heavy libraries only when needed
        from ..graph.neo4j_graph import Neo4jGraph
        from ..processing.chunker import chunk_document
        from ..retrieval.dense_retriever import get_dense_retriever
        from ..retrieval.tfidf_retriever import Financial10QRetriever
        from ..retrieval.ensemble_setup import create_graph_enhanced_retriever

        # Create and populate graph
        neo4j_graph_instance = Neo4jGraph(neo4j_uri, neo4j_user, neo4j_password)
        neo4j_graph_instance.add_document_structure(elements)

        # Recreate ensemble retriever with graph integration
        if ensemble_retriever:
            # Get base retrievers and recreate with graph
            dense_retriever = get_dense_retriever(chunk_document(elements))
            sparse_retriever = Financial10QRetriever(chunk_document(elements))
            ensemble_retriever = create_graph_enhanced_retriever(
                dense_retriever, sparse_retriever, neo4j_graph_instance
            )
            logger.info("Retriever updated with graph integration")

        return "Document structure added to graph and retriever enhanced!"
    except Exception as e:
        logger.error(f"Graph integration failed: {e}")
        return f"Failed to add to graph: {e}"

def answer_question_for_app(question, api_key, cohere_api_key, use_reranker):
    try:
        answer, _ = answer_question_and_context(question, api_key, cohere_api_key, use_reranker)
        return answer
    except Exception as e:
        logger.exception("answer_question_for_app failed")
        return f"Error answering question: {e}"

def answer_question_and_context(question, api_key, cohere_api_key, use_reranker):
    logger.info("answer_question called")
    try:
        if not elements:
            return "Please process a file first.", []

        logger.debug(f"use_reranker={use_reranker}")
        logger.debug(f"GOOGLE_API_KEY provided={'yes' if bool(api_key) else 'no'}")
        # Set key in environment for the model client only at call sites
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Import heavy libraries only when needed
        from ..tools.router import route_query
        from ..tools.general_tool import GeneralTool
        from ..tools.table_tool import TableTool
        from ..tools.mda_tool import MDATool
        from ..tools.risk_tool import RiskTool
        from ..llm.langchain_llm import LangchainLLM
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank

        logger.debug("Initializing ChatGoogleGenerativeAI model")
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)

        retriever = ensemble_retriever
        if use_reranker:
            logger.debug("Enabling Cohere reranker")
            if cohere_api_key:
                os.environ["COHERE_API_KEY"] = cohere_api_key
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
        return "An error occurred while answering the question. Check logs for details.", []

def run_evaluation(question, ground_truth, api_key, cohere_api_key, use_reranker):
    # Import heavy library only when needed
    from ..evaluation.ragas_evaluation import evaluate_ragas

    answer, context = answer_question_and_context(question, api_key, cohere_api_key, use_reranker)
    logger.info("run_evaluation called")
    result = evaluate_ragas(question, answer, context, ground_truth)
    return result

def generate_summary(api_key):
    """Generate comprehensive 10-Q summarization using specialized tools."""
    logger.info("generate_summary called")
    if not elements:
        return "Please process a file first."

    try:
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Import heavy libraries only when needed
        from ..tools.mda_tool import MDATool
        from ..tools.risk_tool import RiskTool
        from ..tools.table_tool import TableTool
        from ..tools.general_tool import GeneralTool
        from ..llm.langchain_llm import LangchainLLM
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Use correct model version
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llama_llm = LangchainLLM(langchain_llm)

        # Use specialized tools instead of raw text processing
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)
        table_tool = TableTool(ensemble_retriever, llama_llm, elements)
        general_tool = GeneralTool(ensemble_retriever, langchain_llm)

        logger.info("Starting comprehensive 10-Q summarization with specialized tools")

        # Generate targeted summaries for each section
        sections_summary = {}

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

        # Create structured comprehensive summary
        comprehensive_summary = f"""# 10-Q Quarterly Report - Comprehensive Summary

## Financial Performance Highlights
{sections_summary["financial_performance"]}

## Management Discussion & Analysis
{sections_summary["mda_analysis"]}

## Risk Factors & Uncertainties
{sections_summary["risk_factors"]}

## Business Operations & Developments
{sections_summary["business_operations"]}

## Summary Assessment
This quarterly report analysis leverages hybrid retrieval with financial domain expertise to provide comprehensive insights across all major sections of the 10-Q filing. Each section has been analyzed using specialized tools optimized for financial document understanding.

*Generated using metadata-driven hybrid RAG with specialized financial tools*
"""

        logger.info("Comprehensive 10-Q summary generated successfully")
        return comprehensive_summary

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Summary generation failed: {e}\n\nPlease check:\n1. Google API key is valid\n2. Document is properly processed\n3. Internet connection is stable"

def query_tables(question, api_key):
    """Specialized interface for table queries."""
    logger.info("query_tables called")
    if not elements:
        return "Please process a file first."

    try:
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Import heavy libraries only when needed
        from ..tools.table_tool import TableTool
        from ..llm.langchain_llm import LangchainLLM
        from langchain_google_genai import ChatGoogleGenerativeAI

        langchain_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        llama_llm = LangchainLLM(langchain_llm)

        # Force routing to table tool
        table_tool = TableTool(ensemble_retriever, llama_llm, elements)
        answer = table_tool.execute(question)

        return answer

    except Exception as e:
        logger.error(f"Table query failed: {e}")
        return f"Table query failed: {e}"

def financial_analysis(api_key):
    """Health assessment and risk factor analysis."""
    logger.info("financial_analysis called")
    if not elements:
        return "Please process a file first."

    try:
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Import heavy libraries only when needed
        from ..tools.mda_tool import MDATool
        from ..tools.risk_tool import RiskTool
        from langchain_google_genai import ChatGoogleGenerativeAI

        langchain_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        # Use specialized tools for analysis
        mda_tool = MDATool(langchain_llm, elements)
        risk_tool = RiskTool(langchain_llm, elements)

        # Generate comprehensive analysis
        mda_analysis = mda_tool.execute("What are the key financial performance trends and management outlook?")
        risk_analysis = risk_tool.execute("What are the primary risk factors and uncertainties?")

        analysis = f"""# Financial Health Assessment

## Management Discussion & Analysis
{mda_analysis}

## Risk Factor Analysis
{risk_analysis}

## Overall Assessment
Based on the MD&A and risk factors, this analysis provides insights into the company's financial health and forward-looking challenges.
"""

        return analysis

    except Exception as e:
        logger.error(f"Financial analysis failed: {e}")
        return f"Financial analysis failed: {e}"

def get_system_info():
    """Display system information and metrics."""
    logger.info("get_system_info called")

    try:
        # Import heavy libraries only when needed
        import psutil
        import platform
        from datetime import datetime
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Configuration info
        config_info = f"""# System Information

## Current Configuration
- **Dense Weight**: {Config.DENSE_WEIGHT} (70%)
- **TF-IDF Weight**: {Config.TFIDF_WEIGHT} (30%) 
- **Graph Enhancement Weight**: {Config.GRAPH_ENHANCEMENT_WEIGHT} (15%)
- **Financial Boost Factor**: {Config.FINANCIAL_BOOST}
- **Max TF-IDF Features**: {Config.MAX_FEATURES}
- **Default Top-K**: {Config.DEFAULT_TOP_K}

## System Performance
- **CPU Usage**: {cpu_percent}%
- **Memory Usage**: {memory.percent}%
- **Available Memory**: {memory.available / (1024**3):.1f} GB
- **Platform**: {platform.system()} {platform.release()}
- **Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Components Status
- **Elements Loaded**: {len(elements) if elements else 0}
- **Ensemble Retriever**: {"✅ Active" if ensemble_retriever else "❌ Not Created"}
- **Graph Integration**: {"✅ Active" if neo4j_graph_instance else "❌ Not Connected"}
- **Graph Enhancement**: {"✅ Enabled" if Config.ENABLE_GRAPH_ENHANCEMENT else "❌ Disabled"}

## Metadata Schema (5 Fields)
1. ✅ **element_type**: SEC semantic element class
2. ✅ **chunk_id**: Unique chunk identifier  
3. ✅ **page_number**: PDF page for citations
4. ✅ **section_path**: SEC section identifier
5. ✅ **content_type**: Content classification
"""
        
        return config_info
        
    except Exception as e:
        logger.error(f"System info failed: {e}")
        return f"System info failed: {e}"

def update_weights(dense_weight, tfidf_weight, graph_weight):
    """Update retrieval weights dynamically."""
    logger.info("update_weights called")
    
    try:
        # Validate weights
        total = dense_weight + tfidf_weight
        if abs(total - 1.0) > 0.01:
            return f"Dense + TF-IDF weights must sum to 1.0 (current: {total})"
        
        if graph_weight < 0 or graph_weight > 0.5:
            return "Graph weight must be between 0 and 0.5"
        
        # Update configuration (Note: this updates the runtime instance only)
        Config.DENSE_WEIGHT = dense_weight
        Config.TFIDF_WEIGHT = tfidf_weight  
        Config.GRAPH_ENHANCEMENT_WEIGHT = graph_weight
        
        logger.info(f"Weights updated: Dense={dense_weight}, TF-IDF={tfidf_weight}, Graph={graph_weight}")
        
        return f"✅ Configuration Updated:\n- Dense: {dense_weight}\n- TF-IDF: {tfidf_weight}\n- Graph Enhancement: {graph_weight}\n\nNote: Restart required for changes to take full effect."
        
    except Exception as e:
        logger.error(f"Weight update failed: {e}")
        return f"Weight update failed: {e}"

with gr.Blocks() as iface:
    gr.Markdown("# Financial RAG System")
    gr.Markdown("Ask questions about your 10-Q documents.")

    with gr.Tabs():
        with gr.TabItem("Q&A"):
            with gr.Row():
                api_key_box = gr.Textbox(label="Google API Key", type="password")
                cohere_api_key_box = gr.Textbox(label="Cohere API Key", type="password")
                use_reranker_checkbox = gr.Checkbox(label="Use Reranker")
            
            with gr.Row():
                file_upload = gr.File(file_types=[".pdf", "pdf", "application/pdf"], file_count="single", type="filepath")
                process_button = gr.Button("Process File")

            process_status = gr.Label()
            process_button.click(process_file, inputs=[file_upload, api_key_box], outputs=[process_status])

            with gr.Row():
                neo4j_uri_box = gr.Textbox(label="Neo4j URI")
                neo4j_user_box = gr.Textbox(label="Neo4j User")
                neo4j_password_box = gr.Textbox(label="Neo4j Password", type="password")
                add_to_graph_button = gr.Button("Add to Graph")
            
            graph_status = gr.Label()
            add_to_graph_button.click(add_to_graph, inputs=[neo4j_uri_box, neo4j_user_box, neo4j_password_box], outputs=[graph_status])

            question_box = gr.Textbox(label="Question")
            answer_button = gr.Button("Answer")
            answer_box = gr.Markdown()

            answer_button.click(answer_question_for_app, inputs=[question_box, api_key_box, cohere_api_key_box, use_reranker_checkbox], outputs=[answer_box])

        with gr.TabItem("Summary"):
            gr.Markdown("## Full Document Summarization")
            gr.Markdown("Generate a comprehensive summary of the 10-Q document using LlamaIndex.")
            
            summary_api_key_box = gr.Textbox(label="Google API Key", type="password")
            generate_summary_button = gr.Button("Generate Summary")
            summary_output = gr.Markdown()
            
            generate_summary_button.click(generate_summary, inputs=[summary_api_key_box], outputs=[summary_output])
        
        with gr.TabItem("Table Queries"):
            gr.Markdown("## Specialized Financial Table Analysis")
            gr.Markdown("Query financial statements, balance sheets, and other tabular data.")
            
            table_api_key_box = gr.Textbox(label="Google API Key", type="password")
            table_question_box = gr.Textbox(label="Table Question", placeholder="e.g., What was the revenue for Q2?")
            query_tables_button = gr.Button("Query Tables")
            table_output = gr.Markdown()
            
            query_tables_button.click(query_tables, inputs=[table_question_box, table_api_key_box], outputs=[table_output])
        
        with gr.TabItem("Financial Analysis"):
            gr.Markdown("## Health Assessment & Risk Analysis")
            gr.Markdown("Comprehensive financial health assessment combining MD&A and risk factor analysis.")
            
            analysis_api_key_box = gr.Textbox(label="Google API Key", type="password")
            run_analysis_button = gr.Button("Run Financial Analysis")
            analysis_output = gr.Markdown()
            
            run_analysis_button.click(financial_analysis, inputs=[analysis_api_key_box], outputs=[analysis_output])
        
        with gr.TabItem("System Info"):
            gr.Markdown("## System Information & Performance Metrics")
            gr.Markdown("Real-time system status, configuration, and component health.")
            
            refresh_info_button = gr.Button("Refresh System Info")
            system_info_output = gr.Markdown()
            
            refresh_info_button.click(get_system_info, outputs=[system_info_output])
            
            # Add maintenance controls
            gr.Markdown("### Maintenance")
            cleanup_button = gr.Button("Clear Global State")
            cleanup_status = gr.Label()
            cleanup_button.click(lambda: (clear_global_state(), "State cleared")[1], outputs=[cleanup_status])
        
        with gr.TabItem("Configuration"):
            gr.Markdown("## System Configuration")
            gr.Markdown("Configure retrieval weights and system parameters.")
            
            with gr.Row():
                dense_weight_slider = gr.Slider(0.1, 0.9, value=Config.DENSE_WEIGHT, step=0.05, label="Dense Weight")
                tfidf_weight_slider = gr.Slider(0.1, 0.9, value=Config.TFIDF_WEIGHT, step=0.05, label="TF-IDF Weight")
                graph_weight_slider = gr.Slider(0.0, 0.5, value=Config.GRAPH_ENHANCEMENT_WEIGHT, step=0.05, label="Graph Enhancement Weight")
            
            update_config_button = gr.Button("Update Configuration")
            config_status = gr.Markdown()
            
            update_config_button.click(
                update_weights,
                inputs=[dense_weight_slider, tfidf_weight_slider, graph_weight_slider],
                outputs=[config_status]
            )

        with gr.TabItem("Evaluation"):
            eval_question_box = gr.Textbox(label="Question")
            ground_truth_box = gr.Textbox(label="Ground Truth")
            eval_button = gr.Button("Evaluate")
            eval_results_box = gr.JSON()

            eval_button.click(run_evaluation, inputs=[eval_question_box, ground_truth_box, api_key_box, cohere_api_key_box, use_reranker_checkbox], outputs=[eval_results_box])

if __name__ == "__main__":
    iface.launch()