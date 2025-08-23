import gradio as gr
import os
import shutil
import logging
from ..logging_setup import initialize_logging
from ..tools.router import route_query
from ..tools.general_tool import GeneralTool
from ..tools.table_tool import TableTool
from ..tools.mda_tool import MDATool
from ..tools.risk_tool import RiskTool
from ..retrieval.dense_retriever import get_dense_retriever
from ..retrieval.tfidf_retriever import Financial10QRetriever
from ..retrieval.ensemble_setup import create_ensemble_retriever
from ..llm.langchain_llm import LangchainLLM
from ..processing.pdf_to_html import convert_pdf_to_html
from ..processing.pdf_parser import load_html
from ..processing.chunker import chunk_document
from ..graph.neo4j_graph import Neo4jGraph
from ..evaluation.ragas_evaluation import evaluate_ragas
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Initialize logging for UI component
initialize_logging(component_name="ui")
logger = logging.getLogger("ui")

# Global variables
elements = []
ensemble_retriever = None

def process_file(file, api_key):
    global elements, ensemble_retriever
    logger.info("process_file called")
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

        # Create retrievers
        dense_retriever = get_dense_retriever(chunks)
        sparse_retriever = Financial10QRetriever(chunks)
        ensemble_retriever = create_ensemble_retriever(dense_retriever, sparse_retriever)
        logger.info("Ensemble retriever created")

        return "File processed successfully!"
    return "Please upload a file first."

def add_to_graph(neo4j_uri, neo4j_user, neo4j_password):
    logger.info("add_to_graph called")
    if not elements:
        return "Please process a file first."
    
    graph = Neo4jGraph(neo4j_uri, neo4j_user, neo4j_password)
    graph.add_document_structure(elements)
    graph.close()
    return "Document structure added to graph!"

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
        os.environ["GOOGLE_API_KEY"] = api_key or ""

        logger.debug("Initializing ChatGoogleGenerativeAI model")
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        llama_llm = LangchainLLM(langchain_llm)

        retriever = ensemble_retriever
        if use_reranker:
            logger.debug("Enabling Cohere reranker")
            os.environ["COHERE_API_KEY"] = cohere_api_key or ""
            reranker = CohereRerank()
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
        logger.debug(f"Routing to tool: {tool_name}")
        answer = tool.execute(question)
        context = retriever.get_relevant_documents(question)
        logger.debug(f"Retrieved {len(context)} context docs")
        return answer, context
    except Exception:
        logger.exception("answer_question_and_context failed")
        return "An error occurred while answering the question. Check logs for details.", []

def run_evaluation(question, ground_truth, api_key, cohere_api_key, use_reranker):
    answer, context = answer_question_and_context(question, api_key, cohere_api_key, use_reranker)
    logger.info("run_evaluation called")
    result = evaluate_ragas(question, answer, context, ground_truth)
    return result

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

        with gr.TabItem("Evaluation"):
            eval_question_box = gr.Textbox(label="Question")
            ground_truth_box = gr.Textbox(label="Ground Truth")
            eval_button = gr.Button("Evaluate")
            eval_results_box = gr.JSON()

            eval_button.click(run_evaluation, inputs=[eval_question_box, ground_truth_box, api_key_box, cohere_api_key_box, use_reranker_checkbox], outputs=[eval_results_box])

if __name__ == "__main__":
    iface.launch()