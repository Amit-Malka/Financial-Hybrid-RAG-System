from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
try:
    from ragas.llms import LangchainLLMWrapper as RagasLangchainLLM
except ImportError:  # version compatibility
    from ragas.llms import LangchainLLM as RagasLangchainLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from ragas.embeddings import LangchainEmbeddings
except ImportError:
    # Fallback for different RAGAS versions
    LangchainEmbeddings = None
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
import logging
import os

class GeminiLLMWrapper:
    """Custom wrapper for Gemini to work with RAGAS."""

    def __init__(self, gemini_llm):
        self.llm = gemini_llm

    def generate(self, prompt, **kwargs):
        """Generate response using Gemini."""
        try:
            # Filter out parameters that Gemini doesn't support
            filtered_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']}
            response = self.llm.invoke(prompt, **filtered_kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            # Fallback without filtering
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

def evaluate_ragas(question, answer, context, ground_truth, api_key=None):
    """Evaluates the RAG system using RAGAS with Gemini and all four metrics."""
    logger = logging.getLogger("evaluation.ragas")

    try:
        # Configure RAGAS to use Gemini instead of OpenAI
        logger.info("Configuring RAGAS to use Gemini for evaluation")

        # Get API key from environment or parameter
        gemini_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            logger.error("No Google API key found for RAGAS evaluation")
            return {"error": "Google API key required for evaluation"}

        # Create Gemini LLM for RAGAS evaluation
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=gemini_api_key,
            temperature=0.1  # Set low temperature for consistent evaluation
        )

        # Wrap Gemini for RAGAS compatibility
        gemini_wrapper = GeminiLLMWrapper(gemini_llm)
        ragas_llm = RagasLangchainLLM(gemini_wrapper)

        # Configure embeddings to use HuggingFace (free, no API key needed)
        logger.info("Setting up HuggingFace embeddings for RAGAS")
        if LangchainEmbeddings:
            # Use RAGAS's LangchainEmbeddings wrapper
            base_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            hf_embeddings = LangchainEmbeddings(base_embeddings)
        else:
            # Direct LangChain embeddings (might work with newer RAGAS versions)
            hf_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # Configure ALL FOUR RAGAS metrics to use Gemini LLM
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = hf_embeddings

        # Make context metrics LLM-based too (as requested)
        context_recall.llm = ragas_llm
        context_precision.llm = ragas_llm

        # Prepare data for RAGAS evaluation
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[c.page_content for c in context]],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        logger.info("Starting RAGAS evaluation with Gemini (ALL 4 metrics LLM-based)")
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,      # LLM: How well the answer matches the ground truth
                answer_relevancy,  # LLM: How relevant the answer is to the question
                context_recall,    # LLM: How much of the ground truth is covered by context
                context_precision, # LLM: How much of the context is relevant
            ],
        )

        # Extract the four key metrics (support both dict and dataframe-like outputs)
        try:
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                row = df.iloc[0] if len(df) > 0 else {}
                result_dict = {
                    "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
                    "context_recall": float(row.get("context_recall", 0.0) or 0.0),
                    "context_precision": float(row.get("context_precision", 0.0) or 0.0),
                }
            elif isinstance(result, dict):
                result_dict = {
                    "faithfulness": float(result.get("faithfulness", 0.0)),
                    "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                    "context_recall": float(result.get("context_recall", 0.0)),
                    "context_precision": float(result.get("context_precision", 0.0)),
                }
            else:
                # Last-resort attribute access
                result_dict = {
                    "faithfulness": float(getattr(result, "faithfulness", 0.0)),
                    "answer_relevancy": float(getattr(result, "answer_relevancy", 0.0)),
                    "context_recall": float(getattr(result, "context_recall", 0.0)),
                    "context_precision": float(getattr(result, "context_precision", 0.0)),
                }
        except Exception as e:
            logger.error(f"Failed to parse RAGAS result: {e}")
            return {"error": f"Failed to parse RAGAS result: {e}"}

        # Calculate overall score
        result_dict["overall_score"] = (
            result_dict["faithfulness"] +
            result_dict["answer_relevancy"] +
            result_dict["context_recall"] +
            result_dict["context_precision"]
        ) / 4.0

        logger.info(f"RAGAS evaluation completed: {result_dict}")
        return result_dict

    except Exception as e:
        logger.error(f"RAGAS LLM evaluation failed: {e}")
        return {"error": f"RAGAS evaluation failed: {e}"}

