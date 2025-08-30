from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingfaceEmbeddings
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
        ragas_llm = LangchainLLMWrapper(gemini_wrapper)

        # Configure embeddings to use HuggingFace (free, no API key needed)
        logger.info("Setting up HuggingFace embeddings for RAGAS")
        hf_embeddings = HuggingfaceEmbeddings(
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

        # Extract the four key metrics
        result_dict = {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
        }

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
        # Fallback to basic evaluation if RAGAS fails
        logger.info("Falling back to rule-based evaluation (all 4 metrics)")
        return _fallback_evaluation(question, answer, context, ground_truth)

def _fallback_evaluation(question, answer, context, ground_truth):
    """Rule-based fallback evaluation when LLM-based RAGAS fails."""
    logger = logging.getLogger("evaluation.ragas")

    try:
        # Basic evaluation as fallback
        answer_lower = answer.lower()
        ground_truth_lower = ground_truth.lower()

        # Basic accuracy check
        accuracy = 1.0 if ground_truth_lower in answer_lower else 0.5

        # Context relevance check
        context_relevance = 0.0
        if context:
            context_text = " ".join([c.page_content for c in context]).lower()
            answer_words = set(answer_lower.split())
            context_words = set(context_text.split())
            common_words = answer_words.intersection(context_words)
            context_relevance = len(common_words) / len(answer_words) if answer_words else 0.0

        # Ground truth match
        exact_match = 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

        result = {
            "faithfulness": accuracy,
            "answer_relevancy": context_relevance,
            "context_recall": exact_match,
            "context_precision": context_relevance,
            "overall_score": (accuracy + context_relevance + exact_match + context_relevance) / 4.0,
            "evaluation_type": "fallback_basic"
        }

        logger.info(f"Fallback evaluation completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Fallback evaluation failed: {e}")
        return {
            "error": "All evaluation methods failed",
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "overall_score": 0.0
        }
