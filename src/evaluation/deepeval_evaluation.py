from typing import List, Dict, Any
import logging


def evaluate_deepeval(question: str, answer: str, context_docs: List[Any], ground_truth: str, *, model_name: str | None = None, provider: str | None = None, api_key: str | None = None) -> Dict[str, Any]:
    """Evaluate a single QA turn using DeepEval metrics.

    Returns a plain dict suitable for the UI JSON component, mirroring our previous shape.
    If DeepEval is unavailable, returns a clear error with a hint.
    """
    logger = logging.getLogger("evaluation.deepeval")
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            AnswerRelevancyMetric,
        )
    except Exception as e:
        logger.error(f"DeepEval import failed: {e}")
        return {
            "error": "DeepEval not installed or failed to import. Please `pip install deepeval`.",
        }

    # Build test case
    retrieval_context = [d.page_content for d in context_docs] if context_docs else []
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=retrieval_context,
    )

    # Configure evaluation model (optional, DeepEval will use defaults if not provided)
    eval_model = None
    try:
        if provider == "google":
            from deepeval.models import GoogleVertexAI
            eval_model = GoogleVertexAI(model_name=model_name or "gemini-1.5-pro-002", api_key=api_key)
        elif provider == "openai":
            from deepeval.models import OpenAIChat
            eval_model = OpenAIChat(model="gpt-4o-mini", api_key=api_key)
    except Exception as e:
        logger.warning(f"Evaluation model init failed, falling back to default: {e}")
        eval_model = None

    # Instantiate metrics
    metrics = [
        ContextualPrecisionMetric(model=eval_model),
        ContextualRecallMetric(model=eval_model),
        FaithfulnessMetric(model=eval_model),
        AnswerRelevancyMetric(model=eval_model),
    ]

    # Run evaluation (DeepEval raises on assert; we capture scores from metric.result)
    results: Dict[str, float] = {}
    try:
        assert_test(test_case, metrics)
    except Exception as e:
        # Still proceed to read metric scores even if assertions fail
        logger.info(f"DeepEval assertion raised: {e}")

    # Collect metric scores (best-effort)
    for m in metrics:
        name = getattr(m, "name", m.__class__.__name__).lower()
        score = getattr(m, "result", None)
        if score is None and hasattr(m, "score"):
            score = getattr(m, "score")
        if isinstance(score, (int, float)):
            results[name] = float(score)

    # Normalize keys to our UI schema
    out = {
        "context_precision": float(
            results.get("contextual precision", results.get("contextualprecisionmetric", results.get("contextual_precision", 0.0)))
        ),
        "context_recall": float(
            results.get("contextual recall", results.get("contextualrecallmetric", results.get("contextual_recall", 0.0)))
        ),
        "faithfulness": float(results.get("faithfulness", results.get("faithfulnessmetric", 0.0))),
        "answer_relevancy": float(results.get("answer relevancy", results.get("answerrelevancymetric", results.get("answer_relevancy", 0.0)))),
    }
    # Overall score
    out["overall_score"] = (out["context_precision"] + out["context_recall"] + out["faithfulness"] + out["answer_relevancy"]) / 4.0
    out["evaluation_type"] = "deepeval"
    return out


