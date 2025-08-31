from typing import List, Dict, Any
import logging


def evaluate_deepeval(question: str, answer: str, context_docs: List[Any], ground_truth: str, *, model_name: str | None = None, provider: str | None = None, api_key: str | None = None) -> Dict[str, Any]:
    """Evaluate a single QA turn using DeepEval metrics.

    Returns a plain dict suitable for the UI JSON component, mirroring our previous shape.
    If DeepEval is unavailable, returns a clear error with a hint.
    """
    logger = logging.getLogger("evaluation.deepeval")
    
    # 🔍 DEBUG: Trace DeepEval entry point
    logger.critical("🔍 DEEPEVAL FUNCTION CALLED")
    logger.critical(f"🔍 Provider: {provider}")
    logger.critical(f"🔍 Model name: {model_name}")
    logger.critical(f"🔍 API key provided: {bool(api_key)}")
    logger.critical(f"🔍 API key length: {len(api_key) if api_key else 0}")
    
    try:
        logger.critical("🔍 Attempting to import DeepEval modules...")
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            AnswerRelevancyMetric,
        )
        logger.critical("🔍 ✅ DeepEval imports successful")
    except Exception as e:
        logger.critical(f"🔍 ❌ DeepEval import failed: {e}")
        return {
            "error": "DeepEval not installed or failed to import. Please `pip install deepeval`.",
        }

    # Build test case with proper structure for DeepEval
    retrieval_context = [d.page_content for d in context_docs] if context_docs else []
    
    logger.critical(f"🔍 Building test case:")
    logger.critical(f"🔍 Input (question): {question[:100]}...")
    logger.critical(f"🔍 Actual output (answer): {answer[:100]}...")
    logger.critical(f"🔍 Expected output (ground truth): {ground_truth[:100]}...")
    logger.critical(f"🔍 Retrieval context chunks: {len(retrieval_context)}")
    
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=retrieval_context,
    )
    
    logger.critical("🔍 ✅ Test case constructed successfully")

    # Configure evaluation model - MUST have explicit model, no DeepEval defaults
    eval_model = None
    logger.critical("🔍 STARTING MODEL CONFIGURATION")
    try:
        if provider == "google":
            logger.critical("🔍 Setting up Google/Gemini model...")
            # Set up environment for Google Generative AI
            import os
            os.environ["GOOGLE_API_KEY"] = api_key
            logger.critical(f"🔍 Set GOOGLE_API_KEY environment variable")
            
            # Use the correct DeepEval Gemini model classes  
            try:
                logger.critical("🔍 Creating GeminiModel with correct parameters...")
                from deepeval.models import GeminiModel
                logger.critical("🔍 ✅ GeminiModel import successful, creating model...")
                
                # Use the correct parameter names: model_name and api_key
                eval_model = GeminiModel(
                    model_name=model_name or "gemini-1.5-pro-002", 
                    api_key=api_key
                )
                logger.critical(f"🔍 ✅ Successfully configured GeminiModel: {model_name or 'gemini-1.5-pro-002'}")
                
            except Exception as e:
                logger.critical(f"🔍 ❌ GeminiModel creation failed: {e}")
                return {
                    "error": f"DeepEval GeminiModel creation failed: {e}"
                }
                    
        elif provider == "openai":
            logger.critical("🔍 ❌ WARNING: Setting up OpenAI model (this should not happen if Google key is available)")
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            from deepeval.models import OpenAIChat
            eval_model = OpenAIChat(model="gpt-4o-mini", api_key=api_key)
            logger.critical("🔍 ✅ Successfully configured OpenAI model: gpt-4o-mini")
            
    except Exception as e:
        logger.critical(f"🔍 ❌ Model setup completely failed: {e}")
        return {
            "error": f"DeepEval model setup failed: {e}"
        }
    
    # CRITICAL: Never allow eval_model = None, as DeepEval will silently use OpenAI default
    if eval_model is None:
        logger.critical(f"🔍 ❌ CRITICAL: eval_model is None for provider '{provider}' - this would cause silent OpenAI fallback")
        return {
            "error": f"DeepEval {provider} model setup failed - cannot use OpenAI fallback. Use RAGAS instead."
        }
    
    logger.critical(f"🔍 ✅ Model configuration complete: {type(eval_model).__name__}")

    # Instantiate metrics with proper configuration
    logger.critical("🔍 Creating metrics with configured model and include_reason=True...")
    metrics = [
        ContextualPrecisionMetric(model=eval_model, include_reason=True),
        ContextualRecallMetric(model=eval_model, include_reason=True),
        FaithfulnessMetric(model=eval_model, include_reason=True),
        AnswerRelevancyMetric(model=eval_model, include_reason=True),
    ]
    logger.critical(f"🔍 ✅ Created {len(metrics)} metrics with explainability enabled")

    # Run evaluation using DeepEval's proper API
    logger.critical("🔍 Starting DeepEval evaluation...")
    
    # Evaluate each metric individually to capture scores and reasons properly
    results: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    for i, metric in enumerate(metrics):
        metric_name = metric_names[i]
        logger.critical(f"🔍 Evaluating {metric_name} using {type(metric).__name__}...")
        
        try:
            # Measure the metric directly
            metric.measure(test_case)
            
            # Access score and reason directly using DeepEval's API
            score = metric.score if hasattr(metric, 'score') else None
            reason = metric.reason if hasattr(metric, 'reason') else None
            
            logger.critical(f"🔍 {metric_name} - Score: {score}, Reason: {reason[:100] if reason else 'None'}...")
            
            if isinstance(score, (int, float)):
                results[metric_name] = float(score)
                reasons[metric_name] = reason if reason else f"No explanation provided for {metric_name}"
                logger.critical(f"🔍 ✅ Successfully captured {metric_name}: {float(score)}")
            else:
                logger.critical(f"🔍 ❌ Invalid score for {metric_name}: {score} (type: {type(score)})")
                results[metric_name] = 0.0
                reasons[metric_name] = f"Failed to evaluate {metric_name}: Invalid score returned"
                
        except Exception as e:
            logger.critical(f"🔍 ❌ Failed to evaluate {metric_name}: {e}")
            results[metric_name] = 0.0
            reasons[metric_name] = f"Evaluation failed: {str(e)}"

    # Build final result using properly extracted scores and reasons
    out = {
        "context_precision": results.get("context_precision", 0.0),
        "context_recall": results.get("context_recall", 0.0),
        "faithfulness": results.get("faithfulness", 0.0),
        "answer_relevancy": results.get("answer_relevancy", 0.0),
        "reasons": {
            "context_precision": reasons.get("context_precision", "No explanation available"),
            "context_recall": reasons.get("context_recall", "No explanation available"),
            "faithfulness": reasons.get("faithfulness", "No explanation available"),
            "answer_relevancy": reasons.get("answer_relevancy", "No explanation available"),
        }
    }
    
    # Calculate overall score
    out["overall_score"] = (out["context_precision"] + out["context_recall"] + out["faithfulness"] + out["answer_relevancy"]) / 4.0
    out["evaluation_type"] = "deepeval"
    
    logger.critical(f"🔍 ✅ Final evaluation result: {out}")
    return out


