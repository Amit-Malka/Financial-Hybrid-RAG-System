from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import logging

def evaluate_ragas(question, answer, context, ground_truth):
    """Evaluates the RAG system using RAGAS."""
    logger = logging.getLogger("evaluation.ragas")
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[c.page_content for c in context]],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    logger.info("Starting RAGAS evaluation")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    logger.info("Completed RAGAS evaluation")
    return result
