from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

def evaluate_ragas(question, answer, context, ground_truth):
    """Evaluates the RAG system using RAGAS."""
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[c.page_content for c in context]],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    return result
