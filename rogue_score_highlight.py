from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np

def calculate_rouge(dataset, model_type="bart", max_input_length=1024, max_summary_length=150):
    """
    Evaluates ROUGE scores for the entire dataset with either T5 or BART.

    Args:
        dataset (Dataset): The dataset with 'article' and 'highlights' columns.
        model_type (str): Type of model, either 't5' or 'bart'.
        max_input_length (int): Maximum token length for input.
        max_summary_length (int): Maximum token length for generated summary.
    
    Returns:
        dict: Average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for example in dataset:
        # Generate the summary
        generated_summary = generate_summary(example["article"], model_type, max_input_length, max_summary_length)

        # Calculate ROUGE scores
        scores = scorer.score(example["highlights"], generated_summary)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Calculate average ROUGE scores
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)

    return {
        "Average ROUGE-1": avg_rouge1,
        "Average ROUGE-2": avg_rouge2,
        "Average ROUGE-L": avg_rougeL
    }

# Example usage
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")  # Adjust to "train" or "validation" as needed

# Evaluate with BART
bart_rouge_scores = calculate_rouge(dataset, model_type="bart")
print("BART ROUGE Scores:", bart_rouge_scores)

# Evaluate with T5
t5_rouge_scores = calculate_rouge(dataset, model_type="t5")
print("T5 ROUGE Scores:", t5_rouge_scores)
