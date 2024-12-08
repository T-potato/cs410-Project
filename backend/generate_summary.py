from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Only loading fine-tuned model if it exists and has model files
fine_tuned_path = "./fine_tuned_summarizer"
fine_tuned_model = None
fine_tuned_tokenizer = None

if os.path.exists(fine_tuned_path) and any(
    os.path.exists(os.path.join(fine_tuned_path, f)) 
    for f in ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
):
    try:
        fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_path)
        fine_tuned_tokenizer = BartTokenizer.from_pretrained(fine_tuned_path)
    except Exception as e:
        print(f"Warning: Failed to load fine-tuned model: {e}")
        fine_tuned_model = None
        fine_tuned_tokenizer = None

def generate_summary(article: str, model_type: str = "bart", max_input_length: int = 1024, max_summary_length: int = 150):
    """
    Generate a summary of the input article using either T5, BART, or fine-tuned BART.
    
    Args:
        article (str): The input article text.
        model_type (str): Type of model, either 't5', 'bart', or 'bart-finetuned'.
        max_input_length (int): Max length of the tokenized input.
        max_summary_length (int): Max length of the generated summary.
    
    Returns:
        str: The generated summary.
    """
    if model_type == "t5":
        model = t5_model
        tokenizer = t5_tokenizer
    elif model_type == "bart":
        model = bart_model
        tokenizer = bart_tokenizer
    elif model_type == "bart-finetuned":
        if fine_tuned_model is None:
            raise ValueError("Fine-tuned model not found. Please run train_summarizer.py first.")
        model = fine_tuned_model
        tokenizer = fine_tuned_tokenizer
    else:
        raise ValueError("Invalid model_type. Choose 't5', 'bart', or 'bart-finetuned'.")

    # Tokenize and generate summary
    tokenized_input = tokenizer(
        article,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    summary_ids = model.generate(
        tokenized_input['input_ids'],
        max_length=max_summary_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
