from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

def generate_summary(article: str, model_type: str = "bart", max_input_length: int = 1024, max_summary_length: int = 150):
    """
    Generate a summary of the input article using either T5 or BART.
    
    Args:
        article (str): The input article text.
        model_type (str): Type of model, either 't5' or 'bart'.
        max_input_length (int): Max length of the tokenized input.
        max_summary_length (int): Max length of the generated summary.
    
    Returns:
        str: The generated summary.
    """
    # Select the appropriate model and tokenizer
    if model_type == "t5":
        tokenized_input = t5_tokenizer(
            article,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        summary_ids = t5_model.generate(
        tokenized_input['input_ids'],
        max_length=max_summary_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
        )
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elif model_type == "bart":
        tokenized_input = bart_tokenizer(
            article,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        summary_ids = bart_model.generate(
        tokenized_input['input_ids'],
        max_length=max_summary_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        raise ValueError("Invalid model_type. Choose 't5' or 'bart'.")
    return summary