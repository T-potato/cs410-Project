from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")


def preprocess_article(article: str, model_type:str = "bart", max_length: int = 1024):
    """
    Preprocess the input article by tokenizing it using BartTokenizer.
    
    Args:
        article (str): The input article text to preprocess.
        max_length (int): Maximum token length for compatibility with BART.
        model_type (str): The model type to use for tokenization. Can be "bart" or "t5".
    
    Returns:
        dict: Tokenized input ready for BART summarization.
    """
    if model_type == "t5":
        tokenized_input = t5_tokenizer(
            article,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    elif model_type == "bart":
        tokenized_input = bart_tokenizer(
            article,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    return tokenized_input
