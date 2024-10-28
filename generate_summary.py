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
        model = t5_model
        tokenizer = t5_tokenizer
    elif model_type == "bart":
        model = bart_model
        tokenizer = bart_tokenizer
    else:
        raise ValueError("Invalid model_type. Choose 't5' or 'bart'.")

    # Preprocess the article
    tokenized_input = preprocess_article(article, model_type=model_type, max_length=max_input_length)
    
    # Generate the summary
    summary_ids = model.generate(
        tokenized_input['input_ids'],
        max_length=max_summary_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
