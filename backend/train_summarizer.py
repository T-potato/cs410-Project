from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm.auto import tqdm
import evaluate
import nltk
import os

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt")

def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=128):
    """Preprocess the data for training."""
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        [doc for doc in examples["highlights"]],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = None

    def _create_training_loop_progress_bar(self, *args, **kwargs):
        self.progress_bar = tqdm(total=self.args.num_train_epochs * len(self.train_dataset))
        return self.progress_bar

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"Training Loss: {loss:.4f}")
        return loss

def compute_metrics(eval_pred):
    """Compute ROUGE metrics."""
    rouge_score = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {key: round(value * 100, 2) for key, value in result.items()}

def train_summarizer(
    model_name="facebook/bart-large-cnn",
    dataset_name="cnn_dailymail",
    dataset_config="3.0.0",
    output_dir="./fine_tuned_summarizer",
    num_train_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_input_length=1024,
    max_target_length=128,
):
    """
    Fine-tune a summarization model on a specific dataset.
    
    Args:
        model_name (str): Name of the pre-trained model
        dataset_name (str): Name of the dataset to use for fine-tuning
        dataset_config (str): Dataset configuration
        output_dir (str): Directory to save the fine-tuned model
        num_train_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        max_input_length (int): Maximum input sequence length
        max_target_length (int): Maximum target sequence length
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, dataset_config)
    
    print("Loading tokenizer and model...")
    global tokenizer  # Make tokenizer accessible to compute_metrics
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_input_length, max_target_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset",
    )

    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=100,
        logging_first_step=True,
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("Initializing trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    print("Starting fine-tuning process...")
    train_summarizer(
        model_name="facebook/bart-large-cnn",
        dataset_name="cnn_dailymail",
        dataset_config="3.0.0",
        output_dir="./fine_tuned_summarizer",
        num_train_epochs=3,
        batch_size=4,
    )
