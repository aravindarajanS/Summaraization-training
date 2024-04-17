from typing import Dict
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)



def compute_metrics(pred: Dict) -> Dict:
    """
    Calculates ROUGE scores for model predictions.

    Args:
        pred: A dictionary containing predictions and labels.

    Returns:
        A dictionary containing ROUGE scores (rouge1, rouge2, rougeL).
    """

    pred_ids = pred.sequences.tolist()
    labels_ids = pred.labels.tolist()

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    get_rouge_scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = get_rouge_scores.score(labels,preds)
    return {k: v.mid for k, v in results.items()}



def train_bart_model(
    model_name: str,
    train_dataset: List[Dict],
    val_dataset: List[Dict],
    output_dir: str,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 8,
    early_stopping_patience: int = 3,
) -> None:
    """
    Training a BART model for extractive summarization.

    Args:
        model_name: The name of the pre-trained BART model.
        train_dataset: A list of processed training data .
        val_dataset: A list of processed validation data .
        output_dir: The directory to save the trained model.
        learning_rate: The learning rate for training.
        epochs: The number of training epochs.
        batch_size: The batch size for training.
        early_stopping_patience: The number of epochs to wait for improvement before stopping.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_steps=args.save_steps,
        evaluation_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        fp16=True if torch.cuda.is_available() else False,
        learning_rate_schedule="linear",
        early_stopping_patience=early_stopping_patience,
    )


     trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

     for epoch in range(1, training_args.num_train_epochs + 1):
        # Train for the epoch
        trainer.train(epoch)

        # Evaluate on the validation set and print metrics
        results = trainer.evaluate(eval_dataset=val_dataset)
        train_loss = trainer.state.train_loss

        print(f"Epoch {epoch}:")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {results['eval_loss']:.4f}")
        print(f"\tValidation ROUGE-1: {results['rouge1']:.4f}")
        print(f"\tValidation ROUGE-2: {results['rouge2']:.4f}")
        print(f"\tValidation ROUGE-L: {results['rougeL']:.4f}")

    trainer.save_model()

    print("Training complete!")
