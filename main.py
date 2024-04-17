from dataset_utils import load_samsum_dataset
from trainer import train_bart_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BART for extractive summarization")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the pre-trained BART model"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the trained model"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before stopping",
    )
    args = parser.parse_args()

    train_dataset = load_samsum_dataset(tokenizer, split="train")
    val_dataset = load_samsum_dataset(tokenizer, split="validation")

    train_bart_model(
        args.model_name,
        train_dataset,
        val_dataset,
        args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
    )
