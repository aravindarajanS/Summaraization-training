from typing import Dict, List
from datasets import load_dataset

def create_training_data(examples: Dict) -> Dict:
    """
    Prepares training data for extractive summarization.

    Args:
        examples: A dictionary containing "context" and "highlights" keys.

    Returns:
        A dictionary containing processed tokens and attention masks.
    """

    source_text = examples["dialogue"]
    target_text = examples["summary"]

    return tokenizer(
        source_text,
        truncation=True,
        padding="max_length",
        max_length=args.max_len,    
        return_tensors="pt",
    )


def load_samsum_dataset(tokenizer, split: str = "train") -> List[Dict]:
    """
    Loads the SAMSum dataset for a specific split.

    Args:
        tokenizer: The tokenizer used for processing text.
        split: The split of the dataset ("train" or "validation").

    Returns:
        A list of processed data points.
    """

    dataset = load_dataset("samsum", split=split)
    return dataset.map(create_training_data, batched=True)
