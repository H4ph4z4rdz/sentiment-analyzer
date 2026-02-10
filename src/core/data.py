"""Data loading and preprocessing for sentiment analysis.

WHAT HAPPENS HERE:
1. Download the IMDb movie review dataset from Hugging Face
2. Tokenize the text (convert words → numbers that BERT understands)
3. Split into train/validation/test sets
4. Create PyTorch DataLoaders for efficient batching

TOKENIZATION EXPLAINED:
BERT doesn't read words — it reads "tokens" (subword pieces).
  "unhappiness" → ["un", "##happi", "##ness"] → [4895, 23920, 2791]

Each token maps to a learned embedding vector. BERT's vocabulary has ~30,000 tokens.

The tokenizer also adds special tokens:
  [CLS] I loved this movie [SEP]
  [CLS] = "start of sequence" — its embedding becomes the sentence representation
  [SEP] = "end of sequence" — tells BERT where the input ends
"""

from typing import Dict, Tuple

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def load_and_prepare_data(config: dict) -> Tuple[DatasetDict, AutoTokenizer]:
    """Load IMDb dataset, tokenize, and prepare for training.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (tokenized DatasetDict, tokenizer).
    """
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    val_ratio = config["data"]["validation_ratio"]

    # Step 1: Load the dataset from Hugging Face Hub
    # This downloads ~80MB of movie reviews (25k train + 25k test)
    print(f"Loading {config['data']['dataset']} dataset...")
    dataset = load_dataset(config["data"]["dataset"])

    # Step 2: Limit samples if configured (useful for quick experiments)
    max_train = config["data"].get("max_train_samples")
    max_test = config["data"].get("max_test_samples")

    if max_train and max_train < len(dataset["train"]):
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_train))
    if max_test and max_test < len(dataset["test"]):
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(max_test))

    # Step 3: Split training data into train + validation
    # We need validation data to monitor for overfitting during training
    train_val = dataset["train"].train_test_split(test_size=val_ratio, seed=42)
    dataset = DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": dataset["test"],
    })

    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    # Step 4: Load the tokenizer
    # The tokenizer must match the model — each model has its own vocabulary
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 5: Tokenize all texts
    # This converts raw text into the format BERT expects:
    #   - input_ids: token IDs (numbers representing each word piece)
    #   - attention_mask: 1 for real tokens, 0 for padding
    #     (BERT needs fixed-length inputs, so shorter texts get padded)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",      # Pad short texts to max_length
            truncation=True,           # Cut long texts at max_length
            max_length=max_length,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,                  # Process in batches (much faster)
        desc="Tokenizing",
    )

    # Step 6: Set format for PyTorch
    # Tell the dataset to return PyTorch tensors instead of Python lists
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized, tokenizer


def create_dataloaders(
    dataset: DatasetDict,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for each split.

    DataLoaders handle:
    - Batching: Group samples into batches for parallel processing on GPU
    - Shuffling: Randomize order each epoch (prevents learning data order)
    - Memory: Load data lazily, not all at once

    Args:
        dataset: Tokenized DatasetDict.
        batch_size: Number of samples per batch.

    Returns:
        Dictionary of DataLoaders for each split.
    """
    loaders = {}

    for split in dataset:
        loaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == "train"),   # Only shuffle training data
            num_workers=0,                # Windows doesn't like multiprocess dataloading
            pin_memory=True,              # Faster GPU transfer
        )

    return loaders
