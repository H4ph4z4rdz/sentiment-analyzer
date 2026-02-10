"""Train the sentiment analysis model.

Usage:
    python src/train.py

This script:
1. Loads the IMDb dataset
2. Tokenizes text for DistilBERT
3. Fine-tunes the model
4. Evaluates on the test set
5. Saves the best model to models/sentiment-model/
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer

from core.data import load_and_prepare_data, create_dataloaders
from core.model import SentimentModel
from core.trainer import train_model
from core.evaluate import evaluate_model
from utils.helpers import load_config, get_device


def main():
    print("=" * 60)
    print("  SENTIMENT ANALYZER — Fine-tuning DistilBERT on IMDb")
    print("=" * 60)

    # Load config
    config = load_config()

    # Setup device
    device = get_device()

    # Load and prepare data
    dataset, tokenizer = load_and_prepare_data(config)
    loaders = create_dataloaders(dataset, config["training"]["batch_size"])

    # Save tokenizer for inference later
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    print(f"Tokenizer saved to {model_dir}")

    # Create model
    model = SentimentModel(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )
    model.to(device)

    # Train
    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        config=config,
        device=device,
    )

    # Load best model for evaluation
    best_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path}")

    # Evaluate on test set
    use_fp16 = config["training"]["fp16"] and device.type == "cuda"
    metrics = evaluate_model(model, loaders["test"], device, use_fp16)

    print(f"\nFinal Test Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Model saved to: {model_dir}")
    print("\nRun the Gradio UI with: python src/app.py")


if __name__ == "__main__":
    main()
