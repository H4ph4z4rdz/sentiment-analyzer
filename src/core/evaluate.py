"""Evaluation metrics for sentiment analysis.

METRICS EXPLAINED:
- Accuracy: % of predictions that are correct (simple but can be misleading)
- Precision: Of all "positive" predictions, how many were actually positive?
- Recall: Of all actually positive samples, how many did we correctly predict?
- F1 Score: Harmonic mean of precision and recall (balances both)

CONFUSION MATRIX:
                 Predicted
              Neg     Pos
  Actual Neg  [TN]    [FP]    ← FP = "false positive" (said positive, was negative)
  Actual Pos  [FN]    [TP]    ← FN = "false negative" (said negative, was positive)

Example: Movie review "This movie was not good" → model says "positive" (tricked by "good")
         That's a False Positive (FP).
"""

import os

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> dict:
    """Evaluate the model on test data.

    Args:
        model: Trained SentimentModel.
        test_loader: Test data DataLoader.
        device: Device to evaluate on.
        use_fp16: Whether to use mixed precision.

    Returns:
        Dictionary with all metrics.
    """
    print("\nEvaluating on test set...")
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=use_fp16):
                outputs = model(input_ids, attention_mask, labels)

            total_loss += outputs["loss"].item()
            predictions = torch.argmax(outputs["logits"], dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Positive"],
    )

    avg_loss = total_loss / len(test_loader)

    # Print results
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Loss:     {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           Neg     Pos")
    print(f"  Neg    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Pos    {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(f"\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "loss": avg_loss,
        "confusion_matrix": cm,
        "report": report,
    }
