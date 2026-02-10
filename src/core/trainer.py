"""Training loop for fine-tuning the sentiment model.

FINE-TUNING VS TRAINING FROM SCRATCH:
When we trained our CNN image classifier, every weight started random.
Here, the transformer weights are already pre-trained — they "know" language.

Fine-tuning differences:
1. Lower learning rate (2e-5 vs 1e-3) — we don't want to destroy what BERT learned
2. Fewer epochs (3 vs 25) — the model already understands language, just needs
   to learn "positive vs negative"
3. Warmup schedule — start with an even smaller LR, ramp up, then decay
   This prevents the randomly initialized classifier head from sending
   huge gradients through the pre-trained layers early on

TRAINING LOOP (same concept as the CNN, different details):
For each epoch:
  For each batch:
    1. Forward pass: text → model → prediction
    2. Compute loss: how wrong was the prediction?
    3. Backward pass: compute gradients (which direction to adjust weights)
    4. Clip gradients: prevent explosion (transformers are deep networks)
    5. Optimizer step: actually update the weights
    6. Scheduler step: adjust learning rate
"""

import os
import time

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    """Fine-tune the sentiment model.

    Args:
        model: The SentimentModel to train.
        train_loader: Training data DataLoader.
        val_loader: Validation data DataLoader.
        config: Training configuration.
        device: Device to train on (cuda/cpu).

    Returns:
        Dictionary with training history (losses, accuracies).
    """
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    warmup_ratio = config["training"]["warmup_ratio"]
    max_grad_norm = config["training"]["max_grad_norm"]
    use_fp16 = config["training"]["fp16"] and device.type == "cuda"
    model_dir = config["output"]["model_dir"]

    # === Optimizer ===
    # AdamW = Adam with proper weight decay (L2 regularization)
    # We use different LR for the transformer vs classifier head:
    #   - Transformer layers: lower LR (don't disturb pre-trained weights much)
    #   - Classifier head: higher LR (needs to learn from scratch)
    optimizer = AdamW(
        [
            {"params": model.transformer.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr * 10},
        ],
        weight_decay=weight_decay,
    )

    # === Learning Rate Scheduler ===
    # Linear warmup then linear decay:
    #   LR: 0 → lr (warmup) → 0 (decay)
    # Warmup prevents the random classifier head from sending huge
    # gradients through the pre-trained transformer early in training
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # === Mixed Precision ===
    # FP16 = use 16-bit floats instead of 32-bit where possible
    # Nearly 2x faster on RTX GPUs with negligible accuracy loss
    scaler = GradScaler(enabled=use_fp16)

    # === Training Loop ===
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  FP16: {use_fp16}")
    print(f"  Device: {device}\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            # Move data to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass (with optional FP16)
            optimizer.zero_grad()

            with autocast(enabled=use_fp16):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (prevents exploding gradients in deep networks)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track metrics
            total_train_loss += loss.item()
            predictions = torch.argmax(outputs["logits"], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = total_train_loss / (step + 1)
                acc = correct / total * 100
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {acc:.1f}% | "
                    f"LR: {current_lr:.2e}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct / total * 100

        # --- Validation Phase ---
        val_loss, val_acc = _validate(model, val_loader, device, use_fp16)

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"\nEpoch {epoch+1}/{epochs} Complete ({epoch_time:.0f}s)\n"
            f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%\n"
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"  [BEST] New best model saved! (Val Acc: {val_acc:.1f}%)")

        print()

    print(f"Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    return history


def _validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> tuple:
    """Run validation and return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=use_fp16):
                outputs = model(input_ids, attention_mask, labels)

            total_loss += outputs["loss"].item()
            predictions = torch.argmax(outputs["logits"], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy
