"""Generate README charts for the sentiment analyzer.

Run this after training to create visualizations:
    python src/generate_charts.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark theme
BG = "#1a1a2e"
TEXT = "#e0e0e0"
GRID = "#2a2a4a"
CYAN = "#00d4ff"
RED = "#ff6b6b"
GREEN = "#51cf66"
YELLOW = "#ffd43b"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": GRID, "grid.alpha": 0.3, "font.size": 11,
})

os.makedirs("assets", exist_ok=True)

# === Training Loss & Accuracy ===
# Data from our actual training run
epochs = [1, 2, 3]
train_loss = [0.3277, 0.1536, 0.0985]
val_loss = [0.2398, 0.2920, 0.4011]
train_acc = [85.6, 94.5, 97.3]
val_acc = [90.5, 90.0, 91.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(epochs, train_loss, color=GREEN, marker="o", linewidth=2, markersize=8, label="Train Loss")
ax1.plot(epochs, val_loss, color=RED, marker="o", linewidth=2, markersize=8, label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
ax1.legend(framealpha=0.3)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# Accuracy
ax2.plot(epochs, train_acc, color=GREEN, marker="o", linewidth=2, markersize=8, label="Train Acc")
ax2.plot(epochs, val_acc, color=CYAN, marker="o", linewidth=2, markersize=8, label="Val Acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
ax2.legend(framealpha=0.3)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_ylim(80, 100)

plt.tight_layout()
plt.savefig("assets/training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: assets/training_history.png")

# === Confusion Matrix ===
cm = np.array([[2244, 250], [205, 2301]])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues", alpha=0.8)

labels = ["Negative", "Positive"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")

# Add text annotations
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > 1500 else TEXT
        ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/50:.1f}%)",
                ha="center", va="center", fontsize=16, fontweight="bold", color=color)

plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: assets/confusion_matrix.png")

# === Architecture Diagram (text-based in chart) ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

# Pipeline steps
steps = [
    ("Input Text", '"This movie was amazing!"', CYAN),
    ("Tokenizer", "[CLS] this movie was amazing ! [SEP]\n[101, 2023, 3185, 2001, 6429, 999, 102]", YELLOW),
    ("DistilBERT\n(6 layers)", "Self-attention across all tokens\n768-dim embeddings per token", GREEN),
    ("[CLS] Token", "Single 768-dim vector\nrepresenting whole sentence", CYAN),
    ("Classifier", "Linear(768 -> 2)\nDropout(0.1)", RED),
    ("Output", "Positive: 97.3%\nNegative: 2.7%", GREEN),
]

y_pos = 0.5
for i, (title, desc, color) in enumerate(steps):
    x = 0.08 + i * 0.155
    # Box
    rect = plt.Rectangle((x, 0.2), 0.13, 0.6, linewidth=2,
                          edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(rect)
    # Title
    ax.text(x + 0.065, 0.65, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color)
    # Description
    ax.text(x + 0.065, 0.4, desc, ha="center", va="center",
            fontsize=7, color=TEXT)
    # Arrow
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 0.15, 0.5), xytext=(x + 0.13, 0.5),
                    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("DistilBERT Sentiment Analysis Pipeline", fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("assets/architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: assets/architecture.png")

print("\nAll charts generated!")
