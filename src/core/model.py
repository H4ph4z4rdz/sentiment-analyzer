"""Sentiment classification model using a pre-trained transformer.

WHAT IS FINE-TUNING?
Instead of training a model from scratch (like our CNN), we start with a model
that already "understands" language. BERT was trained on billions of words from
books and Wikipedia. It learned grammar, meaning, context, even some world knowledge.

Fine-tuning = take that pre-trained knowledge and specialize it for our task.
We add a small classification layer on top and train the whole thing on our
sentiment data. The pre-trained layers adjust slightly, the new layer learns
from scratch.

It's like hiring someone who already speaks English fluently and just teaching
them to be a movie critic — much easier than teaching a baby from scratch.

WHY DISTILBERT?
DistilBERT is a smaller version of BERT:
- 66M parameters vs BERT's 110M (40% smaller)
- 60% faster inference
- Retains 97% of BERT's performance
Perfect for learning — same concepts, less compute needed.

ARCHITECTURE:
  Input text
      ↓
  [Tokenizer] → token IDs + attention mask
      ↓
  [DistilBERT] → 768-dimensional embedding for each token
      ↓
  [CLS token embedding] → single 768-dim vector representing the whole text
      ↓
  [Dropout] → regularization (prevents overfitting)
      ↓
  [Linear layer] → 768 → 2 (positive / negative)
      ↓
  Output logits → softmax → probabilities
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SentimentModel(nn.Module):
    """Transformer-based sentiment classifier.

    Uses a pre-trained transformer as the backbone and adds a classification
    head on top. The entire model is fine-tuned end-to-end.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):
        """Initialize the sentiment model.

        Args:
            model_name: Name of the pre-trained transformer model.
            num_labels: Number of output classes (2 for binary sentiment).
        """
        super().__init__()

        self.num_labels = num_labels

        # Load pre-trained transformer
        # This downloads the model weights (~250MB for DistilBERT)
        print(f"Loading pre-trained model: {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Classification head
        # Takes the [CLS] token's hidden state and maps it to class logits
        hidden_size = self.config.hidden_size  # 768 for DistilBERT
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """Forward pass through the model.

        How it works step by step:
        1. Feed tokens through the transformer → get hidden states for ALL tokens
        2. Extract only the [CLS] token's hidden state (first token)
           - The [CLS] token learns to encode the "meaning" of the whole sentence
        3. Apply dropout (randomly zero out some values during training)
        4. Feed through linear layer → get raw scores (logits) for each class
        5. If labels provided, compute cross-entropy loss

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Optional ground truth labels [batch_size]

        Returns:
            Dictionary with 'logits' and optionally 'loss'.
        """
        # Step 1: Run through transformer
        # Output shape: [batch_size, seq_length, hidden_size]
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Step 2: Get [CLS] token representation
        # The [CLS] token is always at position 0
        # Shape: [batch_size, hidden_size] = [batch_size, 768]
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Step 3: Dropout for regularization
        cls_output = self.dropout(cls_output)

        # Step 4: Classification
        # Shape: [batch_size, num_labels] = [batch_size, 2]
        logits = self.classifier(cls_output)

        result = {"logits": logits}

        # Step 5: Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Make a prediction with probabilities.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.

        Returns:
            Dictionary with 'label', 'confidence', and 'probabilities'.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            probs = torch.softmax(output["logits"], dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)

        labels = ["negative", "positive"]

        return {
            "label": labels[predicted.item()],
            "confidence": confidence.item(),
            "probabilities": {
                "negative": probs[0][0].item(),
                "positive": probs[0][1].item(),
            },
        }
