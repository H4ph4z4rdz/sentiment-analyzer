"""Sentiment Analyzer — Gradio Web UI.

Launch with:
    python src/app.py

Then open http://localhost:7862 in your browser.
Type or paste any text to analyze its sentiment!
"""

import os
import sys

import yaml
import torch
import gradio as gr
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from core.model import SentimentModel
from utils.helpers import load_config, get_device


def load_model(config: dict, device: torch.device):
    """Load the trained model and tokenizer."""
    model_dir = config["output"]["model_dir"]
    model_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Run 'python src/train.py' first to train the model.")
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model
    model = SentimentModel(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, tokenizer


# Initialize
config = load_config()
device = get_device()
model, tokenizer = load_model(config, device)


def analyze_sentiment(text: str) -> tuple:
    """Analyze sentiment of input text.

    Returns:
        Tuple of (label_with_emoji, confidence_dict_for_gradio_label).
    """
    if not text.strip():
        return "Please enter some text.", {"Positive 😊": 0.5, "Negative 😞": 0.5}

    # Tokenize
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Predict
    result = model.predict(input_ids, attention_mask)

    # Format output
    label = result["label"]
    confidence = result["confidence"]
    probs = result["probabilities"]

    emoji = "😊" if label == "positive" else "😞"
    label_text = f"{label.upper()} {emoji} ({confidence:.1%} confident)"

    # Gradio Label component expects {label: score} dict
    confidence_dict = {
        "Positive 😊": probs["positive"],
        "Negative 😞": probs["negative"],
    }

    return label_text, confidence_dict


def analyze_batch(texts: str) -> str:
    """Analyze multiple texts (one per line)."""
    if not texts.strip():
        return "Please enter one or more texts (one per line)."

    lines = [line.strip() for line in texts.strip().split("\n") if line.strip()]
    results = []

    for i, text in enumerate(lines, 1):
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config["model"]["max_length"],
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        result = model.predict(input_ids, attention_mask)
        emoji = "😊" if result["label"] == "positive" else "😞"
        short_text = text[:80] + "..." if len(text) > 80 else text
        results.append(
            f"{i}. {emoji} {result['label'].upper()} ({result['confidence']:.1%}) — \"{short_text}\""
        )

    return "\n".join(results)


# Build the Gradio UI
with gr.Blocks(title="Sentiment Analyzer") as demo:

    gr.Markdown(
        """
        # 🎭 Sentiment Analyzer
        ### Powered by Fine-tuned DistilBERT

        Analyze the sentiment of any text — movie reviews, product feedback, tweets, anything!
        The model was fine-tuned on 25,000 IMDb movie reviews.
        """
    )

    with gr.Tab("Single Analysis"):
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type or paste any text to analyze...",
                    lines=4,
                )
                analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Label(label="Confidence", num_top_classes=2)

        gr.Examples(
            examples=config["ui"]["examples"],
            inputs=text_input,
            label="Try These Examples",
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown("Analyze multiple texts at once — one per line.")
        batch_input = gr.Textbox(
            label="Texts (one per line)",
            lines=8,
            placeholder="Enter multiple texts, one per line...",
        )
        batch_btn = gr.Button("🔍 Analyze All", variant="primary")
        batch_output = gr.Textbox(label="Results", lines=10, interactive=False)

    gr.Markdown(
        """
        ---
        **How it works:**
        - Text is tokenized into subword pieces using DistilBERT's vocabulary
        - Each token is converted to a 768-dimensional embedding vector
        - The transformer processes all tokens with self-attention (each token looks at every other token)
        - The [CLS] token's final embedding captures the overall sentiment
        - A classification layer maps this to positive/negative probabilities

        *Fine-tuned on IMDb reviews • Built with PyTorch, Hugging Face Transformers & Gradio*
        """
    )

    # Wire up events
    analyze_btn.click(analyze_sentiment, [text_input], [label_output, confidence_output])
    text_input.submit(analyze_sentiment, [text_input], [label_output, confidence_output])
    batch_btn.click(analyze_batch, [batch_input], [batch_output])


if __name__ == "__main__":
    print(f"\n🎭 Launching Sentiment Analyzer...")
    print(f"   Model: {config['model']['name']}")
    print(f"   Open http://localhost:{config['ui']['port']} in your browser\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=config["ui"]["port"],
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
