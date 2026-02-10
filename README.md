# Sentiment Analyzer

A sentiment analysis model fine-tuned on IMDb movie reviews using DistilBERT. Classifies text as positive or negative with confidence scores. Includes a Gradio web UI for interactive analysis.

## How It Works

```
Input Text: "This movie was absolutely fantastic!"
     ↓
[Tokenizer] → Split into subword tokens → Convert to IDs
     ↓
[DistilBERT] → Self-attention across all tokens → Contextual embeddings
     ↓
[CLS Token] → Single vector representing the whole sentence
     ↓
[Classifier] → Linear layer → Positive: 97.3% | Negative: 2.7%
```

## Features

- **Fine-tuned DistilBERT** — Pre-trained on billions of words, specialized for sentiment
- **IMDb Dataset** — Trained on 25,000 movie reviews (positive/negative)
- **GPU Accelerated** — Mixed precision (FP16) training on NVIDIA GPUs
- **Web UI** — Gradio interface for single and batch analysis
- **Batch Mode** — Analyze multiple texts at once

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** recommended (works on CPU, just slower)

## Setup

```bash
# Clone the repo
git clone git@github.com:H4ph4z4rdz/sentiment-analyzer.git
cd sentiment-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python src/train.py
```

This will:
- Download the IMDb dataset (~80MB)
- Download DistilBERT weights (~250MB)
- Fine-tune for 3 epochs (~5-10 minutes on GPU)
- Save the best model to `models/sentiment-model/`

### 2. Launch the Web UI

```bash
python src/app.py
```

Open **http://localhost:7862** in your browser.

## Project Structure

```
sentiment-analyzer/
├── configs/
│   └── default.yaml          # All configuration
├── models/                    # Saved model weights (auto-generated)
└── src/
    ├── train.py               # Training script
    ├── app.py                 # Gradio web UI
    └── core/
        ├── data.py            # Dataset loading & tokenization
        ├── model.py           # DistilBERT + classification head
        ├── trainer.py         # Fine-tuning training loop
        └── evaluate.py        # Test metrics & evaluation
```

## Configuration

Edit `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | distilbert-base-uncased | Pre-trained transformer |
| `model.max_length` | 256 | Max input tokens |
| `training.epochs` | 3 | Fine-tuning epochs |
| `training.batch_size` | 16 | Batch size |
| `training.learning_rate` | 2e-5 | Learning rate |
| `training.fp16` | true | Mixed precision training |

## Tech Stack

- **PyTorch** — Deep learning framework
- **Hugging Face Transformers** — Pre-trained models
- **Hugging Face Datasets** — Dataset loading
- **scikit-learn** — Evaluation metrics
- **Gradio** — Web UI

## License

MIT
