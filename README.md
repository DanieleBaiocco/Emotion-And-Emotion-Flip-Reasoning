# MELD Emotion & Emotion-Flip Reasoning

A command-line project for **emotion recognition** and **emotion-flip trigger detection** in multi-speaker conversations.

Given a dialogue made of utterances and speaker IDs, the model predicts for each turn:

- the **emotion** expressed in the utterance;
- the probability that the utterance acts as an **emotion-flip trigger**;
- a binary `trigger` decision obtained using the threshold selected during validation.

The project is designed so that training and inference use the same Python package and the same CLI. A trained repository can therefore be cloned and used directly from a terminal without opening a notebook.

---

## What the model does

Input:

```json
{
  "utterances": [
    "I finally got the promotion!",
    "That is amazing, congratulations!"
  ],
  "speakers": [
    "A",
    "B"
  ]
}
```

For every utterance, the model returns:

- `emotion`: predicted emotion class;
- `trigger_probability`: estimated probability of being an emotion-flip trigger;
- `trigger`: binary prediction (`0` or `1`);
- the original utterance, speaker and index for convenience.

The model uses a Transformer encoder for utterance representations and a contextual sequence model to reason across the dialogue.

---

## Project structure

```text
meld-efr-cli/
├── artifacts/                 # Trained model and inference assets
│   ├── best_model.pt
│   ├── tokenizer/
│   ├── encoder_config/
│   └── ...
├── data/                      # Training / validation / test datasets
├── examples/                  # Example input dialogues
├── src/
│   └── meld_efr/
│       ├── cli.py
│       ├── data.py
│       ├── inference.py
│       ├── model.py
│       ├── training.py
│       └── utils.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.10+**
- PyTorch
- Transformers
- scikit-learn
- NumPy

A CUDA-compatible NVIDIA GPU is recommended for training, but inference can also run on CPU.

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd meld-efr-cli
```

Optionally create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the project:

```bash
pip install -e .
```

The `-e` option installs the project in **editable mode**, so changes made under `src/` are immediately reflected without reinstalling the package.

After installation, the CLI is available as:

```bash
meld-efr --help
```

The equivalent module syntax is:

```bash
python -m meld_efr --help
```

---

# Inference

Inference requires a trained checkpoint, normally stored at:

```text
artifacts/best_model.pt
```

Run prediction with:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

Equivalent command:

```bash
python -m meld_efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

## Prediction options

| Option | Required | Description |
|---|---:|---|
| `--checkpoint PATH` | Yes | Path to the trained `.pt` checkpoint, usually `artifacts/best_model.pt`. |
| `--input PATH` | Yes | JSON file containing one dialogue or a list of dialogues. |
| `--output PATH` | No | File where predictions are saved as JSON. If omitted, predictions are still printed to the terminal. |
| `--device DEVICE` | No | Inference device. Use `auto`, `cpu`, `cuda`, `cuda:0`, etc. Default: `auto`. |

With:

```bash
--device auto
```

the program automatically uses CUDA when a compatible GPU is available; otherwise it falls back to CPU.

---

## Input format

A dialogue must contain:

- `utterances`: ordered list of dialogue turns;
- `speakers`: speaker ID for each corresponding utterance.

The two arrays must have the same length.

Example:

```json
{
  "utterances": [
    "Hello!",
    "Hi, how are you?"
  ],
  "speakers": [
    "A",
    "B"
  ]
}
```

No emotion or trigger labels are required for inference.

---

# Complete inference example

Assume the following file exists:

```text
examples/dialogue1.json
```

with this content:

```json
{
  "utterances": [
    "I finally got the promotion!",
    "That is amazing, congratulations!",
    "Thanks. I was sure they would choose someone else.",
    "You earned it. We should celebrate tonight."
  ],
  "speakers": [
    "A",
    "B",
    "A",
    "B"
  ]
}
```

Run:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output artifacts/dialogue1_predictions.json \
  --device auto
```

Example output produced by the trained model:

```json
[
  {
    "trigger_threshold": 0.7500000000000002,
    "predictions": [
      {
        "index": 0,
        "speaker": "A",
        "utterance": "I finally got the promotion!",
        "emotion": "joy",
        "trigger_probability": 0.083826,
        "trigger": 0
      },
      {
        "index": 1,
        "speaker": "B",
        "utterance": "That is amazing, congratulations!",
        "emotion": "joy",
        "trigger_probability": 0.155064,
        "trigger": 0
      },
      {
        "index": 2,
        "speaker": "A",
        "utterance": "Thanks. I was sure they would choose someone else.",
        "emotion": "joy",
        "trigger_probability": 0.881737,
        "trigger": 1
      },
      {
        "index": 3,
        "speaker": "B",
        "utterance": "You earned it. We should celebrate tonight.",
        "emotion": "joy",
        "trigger_probability": 0.900187,
        "trigger": 1
      }
    ]
  }
]
```

### How to read the output

For example:

```json
{
  "emotion": "joy",
  "trigger_probability": 0.881737,
  "trigger": 1
}
```

means that the model:

1. classified the utterance as `joy`;
2. assigned an emotion-flip trigger probability of approximately `0.882`;
3. classified it as a trigger because its probability is above the learned threshold (`0.75` in this example).

`trigger_threshold` is selected from validation data during training and stored with the trained model, rather than being arbitrarily fixed at `0.5`.

---

# Training

Training is also available directly from the CLI.

Expected dataset files:

```text
data/
├── MELD_train_efr.json
├── MELD_val_efr.json
└── MELD_test_efr.json
```

Example:

```bash
meld-efr train \
  --train-path data/MELD_train_efr.json \
  --val-path data/MELD_val_efr.json \
  --test-path data/MELD_test_efr.json \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --encoder-lr 8e-6 \
  --head-lr 7e-5 \
  --dropout 0.5 \
  --weight-decay 0.05 \
  --patience 5 \
  --device auto
```

During training, the program evaluates the model on the validation set and saves the best checkpoint in the output directory.

The main validation metrics are:

- `emotion_macro_f1`: macro-averaged F1 across emotion classes;
- `trigger_f1+`: F1 score for the positive emotion-flip trigger class.

After training, the important inference files are kept under `artifacts/`, including the trained checkpoint and the resources required to reconstruct the model.

---

## Training options: most important parameters

| Option | Description |
|---|---|
| `--train-path` | Training JSON dataset. |
| `--val-path` | Validation JSON dataset. |
| `--test-path` | Test JSON dataset. |
| `--output-dir` | Directory where checkpoints, metrics and model assets are saved. |
| `--epochs` | Maximum number of training epochs. |
| `--patience` | Number of validation rounds without improvement before early stopping. |
| `--batch-size` | Number of dialogues processed per batch. |
| `--grad-accum-steps` | Gradient accumulation steps; useful when GPU memory is limited. |
| `--encoder-lr` | Learning rate of the pretrained Transformer encoder. |
| `--head-lr` | Learning rate of the task-specific/contextual layers. |
| `--dropout` | Dropout probability used for regularization. |
| `--weight-decay` | Weight decay used for regularization. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |

For the full list:

```bash
meld-efr train --help
```

---

# Using the trained repository after `git clone`

Once `artifacts/best_model.pt` and the associated inference assets are committed to the repository, another user can run:

```bash
git clone <REPOSITORY_URL>
cd meld-efr-cli
pip install -e .
```

and then immediately execute:

```bash
meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --device auto
```

No notebook is required for inference.

> Large model files such as `.pt`, `.bin` or `.safetensors` should normally be stored with **Git LFS** when pushing the trained repository to GitHub.

---

## Quick start

```bash
# 1. Clone

git clone <REPOSITORY_URL>
cd meld-efr-cli

# 2. Install

pip install -e .

# 3. Predict

meld-efr predict \
  --checkpoint artifacts/best_model.pt \
  --input examples/dialogue1.json \
  --output predictions.json \
  --device auto
```

