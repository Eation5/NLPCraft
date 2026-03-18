# NLPCraft

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21C?style=flat-square&logo=huggingface)
![License](https://img.shields.io/github/license/Eation5/NLPCraft?style=flat-square)

## Overview

NLPCraft is a Python library designed for advanced Natural Language Processing (NLP) tasks, focusing on building and deploying custom NLP models with ease. It leverages state-of-the-art deep learning frameworks like PyTorch and integrates seamlessly with Hugging Face Transformers for various applications, including text classification, named entity recognition (NER), sentiment analysis, and question answering.

## Features

- **Modular Architecture**: Easily swap out components for different models and datasets.
- **Hugging Face Integration**: Utilize pre-trained models and tokenizers from the Transformers library.
- **Customizable Training Loops**: Fine-tune models with custom training and evaluation logic.
- **Data Preprocessing Utilities**: Tools for tokenization, padding, and dataset creation.
- **Evaluation Metrics**: Support for common NLP metrics like accuracy, precision, recall, F1-score.
- **Deployment Ready**: Designed with considerations for easy integration into production systems.

## Installation

To get started with NLPCraft, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Eation5/NLPCraft.git
cd NLPCraft
pip install -r requirements.txt
```

## Usage

Here's a quick example of how to use NLPCraft for text classification:

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from nlpcraft.models import TextClassifier
from nlpcraft.datasets import TextClassificationDataset
from nlpcraft.trainers import Trainer

# 1. Configuration
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2 # Example: positive/negative sentiment
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 2. Sample Data
train_texts = ["This movie is great!", "I hate this film.", "Amazing performance.", "So boring."]
train_labels = [1, 0, 1, 0]

val_texts = ["What a wonderful experience.", "Terrible acting."]
val_labels = [1, 0]

# 3. Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Create Datasets
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 5. Initialize Model
model = TextClassifier(MODEL_NAME, NUM_LABELS)

# 6. Initialize Trainer and Train
trainer = Trainer(model, LEARNING_RATE)
trainer.train(train_dataloader, val_dataloader, EPOCHS)

# 7. Make a prediction
sample_text = "This is an excellent tool for NLP."
encoded_input = tokenizer(sample_text, return_tensors=\'pt\', padding=\'max_length\', truncation=True, max_length=MAX_LEN)

model.eval()
with torch.no_grad():
    outputs = model(encoded_input["input_ids"], encoded_input["attention_mask"])
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Prediction for \"{sample_text}\": {prediction} (0: Negative, 1: Positive)")
```

## Project Structure

```
NLPCraft/
├── README.md
├── requirements.txt
├── setup.py
├── nlpcraft/
│   ├── __init__.py
│   ├── models.py
│   ├── datasets.py
│   ├── trainers.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_models.py
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries, please open an issue on GitHub or contact Matthew Wilson at [matthew.wilson.ai@example.com](mailto:matthew.wilson.ai@example.com).
