import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForCausalLM

class TextClassifier(nn.Module):
    """A text classification model using a pre-trained Transformer encoder."""
    def __init__(self, model_name, num_labels):
        super(TextClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class SequenceTagger(nn.Module):
    """A sequence tagging model (e.g., for NER) using a pre-trained Transformer encoder."""
    def __init__(self, model_name, num_labels):
        super(SequenceTagger, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class TextGenerator(nn.Module):
    """A text generation model using a pre-trained causal language model."""
    def __init__(self, model_name):
        super(TextGenerator, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(self, input_ids, max_length=50, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0):
        return self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.model.config.eos_token_id,
        )
