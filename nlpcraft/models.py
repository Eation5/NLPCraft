import torch
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):
    """A text classification model using a pre-trained Transformer encoder."""
    def __init__(self, model_name, num_labels):
        super(TextClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use pooled output for classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
