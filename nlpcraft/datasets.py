import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks, handling tokenization and formatting."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors=\'pt\',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SequenceTaggingDataset(Dataset):
    """Dataset for sequence tagging tasks (e.g., NER), handling tokenization and label alignment."""
    def __init__(self, sentences, tags, tokenizer, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag_to_id = self._get_tag_mapping(tags)

    def _get_tag_mapping(self, tags):
        unique_tags = sorted(list(set(tag for sublist in tags for tag in sublist)))
        return {tag: i for i, tag in enumerate(unique_tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.tags[idx]

        tokens = []
        labels = []

        for word, label in zip(sentence, word_labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # For simplicity, assign the label to the first token of the word
            # and -100 (ignore index) to subsequent tokens of the same word
            labels.extend([self.tag_to_id[label]] + [-100] * (len(word_tokens) - 1))

        # Truncate or pad tokens and labels
        if len(tokens) > self.max_len - 2: # -2 for [CLS] and [SEP]
            tokens = tokens[:self.max_len - 2]
            labels = labels[:self.max_len - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        labels = [-100] + labels + [-100]

        attention_mask = [1] * len(input_ids)

        # Pad to max_len
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        labels = labels + ([-100] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class TextGenerationDataset(Dataset):
    """Dataset for text generation tasks, preparing text for causal language models."""
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors=\'pt\',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten() # For causal LM, labels are usually the input_ids shifted
        }

class NLPCraftDataLoader:
    """A utility class to load and prepare data for various NLP tasks."""
    def __init__(self, model_name, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def get_classification_dataloader(self, texts, labels, batch_size=16, shuffle=True):
        dataset = TextClassificationDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_tagging_dataloader(self, sentences, tags, batch_size=16, shuffle=True):
        dataset = SequenceTaggingDataset(sentences, tags, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_generation_dataloader(self, texts, batch_size=16, shuffle=True):
        dataset = TextGenerationDataset(texts, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
