import torch
from tqdm import tqdm

class Trainer:
    """Handles the training and evaluation loop for NLP models."""
    def __init__(self, model, learning_rate, device=\"cuda\" if torch.cuda.is_available() else \"cpu\"):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=\"Training\", leave=False):
            self.optimizer.zero_grad()
            input_ids = batch[\"input_ids\"].to(self.device)
            attention_mask = batch[\"attention_mask\"].to(self.device)
            labels = batch[\"labels\"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return total_loss / len(dataloader)

    def evaluate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=\"Evaluating\", leave=False):
                input_ids = batch[\"input_ids\"].to(self.device)
                attention_mask = batch[\"attention_mask\"].to(self.device)
                labels = batch[\"labels\"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted_labels = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
        
        accuracy = correct_predictions / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy

    def train(self, train_dataloader, val_dataloader, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_dataloader)
            val_loss, val_accuracy = self.evaluate_epoch(val_dataloader)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


