from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from Covid19PrepareData import TweetDataset, preprocess_data, clean_text

# Load preprocessed data
train_dataset, test_dataset, vocab = preprocess_data('COVIDSenti.csv')

# Tokenization function
def text_pipeline(text):
    return [vocab[token] if token in vocab else vocab["<UNK>"] for token in text.split()]

# Collate function to pad sequences to the same length
def collate_fn(batch):
    texts, labels = zip(*batch)
    text_indices = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in texts]
    text_indices_padded = pad_sequence(text_indices, batch_first=True, padding_value=vocab["<PAD>"])
    return text_indices_padded, torch.tensor(labels, dtype=torch.long)

# DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model Definition
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=None):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss, correct, total = 0, 0, 0
        for text, label in dataloader:
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')

# Parameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 3
padding_idx = vocab["<PAD>"]
num_epochs = 5

# Instantiate and train the model
model = SentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, label in dataloader:
            output = model(text)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    return all_labels, all_preds

# Test on COVIDSenti dataset
true_labels, predicted_labels = evaluate_model(model, test_loader)
print(f'Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}')
print("Classification Report for COVIDSenti:")
print(classification_report(true_labels, predicted_labels, target_names=['Negative', 'Neutral', 'Positive']))