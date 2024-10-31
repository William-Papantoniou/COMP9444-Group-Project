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

# Check if a GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
    labels = torch.tensor(labels, dtype=torch.long)
    return text_indices_padded.to(device), labels.to(device)

# DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model Definition
class TransformerSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, output_dim, num_layers, padding_idx=None):
        super(TransformerSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Define a single Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.5
        )

        # Stack multiple Transformer Encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.padding_idx = padding_idx

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Create padding mask (True for padding tokens)
        src_key_padding_mask = (x == self.padding_idx)  # [batch_size, seq_len]

        # Permute for Transformer [seq_len, batch_size, embedding_dim]
        embedded = embedded.permute(1, 0, 2)

        # Apply Transformer Encoder
        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch_size, embedding_dim]

        # Permute back to [batch_size, seq_len, embedding_dim]
        transformer_out = transformer_out.permute(1, 0, 2)

        # Create mask for non-padding tokens
        mask = ~src_key_padding_mask  # [batch_size, seq_len]
        mask = mask.unsqueeze(2).float()

        # Apply mask and compute mean over non-padded tokens
        transformer_out = transformer_out * mask
        lengths = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        pooled = transformer_out.sum(dim=1) / lengths  # [batch_size, embedding_dim]

        # Apply dropout and fully connected layer
        output = self.dropout(pooled)
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
num_heads = 4
hidden_dim = 256
output_dim = 3
num_layers = 2
padding_idx = vocab["<PAD>"]
num_epochs = 10

# Instantiate and train the model
model = TransformerSentimentModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    padding_idx=padding_idx
).to(device)

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
