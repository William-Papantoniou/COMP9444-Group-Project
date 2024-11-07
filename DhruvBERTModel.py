from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_map = {'neg': 0, 'neu': 1, 'pos': 2}

def text_pipeline(text):
    return tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [label_map[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = text_pipeline(text)
        return encoding, label

df = pd.read_csv('COVIDSenti.csv')
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
train_texts, train_labels = train_df['tweet'].tolist(), train_df['label'].tolist()
test_texts, test_labels = test_df['tweet'].tolist(), test_df['label'].tolist()

train_dataset = TweetDataset(train_texts, train_labels)
test_dataset = TweetDataset(test_texts, test_labels)

def collate_fn(batch):
    texts, labels = zip(*batch)
    input_ids = torch.cat([item['input_ids'] for item in texts], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in texts], dim=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)  # Move model to GPU
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_model(model, dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            # Move inputs and labels to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            inputs['labels'] = labels  
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')

num_epochs = 5
train_model(model, train_loader, optimizer, num_epochs)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            # Move inputs and labels to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

true_labels, predicted_labels = evaluate_model(model, test_loader)
print(f'Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}')
print("Classification Report for COVIDSenti:")
print(classification_report(true_labels, predicted_labels, target_names=['Negative', 'Neutral', 'Positive']))
