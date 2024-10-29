import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset

# Function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove @ mentions
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace('\n', ' ').lower().strip()  # Convert to lowercase and trim
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Class for the dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Custom vocabulary building
def build_vocab(sentences):
    word_to_index = defaultdict(lambda: len(word_to_index))
    word_to_index["<PAD>"] = 0  # Reserve index 0 for padding
    word_to_index["<UNK>"] = 1  # Reserve index 1 for unknown words
    
    for sentence in sentences:
        tokens = sentence.split()  # Tokenization
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = len(word_to_index)
    
    return word_to_index

# Function to preprocess data and return train/test datasets
def preprocess_data(file_path, test_size=0.2, random_state=42):
    # Load data
    data = pd.read_csv(file_path, header=None, names=['tweet', 'label'], quotechar='"', lineterminator='\n')
    data = data.iloc[1:].reset_index(drop=True)
    
    # Clean tweets and strip labels
    data['cleaned_tweet'] = data['tweet'].apply(clean_text)
    data['label'] = data['label'].str.strip()

    # Define valid labels and filter data
    valid_labels = ['neu', 'neg', 'pos']
    label_mapping = {'neu': 1, 'neg': 0, 'pos': 2}
    filtered_data = data[data['label'].isin(valid_labels)].copy()
    filtered_data['encoded_label'] = filtered_data['label'].map(label_mapping)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_data['cleaned_tweet'].values,
        filtered_data['encoded_label'].values,
        test_size=test_size,
        random_state=random_state
    )

    # Build vocabulary from training data
    vocab = build_vocab(X_train)
    
    # Create Dataset objects
    train_dataset = TweetDataset(X_train, y_train)
    test_dataset = TweetDataset(X_test, y_test)
    
    return train_dataset, test_dataset, vocab