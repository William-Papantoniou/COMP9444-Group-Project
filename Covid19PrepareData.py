from typing import Counter
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.corpus import stopwords
import nltk

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

## Data anaylisis section
def word_frequencies(data):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_counts = {'pos': Counter(), 'neg': Counter(), 'neu': Counter()}
    
    for _, row in data.iterrows():
        label = row['label']
        words = row['cleaned_tweet'].split()
        filtered_words = [word for word in words if word not in stop_words]
        word_counts[label].update(filtered_words)

    top_words = {label: count.most_common(10) for label, count in word_counts.items()}
    return top_words
    
def tweet_length_analysis(data):
    data['tweet_length'] = data['cleaned_tweet'].apply(lambda x: len(x.split()))
    avg_length = data.groupby('label')['tweet_length'].mean()
    return avg_length.to_dict()


def analyze_data(data):
    label_counts = data['label'].value_counts(normalize=True) * 100
    label_summary = {
        'Positive %': label_counts.get('pos', 0),
        'Negative %': label_counts.get('neg', 0),
        'Neutral %': label_counts.get('neu', 0),
        'Total Samples': len(data)
    }
    
    avg_tweet_length = tweet_length_analysis(data)
    word_freqs = word_frequencies(data)

    analysis_summary = {
        'Label Summary': label_summary,
        'Average Tweet Length per Sentiment': avg_tweet_length,
        'Top Words per Sentiment': word_freqs,
    }
    
    return analysis_summary

def print_analysis(analysis_summary):
    print("\n--- Data Analysis Summary ---\n")
    
    print("Label Summary:")
    for label, value in analysis_summary['Label Summary'].items():
        if isinstance(value, float):
            print(f"  {label:<15}: {value:.2f}%")
        else:
            print(f"  {label:<15}: {value}")
    
    print("\nAverage Tweet Length per Sentiment:")
    for sentiment, avg_length in analysis_summary['Average Tweet Length per Sentiment'].items():
        print(f"  {sentiment.capitalize():<10}: {avg_length:.2f} words")
    
    print("\nTop Words per Sentiment:")
    for sentiment, words in analysis_summary['Top Words per Sentiment'].items():
        print(f"  {sentiment.capitalize()}:")
        for word, count in words:
            print(f"    {word:<15} - {count} occurrences")

    print("\n--- End of Summary ---\n")

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

    # Analyze the data and print the results
    if (1 == 1):
        analysis_results = analyze_data(filtered_data)
        print_analysis(analysis_results)
    
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


train_dataset, test_dataset, vocab = preprocess_data('COVIDSenti.csv')


# Output
# --- Data Analysis Summary ---

# Label Summary:
#   Positive %     : 6.98%
#   Negative %     : 18.15%
#   Neutral %      : 74.87%
#   Total Samples  : 90000

# Average Tweet Length per Sentiment:
#   Neg       : 15.09 words
#   Neu       : 13.09 words
#   Pos       : 14.33 words

# Top Words per Sentiment:
#   Pos:
#     coronavirus     - 5353 occurrences
#     virus           - 908 occurrences
#     corona          - 849 occurrences
#     china           - 532 occurrences
#     latest          - 493 occurrences
#     cases           - 466 occurrences
#     good            - 446 occurrences
#     people          - 436 occurrences
#     many            - 420 occurrences
#     covid19         - 406 occurrences
#   Neg:
#     coronavirus     - 13796 occurrences
#     virus           - 2465 occurrences
#     corona          - 2250 occurrences
#     due             - 1686 occurrences
#     people          - 1194 occurrences
#     china           - 1113 occurrences
#     outbreak        - 857 occurrences
#     us              - 773 occurrences
#     covid19         - 762 occurrences
#     trump           - 754 occurrences
#   Neu:
#     coronavirus     - 59363 occurrences
#     virus           - 7991 occurrences
#     corona          - 7035 occurrences
#     china           - 5013 occurrences
#     covid19         - 4584 occurrences
#     new             - 4289 occurrences
#     cases           - 4180 occurrences
#     us              - 3995 occurrences
#     via             - 3825 occurrences
#     outbreak        - 3551 occurrences
