from mmsdk import mmdatasdk
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import numpy as np
from sklearn.model_selection import train_test_split

# Pre-trained GloVe embeddings for text
glove = GloVe(name='6B', dim=100)
vocab = glove.stoi  # String-to-index mapping

# Step 1: Load CMU-MOSI dataset and align for all three modalities
def load_and_align_multimodal_dataset(dataset_path='cmumosi/'):
    if not os.path.exists(dataset_path):
        # Load CMU-MOSI dataset if not already present
        cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, dataset_path)
        
        # Add opinion segment labels for alignment
        cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels, dataset_path)
        
        # Align with opinion segment labels
        cmumosi_highlevel.align('Opinion Segment Labels')
    else:
        print(f"Dataset already exists at {dataset_path}, skipping download and alignment.")
        cmumosi_highlevel = mmdatasdk.mmdataset(dataset_path)

    return cmumosi_highlevel

# Step 2: Preprocess text modality (similar to previous text preprocessing)
def preprocess_text(sentences, max_len=100):
    tokenizer = get_tokenizer("basic_english")
    tokenized_sentences = [tokenizer(sentence) for sentence in sentences]
    tokenized_sentences = [sentence[:max_len] for sentence in tokenized_sentences]
    indexed_sentences = [[vocab[word] if word in vocab else 0 for word in sentence] for sentence in tokenized_sentences]
    tensor_sentences = [torch.tensor(sentence) for sentence in indexed_sentences]
    padded_sentences = pad_sequence(tensor_sentences, batch_first=True, padding_value=0)
    padded_sentences = padded_sentences[:, :max_len]
    return padded_sentences

# Step 3: Preprocess audio and video modalities (COVAREP for audio, FACET for video)
def preprocess_features(features_list, max_len=100):
    """
    Preprocess the feature matrices for audio and video modalities.
    
    Args:
        features_list (list): List containing the feature sequences.
        max_len (int): Maximum number of time steps to keep (truncate or pad to this length).
    
    Returns:
        torch.Tensor: Padded tensor of shape (num_samples, max_len, num_features).
    """
    all_sequences = []
    
    # Determine the maximum number of features across all sequences
    max_features = max(features.shape[1] for features in features_list)
    
    for features in features_list:
        # Truncate or pad the features to max_len
        if len(features) > max_len:
            features = features[:max_len]
        elif len(features) < max_len:
            padding = np.zeros((max_len - len(features), features.shape[1]))  # Adjusted to use features.shape[1]
            features = np.vstack((features, padding))
        
        # Pad features to have the same number of features
        if features.shape[1] < max_features:
            padding = np.zeros((features.shape[0], max_features - features.shape[1]))
            features = np.hstack((features, padding))  # Append zeros to the right side
        
        all_sequences.append(torch.tensor(features))
    
    return torch.stack(all_sequences)

# Step 4: Prepare dataset for each modality
def prepare_multimodal_dataset(dataset, max_len=100):
    """
    Prepares the dataset for text, audio, and video modalities.
    
    Args:
        dataset: The aligned CMU-MOSI dataset.
        max_len (int): Maximum number of tokens/time steps per sequence.
        
    Returns:
        Tuple: (text_data, audio_data, video_data, labels)
    """

    print(dataset.computational_sequences.keys())
    
    # Extract the GloVe embeddings, COVAREP (audio), and FACET (video) features
    glove_vectors = dataset.computational_sequences['glove_vectors'].data
    covarep_features = dataset.computational_sequences['COVAREP'].data
    facet_features = dataset.computational_sequences['FACET_4.1'].data
    labels = dataset.computational_sequences['Opinion Segment Labels'].data

    # Extract text, audio, video, and labels for each segment
    texts, audios, videos, label_list = [], [], [], []

    for video, segments in glove_vectors.items():
        for segment_id, segment_data in segments.items():
            print(segment_data)  # Debug print to inspect segment data
            
            # Assuming segment_data is a 2D array, convert it to a string if necessary
            # Example: take mean and convert to string
            words = " ".join(map(str, np.mean(segment_data, axis=0)))  
            texts.append(words)  # Store the processed feature as a string

            # Get the audio and video features for the same segment
            audio_features = covarep_features[video][segment_id]
            video_features = facet_features[video][segment_id]
            audios.append(audio_features)
            videos.append(video_features)

            # Get the label for the current segment
            label = labels[video][segment_id]  # Assuming label might be an array
            label_list.append(label)  # Ensure label is in the right format

    # Preprocess text
    processed_texts = preprocess_text(texts, max_len=max_len)
    
    # Preprocess audio and video features
    processed_audios = preprocess_features(audios, max_len=max_len)
    processed_videos = preprocess_features(videos, max_len=max_len)

    return processed_texts, processed_audios, processed_videos

# Step 5: Load and preprocess dataset
cmumosi_dataset = load_and_align_multimodal_dataset()
text_data, audio_data, video_data = prepare_multimodal_dataset(cmumosi_dataset, max_len=100)

# Split data into training and testing sets (using the same split for all modalities)
train_text, test_text, train_audio, test_audio, train_video, test_video= train_test_split(
    text_data, audio_data, video_data, test_size=0.2, random_state=42)

# Now we have separate input streams for text, audio, and video ready for the LSTM models.

def print_sample_data(text_data, audio_data, video_data, sample_index=0):
    print("Sample Text Data (Tokenized and Padded):")
    print(text_data[sample_index])
    print("\nSample Audio Data (Padded):")
    print(audio_data[sample_index])
    print("\nSample Video Data (Padded):")
    print(video_data[sample_index])

# Print a sample from the training data
print_sample_data(train_text, train_audio, train_video, sample_index=0)
