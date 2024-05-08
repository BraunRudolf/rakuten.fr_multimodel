import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence 
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
import spacy
from collections import Counter

data_importer = DataImporter()
df = data_importer.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)

# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor()
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_test, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)
image_preprocessor.preprocess_images_in_df(X_test)

# Write Preprocessed Files
train_dataset = pd.concat([ X_train, y_train ], axis=1)
test_dataset = pd.concat([ X_test, y_test ], axis=1)
val_dataset = pd.concat([ X_val, y_val ], axis=1)
train_dataset.to_csv('./data/preprocessed/train_dataset.csv')
test_dataset.to_csv('./data/preprocessed/test_dataset.csv')
val_dataset.to_csv('./data/preprocessed/val_dataset.csv')

train_file = "./data/preprocessed/train_dataset.csv"
test_file = "./data/preprocessed/test_dataset.csv"
val_file = "./data/preprocessed/val_dataset.csv"
class TextIterator:
    def __init__(self, file_path, text_column, chunk_size=100):
        self.file_path = file_path
        self.text_column = text_column
        self.chunk_size = chunk_size
    
    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, usecols=[self.text_column], chunksize=self.chunk_size):
            for text in chunk[self.text_column]:
                yield text
# Tokenize the text data
# tokenizer = get_tokenizer('frensh')
# tokenizer = spacy.load("fr_core_news_sm") #TODO: change to nlp to follow standard 

def build_vocab(data, spacy_model="fr_core_news_sm"):
    counter = Counter()
    nlp = spacy.load(spacy_model)
    for text in data:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        counter.update(tokens)
    vocab = {token: index for index, (token, _) in enumerate(counter.most_common())}
    return vocab, nlp
vocab, nlp = build_vocab(train_dataset['description'].to_list())

# text_iterator = TextIterator(train_file, 'description')
# for chunk in text_iterator:
#     doc = tokenizer(chunk)
#     for token in doc:
#         # Check if token has a vector before setting it
#         if token.text in vocab:
#             index = vocab[token.text].orth
            # print("Token:", token.text, "Index:", index)
# Build the vocabulary
# def build_vocab(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         tokens = tokenizer(f.read())
#     return tokens

# train_tokens = build_vocab(train_file)
# vocab = build_vocab_from_iterator(train_tokens)


class RakutenTextDataset(Dataset):
    def __init__(self, file_path, text_column, label_column, vocab, spacy_model):
        self.vocab = vocab
        self.text_data, self.labels = self.load_data(file_path, text_column, label_column)
        self.tokenizer =  spacy_model.tokenizer

    def load_data(self, file_path, text_column, label_column):
        # Load the CSV file
        df = pd.read_csv(file_path)
        text_data = df[text_column].tolist()
        labels = df[label_column].tolist()
        return text_data, labels
    
    def text_to_tensor(self, text):
        tokens = self.tokenizer(text)
        tensor = [self.vocab[token.text] for token in tokens if token.text in self.vocab]
        return torch.tensor(tensor, dtype=torch.long) 

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, index):
        text = self.text_data[index]
        label = self.labels[index]
        text_tensor = self.text_to_tensor(text)
        return text_tensor, label

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits
def collate_fn(batch):
    # Separate inputs and targets
    inputs, targets = zip(*batch)

    # Pad sequences to the maximum length in the batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return padded_inputs, targets_tensor

#TRAINING
# Training parameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
num_classes = 27
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create the model
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders for the training and validation sets
train_dataset = RakutenTextDataset(train_file, 'description', 'prdtypecode', vocab, nlp)#, tokenizer)
valid_dataset = RakutenTextDataset(val_file, 'description', 'prdtypecode', vocab, nlp)# tokenizer.vocab, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Iterate over the training data for the specified number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs
        targets = targets
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(inputs)
        total_samples += len(inputs)

    # Evaluate on the validation set after every epoch
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs
            targets = targets
            outputs = model(inputs)
            val_loss = criterion(outputs.view(-1, num_classes), targets.view(-1))

            total_val_loss += val_loss.item() * len(inputs)
            total_val_samples += len(inputs)

    avg_loss = total_loss / total_samples
    avg_val_loss = total_val_loss / total_val_samples

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

### EVALUATION
# Load the test set or a separate evaluation set
test_dataset = RakutenTextDataset(test_file, 'description', 'prdtypecode', vocab, nlp)# tokenizer.vocab, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Use the trained model to make predictions on the test set
model.eval()
with torch.no_grad():
    total_test_samples = 0
    correct_predictions = 0

    predicted_labels_list = []
    targets_list = []

    for inputs, targets in test_loader:
        iinputsnputs = inputs
        targets = targets
        outputs = model(inputs)

        _, predicted_labels = torch.max(outputs, dim=1)

        total_test_samples += len(inputs)
        correct_predictions += (predicted_labels == targets).sum().item()

        predicted_labels_list.extend(predicted_labels.tolist())
        targets_list.extend(targets.tolist())

    # Calculate evaluation metrics
    accuracy = correct_predictions / total_test_samples
    precision = precision_score(targets_list, predicted_labels_list, average='weighted')
    recall = recall_score(targets_list, predicted_labels_list, average='weighted')
    f1 = f1_score(targets_list, predicted_labels_list, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
