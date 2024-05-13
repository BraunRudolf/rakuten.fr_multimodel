import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence 
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
import spacy
import numpy as np
from model.text_classifier import TextClassifier
from dataset.dataset import RakutenTextDataset
from dataset.preprocess import build_vocab, collate_fn

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

vocab, nlp = build_vocab(train_dataset['description'].to_list())


#TRAINING
# Training parameters
vocab_size = len(vocab)
embedding_dim = 250
hidden_dim = 128
num_classes = 27
batch_size = 50
num_epochs = 50
learning_rate = 0.001

# Create the model
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders for the training and validation sets
train_dataset = RakutenTextDataset(train_file, 'description', 'prdtypecode', vocab, nlp)
valid_dataset = RakutenTextDataset(val_file, 'description', 'prdtypecode', vocab, nlp)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Iterate over the training data for the specified number of epochs
for epoch in range(num_epochs):
    print("Start Training")
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()
        #test 
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets.view(-1)).sum().item()

        total_loss += loss.item() * len(inputs)
        total_samples += len(inputs)

    # Evaluate on the validation set after every epoch
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            total_val_loss += val_loss.item() * len(inputs)
            total_val_samples += len(inputs)

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == targets.view(-1)).sum().item()

    avg_loss = total_loss / total_samples
    # test
    avg_acc = 100* correct / total_samples
    avg_val_acc = 100* val_correct / total_samples

    avg_val_loss = total_val_loss / total_val_samples

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

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
        inputs = inputs.to(device)
        targets = targets.to(device)
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
