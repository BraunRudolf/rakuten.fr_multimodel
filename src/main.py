#!/usr/bin/env python
import os

import dotenv
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import save
from sqlalchemy import label
from torch.utils.data import DataLoader

from src.callbacks.callbacks import EarlyStopping, LearningRateScheduler
from src.dataset.dataset import RakutenTextDataset
from src.dataset.preprocess import (
    build_vocab,
    collate_fn,
    load_vocab_and_nlp,
    retrieve_indices,
    retrieve_vocab_dataset,
    train_val_test_indices,
)
from src.features.build_features import DataImporter, ImagePreprocessor, TextPreprocessor
from src.model.text_classifier import TextClassifier
from src.model.train_model import evaluate_text_model, train_text_model, validate_text_model

dotenv.load_dotenv()


# # TODO: Create function that creates splits based on pct
# data_importer = DataImporter(
#     target_col=["designation", "description"], label_col="label"
# )
# df = data_importer.load_data()
# X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)
# # TODO: create separate preprocessing functions and make usable by dataloader
# # Preprocess text and images
# text_preprocessor = TextPreprocessor()
# image_preprocessor = ImagePreprocessor()
#
#
# X_train = text_preprocessor.preprocess_text_in_df(X_train)
# X_val = text_preprocessor.preprocess_text_in_df(X_val)
# X_test = text_preprocessor.preprocess_text_in_df(X_test)
#
# # HACK: test as '' strings which load as NaN. check why and resolve
# X_test = X_test.replace("", " ")

# Write Preprocessed Files
# train_dataset = pd.concat([X_train, y_train], axis=1)
# test_dataset = pd.concat([X_test, y_test], axis=1)
# val_dataset = pd.concat([X_val, y_val], axis=1)
# train_dataset.to_csv("./data/preprocessed/train_dataset.csv")
# test_dataset.to_csv("./data/preprocessed/test_dataset.csv")
# val_dataset.to_csv("./data/preprocessed/val_dataset.csv")

# test = pd.read_csv("./data/preprocessed/test_dataset.csv")
#
# train_file = "./data/preprocessed/train_dataset.csv"
# test_file = "./data/preprocessed/test_dataset.csv"
# val_file = "./data/preprocessed/val_dataset.csv"
#
# assert val_dataset["target"].isna().sum() == 0, "none value"
#


# TRAINING
# Training parameters
vocab_size = 500
embedding_dim = 250
hidden_dim = 128
num_classes = 27
batch_size = 32
num_epochs = 200
learning_rate = 0.001
print(f"vocab size: {vocab_size}")
MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

mlflow.set_tracking_uri(MLFLOW_SERVER_URI)  # type: ignore
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Dataloader settings
# 8 workers enough, bottleneck on RTX 4090
NUM_OF_WORKERS = 8
PIN_MEMORY = False

DB_URL = "postgresql://admin:admin@192.168.2.105:5432/local"
TABLE_NAME = "rakuten_products"
MAPPING_TABLE_NAME = "prdtypecode_label_mapping"
TEXT_COLUMN = "designation"
LABLE_COLUMN = "label"
MAPPING_COLUMN = "prdtypecode"

indices = retrieve_indices(DB_URL, TABLE_NAME, "id")
train_indices, val_indices, test_indices = train_val_test_indices(indices, 0.8, 0.1, 0.1)
num_of_batches = len(train_indices) // batch_size

# TODO: adjust for db! remove from main?
# create/load vocab
vocab_path = os.getenv("VOCAB_PATH")
# HACK change to more precise check
if len(os.listdir(vocab_path)) == 0:
    print(len(os.listdir(vocab_path)))
    # create new tokenizer + vocab
    vocab, nlp = build_vocab(
        retrieve_vocab_dataset(DB_URL, TABLE_NAME, "id", TEXT_COLUMN, train_indices),
        save_dir=vocab_path,
    )
else:
    vocab, nlp = load_vocab_and_nlp(vocab_path)

vocab_size = len(vocab)

# Create data loaders for the training and validation sets
# train_dataset = RakutenTextDataset(train_file, "target", "label", vocab, nlp)
# valid_dataset = RakutenTextDataset(val_file, "target", "label", vocab, nlp)
# test_dataset = RakutenTextDataset(test_file, "target", "label", vocab, nlp)
train_dataset = RakutenTextDataset(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    text_column=TEXT_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    vocab=vocab,
    spacy_model=nlp,
    indices=train_indices,
)

valid_dataset = RakutenTextDataset(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    text_column=TEXT_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    vocab=vocab,
    spacy_model=nlp,
    indices=val_indices,
)

test_dataset = RakutenTextDataset(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    text_column=TEXT_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    vocab=vocab,
    spacy_model=nlp,
    indices=test_indices,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_OF_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_OF_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_OF_WORKERS,
    pin_memory=PIN_MEMORY,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Define Callbacks
scheduler = LearningRateScheduler(optimizer)
early_stopping = EarlyStopping(verbose=True)

params = {
    "vocab_size": len(vocab),
    "embedding_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
}

with mlflow.start_run():
    # Log parameters to MLflow
    for key, value in params.items():
        mlflow.log_param(key, value)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_text_model(
            model, train_loader, optimizer, criterion, device, params["num_classes"]
        )
        val_loss, val_accuracy = validate_text_model(
            model, val_loader, criterion, device, params["num_classes"]
        )

        lr_adjusted = scheduler.on_epoch_end(epoch, val_loss)
        early_stopping(val_loss, lr_adjusted=lr_adjusted)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, epoch)
        mlflow.log_metric("val_loss", val_loss, epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, epoch)

        # Log model checkpoint to MLflow
        # mlflow.pytorch.log_model(model, "models")

        # Adjust learning rate based on epoch loss
        # Check if validation loss has improved, if not, trigger early stopping
        if early_stopping.early_stop:
            print("Early stopping activated. Training halted.")
            break

    # Make predictions on the test data
    accuracy, precision, recall, f1 = evaluate_text_model(model, test_loader, device)

    # Log testing metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
