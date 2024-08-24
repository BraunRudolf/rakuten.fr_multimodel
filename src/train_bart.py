#!/usr/bin/env python
import os

import dotenv
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForSequenceClassification,
    BarthezTokenizer,
)

from src.callbacks.callbacks import EarlyStopping, LearningRateScheduler
from src.dataset.data_loader import create_dataloaders, create_text_transformer_datasets
from src.dataset.preprocess import (
    remove_html,
    remove_white_space,
    retrieve_indices,
    to_lower,
    train_val_test_indices,
)
from src.model.train_model import train_eval_bart_classification_loop

dotenv.load_dotenv()

# TRAINING
# Training parameters
vocab_size = 500
embedding_dim = 250
hidden_dim = 128
num_classes = 27
batch_size = 16
num_epochs = 20
learning_rate = 0.001

MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

mlflow.set_tracking_uri(MLFLOW_SERVER_URI)  # type: ignore
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Dataloader settings
# 8 workers enough, bottleneck on RTX 4090
NUM_OF_WORKERS = int(os.getenv("NUM_OF_WORKERS"))  # type: ignore # NUM_OF_WORKERS
PIN_MEMORY = os.getenv("PIN_MEMORY")  # type: ignore # PIN_MEMORY

DB_URL = os.getenv("DB_SERVER_URL")  # type: ignore # DB_URL
ID_COLUMN = os.getenv("ID_COLUMN")
IMAGE_COLUMN = os.getenv("IMAGE_COLUMN")
TABLE_NAME = os.getenv("TABLE_NAME")
MAPPING_TABLE_NAME = os.getenv("MAPPING_TABLE_NAME")
TEXT_COLUMN = os.getenv("TEXT_COLUMN")
LABLE_COLUMN = os.getenv("LABLE_COLUMN")
MAPPING_COLUMN = os.getenv("MAPPING_COLUMN")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")
VOCAB_BASE_PATH = os.getenv("VOCAB_PATH")

indices = retrieve_indices(DB_URL, TABLE_NAME, ID_COLUMN)  # type: ignore
train_indices, val_indices, test_indices = train_val_test_indices(indices, 0.8, 0.1, 0.1)
train_indices.sort(reverse=True)
val_indices.sort(reverse=True)
test_indices.sort(reverse=True)
preprocessing_pipeline = [remove_html, remove_white_space, to_lower]
text = True
image = False
fusion = False

model_name = "moussaKam/barthez"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_text_dataset, val_text_dataset, test_text_dataset = create_text_transformer_datasets(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    text_column=TEXT_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    tokenizer=tokenizer,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices,
)
train_text_loader, val_text_loader, test_text_loader = create_dataloaders(
    train_dataset=train_text_dataset,
    val_dataset=val_text_dataset,
    test_dataset=test_text_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=NUM_OF_WORKERS,
    pin_memory=PIN_MEMORY,
    # collate_fn=text_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(
    device
)

text_optimizer = optim.Adam(model.parameters(), lr=0.001)
text_criterion = nn.CrossEntropyLoss()

# Define Callbacks
text_scheduler = LearningRateScheduler(text_optimizer)
early_stopping = EarlyStopping(verbose=True)

params = {
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
}

with mlflow.start_run():
    # Log parameters to MLflow
    for key, value in params.items():
        mlflow.log_param(key, value)
    train_eval_bart_classification_loop(
        model=model,
        train_loader=train_text_loader,
        val_loader=val_text_loader,
        test_loader=test_text_loader,
        optimizer=text_optimizer,
        criterion=text_criterion,
        scheduler=text_scheduler,
        early_stopping=early_stopping,
        num_epochs=num_epochs,
        device=device,
        params=params,
        log_to_mlflow=True,
    )
