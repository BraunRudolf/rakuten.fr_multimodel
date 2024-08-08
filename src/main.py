#!/usr/bin/env python
import os

import dotenv
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.callbacks.callbacks import EarlyStopping, LearningRateScheduler
from src.dataset.data_loader import (
    create_dataloaders,
    create_fusion_datasets,
    create_image_datasets,
    create_text_datasets,
)
from src.dataset.preprocess import (
    build_vocab,
    fusion_collate_fn,
    load_vocab_and_nlp,
    retrieve_indices,
    retrieve_vocab_dataset,
    text_collate_fn,
    train_val_test_indices,
)
from src.model.fusion_model import FusionModel
from src.model.image_classifier import ImageClassifier
from src.model.text_classifier import HeadlessTextClassifier, TextClassifier
from src.model.train_model import (
    train_eval_classification_loop,
    train_eval_fusion_classification_loop,
)

dotenv.load_dotenv()

# TRAINING
# Training parameters
vocab_size = 500
embedding_dim = 250
hidden_dim = 128
num_classes = 27
batch_size = 32
num_epochs = 20
learning_rate = 0.001

MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

mlflow.set_tracking_uri(MLFLOW_SERVER_URI)  # type: ignore
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Dataloader settings
# 8 workers enough, bottleneck on RTX 4090
NUM_OF_WORKERS = 16
PIN_MEMORY = False

DB_URL = os.getenv("DB_SERVER_URL")  # type: ignore # DB_URL
ID_COLUMN = os.getenv("ID_COLUMN")
IMAGE_COLUMN = os.getenv("IMAGE_COLUMN")
TABLE_NAME = os.getenv("TABLE_NAME")
MAPPING_TABLE_NAME = os.getenv("MAPPING_TABLE_NAME")
TEXT_COLUMN = os.getenv("TEXT_COLUMN")
LABLE_COLUMN = os.getenv("LABLE_COLUMN")
MAPPING_COLUMN = os.getenv("MAPPING_COLUMN")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")

indices = retrieve_indices(DB_URL, TABLE_NAME, ID_COLUMN)  # type: ignore
train_indices, val_indices, test_indices = train_val_test_indices(indices, 0.8, 0.1, 0.1)
train_indices.sort(reverse=True)
val_indices.sort(reverse=True)
test_indices.sort(reverse=True)

# TODO: remove from main to seperate script that run periodically with full dataset?
# create/load vocab
vocab_path = os.getenv("VOCAB_PATH")
# HACK: change to more precise check
if len(os.listdir(vocab_path)) == 0:
    print(len(os.listdir(vocab_path)))
    # create new tokenizer + vocab
    vocab, nlp = build_vocab(
        retrieve_vocab_dataset(DB_URL, TABLE_NAME, "id", TEXT_COLUMN, train_indices),  # type: ignore
        save_dir=vocab_path,
    )
else:
    vocab, nlp = load_vocab_and_nlp(vocab_path)

vocab_size = len(vocab)

train_text_dataset, val_text_dataset, test_text_dataset = create_text_datasets(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    text_column=TEXT_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    vocab=vocab,
    spacy_model=nlp,
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
    collate_fn=text_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
text_model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

text_optimizer = optim.Adam(text_model.parameters(), lr=0.001)
text_criterion = nn.CrossEntropyLoss()

# Define Callbacks
text_scheduler = LearningRateScheduler(text_optimizer)
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

# with mlflow.start_run():
#     # Log parameters to MLflow
#     for key, value in params.items():
#         mlflow.log_param(key, value)
#     train_eval_classification_loop(
#         model=text_model,
#         train_loader=train_text_loader,
#         val_loader=val_text_loader,
#         test_loader=test_text_loader,
#         optimizer=text_optimizer,
#         criterion=text_criterion,
#         scheduler=text_scheduler,
#         early_stopping=early_stopping,
#         num_epochs=num_epochs,
#         device=device,
#         params=params,
#         log_to_mlflow=True,
#     )

### IMAGE CLASSIFICATION ###
transform = transforms.Compose([transforms.Resize((224, 224))])

train_image_dataset, val_image_dataset, test_image_dataset = create_image_datasets(
    db_url=DB_URL,
    table_name=TABLE_NAME,
    mapping_table_name=MAPPING_TABLE_NAME,
    id_column=ID_COLUMN,
    image_column=IMAGE_COLUMN,
    label_column=LABLE_COLUMN,
    mapping_column=MAPPING_COLUMN,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices,
    image_folder=IMAGE_FOLDER,
    transform=transform,
)
train_image_loader, val_image_loader, test_image_loader = create_dataloaders(
    train_dataset=train_image_dataset,
    val_dataset=val_image_dataset,
    test_dataset=test_image_dataset,
    shuffle=False,
    batch_size=batch_size,  # WARNING: HARDCODED
    num_workers=NUM_OF_WORKERS,
    pin_memory=PIN_MEMORY,
)
image_params = {
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
}
# Create the model
image_model = ImageClassifier().to(device)

image_optimizer = optim.Adam(image_model.parameters(), lr=0.001)
image_criterion = nn.CrossEntropyLoss()
# Define Callback
image_scheduler = LearningRateScheduler(image_optimizer)
early_stopping = EarlyStopping(verbose=True)
# with mlflow.start_run():
#     # Log parameters to MLflow
#     for key, value in params.items():
#         mlflow.log_param(key, value)
#     accuracy, precision, recall, f1 = train_eval_classification_loop(
#         model=image_model,
#         train_loader=train_image_loader,
#         val_loader=val_image_loader,
#         test_loader=test_image_loader,
#         optimizer=image_optimizer,
#         criterion=image_criterion,
#         scheduler=image_scheduler,
#         early_stopping=early_stopping,
#         num_epochs=num_epochs,
#         device=device,
#         params=params,
#         log_to_mlflow=True,
#     )

torch.save(text_model.state_dict(), "text_model.pth")
torch.save(image_model.state_dict(), "image_model.pth")

# Fusion Model
text_model.load_state_dict(torch.load("text_model.pth"))
image_model.load_state_dict(torch.load("image_model.pth"))

headless_text_model = HeadlessTextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
headless_text_model.load_state_dict(text_model.state_dict(), strict=False)
cut_image_model = nn.Sequential(*list(image_model.children())[:-3])
# headless_text_model.eval()
# cut_image_model.eval()

train_fusion_dataset, val_fusion_dataset, test_fusion_dataset = create_fusion_datasets(
    train_text_dataset=train_text_dataset,
    val_text_dataset=val_text_dataset,
    test_text_dataset=test_text_dataset,
    train_image_dataset=train_image_dataset,
    val_image_dataset=val_image_dataset,
    test_image_dataset=test_image_dataset,
)
train_fusion_loader, val_fusion_loader, test_fusion_loader = create_dataloaders(
    train_fusion_dataset,
    val_fusion_dataset,
    test_fusion_dataset,
    batch_size,
    NUM_OF_WORKERS,
    PIN_MEMORY,
    collate_fn=fusion_collate_fn,
    shuffle=True,
)

image_params = {
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
}


# # Create the model
fusion_model = FusionModel(headless_text_model, cut_image_model, num_classes).to(device)


fusion_optimizer = optim.Adam(image_model.parameters(), lr=0.001)
fusion_criterion = nn.CrossEntropyLoss()
# Define Callback
fusion_scheduler = LearningRateScheduler(fusion_optimizer)
early_stopping = EarlyStopping(verbose=True)


fusion_model.train()
total_loss = 0.0
total_samples = 0
correct = 0

with mlflow.start_run():
    # Log parameters to MLflow
    for key, value in params.items():
        mlflow.log_param(key, value)
    accuracy, precision, recall, f1 = train_eval_fusion_classification_loop(
        model=fusion_model,
        train_loader=train_fusion_loader,
        val_loader=val_fusion_loader,
        test_loader=test_fusion_loader,
        optimizer=fusion_optimizer,
        criterion=fusion_criterion,
        scheduler=fusion_scheduler,
        early_stopping=early_stopping,
        num_epochs=15,
        device=device,
        params=params,
        log_to_mlflow=True,
    )
