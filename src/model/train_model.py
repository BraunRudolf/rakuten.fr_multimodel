#!/usr/bin/env
import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO: Add Logging
# TODO: Fix formatting of procentage values


def train_classification_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for inputs, targets in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets.view(-1)).sum().item()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    print("Training done")
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples

    return avg_loss, avg_acc


def validate_classification_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    val_correct = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            total_val_loss += val_loss.item() * inputs.size(0)
            total_val_samples += inputs.size(0)

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == targets.view(-1)).sum().item()

    avg_val_loss = total_val_loss / total_val_samples
    avg_val_acc = val_correct / total_val_samples

    return avg_val_loss, avg_val_acc


def evaluate_classification_model(model: nn.Module, test_loader: DataLoader, device: torch.device):

    model.eval()
    total_test_samples = 0
    correct_predictions = 0

    predicted_labels_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, dim=1)

            total_test_samples += inputs.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()

            predicted_labels_list.extend(predicted_labels.cpu().tolist())
            targets_list.extend(targets.cpu().tolist())

    accuracy = correct_predictions / total_test_samples
    precision = precision_score(targets_list, predicted_labels_list, average="weighted")
    recall = recall_score(targets_list, predicted_labels_list, average="weighted")
    f1 = f1_score(targets_list, predicted_labels_list, average="weighted")

    return accuracy, precision, recall, f1


def train_eval_classification_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    early_stopping,
    num_epochs,
    device,
    params,
    log_to_mlflow,
):
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_classification_model(
            model, train_loader, optimizer, criterion, device, params["num_classes"]
        )
        val_loss, val_accuracy = validate_classification_model(
            model, val_loader, criterion, device, params["num_classes"]
        )

        lr_adjusted = scheduler.on_epoch_end(epoch, val_loss)
        early_stopping(val_loss, lr_adjusted=lr_adjusted)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)
            mlflow.log_metric("val_loss", val_loss, epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, epoch)

        if early_stopping.early_stop:
            print("Early stopping activated. Training halted.")
            break

    accuracy, precision, recall, f1 = evaluate_classification_model(model, test_loader, device)

    if log_to_mlflow:
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

    return accuracy, precision, recall, f1


def train_classification_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for inputs, targets in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        inputs = (inputs[0].to(device), inputs[1].to(device))
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets.view(-1)).sum().item()
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
    print("Training done")
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples

    return avg_loss, avg_acc


def validate_classification_fusion_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    val_correct = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluation"):
            inputs = (inputs[0].to(device), inputs[1].to(device))
            targets = targets.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            total_val_loss += val_loss.item() * targets.size(0)
            total_val_samples += targets.size(0)

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == targets.view(-1)).sum().item()

    avg_val_loss = total_val_loss / total_val_samples
    avg_val_acc = val_correct / total_val_samples

    return avg_val_loss, avg_val_acc


def evaluate_classification_fusion_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
):

    model.eval()
    total_test_samples = 0
    correct_predictions = 0

    predicted_labels_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Training"):
            inputs = (inputs[0].to(device), inputs[1].to(device))
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, dim=1)

            total_test_samples += targets.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()

            predicted_labels_list.extend(predicted_labels.cpu().tolist())
            targets_list.extend(targets.cpu().tolist())

    accuracy = correct_predictions / total_test_samples
    precision = precision_score(targets_list, predicted_labels_list, average="weighted")
    recall = recall_score(targets_list, predicted_labels_list, average="weighted")
    f1 = f1_score(targets_list, predicted_labels_list, average="weighted")

    return accuracy, precision, recall, f1


def train_eval_fusion_classification_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    early_stopping,
    num_epochs,
    device,
    params,
    log_to_mlflow,
):
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_classification_fusion_model(
            model, train_loader, optimizer, criterion, device, params["num_classes"]
        )
        val_loss, val_accuracy = validate_classification_fusion_model(
            model, val_loader, criterion, device, params["num_classes"]
        )

        lr_adjusted = scheduler.on_epoch_end(epoch, val_loss)
        early_stopping(val_loss, lr_adjusted=lr_adjusted)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)
            mlflow.log_metric("val_loss", val_loss, epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, epoch)

        if early_stopping.early_stop:
            print("Early stopping activated. Training halted.")
            break

    accuracy, precision, recall, f1 = evaluate_classification_fusion_model(
        model, test_loader, device
    )

    if log_to_mlflow:
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

    return accuracy, precision, recall, f1


# BART
def train_classification_bart_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for input_ids, attention_mask, targets in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs.logits.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == targets.view(-1)).sum().item()
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
    print("Training done")
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples

    return avg_loss, avg_acc


def validate_classification_bart_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):

    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    val_correct = 0

    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(val_loader, desc="Evaluation"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_loss = criterion(outputs.logits.view(-1, num_classes), targets.view(-1))
            total_val_loss += val_loss.item() * targets.size(0)
            total_val_samples += targets.size(0)

            predictions = torch.argmax(outputs.logits, dim=1)
            val_correct += (predictions == targets.view(-1)).sum().item()

    avg_val_loss = total_val_loss / total_val_samples
    avg_val_acc = val_correct / total_val_samples

    return avg_val_loss, avg_val_acc


def evaluate_classification_bart_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
):

    model.eval()
    total_test_samples = 0
    correct_predictions = 0

    predicted_labels_list = []
    targets_list = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(test_loader, desc="Training"):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, predicted_labels = torch.max(outputs.logits, dim=1)

            total_test_samples += targets.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()

            predicted_labels_list.extend(predicted_labels.cpu().tolist())
            targets_list.extend(targets.cpu().tolist())

    accuracy = correct_predictions / total_test_samples
    precision = precision_score(targets_list, predicted_labels_list, average="weighted")
    recall = recall_score(targets_list, predicted_labels_list, average="weighted")
    f1 = f1_score(targets_list, predicted_labels_list, average="weighted")

    return accuracy, precision, recall, f1


def train_eval_bart_classification_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    early_stopping,
    num_epochs,
    device,
    params,
    log_to_mlflow,
):
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_classification_bart_model(
            model, train_loader, optimizer, criterion, device, params["num_classes"]
        )
        val_loss, val_accuracy = validate_classification_bart_model(
            model, val_loader, criterion, device, params["num_classes"]
        )

        lr_adjusted = scheduler.on_epoch_end(epoch, val_loss)
        early_stopping(val_loss, lr_adjusted=lr_adjusted)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)
            mlflow.log_metric("val_loss", val_loss, epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, epoch)

        if early_stopping.early_stop:
            print("Early stopping activated. Training halted.")
            break

    accuracy, precision, recall, f1 = evaluate_classification_bart_model(model, test_loader, device)

    if log_to_mlflow:
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

    return accuracy, precision, recall, f1
