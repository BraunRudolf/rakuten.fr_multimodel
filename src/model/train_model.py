import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

def train_text_model(model: nn.Module,
                     train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module,
                     device: torch.device,
                     num_classes: int):
    
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

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets.view(-1)).sum().item()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples

    return avg_loss, avg_acc

def validate_text_model(model: nn.Module,
                         val_loader: DataLoader,
                         criterion: nn.Module,
                         device: torch.device,
                         num_classes: int):
    
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

def evaluate_text_model(model: nn.Module,
                        test_loader: DataLoader,
                        device: torch.device):
    
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
    precision = precision_score(targets_list, predicted_labels_list, average='weighted')
    recall = recall_score(targets_list, predicted_labels_list, average='weighted')
    f1 = f1_score(targets_list, predicted_labels_list, average='weighted')

    return accuracy, precision, recall, f1

