from src.callbacks.callbacks import EarlyStopping, LearningRateScheduler
import torch
import pytest

# Test LearningRateScheduler
@pytest.fixture
def optimizer():
    return torch.optim.SGD([torch.tensor(1.0)], lr=0.001)

def test_learning_rate_scheduler(optimizer):
    lr_scheduler = LearningRateScheduler(optimizer)
    assert lr_scheduler.initial_lr == 0.001
    assert lr_scheduler.factor == 0.001
    assert lr_scheduler.patience == 3

def test_learning_rate_scheduler_on_epoch_end(optimizer):
    lr_scheduler = LearningRateScheduler(optimizer)
    # Assuming some epoch and loss values for testing
    epoch = 5
    loss = 0.1
    old_lr = lr_scheduler.lr_scheduler.get_last_lr()
    lr_adjusted = lr_scheduler.on_epoch_end(epoch, loss)
    new_lr = lr_scheduler.lr_scheduler.get_last_lr()
    # Assert that the learning rate has been updated
    assert old_lr == new_lr
    assert lr_adjusted == False

    # Testing with different loss to ensure the learning rate is not updated
    old_lr = lr_scheduler.lr_scheduler.get_last_lr()
    lr_adjusted = lr_scheduler.on_epoch_end(epoch, 0.3)
    new_lr = lr_scheduler.lr_scheduler.get_last_lr()
    # Assert that the learning rate remains the same
    assert old_lr == new_lr
    assert lr_adjusted == False

    # patience 1
    old_lr = lr_scheduler.lr_scheduler.get_last_lr()
    lr_adjusted = lr_scheduler.on_epoch_end(epoch, 0.3)
    new_lr = lr_scheduler.lr_scheduler.get_last_lr()
    # Assert that the learning rate remains the same
    assert old_lr == new_lr
    assert lr_adjusted == False

    # Patience 2
    old_lr = lr_scheduler.lr_scheduler.get_last_lr()
    lr_adjusted = lr_scheduler.on_epoch_end(epoch, 0.3)
    new_lr = lr_scheduler.lr_scheduler.get_last_lr()
    # Assert that the learning rate remains the same
    assert old_lr == new_lr
    assert lr_adjusted == False

    # Adjustedment of learning rate
    old_lr = lr_scheduler.lr_scheduler.get_last_lr()
    lr_adjusted = lr_scheduler.on_epoch_end(epoch, 0.3)
    new_lr = lr_scheduler.lr_scheduler.get_last_lr()
    # Assert that the learning rate remains the same
    assert old_lr != new_lr
    assert lr_adjusted == True


# Testing EarlyStopping
@pytest.fixture
def early_stopping_instance():
    return EarlyStopping()

def test_early_stopping_with_lr_adjustment(early_stopping_instance):
    # Simulate a training process with validation losses
    val_losses = [0.5, 0.4, 0.35, 0.36, 0.36, 0.36, 0.37, 0.38, 0.36, 0.36, 0.36, 0.36]
    lr_rate = [False, False, False, False, False, False, True, False, False, False, False, False] 

    for val_loss, lr_rate_adjusted in zip(val_losses, lr_rate):
        early_stopping_instance(val_loss, lr_rate_adjusted)
        if early_stopping_instance.early_stop:
            break

    # Assert that no early stopping was triggered
    assert early_stopping_instance.early_stop == True
    assert early_stopping_instance.counter == 5
    assert early_stopping_instance.epochs_since_lr_adjust == 5

def test_no_early_stopping_with_lr_adjustment_no_same_val_loss(early_stopping_instance):
    # Simulate a training process with validation losses
    val_losses = [0.502, 0.401, 0.354, 0.341, 0.339, 0.327, 0.318, 0.309, 0.295, 0.288, 0.276, 0.262]
    lr_rate = [False, False, False, False, False, False, False, False, False, False, False, False] 

    for val_loss, lr_rate_adjusted in zip(val_losses, lr_rate):
        early_stopping_instance(val_loss, lr_rate_adjusted)
        if early_stopping_instance.early_stop:
            break

    # Assert that no early stopping was triggered
    assert early_stopping_instance.early_stop == False
    assert early_stopping_instance.counter == 0
    assert early_stopping_instance.epochs_since_lr_adjust == 0
    
def test_no_early_stopping_with_same_val_losses(early_stopping_instance):
    # Simulate a training process with validation losses
    val_losses = [0.502, 0.401, 0.354, 0.341, 0.341, 0.327, 0.327, 0.309, 0.309, 0.295, 0.288, 0.288]
    lr_rate = [False, False, False, False, False, False, False, False, False, False, False, False] 

    for val_loss, lr_rate_adjusted in zip(val_losses, lr_rate):
        early_stopping_instance(val_loss, lr_rate_adjusted)
        if early_stopping_instance.early_stop:
            break

    # Assert that no early stopping was triggered
    assert early_stopping_instance.early_stop == False
    assert early_stopping_instance.counter == 0
    assert early_stopping_instance.epochs_since_lr_adjust == 0

def test_no_early_stopping_finish_before_cooldown(early_stopping_instance):
    # Simulate a training process with validation losses
    val_losses = [0.502, 0.401, 0.346, 0.361, 0.361, 0.367, 0.357, 0.359, 0.357]
    lr_rate = [False, False, False, False, False, True, False, False, False] 

    for val_loss, lr_rate_adjusted in zip(val_losses, lr_rate):
        early_stopping_instance(val_loss, lr_rate_adjusted)
        if early_stopping_instance.early_stop:
            break

    # Assert that no early stopping was triggered
    assert early_stopping_instance.early_stop == False
    assert early_stopping_instance.counter == 3
    assert early_stopping_instance.epochs_since_lr_adjust == 3

