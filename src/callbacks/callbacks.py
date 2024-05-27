import torch

class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr=0.001, factor=0.001, patience=3, cooldown=3):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=factor,
            patience=patience,
            cooldown=cooldown
        )

    def on_epoch_end(self, epoch, loss, **kwargs):
        old_lr = self.lr_scheduler.get_last_lr() # Get the current learning rate
        self.lr_scheduler.step(loss)
        new_lr = self.lr_scheduler.get_last_lr() # Get the new learning rate
        if old_lr != new_lr:
            print(f"Learning rate adjusted from {old_lr} to {new_lr}.") 
            return True
        return False

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, cooldown_epochs=5):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.cooldown_epochs = cooldown_epochs
        self.epochs_since_lr_adjust = 0

    def __call__(self, current_metric, lr_adjusted=False):
        if self.best_metric is None:
            self.best_metric = current_metric
        elif ( self.best_metric - current_metric ) < -0.01:
            self.counter += 1
            self.epochs_since_lr_adjust += 1
            if self.verbose:
                print(f'Patience Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience and self.epochs_since_lr_adjust >= self.cooldown_epochs:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping activated.")
        else:
            self.best_metric = current_metric
            self.counter = 0
            self.epochs_since_lr_adjust = 0
        
        if lr_adjusted:
            self.epochs_since_lr_adjust = 0
            self.counter = 0

