import torch
from tqdm import tqdm
import copy
import numpy as np


class Trainer(torch.nn.Module):
    def __init__(self, model, optimizer, loss_function, device):
        super(Trainer, self).__init__()
        """
        Args:
            model (torch.nn.Module): Neural network model.
            optimizer (torch.optim): NN optimizer, e.g. Adam.
            loss_function (torch.nn): Loss function to be minimized, 
                                      e.g. MSELoss.
            device (string): Device to train on, e.g. CPU or cuda.
        """

        self.model = model
        self.model.train()  # Model in training mode

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        val_metrics=None,
        early_stopping=True,
        patience=10,
    ):
        """Fit model to training data

        Args:
            train_loader (torch.utils.data.dataloader): Training data loader.
            val_loader (torch.utils.data.dataloader): Validation data loader.
            num_epochs (int): Number of epochs to train.
            val_metrics (list with metrics): list of metrics to record on
                                             validation data, e.g. MSELoss.
                                             The first metric is used for
                                             early stopping if early stopping
                                             is on.
            early_stopping (bool): Determines if early stopping is utilized.
            patience (int): Number of accepted non improvements before early
                            stopping
        """

        if val_metrics is not None:
            self.val_metrics = val_metrics
        else:
            self.val_metrics = [self.loss_function]

        if early_stopping:
            self.best_loss = 1e8
            self.num_no_improvements = 0
            self.patience = patience

        train_loss = []
        val_metrics = []
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            for _, train_data in enumerate(train_loader):
                x_train = train_data["features"].to(self.device)
                y_train = train_data["target"].to(self.device)

                train_epoch_loss = self.train_batch(x_train, y_train)

            val_epoch_metrics = self.validation_loss(val_loader)
            val_metrics.append(val_epoch_metrics)

            progress_bar.set_postfix(
                {"Train loss": train_epoch_loss, "Val metrics": val_epoch_metrics}
            )

            train_loss.append(train_epoch_loss)

            early_stop = self.early_stopping(val_epoch_metrics[0])

            if early_stop:
                self.model.load_state_dict(self.best_model)
                print("Early Stopping!")
                break

        self.model.eval()  # Model in eval mode
        return np.asarray(train_loss), np.asarray(val_metrics)

    def train_batch(self, train_x, train_y):
        """Train model on one minibatch"""

        self.optimizer.zero_grad()
        train_y_pred = self.model(train_x)
        train_loss = self.loss_function(train_y, train_y_pred)
        train_loss.backward()
        self.optimizer.step()

        return train_loss.detach().item()

    def validation_loss(self, val_loader):
        """Compute validation metrics"""

        self.model.eval()
        val_metrics_total = [0] * len(self.val_metrics)

        # Test validation data
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                x_val = val_data["features"].to(self.device)
                y_val = val_data["target"].to(self.device)

                val_y_hat = self.model(x_val)
                metrics = [
                    val_metric(val_y_hat, y_val) for val_metric in self.val_metrics
                ]
                val_metrics_total = [
                    value + metric
                    for (value, metric) in zip(val_metrics_total, metrics)
                ]

        self.model.train()
        return [
            (val_metric / len(val_loader)).item() for val_metric in val_metrics_total
        ]

    def early_stopping(self, val_loss):
        """Early stopping"""

        if val_loss < self.best_loss:
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.best_loss = val_loss
            self.num_no_improvements = 0
        else:
            self.num_no_improvements += 1

        if self.num_no_improvements > self.patience:
            return True
        else:
            return False
