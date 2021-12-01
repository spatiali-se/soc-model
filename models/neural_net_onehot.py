import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
#from modules.SoilDataset import SoilDataset


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_hidden_layers = len(hidden_dim)
        self.activation = nn.LeakyReLU()

        self.out_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.50)

        self.linear_input = nn.Linear(
            in_features=input_dim, out_features=hidden_dim[0], bias=True
        )

        self.linear = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(1, self.num_hidden_layers):
            self.batchnorm.append(nn.BatchNorm1d(num_features=hidden_dim[i - 1]))
            self.linear.append(
                nn.Linear(
                    in_features=hidden_dim[i - 1], out_features=hidden_dim[i], bias=True
                )
            )

        self.linear_output = nn.Linear(
            in_features=hidden_dim[-1], out_features=output_dim, bias=False
        )

    def forward(self, x):
        x = self.linear_input(x)
        x = self.dropout(x)
        x = self.activation(x)

        for (linear, batchnorm) in zip(self.linear, self.batchnorm):
            x = batchnorm(x)
            x = linear(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.linear_output(x)
        return x  # self.out_activation(x)

'''
class TrainModel(torch.nn.Module):
    def __init__(self, model, optimizer, loss_function, device):
        super(TrainModel, self).__init__()

        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def fit(self, X, Y, num_epochs, batch_size):
        """Fit model to training data"""

        X = X.to(self.device)
        Y = Y.to(self.device)

        train_loss = []
        for epoch in range(num_epochs):

            permutation = torch.randperm(X.size()[0])
            for i in range(0, X.size()[0], batch_size):

                indices = permutation[i : i + batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                train_epoch_loss = self.train_step(batch_x, batch_y)

            if epoch % 100 == 0:
                print(f"epoch {epoch}, loss={train_epoch_loss}")

            train_loss.append(train_epoch_loss.item())

        self.model.eval()
        return train_loss

    def train_step(self, train_x, train_y):
        """Train one step"""

        self.optimizer.zero_grad()
        train_y_hat = self.model(train_x)
        train_loss = self.loss_function(train_y, train_y_hat)
        train_loss.backward()
        self.optimizer.step()

        return train_loss.detach()
'''

class TrainModel(torch.nn.Module):

    def __init__(self, model, optimizer, loss_function, device):
        super(TrainModel, self).__init__()
        """
        Args:
            model (torch.nn.Module): Neural network model.
            optimizer (torch.optim): NN optimizer, e.g. Adam.
            loss_function (torch.nn): Loss function to be minimized, 
                                      e.g. MSELoss.
            device (string): Device to train on, e.g. CPU or cuda.
        """

        self.model = model
        self.model.train() # Model in training mode

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def fit(self, train_loader, val_loader, num_epochs, val_metrics=None,
            early_stopping=True, patience=10):
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
            for i, (x_train, y_train) in enumerate(train_loader):
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                train_epoch_loss = self.train_batch(x_train, y_train)

            val_epoch_metrics = self.validation_loss(val_loader)
            val_metrics.append(val_epoch_metrics)

            progress_bar.set_postfix({'Train loss': train_epoch_loss,
                                      'Val metrics': val_epoch_metrics})

            train_loss.append(train_epoch_loss.item())

            early_stop = self.early_stopping(val_epoch_metrics[0])

            if early_stop:
                self.model.load_state_dict(self.best_model)
                print('Early Stopping!')
                break

        self.model.eval() # Model in eval mode
        return train_loss, val_metrics

    def train_batch(self, train_x, train_y):
        """Train model on one minibatch"""

        self.optimizer.zero_grad()
        train_y_pred = self.model(train_x)
        train_loss = self.loss_function(train_y, train_y_pred)
        train_loss.backward()
        self.optimizer.step()

        return train_loss.detach().item()

    def validation_loss(self, val_loader):
        "Compute validation metrics"

        self.model.eval()
        val_metrics_total = [0] * len(self.val_metrics)

        # Test validation data
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)

                val_y_hat = self.model(x_val)
                metrics = [val_metric(val_y_hat, y_val) for
                           val_metric in self.val_metrics]
                val_metrics_total = [value+metric for (value, metric)
                                     in zip(val_metrics_total, metrics)]

        self.model.train()
        return [val_metric/len(val_loader) for val_metric in val_metrics_total]

    def early_stopping(self, val_loss):
        "Early stopping"

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


if __name__ == "__main__":

    min_max_transform = False
    if min_max_transform:
        transform = MinMaxScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    data_string = "../dataset_one_hot.npy"  # one-hot encoded
    df = np.load(data_string)
    X = torch.tensor(df[:, 1:], dtype=torch.float32)
    Y = df[:, 0:1]

    non_outlier_ids = Y[:, 0] < 300
    X = X[non_outlier_ids]
    Y = Y[non_outlier_ids]

    train_ratio = 0.80
    validation_ratio = 0.0
    test_ratio = 0.20

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)

    # min-max tranform y
    if min_max_transform:
        y_train = transform.fit_transform(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    input_dim = X.shape[1]
    output_dim = 1

    # Neural net hyperparameters
    hidden_dim = [8, 8]
    batch_size = 1024
    learning_rate = 1e-2
    num_epochs = 2000
    weight_decay = 1e-5

    MLP = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(
        device
    )

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(
        params=MLP.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    trainer = TrainModel(
        model=MLP, optimizer=optimizer, loss_function=loss_function, device=device
    )

    train_loss = trainer.fit(
        x_train, y_train, num_epochs=num_epochs, batch_size=batch_size
    )

    plt.figure()
    plt.plot(train_loss)
    plt.show()

    MLP = MLP.to("cpu")

    y_pred = MLP(x_test).detach().numpy()
    if min_max_transform:
        y_pred = transform.inverse_transform(y_pred)

    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    print(f"RMSE: {RMSE:0.3f}")

    plt.figure()
    plt.plot(y_pred, y_test, ".")
    plt.plot(
        [0, np.max(np.maximum(y_test[:, 0], y_pred[:, 0]))],
        [0, np.max(np.maximum(y_test[:, 0], y_pred[:, 0]))],
        "-",
    )
    plt.xlabel("Model prediction")
    plt.ylabel("True value")
    plt.show()
