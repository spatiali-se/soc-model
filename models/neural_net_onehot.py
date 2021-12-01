import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from modules.SoilDataset import SoilDataset


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


if __name__ == "__main__":

    min_max_transform = False
    if min_max_transform:
        transform = MinMaxScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    data_string = "dataset_one_hot.npy"  # one-hot encoded
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
