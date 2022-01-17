import pdb

import torch.nn as nn
import torch
import models.dense_neural_net as models
from modules.training import Trainer


class EnsembleNN(torch.nn.Module):
    """Dense Neural Network"""
    def __init__(self, num_models, input_dim, hidden_dims, output_dim,
                 activation=nn.LeakyReLU(), dropout_rate=0.5):
        """Initialize dense neural net

        args:
            num_models (int): Number of models in ensemble.
            input_dim (int): Input dimension.
            hidden_dims (list of ints): List of dimension of each hidden layer.
            output_dim (int): Output dimension
            activation (torch.nn.activation): Activation function.
            dropout_rate (float): Dropout rate.
        """
        super(EnsembleNN, self).__init__()

        self.num_models = num_models
        self.output_dim = output_dim


        self.ensemble = nn.ModuleList([models.DenseNN(input_dim=input_dim,
                                                      hidden_dims=hidden_dims,
                                                      output_dim=output_dim,
                                                      activation=activation,
                                                      dropout_rate=dropout_rate)
                                       for i in range(self.num_models)])

    def forward(self, x):
        """Forward propagation"""

        # Preallocate prediction tensor
        pred = torch.zeros([x.shape[0], self.output_dim, self.num_models])

        # Compute prediction mean and variance for each model
        for i, model in enumerate(self.ensemble):
            pred[:, :, i] = model(x)

        # Compute prediction mean
        pred_mean = pred.mean(dim=1)

        return pred_mean

    def compile(self, optimizers, loss_function, metrics, device):
        """Compile models and set up trainers

        args:
            optimizers (list of optimizers): List of optimizers. Optimizer i is
                                             used to train model i.
            loss_function (callable): Loss function used to train the models
            device (string): Device to train on, e.g. CPU or cuda.
        """

        self.device = device
        self.optimizers = optimizers
        self.loss_function = loss_function
        self.metrics = metrics

        for model in self.ensemble:
            model = model.to(self.device)

        self.trainers = [Trainer(model=model,
                                  optimizer=optimizer,
                                  loss_function=loss_function,
                                  device=device) for (model, optimizer) in
                                  zip(self.ensemble, self.optimizers)]



    def fit(self, train_loader, val_loader,
            num_epochs=100, patience=20, early_stopping=True):
        """Fit ensemble model to training data.

        args:
            train_loader (torch.utils.data.dataloader): Training data loader.
            val_loader (torch.utils.data.dataloader): Validation data loader.
            num_epochs (int): Number of epochs to train.
            early_stopping (bool): Determines if early stopping is utilized.
            patience (int): Number of accepted non improvements before early
                            stopping
        """

        train_loss_list = []
        val_metric_list = []
        for trainer in self.trainers:
            train_loss, val_metric = trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    val_metrics=self.metrics,
                    num_epochs=num_epochs,
                    patience=patience,
                    early_stopping=early_stopping,
            )
            train_loss_list.append(train_loss)
            val_metric_list.append(val_metric)

        return train_loss_list, val_metric_list




