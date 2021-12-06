import pdb

import torch.nn as nn
import torch
import models.dense_neural_net_with_uncertainty as models
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
        pred = torch.zeros([x.shape[0], self.output_dim, self.num_models])

        for i, model in enumerate(self.ensemble):
            pred[:, :, i] = model(x)

        pred_mean = pred[:,0].mean(dim=1)
        pred_var = torch.mean(pred[:,1] + torch.pow(pred[:,0], 2), dim=1)\
                   - torch.pow(pred_mean, 2)

        pred_mean = torch.unsqueeze(pred_mean, dim=1)
        pred_var = torch.unsqueeze(pred_var, dim=1)

        pred = torch.cat([pred_mean, pred_var], dim=1)
        return pred

    def compile(self, optimizers, loss_function, device):
        """"""

        self.device = device
        self.optimizers = optimizers
        self.loss_function = loss_function

        for model in self.ensemble:
            model = model.to(self.device)

        self.trainers = [Trainer(model=model,
                                  optimizer=optimizer,
                                  loss_function=loss_function,
                                  device=device) for _, (model, optimizer) in
                          enumerate(zip(self.ensemble, self.optimizers))]



    def fit(self, train_loader, val_loader, val_metrics,
            num_epochs, patience, early_stopping):

        train_loss_list = []
        val_metric_list = []
        for trainer in self.trainers:
            train_loss, val_metric = trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    val_metrics=val_metrics,
                    num_epochs=num_epochs,
                    patience=patience,
                    early_stopping=early_stopping,
            )
            train_loss_list.append(train_loss)
            val_metric_list.append(val_metric)
        return train_loss_list, val_metric_list




