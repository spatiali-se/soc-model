import pdb

import torch.nn as nn
import torch


class DenseNN(torch.nn.Module):
    """Dense Neural Network"""
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation=nn.LeakyReLU(), dropout_rate=0.5):
        """Initialize dense neural net

        args:
            input_dim (int): Input dimension.
            hidden_dims (list of ints): List of dimension of each hidden layer.
            output_dim (int): Output dimension
            activation (torch.nn.activation): Activation function.
            dropout_rate (float): Dropout rate.
        """
        super(DenseNN, self).__init__()

        self.num_hidden_layers = len(hidden_dims)
        self.activation = activation

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear_input = nn.Linear(in_features=input_dim,
                                      out_features=hidden_dims[0],
                                      bias=True)

        self.linear = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(1, self.num_hidden_layers):
            self.batchnorm.append(nn.BatchNorm1d(num_features=hidden_dims[i - 1]))
            self.linear.append(nn.Linear(in_features=hidden_dims[i - 1],
                                         out_features=hidden_dims[i],
                                         bias=True))

        self.linear_output = nn.Linear(in_features=hidden_dims[-1],
                                       out_features=output_dim,
                                       bias=False)

    def forward(self, x):
        """Forward propagation"""

        x = self.linear_input(x)
        x = self.dropout(x)
        x = self.activation(x)

        for (linear, batchnorm) in zip(self.linear, self.batchnorm):
            x = batchnorm(x)
            x = linear(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.linear_output(x)
        x[:, 1] = torch.log(1 + torch.exp(x[:, 1])) + 1e-6
        return x
