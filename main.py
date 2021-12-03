import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import modules.datasets as dataset
import modules.training_neural_networks as training
import models.dense_neural_net as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy



if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters - Needs to be incorporated into a parser!
    learning_rate = 1e-2
    weight_decay = 1e-6
    hidden_dims = [16, 16, 16]
    activation = nn.LeakyReLU()
    batch_size = 64
    dropout_rate = 0.20
    num_epochs = 100
    patience = 10

    # Load data
    data_string = 'dataset_one_hot.npy'  # one-hot encoded
    df = np.load(data_string)

    # Define split ratios
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    split = [train_ratio, val_ratio, test_ratio]

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = \
        dataset.get_data_loaders(data=df,
                                 split=split,
                                 batch_size=batch_size)

    # Create NN
    input_dim = train_dataloader.dataset[0]['features'].shape[0]
    output_dim = 1
    model = models.DenseNN(input_dim=input_dim,
                           hidden_dims=hidden_dims,
                           output_dim=output_dim,
                           activation=activation,
                           dropout_rate=dropout_rate)

    # Set up optimizer
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # Set up loss function
    loss_function = nn.MSELoss()

    # Set up val metrics
    val_metrics = [nn.MSELoss(), nn.L1Loss()]

    # Set up NN trainer
    NN_trainer = training.TrainModel(model=model,
                                     optimizer=optimizer,
                                     loss_function=loss_function,
                                     device=device)
    train_loss, val_metrics = NN_trainer.fit(train_loader=train_dataloader,
                                             val_loader=val_dataloader,
                                             num_epochs=num_epochs,
                                             val_metrics=val_metrics,
                                             early_stopping=True,
                                             patience=10)







