
import torch.nn as nn
import torch.optim as optim
import torch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from modules.datasets import get_data_loaders
from modules.training import Trainer
import models.dense_neural_net as models
from transforms.preprocess import preprocessor
import utils.test_routines as test_routines
from utils.seed_everything import seed_everything
from utils.loss_functions import negative_log_likelighood
from sklearn.preprocessing import MinMaxScaler


def fitness_function(config, data, preproc, device='cpu'):

    dataloader_params = {
        "data": data,
        "batch_size": config['batch_size'],
        "train_ratio": 0.7,
        "test_ratio": 0.15,
        "preprocessor": preproc,
    }
    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        **dataloader_params
    )

    if config['activation'] == 'relu':
        activation = nn.ReLU()
    elif config['activation'] == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif config['activation'] == 'sigmoid':
        activation = nn.Sigmoid()
    elif config['activation'] == 'tanh':
        activation = nn.Tanh()
    elif config['activation'] == 'elu':
        activation = nn.ELU()

    nn_params = {
        "input_dim": train_dataloader.dataset[0]["features"].shape[0],
        "output_dim": 1,
        "hidden_dims": [config['num_neurons_pr_layer'] 
                            for i in range(config['num_layers'])],
        "activation": activation,
        "dropout_rate": config['dropout_rate'],
    }
    train_params = {"num_epochs": 100000, "patience": 500, "early_stopping": False}
    optim_params = {"lr": config['lr'], "weight_decay": config['weight_decay']}

    # Create NN
    model = models.DenseNN(**nn_params)
    # Set up optimizer
    optimizer = optim.SGD(params=model.parameters(), **optim_params)
    # Set up loss function
    loss_function = nn.MSELoss()#negative_log_likelighood
    # Set up val metrics
    val_metrics = [nn.MSELoss(), nn.L1Loss()]



    # Set up NN trainer
    trainer = Trainer(
        model=model, optimizer=optimizer, loss_function=loss_function, device=device
    )
    train_loss, val_metrics = trainer.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,  # Currently
        val_metrics=val_metrics,
        **train_params,
        with_tune=True,
        print_progress=False
        )
