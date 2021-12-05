import warnings
import argparse
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import mlflow

from modules.datasets import get_data_loaders
from modules.training import Trainer
import models.dense_neural_net as models
from transforms.preprocess import preprocessor
import utils.test_routines as test_routines
from utils.seed_everything import seed_everything


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seed_everything()
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {str(device).upper()} as ´device´")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="ratio of train data per split"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="ratio of test data per split"
    )
    parser.add_argument(
        "--data", type=str, default="lucas_2015.parquet", help="name of data file"
    )
    parser.add_argument("--preprocess", type=bool, default=True, help="preprocess data")
    parser.add_argument(
        "--plot", type=bool, default=False, help="plot training metrics"
    )
    parser.add_argument(
        "--print_metrics", type=bool, default=False, help="print metrics"
    )
    args = parser.parse_args()

    # Read args
    train_ratio = float(args.train_ratio)
    test_ratio = float(args.test_ratio)
    data_file = str(args.data)
    preprocess = bool(args.preprocess)
    plot_metrics = bool(args.plot)
    print_metrics = bool(args.print_metrics)

    if (train_ratio + test_ratio > 1) or (train_ratio + test_ratio < 0):
        raise ValueError("train_ratio + test_ratio must equal number in range [0-1]")

    # Read data (make sure you're running this from the root of the MLFlow)

    current_dir = pathlib.Path(__file__).parent.absolute()
    data_dir = current_dir.joinpath("data")
    data_path = data_dir.joinpath(data_file)

    # Make sure the data is in the correct format
    # TODO: Add logic elsewhere to deal with other formats than parquet
    if data_file.split(".")[-1] == "parquet":
        data = pd.read_parquet(data_path)
    elif data_file.split(".")[-1] == "csv":
        data = pd.read_csv(data_path)
    elif data_file.split(".")[-1] == "np":
        data = np.load(data_path)
    else:
        raise ValueError("data file must be .parquet, .csv or .np")

    if preprocess:
        preproc = preprocessor(data)
    else:
        preproc = None

    # Hyperparameters
    # TODO: Incoroporate variable hparams into cli args parsing
    dataloader_params = {
        "data": data,
        "batch_size": 64,
        "train_ratio": 0.7,
        "test_ratio": 0.15,
        "preprocessor": preproc,
    }
    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        **dataloader_params
    )
    nn_params = {
        "input_dim": train_dataloader.dataset[0]["features"].shape[0],
        "output_dim": 1,
        "hidden_dims": [16, 16, 16],
        "activation": nn.LeakyReLU(),
        "dropout_rate": 0.20,
    }
    train_params = {"num_epochs": 10, "patience": 10, "early_stopping": True}
    optim_params = {"lr": 1e-2, "weight_decay": 1e-6}

    with mlflow.start_run():
        # Create NN
        model = models.DenseNN(**nn_params)
        # Set up optimizer
        optimizer = optim.Adam(params=model.parameters(), **optim_params)
        # Set up loss function
        loss_function = nn.MSELoss()
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
        )

        # Plot training metrics
        if plot_metrics:
            plt.figure()
            plt.semilogy(train_loss, linewidth=2.0, label="Train MSE loss")
            plt.semilogy(val_metrics[:, 0], linewidth=2.0, label="Val MSE loss")
            plt.semilogy(val_metrics[:, 1], linewidth=2.0, label="Val L1 loss")
            plt.grid()
            plt.legend()
            plt.show()

        # Run test routine
        test_model = test_routines.Tester(model, test_dataloader)

        # TODO: Fix why it's printing metrics when args is set to False
        # Print test metrics
        if print_metrics:
            test_model.print_metrics()

        mlflow.log_param("dropout_rate", nn_params["dropout_rate"])
        mlflow.log_metric("MAE", test_model.MAE)
        mlflow.log_metric("MSE", test_model.MSE)
        mlflow.log_metric("APE", test_model.APE)
        mlflow.log_metric("APC", test_model.APC)
        mlflow.log_metric("RMSE", test_model.RMSE)
        mlflow.log_metric("rRMSE", test_model.rRMSE)

        mlflow.pytorch.log_model(model, "model")
