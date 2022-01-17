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
import models.ensemble_model_with_uncertainty as models
from transforms.preprocess import preprocessor
import utils.test_routines as test_routines
from utils.seed_everything import seed_everything
from utils.loss_functions import negative_log_likelighood


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
        "num_models": 5,
        "input_dim": train_dataloader.dataset[0]["features"].shape[0],
        "output_dim": 2,
        "hidden_dims": [8, 8],
        "activation": nn.LeakyReLU(),
        "dropout_rate": 0.20,
    }
    train_params = {"num_epochs": 500, "patience": 30, "early_stopping": True}
    optim_params = {"lr": 1e-4, "weight_decay": 1e-7}

    with mlflow.start_run():
        # Create NN
        model = models.EnsembleNN(**nn_params)
        # Set up optimizer
        optimizer = [
            optim.Adam(params=model_i.parameters(), **optim_params)
            for model_i in model.ensemble
        ]
        # Set up loss function
        loss_function = negative_log_likelighood
        # Set up val metrics
        val_metrics = [negative_log_likelighood, nn.MSELoss()]
        # Compile ensemble model
        model.compile(
            optimizers=optimizer,
            loss_function=loss_function,
            metrics=val_metrics,
            device=device,
        )

        # Train ensemble models
        train_loss_list, val_metric_list = model.fit(
            train_loader=train_dataloader, val_loader=val_dataloader, **train_params
        )

        # Plot training metrics
        if plot_metrics:
            plt.figure()
            for i in range(nn_params["num_models"]):
                plt.semilogy(
                    train_loss_list[i],
                    linewidth=1.5,
                    label="Train MSE loss",
                    color="tab:blue",
                )
                plt.semilogy(
                    val_metric_list[i][:, 0],
                    linewidth=1.5,
                    label="Val MSE loss",
                    color="tab:orange",
                )
                plt.semilogy(
                    val_metric_list[i][:, 1],
                    linewidth=1.5,
                    label="Val L1 loss",
                    color="tab:green",
                )
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
