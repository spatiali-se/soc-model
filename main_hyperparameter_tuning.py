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
import pdb
import seaborn as sns

from modules.datasets import get_data_loaders
from modules.training import Trainer
import models.dense_neural_net as models
from modules.hyperparameter_tuning import fitness_function
from transforms.preprocess import preprocessor
import utils.test_routines as test_routines
from utils.seed_everything import seed_everything
from utils.loss_functions import negative_log_likelighood
from sklearn.preprocessing import MinMaxScaler
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.tune.suggest.hyperopt import HyperOptSearch




if __name__ == "__main__":

    seed_everything()
    torch.set_default_dtype(torch.float32)
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
        "--print_metrics", type=bool, default=True, help="print metrics"
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

    #data = data[data['OC'] < 15]

    if preprocess:
        preproc = preprocessor(data)
    else:
        preproc = None
    


    config = {
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "num_neurons_pr_layer": tune.choice([4, 8, 16, 32, 64]),
        "lr": tune.uniform(1e-4, 1e-1),
        "weight_decay": tune.uniform(1e-12, 1e-1),
        "dropout_rate": tune.uniform(0, 0.5),
        "batch_size": tune.choice([64, 128, 256, 512, 1024]),
        "activation": tune.choice(['relu', 'leakyrelu', 'sigmoid', 'tanh', 'elu'])
    }

    initial_config = [{'num_layers': 2,
     'num_neurons_pr_layer': 32, 
     'lr': 0.022716884904880226, 
     'weight_decay': 0.016084283856934195, 
     'dropout_rate': 0.3306179285688111, 
     'batch_size': 256,
     "activation": 'relu'
     }]
    

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10000,
        grace_period=200,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])

    search_alg = HyperOptSearch(metric="loss", mode="min",
                                points_to_evaluate=initial_config)


    #fitness_function(config=config, data=data, preproc=preproc)
    #pdb.set_trace()
    result = tune.run(
        partial(fitness_function, data=data, preproc=preproc),
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=30000,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_failures=3,
        verbose=1,
        search_alg=search_alg)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))