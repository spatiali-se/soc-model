import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

def negative_log_likelighood(y_pred_mean, y_pred_std, y_true):
    """ Compute negative log-likelihood loss """
    std_squared = torch.pow(y_pred_std, 2)
    NLL = torch.pow(y_pred_mean - y_true, 2)
    return torch.divide(NLL, std_squared) + torch.log(std_squared)
