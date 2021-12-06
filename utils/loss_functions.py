import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

def negative_log_likelighood(y_pred, y_true):
    """ Compute negative log-likelihood loss """
    NLL = torch.pow(y_pred[:, 0:1] - y_true, 2)
    NLL = torch.divide(NLL, y_pred[:, 1:2]) + torch.log(y_pred[:, 1:2])
    return NLL.mean()
