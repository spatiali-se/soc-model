import numpy as np
from sklearn.metrics import r2_score


def eval_metrics(y_true, y_pred):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    rmse = np.sqrt(mse)
    mae = np.absolute(np.subtract(y_true, y_pred)).mean()
    # r2 = np.square(np.corrcoef(y_true, y_pred)[0,1])
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
