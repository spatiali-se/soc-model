import numpy as np
import torch.nn as nn
import torch


class Tester:
    """Class to compute and store all test metrics"""

    def __init__(self, model, test_loader, inverse_transform_target=None, device="cpu"):
        """Initialize class

        args:
            model (Torch model):
            test_loader (dataloader):
            inverse_transform_target (callable, optional): Optional transform
                                                           to be applied on a
                                                           target sample.
            device (string): device to run the testing, e.g. 'cuda'
        """

        super().__init__()

        self.model = model
        self.test_loader = test_loader
        self.inverse_transform_target = inverse_transform_target
        self.device = device
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss(reduce="mean")

        self.compute_metrics(model, test_loader)

    def print_metrics(self):
        print(f"MSE = {self.MSE:0.3f}")
        print(f"RMSE = {self.RMSE:0.3f}")
        print(f"Relative RMSE = {self.relative_RMSE:0.3f}")
        print(f"MAE = {self.MAE:0.3f}")
        print(f"Average percentage error = {self.average_percentage_error:0.3f}")
        print(f"Average percentage covering = {self.average_percentage_covering:0.3f}")

    # TODO: return metrics
    def compute_metrics(self, model, test_loader):
        """Compute test metrics"""

        self.MSE = 0
        self.MAE = 0
        self.RMSE = 0
        self.relative_RMSE = 0
        self.average_percentage_error = 0
        self.average_percentage_covering = 0
        test_norm = 0

        with torch.no_grad():
            for i, val_data in enumerate(test_loader):
                x_test = val_data["features"].to(self.device)
                y_test = val_data["target"].to(self.device)

                test_y_hat = model(x_test)
                if self.inverse_transform_target:
                    test_y_hat = self.inverse_transform_target(test_y_hat)

                self.MSE += self.MSELoss(y_test, test_y_hat).item()
                self.MAE += self.L1Loss(y_test, test_y_hat).item()
                self.average_percentage_error += torch.mean(
                    torch.abs((y_test - test_y_hat) / y_test)
                )
                self.average_percentage_covering += torch.mean(
                    1 - torch.abs((y_test - test_y_hat) / y_test)
                )

                test_norm += torch.sum(torch.pow(y_test, 2)).item()

        self.MAE /= i
        self.MSE /= i
        self.average_percentage_error *= 100 / i
        self.average_percentage_covering *= 100 / i
        self.RMSE = np.sqrt(self.MSE)
        self.relative_RMSE = np.sqrt(self.RMSE / test_norm)
        # return {
        #     "MSE": self.MSE,
        #     "RMSE": self.RMSE,
        #     "rRMSE": self.relative_RMSE,
        #     "MAE": self.MAE,
        # }
