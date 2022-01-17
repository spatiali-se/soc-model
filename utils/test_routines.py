import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pdb


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
        #print(f"MSE = {self.MSE:0.3f}")
        print(f"RMSE = {self.RMSE:0.3f}")
        #print(f"Relative RMSE = {self.rRMSE:0.3f}")
        #print(f"RMS Realtive error = {self.RMSrE:0.3f}")
        #print(f"Mean Absolute error = {self.MAE:0.3f}")
        print(f"Mean percentage error = {self.MAPE:0.3f}")
        print(f"Reliability at 20% accuracy = {self.reliability:0.3f}")

    def plot_predictions(self,):
        plt.figure()
        plt.plot(self.preds, self.true_values, '.')
        plt.plot(self.true_values, self.true_values, '.')
        plt.show()




    # TODO: return metrics
    def compute_metrics(self, model, test_loader):
        """Compute test metrics"""
        batch_size = test_loader.batch_size
        
        self.true_values = torch.zeros([len(test_loader.dataset), 1])
        self.preds = torch.zeros([len(test_loader.dataset), 1])
        self.vars = torch.zeros([len(test_loader.dataset), 1])
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                x_test = test_data["features"].to(self.device)
                y_test = test_data["target"].to(self.device)

                y_pred = model(x_test)
                if self.inverse_transform_target:
                    y_pred = self.inverse_transform_target(y_pred[:,0:1])
                    y_test = self.inverse_transform_target(y_test[:,0:1])

                self.preds[i*batch_size:(i*batch_size+y_test.shape[0])]\
                    = y_pred[:, 0:1].detach()

                self.true_values[i*batch_size:(i*batch_size+y_test.shape[0])] \
                    = y_test.detach()


        self.true_L2_norm = torch.sum(torch.pow(self.true_values,2))
        self.true_L1_norm = torch.sum(torch.abs(self.true_values))

        self.SE = torch.pow(self.preds-self.true_values,2)
        self.MSE = torch.mean(self.SE)
        self.SrE = torch.divide(self.SE,torch.pow(self.true_values,2))
        self.RMSE = torch.sqrt(self.MSE)
        self.rRMSE = self.MSE/self.true_L2_norm
        self.MSrE = torch.mean(self.SrE)
        self.RMSrE = torch.sqrt(self.MSrE)
        self.AE = torch.abs(self.preds-self.true_values)
        self.ArE = torch.divide(self.AE,torch.abs(self.true_values))
        self.MAE = torch.mean(self.AE)
        self.rMAE = torch.mean(self.AE)/self.true_L1_norm
        self.MArE = torch.mean(self.ArE)
        self.APE = self.ArE * 100
        self.MAPE = torch.mean(self.APE)
        self.APC = torch.abs(100 - self.APE)
        self.MAPC = torch.mean(self.APC)
        self.MPC = torch.mean(self.APC)
        
        self.reliability = self.APE<20.
        self.reliability = torch.sum(self.reliability)/len(self.reliability) * 100
        self.reliability = self.reliability.item()