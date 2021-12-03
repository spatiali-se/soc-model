import torch
from sklearn.model_selection import train_test_split
import pdb

class SoilDataset(torch.utils.data.TensorDataset):
    """Soil Features dataset."""

    def __init__(self, data, features_transform=None, target_transform=None):
        super().__init__()
        """
        Args:
            data (pandas DataFrame): The pandas DataFrame containing the targets and features.
            feature_transform (callable, optional): Optional transform to be applied on a feature sample.
            target_transform (callable, optional): Optional transform to be applied on a target sample.
        """
        self.targets = data[:, 0:1]
        self.features = data[:, 1:]

        self.features_transform = features_transform
        self.target_transform = target_transform

    def __len__(self):
        """Get length of dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        "Get one sample."
        if torch.is_tensor(idx):
            idx = idx.tolist()
        targ = self.targets[idx]
        feat = self.features[idx]

        if self.features_transform:
            feat = self.features_transform(feat)
        if self.target_transform:
            targ = self.target_transform(targ)

        return {"target": targ, "features": feat}

def split_data(data, split):
    """Splits dataset to train, val, and test data"""

    train_ratio = split[0]
    val_ratio = split[1]
    test_ratio = split[2]

    train_data, test_data = train_test_split(data, test_size=1 - train_ratio)

    val_data, test_data = train_test_split(test_data,
                                           test_size=test_ratio / (
                                                       test_ratio + val_ratio))
    return train_data, val_data, test_data

def get_data_loaders(data, split=[0.7, 0.15, 0.15],
                     batch_size=64,
                     features_transform=None,
                     target_transform=None):
    """Generates train, val, and test dataloaders

    args:
        data (pandas dataframe): The pandas DataFrame containing the targets and features.
        split (list): list with train, val, and test ratio
        feature_transform (callable, optional): Optional transform to be applied on a feature sample.
        target_transform (callable, optional): Optional transform to be applied on a target sample.
    """

    # Split data
    train_data, val_data, test_data = split_data(data=data, split=split)

    # Train dataset
    train_data = SoilDataset(train_data,
                             features_transform=features_transform,
                             target_transform=target_transform)
    # Val dataset
    val_data = SoilDataset(val_data,
                           features_transform=features_transform,
                           target_transform=target_transform)
    # Test dataset
    test_data = SoilDataset(test_data)

    # Train dataloader
    train_dataloader =torch.utils.data.DataLoader(train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    # Val dataloader
    val_dataloader =torch.utils.data.DataLoader(val_data,
                                                batch_size=batch_size)
    # Test dataloader
    test_dataloader =torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader