import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader


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

        return {"features": feat, "target": targ}


def split_data(data, train_ratio, test_ratio):
    """Splits dataset into train, val, and test data"""

    # Split into train(-val) and test
    train_split, test_split = train_test_split(data, test_size=test_ratio)
    # Split train(-val) into train and val if wanted
    if train_ratio + test_ratio == 1:
        return train_split, test_split

    train_split, val_split = train_test_split(
        train_split, test_size=1 - train_ratio - test_ratio
    )

    return train_split, val_split, test_split


def get_data_loaders(
    data,
    train_ratio=0.7,
    test_ratio=0.15,
    batch_size=64,
    preprocessor=None,
    features_transform=None,
    target_transform=None,
):
    """Generates train, val, and test dataloaders

    args:
        data (pandas dataframe): The pandas DataFrame containing the targets and features.
        split (list): list with train, val, and test ratio
        feature_transform (callable, optional): Optional transform to be applied on a feature sample.
        target_transform (callable, optional): Optional transform to be applied on a target sample.
    """
    # TODO: num_workers on batchloaders?

    # Split data
    train_data, *test_data = split_data(
        data=data, train_ratio=train_ratio, test_ratio=test_ratio
    )

    if len(test_data) == 2:
        val_data, test_data = test_data

    # Preprocess data
    if preprocessor != None:
        train_data = preprocessor.fit_transform(train_data)
        test_data = preprocessor.transform(test_data)
        if "val_data" in locals():
            val_data = preprocessor.transform(val_data)

    # Train dataset
    train_data = SoilDataset(
        train_data,
        features_transform=features_transform,
        target_transform=target_transform,
    )

    # Test dataset
    test_data = SoilDataset(test_data)

    # Train dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # Test dataloader
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Check for val dataset
    if "val_data" in locals():
        # Val dataset
        val_data = SoilDataset(
            val_data,
            features_transform=features_transform,
            target_transform=target_transform,
        )
        # Val dataloader
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    return train_dataloader, test_dataloader
