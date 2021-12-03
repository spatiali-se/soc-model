import torch


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
        self.targets = data[:, 0]
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

