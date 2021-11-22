import torch


class SoilDataset(torch.utils.data.TensorDataset):
    """Soil Features dataset."""

    def __init__(self, data):
        super().__init__()
        """
        Args:
            data (pandas DataFrame): The pandas DataFrame containing the targets and features.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.targets = data[:, 0]
        self.features = data[:, 1:]

    def __len__(self):
        """Get length of dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        "Get one sample."
        if torch.is_tensor(idx):
            idx = idx.tolist()
        targ = self.targets[idx]
        feat = self.features[idx]
        return {"target": targ, "features": feat}
