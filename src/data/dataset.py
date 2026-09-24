import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):


    def __init__(self, data_x, data_y, domain_y=None):
        self.data_x = torch.as_tensor(data_x, dtype=torch.float32)
        self.data_y = self._labels(data_y, 'data_y')
        self.domain_y = None if domain_y is None else self._labels(domain_y, 'domain_y')
        if self.data_x.ndim != 3 or min(self.data_x.shape[1:]) < 1:
            raise ValueError('data_x must have shape [N,C,T] with positive C and T')
        if len(self.data_x) != len(self.data_y):
            raise ValueError('Sample and class label counts differ')
        if self.domain_y is not None and len(self.domain_y) != len(self.data_x):
            raise ValueError('Sample and domain label counts differ')
        if (self.data_y < 0).any():
            raise ValueError('Class labels must be nonnegative')

    @staticmethod
    def _labels(values, name):
        labels = torch.as_tensor(values)
        if labels.ndim != 1 or labels.is_complex() or (
            labels.is_floating_point() and
            (not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()))
        ):
            raise ValueError(f'{name} must contain one integer per sample')
        return labels.to(dtype=torch.long)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        x, y = self.data_x[index].unsqueeze(-1), self.data_y[index]
        return (x, y) if self.domain_y is None else (x, (y, self.domain_y[index]))
