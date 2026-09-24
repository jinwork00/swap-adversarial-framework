import torch
from torch import nn


def negative_entropy(probabilities, mi_lambda=0.01, target_labels=None):


    per_sample = (probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)
    if isinstance(mi_lambda, nn.Parameter):
        if target_labels is None:
            raise ValueError('target_labels are required for class-specific lambda')
        return (mi_lambda[target_labels] * per_sample).mean()
    return mi_lambda * per_sample.mean()
