import random
from typing import Tuple
import torch

def _get_swap_probabilities(num_channels: int, method: str, p=0.5) -> torch.Tensor:
    if method == 'normal':
        return torch.abs(torch.randn(num_channels)) % 1.0
    elif method == 'beta':
        return torch.distributions.Beta(0.5, 0.5).sample((num_channels,))
    elif method == 'uniform':
        return torch.full((num_channels,), p)
    else:
        raise ValueError(f'Unsupported swap probability method: {method}')

def _decide_swap(swap_probabilities: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(swap_probabilities).bool()

class SubjectLevelChannelSwap:


    def __init__(self, channel_num, swap_probability_method='uniform',
                 channel_swap_p=0.5, exclude_swap_channels=None):
        if channel_num <= 0:
            raise ValueError('channel_num must be positive')
        if swap_probability_method not in ('uniform', 'normal', 'beta'):
            raise ValueError('Unknown swap probability method')
        if not 0 <= channel_swap_p <= 1:
            raise ValueError('channel_swap_p must be in [0, 1]')
        self.channel_num = channel_num
        self.swap_probability_method = swap_probability_method
        self.channel_swap_p = float(channel_swap_p)
        self.exclude_swap_channels = set(exclude_swap_channels or [])
        if any(c < 0 or c >= channel_num for c in self.exclude_swap_channels):
            raise ValueError('Excluded channel index out of range')

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        data_tensor, y_labels, domain_labels = data
        if domain_labels is None:
            raise ValueError('domain_labels must be provided for SubjectLevelChannelSwap.')
        if y_labels is None:
            raise ValueError('y_labels must be provided for SubjectLevelChannelSwap.')
        augmented_data = data_tensor.clone()
        subjects_in_batch = torch.unique(domain_labels).tolist()
        if len(subjects_in_batch) < 2:
            return augmented_data
        subject_indices = {sid: (domain_labels == sid).nonzero(as_tuple=True)[0] for sid in subjects_in_batch}
        channels_to_swap = _decide_swap(_get_swap_probabilities(self.channel_num, self.swap_probability_method, p=self.channel_swap_p))
        for ch_idx in self.exclude_swap_channels:
            channels_to_swap[ch_idx] = False
        for channel_idx in torch.where(channels_to_swap)[0]:
            shuffled_subjects = random.sample(subjects_in_batch, len(subjects_in_batch))
            subject_map = {orig: shuffled for orig, shuffled in zip(subjects_in_batch, shuffled_subjects)}
            temp_channel_data = augmented_data[:, channel_idx, :, :].clone()
            for orig_subject, target_subject in subject_map.items():
                orig_indices, target_indices = (subject_indices[orig_subject], subject_indices[target_subject])
                if len(orig_indices) == len(target_indices):
                    for orig_idx, target_idx in zip(orig_indices, target_indices):
                        if y_labels[orig_idx] == y_labels[target_idx]:
                            augmented_data[target_idx, channel_idx, :, :] = temp_channel_data[orig_idx]
        return augmented_data

