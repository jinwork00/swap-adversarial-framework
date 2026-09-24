import random
import torch
from torch.utils.data import DataLoader, default_collate


class AugmentedCollateFunction:


    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, batch):
        x, labels = default_collate(batch)
        if not isinstance(labels, (tuple, list)) or len(labels) != 2:
            raise ValueError('ISBCS requires both class and subject/domain labels')
        y, domains = labels
        if (domains < 0).any():
            raise ValueError('ISBCS requires known, nonnegative source-domain labels')
        return self.augmentation((x, y, domains)), (y, domains)


def _seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


def create_dataloaders(datasets, batch_size=16, *, augmentation=None,
                       seed=42, num_workers=0, drop_last=True):


    if seed is None:
        raise ValueError('Provide an integer seed')
    if augmentation is not None:
        domains = datasets.train.domain_y
        if domains is None or (domains < 0).any():
            raise ValueError('Training augmentation requires known domain labels')
    loaders = {}
    for split in ('train', 'val', 'test'):
        train = split == 'train'
        loaders[split] = DataLoader(
            getattr(datasets, split), batch_size=batch_size,
            shuffle=train, drop_last=drop_last if train else False,
            collate_fn=AugmentedCollateFunction(augmentation) if train and augmentation is not None else None,
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers, worker_init_fn=_seed_worker,
        )
    return loaders
