from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import EEGDataset


@dataclass
class EEGDatasets:
    train: EEGDataset
    val: EEGDataset
    test: EEGDataset
    label_to_index: dict
    domain_label_to_index: dict

    @property
    def chans(self):
        return self.train.data_x.shape[1]

    @property
    def samples(self):
        return self.train.data_x.shape[2]

    @property
    def num_classes(self):
        return len(self.label_to_index)

    @property
    def num_domain_classes(self):
        return len(self.domain_label_to_index)


def load_datasets(data_config, root='.', *, seed=42):


    splits = ('train', 'val', 'test')
    if seed is None:
        raise ValueError('Provide an integer seed for reproducible splitting')
    base = Path(root)
    def file_map(section):
        if set(section) - set(splits):
            raise ValueError('Only train, val, and test split names are supported')
        result = {s: {} for s in splits}
        for split, groups in section.items():
            for label, paths in groups.items():
                if label is None or isinstance(paths, (str, Path)):
                    raise ValueError('Use non-null labels and lists of file paths')
                for path in paths:
                    path = (base / path).resolve()
                    if path in result[split] and result[split][path] != label:
                        raise ValueError(f'Conflicting labels for {path.name}')
                    result[split][path] = label
        return result
    files = file_map(data_config['data_list'])
    domains = file_map(data_config['domain_list']) if 'domain_list' in data_config else None
    seen = set()
    for split in splits:
        if seen.intersection(files[split]):
            raise ValueError('The same file is assigned to multiple data splits')
        seen.update(files[split])
        if domains is not None and files[split].keys() != domains[split].keys():
            raise ValueError(f'{split}: class and domain file lists must match')
    arrays, targets, subjects = {}, {}, {}
    shape = None
    for split in splits:
        blocks, labels, domain_labels = [], [], []
        for path, label in files[split].items():
            values = np.load(path, allow_pickle=False)
            if (not isinstance(values, np.ndarray) or values.ndim != 3
                    or not np.issubdtype(values.dtype, np.number)
                    or np.iscomplexobj(values) or min(values.shape) < 1):
                raise ValueError(f'{path.name}: expected nonempty real numeric [N,C,T] .npy array')
            if shape is not None and values.shape[1:] != shape:
                raise ValueError('All files must have the same channel/time dimensions')
            shape = values.shape[1:]
            blocks.append(values)
            labels.extend([label] * len(values))
            if domains is not None:
                domain_labels.extend([domains[split][path]] * len(values))
        arrays[split] = np.concatenate(blocks) if blocks else None
        targets[split], subjects[split] = labels, domain_labels
    if arrays['train'] is None:
        raise ValueError('Training files are required')
    label_map = {label: i for i, label in enumerate(sorted(set(sum(targets.values(), []))))}
    domain_map = {label: i for i, label in enumerate(sorted(set(subjects['train'])))}
    for split in splits:
        targets[split] = np.array([label_map[v] for v in targets[split]], dtype=np.int64)
        if domains is not None:
            if split != 'test' and any(v not in domain_map for v in subjects[split]):
                raise ValueError('Validation domains must occur in training')
            subjects[split] = np.array([domain_map.get(v, -1) for v in subjects[split]], dtype=np.int64)
        else:
            subjects[split] = None
        if arrays[split] is None:
            arrays[split] = np.empty((0, *shape), dtype=np.float32)
    if not len(arrays['val']):
        def partition(indices, fraction):
            y, d = targets['train'][indices], subjects['train']
            key = y if d is None else np.array([f'{a}_{b}' for a, b in zip(y, d[indices])])
            return train_test_split(indices, test_size=fraction, random_state=seed,
                                    shuffle=True, stratify=key)
        train_ids, held_ids = partition(np.arange(len(arrays['train'])), 0.2)
        chosen = {'train': train_ids}
        if len(arrays['test']):
            chosen['val'] = held_ids
        else:
            chosen['val'], chosen['test'] = partition(held_ids, 0.5)
        original_x, original_y, original_d = arrays['train'], targets['train'], subjects['train']
        for split, indices in chosen.items():
            arrays[split], targets[split] = original_x[indices], original_y[indices]
            subjects[split] = None if original_d is None else original_d[indices]
    datasets = {s: EEGDataset(arrays[s], targets[s], subjects[s]) for s in splits}
    return EEGDatasets(**datasets, label_to_index=label_map, domain_label_to_index=domain_map)
