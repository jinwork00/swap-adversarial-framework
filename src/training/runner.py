import csv
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..data import load_datasets, create_dataloaders
from ..data.augmentation import SubjectLevelChannelSwap
from ..models import EEGNetDAL
from .config import TrainingConfig
from .engine import create_optimizers, run_epoch
from .checkpoints import EarlyStopping, save_checkpoint, load_model


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fit(data_config, data_root='.', output_dir=None, config=None):


    config = TrainingConfig() if config is None else config
    config.validate()
    device = torch.device(config.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu or explicitly selected cuda device')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    output = Path(output_dir) if output_dir is not None else Path('runs') / datetime.now(timezone.utc).strftime('saf_%Y%m%d_%H%M%S_%f')
    if output.exists():
        raise FileExistsError(f'Output directory already exists: {output}')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    datasets = load_datasets(data_config, root=data_root, seed=config.seed)
    if datasets.num_classes < 2 or len(config.class_weights) != datasets.num_classes:
        raise ValueError('class_weights must match the number of classes (at least two)')
    if datasets.num_domain_classes < 1:
        raise ValueError('DAL training requires domain_list with source subjects')
    if not len(datasets.val):
        raise ValueError('Validation samples are required')
    augmentation = SubjectLevelChannelSwap(datasets.chans, 'uniform', config.channel_swap_p) if config.augmentation else None
    loaders = create_dataloaders(datasets, batch_size=config.batch_size,
                                augmentation=augmentation, seed=config.seed,
                                num_workers=config.num_workers)
    if not len(loaders['train']):
        raise ValueError('Training set must contain at least batch_size samples (drop_last=True)')
    model_args = dict(nb_classes=datasets.num_classes, domain_classes=datasets.num_domain_classes,
                      Chans=datasets.chans, Samples=datasets.samples, kernLength=config.kern_length,
                      F1=config.f1, D=config.depth_multiplier, F2=config.f2,
                      dropoutRate=config.dropout, dropoutType='Dropout', channel_norm=config.channel_norm,
                      norm_rate=config.norm_rate, class_weight=list(config.class_weights),
                      grl_lambda=config.grl_lambda, mi_lambda=config.mi_lambda,
                      mi_lambda_learnable=config.mi_lambda_learnable,
                      domain_label_smoothing=config.domain_label_smoothing)
    model = EEGNetDAL(**model_args).to(device)
    optimizers, schedulers = create_optimizers(model, config)
    stop = EarlyStopping(config.early_stop_patience, config.early_stop_min_delta)
    best_score, best_epoch = -float('inf'), 0
    resolved = dict(training=asdict(config), model=model_args,
                    data_root=str(Path(data_root).resolve()), data_config=data_config,
                    split_sizes={key: len(getattr(datasets, key)) for key in ('train', 'val', 'test')},
                    monitors={'checkpoint': 'val_target_macro_acc', 'early_stopping': 'val_target_macro_acc',
                              'target_scheduler': 'val_target_macro_acc (max)',
                              'domain_scheduler': 'val_domain_macro_acc (min)'},
                    versions={'torch': str(torch.__version__), 'numpy': str(np.__version__)})
    
    mappings = {name: [{'label': label, 'index': index} for label, index in mapping.items()]
                for name, mapping in [('class', datasets.label_to_index), ('domain', datasets.domain_label_to_index)]}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'config.json', resolved)
    _write_json(output / 'label_mappings.json', mappings)
    writer = None
    with (output / 'history.csv').open('w', newline='') as stream:
        for epoch in range(1, config.max_epochs + 1):
            train = run_epoch(model, loaders['train'], device, optimizers=optimizers, grad_clip_norm=config.grad_clip_norm)
            val = run_epoch(model, loaders['val'], device)
            row = {'epoch': epoch, 'target_lr': optimizers[0].param_groups[0]['lr'],
                   'domain_lr': optimizers[1].param_groups[0]['lr']}
            row.update({'train_' + key: value for key, value in train.items()})
            row.update({'val_' + key: value for key, value in val.items()})
            score = val['target_macro_acc']
            schedulers[0].step(score)
            schedulers[1].step(val['domain_macro_acc'])
            stop_requested = stop.update(score)
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            payload = dict(epoch=epoch, model_args=model_args, model_state_dict=model.state_dict(),
                           optimizer_states=[o.state_dict() for o in optimizers],
                           scheduler_states=[s.state_dict() for s in schedulers],
                           early_stopping=asdict(stop), best_score=best_score, best_epoch=best_epoch,
                           training_config=asdict(config), label_mappings=mappings, metrics=row)
            if improved:
                save_checkpoint(output / 'best.pt', payload)
            save_checkpoint(output / 'last.pt', payload)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(f'Epoch {epoch}/{config.max_epochs}: val_macro_acc={score:.6f}, best={best_score:.6f}', flush=True)
            if epoch >= config.min_epochs and stop_requested:
                break
    result = dict(best_epoch=best_epoch, best_val_target_macro_acc=best_score,
                  epochs_completed=epoch, early_stopped=epoch < config.max_epochs)
    if len(datasets.test):
        best_model, _ = load_model(output / 'best.pt', device)
        test = run_epoch(best_model, loaders['test'], device, target_only=True)
        _write_json(output / 'test_metrics.json', {'checkpoint': 'best.pt', 'epoch': best_epoch,
                                                  **{'test_' + key: value for key, value in test.items()}})
    result['test_evaluated'] = bool(len(datasets.test))
    _write_json(output / 'summary.json', result)
    return output
