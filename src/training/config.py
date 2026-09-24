from dataclasses import dataclass, asdict
import math


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    device: str = 'cpu'
    kern_length: int = 256
    f1: int = 8
    depth_multiplier: int = 2
    f2: int = 16
    dropout: float = 0.5
    channel_norm: bool = True
    norm_rate: float = 0.25
    class_weights: tuple = (1.0, 1.0)
    grl_lambda: float = 0.1
    mi_lambda: float = 0.01
    mi_lambda_learnable: bool = False
    augmentation: bool = True
    channel_swap_p: float = 0.5
    lr: float = 0.001
    domain_lr_factor: float = 0.1
    grad_clip_norm: float = 1.0
    domain_label_smoothing: float = 0.1
    min_epochs: int = 60
    max_epochs: int = 250
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0005
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_threshold: float = 0.001
    scheduler_cooldown: int = 10

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        for key in ('batch_size', 'kern_length', 'f1', 'depth_multiplier', 'f2', 'max_epochs', 'early_stop_patience'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if not 0 <= self.min_epochs <= self.max_epochs:
            raise ValueError('Require 0 <= min_epochs <= max_epochs')
        if min(self.num_workers, self.scheduler_patience, self.scheduler_cooldown,
               self.early_stop_min_delta, self.scheduler_threshold, self.grad_clip_norm) < 0:
            raise ValueError('Worker counts, patience, cooldown, thresholds and clipping must be nonnegative')
        if self.lr <= 0 or self.domain_lr_factor <= 0 or self.norm_rate <= 0:
            raise ValueError('Learning rates and norm_rate must be positive')
        if not 0 < self.scheduler_factor < 1:
            raise ValueError('scheduler_factor must be between 0 and 1')
        if not 0 <= self.dropout < 1 or not 0 <= self.channel_swap_p <= 1 or not 0 <= self.domain_label_smoothing <= 1:
            raise ValueError('Invalid dropout, swap probability or label smoothing')
        if not self.class_weights or any(not math.isfinite(v) or v <= 0 for v in self.class_weights):
            raise ValueError('class_weights must be finite and positive')
