from dataclasses import dataclass
from pathlib import Path
import math
import torch


@dataclass
class EarlyStopping:
    patience: int = 20
    min_delta: float = 0.0005
    best: float = -math.inf
    bad_epochs: int = 0
    triggered: bool = False

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError('Non-finite validation score')
        if score > self.best + self.min_delta:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        
        self.triggered |= self.bad_epochs >= self.patience
        return self.triggered


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device='cpu'):

    from ..models import EEGNetDAL
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = EEGNetDAL(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval(), checkpoint
