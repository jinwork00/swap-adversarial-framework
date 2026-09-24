import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


class ClassificationMetrics:
    def __init__(self, classes):
        if classes < 1:
            raise ValueError('classes must be positive')
        self.classes = classes
        self.samples = 0
        args = {'num_classes': classes, 'validate_args': classes > 1}
        self.metrics = MetricCollection({
            'macro_acc': MulticlassAccuracy(average='macro', **args),
            'accuracy': MulticlassAccuracy(average='micro', **args),
            'macro_precision': MulticlassPrecision(average='macro', **args),
            'macro_recall': MulticlassRecall(average='macro', **args),
            'macro_f1': MulticlassF1Score(average='macro', **args),
        })

    def update(self, logits, labels):
        labels = labels.detach().to('cpu', dtype=torch.long)
        if (logits.ndim != 2 or logits.shape[1] != self.classes
                or labels.ndim != 1 or len(labels) != len(logits)
                or ((labels < 0) | (labels >= self.classes)).any()):
            raise ValueError('Invalid classification labels or logits')
        preds = logits.detach().argmax(1).cpu()
        self.metrics.update(preds, labels)
        self.samples += len(labels)

    def compute(self):
        if not self.samples:
            raise ValueError('Cannot compute metrics without samples')
        return {name: value.item() for name, value in self.metrics.compute().items()}
