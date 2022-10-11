from typing import Dict, List
from collections import defaultdict
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def accuracy(y: Tensor, yhat: Tensor) -> float:
    pred = torch.argmax(F.softmax(yhat, dim=1), dim=1)
    acc = (torch.nonzero(y)[:, 1] == pred).sum() / y.shape[0]
    return acc


def class_hits(logits: Tensor, targets: Tensor) -> Dict[int, List[int]]:
    """Compute the per-class accuracy for imbalanced datasets."""
    cls_hits = defaultdict(list)
    y_hat = torch.argmax(logits, axis=-1)
    y = torch.nonzero(targets)[:, -1]
    for i, y_i in enumerate(y):
        cls_hits[y_i.item()].append(1 if y_i == y_hat[i] else 0)
    return cls_hits
