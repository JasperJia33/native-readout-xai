import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score,
)

CLASS_NAMES = [
    'None', 'Edge-Loc', 'Edge-Ring', 'Loc',
    'Center', 'Scratch', 'Random', 'Donut', 'Near-Full',
]


def compute_class_weights(labels, device='cpu'):
    """Inverse-frequency class weights."""
    counts = Counter(int(l) for l in labels)
    n = len(labels)
    nc = len(counts)
    weights = torch.tensor(
        [n / (nc * counts[i]) for i in range(nc)],
        dtype=torch.float32,
    )
    return weights.to(device)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class WaferMapEvaluator:
    """Compute and print comprehensive classification metrics."""

    def __init__(self, num_classes=9):
        self.num_classes = num_classes

    def evaluate(self, y_true, y_pred, y_prob=None):
        r = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }
        if y_prob is not None:
            try:
                r['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
            except Exception:
                r['roc_auc_macro'] = None
        return r

    def print_report(self, results):
        print("=" * 60)
        print("WAFER MAP CLASSIFICATION EVALUATION REPORT")
        print("=" * 60)
        print(f"  Accuracy:           {results['accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
        print(f"  MCC:                {results['mcc']:.4f}")
        print(f"  F1 (macro):         {results['f1_macro']:.4f}")
        print(f"  F1 (weighted):      {results['f1_weighted']:.4f}")
        if results.get('roc_auc_macro'):
            print(f"  ROC-AUC (macro):    {results['roc_auc_macro']:.4f}")
        print("\nPer-Class F1:")
        for name, f1 in zip(CLASS_NAMES, results['f1_per_class']):
            print(f"  {name:15s}: {f1:.4f}")
        print("=" * 60)
