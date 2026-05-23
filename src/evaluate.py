import logging
import torch
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device):
    """Run inference and return labels, predictions, and probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Evaluating', leave=False):
            X = X.to(device)
            out = model(X)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)
