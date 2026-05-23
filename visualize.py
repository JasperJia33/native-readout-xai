"""Generate all plots for a single experiment run."""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.evaluate import evaluate_model
from utils.metrics import CLASS_NAMES, WaferMapEvaluator
from utils.visualize import (
    plot_class_distribution, plot_sample_wafers, plot_training_curves,
    plot_confusion_matrix, plot_per_class_f1, plot_misclassified,
    explain_prediction,
)


def main():
    parser = argparse.ArgumentParser(description='Generate plots for a run')
    parser.add_argument('--run_dir', required=True, help='Path to run folder')
    args = parser.parse_args()

    run_dir = args.run_dir
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nc = cfg['model']['num_classes']
    class_names = CLASS_NAMES[:nc]

    # Load data
    loader = WM811KLoader(cfg)
    loader.load()
    loader.get_labeled_data()
    _, _, test_df = loader.split_data()

    # Data plots
    plot_class_distribution(loader.df, class_names, os.path.join(plots_dir, 'class_distribution.png'))
    plot_sample_wafers(loader.df, class_names=class_names, save_path=os.path.join(plots_dir, 'sample_wafers.png'))

    # Training curves
    hist_path = os.path.join(run_dir, 'training_history.json')
    if os.path.isfile(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(plots_dir, 'training_curves.png'))

    # Load model
    model = build_model(cfg).to(device)
    weights_path = os.path.join(run_dir, 'best_model.pth')
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # Evaluate for confusion matrix
    val_cfg = {**cfg, 'augmentation': {**cfg['augmentation'], 'enabled': False}}
    test_ds = WM811KDataset(test_df, val_cfg)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))

    results = WaferMapEvaluator(nc).evaluate(y_true, y_pred, y_prob)
    plot_per_class_f1(results['f1_per_class'], class_names, os.path.join(plots_dir, 'f1_per_class.png'))
    plot_misclassified(model, test_ds, device, class_names, n=12,
                       save_path=os.path.join(plots_dir, 'misclassified.png'))

    # Explainability — one sample per class using model-specific best method
    seen_classes = set()
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        if y.item() not in seen_classes:
            seen_classes.add(y.item())
            explain_prediction(model, x, y.item(), device, class_names,
                               os.path.join(plots_dir, f'explain_class{y.item()}.png'))
        if len(seen_classes) >= min(6, nc):
            break

    print(f"Plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
