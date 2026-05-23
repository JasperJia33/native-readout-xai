import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from utils.metrics import compute_class_weights, WaferMapEvaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(run_dir):
    log_fmt = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(run_dir, 'train.log')),
    ]
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=handlers, force=True)


def main():
    parser = argparse.ArgumentParser(description='WM-811K Wafer Map Training')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--run_name', default=None, help='Run name (default: {model}_{datetime})')
    parser.add_argument('--seed', type=int, default=None, help='Override cfg[training][seed]')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed override (CLI wins over config)
    if args.seed is not None:
        cfg['training']['seed'] = args.seed

    # Run name
    model_name = cfg['model']['name']
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join('outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Copy config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Logging & seed
    setup_logging(run_dir)
    log = logging.getLogger(__name__)
    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Run: {run_name}  Device: {device}")

    # --- Data ---
    loader = WM811KLoader(cfg)
    loader.load()
    loader.get_labeled_data()

    log.info("Class distribution:\n" + str(loader.df['failurePatternType'].value_counts().sort_index()))
    train_df, val_df, test_df = loader.split_data()

    class_counts = Counter(int(l) for l in train_df['failurePatternType'])

    # Datasets — disable augmentation for val/test
    val_cfg = {**cfg, 'augmentation': {**cfg['augmentation'], 'enabled': False}}
    train_ds = WM811KDataset(train_df, cfg, class_counts)
    val_ds = WM811KDataset(val_df, val_cfg, class_counts)
    test_ds = WM811KDataset(test_df, val_cfg, class_counts)

    tc = cfg['training']
    dc = cfg['data']
    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'], shuffle=True,
                              num_workers=dc['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc['batch_size_eval'], shuffle=False,
                            num_workers=dc['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=tc['batch_size_eval'], shuffle=False,
                             num_workers=dc['num_workers'], pin_memory=True)

    # Class weights
    class_weights = compute_class_weights(train_df['failurePatternType'].values, device)
    log.info(f"Class weights: {class_weights}")

    # --- Model ---
    model = build_model(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {model_name}  Parameters: {total_p:,}")

    # --- Train ---
    model, history = train_model(model, train_loader, val_loader, cfg, class_weights, device, run_dir=run_dir)

    # Save model & history
    torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    log.info("Model and history saved.")

    # --- Evaluate ---
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    evaluator = WaferMapEvaluator(num_classes=cfg['model']['num_classes'])
    results = evaluator.evaluate(y_true, y_pred, y_prob)
    evaluator.print_report(results)

    # Save results
    results_save = {
        'run_name': run_name,
        'model': model_name,
        'params': total_p,
        'accuracy': results['accuracy'],
        'balanced_accuracy': results['balanced_accuracy'],
        'mcc': results['mcc'],
        'f1_macro': results['f1_macro'],
        'f1_weighted': results['f1_weighted'],
        'per_class_f1': results['f1_per_class'].tolist(),
        'epochs_trained': len(history['train_loss']),
    }
    if results.get('roc_auc_macro') is not None:
        results_save['roc_auc_macro'] = results['roc_auc_macro']

    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    log.info(f"Results saved to {run_dir}/test_results.json")
    log.info(f"Done. Run folder: {run_dir}")


if __name__ == '__main__':
    main()
