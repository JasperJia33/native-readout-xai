"""Train and evaluate on MVTec AD (binary: good vs defective).

Usage:
    python main_mvtec.py --config configs_mvtec/resnet18_cbam_mvtec.yaml
    python main_mvtec.py --config configs_mvtec/vit_tiny_mvtec.yaml --seed 123
"""
import argparse
import json
import logging
import os
import random
from collections import Counter
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset_mvtec import MVTecADDataset, load_mvtec_samples
from src.models_mvtec import build_mvtec_model
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
    parser = argparse.ArgumentParser(description='MVTec AD Training')
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg['training']['seed'] = args.seed

    seed = cfg['training']['seed']
    model_name = cfg['model']['name']
    short_name = model_name.replace('_rgb', '')

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{short_name}_seed{seed}"

    run_dir = os.path.join('outputs_mvtec', run_name)

    # Skip if already completed
    if os.path.isfile(os.path.join(run_dir, 'test_results.json')):
        print(f"[SKIP] {run_name} already completed (test_results.json exists)")
        return

    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    setup_logging(run_dir)
    log = logging.getLogger(__name__)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Run: {run_name}  Device: {device}  Seed: {seed}")

    # --- Data ---
    data_root = cfg['data']['path']
    train_samples, test_samples, train_indices, test_indices = load_mvtec_samples(data_root)

    # Pool all samples for proper stratified split
    all_samples = train_samples + test_samples
    all_indices = train_indices + test_indices
    all_labels = [0 if s['defect']['label'] == 'good' else 1 for s in all_samples]
    log.info(f"Total pooled: {len(all_samples)} (good={all_labels.count(0)}, defective={all_labels.count(1)})")

    # 70% train, 15% val, 15% test — stratified
    idx_range = list(range(len(all_samples)))
    train_idx, temp_idx = train_test_split(
        idx_range, test_size=0.3, random_state=seed,
        stratify=all_labels
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed,
        stratify=temp_labels
    )

    train_data = [all_samples[i] for i in train_idx]
    val_data = [all_samples[i] for i in val_idx]
    test_data = [all_samples[i] for i in test_idx]
    train_global_idx = [all_indices[i] for i in train_idx]
    val_global_idx = [all_indices[i] for i in val_idx]
    test_global_idx = [all_indices[i] for i in test_idx]

    train_labels_final = [all_labels[i] for i in train_idx]
    log.info(f"Final split — Train: {len(train_data)} (good={train_labels_final.count(0)}, def={train_labels_final.count(1)})")
    log.info(f"  Val: {len(val_data)}, Test: {len(test_data)}")

    # Datasets
    val_cfg = {**cfg, 'augmentation': {**cfg.get('augmentation', {}), 'enabled': False}}
    train_ds = MVTecADDataset(train_data, data_root, cfg, augmentation=True, sample_indices=train_global_idx)
    val_ds = MVTecADDataset(val_data, data_root, val_cfg, augmentation=False, sample_indices=val_global_idx)
    test_ds = MVTecADDataset(test_data, data_root, val_cfg, augmentation=False, sample_indices=test_global_idx)

    tc = cfg['training']
    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'], shuffle=True,
                              num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc['batch_size_eval'], shuffle=False,
                            num_workers=cfg['data']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=tc['batch_size_eval'], shuffle=False,
                             num_workers=cfg['data']['num_workers'], pin_memory=True)

    # Class weights (handle imbalance: ~4096 good vs ~1258 defective)
    class_counts = Counter(train_labels_final)
    total = sum(class_counts.values())
    n_classes = cfg['model']['num_classes']
    weights = torch.tensor([total / (n_classes * class_counts[c]) for c in range(n_classes)],
                           dtype=torch.float32).to(device)
    log.info(f"Class weights: {weights}")

    # --- Model ---
    model = build_mvtec_model(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {model_name}  Parameters: {total_p:,}")

    # --- Train ---
    model, history = train_model(model, train_loader, val_loader, cfg, weights, device, run_dir=run_dir)

    # Save
    torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # --- Evaluate ---
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    evaluator = WaferMapEvaluator(num_classes=n_classes)
    results = evaluator.evaluate(y_true, y_pred, y_prob)
    evaluator.print_report(results)

    results_save = {
        'run_name': run_name,
        'model': model_name,
        'dataset': 'mvtec_ad',
        'params': total_p,
        'accuracy': results['accuracy'],
        'balanced_accuracy': results['balanced_accuracy'],
        'f1_macro': results['f1_macro'],
        'epochs_trained': len(history['train_loss']),
    }
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    log.info(f"Done. Run folder: {run_dir}")


if __name__ == '__main__':
    main()
