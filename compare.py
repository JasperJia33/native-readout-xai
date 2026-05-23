"""Compare the latest run of each model. Produces outputs/comparison/."""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import build_model
from utils.metrics import CLASS_NAMES


def find_all_runs(outputs_dir='outputs'):
    """Scan outputs/ and return ALL run_dirs with results (keyed by folder name)."""
    runs = {}
    if not os.path.isdir(outputs_dir):
        return runs
    for name in sorted(os.listdir(outputs_dir)):
        cfg_path = os.path.join(outputs_dir, name, 'config.yaml')
        res_path = os.path.join(outputs_dir, name, 'test_results.json')
        if not os.path.isfile(cfg_path) or not os.path.isfile(res_path):
            continue
        runs[name] = os.path.join(outputs_dir, name)
    return runs


def find_latest_runs(outputs_dir='outputs'):
    """Scan outputs/ and return the latest run_dir per model name."""
    latest = {}
    if not os.path.isdir(outputs_dir):
        return latest
    for name in sorted(os.listdir(outputs_dir)):
        cfg_path = os.path.join(outputs_dir, name, 'config.yaml')
        res_path = os.path.join(outputs_dir, name, 'test_results.json')
        if not os.path.isfile(cfg_path) or not os.path.isfile(res_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg['model']['name']
        latest[model_name] = os.path.join(outputs_dir, name)
    return latest


def load_run(run_dir):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, 'test_results.json')) as f:
        results = json.load(f)
    hist_path = os.path.join(run_dir, 'training_history.json')
    history = json.load(open(hist_path)) if os.path.isfile(hist_path) else None
    return cfg, results, history


def measure_speed(cfg, device, n=100):
    model = build_model(cfg).to(device).eval()
    sz = cfg['data']['image_size']
    x = torch.randn(1, 1, sz, sz, device=device)
    # Warmup
    for _ in range(10):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main():
    parser = argparse.ArgumentParser(description='Compare model runs')
    parser.add_argument('--runs', nargs='*', default=None, help='Explicit run dirs')
    parser.add_argument('--latest-only', action='store_true', help='Only latest run per model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.runs:
        run_dirs = {}
        for r in args.runs:
            run_dirs[os.path.basename(r)] = r
    elif args.latest_only:
        run_dirs = find_latest_runs()
    else:
        run_dirs = find_all_runs()

    if not run_dirs:
        print("No runs found in outputs/. Train models first.")
        return

    out_dir = os.path.join('outputs', 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    runs = {}
    for model_name, run_dir in run_dirs.items():
        cfg, results, history = load_run(run_dir)
        speed = measure_speed(cfg, device)
        runs[model_name] = {'cfg': cfg, 'results': results, 'history': history,
                            'speed_ms': speed, 'run_dir': run_dir}

    # --- Comparison table ---
    header = f"{'Model':<20} {'Acc':>7} {'BAcc':>7} {'F1mac':>7} {'F1wt':>7} {'MCC':>7} {'Params':>10} {'ms/img':>8}"
    sep = '-' * len(header)
    lines = [sep, header, sep]
    for name, r in runs.items():
        res = r['results']
        lines.append(f"{name:<20} {res['accuracy']:>7.4f} {res['balanced_accuracy']:>7.4f} "
                      f"{res['f1_macro']:>7.4f} {res['f1_weighted']:>7.4f} {res['mcc']:>7.4f} "
                      f"{res['params']:>10,} {r['speed_ms']:>7.2f}")
    lines.append(sep)

    # Per-class F1
    class_names = CLASS_NAMES[:len(list(runs.values())[0]['results']['per_class_f1'])]
    lines.append(f"\n{'Per-Class F1':<20} " + ' '.join(f'{n:>10}' for n in class_names))
    for name, r in runs.items():
        f1s = r['results']['per_class_f1']
        lines.append(f"{name:<20} " + ' '.join(f'{v:>10.4f}' for v in f1s))
    lines.append(sep)

    table_text = '\n'.join(lines)
    print(table_text)
    with open(os.path.join(out_dir, 'comparison_table.txt'), 'w') as f:
        f.write(table_text)

    # --- Plots ---
    names = list(runs.keys())

    # 1. Accuracy metrics bar chart
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [runs[name]['results'][m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Balanced Acc', 'F1 Macro', 'F1 Weighted'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title('Model Comparison — Accuracy Metrics')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison_accuracy.png'), dpi=150)
    plt.close(fig)

    # 2. F1 radar chart
    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name in names:
        vals = runs[name]['results']['per_class_f1'] + [runs[name]['results']['per_class_f1'][0]]
        ax.plot(angles, vals, label=name, linewidth=2)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Per-Class F1 Radar')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison_f1_radar.png'), dpi=150)
    plt.close(fig)

    # 3. Confusion matrices side by side
    # (requires y_true/y_pred which we don't store — skip if not available)

    # 4. Training curves overlay
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name in names:
        h = runs[name].get('history')
        if not h:
            continue
        ep = range(1, len(h['train_loss']) + 1)
        axes[0].plot(ep, h['val_loss'], label=name)
        axes[1].plot(ep, h['val_acc'], label=name)
        axes[2].plot(ep, h['val_balanced_acc'], label=name)
    axes[0].set_title('Val Loss')
    axes[1].set_title('Val Accuracy')
    axes[2].set_title('Val Balanced Accuracy')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison_training_curves.png'), dpi=150)
    plt.close(fig)

    # 5. Params & speed
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    params = [runs[n]['results']['params'] / 1e6 for n in names]
    speeds = [runs[n]['speed_ms'] for n in names]
    axes[0].bar(names, params, color='steelblue')
    axes[0].set_ylabel('Parameters (M)')
    axes[0].set_title('Model Size')
    axes[1].bar(names, speeds, color='coral')
    axes[1].set_ylabel('Inference (ms/image)')
    axes[1].set_title('Inference Speed')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison_params_speed.png'), dpi=150)
    plt.close(fig)

    print(f"\nComparison saved to {out_dir}/")


if __name__ == '__main__':
    main()
