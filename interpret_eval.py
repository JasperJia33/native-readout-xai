"""Quantitative interpretability evaluation — Deletion/Insertion AUC + Stability.

Runs the three metrics for every (family, seed) combination listed in the
paper-runs registry and writes aggregated artifacts for the report/paper.

Usage:
    python interpret_eval.py
    python interpret_eval.py --paper_runs configs/paper_runs.yaml \
                             --n_samples 200 --topk 0.1 \
                             --output outputs/interpretability

Outputs (all under --output):
    metrics.json                    per-sample results (full)
    metrics_per_sample.csv          one row per (family, seed, sample)
    summary.csv                     mean ± std per family
    deletion_insertion_curves.png   mean curves (one line per family)
    stability_boxplot.png           stability distribution per family
    qualitative_panel.png           one sample per class × 3 families
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.interpretability import (
    get_heatmap, heatmap_to_topk_mask,
    deletion_auc, insertion_auc, stability_score,
)
from utils.metrics import CLASS_NAMES


def defect_heatmap_correlation(heatmap, wafer_img):
    """Spearman rank correlation between heatmap intensity and defect-pixel
    indicator at each pixel.

    heatmap: (H, W) float array, model's saliency/attention map
    wafer_img: (H, W) float array, wafer map (defect pixels are brightest)

    Returns a scalar in [-1, 1]. Near zero means the heatmap is not
    aligned with the defect distribution; high positive means they
    co-vary spatially. Uses rank correlation to be robust to the
    different dynamic ranges of Grad-CAM vs. Attention Rollout.
    """
    h = np.asarray(heatmap, dtype=np.float64).ravel()
    w = np.asarray(wafer_img, dtype=np.float64).ravel()
    if h.std() < 1e-10 or w.std() < 1e-10:
        return 0.0  # degenerate heatmap or blank wafer
    # Rank-transform both
    from scipy.stats import spearmanr
    rho, _ = spearmanr(h, w)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def load_paper_runs(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc['paper_runs']


def load_model_from_run(run_dir, device):
    """Build model from run's config and load best_model.pth."""
    cfg_path = os.path.join(run_dir, 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(device)
    state = torch.load(os.path.join(run_dir, 'best_model.pth'),
                       map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def stratified_sample(test_df, n_samples, num_classes, seed=0):
    """Return a balanced stratified sample of test indices."""
    per_class = max(1, n_samples // num_classes)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(num_classes):
        idxs = test_df.index[test_df['failurePatternType'] == c].to_numpy()
        if len(idxs) == 0:
            continue
        take = min(per_class, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
    return np.array(chosen[:n_samples])


def evaluate_run(run_dir, test_ds, sample_indices, family_name, device,
                 topk=0.10, n_steps=20, n_augs=5, seed=0):
    """Run Del/Ins/Stability on one model for the given sample indices."""
    print(f"  [{family_name}] Loading {run_dir}...")
    model, cfg = load_model_from_run(run_dir, device)
    model_name_raw = cfg['model']['name']  # 'resnet18_cbam' etc.

    results = []
    t0 = time.time()
    for k, idx in enumerate(sample_indices):
        x, y = test_ds[int(idx)]   # (1, H, W) tensor, int label
        y = int(y.item())

        # Single forward pass to pick the class to explain (pred class)
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device))
            pred = int(logits.argmax(1).item())

        heatmap, _ = get_heatmap(model, x, model_name_raw, device,
                                 class_idx=pred)

        d = deletion_auc(model, x, heatmap, device, class_idx=pred,
                         n_steps=n_steps)
        ins = insertion_auc(model, x, heatmap, device, class_idx=pred,
                            n_steps=n_steps)
        stab = stability_score(model, x, model_name_raw, device,
                               n_augs=n_augs, seed=seed + k,
                               class_idx=pred)

        # Spatial alignment: correlation between heatmap and wafer defect
        # pixels. High positive means the heatmap tracks the defect
        # distribution; near zero means they are spatially unrelated.
        wafer_np = x.squeeze().cpu().numpy()
        defect_corr = defect_heatmap_correlation(heatmap, wafer_np)

        results.append({
            'family': family_name,
            'run_dir': run_dir,
            'sample_idx': int(idx),
            'true_label': y,
            'pred_label': pred,
            'deletion_auc': d['auc'],
            'insertion_auc': ins['auc'],
            'stability': stab['mean'],
            'defect_corr': defect_corr,
            'deletion_curve': d['curve'].tolist(),
            'insertion_curve': ins['curve'].tolist(),
            'fractions': d['fractions'].tolist(),
        })

        if (k + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"    [{family_name}] {k+1}/{len(sample_indices)} "
                  f"({dt:.1f}s, {dt/(k+1):.2f}s/sample)")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


def plot_curves(all_results, out_path):
    """Deletion/Insertion mean-curve comparison across families."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Group by family
    by_family = defaultdict(list)
    for r in all_results:
        by_family[r['family']].append(r)

    colors = {'resnet18_cbam': 'tab:orange', 'densenet121': 'tab:blue',
              'vit_tiny': 'tab:green'}

    for fam, rows in by_family.items():
        frac = np.asarray(rows[0]['fractions'])
        del_curves = np.vstack([r['deletion_curve'] for r in rows])
        ins_curves = np.vstack([r['insertion_curve'] for r in rows])
        color = colors.get(fam, None)

        mean_d, std_d = del_curves.mean(0), del_curves.std(0)
        axes[0].plot(frac, mean_d, label=fam, color=color, lw=2)
        axes[0].fill_between(frac, mean_d - std_d, mean_d + std_d,
                             alpha=0.15, color=color)

        mean_i, std_i = ins_curves.mean(0), ins_curves.std(0)
        axes[1].plot(frac, mean_i, label=fam, color=color, lw=2)
        axes[1].fill_between(frac, mean_i - std_i, mean_i + std_i,
                             alpha=0.15, color=color)

    axes[0].set_title('Deletion (lower = more faithful)')
    axes[0].set_xlabel('Fraction of pixels removed')
    axes[0].set_ylabel('P(predicted class)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title('Insertion (higher = more faithful)')
    axes[1].set_xlabel('Fraction of pixels inserted')
    axes[1].set_ylabel('P(predicted class)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle('Deletion / Insertion AUC — mean ± std across samples',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_stability_boxplot(all_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    by_family = defaultdict(list)
    for r in all_results:
        by_family[r['family']].append(r['stability'])
    fams = sorted(by_family.keys())
    data = [by_family[f] for f in fams]
    bp = ax.boxplot(data, tick_labels=fams, patch_artist=True, showmeans=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_ylabel('Cosine similarity under perturbation')
    ax.set_title('Explanation Stability (higher = more robust)')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_qualitative_panel(families_dict, test_ds, test_df, device, out_path,
                            class_names=None):
    """One row per class (up to 9), columns = original + 3 family heatmaps.

    Picks the highest-confidence correctly-classified sample per class
    using the first family's model as reference.
    """
    class_names = class_names or CLASS_NAMES
    fam_names = list(families_dict.keys())

    # Use first family to pick representative samples
    ref_model, ref_cfg = load_model_from_run(families_dict[fam_names[0]][0], device)
    ref_name = ref_cfg['model']['name']

    # Find best sample per class
    samples_per_class = {}
    ref_model.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            y_int = int(y.item())
            logits = ref_model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze()
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())
            if pred == y_int:
                if y_int not in samples_per_class or conf > samples_per_class[y_int][0]:
                    samples_per_class[y_int] = (conf, i)
            if len(samples_per_class) == len(class_names) and all(
                    v[0] > 0.95 for v in samples_per_class.values()):
                break
    del ref_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    samples_per_class = {c: idx for c, (_, idx) in samples_per_class.items()}
    classes = sorted(samples_per_class.keys())
    n_rows = len(classes)
    n_cols = 1 + len(fam_names)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Load one model per family
    loaded = {}
    for fam, runs in families_dict.items():
        model, cfg = load_model_from_run(runs[0], device)
        loaded[fam] = (model, cfg['model']['name'])

    for r, c in enumerate(classes):
        idx = samples_per_class[c]
        x, y = test_ds[idx]
        img = x.squeeze().numpy()
        axes[r, 0].imshow(img, cmap='inferno')
        axes[r, 0].set_ylabel(class_names[c], fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title('Input', fontsize=11, pad=12)

        for j, fam in enumerate(fam_names):
            model, m_name = loaded[fam]
            heatmap, _ = get_heatmap(model, x, m_name, device)
            axes[r, 1 + j].imshow(img, cmap='gray', alpha=0.5)
            axes[r, 1 + j].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[r, 1 + j].set_xticks([]); axes[r, 1 + j].set_yticks([])
            if r == 0:
                axes[r, 1 + j].set_title(fam, fontsize=11, pad=12)

    fig.suptitle('Qualitative interpretability comparison (per class)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Free models
    for fam in list(loaded.keys()):
        del loaded[fam]
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs/paper_runs.yaml')
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--topk', type=float, default=0.10)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--n_augs', type=int, default=5)
    ap.add_argument('--output', default='outputs/interpretability')
    ap.add_argument('--sample_seed', type=int, default=0,
                    help='Seed for stratified sampling (fixed for reproducibility)')
    ap.add_argument('--family', type=str, default=None,
                    help='Evaluate only this family (e.g. swin_tiny)')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load test set (using the primary config of each family -> they
    # all share the same data settings other than val/test split which is
    # deterministic; we load the ResNet config for data.) ---
    paper_runs = load_paper_runs(args.paper_runs)
    print(f"Paper runs: {list(paper_runs.keys())}")

    any_cfg_path = list(paper_runs.values())[0]['config']
    with open(any_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    # Disable augmentation on the test dataset
    data_cfg = {**data_cfg,
                'augmentation': {**data_cfg['augmentation'], 'enabled': False}}
    loader = WM811KLoader(data_cfg)
    loader.load()
    loader.get_labeled_data()
    _, _, test_df = loader.split_data()
    test_df = test_df.reset_index(drop=True)
    class_counts = Counter(int(l) for l in test_df['failurePatternType'])
    test_ds = WM811KDataset(test_df, data_cfg, class_counts)

    num_classes = data_cfg['model']['num_classes']
    sample_indices = stratified_sample(test_df, args.n_samples, num_classes,
                                       seed=args.sample_seed)
    print(f"Sampled {len(sample_indices)} test indices (stratified across "
          f"{num_classes} classes)")

    # --- Per-run evaluation ---
    all_results = []
    families_to_eval = paper_runs.items()
    if args.family:
        families_to_eval = [(k, v) for k, v in paper_runs.items() if k == args.family]
        if not families_to_eval:
            print(f"Family '{args.family}' not found. Available: {list(paper_runs.keys())}")
            return
    for fam, meta in families_to_eval:
        for run_dir in meta['runs']:
            if not os.path.isfile(os.path.join(run_dir, 'best_model.pth')):
                print(f"  [SKIP] {run_dir} missing best_model.pth")
                continue
            rows = evaluate_run(run_dir, test_ds, sample_indices, fam, device,
                                topk=args.topk, n_steps=args.n_steps,
                                n_augs=args.n_augs, seed=args.sample_seed)
            all_results.extend(rows)

    if not all_results:
        print("No runs evaluated — aborting.")
        return

    # --- Save per-sample metrics ---
    # Heavy JSON (includes curves)
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(all_results, f)
    # Light CSV (scalar metrics only)
    light = [{k: v for k, v in r.items()
              if k not in ('deletion_curve', 'insertion_curve', 'fractions')}
             for r in all_results]
    df = pd.DataFrame(light)
    df.to_csv(os.path.join(args.output, 'metrics_per_sample.csv'), index=False)

    # --- Aggregate per family (mean±std across all seeds × samples) ---
    summary_rows = []
    for fam in df['family'].unique():
        sub = df[df['family'] == fam]
        row = {
            'family': fam,
            'n_runs': sub['run_dir'].nunique(),
            'n_samples': len(sub),
            'deletion_auc_mean': sub['deletion_auc'].mean(),
            'deletion_auc_std': sub['deletion_auc'].std(),
            'insertion_auc_mean': sub['insertion_auc'].mean(),
            'insertion_auc_std': sub['insertion_auc'].std(),
            'stability_mean': sub['stability'].mean(),
            'stability_std': sub['stability'].std(),
            'defect_corr_mean': sub['defect_corr'].mean(),
            'defect_corr_std': sub['defect_corr'].std(),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.output, 'summary.csv'), index=False)
    print("\n=== Interpretability Summary (mean ± std) ===")
    print(summary.to_string(index=False))

    # --- Figures ---
    plot_curves(all_results, os.path.join(args.output,
                                          'deletion_insertion_curves.png'))
    plot_stability_boxplot(all_results, os.path.join(args.output,
                                                     'stability_boxplot.png'))
    families_dict = {fam: meta['runs'] for fam, meta in paper_runs.items()}
    plot_qualitative_panel(families_dict, test_ds, test_df, device,
                           os.path.join(args.output, 'qualitative_panel.png'))

    print(f"\nAll outputs -> {args.output}/")


if __name__ == '__main__':
    main()
