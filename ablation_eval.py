"""Ablation experiments for reviewer response.

Produces four supplementary analyses using existing trained checkpoints
(no retraining required):

1. Random-ranking baseline for Deletion/Insertion AUC
2. Grad-CAM on ViT (non-native control — tests model–explainer compatibility)
3. Commonly-correct subset (samples all 3 families classify correctly)
4. Top-k sensitivity (5%, 10%, 20%)

Usage:
    python ablation_eval.py [--paper_runs configs/paper_runs.yaml]
                            [--n_samples 200] [--output outputs/ablation]
"""
import argparse, json, os, time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.interpretability import (
    get_heatmap, deletion_auc, insertion_auc, _ordered_indices,
    _predict_prob, heatmap_to_topk_mask,
)
from utils.visualize import GradCAM
from utils.metrics import CLASS_NAMES


def load_paper_runs(path):
    with open(path) as f:
        return yaml.safe_load(f)['paper_runs']


def load_model_from_run(run_dir, device):
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


# ---------------------------------------------------------------------------
# Ablation 1: Random baseline
# ---------------------------------------------------------------------------
def random_deletion_insertion(model, image_tensor, device, class_idx,
                              n_steps=20, seed=42):
    """Deletion/Insertion with random pixel ordering (no heatmap)."""
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    N = H * W
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    step = max(1, N // n_steps)

    # Deletion
    x_del = image_tensor.clone().float()
    del_fracs, del_probs = [0.0], [float(_predict_prob(model, x_del, device, class_idx)[0])]
    removed = 0
    for _ in range(n_steps):
        end = min(removed + step, N)
        idxs = order[removed:end]
        rr, cc = np.unravel_index(idxs, (H, W))
        x_del[..., rr, cc] = 0.0
        removed = end
        del_fracs.append(removed / N)
        del_probs.append(float(_predict_prob(model, x_del, device, class_idx)[0]))

    # Insertion
    x_ins = torch.zeros_like(image_tensor, dtype=torch.float32)
    x_orig = image_tensor.float()
    ins_fracs, ins_probs = [0.0], [float(_predict_prob(model, x_ins, device, class_idx)[0])]
    added = 0
    for _ in range(n_steps):
        end = min(added + step, N)
        idxs = order[added:end]
        rr, cc = np.unravel_index(idxs, (H, W))
        x_ins[..., rr, cc] = x_orig[..., rr, cc]
        added = end
        ins_fracs.append(added / N)
        ins_probs.append(float(_predict_prob(model, x_ins, device, class_idx)[0]))

    del_auc = float(np.trapezoid(del_probs, del_fracs))
    ins_auc = float(np.trapezoid(ins_probs, ins_fracs))
    return del_auc, ins_auc


# ---------------------------------------------------------------------------
# Ablation 2: Grad-CAM on ViT (non-native control)
# ---------------------------------------------------------------------------
def get_heatmap_gradcam_vit(model, image_tensor, device, class_idx=None):
    """Grad-CAM on ViT's last encoder layer (deliberately non-native)."""
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    target_layer = model.encoder.layers[-1]
    gc = GradCAM(model, target_layer)
    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    cam, pred_idx = gc(x, class_idx=class_idx)
    return cam.astype(np.float32), pred_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs/paper_runs.yaml')
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--output', default='outputs/ablation')
    ap.add_argument('--sample_seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    paper_runs = load_paper_runs(args.paper_runs)
    any_cfg_path = list(paper_runs.values())[0]['config']
    with open(any_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    data_cfg['augmentation']['enabled'] = False
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
    print(f"Sampled {len(sample_indices)} test indices")

    # ===== Ablation 1: Random baseline =====
    print("\n=== Ablation 1: Random-ranking baseline ===")
    random_results = []
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]  # seed-42 only for speed
        model, cfg = load_model_from_run(run_dir, device)
        model_name = cfg['model']['name']
        print(f"  [{fam}] {run_dir}")
        for k, idx in enumerate(sample_indices):
            x, y = test_ds[int(idx)]
            with torch.no_grad():
                logits = model(x.unsqueeze(0).to(device))
                pred = int(logits.argmax(1).item())
            d, ins = random_deletion_insertion(model, x, device, pred,
                                              n_steps=args.n_steps,
                                              seed=args.sample_seed + k)
            random_results.append({
                'family': fam, 'sample_idx': int(idx),
                'deletion_auc': d, 'insertion_auc': ins,
            })
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    df_rand = pd.DataFrame(random_results)
    df_rand.to_csv(os.path.join(args.output, 'random_baseline.csv'), index=False)
    print("\nRandom baseline (mean):")
    print(df_rand.groupby('family')[['deletion_auc', 'insertion_auc']].mean())

    # ===== Ablation 2: Grad-CAM on ViT (non-native control) =====
    print("\n=== Ablation 2: Grad-CAM on ViT (non-native control) ===")
    vit_gc_results = []
    vit_meta = paper_runs['vit_tiny']
    run_dir = vit_meta['runs'][0]
    model, cfg = load_model_from_run(run_dir, device)
    print(f"  [vit_tiny+gradcam] {run_dir}")
    for k, idx in enumerate(sample_indices):
        x, y = test_ds[int(idx)]
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device))
            pred = int(logits.argmax(1).item())
        heatmap, _ = get_heatmap_gradcam_vit(model, x, device, class_idx=pred)
        d = deletion_auc(model, x, heatmap, device, class_idx=pred,
                         n_steps=args.n_steps)
        ins = insertion_auc(model, x, heatmap, device, class_idx=pred,
                            n_steps=args.n_steps)
        vit_gc_results.append({
            'family': 'vit_tiny_gradcam', 'sample_idx': int(idx),
            'pred_label': pred, 'true_label': int(y.item()),
            'deletion_auc': d['auc'], 'insertion_auc': ins['auc'],
        })
        if (k + 1) % 50 == 0:
            print(f"    {k+1}/{len(sample_indices)}")
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    df_vit_gc = pd.DataFrame(vit_gc_results)
    df_vit_gc.to_csv(os.path.join(args.output, 'vit_gradcam.csv'), index=False)
    print(f"\nViT + Grad-CAM: Del={df_vit_gc['deletion_auc'].mean():.3f}, "
          f"Ins={df_vit_gc['insertion_auc'].mean():.3f}")

    # ===== Ablation 3: Commonly-correct subset =====
    print("\n=== Ablation 3: Commonly-correct subset ===")
    # Load existing per-sample metrics
    existing = pd.read_csv('outputs/interpretability/metrics_per_sample.csv')
    # For each sample, check if all 3 families predicted correctly (any seed)
    # Use seed-42 runs only for fair comparison
    seed42_runs = {fam: meta['runs'][0] for fam, meta in paper_runs.items()}
    seed42 = existing[existing['run_dir'].isin(seed42_runs.values())]

    # Pivot: for each sample_idx, check pred_label == true_label per family
    correct_by_sample = seed42.groupby('sample_idx').apply(
        lambda g: (g['pred_label'] == g['true_label']).all()
    )
    common_correct_idx = correct_by_sample[correct_by_sample].index.tolist()
    print(f"  Samples correct by all 3 families (seed 42): "
          f"{len(common_correct_idx)} / {seed42['sample_idx'].nunique()}")

    subset = seed42[seed42['sample_idx'].isin(common_correct_idx)]
    cc_summary = subset.groupby('family')[['deletion_auc', 'insertion_auc']].agg(
        ['mean', 'std'])
    cc_summary.to_csv(os.path.join(args.output, 'commonly_correct.csv'))
    print("\nCommonly-correct subset (mean ± std):")
    print(cc_summary)

    # ===== Ablation 4: Top-k sensitivity =====
    print("\n=== Ablation 4: Top-k sensitivity (5%, 10%, 20%) ===")
    topk_results = []
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]
        model, cfg = load_model_from_run(run_dir, device)
        model_name = cfg['model']['name']
        print(f"  [{fam}] {run_dir}")
        for k, idx in enumerate(sample_indices):
            x, y = test_ds[int(idx)]
            with torch.no_grad():
                logits = model(x.unsqueeze(0).to(device))
                pred = int(logits.argmax(1).item())
            heatmap, _ = get_heatmap(model, x, model_name, device,
                                     class_idx=pred)
            for topk in [0.05, 0.10, 0.20]:
                # Measure confidence drop when removing only top-k% pixels
                mask = heatmap_to_topk_mask(heatmap, ratio=topk)
                x_del = x.clone().float()
                if x_del.dim() == 2:
                    x_del = x_del.unsqueeze(0)
                rr, cc = np.where(mask)
                x_del[..., rr, cc] = 0.0
                p_orig = float(_predict_prob(model, x, device, pred)[0])
                p_del = float(_predict_prob(model, x_del, device, pred)[0])
                drop = p_orig - p_del  # higher drop = more faithful
                topk_results.append({
                    'family': fam, 'topk': topk, 'sample_idx': int(idx),
                    'conf_orig': p_orig, 'conf_after_del': p_del,
                    'conf_drop': drop,
                })
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    df_topk = pd.DataFrame(topk_results)
    df_topk.to_csv(os.path.join(args.output, 'topk_sensitivity.csv'), index=False)
    print("\nTop-k sensitivity (mean confidence drop after removing top-k%):")
    print(df_topk.groupby(['family', 'topk'])['conf_drop'].mean().unstack())

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("ALL ABLATION RESULTS SUMMARY")
    print("=" * 60)

    # Compare heatmap-based vs random
    existing_seed42 = seed42.groupby('family')[['deletion_auc', 'insertion_auc']].mean()
    rand_mean = df_rand.groupby('family')[['deletion_auc', 'insertion_auc']].mean()
    print("\n--- Heatmap vs Random baseline (seed 42) ---")
    comparison = existing_seed42.copy()
    comparison.columns = ['del_heatmap', 'ins_heatmap']
    comparison['del_random'] = rand_mean['deletion_auc']
    comparison['ins_random'] = rand_mean['insertion_auc']
    print(comparison)

    print(f"\n--- ViT: Rollout vs Grad-CAM (non-native control) ---")
    vit_rollout = seed42[seed42['family'] == 'vit_tiny'][['deletion_auc', 'insertion_auc']].mean()
    print(f"  Rollout:  Del={vit_rollout['deletion_auc']:.3f}, Ins={vit_rollout['insertion_auc']:.3f}")
    print(f"  Grad-CAM: Del={df_vit_gc['deletion_auc'].mean():.3f}, Ins={df_vit_gc['insertion_auc'].mean():.3f}")

    print(f"\nAll outputs saved to {args.output}/")


if __name__ == '__main__':
    main()
