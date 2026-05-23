"""RISE (Randomized Input Sampling for Explanation) evaluation.

Applies RISE — a model-agnostic explanation method — to all paper-run families
and computes Deletion/Insertion AUC using the same protocol as interpret_eval.py.

Purpose: Test whether the ViT faithfulness advantage is due to the native-readout
explainer (Attention Rollout) or the architecture itself. If RISE equalizes
families, the native-readout hypothesis is supported. If RISE preserves the
ViT advantage, the effect is architectural.

Usage:
    python rise_eval.py
    python rise_eval.py --n_masks 4000 --mask_res 8 --n_samples 198

Outputs (under outputs/rise/):
    rise_metrics.csv        per-sample Del/Ins AUC
    rise_summary.csv        mean ± std per family
    rise_comparison.csv     side-by-side: native vs RISE per family
"""
import argparse
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.interpretability import deletion_auc, insertion_auc
from utils.metrics import CLASS_NAMES


# ---------------------------------------------------------------------------
# RISE implementation
# ---------------------------------------------------------------------------

def generate_masks(n_masks, input_size, mask_res, rng, p=0.5):
    """Generate random binary masks upsampled to input_size.

    Args:
        n_masks: number of masks
        input_size: (H, W) of the input image
        mask_res: resolution of the small random grid (e.g., 8 means 8x8)
        rng: numpy random generator
        p: probability of each cell being 1 (revealed)

    Returns:
        masks: np.ndarray of shape (n_masks, H, W), float32 in [0, 1]
    """
    H, W = input_size
    # Random binary grids at low resolution
    small = (rng.random((n_masks, 1, mask_res, mask_res)) < p).astype(np.float32)
    # Upsample to input size with bilinear interpolation
    small_t = torch.from_numpy(small)
    up = F.interpolate(small_t, size=(H, W), mode='bilinear', align_corners=False)
    masks = up.squeeze(1).numpy()  # (n_masks, H, W)
    return masks


@torch.no_grad()
def rise_heatmap(model, image_tensor, device, n_masks=4000, mask_res=8,
                 batch_size=128, seed=42, class_idx=None):
    """Compute RISE saliency map for a single image.

    Args:
        model: torch model in eval mode
        image_tensor: (1, H, W) or (H, W) tensor
        device: torch device
        n_masks: number of random masks
        mask_res: low-res grid size
        batch_size: forward-pass batch size
        seed: random seed for mask generation
        class_idx: target class (None = use predicted class)

    Returns:
        heatmap: (H, W) numpy array, normalized to [0, 1]
        pred_idx: predicted class index
    """
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    x = image_tensor.float().to(device)  # (1, H, W)

    # Get predicted class
    if class_idx is None:
        logits = model(x.unsqueeze(0))
        class_idx = int(logits.argmax(1).item())

    # Generate masks
    rng = np.random.default_rng(seed)
    masks = generate_masks(n_masks, (H, W), mask_res, rng, p=0.5)

    # Weighted accumulation
    sal = np.zeros((H, W), dtype=np.float64)
    n_total = np.zeros((H, W), dtype=np.float64)

    for i in range(0, n_masks, batch_size):
        batch_masks = masks[i:i + batch_size]  # (B, H, W)
        B = batch_masks.shape[0]
        # Mask the image: element-wise multiply
        mask_t = torch.from_numpy(batch_masks).float().to(device)  # (B, H, W)
        masked_imgs = x.expand(B, -1, -1, -1) * mask_t.unsqueeze(1)  # (B, 1, H, W)

        logits = model(masked_imgs)
        probs = F.softmax(logits, dim=-1)[:, class_idx].cpu().numpy()  # (B,)

        # Accumulate: weight each mask by the probability it produced
        for j in range(B):
            sal += probs[j] * batch_masks[j]
            n_total += batch_masks[j]

    # Normalize by expected mask value at each pixel
    n_total = np.maximum(n_total, 1e-8)
    sal = sal / n_total

    # Min-max normalize to [0, 1]
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max - sal_min > 1e-10:
        sal = (sal - sal_min) / (sal_max - sal_min)
    else:
        sal = np.zeros_like(sal)

    return sal.astype(np.float32), class_idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_paper_runs(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc['paper_runs']


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs/paper_runs.yaml')
    ap.add_argument('--n_samples', type=int, default=198,
                    help='Number of test samples (22 per class × 9)')
    ap.add_argument('--n_masks', type=int, default=4000)
    ap.add_argument('--mask_res', type=int, default=8)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--output', default='outputs/rise')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"RISE config: n_masks={args.n_masks}, mask_res={args.mask_res}, "
          f"n_steps={args.n_steps}")

    # Load test set
    paper_runs = load_paper_runs(args.paper_runs)
    any_cfg_path = list(paper_runs.values())[0]['config']
    with open(any_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
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
                                       seed=0)  # same seed as interpret_eval
    print(f"Sampled {len(sample_indices)} test indices")

    # Evaluate each family (seed 42 only for speed — same as ablation protocol)
    all_results = []
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]  # seed 42
        if not os.path.isfile(os.path.join(run_dir, 'best_model.pth')):
            print(f"  [SKIP] {run_dir} missing best_model.pth")
            continue

        print(f"\n  [{fam}] Loading {run_dir}...")
        model, cfg = load_model_from_run(run_dir, device)
        t0 = time.time()

        for k, idx in enumerate(sample_indices):
            x, y = test_ds[int(idx)]
            y = int(y.item())

            # RISE heatmap
            heatmap, pred = rise_heatmap(
                model, x, device,
                n_masks=args.n_masks, mask_res=args.mask_res,
                batch_size=args.batch_size, seed=args.seed + k,
                class_idx=None
            )

            # Deletion/Insertion using same protocol
            d = deletion_auc(model, x, heatmap, device, class_idx=pred,
                             n_steps=args.n_steps)
            ins = insertion_auc(model, x, heatmap, device, class_idx=pred,
                                n_steps=args.n_steps)

            all_results.append({
                'family': fam,
                'sample_idx': int(idx),
                'true_label': y,
                'pred_label': pred,
                'deletion_auc': d['auc'],
                'insertion_auc': ins['auc'],
            })

            if (k + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"    [{fam}] {k+1}/{len(sample_indices)} "
                      f"({dt:.1f}s, {dt/(k+1):.2f}s/sample)")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output, 'rise_metrics.csv'), index=False)

    # Summary
    summary_rows = []
    for fam in df['family'].unique():
        sub = df[df['family'] == fam]
        summary_rows.append({
            'family': fam,
            'method': 'RISE',
            'n_samples': len(sub),
            'deletion_auc_mean': sub['deletion_auc'].mean(),
            'deletion_auc_std': sub['deletion_auc'].std(),
            'insertion_auc_mean': sub['insertion_auc'].mean(),
            'insertion_auc_std': sub['insertion_auc'].std(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.output, 'rise_summary.csv'), index=False)

    print("\n\n=== RISE Evaluation Summary ===")
    print(summary.to_string(index=False))

    # Load native method results for comparison
    native_path = 'outputs/interpretability/metrics_per_sample.csv'
    if os.path.isfile(native_path):
        native_df = pd.read_csv(native_path)
        # Filter to seed-42 runs only for fair comparison
        seed42_runs = {fam: meta['runs'][0] for fam, meta in paper_runs.items()}
        native_s42 = native_df[native_df['run_dir'].isin(seed42_runs.values())]

        print("\n\n=== Comparison: Native Method vs RISE (seed 42) ===")
        print(f"{'Family':<18} {'Native Del':>12} {'RISE Del':>12} {'Native Ins':>12} {'RISE Ins':>12}")
        print("-" * 70)
        for fam in paper_runs.keys():
            n_sub = native_s42[native_s42['family'] == fam]
            r_sub = df[df['family'] == fam]
            if len(n_sub) == 0 or len(r_sub) == 0:
                continue
            print(f"{fam:<18} {n_sub['deletion_auc'].mean():>12.3f} "
                  f"{r_sub['deletion_auc'].mean():>12.3f} "
                  f"{n_sub['insertion_auc'].mean():>12.3f} "
                  f"{r_sub['insertion_auc'].mean():>12.3f}")

        # Save comparison
        comp_rows = []
        for fam in paper_runs.keys():
            n_sub = native_s42[native_s42['family'] == fam]
            r_sub = df[df['family'] == fam]
            if len(n_sub) == 0 or len(r_sub) == 0:
                continue
            comp_rows.append({
                'family': fam,
                'native_del_mean': n_sub['deletion_auc'].mean(),
                'native_del_std': n_sub['deletion_auc'].std(),
                'rise_del_mean': r_sub['deletion_auc'].mean(),
                'rise_del_std': r_sub['deletion_auc'].std(),
                'native_ins_mean': n_sub['insertion_auc'].mean(),
                'native_ins_std': n_sub['insertion_auc'].std(),
                'rise_ins_mean': r_sub['insertion_auc'].mean(),
                'rise_ins_std': r_sub['insertion_auc'].std(),
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(os.path.join(args.output, 'rise_comparison.csv'), index=False)

    print(f"\nAll outputs -> {args.output}/")


if __name__ == '__main__':
    main()
