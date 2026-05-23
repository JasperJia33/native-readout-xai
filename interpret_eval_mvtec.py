"""Interpretability evaluation for MVTec AD — native methods + RISE + IoU.

Runs Deletion/Insertion AUC with native methods (Grad-CAM, Rollout) and RISE,
plus ground-truth mask IoU for defective samples.

Usage:
    python interpret_eval_mvtec.py
    python interpret_eval_mvtec.py --n_masks 4000 --mask_res 8
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset_mvtec import MVTecADDataset, load_mvtec_samples
from src.models_mvtec import build_mvtec_model
from src.interpretability import (
    deletion_auc, insertion_auc, _gradcam_target_layer,
)
from utils.visualize import GradCAM, attention_rollout
from rise_eval import rise_heatmap


# ---------------------------------------------------------------------------
# Heatmap generation for RGB models
# ---------------------------------------------------------------------------

def get_heatmap_mvtec(model, image_tensor, model_name, device, class_idx=None):
    """Heatmap for MVTec RGB models. Same logic as src/interpretability.get_heatmap
    but handles 3-channel input."""
    if image_tensor.dim() == 3:  # (3, H, W)
        pass
    elif image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    model.eval()

    if 'vit' in model_name.lower() and 'swin' not in model_name.lower():
        heatmap, pred_idx = attention_rollout(model, image_tensor, device)
        if heatmap is None:
            raise RuntimeError("attention_rollout returned None")
        return heatmap.astype(np.float32), pred_idx

    # CNN / Swin -> Grad-CAM
    if 'resnet' in model_name.lower():
        target_layer = model.cbam4 if hasattr(model, 'cbam4') else model.backbone.layer4
    elif 'densenet' in model_name.lower():
        target_layer = model.backbone.features.denseblock4
    elif 'swin' in model_name.lower():
        target_layer = model.final_stage
    else:
        raise ValueError(f"No Grad-CAM target for {model_name}")

    gc = GradCAM(model, target_layer)
    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    cam, pred_idx = gc(x, class_idx=class_idx)
    return cam.astype(np.float32), pred_idx


def compute_iou(heatmap, mask, threshold=0.5):
    """IoU between thresholded heatmap and ground-truth binary mask."""
    pred = (heatmap >= threshold).astype(np.float32)
    gt = (mask > 0.5).astype(np.float32)
    intersection = (pred * gt).sum()
    union = ((pred + gt) > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model_from_run(run_dir, device):
    cfg_path = os.path.join(run_dir, 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_mvtec_model(cfg).to(device)
    state = torch.load(os.path.join(run_dir, 'best_model.pth'),
                       map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs_mvtec/paper_runs_mvtec.yaml')
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--n_masks', type=int, default=4000)
    ap.add_argument('--mask_res', type=int, default=8)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--output', default='outputs_mvtec/interpretability')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load paper runs
    with open(args.paper_runs) as f:
        paper_runs = yaml.safe_load(f)['paper_runs']

    # Load dataset
    any_cfg_path = list(paper_runs.values())[0]['config']
    with open(any_cfg_path) as f:
        cfg = yaml.safe_load(f)
    data_root = cfg['data']['path']
    _, test_samples, _, test_indices = load_mvtec_samples(data_root)

    # Filter to defective samples only (they have masks for IoU)
    defective_samples = [s for s in test_samples if s['defect']['label'] != 'good']
    print(f"Defective test samples: {len(defective_samples)}")

    # Subsample if needed
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(defective_samples))
    indices = rng.choice(len(defective_samples), size=n, replace=False)
    eval_samples = [defective_samples[i] for i in indices]
    print(f"Evaluating {n} defective samples")

    eval_cfg = {**cfg, 'augmentation': {'enabled': False}}
    eval_ds = MVTecADDataset(eval_samples, data_root, eval_cfg, augmentation=False)

    # Evaluate each family (seed 42 only)
    all_results = []
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]  # seed 42
        if not os.path.isfile(os.path.join(run_dir, 'best_model.pth')):
            print(f"  [SKIP] {run_dir}")
            continue

        print(f"\n  [{fam}] Loading {run_dir}...")
        model, run_cfg = load_model_from_run(run_dir, device)
        model_name = run_cfg['model']['name']
        t0 = time.time()

        for k in range(len(eval_ds)):
            x, y = eval_ds[k]

            # Native heatmap
            native_hmap, pred = get_heatmap_mvtec(model, x, model_name, device)

            # RISE heatmap
            rise_hmap, _ = rise_heatmap(
                model, x, device,
                n_masks=args.n_masks, mask_res=args.mask_res,
                batch_size=args.batch_size, seed=args.seed + k,
                class_idx=pred
            )

            # Deletion/Insertion for both
            d_native = deletion_auc(model, x, native_hmap, device, class_idx=pred, n_steps=args.n_steps)
            i_native = insertion_auc(model, x, native_hmap, device, class_idx=pred, n_steps=args.n_steps)
            d_rise = deletion_auc(model, x, rise_hmap, device, class_idx=pred, n_steps=args.n_steps)
            i_rise = insertion_auc(model, x, rise_hmap, device, class_idx=pred, n_steps=args.n_steps)

            # IoU with ground-truth mask
            mask = eval_ds.get_mask(k)
            iou_native = compute_iou(native_hmap, mask) if mask is not None else None
            iou_rise = compute_iou(rise_hmap, mask) if mask is not None else None

            all_results.append({
                'family': fam,
                'sample_idx': k,
                'category': eval_samples[k]['category']['label'],
                'defect_type': eval_samples[k]['defect']['label'],
                'pred_label': pred,
                'true_label': int(y.item()),
                'native_del': d_native['auc'],
                'native_ins': i_native['auc'],
                'rise_del': d_rise['auc'],
                'rise_ins': i_rise['auc'],
                'native_iou': iou_native,
                'rise_iou': iou_rise,
            })

            if (k + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"    [{fam}] {k+1}/{len(eval_ds)} ({dt:.1f}s)")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output, 'mvtec_interp_metrics.csv'), index=False)

    # Summary
    print("\n\n=== MVTec AD Interpretability Summary ===")
    print(f"{'Family':<18} {'Native Del':>10} {'RISE Del':>10} {'Native Ins':>10} {'RISE Ins':>10} {'Native IoU':>10} {'RISE IoU':>10}")
    print("-" * 80)
    for fam in paper_runs.keys():
        sub = df[df['family'] == fam]
        if len(sub) == 0:
            continue
        nd = sub['native_del'].mean()
        rd = sub['rise_del'].mean()
        ni = sub['native_ins'].mean()
        ri = sub['rise_ins'].mean()
        n_iou = sub['native_iou'].dropna().mean() if sub['native_iou'].notna().any() else 0
        r_iou = sub['rise_iou'].dropna().mean() if sub['rise_iou'].notna().any() else 0
        print(f"{fam:<18} {nd:>10.3f} {rd:>10.3f} {ni:>10.3f} {ri:>10.3f} {n_iou:>10.3f} {r_iou:>10.3f}")

    # Save summary
    summary_rows = []
    for fam in paper_runs.keys():
        sub = df[df['family'] == fam]
        if len(sub) == 0:
            continue
        summary_rows.append({
            'family': fam,
            'native_del_mean': sub['native_del'].mean(),
            'native_del_std': sub['native_del'].std(),
            'rise_del_mean': sub['rise_del'].mean(),
            'rise_del_std': sub['rise_del'].std(),
            'native_ins_mean': sub['native_ins'].mean(),
            'native_ins_std': sub['native_ins'].std(),
            'rise_ins_mean': sub['rise_ins'].mean(),
            'rise_ins_std': sub['rise_ins'].std(),
            'native_iou_mean': sub['native_iou'].dropna().mean(),
            'rise_iou_mean': sub['rise_iou'].dropna().mean(),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output, 'mvtec_summary.csv'), index=False)
    print(f"\nAll outputs -> {args.output}/")


if __name__ == '__main__':
    main()
