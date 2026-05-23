"""Blur-fill perturbation baseline ablation.

Tests whether the WM-811K family ordering is robust to perturbation
baseline choice by replacing zero-fill with Gaussian blur-fill.

Usage:
    CUDA_VISIBLE_DEVICES=7 python ablation_blur_fill.py
"""
import os, json, yaml, time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from collections import defaultdict

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.interpretability import get_heatmap, _ordered_indices, _predict_prob

torch.set_num_threads(4)


def deletion_auc_blur(model, image_tensor, heatmap, device, class_idx=None,
                      n_steps=20, sigma=3.0):
    """Deletion AUC with blur-fill baseline instead of zero-fill."""
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    if class_idx is None:
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            class_idx = int(logits.argmax(1).item())

    # Create blurred version as baseline
    img_np = image_tensor.cpu().numpy()
    blurred_np = scipy.ndimage.gaussian_filter(img_np, sigma=(0, sigma, sigma))
    blurred = torch.from_numpy(blurred_np).float()

    order = _ordered_indices(heatmap)
    N = order.size
    step = max(1, N // n_steps)
    H, W = image_tensor.shape[-2], image_tensor.shape[-1]

    x_cur = image_tensor.clone().float()
    fractions = [0.0]
    probs = [float(_predict_prob(model, x_cur, device, class_idx)[0])]

    removed = 0
    for i in range(n_steps):
        end = min(removed + step, N)
        idxs = order[removed:end]
        rr, cc = np.unravel_index(idxs, (H, W))
        x_cur[..., rr, cc] = blurred[..., rr, cc]  # blur-fill instead of zero
        removed = end
        fractions.append(removed / N)
        probs.append(float(_predict_prob(model, x_cur, device, class_idx)[0]))

    fractions = np.asarray(fractions)
    probs = np.asarray(probs)
    auc = float(np.trapezoid(probs, fractions))
    return auc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs/paper_runs.yaml')
    ap.add_argument('--n_samples', type=int, default=198)
    ap.add_argument('--n_steps', type=int, default=20)
    ap.add_argument('--sigma', type=float, default=3.0)
    ap.add_argument('--output', default='outputs/ablation')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, sigma={args.sigma}")

    with open(args.paper_runs) as f:
        paper_runs = yaml.safe_load(f)['paper_runs']

    # Load dataset
    any_cfg = yaml.safe_load(open(list(paper_runs.values())[0]['config']))
    any_cfg['augmentation'] = {'enabled': False}
    loader = WM811KLoader(any_cfg)
    loader.load()
    loader.get_labeled_data()
    _, _, test_df = loader.split_data()
    test_df = test_df.reset_index(drop=True)
    from collections import Counter
    class_counts = Counter(int(l) for l in test_df['failurePatternType'])
    test_ds = WM811KDataset(test_df, any_cfg, class_counts)

    # Stratified sample (same as interpret_eval)
    num_classes = any_cfg['model']['num_classes']
    per_class = max(1, args.n_samples // num_classes)
    rng = np.random.default_rng(0)
    chosen = []
    for c in range(num_classes):
        idxs = test_df.index[test_df['failurePatternType'] == c].values
        take = min(per_class, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
    sample_indices = np.array(chosen[:args.n_samples])
    print(f"Evaluating {len(sample_indices)} samples")

    results = []
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]
        ckpt = os.path.join(run_dir, 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f"  [SKIP] {run_dir}")
            continue

        cfg = yaml.safe_load(open(os.path.join(run_dir, 'config.yaml')))
        model = build_model(cfg).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        model_name = cfg['model']['name']

        print(f"  [{fam}] {run_dir}")
        t0 = time.time()
        fam_aucs = []

        for k, idx in enumerate(sample_indices):
            x, y = test_ds[int(idx)]
            with torch.no_grad():
                logits = model(x.unsqueeze(0).to(device))
                pred = int(logits.argmax(1).item())

            heatmap, _ = get_heatmap(model, x, model_name, device, class_idx=pred)
            auc = deletion_auc_blur(model, x, heatmap, device, class_idx=pred,
                                    n_steps=args.n_steps, sigma=args.sigma)
            fam_aucs.append(auc)

            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{len(sample_indices)} ({time.time()-t0:.1f}s)")

        results.append({
            'family': fam,
            'display_name': meta['display_name'],
            'blur_del_mean': float(np.mean(fam_aucs)),
            'blur_del_std': float(np.std(fam_aucs)),
            'n_samples': len(fam_aucs),
        })
        print(f"    Blur-fill Del AUC: {np.mean(fam_aucs):.3f} ± {np.std(fam_aucs):.3f}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save and print comparison
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output, 'blur_fill_deletion.csv'), index=False)

    # Load zero-fill for comparison
    zero_fill = pd.read_csv('outputs/interpretability/summary.csv')
    print("\n=== Perturbation Baseline Comparison ===")
    print(f"{'Family':<16} {'Zero-fill Del':>13} {'Blur-fill Del':>13} {'Ordering preserved?'}")
    print("-" * 60)
    for _, row in df.iterrows():
        zf = zero_fill[zero_fill['family'] == row['family']]
        zf_val = zf['deletion_auc_mean'].values[0] if len(zf) > 0 else float('nan')
        print(f"{row['display_name']:<16} {zf_val:>10.3f}     {row['blur_del_mean']:>10.3f}")


if __name__ == '__main__':
    main()
