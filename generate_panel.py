"""Generate the qualitative interpretability panel (one sample per class × 3 families).

Picks the highest-confidence correctly-classified test sample per class,
then overlays heatmaps from each model family.

Usage:
    CUDA_VISIBLE_DEVICES=7 python generate_panel.py
    CUDA_VISIBLE_DEVICES=7 python generate_panel.py --output outputs/interpretability/qualitative_panel.png
"""
import argparse
import os

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from src.dataset import WM811KLoader, WM811KDataset
from src.models import build_model
from src.interpretability import get_heatmap
from utils.metrics import CLASS_NAMES


def load_paper_runs(path='configs/paper_runs.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)['paper_runs']


def pick_best_samples(model, test_ds, test_df, device, num_classes):
    """For each class, find the correctly-classified sample with highest softmax confidence."""
    model.eval()
    best = {}  # class -> (confidence, dataset_index)

    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            y = int(y.item())
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze()
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())

            if pred == y:  # correctly classified
                if y not in best or conf > best[y][0]:
                    best[y] = (conf, i)

            # Early exit once we have high-confidence samples for all classes
            if len(best) == num_classes and all(v[0] > 0.95 for v in best.values()):
                break

    return {c: idx for c, (_, idx) in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_runs', default='configs/paper_runs.yaml')
    ap.add_argument('--output', default='outputs/interpretability/qualitative_panel.png')
    ap.add_argument('--mode', default='overlay',
                    choices=['overlay', 'raw', 'both'],
                    help="'overlay' shows heatmap over wafer (default); "
                         "'raw' shows heatmap only (no wafer overlay); "
                         "'both' writes two files: --output and --output.replace('.png','_raw.png')")
    ap.add_argument('--reference_family', default='resnet18_cbam',
                    help='Family used to pick representative samples (highest-confidence correct)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paper_runs = load_paper_runs(args.paper_runs)

    # Load test set
    any_cfg_path = list(paper_runs.values())[0]['config']
    cfg = yaml.safe_load(open(any_cfg_path))
    cfg['augmentation'] = {'enabled': False}
    loader = WM811KLoader(cfg); loader.load(); loader.get_labeled_data()
    _, _, test_df = loader.split_data()
    test_df = test_df.reset_index(drop=True)
    class_counts = Counter(int(l) for l in test_df['failurePatternType'])
    test_ds = WM811KDataset(test_df, cfg, class_counts)
    num_classes = cfg['model']['num_classes']

    # Pick samples using reference family (first available run)
    ref_run = paper_runs[args.reference_family]['runs'][0]
    ref_cfg = yaml.safe_load(open(os.path.join(ref_run, 'config.yaml')))
    ref_model = build_model(ref_cfg).to(device)
    ref_model.load_state_dict(torch.load(
        os.path.join(ref_run, 'best_model.pth'), map_location=device, weights_only=False))
    print(f"Picking best samples using {args.reference_family}...")
    samples = pick_best_samples(ref_model, test_ds, test_df, device, num_classes)
    del ref_model; torch.cuda.empty_cache()

    classes = sorted(samples.keys())
    print(f"Found representative samples for {len(classes)}/{num_classes} classes")
    for c in classes:
        print(f"  {CLASS_NAMES[c]:12s} -> test index {samples[c]}")

    # Load one model per family
    fam_names = list(paper_runs.keys())
    loaded = {}
    for fam, meta in paper_runs.items():
        run_dir = meta['runs'][0]
        c = yaml.safe_load(open(os.path.join(run_dir, 'config.yaml')))
        model = build_model(c).to(device)
        model.load_state_dict(torch.load(
            os.path.join(run_dir, 'best_model.pth'), map_location=device, weights_only=False))
        model.eval()
        loaded[fam] = (model, c['model']['name'], meta.get('display_name', fam))

    # Plot
    def render_panel(mode, out_path):
        """mode: 'overlay' (default) or 'raw' (heatmap only, no wafer)."""
        n_rows = len(classes)
        n_cols = 1 + len(fam_names)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.5 * n_cols, 2.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for r, c in enumerate(classes):
            idx = samples[c]
            x, y = test_ds[idx]
            img = x.squeeze().numpy()

            axes[r, 0].imshow(img, cmap='inferno')
            axes[r, 0].set_ylabel(CLASS_NAMES[c], fontsize=10)
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            if r == 0:
                axes[r, 0].set_title('Input', fontsize=11, pad=12)

            for j, fam in enumerate(fam_names):
                model, m_name, display_name = loaded[fam]
                heatmap, _ = get_heatmap(model, x, m_name, device)
                # Per-heatmap min-max normalisation so each map uses
                # its own full dynamic range (fairer visual comparison).
                hmin, hmax = float(heatmap.min()), float(heatmap.max())
                if hmax > hmin:
                    heatmap_n = (heatmap - hmin) / (hmax - hmin)
                else:
                    heatmap_n = heatmap

                if mode == 'overlay':
                    axes[r, 1 + j].imshow(img, cmap='gray', alpha=0.5)
                    axes[r, 1 + j].imshow(heatmap_n, cmap='jet', alpha=0.5)
                else:  # 'raw'
                    axes[r, 1 + j].imshow(heatmap_n, cmap='jet', vmin=0, vmax=1)
                axes[r, 1 + j].set_xticks([]); axes[r, 1 + j].set_yticks([])
                if r == 0:
                    axes[r, 1 + j].set_title(display_name, fontsize=11, pad=12)

        title = 'Qualitative interpretability comparison (per class)'
        if mode == 'raw':
            title += ' — raw heatmaps (no wafer overlay)'
        fig.suptitle(title, fontsize=13, y=1.01)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {out_path}")

    if args.mode == 'both':
        render_panel('overlay', args.output)
        raw_out = args.output.replace('.png', '_raw.png')
        render_panel('raw', raw_out)
    else:
        render_panel(args.mode, args.output)

    # Cleanup
    for fam in list(loaded.keys()):
        del loaded[fam]


if __name__ == '__main__':
    main()
