"""Final-layer CLS attention ablation for ViT-Tiny (all 3 seeds).

Evaluates final-layer CLS-to-patch attention as a diagnostic ablation
to separate native attention access from multi-layer rollout depth.

Usage:
    # WM-811K (all 3 seeds):
    CUDA_VISIBLE_DEVICES=7 python ablation_final_layer_attn.py

    # WM-811K single seed:
    CUDA_VISIBLE_DEVICES=7 python ablation_final_layer_attn.py --seed 42

    # MVTec AD (seed 42 only):
    CUDA_VISIBLE_DEVICES=7 python ablation_final_layer_attn.py --dataset mvtec
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import argparse
import json
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
import torch.nn.functional as F
import yaml
from collections import Counter

from src.models import build_model
from src.dataset import WM811KLoader, WM811KDataset
from src.interpretability import deletion_auc, insertion_auc


SEED_RUN_MAP = {
    42: 'outputs/run0_vit_tiny_v3',
    123: 'outputs/vit_tiny_v3_seed123',
    456: 'outputs/vit_tiny_v3_seed456',
}

MVTEC_RUN = 'outputs_mvtec/vit_tiny_pretrained_seed42'


def final_layer_cls_attention(model, image_tensor, device):
    """Extract final-layer CLS-to-patch attention (averaged over heads).
    Works with nn.TransformerEncoder (WM-811K ViT)."""
    x = image_tensor.unsqueeze(0).to(device)
    captured = []
    layer = model.encoder.layers[-1]
    sa = layer.self_attn
    orig_fwd = sa.forward

    def patched(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True
        out, attn_w = orig_fwd(*args, **kwargs)
        captured.append(attn_w.detach())
        return out, attn_w

    sa.forward = patched
    with torch.no_grad():
        out = model(x)
    sa.forward = orig_fwd
    pred_idx = out.argmax(1).item()

    if not captured:
        return None, pred_idx

    # CLS row (index 0) attending to patch tokens (indices 1:)
    cls_attn = captured[0][0, 0, 1:].cpu().numpy()
    grid = int(cls_attn.shape[0] ** 0.5)
    heatmap = cls_attn.reshape(grid, grid)

    img_size = image_tensor.shape[-1]
    heatmap = F.interpolate(
        torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, pred_idx


def final_layer_cls_attention_timm(model, image_tensor, device):
    """Extract final-layer CLS-to-patch attention for timm/DeiT models (MVTec)."""
    x = image_tensor.unsqueeze(0).to(device)
    captured = []
    block = model.encoder[-1]  # last timm Block
    attn_mod = block.attn

    def hook_fn(module, input, output):
        # Recompute attention weights from qkv
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads,
                                           module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        # Average over heads: [B, N, N]
        captured.append(attn.mean(dim=1).detach())

    handle = attn_mod.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(x)
    handle.remove()
    pred_idx = out.argmax(1).item()

    if not captured:
        return None, pred_idx

    cls_attn = captured[0][0, 0, 1:].cpu().numpy()
    grid = int(cls_attn.shape[0] ** 0.5)
    heatmap = cls_attn.reshape(grid, grid)

    img_size = image_tensor.shape[-1]
    heatmap = F.interpolate(
        torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, pred_idx


def get_stratified_indices(test_df, n_per_class=22, seed=42):
    """Same stratified sampling as interpret_eval.py."""
    rng = np.random.default_rng(seed)
    labels = np.array([int(test_df.iloc[i]['failurePatternType'])
                       for i in range(len(test_df))])
    indices = []
    for c in range(9):
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, min(n_per_class, len(class_idx)),
                            replace=False)
        indices.extend(chosen.tolist())
    return indices


def evaluate_seed(seed, test_ds, indices, device):
    """Run final-layer attention evaluation for one seed."""
    run_dir = SEED_RUN_MAP[seed]
    model_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.isfile(model_path):
        print(f"  [SKIP] {model_path} not found")
        return None

    vit_cfg = yaml.safe_load(open('configs/vit_tiny_v3.yaml'))
    model = build_model(vit_cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    del_aucs, ins_aucs = [], []
    for i, idx in enumerate(indices):
        img, label = test_ds[idx]
        img_t = img.to(device)
        heatmap, pred = final_layer_cls_attention(model, img_t, device)
        if heatmap is None:
            continue
        d = deletion_auc(model, img_t, heatmap, device,
                         class_idx=pred, n_steps=20)['auc']
        ins = insertion_auc(model, img_t, heatmap, device,
                            class_idx=pred, n_steps=20)['auc']
        del_aucs.append(d)
        ins_aucs.append(ins)
        if (i + 1) % 50 == 0:
            print(f"    seed {seed}: {i+1}/{len(indices)} "
                  f"Del={np.mean(del_aucs):.3f} Ins={np.mean(ins_aucs):.3f}")

    return {
        'method': 'final_layer_cls_attention',
        'model': 'vit_tiny',
        'seed': seed,
        'run_dir': run_dir,
        'n_samples': len(del_aucs),
        'deletion_auc_mean': float(np.mean(del_aucs)),
        'deletion_auc_std': float(np.std(del_aucs)),
        'insertion_auc_mean': float(np.mean(ins_aucs)),
        'insertion_auc_std': float(np.std(ins_aucs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Run single seed (42/123/456). Default: all.')
    parser.add_argument('--dataset', default='wm811k',
                        choices=['wm811k', 'mvtec'],
                        help='Dataset to evaluate on.')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = ('outputs/ablation' if args.dataset == 'wm811k'
                       else 'outputs_mvtec/ablation')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.dataset == 'mvtec':
        run_mvtec(args, device)
    else:
        run_wm811k(args, device)


def run_mvtec(args, device):
    """Run final-layer CLS attention on MVTec AD (seed 42 only)."""
    from src.dataset_mvtec import MVTecADDataset, load_mvtec_samples
    from src.models_mvtec import build_mvtec_model

    cfg = yaml.safe_load(open('configs_mvtec/vit_tiny_pretrained_mvtec.yaml'))
    data_root = cfg['data']['path']

    _, test_samples, _, _ = load_mvtec_samples(data_root)
    defective = [s for s in test_samples if s['defect']['label'] != 'good']
    rng = np.random.default_rng(42)
    n = min(200, len(defective))
    indices = rng.choice(len(defective), size=n, replace=False)
    eval_samples = [defective[i] for i in indices]
    print(f"Evaluating {n} defective MVTec samples")

    eval_cfg = {**cfg, 'augmentation': {'enabled': False}}
    eval_ds = MVTecADDataset(eval_samples, data_root, eval_cfg, augmentation=False)

    # Load model
    model = build_mvtec_model(cfg).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MVTEC_RUN, 'best_model.pth'),
        map_location=device, weights_only=True))
    model.eval()

    del_aucs, ins_aucs = [], []
    for i in range(len(eval_ds)):
        img, label = eval_ds[i]
        img_t = img.to(device)
        heatmap, pred = final_layer_cls_attention_timm(model, img_t, device)
        if heatmap is None:
            continue
        d = deletion_auc(model, img_t, heatmap, device,
                         class_idx=pred, n_steps=20)['auc']
        ins = insertion_auc(model, img_t, heatmap, device,
                            class_idx=pred, n_steps=20)['auc']
        del_aucs.append(d)
        ins_aucs.append(ins)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(eval_ds)}: "
                  f"Del={np.mean(del_aucs):.3f} Ins={np.mean(ins_aucs):.3f}")

    result = {
        'method': 'final_layer_cls_attention',
        'model': 'vit_tiny_pretrained',
        'dataset': 'mvtec',
        'seed': 42,
        'n_samples': len(del_aucs),
        'deletion_auc_mean': float(np.mean(del_aucs)),
        'deletion_auc_std': float(np.std(del_aucs)),
        'insertion_auc_mean': float(np.mean(ins_aucs)),
        'insertion_auc_std': float(np.std(ins_aucs)),
    }

    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, 'final_layer_attention_mvtec.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL-LAYER CLS ATTENTION — MVTec AD")
    print(f"{'='*60}")
    print(f"  Final-layer CLS attn:  Del {result['deletion_auc_mean']:.3f}")
    print(f"  Full Rollout (paper):  Del 0.770")
    print(f"  RISE (paper):          Del 0.656")
    print(f"{'='*60}")
    print(f"Saved to {out_file}")


def run_wm811k(args, device):
    """Run final-layer CLS attention on WM-811K."""
    cfg = yaml.safe_load(open('configs/resnet18_cbam.yaml'))
    cfg['augmentation']['enabled'] = False
    loader = WM811KLoader(cfg)
    loader.load()
    loader.get_labeled_data()
    _, _, test_df = loader.split_data()
    test_df = test_df.reset_index(drop=True)
    class_counts = Counter(int(l) for l in test_df['failurePatternType'])
    test_ds = WM811KDataset(test_df, cfg, class_counts)

    indices = get_stratified_indices(test_df, n_per_class=22, seed=42)
    print(f"Evaluating {len(indices)} samples")

    seeds = [args.seed] if args.seed else [42, 123, 456]
    results = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        r = evaluate_seed(seed, test_ds, indices, device)
        if r:
            results.append(r)
            print(f"  Del: {r['deletion_auc_mean']:.4f} ± "
                  f"{r['deletion_auc_std']:.4f}")
            print(f"  Ins: {r['insertion_auc_mean']:.4f} ± "
                  f"{r['insertion_auc_std']:.4f}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, 'final_layer_attention.json')
    # If single seed, append to existing file
    if args.seed and os.path.isfile(out_file):
        try:
            existing = json.load(open(out_file))
            if isinstance(existing, list):
                existing = [r for r in existing if r['seed'] != args.seed]
                existing.extend(results)
                results = existing
            elif existing.get('seed') != args.seed:
                results = [existing] + results
        except (json.JSONDecodeError, KeyError):
            pass  # overwrite corrupt file

    with open(out_file, 'w') as f:
        json.dump(results if len(results) > 1 else results[0], f, indent=2)
    print(f"\nSaved to {out_file}")

    # Print comparison
    if results:
        mean_del = np.mean([r['deletion_auc_mean'] for r in results])
        mean_ins = np.mean([r['insertion_auc_mean'] for r in results])
        print(f"\n{'='*60}")
        print(f"FINAL-LAYER CLS ATTENTION ABLATION RESULTS")
        print(f"{'='*60}")
        print(f"  Final-layer CLS attn:  Del {mean_del:.3f}, Ins {mean_ins:.3f}")
        print(f"  Full Rollout (paper):  Del 0.211, Ins 0.689")
        print(f"  Grad-CAM on ViT (S9):  Del 0.492")
        print(f"  CNN Grad-CAM (paper):  Del 0.495–0.525")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
