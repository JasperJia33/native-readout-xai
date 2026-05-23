"""Interpretability protocol: unified heatmap API + faithfulness metrics.

Implements Deletion/Insertion AUC (Petsiuk et al. 2018, RISE) and Stability
(Alvarez-Melis & Jaakkola 2018) under a common protocol:

    * One heatmap per (model, sample) using a fixed method per family:
        - ResNet18+CBAM   -> Grad-CAM on backbone.layer4
        - DenseNet121     -> Grad-CAM on backbone.features.denseblock4
        - ViT-Tiny        -> Attention Rollout
    * Top-k fraction of pixels (default 10%) defines the "important" mask.
    * Deletion/Insertion perturbation: zero-fill.
    * Stability perturbations: rotate ±15°, translate ±3px, Gaussian noise σ=0.02.

All functions expect PyTorch tensors on the model's device.
"""
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage

from utils.visualize import GradCAM, attention_rollout


# ---------------------------------------------------------------------------
# Unified heatmap API
# ---------------------------------------------------------------------------

def _gradcam_target_layer(model, model_name):
    """Return the fixed Grad-CAM layer per family (per paper protocol).

    - resnet18_cbam: cbam4 (post-attention, after last residual stage)
    - densenet121:   features.denseblock4 (last dense block, pre-classifier)
    - swin_tiny:     final_stage (last Swin Transformer stage)
    """
    if 'resnet' in model_name.lower():
        if hasattr(model, 'cbam4'):
            return model.cbam4
        return model.backbone.layer4
    if 'densenet' in model_name.lower():
        return model.backbone.features.denseblock4
    if 'swin' in model_name.lower():
        return model.final_stage
    raise ValueError(f"No Grad-CAM target layer defined for {model_name}")


def get_heatmap(model, image_tensor, model_name, device, class_idx=None):
    """Return a (H, W) heatmap in [0, 1] for the given model and input.

    Args:
        model:        a torch model, already on `device`, in eval mode.
        image_tensor: (1, H, W) or (H, W) tensor, single sample.
        model_name:   family name; used to pick the method.
        device:       torch.device.
        class_idx:    if None, use the model's predicted class.

    Returns:
        heatmap: np.ndarray of shape (H, W), float, max-normalized to [0, 1].
        pred_idx: int, predicted class (or class_idx if provided).
    """
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    model.eval()

    if 'vit' in model_name.lower() and 'swin' not in model_name.lower():
        # ViT family -> Attention Rollout
        heatmap, pred_idx = attention_rollout(model, image_tensor, device)
        if heatmap is None:
            raise RuntimeError("attention_rollout returned None for ViT")
        return heatmap.astype(np.float32), pred_idx

    # CNN family -> Grad-CAM
    target_layer = _gradcam_target_layer(model, model_name)
    gc = GradCAM(model, target_layer)
    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    cam, pred_idx = gc(x, class_idx=class_idx)
    return cam.astype(np.float32), pred_idx


def heatmap_to_topk_mask(heatmap, ratio=0.10):
    """Return a boolean mask of the top-`ratio` fraction of pixels."""
    flat = heatmap.flatten()
    k = max(1, int(round(ratio * flat.size)))
    thresh = np.partition(flat, -k)[-k]
    return heatmap >= thresh


# ---------------------------------------------------------------------------
# Deletion / Insertion AUC  (RISE, Petsiuk et al. 2018)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_prob(model, x, device, class_idx):
    """Return the probability of class_idx for batched or single input."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    logits = model(x.to(device))
    probs = F.softmax(logits, dim=-1)
    return probs[:, class_idx].cpu().numpy()


def _ordered_indices(heatmap, rng=None):
    """Return flat pixel indices sorted by heatmap value, descending.

    When the heatmap is degenerate (all-equal, e.g., all zeros from a
    sign-cancelled Grad-CAM), ordering is arbitrary, so we jitter with
    a tiny random noise to randomize tie-breaking. The caller should
    check `heatmap.max() > 0` to decide whether the metric is trustworthy.
    """
    flat = heatmap.flatten().astype(np.float64)
    if flat.max() - flat.min() < 1e-8:
        rng = rng if rng is not None else np.random.default_rng(0)
        flat = flat + rng.normal(0, 1e-6, flat.shape)
    return np.argsort(-flat)


def deletion_auc(model, image_tensor, heatmap, device, class_idx=None, n_steps=20):
    """Deletion AUC. Progressively zero-out the top-ranked pixels.

    Lower AUC = more faithful explanation (removing the top pixels kills
    confidence fast).

    Returns dict with keys: auc (float), curve (np.ndarray, shape [n_steps+1]),
    fractions (np.ndarray, same shape), pred_idx (int).
    """
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    # Determine class_idx from initial prediction if not given
    if class_idx is None:
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            class_idx = int(logits.argmax(1).item())

    order = _ordered_indices(heatmap)
    N = order.size
    step = max(1, N // n_steps)
    fractions = []
    probs = []

    # Start from the full image
    x_cur = image_tensor.clone().float()
    H, W = x_cur.shape[-2], x_cur.shape[-1]

    fractions.append(0.0)
    probs.append(float(_predict_prob(model, x_cur, device, class_idx)[0]))

    removed = 0
    for i in range(n_steps):
        end = min(removed + step, N)
        idxs = order[removed:end]
        rr, cc = np.unravel_index(idxs, (H, W))
        x_cur[..., rr, cc] = 0.0
        removed = end
        fractions.append(removed / N)
        probs.append(float(_predict_prob(model, x_cur, device, class_idx)[0]))

    fractions = np.asarray(fractions)
    probs = np.asarray(probs)
    auc = float(np.trapezoid(probs, fractions))
    return {'auc': auc, 'curve': probs, 'fractions': fractions,
            'pred_idx': class_idx}


def insertion_auc(model, image_tensor, heatmap, device, class_idx=None, n_steps=20):
    """Insertion AUC. Start from a zero image; progressively reveal top-ranked
    pixels from the original image.

    Higher AUC = more faithful explanation (revealing the top pixels restores
    confidence fast).
    """
    model.eval()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    if class_idx is None:
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            class_idx = int(logits.argmax(1).item())

    order = _ordered_indices(heatmap)
    N = order.size
    step = max(1, N // n_steps)
    fractions = []
    probs = []

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    x_cur = torch.zeros_like(image_tensor, dtype=torch.float32)
    x_orig = image_tensor.float()

    fractions.append(0.0)
    probs.append(float(_predict_prob(model, x_cur, device, class_idx)[0]))

    added = 0
    for i in range(n_steps):
        end = min(added + step, N)
        idxs = order[added:end]
        rr, cc = np.unravel_index(idxs, (H, W))
        x_cur[..., rr, cc] = x_orig[..., rr, cc]
        added = end
        fractions.append(added / N)
        probs.append(float(_predict_prob(model, x_cur, device, class_idx)[0]))

    fractions = np.asarray(fractions)
    probs = np.asarray(probs)
    auc = float(np.trapezoid(probs, fractions))
    return {'auc': auc, 'curve': probs, 'fractions': fractions,
            'pred_idx': class_idx}


# ---------------------------------------------------------------------------
# Stability (Alvarez-Melis & Jaakkola 2018)
# ---------------------------------------------------------------------------

def _semantic_augment(image_np, rng):
    """Apply a single semantics-preserving perturbation.

    Sampled uniformly: rotation in [-15, 15], translation in [-3, 3] px,
    and additive Gaussian noise sigma=0.02 on top.
    """
    angle = rng.uniform(-15, 15)
    dx = rng.integers(-3, 4)   # -3..3 inclusive
    dy = rng.integers(-3, 4)
    # Rotate (order=0: nearest-neighbor to keep binary structure)
    out = scipy.ndimage.rotate(image_np, angle, reshape=False, order=0,
                               mode='constant', cval=0.0)
    # Translate
    out = scipy.ndimage.shift(out, (dy, dx), order=0,
                              mode='constant', cval=0.0)
    # Gaussian noise
    noise = rng.normal(0, 0.02, out.shape).astype(np.float32)
    out = np.clip(out + noise, 0.0, 1.0)
    return out.astype(np.float32)


def _cosine_similarity_on_wafer(a, b, valid_mask):
    """Cosine similarity restricted to the wafer region."""
    av = a[valid_mask].flatten()
    bv = b[valid_mask].flatten()
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def stability_score(model, image_tensor, model_name, device,
                    n_augs=5, seed=0, class_idx=None):
    """Stability: mean cosine similarity between the heatmap on the original
    image and heatmaps on `n_augs` semantics-preserving perturbations.

    Returns dict with keys: mean (float in [0,1]), per_aug (list of floats).
    """
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    rng = np.random.default_rng(seed)
    orig_np = image_tensor.squeeze().cpu().numpy().astype(np.float32)
    valid_mask = orig_np > 0   # wafer region (non-background)

    # Original heatmap
    h_orig, pred_idx = get_heatmap(model, image_tensor, model_name, device,
                                   class_idx=class_idx)

    sims = []
    for _ in range(n_augs):
        aug_np = _semantic_augment(orig_np, rng)
        aug_tensor = torch.tensor(aug_np).unsqueeze(0)
        h_aug, _ = get_heatmap(model, aug_tensor, model_name, device,
                               class_idx=pred_idx)  # fix class for fair compare
        sims.append(_cosine_similarity_on_wafer(h_orig, h_aug, valid_mask))

    return {'mean': float(np.mean(sims)),
            'per_aug': [float(s) for s in sims],
            'pred_idx': int(pred_idx)}
