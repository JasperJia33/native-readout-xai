import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

from utils.metrics import CLASS_NAMES


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------

def plot_class_distribution(df, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    counts = df['failurePatternType'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [class_names[int(i)] if int(i) < len(class_names) else str(int(i)) for i in counts.index]
    sns.barplot(x=labels, y=counts.values, ax=ax, palette='viridis')
    ax.set_title('Class Distribution'); ax.set_ylabel('Count'); ax.set_xlabel('Class')
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha='center', fontsize=8)
    plt.tight_layout(); _save_or_show(fig, save_path)


def plot_sample_wafers(df, n_per_class=3, target_size=64, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    classes = sorted(df['failurePatternType'].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2, len(classes) * 2))
    if len(classes) == 1:
        axes = axes[np.newaxis, :]
    for r, cls in enumerate(classes):
        subset = df[df['failurePatternType'] == cls]
        samples = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        name = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
        for c in range(n_per_class):
            ax = axes[r, c]
            if c < len(samples):
                wm = samples.iloc[c]['waferMap']
                if isinstance(wm, list):
                    wm = np.array(wm)
                ax.imshow(wm, cmap='inferno', interpolation='nearest')
            ax.axis('off')
            if c == 0:
                ax.set_title(name, fontsize=9)
    plt.suptitle('Sample Wafer Maps per Class', fontsize=12)
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'], label='Val')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    axes[2].plot(epochs, history['val_balanced_acc'], label='Balanced Acc', color='green')
    axes[2].plot(epochs, history['val_f1_macro'], label='F1 Macro', color='orange')
    axes[2].set_title('Validation Metrics'); axes[2].set_xlabel('Epoch'); axes[2].legend()
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    from sklearn.metrics import confusion_matrix
    class_names = class_names or CLASS_NAMES
    n = len(set(y_true)); names = class_names[:n]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
    plt.tight_layout(); _save_or_show(fig, save_path)


def plot_per_class_f1(per_class_f1, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    names = class_names[:len(per_class_f1)]
    colors = ['#d32f2f' if f < 0.5 else '#ff9800' if f < 0.8 else '#4caf50' for f in per_class_f1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    ax.barh(names, per_class_f1, color=colors)
    ax.set_xlim(0, 1); ax.set_xlabel('F1 Score'); ax.set_title('Per-Class F1 Score')
    for i, v in enumerate(per_class_f1):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout(); _save_or_show(fig, save_path)


def plot_misclassified(model, dataset, device, class_names=None, n=12, save_path=None):
    class_names = class_names or CLASS_NAMES
    model.eval()
    mis = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
            if pred != y.item():
                mis.append((x.squeeze().numpy(), y.item(), pred))
            if len(mis) >= n:
                break
    if not mis:
        return
    cols = min(4, len(mis))
    rows = (len(mis) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i, (img, true, pred) in enumerate(mis):
        axes[i].imshow(img, cmap='inferno')
        tn = class_names[true] if true < len(class_names) else str(true)
        pn = class_names[pred] if pred < len(class_names) else str(pred)
        axes[i].set_title(f'T:{tn}\nP:{pn}', fontsize=8, color='red'); axes[i].axis('off')
    for j in range(len(mis), len(axes)):
        axes[j].axis('off')
    plt.suptitle('Misclassified Samples', fontsize=12)
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Grad-CAM (all models)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def _to_4d(self, t):
        if t.dim() == 4 and t.shape[1] != t.shape[-1]:
            # Check if (B, H, W, C) format (Swin) — last dim >> spatial dims
            if t.shape[-1] > t.shape[1]:
                t = t.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            return t
        if t.dim() == 3:
            grid = int((t.shape[1] - 1) ** 0.5)
            if grid * grid == t.shape[1] - 1:
                t = t[:, 1:, :]
            else:
                grid = int(t.shape[1] ** 0.5)
            t = t.transpose(1, 2).reshape(t.shape[0], t.shape[2], grid, grid)
        return t

    def __call__(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        acts = self._to_4d(self.activations)
        grads = self._to_4d(self.gradients)
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def get_gradcam_target_layer(model):
    from src.models import ResNet18CBAM, DenseNet121Classifier, ViTTiny, DepthwiseSeparableCNN, SwinTiny
    if isinstance(model, ResNet18CBAM):
        return model.backbone.layer4
    if isinstance(model, DenseNet121Classifier):
        return model.backbone.features.denseblock4
    if isinstance(model, DepthwiseSeparableCNN):
        return model.features[-1]
    if isinstance(model, ViTTiny):
        return model.encoder.layers[-1]
    if isinstance(model, SwinTiny):
        return model.final_stage
    raise ValueError(f"No Grad-CAM target for {type(model).__name__}")


def plot_gradcam(model, image_tensor, label, device, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    gc = GradCAM(model, get_gradcam_target_layer(model))
    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    cam, pred_idx = gc(x)
    img = image_tensor.squeeze().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap='inferno')
    axes[0].set_title(f'True: {class_names[label]}'); axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(img, cmap='gray', alpha=0.5)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Pred: {class_names[pred_idx]}'); axes[2].axis('off')
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# CBAM attention (ResNet18+CBAM best method)
# ---------------------------------------------------------------------------

def get_cbam_attention_maps(model, image_tensor, device):
    from src.models import CBAM
    cbams = [(n, m) for n, m in model.named_modules() if isinstance(m, CBAM)]
    if not cbams:
        return None
    for _, m in cbams:
        m.store_attention = True
    model.eval()
    with torch.no_grad():
        model(image_tensor.unsqueeze(0).to(device))
    maps = [(n, m.last_spatial_attn.squeeze().cpu().numpy()) for n, m in cbams]
    for _, m in cbams:
        m.store_attention = False
    return maps


def plot_cbam_attention(model, image_tensor, device, save_path=None):
    maps = get_cbam_attention_maps(model, image_tensor, device)
    if maps is None:
        return
    img = image_tensor.squeeze().numpy()
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3 * (len(maps) + 1), 3))
    axes[0].imshow(img, cmap='inferno'); axes[0].set_title('Original'); axes[0].axis('off')
    for i, (name, attn) in enumerate(maps):
        axes[i + 1].imshow(attn, cmap='hot')
        axes[i + 1].set_title(name.split('.')[-1] if '.' in name else name, fontsize=8)
        axes[i + 1].axis('off')
    plt.suptitle('CBAM Spatial Attention')
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Occlusion Sensitivity (DenseNet best method)
# ---------------------------------------------------------------------------

def occlusion_sensitivity(model, image_tensor, device, patch_size=8, stride=4):
    """Slide a zero-patch across the image and measure prediction drop.
    High values = important regions (masking them hurts confidence)."""
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        base_prob = torch.softmax(model(x), 1)
    pred_class = base_prob.argmax(1).item()
    base_conf = base_prob[0, pred_class].item()

    _, _, H, W = x.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for xp in range(0, W - patch_size + 1, stride):
            masked = x.clone()
            masked[:, :, y:y + patch_size, xp:xp + patch_size] = 0
            with torch.no_grad():
                prob = torch.softmax(model(masked), 1)[0, pred_class].item()
            drop = max(base_conf - prob, 0)
            heatmap[y:y + patch_size, xp:xp + patch_size] += drop
            count[y:y + patch_size, xp:xp + patch_size] += 1

    count[count == 0] = 1
    heatmap /= count
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, pred_class


def plot_occlusion(model, image_tensor, label, device, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    occ_map, pred_idx = occlusion_sensitivity(model, image_tensor, device)
    img = image_tensor.squeeze().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap='inferno')
    axes[0].set_title(f'True: {class_names[label]}'); axes[0].axis('off')
    axes[1].imshow(occ_map, cmap='jet')
    axes[1].set_title('Occlusion Sensitivity'); axes[1].axis('off')
    axes[2].imshow(img, cmap='gray', alpha=0.5)
    axes[2].imshow(occ_map, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Pred: {class_names[pred_idx]}'); axes[2].axis('off')
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Attention Rollout (ViT best method)
# ---------------------------------------------------------------------------

def attention_rollout(model, image_tensor, device):
    """Aggregate self-attention across all ViT layers to show cumulative
    information flow from each patch to the CLS token."""
    from src.models import ViTTiny
    try:
        from src.models_mvtec import ViTTiny_RGB, ViTTinyPretrained_RGB
        vit_classes = (ViTTiny, ViTTiny_RGB, ViTTinyPretrained_RGB)
    except ImportError:
        vit_classes = (ViTTiny,)
    if not isinstance(model, vit_classes):
        return None, None

    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    captured = []
    original_forwards = []

    # Determine encoder layers based on model type
    try:
        from src.models_mvtec import ViTTinyPretrained_RGB as _Pretrained
        is_timm = isinstance(model, _Pretrained)
    except ImportError:
        is_timm = False

    if is_timm:
        # timm blocks: each block.attn is a timm Attention module
        for block in model.encoder:
            attn_mod = block.attn
            orig_fwd = attn_mod.forward
            original_forwards.append((attn_mod, orig_fwd))

            def make_patched(orig, mod):
                def patched_forward(x, **kwargs):
                    B, N, C = x.shape
                    qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    attn = (q @ k.transpose(-2, -1)) * mod.scale
                    attn = attn.softmax(dim=-1)
                    captured.append(attn.mean(dim=1).detach())  # average over heads
                    attn = mod.attn_drop(attn)
                    x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x_out = mod.proj(x_out)
                    x_out = mod.proj_drop(x_out)
                    return x_out
                return patched_forward

            attn_mod.forward = make_patched(orig_fwd, attn_mod)
    else:
        # Custom ViT with nn.TransformerEncoder
        for layer in model.encoder.layers:
            sa = layer.self_attn
            orig_fwd = sa.forward
            original_forwards.append((sa, orig_fwd))

            def make_patched(orig):
                def patched_forward(*args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = True
                    out, attn_w = orig(*args, **kwargs)
                    captured.append(attn_w.detach())
                    return out, attn_w
                return patched_forward

            sa.forward = make_patched(orig_fwd)

    with torch.no_grad():
        out = model(x)
    pred_idx = out.argmax(1).item()

    # Restore original forwards
    for mod, orig_fwd in original_forwards:
        mod.forward = orig_fwd

    if not captured:
        return None, pred_idx

    # Rollout: multiply attention matrices with residual connections
    rollout = torch.eye(captured[0].shape[-1], device=device).unsqueeze(0)
    for attn in captured:
        attn = attn + torch.eye(attn.shape[-1], device=device).unsqueeze(0)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(attn, rollout)

    # CLS token attention to all patches (row 0, skip CLS at col 0)
    cls_attn = rollout[0, 0, 1:].cpu().numpy()
    grid = int(cls_attn.shape[0] ** 0.5)
    heatmap = cls_attn.reshape(grid, grid)

    # Upsample to image size
    img_size = image_tensor.shape[-1]
    heatmap = F.interpolate(
        torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, pred_idx


def plot_attention_rollout(model, image_tensor, label, device, class_names=None, save_path=None):
    class_names = class_names or CLASS_NAMES
    heatmap, pred_idx = attention_rollout(model, image_tensor, device)
    if heatmap is None:
        return
    img = image_tensor.squeeze().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap='inferno')
    axes[0].set_title(f'True: {class_names[label]}'); axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Attention Rollout'); axes[1].axis('off')
    axes[2].imshow(img, cmap='gray', alpha=0.5)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Pred: {class_names[pred_idx]}'); axes[2].axis('off')
    plt.tight_layout(); _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Unified explain_prediction — auto-selects best method per model
# ---------------------------------------------------------------------------

def explain_prediction(model, image_tensor, label, device, class_names=None, save_path=None):
    """Model-aware explanation that picks the most suitable method(s).

    ResNet18+CBAM -> Grad-CAM + CBAM spatial attention (dual cross-validation)
    DenseNet121   -> Grad-CAM + Occlusion Sensitivity (two independent post-hoc)
    ViT-Tiny      -> Attention Rollout + Grad-CAM (mechanism + gradient views)
    """
    from src.models import ResNet18CBAM, DenseNet121Classifier, ViTTiny
    class_names = class_names or CLASS_NAMES
    img = image_tensor.squeeze().numpy()

    # Grad-CAM (all models)
    gc = GradCAM(model, get_gradcam_target_layer(model))
    x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    cam, pred_idx = gc(x)
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    true_name = class_names[label] if label < len(class_names) else str(label)

    if isinstance(model, ResNet18CBAM):
        cbam_maps = get_cbam_attention_maps(model, image_tensor, device) or []
        ncols = 3 + len(cbam_maps)
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.5))
        axes[0].imshow(img, cmap='inferno')
        axes[0].set_title(f'True: {true_name}', fontsize=9); axes[0].axis('off')
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM', fontsize=9); axes[1].axis('off')
        axes[2].imshow(img, cmap='gray', alpha=0.5)
        axes[2].imshow(cam, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Pred: {pred_name}', fontsize=9); axes[2].axis('off')
        for i, (name, attn) in enumerate(cbam_maps):
            axes[3 + i].imshow(attn, cmap='hot')
            axes[3 + i].set_title(name.split('.')[-1], fontsize=8); axes[3 + i].axis('off')
        plt.suptitle('ResNet+CBAM: Grad-CAM + CBAM Attention', fontsize=11)

    elif isinstance(model, DenseNet121Classifier):
        occ_map, _ = occlusion_sensitivity(model, image_tensor, device)
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        axes[0].imshow(img, cmap='inferno')
        axes[0].set_title(f'True: {true_name}', fontsize=9); axes[0].axis('off')
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM', fontsize=9); axes[1].axis('off')
        axes[2].imshow(occ_map, cmap='jet')
        axes[2].set_title('Occlusion Sensitivity', fontsize=9); axes[2].axis('off')
        axes[3].imshow(img, cmap='gray', alpha=0.5)
        axes[3].imshow(occ_map, cmap='jet', alpha=0.5)
        axes[3].set_title(f'Pred: {pred_name}', fontsize=9); axes[3].axis('off')
        plt.suptitle('DenseNet: Grad-CAM + Occlusion Sensitivity', fontsize=11)

    elif isinstance(model, ViTTiny):
        rollout_map, _ = attention_rollout(model, image_tensor, device)
        has_rollout = rollout_map is not None
        ncols = 4 if has_rollout else 3
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5))
        axes[0].imshow(img, cmap='inferno')
        axes[0].set_title(f'True: {true_name}', fontsize=9); axes[0].axis('off')
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM', fontsize=9); axes[1].axis('off')
        if has_rollout:
            axes[2].imshow(rollout_map, cmap='jet')
            axes[2].set_title('Attention Rollout', fontsize=9); axes[2].axis('off')
            axes[3].imshow(img, cmap='gray', alpha=0.5)
            axes[3].imshow(rollout_map, cmap='jet', alpha=0.5)
            axes[3].set_title(f'Pred: {pred_name}', fontsize=9); axes[3].axis('off')
        else:
            axes[2].imshow(img, cmap='gray', alpha=0.5)
            axes[2].imshow(cam, cmap='jet', alpha=0.5)
            axes[2].set_title(f'Pred: {pred_name}', fontsize=9); axes[2].axis('off')
        plt.suptitle('ViT: Attention Rollout + Grad-CAM', fontsize=11)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(img, cmap='inferno')
        axes[0].set_title(f'True: {true_name}', fontsize=9); axes[0].axis('off')
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM', fontsize=9); axes[1].axis('off')
        axes[2].imshow(img, cmap='gray', alpha=0.5)
        axes[2].imshow(cam, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Pred: {pred_name}', fontsize=9); axes[2].axis('off')
        plt.suptitle('Grad-CAM', fontsize=11)

    plt.tight_layout(); _save_or_show(fig, save_path)


# Backward compatibility alias
plot_explainability = explain_prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
