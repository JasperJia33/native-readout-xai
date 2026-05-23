import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils.metrics import FocalLoss

log = logging.getLogger(__name__)


def _build_optimizer(model, cfg):
    tc = cfg['training']
    lr, wd = tc['lr'], tc.get('weight_decay', 0.0)
    name = tc['optimizer'].lower()
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(optimizer, cfg):
    sc = cfg['training']['scheduler']
    stype = sc['type'].lower()
    if stype == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=sc.get('factor', 0.5), patience=sc.get('patience', 5))
    if stype == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    if stype == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=sc.get('factor', 0.5))
    raise ValueError(f"Unknown scheduler: {stype}")


def _build_criterion(cfg, class_weights, device):
    tc = cfg['training']
    if tc['loss'] == 'focal':
        return FocalLoss(alpha=class_weights.to(device), gamma=tc.get('focal_gamma', 2.0))
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


def _save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                     history, best_metric, patience_counter):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'best_metric': best_metric,
        'patience_counter': patience_counter,
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return (ckpt['epoch'], ckpt['history'],
            ckpt['best_metric'], ckpt['patience_counter'])


def train_model(model, train_loader, val_loader, cfg, class_weights,
                device, run_dir=None):
    """Config-driven training with AMP, TensorBoard, checkpoints,
    and flexible early stopping."""
    tc = cfg['training']
    epochs = tc['epochs']
    use_amp = tc.get('mixed_precision', False) and device.type == 'cuda'
    metric_name = tc.get('early_stop_metric', 'val_loss')
    higher_is_better = metric_name != 'val_loss'

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    criterion = _build_criterion(cfg, class_weights, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss',
                                'val_acc', 'val_balanced_acc',
                                'val_f1_macro', 'lr']}
    best_metric = -float('inf') if higher_is_better else float('inf')
    best_state = None
    patience_counter = 0
    start_epoch = 0

    # --- Resume from checkpoint ---
    ckpt_path = os.path.join(run_dir, 'checkpoint.pth') if run_dir else None
    if ckpt_path and os.path.isfile(ckpt_path):
        start_epoch, history, best_metric, patience_counter = \
            _load_checkpoint(ckpt_path, model, optimizer, scheduler,
                             scaler, device)
        best_state = {k: v.cpu().clone()
                      for k, v in model.state_dict().items()}
        log.info(f"Resumed from checkpoint at epoch {start_epoch}, "
                 f"best {metric_name}={best_metric:.4f}")

    # --- TensorBoard ---
    writer = None
    if run_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=os.path.join(run_dir, 'tensorboard'))
            log.info(f"TensorBoard: {run_dir}/tensorboard/")
        except ImportError:
            log.warning("tensorboard not installed, skipping")

    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in tqdm(train_loader,
                         desc=f'Epoch {epoch+1}/{epochs} [Train]',
                         leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), tc.get('grad_clip_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_correct += out.argmax(1).eq(y).sum().item()
            t_total += y.size(0)

        # --- Validate ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in tqdm(val_loader,
                             desc=f'Epoch {epoch+1}/{epochs} [Val]',
                             leave=False):
                X, y = X.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(X)
                    loss = criterion(out, y)
                v_loss += loss.item()
                preds = out.argmax(1)
                v_correct += preds.eq(y).sum().item()
                v_total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_tl = t_loss / len(train_loader)
        avg_vl = v_loss / len(val_loader)
        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        v_bacc = balanced_accuracy_score(all_labels, all_preds)
        v_f1 = f1_score(all_labels, all_preds, average='macro',
                        zero_division=0)
        cur_lr = optimizer.param_groups[0]['lr']

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_vl)
        else:
            scheduler.step()

        history['train_loss'].append(avg_tl)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(avg_vl)
        history['val_acc'].append(v_acc)
        history['val_balanced_acc'].append(v_bacc)
        history['val_f1_macro'].append(v_f1)
        history['lr'].append(cur_lr)

        log.info(
            f"Epoch {epoch+1}/{epochs}  "
            f"TrLoss={avg_tl:.4f} TrAcc={t_acc:.4f}  "
            f"VlLoss={avg_vl:.4f} VlAcc={v_acc:.4f} "
            f"VlBAcc={v_bacc:.4f} VlF1={v_f1:.4f} "
            f"LR={cur_lr:.6f}")

        # --- TensorBoard ---
        if writer:
            writer.add_scalars('Loss',
                               {'train': avg_tl, 'val': avg_vl}, epoch+1)
            writer.add_scalars('Accuracy',
                               {'train': t_acc, 'val': v_acc}, epoch+1)
            writer.add_scalar('Val/balanced_acc', v_bacc, epoch+1)
            writer.add_scalar('Val/f1_macro', v_f1, epoch+1)
            writer.add_scalar('LR', cur_lr, epoch+1)

        # --- Early stopping ---
        current = {'val_loss': avg_vl, 'val_balanced_acc': v_bacc,
                   'val_f1_macro': v_f1}[metric_name]
        improved = ((current > best_metric) if higher_is_better
                    else (current < best_metric))
        if improved:
            best_metric = current
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            log.info(f"  New best {metric_name}={current:.4f}")
        else:
            patience_counter += 1

        # --- Save checkpoint every epoch ---
        if ckpt_path:
            _save_checkpoint(ckpt_path, epoch + 1, model, optimizer,
                             scheduler, scaler, history,
                             best_metric, patience_counter)

        if patience_counter >= tc.get('early_stop_patience', 10):
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    if writer:
        writer.close()
    if best_state:
        model.load_state_dict(best_state)
    return model, history
