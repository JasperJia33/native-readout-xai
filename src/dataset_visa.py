"""VisA dataset loader for binary classification (normal vs anomaly).

Preloads all images into RAM at init to avoid shared-memory issues
with multi-worker DataLoader. Uses num_workers=0 by design.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisADataset(Dataset):
    """VisA binary classification dataset with in-memory preloading.

    Args:
        samples: list of dicts with keys 'image', 'label', 'mask', 'object'
        data_root: path to Data_VisA directory
        cfg: config dict
        augmentation: whether to apply training augmentation
    """

    def __init__(self, samples, data_root, cfg, augmentation=False):
        self.samples = samples
        self.data_root = data_root
        self.image_size = cfg['data']['image_size']
        self.augmentation = augmentation and cfg.get('augmentation', {}).get('enabled', False)
        self.aug_cfg = cfg.get('augmentation', {})

        # Preload all images into RAM (uint8 to save memory)
        print(f"  Preloading {len(samples)} images into RAM...", end=' ', flush=True)
        imgs = []
        for s in samples:
            img_path = os.path.join(data_root, s['image'])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.uint8))
        self._images = np.stack(imgs)  # (N, H, W, 3) uint8
        mem_mb = self._images.nbytes / 1024 / 1024
        print(f"done ({mem_mb:.0f} MB)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self._images[idx].astype(np.float32) / 255.0

        if self.augmentation:
            img = self._augment(img)

        img = img.transpose(2, 0, 1)  # (H,W,3) -> (3,H,W)
        label = 0 if self.samples[idx]['label'] == 'normal' else 1
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment(self, img):
        rng = np.random.default_rng()
        if rng.random() < self.aug_cfg.get('flip_prob', 0.5):
            img = np.flip(img, axis=1).copy()
        if rng.random() < self.aug_cfg.get('flip_prob', 0.5):
            img = np.flip(img, axis=0).copy()
        if rng.random() < self.aug_cfg.get('rotation_prob', 0.5):
            k = rng.integers(1, 4)
            img = np.rot90(img, k, axes=(0, 1)).copy()
        if rng.random() < self.aug_cfg.get('noise_prob', 0.3):
            noise = rng.normal(0, self.aug_cfg.get('noise_std', 0.02), img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
        return img

    def get_mask(self, idx):
        """Return the defect mask for a sample (None if normal)."""
        s = self.samples[idx]
        if not s.get('mask'):
            return None
        mask_path = os.path.join(self.data_root, s['mask'])
        if not os.path.exists(mask_path):
            return None
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        return np.array(mask, dtype=np.float32) / 255.0


def load_visa_samples(data_root, split_csv='split_csv/2cls_highshot.csv'):
    """Load VisA samples from the official split CSV.

    Returns:
        train_samples, test_samples: lists of dicts
    """
    import csv
    csv_path = os.path.join(data_root, split_csv)
    train_samples, test_samples = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {
                'object': row['object'],
                'label': row['label'],
                'image': row['image'],
                'mask': row['mask'] if row.get('mask') else None,
            }
            if row['split'] == 'train':
                train_samples.append(sample)
            else:
                test_samples.append(sample)

    return train_samples, test_samples
