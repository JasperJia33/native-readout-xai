"""MVTec AD dataset loader for the FiftyOne/Voxel51 export format.

Provides a PyTorch Dataset compatible with the existing training pipeline.
Binary classification: 0 = good, 1 = defective.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecADDataset(Dataset):
    """MVTec AD dataset — binary classification (good vs defective).

    Args:
        samples: list of dicts from samples.json (filtered to desired split)
        data_root: path to the mvtec-ad directory
        cfg: config dict (uses cfg['data']['image_size'])
        augmentation: whether to apply training augmentation
        sample_indices: optional list of original indices (for cache lookup)
    """

    def __init__(self, samples, data_root, cfg, augmentation=False, sample_indices=None):
        self.samples = samples
        self.data_root = data_root
        self.image_size = cfg['data']['image_size']
        self.augmentation = augmentation and cfg.get('augmentation', {}).get('enabled', False)
        self.aug_cfg = cfg.get('augmentation', {})
        self.sample_indices = sample_indices  # maps local idx -> global cache idx
        self.cache_dir = cfg['data'].get('cache_dir', None)

        # Preload all images into RAM if cache exists (~1GB for full dataset)
        self._mem_cache = None
        if self.cache_dir and self.sample_indices is not None:
            imgs = []
            for global_idx in self.sample_indices:
                npy_path = os.path.join(self.cache_dir, f'{global_idx:05d}.npy')
                imgs.append(np.load(npy_path))
            self._mem_cache = np.stack(imgs)  # (N, 256, 256, 3) uint8

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Use in-memory cache (fastest)
        if self._mem_cache is not None:
            img = self._mem_cache[idx].astype(np.float32) / 255.0
        elif self.cache_dir and self.sample_indices is not None:
            global_idx = self.sample_indices[idx]
            npy_path = os.path.join(self.cache_dir, f'{global_idx:05d}.npy')
            if os.path.exists(npy_path):
                img = np.load(npy_path).astype(np.float32) / 255.0
            else:
                img = self._load_pil(s)
        else:
            img = self._load_pil(s)

        if self.augmentation:
            img = self._augment(img)

        # (H, W, 3) -> (3, H, W)
        img = img.transpose(2, 0, 1)
        label = 0 if s['defect']['label'] == 'good' else 1

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _load_pil(self, s):
        img_path = os.path.join(self.data_root, s['filepath'])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _augment(self, img):
        """Simple augmentation: random flip + rotation + noise."""
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
        """Return the defect mask for a sample (None if good/no mask)."""
        # Try cache first
        if self.cache_dir and self.sample_indices is not None:
            global_idx = self.sample_indices[idx]
            mask_npy = os.path.join(self.cache_dir, 'masks', f'{global_idx:05d}.npy')
            if os.path.exists(mask_npy):
                return np.load(mask_npy).astype(np.float32) / 255.0

        s = self.samples[idx]
        if 'defect_mask' not in s or s['defect_mask'] is None:
            return None
        mask_path = os.path.join(self.data_root, s['defect_mask']['mask_path'])
        if not os.path.exists(mask_path):
            return None
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        return np.array(mask, dtype=np.float32) / 255.0  # binary mask


def load_mvtec_samples(data_root, categories=None):
    """Load and parse samples.json, optionally filtering by category.

    Returns:
        train_samples, test_samples: lists of sample dicts
        train_indices, test_indices: global indices into samples.json (for cache)
    """
    with open(os.path.join(data_root, 'samples.json')) as f:
        data = json.load(f)

    all_samples = data['samples']

    train_samples, test_samples = [], []
    train_indices, test_indices = [], []

    for i, s in enumerate(all_samples):
        if categories and s['category']['label'] not in categories:
            continue
        if s['split'] == 'train':
            train_samples.append(s)
            train_indices.append(i)
        else:
            test_samples.append(s)
            test_indices.append(i)

    return train_samples, test_samples, train_indices, test_indices
