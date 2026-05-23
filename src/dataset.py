import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter

log = logging.getLogger(__name__)


class WM811KDataset(Dataset):
    """Config-driven dataset with optional caching and minority-boost augmentation."""

    def __init__(self, df, cfg, class_counts=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        sz = cfg['data']['image_size']
        self.target_size = (sz, sz)
        self.augment = cfg['augmentation']['enabled']
        self.aug_cfg = cfg['augmentation']
        self.class_counts = class_counts or Counter(int(l) for l in df['failurePatternType'])

        # Preprocessing cache
        self.cache = None
        if cfg['data'].get('cache_preprocessed', False):
            log.info("Caching preprocessed wafer maps in RAM...")
            self.cache = [self._preprocess(row['waferMap']) for _, row in self.df.iterrows()]
            log.info(f"Cached {len(self.cache)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wafermap = self.cache[idx] if self.cache is not None else self._preprocess(row['waferMap'])
        label = int(row['failurePatternType'])

        if self.augment:
            is_minority = self.class_counts.get(label, 0) < 1000
            wafermap = self._augment(wafermap, boost=is_minority and self.aug_cfg.get('minority_boost', False))

        wafermap = torch.tensor(wafermap, dtype=torch.float32).unsqueeze(0)
        return wafermap, torch.tensor(label, dtype=torch.long)

    def _preprocess(self, wafermap):
        if isinstance(wafermap, list):
            wafermap = np.array(wafermap)
        processed = wafermap.astype(np.float64)
        processed[wafermap == 0] = np.nan
        processed[wafermap == 1] = 0
        processed[wafermap == 2] = 1
        zoom_factors = (self.target_size[0] / processed.shape[0],
                        self.target_size[1] / processed.shape[1])
        processed = ndimage.zoom(processed, zoom_factors, order=0)
        return np.nan_to_num(processed, nan=0.0)

    def _augment(self, wafermap, boost=False):
        aug = wafermap.copy()
        rot_p = min(self.aug_cfg['rotation_prob'] * (2 if boost else 1), 1.0)
        flip_p = min(self.aug_cfg['flip_prob'] * (2 if boost else 1), 1.0)
        noise_p = min(self.aug_cfg['noise_prob'] * (2 if boost else 1), 1.0)

        if np.random.random() < rot_p:
            aug = np.rot90(aug, k=np.random.randint(1, 4))
        if np.random.random() < flip_p:
            aug = np.fliplr(aug)
        if np.random.random() < flip_p:
            aug = np.flipud(aug)
        if np.random.random() < noise_p:
            aug = np.clip(aug + np.random.normal(0, self.aug_cfg['noise_std'], aug.shape), 0, 1)
        return aug.copy()


class WM811KLoader:
    """Load, filter, and split the WM-811K pickle dataset."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.filepath = cfg['data']['path']
        self.df = None

    def load(self):
        log.info(f"Loading {self.filepath}...")
        # Patch for old pandas pickle files that reference removed pandas.indexes
        import sys
        if 'pandas.indexes' not in sys.modules:
            import pandas.core.indexes as _idx
            sys.modules['pandas.indexes'] = _idx
            sys.modules['pandas.indexes.base'] = _idx.base
            sys.modules['pandas.indexes.numeric'] = _idx.numeric if hasattr(_idx, 'numeric') else _idx.base
            sys.modules['pandas.indexes.multi'] = _idx.multi
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.df = pd.DataFrame(data)
        log.info(f"Total wafer maps: {len(self.df)}")
        log.info(f"Columns: {list(self.df.columns)}")
        self._normalize_columns()
        return self.df

    def _normalize_columns(self):
        """Map actual WM-811K column names to the names used throughout the code.

        The real pickle stores failureType as 2-D numpy arrays:
          shape (1,1) → labeled, e.g. [['none']], [['Edge-Loc']]
          shape (0,0) → unlabeled, i.e. []
        """
        LABEL_MAP = {
            'none': 0, 'Center': 4, 'Donut': 7, 'Edge-Loc': 1,
            'Edge-Ring': 2, 'Loc': 3, 'Random': 6, 'Scratch': 5, 'Near-full': 8,
        }
        if 'failurePatternType' not in self.df.columns and 'failureType' in self.df.columns:
            def _parse_failure(ft):
                if isinstance(ft, np.ndarray):
                    if ft.size == 0:
                        return np.nan
                    label_str = ft.flat[0]  # get the single string from [[...]]
                else:
                    return np.nan
                return LABEL_MAP.get(label_str, np.nan)
            self.df['failurePatternType'] = self.df['failureType'].apply(_parse_failure)
            log.info("Mapped 'failureType' → 'failurePatternType'")

    def get_labeled_data(self):
        self.df = self.df[self.df['failurePatternType'].notna()].copy()
        log.info(f"Labeled samples: {len(self.df)}")

        dcfg = self.cfg['data']

        # Exclude None class
        if dcfg.get('exclude_none', False):
            self.df = self.df[self.df['failurePatternType'] != 0].copy()
            remap = {old: new for new, old in enumerate(sorted(self.df['failurePatternType'].unique()))}
            self.df['failurePatternType'] = self.df['failurePatternType'].map(remap)
            self.cfg['model']['num_classes'] = len(remap)
            log.info(f"Excluded None class. Remapped to {len(remap)} classes, {len(self.df)} samples.")

        # Downsample majority classes
        cap = dcfg.get('max_samples_per_class')
        if cap:
            frames = []
            for cls in self.df['failurePatternType'].unique():
                subset = self.df[self.df['failurePatternType'] == cls]
                frames.append(subset.sample(n=min(len(subset), cap), random_state=self.cfg['training']['seed']))
            self.df = pd.concat(frames).reset_index(drop=True)
            log.info(f"Downsampled to max {cap}/class, total {len(self.df)} samples.")

        return self.df

    def split_data(self):
        dcfg = self.cfg['data']
        seed = self.cfg['training']['seed']
        test_size = dcfg['test_size']
        val_size = dcfg['val_size']

        if dcfg.get('split_mode') == 'lot_group':
            return self._lot_group_split(test_size, val_size, seed)
        return self._stratified_split(test_size, val_size, seed)

    def _lot_group_split(self, test_size, val_size, seed):
        groups = self.df['lotName'].values
        # Split into train_val / test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tv_idx, test_idx = next(gss1.split(self.df, groups=groups))
        train_val_df = self.df.iloc[tv_idx]
        test_df = self.df.iloc[test_idx]
        # Split train_val into train / val
        tv_groups = train_val_df['lotName'].values
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        train_idx, val_idx = next(gss2.split(train_val_df, groups=tv_groups))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        log.info(f"Lot-group split: Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
        return train_df, val_df, test_df

    def _stratified_split(self, test_size, val_size, seed):
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size,
            stratify=self.df['failurePatternType'], random_state=seed)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size),
            stratify=train_val_df['failurePatternType'], random_state=seed)
        log.info(f"Stratified split: Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
        return train_df, val_df, test_df
