import numpy as np


def random_augment(wafermap):
    """Standalone augmentation utility (rotation, flip, noise)."""
    aug = wafermap.copy()

    if np.random.random() < 0.7:
        aug = np.rot90(aug, k=np.random.randint(1, 4))

    if np.random.random() < 0.5:
        aug = np.fliplr(aug)
    if np.random.random() < 0.5:
        aug = np.flipud(aug)

    if np.random.random() < 0.3:
        aug = np.clip(aug + np.random.normal(0, 0.05, aug.shape), 0, 1)

    return aug.copy()
