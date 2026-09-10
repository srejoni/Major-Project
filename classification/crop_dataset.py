"""
classification/crop_dataset.py

Loads pre-cropped classification patches saved by crop_extraction.py.
Every crop is a genuine (H, W, 3) array -- sagittal replicated to 3
identical channels, axial as 3 real neighboring slices. This class only
transposes and normalizes; it does NOT replicate channels itself.

Each item is one crop = one (study, condition, level). study_id/condition/
level are carried through __getitem__ so predictions can be reassembled
into the per-study findings JSON for the LLM integration layer.

Normalization is imported from normalization.py -- the single shared
function also used by assemble_study_output.py at inference time, so
train and inference can never drift apart on this step again.

AUGMENTATION STRENGTHENED (overfitting fix, alongside train.py's weight
decay / label smoothing / early stopping / LR scheduling): the previous
augmentation was mild -- +-10 deg rotation, +-10% brightness/contrast,
p=0.3 light noise. Widened below and a random crop/zoom step added, since
stronger augmentation is a direct lever against a model that's overfitting
the training set specifically (memorizing exact crop framing/exposure
rather than learning the underlying anatomy).

CAUTION (project doc, unchanged): several RSNA conditions are left/right-
specific. Horizontal flip is intentionally OMITTED from the augmentation
set below -- do not add it unless you also swap the corresponding
left/right labels.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from classification.normalization import normalize_crop


class ClassificationCropDataset(Dataset):
    def __init__(self, crop_paths, labels, mean, std, augment=False,
                 study_ids=None, conditions=None, levels=None):
        self.crop_paths = crop_paths
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment
        # Optional so this stays backward-compatible with any existing
        # call site that only passes crop_paths/labels/mean/std/augment.
        self.study_ids = study_ids if study_ids is not None else [None] * len(crop_paths)
        self.conditions = conditions if conditions is not None else [None] * len(crop_paths)
        self.levels = levels if levels is not None else [None] * len(crop_paths)

    def __len__(self):
        return len(self.crop_paths)

    def _augment(self, tensor):
        """Training-only, applied AFTER normalization (tensor is already
        (C, H, W) float). Widened ranges + added random crop/zoom vs. the
        previous version -- see module docstring. Still deliberately NO
        horizontal flip (see caution above)."""
        # Rotation: widened +-10 -> +-15 deg, p unchanged at effectively
        # always-on (matches previous behavior of applying at p=0.5).
        if random.random() < 0.5:
            tensor = TF.rotate(tensor, random.uniform(-15, 15))

        # Brightness/contrast: widened +-10% -> +-15%.
        if random.random() < 0.5:
            tensor = TF.adjust_brightness(tensor, random.uniform(0.85, 1.15))
            tensor = TF.adjust_contrast(tensor, random.uniform(0.85, 1.15))

        # Random crop/zoom -- NEW. Zooms in slightly (crops a smaller region,
        # then resizes back to the original size), forcing the model to not
        # rely on exact framing/margin consistency across every sample.
        if random.random() < 0.4:
            c, h, w = tensor.shape
            zoom = random.uniform(1.0, 1.15)
            zh, zw = int(h / zoom), int(w / zoom)
            y0 = random.randint(0, max(h - zh, 1))
            x0 = random.randint(0, max(w - zw, 1))
            tensor = tensor[:, y0:y0 + zh, x0:x0 + zw]
            tensor = TF.resize(tensor, [h, w], antialias=True)

        # Gaussian noise: probability raised 0.3 -> 0.5.
        if random.random() < 0.5:
            tensor = tensor + torch.randn_like(tensor) * 0.02

        return tensor

    def __getitem__(self, idx):
        img = np.load(self.crop_paths[idx]).astype(np.float32)
        tensor = normalize_crop(img, self.mean, self.std)
        if self.augment:
            tensor = self._augment(tensor)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label, self.study_ids[idx], self.conditions[idx], self.levels[idx]
