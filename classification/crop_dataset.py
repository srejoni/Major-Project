"""
classification/crop_dataset.py

Loads pre-cropped classification patches saved by crop_extraction.py.
Every crop is a genuine (H, W, 3) array -- sagittal replicated to 3
identical channels, axial as 3 real neighboring slices. This class only
transposes and normalizes; it does NOT replicate channels itself.

Each item is one crop = one (study, condition, level). study_id/condition/
level are carried through __getitem__ so predictions can be reassembled
into the per-study findings JSON for the LLM integration layer.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


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
        if random.random() < 0.5:
            tensor = TF.rotate(tensor, random.uniform(-10, 10))
        if random.random() < 0.5:
            tensor = TF.adjust_brightness(tensor, random.uniform(0.9, 1.1))
            tensor = TF.adjust_contrast(tensor, random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            tensor = tensor + torch.randn_like(tensor) * 0.02
        return tensor

    def __getitem__(self, idx):
        img = np.load(self.crop_paths[idx]).astype(np.float32)
        img = (img - self.mean) / (self.std + 1e-8)
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        if self.augment:
            tensor = self._augment(tensor)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label, self.study_ids[idx], self.conditions[idx], self.levels[idx]
