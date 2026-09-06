"""
PDSCD - streaming PyTorch Dataset for RSNA DICOM images.

Reads one image at a time (never loads the whole fold into RAM), applies
the SAME preprocess() function for train and val, and layers on
augmentation only when augment=True (training only - regenerated fresh
every epoch automatically, since it happens inside __getitem__, which
PyTorch calls again each epoch).

CAUTION (project doc §6): several RSNA conditions are left/right-specific.
Horizontal flip is intentionally OMITTED from the augmentation set below -
do not add it unless you also swap the corresponding left/right labels.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from preprocess import load_dicom_pixels, preprocess, TARGET_SIZE


class RSNAImageDataset(Dataset):
    def __init__(self, dicom_paths, labels, mean, std, augment=False,
                 target_size=TARGET_SIZE):
        """
        dicom_paths: list[str] - full paths to .dcm files for THIS split
                     only (one fold's train OR val - never mixed together).
        labels: list - label per path, aligned by index.
        mean, std: this fold's TRAIN-set-only stats (from
                   compute_fold_stats.py), applied identically to both
                   this fold's train and val data.
        augment: True for training data only. Always False for val/test.
        """
        assert len(dicom_paths) == len(labels), "paths and labels must be same length"
        self.dicom_paths = dicom_paths
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.dicom_paths)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Training-only. Rotation (+-10 deg), brightness/contrast jitter,
        random crop/zoom, Gaussian noise - per project doc §6.
        Deliberately NO horizontal flip (see caution above).
        """
        h, w = image.shape[:2]

        # random rotation +-10 deg
        angle = random.uniform(-10, 10)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

        # brightness/contrast jitter
        brightness = random.uniform(-0.1, 0.1)
        contrast = random.uniform(0.9, 1.1)
        image = image * contrast + brightness

        # random crop/zoom (zoom in slightly, then resize back to original size)
        zoom = random.uniform(1.0, 1.15)
        zh, zw = int(h / zoom), int(w / zoom)
        y0 = random.randint(0, max(h - zh, 1))
        x0 = random.randint(0, max(w - zw, 1))
        image = image[y0:y0 + zh, x0:x0 + zw]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        # Gaussian noise
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = image + noise

        return image

    def __getitem__(self, idx):
        path = self.dicom_paths[idx]
        label = self.labels[idx]

        pixels = load_dicom_pixels(path)                                  # raw DICOM
        image = preprocess(pixels, self.mean, self.std, self.target_size)  # deterministic

        if self.augment:
            image = self._augment(image)                                  # train-only, random

        tensor = torch.from_numpy(image).unsqueeze(0).float()  # add channel dim -> (1, H, W)
        return tensor, torch.tensor(label, dtype=torch.long)
