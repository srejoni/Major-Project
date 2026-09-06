"""
PDSCD - shared preprocessing.

CRITICAL RULE (project doc, section 8): this preprocess() function must be
called IDENTICALLY by both training and inference/validation code. Never
write separate logic for train vs. test - that's how train/test skew
creeps in.

Install once per Kaggle notebook:
    pip install pydicom opencv-python-headless
"""

import numpy as np
import pydicom
import cv2

TARGET_SIZE = 224  # 224x224 default; raise to 512 only if your GPU/RAM allow it


def load_dicom_pixels(dicom_path: str) -> np.ndarray:
    """Load a single DICOM file and return its pixel array as float32."""
    dcm = pydicom.dcmread(dicom_path)
    pixels = dcm.pixel_array.astype(np.float32)
    return pixels


def resize_image(pixels: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Resize to a fixed square size. Deterministic - no randomness here."""
    return cv2.resize(pixels, (target_size, target_size), interpolation=cv2.INTER_AREA)


def preprocess(pixels: np.ndarray, mean: float, std: float,
                target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    THE single reusable preprocessing function. Call this identically from
    training and inference/validation code - never write two versions.

    Steps: resize -> per-fold z-score normalize using mean/std computed
    ONLY from that fold's training patients (see compute_fold_stats.py).
    Val/test images get normalized with their fold's TRAIN stats too -
    that's correct and intentional, not a leak (§6: "apply that same
    deterministic preprocessing to both training and validation images").
    """
    resized = resize_image(pixels, target_size)
    normalized = (resized - mean) / (std + 1e-8)
    return normalized.astype(np.float32)
