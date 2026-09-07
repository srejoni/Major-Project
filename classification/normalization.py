"""
classification/normalization.py

THE single normalization function for classification crops. Both training
(crop_dataset.py) and inference (assemble_study_output.py) must import and
call this -- never reimplement it. This project has already hit two
separate train/inference normalization-mismatch bugs (BOX_SIZE_FRAC,
percentile-vs-zscore); this file exists so a third can't happen here.
"""
import numpy as np
import torch


def normalize_crop(crop_array: np.ndarray, mean: float, std: float) -> torch.Tensor:
    img = crop_array.astype(np.float32)
    img = (img - mean) / (std + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1)
