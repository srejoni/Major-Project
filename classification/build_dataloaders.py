"""
classification/build_dataloaders.py

Builds train/val DataLoaders using the fold1_train / fold1_val split from
splits.py (Issue 1 fix -- NOT merged into one pool).

RSNAImageDataset's constructor params and fold1_stats.json's key names are
verified at runtime, not assumed, given this project's prior mixups with both.
"""

import sys
import json
import inspect

import pandas as pd
from torch.utils.data import DataLoader

sys.path.append("/kaggle/working/PDSCD")
sys.path.append("/kaggle/working/PDSCD/data_loading")
from data_loading.dataset import RSNAImageDataset
from classification.splits import load_classification_split

STATS_PATH = "/kaggle/working/PDSCD/configs/splits/stats/fold1_stats.json"
EXPECTED_PARAMS = {"dicom_paths", "labels", "mean", "std", "augment", "target_size"}


def _verify_dataset_signature():
    sig = inspect.signature(RSNAImageDataset.__init__)
    actual = set(sig.parameters.keys()) - {"self"}
    missing = EXPECTED_PARAMS - actual
    if missing:
        raise TypeError(
            f"RSNAImageDataset.__init__ missing expected params {missing}. "
            f"Actual signature: {sig}. Fix this function before proceeding."
        )


def _load_fold1_stats():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean_key = next((k for k in ("mean", "global_mean") if k in stats), None)
    std_key = next((k for k in ("std", "global_std") if k in stats), None)
    if mean_key is None or std_key is None:
        raise KeyError(
            f"fold1_stats.json has keys {list(stats.keys())} -- expected "
            f"mean/std or global_mean/global_std. Update this function."
        )
    print(f"[build_dataloaders] stats: {mean_key}={stats[mean_key]:.4f}  "
          f"{std_key}={stats[std_key]:.4f}")
    return stats[mean_key], stats[std_key]


def build_classification_dataloaders(labels_df, batch_size=16, num_workers=2):
    """labels_df needs columns: study_id, crop_image_path, severity."""
    _verify_dataset_signature()
    mean, std = _load_fold1_stats()

    train_ids, val_ids = load_classification_split()
    train_df = labels_df[labels_df.study_id.isin(train_ids)].reset_index(drop=True)
    val_df = labels_df[labels_df.study_id.isin(val_ids)].reset_index(drop=True)
    print(f"[build_dataloaders] train rows={len(train_df)}  val rows={len(val_df)}")

    train_dataset = RSNAImageDataset(
        dicom_paths=train_df["crop_image_path"].tolist(),
        labels=train_df["severity"].tolist(),
        mean=mean, std=std, augment=True, target_size=224,
    )
    val_dataset = RSNAImageDataset(
        dicom_paths=val_df["crop_image_path"].tolist(),
        labels=val_df["severity"].tolist(),
        mean=mean, std=std, augment=False, target_size=224,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
