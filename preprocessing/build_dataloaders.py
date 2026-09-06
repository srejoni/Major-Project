"""
PDSCD - example: assemble the train/val DataLoaders for one fold.

Run this AFTER compute_fold_stats.py has produced
splits/stats/fold{FOLD}_stats.json for your fold.

NOTE on labels: build_paths_and_labels() below is a placeholder. Real
per-condition severity labels get attached after the localization step
(cropping around each detected disc level) - that's step 4-5 in the
pipeline. This script exists to prove the loading/preprocessing wiring
works end-to-end; swap in real (path, label) pairs once cropped patches
exist.
"""

import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from dataset import RSNAImageDataset

FOLD = 1
BATCH_SIZE = 16  # 8-16 typical for 224x224 on a single 16GB T4 (project doc §8)

RSNA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
SPLITS_DIR = "splits"

# 1. load this fold's TRAIN-only stats (computed once, cached)
with open(os.path.join(SPLITS_DIR, "stats", f"fold{FOLD}_stats.json")) as f:
    stats = json.load(f)
mean, std = stats["mean"], stats["std"]

# 2. load this fold's patient ID lists (from the splitting script)
train_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{FOLD}_train_ids.csv"))["study_id"])
val_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{FOLD}_val_ids.csv"))["study_id"])


def build_paths_and_labels(patient_ids):
    """
    PLACEHOLDER - replace with your real (path, label) construction once
    the localization/cropping step is producing cropped patches with
    per-condition severity labels attached.
    """
    paths, labels = [], []
    return paths, labels


train_paths, train_labels = build_paths_and_labels(train_ids)
val_paths, val_labels = build_paths_and_labels(val_ids)

train_ds = RSNAImageDataset(train_paths, train_labels, mean, std, augment=True)
val_ds = RSNAImageDataset(val_paths, val_labels, mean, std, augment=False)  # never augment val/test

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Fold {FOLD}: {len(train_ds)} train images, {len(val_ds)} val images "
      f"(mean={mean:.4f}, std={std:.4f})")
