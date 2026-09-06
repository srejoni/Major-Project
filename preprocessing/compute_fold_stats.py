"""
PDSCD - compute per-fold normalization mean/std using ONLY that fold's
TRAINING patients (project doc §6: "fit preprocessing parameters using
only their fold's training patients"). Deterministic, no randomness -
this only needs to run ONCE per fold, then cache and reuse (§6: "val/test
preprocessing never needs regenerating since it has no randomness").

Each of the 4 teammates runs this once for THEIR fold (change FOLD below),
then commits the resulting stats/foldN_stats.json to the repo.
"""

import os
import json
import pandas as pd
import numpy as np
from preprocess import load_dicom_pixels, resize_image

RSNA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
IMAGES_ROOT = os.path.join(RSNA_ROOT, "train_images")
SPLITS_DIR = "../configs/splits"
STATS_DIR = os.path.join(SPLITS_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)

FOLD = 1  # <-- change to 1, 2, 3, or 4 depending on which teammate is running this
MAX_IMAGES = None  # set e.g. 3000 for a fast approximate stat if you're extremely
                    # time-constrained; leave None for the accurate full pass


def get_dicom_paths_for_patients(study_ids, series_df):
    """
    series_df: train_series_descriptions.csv, filtered by study_id
    membership - this is the leakage-safe join pattern (never slice this
    file by row position, always filter by the fold's ID list).
    """
    paths = []
    fold_series = series_df[series_df["study_id"].isin(study_ids)]
    for _, row in fold_series.iterrows():
        series_dir = os.path.join(IMAGES_ROOT, str(row["study_id"]), str(row["series_id"]))
        if not os.path.isdir(series_dir):
            continue
        for fname in os.listdir(series_dir):
            if fname.endswith(".dcm"):
                paths.append(os.path.join(series_dir, fname))
    return paths


def compute_stats(dicom_paths, target_size=224, max_images=None):
    """
    Vectorized running sum / sum-of-squares (float64 accumulators for
    numerical stability) - one pass over the fold's training images only,
    O(1) memory regardless of dataset size.
    """
    if max_images is not None:
        dicom_paths = dicom_paths[:max_images]

    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for i, path in enumerate(dicom_paths):
        try:
            pixels = load_dicom_pixels(path)
            resized = resize_image(pixels, target_size).astype(np.float64)
        except Exception as e:
            print(f"Skipping unreadable file {path}: {e}")
            continue

        total_sum += resized.sum()
        total_sumsq += np.square(resized).sum()
        total_count += resized.size

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(dicom_paths)} images...")

    if total_count == 0:
        raise RuntimeError("No readable DICOM files found - check IMAGES_ROOT / fold IDs.")

    mean = total_sum / total_count
    var = (total_sumsq / total_count) - mean ** 2
    std = np.sqrt(max(var, 1e-8))
    return float(mean), float(std)


if __name__ == "__main__":
    series_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_series_descriptions.csv"))
    train_ids = pd.read_csv(
        os.path.join(SPLITS_DIR, f"fold{FOLD}_train_ids.csv")
    )["study_id"].tolist()

    print(f"Fold {FOLD}: computing stats from {len(train_ids)} training patients only "
          f"(never validation or the locked test set)...")

    dicom_paths = get_dicom_paths_for_patients(train_ids, series_df)
    print(f"Found {len(dicom_paths)} DICOM files in fold {FOLD}'s training set.")

    mean, std = compute_stats(dicom_paths, max_images=MAX_IMAGES)
    print(f"Fold {FOLD} stats: mean={mean:.4f}, std={std:.4f}")

    out_path = os.path.join(STATS_DIR, f"fold{FOLD}_stats.json")
    with open(out_path, "w") as f:
        json.dump({"mean": mean, "std": std, "n_images": len(dicom_paths)}, f, indent=2)
    print(f"Saved -> {out_path}. Commit this file to the repo - it never needs "
          f"regenerating unless fold{FOLD}_train_ids.csv changes.")
