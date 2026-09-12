"""
classification/build_dataloaders.py

Builds train/val DataLoaders using the fold1_train / fold1_val split from
splits.py (Issue 1 fix -- NOT merged into one pool). Unified pool, diversity
by architecture only: every classification architecture calls
build_classification_dataloaders() with the same manifest and split.

ClassificationCropDataset's constructor params and fold1_stats.json's key
names are verified at runtime, not assumed, given this project's prior
mixups with both. crop_image_path is also re-rooted at load time, since the
manifest may carry a stale absolute prefix from whatever session originally
ran crop_extraction.py.

CHANGE: _verify_dataset_signature() now checks against every parameter this
file's two constructor calls actually pass (crop_paths, labels, mean, std,
augment, study_ids, conditions, levels) -- not just the original five. The
previous version only checked the first five, so a real mismatch (the
imported ClassificationCropDataset missing study_ids/conditions/levels)
slipped past it silently and only surfaced as a raw TypeError three calls
deep. That mismatch turned out to be a stale cached import in a long-running
kernel session (file on disk was already correct; the already-imported
module wasn't) -- this check can't fix a stale import by itself, but it now
at least fails with a clear, specific message instead of a confusing one.
"""
import sys
import json
import inspect
import os
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append("/kaggle/working/PDSCD")
sys.path.append("/kaggle/working/PDSCD/data_loading")
from classification.crop_dataset import ClassificationCropDataset
from classification.splits import load_classification_split

STATS_PATH = "/kaggle/working/PDSCD/configs/splits/stats/fold1_stats.json"

# Every parameter this file's two constructor calls actually pass below --
# not a partial list. If ClassificationCropDataset's __init__ is missing any
# of these, the calls further down WILL fail; better to fail here with a
# clear message than as a raw TypeError deep in build_classification_dataloaders().
REQUIRED_DATASET_PARAMS = {
    "crop_paths", "labels", "mean", "std", "augment",
    "study_ids", "conditions", "levels",
}
REQUIRED_MANIFEST_COLS = {"study_id", "crop_image_path", "severity", "condition", "level"}


def _verify_dataset_signature():
    sig = inspect.signature(ClassificationCropDataset.__init__)
    actual = set(sig.parameters.keys()) - {"self"}
    missing = REQUIRED_DATASET_PARAMS - actual
    if missing:
        raise TypeError(
            f"ClassificationCropDataset.__init__ is missing parameter(s) {sorted(missing)} "
            f"that this file's constructor calls require. Actual signature: {sig}. "
            f"Most likely cause: a stale cached import -- the file on disk was updated "
            f"(e.g. via git pull) after this module was already imported into this "
            f"kernel session. Restarting the kernel and re-running all cells resolves "
            f"this; simply re-running the import does not, since Python will not "
            f"re-execute an already-imported module. If restarting doesn't fix it, "
            f"confirm the committed file itself has these parameters before assuming "
            f"anything else."
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


def _resolve_crop_root(sample_path: str) -> str:
    """crop_image_path in the manifest may carry a stale absolute prefix from
    whatever session originally ran crop_extraction.py. Find the real current
    root by matching the path's tail against known candidate mount points,
    instead of trusting the baked-in prefix.

    ORDER FIX (this pass): this session's own /kaggle/working output is now
    checked BEFORE the read-only cross-account input Dataset
    (srejoni/crop-extraction). Previously the input Dataset was checked
    first and silently won any time its directory structure happened to
    match the manifest's tail path -- which is every time, since the crop
    naming convention (series_slug + level_slug + .npy) hasn't changed.
    That meant a from-scratch rerun's freshly-generated crops in
    /kaggle/working could be silently ignored in favor of an older,
    possibly pre-fix dataset, with no error or warning -- the function
    would report success either way. The input Dataset entry is kept as a
    fallback for legitimate cases where crops genuinely only exist there
    (e.g. an inference-only session that never ran crop_extraction.py
    itself), but it should never win over this session's own live output
    when both exist.
    """
    candidates = [
    "/kaggle/working/classification_data",
    "/kaggle/working/PDSCD/classification_data",
    "/kaggle/input/datasets/elenoremadams/crop-extraction/classification_data",
]
    tail = sample_path.split("classification_data", 1)[-1]
    for root in candidates:
        if os.path.exists(root + tail):
            print(f"[build_dataloaders] resolved crop root: {root}")
            return root
    raise FileNotFoundError(
        f"Could not find classification_data crops under any candidate root. "
        f"Checked: {candidates}"
    )


def _fix_crop_paths(df, root):
    fixed = df.copy()
    fixed["crop_image_path"] = fixed["crop_image_path"].apply(
        lambda p: root + p.split("classification_data", 1)[-1]
    )
    return fixed


def _load_and_check_manifest(manifest_path):
    """Loads labels_df here (so it can't be run out of order / undefined),
    checks the missing-severity assumption, and re-roots crop_image_path
    against wherever this session's data actually lives."""
    labels_df = pd.read_csv(manifest_path)

    missing_cols = REQUIRED_MANIFEST_COLS - set(labels_df.columns)
    if missing_cols:
        raise KeyError(f"Manifest at {manifest_path} missing columns: {missing_cols}")

    n_missing = labels_df["severity"].isna().sum()
    if n_missing > 0:
        print(f"[build_dataloaders] WARNING: {n_missing} rows with missing severity "
              f"found -- dropping them now (crop_extraction_fix.md §5 step not "
              f"confirmed applied upstream).")
        labels_df = labels_df.dropna(subset=["severity"]).reset_index(drop=True)

    crop_root = _resolve_crop_root(labels_df["crop_image_path"].iloc[0])
    labels_df = _fix_crop_paths(labels_df, crop_root)

    return labels_df


def build_classification_dataloaders(manifest_path, batch_size=16, num_workers=2):
    _verify_dataset_signature()
    mean, std = _load_fold1_stats()
    labels_df = _load_and_check_manifest(manifest_path)

    train_ids, val_ids = load_classification_split()
    train_df = labels_df[labels_df.study_id.isin(train_ids)].reset_index(drop=True)
    val_df = labels_df[labels_df.study_id.isin(val_ids)].reset_index(drop=True)

    assert set(train_df["study_id"]) & set(val_df["study_id"]) == set(), \
        "Leakage between classification train/val manifests"

    print(f"[build_dataloaders] train rows={len(train_df)}  val rows={len(val_df)}")

    train_dataset = ClassificationCropDataset(
        crop_paths=train_df["crop_image_path"].tolist(),
        labels=train_df["severity"].tolist(),
        mean=mean, std=std, augment=True,
        study_ids=train_df["study_id"].tolist(),
        conditions=train_df["condition"].tolist(),
        levels=train_df["level"].tolist(),
    )
    val_dataset = ClassificationCropDataset(
        crop_paths=val_df["crop_image_path"].tolist(),
        labels=val_df["severity"].tolist(),
        mean=mean, std=std, augment=False,
        study_ids=val_df["study_id"].tolist(),
        conditions=val_df["condition"].tolist(),
        levels=val_df["level"].tolist(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
