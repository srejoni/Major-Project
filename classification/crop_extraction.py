"""
classification/crop_extraction.py

Fixes applied after an OSError (disk quota exceeded) on the first full run:
- Crops saved as uint8, not float32 -- source data is inherently 8-bit
  (percentile_clip_to_uint8), so float32 was a needless 4x size inflation.
  Normalization to float happens at load time in crop_dataset.py instead.
- Crops are now keyed by (series_desc, level), not (condition, level) --
  left/right foraminal share one Sagittal T1 crop per level, left/right
  subarticular share one Axial T2 crop per level. Previously each was
  saved twice under different filenames.
- Manifest is written incrementally, one study at a time, not only at the
  end -- a crash now loses at most the in-progress study, not the whole run.
- Resumable: on restart, study_ids already present in an existing
  manifest.csv are skipped.
"""

import os
import csv
import glob
import sys

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from ultralytics import YOLO

sys.path.append("/kaggle/working/PDSCD")
from data_loading.preprocess import load_dicom_pixels
from classification.splits import load_classification_split

REPO_ROOT = "/kaggle/working/PDSCD"
DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
YOLO_WEIGHTS = f"{REPO_ROOT}/localization/weights/best.pt"
CROPS_DIR = "/kaggle/working/classification_data/crops"
MANIFEST_CSV = "/kaggle/working/classification_data/manifest.csv"

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

SERIES_BY_CONDITION = {
    "spinal_canal_stenosis": "Sagittal T2/STIR",
    "left_neural_foraminal_narrowing": "Sagittal T1",
    "right_neural_foraminal_narrowing": "Sagittal T1",
    "left_subarticular_stenosis": "Axial T2",
    "right_subarticular_stenosis": "Axial T2",
}

PER_SERIES_CONFIG = {
    "Sagittal T1":      {"margin": 0.5, "n_slices": 1},
    "Sagittal T2/STIR": {"margin": 0.5, "n_slices": 1},
    "Axial T2":         {"margin": 0.7, "n_slices": 3},
}

SEVERITY_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
YOLO_CONF_THRESHOLD = 0.25
CROP_SIZE = 224
MANIFEST_FIELDS = ["study_id", "crop_image_path", "condition", "level", "severity", "detection_conf"]


def level_slug(level):
    return level.lower().replace("/", "_")


def series_slug(series_desc):
    return series_desc.lower().replace(" ", "_").replace("/", "_")


def percentile_clip_to_uint8(img, lo=1, hi=99):
    lo_v, hi_v = np.percentile(img, [lo, hi])
    img = np.clip(img, lo_v, hi_v)
    if hi_v - lo_v < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo_v) / (hi_v - lo_v) * 255).astype(np.uint8)


def get_ordered_slices(series_dir):
    dcm_paths = glob.glob(os.path.join(series_dir, "*.dcm"))
    tagged = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            tagged.append((int(ds.InstanceNumber), p))
        except Exception:
            continue
    tagged.sort(key=lambda t: t[0])
    return [p for _, p in tagged]


def detect_best_per_level(model, ordered_slices):
    best = {}
    for dcm_path in ordered_slices:
        pixels = load_dicom_pixels(dcm_path)
        img8 = percentile_clip_to_uint8(pixels)
        rgb = np.stack([img8] * 3, axis=-1)
        results = model.predict(rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        for box in results.boxes:
            lvl, conf = int(box.cls.item()), float(box.conf.item())
            if lvl not in best or conf > best[lvl][0]:
                best[lvl] = (conf, dcm_path, box.xyxy[0].cpu().numpy(), img8.shape)
    return best


def crop_one_slice(dcm_path, xyxy, shape, margin):
    """Returns uint8 -- no float32 cast. Source is already 8-bit."""
    pixels = load_dicom_pixels(dcm_path)
    img8 = percentile_clip_to_uint8(pixels)
    h, w = shape
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - bw * margin), max(0, y1 - bh * margin)
    x2, y2 = min(w, x2 + bw * margin), min(h, y2 + bh * margin)
    crop = img8[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None
    return np.array(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE)))  # uint8


def build_stacked_crop(ordered_slices, center_dcm_path, xyxy, shape, series_desc):
    cfg = PER_SERIES_CONFIG[series_desc]
    margin, n_slices = cfg["margin"], cfg["n_slices"]

    if n_slices == 1:
        crop = crop_one_slice(center_dcm_path, xyxy, shape, margin)
        if crop is None:
            return None
        return np.stack([crop, crop, crop], axis=-1)  # uint8, (H,W,3)

    center_idx = ordered_slices.index(center_dcm_path)
    idxs = [max(0, center_idx - 1), center_idx, min(len(ordered_slices) - 1, center_idx + 1)]
    crops = []
    for idx in idxs:
        c = crop_one_slice(ordered_slices[idx], xyxy, shape, margin)
        if c is None:
            return None
        crops.append(c)
    return np.stack(crops, axis=-1)  # uint8, (H,W,3)


def load_completed_study_ids():
    """Resume support -- study_ids already fully written to an existing manifest."""
    if not os.path.exists(MANIFEST_CSV):
        return set()
    existing = pd.read_csv(MANIFEST_CSV)
    return set(existing["study_id"].unique())


def append_rows_to_manifest(rows):
    file_exists = os.path.exists(MANIFEST_CSV)
    with open(MANIFEST_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def process_study(model, study_id, train_df, series_df, study_dir):
    """Returns this study's manifest rows, deduping shared crops by
    (series_desc, level) instead of writing one per condition."""
    cache = {}
    for series_desc in set(SERIES_BY_CONDITION.values()):
        series_ids = series_df[
            (series_df.study_id == study_id) &
            (series_df.series_description == series_desc)
        ].series_id.tolist()

        merged_best, merged_ordered = {}, {}
        for sid in series_ids:
            sdir = f"{DATA_ROOT}/train_images/{study_id}/{sid}"
            ordered = get_ordered_slices(sdir)
            if not ordered:
                continue
            best = detect_best_per_level(model, ordered)
            for lvl, val in best.items():
                if lvl not in merged_best or val[0] > merged_best[lvl][0]:
                    merged_best[lvl] = val
                    merged_ordered[lvl] = ordered
        cache[series_desc] = (merged_best, merged_ordered)

    # Save each (series_desc, level) crop ONCE, cache the path for reuse
    # across conditions that share the same source series.
    crop_path_by_series_level = {}
    for series_desc, (merged_best, merged_ordered) in cache.items():
        for level_idx, level in enumerate(LEVELS):
            det = merged_best.get(level_idx)
            if det is None:
                continue
            conf, dcm_path, xyxy, shape = det
            ordered_slices = merged_ordered[level_idx]
            stacked = build_stacked_crop(ordered_slices, dcm_path, xyxy, shape, series_desc)
            if stacked is None:
                continue
            crop_path = os.path.join(
                study_dir, f"{series_slug(series_desc)}_{level_slug(level)}.npy"
            )
            np.save(crop_path, stacked)
            crop_path_by_series_level[(series_desc, level_idx)] = (crop_path, conf)

    rows = []
    for condition, series_desc in SERIES_BY_CONDITION.items():
        for level_idx, level in enumerate(LEVELS):
            entry = crop_path_by_series_level.get((series_desc, level_idx))
            if entry is None:
                continue
            crop_path, conf = entry

            col = f"{condition}_{level_slug(level)}"
            sev_raw = train_df.loc[study_id].get(col)
            if pd.isna(sev_raw):
                continue

            rows.append({
                "study_id": study_id,
                "crop_image_path": crop_path,
                "condition": condition,
                "level": level,
                "severity": SEVERITY_MAP[sev_raw],
                "detection_conf": conf,
            })
    return rows


def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    train_df = pd.read_csv(f"{DATA_ROOT}/train.csv").set_index("study_id")
    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    model = YOLO(YOLO_WEIGHTS)

    train_ids, val_ids = load_classification_split()
    unified_ids = sorted(set(train_ids) | set(val_ids))  # excludes locked test set entirely

    completed = load_completed_study_ids()
    if completed:
        print(f"Resuming -- {len(completed)} studies already in manifest, skipping them")

    total_rows_written = 0
    for i, study_id in enumerate(unified_ids):
        if study_id in completed:
            continue
        study_dir = os.path.join(CROPS_DIR, str(study_id))
        os.makedirs(study_dir, exist_ok=True)
        rows = process_study(model, study_id, train_df, series_df, study_dir)
        if rows:
            append_rows_to_manifest(rows)
            total_rows_written += len(rows)
        if i % 50 == 0:
            print(f"[{i}/{len(unified_ids)}] processed, {total_rows_written} rows written, "
                  f"{i/len(unified_ids)*100:.0f}% done")

    print(f"Done. {total_rows_written} new rows written to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
