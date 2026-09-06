"""
classification/margin_tuning.py

Step 1 tooling (crop_extraction_fix.md, Option 2): visual tuning of
per-series crop margins on a small sample rather than assuming or fully
sweeping. This run doubles as the required Step 8 spot-check pass later,
so it isn't extra work on top of what the pipeline already needed.

Run this AFTER verify_paths.py confirms all 4 required paths as FOUND --
DATA_ROOT and YOLO_WEIGHTS_PATH below should be updated to match its
output before running, not assumed.

get_reference_frame() is wired directly to the project's existing,
QA-verified data_loading.preprocess.load_dicom_pixels -- this was a
deliberate NotImplementedError stub until this point.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

sys.path.append("/kaggle/working/PDSCD")
from data_loading.preprocess import load_dicom_pixels

# ---- Paths -- confirm/update these from verify_paths.py's printed output ----
DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
REPO_ROOT = "/kaggle/working/PDSCD"
YOLO_WEIGHTS_PATH = f"{REPO_ROOT}/localization/weights/best.pt"  # confirmed location, per progress doc §4

TRAIN_SERIES_CSV = f"{DATA_ROOT}/train_series_descriptions.csv"
TRAIN_COORDS_CSV = f"{DATA_ROOT}/train_label_coordinates.csv"
TRAIN_IMAGES_DIR = f"{DATA_ROOT}/train_images"

OUTPUT_DIR = "/kaggle/working/margin_tuning_output"

N_STUDIES_PER_SERIES = 18
YOLO_CONF_THRESHOLD = 0.25
CROP_SIZE = 224
SEED = 42

# Candidate margins per series -- seeded from anatomy priors:
#   Sagittal (foraminal/canal): foramen sits laterally off the vertebral
#     edge, canal roughly centered on the box -- moderate range.
#   Axial (subarticular): lateral recess spreads widest across the canal
#     cross-section -- needs the most generous range.
CANDIDATE_MARGINS = {
    "Sagittal T1":      [0.30, 0.40, 0.50],
    "Sagittal T2/STIR": [0.30, 0.40, 0.50],
    "Axial T2":         [0.50, 0.60, 0.70],
}


def percentile_clip_to_uint8(img, lo=1, hi=99):
    lo_v, hi_v = np.percentile(img, [lo, hi])
    img = np.clip(img, lo_v, hi_v)
    if hi_v - lo_v < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo_v) / (hi_v - lo_v) * 255).astype(np.uint8)


def get_reference_frame(dcm_path):
    """Wired to the existing, verified loader -- no longer a stub."""
    pixels = load_dicom_pixels(dcm_path)
    return percentile_clip_to_uint8(pixels)


def best_detection_for_study(model, series_dir):
    """Single highest-confidence detection across the series -- enough for
    a quick visual check; NOT the multi-slice logic crop_extraction.py
    uses for axial at extraction time."""
    best = None
    for fname in sorted(os.listdir(series_dir)):
        if not fname.endswith(".dcm"):
            continue
        dcm_path = os.path.join(series_dir, fname)
        img8 = get_reference_frame(dcm_path)
        rgb = np.stack([img8] * 3, axis=-1)
        results = model.predict(rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf.item())
            if best is None or conf > best[0]:
                best = (conf, dcm_path, box.xyxy[0].cpu().numpy(), img8.shape)
    return best


def crop_at_margin(dcm_path, xyxy, shape, margin):
    img8 = get_reference_frame(dcm_path)
    h, w = shape
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - bw * margin), max(0, y1 - bh * margin)
    x2, y2 = min(w, x2 + bw * margin), min(h, y2 + bh * margin)
    crop = img8[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    return np.array(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE)))


def sample_studies_for_series(series_desc, series_df, n):
    study_ids = series_df[series_df.series_description == series_desc].study_id.unique().tolist()
    random.Random(SEED).shuffle(study_ids)
    return study_ids[:n]


def print_adjacent_level_spacing_warning(coords_df):
    """Cheap sanity check only -- NOT used to set margins, just flags
    whether some studies have unusually tight adjacent-level spacing,
    which raises bleed risk regardless of the visually-chosen margin."""
    spacings = []
    for _, group in coords_df.groupby("study_id"):
        ys = sorted(group["y"].tolist())
        spacings.extend(b - a for a, b in zip(ys, ys[1:]))
    if spacings:
        print(f"[bleed-risk check] median adjacent-level y-spacing across "
              f"sample: {float(np.median(spacings)):.1f}px -- sanity check "
              f"only, not used to set margins")


def build_grid(study_crops, series_desc, margins):
    n_studies, n_margins = len(study_crops), len(margins)
    fig, axes = plt.subplots(n_studies, n_margins, figsize=(3 * n_margins, 3 * n_studies))
    if n_studies == 1:
        axes = axes.reshape(1, -1)
    for row, (study_id, crops) in enumerate(study_crops.items()):
        for col, margin in enumerate(margins):
            axes[row, col].imshow(crops[col], cmap="gray")
            axes[row, col].set_title(f"study={study_id}\nmargin={margin}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle(f"{series_desc} -- candidate margin comparison")
    fig.tight_layout()
    safe_name = series_desc.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_grid.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[margin_tuning] wrote {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    series_df = pd.read_csv(TRAIN_SERIES_CSV)
    coords_df = pd.read_csv(TRAIN_COORDS_CSV)
    model = YOLO(YOLO_WEIGHTS_PATH)

    print_adjacent_level_spacing_warning(coords_df)

    for series_desc, margins in CANDIDATE_MARGINS.items():
        sample_ids = sample_studies_for_series(series_desc, series_df, N_STUDIES_PER_SERIES)
        study_crops = {}

        for study_id in sample_ids:
            series_ids = series_df[
                (series_df.study_id == study_id) &
                (series_df.series_description == series_desc)
            ].series_id.tolist()
            if not series_ids:
                continue
            series_dir = os.path.join(TRAIN_IMAGES_DIR, str(study_id), str(series_ids[0]))
            if not os.path.isdir(series_dir):
                continue

            det = best_detection_for_study(model, series_dir)
            if det is None:
                continue
            conf, dcm_path, xyxy, shape = det
            study_crops[study_id] = [crop_at_margin(dcm_path, xyxy, shape, m) for m in margins]

        if study_crops:
            build_grid(study_crops, series_desc, margins)
        else:
            print(f"[margin_tuning] no usable studies found for {series_desc} -- check paths")


if __name__ == "__main__":
    main()
