"""
PDSCD - preprocessing QA pass (steps 1-3).

Run this once per fold, BY THE PERSON WHO OWNS THAT FOLD, after
compute_fold_stats.py has finished. Budget ~15-20 minutes of your own
time to read the output and glance at two saved PNGs - the code itself
runs in a couple minutes.

Covers:
  Step 1 - patient-leakage check across ALL split files (run first,
           cheapest check, catches the single most damaging possible bug)
         + automated whole-dataset scan (NaN / blank / bad shape /
           missing files - stops and prints details if anything's wrong)
         + coordinate bounds check (after rescaling to the resized
           image's dimensions)
  Step 2 - visual grid spot-check (25 samples, spread across all 3
           series types) - saved as spot_check.png
  Step 3 - raw vs. processed comparison (n=10, not n=3) - saved as
           raw_vs_processed.png - PLUS an explicit left/right laterality
           check, since a flipped image with an unswapped label looks
           completely normal on a casual visual scan and RSNA's
           conditions are explicitly laterality-specific.

Run order matters: this script stops at step 1 if anything is wrong -
fix that before looking at any images. Steps 2-3 end in a judgment call
you make by eye.
"""

import os
import sys
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_loading"))
from preprocess import load_dicom_pixels, resize_image, TARGET_SIZE  # noqa: E402

RSNA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
IMAGES_ROOT = os.path.join(RSNA_ROOT, "train_images")
SPLITS_DIR = "../configs/splits"

FOLD = 4  # <-- set to YOUR OWN fold number before running. Run this on
          #     your own fold's output, not a teammate's - a bug specific
          #     to how one person's fold got processed won't show up
          #     when someone else runs this on their own fold.
N_GRID_SAMPLES = 25
N_RAW_VS_PROCESSED = 10


# ---------------------------------------------------------------------
# Step 1a: patient-leakage check across ALL split files
# ---------------------------------------------------------------------
def check_patient_leakage():
    print("=" * 60)
    print("STEP 1a: patient-leakage check across split files")
    print("=" * 60)

    test_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, "locked_test_ids.csv"))["study_id"])

    fold_ids = {}
    for f in range(1, 5):
        tr = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{f}_train_ids.csv"))["study_id"])
        va = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{f}_val_ids.csv"))["study_id"])
        fold_ids[f] = {"train": tr, "val": va}

    problems = []
    for f, d in fold_ids.items():
        if test_ids & d["train"]:
            problems.append(f"locked test overlaps fold{f} train: {len(test_ids & d['train'])} patients")
        if test_ids & d["val"]:
            problems.append(f"locked test overlaps fold{f} val: {len(test_ids & d['val'])} patients")

    for f, d in fold_ids.items():
        overlap = d["train"] & d["val"]
        if overlap:
            problems.append(f"fold{f} train/val overlap: {len(overlap)} patients")

    if problems:
        print("LEAKAGE FOUND - stop here, do not proceed:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print(f"Clean: 0 overlaps across locked test ({len(test_ids)} patients) "
          f"and all 4 folds' train/val sets.")


# ---------------------------------------------------------------------
# Step 1b: automated whole-dataset scan + count check
# ---------------------------------------------------------------------
def scan_fold_images(fold, expected_count=None):
    """
    expected_count: pass this in if you have an independent ground-truth
    count (e.g. from a manifest built before any processing touched the
    files). If you don't have one, leave it None - the scan still catches
    missing series directories, which is the most common silent-drop cause.
    """
    print("=" * 60)
    print(f"STEP 1b: automated scan - fold {fold} images")
    print("=" * 60)

    series_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_series_descriptions.csv"))
    train_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{fold}_train_ids.csv"))["study_id"])
    fold_series = series_df[series_df["study_id"].isin(train_ids)]

    issues = []
    n_scanned = 0
    n_attempted = 0

    for _, row in fold_series.iterrows():
        series_dir = os.path.join(IMAGES_ROOT, str(row["study_id"]), str(row["series_id"]))
        if not os.path.isdir(series_dir):
            issues.append((row["study_id"], row["series_id"], "missing_series_dir"))
            continue
        for fname in os.listdir(series_dir):
            if not fname.endswith(".dcm"):
                continue
            n_attempted += 1
            path = os.path.join(series_dir, fname)
            try:
                pixels = load_dicom_pixels(path)
                resized = resize_image(pixels, TARGET_SIZE)
            except Exception as e:
                issues.append((row["study_id"], fname, f"unreadable: {e}"))
                continue

            if np.isnan(resized).any():
                issues.append((row["study_id"], fname, "nan"))
            if resized.max() == resized.min():
                issues.append((row["study_id"], fname, "blank"))
            if resized.shape != (TARGET_SIZE, TARGET_SIZE):
                issues.append((row["study_id"], fname, "bad_shape"))
            n_scanned += 1

    print(f"Attempted: {n_attempted} files | Successfully scanned: {n_scanned} | "
          f"Issues: {len(issues)}")

    if expected_count is not None and n_attempted != expected_count:
        issues.append((None, None, f"count_mismatch: found {n_attempted}, "
                                    f"expected {expected_count} - something was silently "
                                    f"dropped upstream"))

    if issues:
        print(f"{len(issues)} issues found - STOP, fix these before step 2:")
        for i in issues[:30]:
            print(f"  - {i}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")
        raise SystemExit(1)

    print("Clean: 0 issues (no NaNs, no blank images, no shape mismatches, no missing files).")
    return n_attempted


# ---------------------------------------------------------------------
# Step 1c: coordinate bounds check (rescaled to match resized dimensions)
# ---------------------------------------------------------------------
def scan_coordinates(fold):
    print("=" * 60)
    print(f"STEP 1c: coordinate bounds check - fold {fold}")
    print("=" * 60)

    coords_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_label_coordinates.csv"))
    train_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{fold}_train_ids.csv"))["study_id"])
    fold_coords = coords_df[coords_df["study_id"].isin(train_ids)]

    issues = []
    for _, row in fold_coords.iterrows():
        dcm_path = os.path.join(IMAGES_ROOT, str(row["study_id"]), str(row["series_id"]),
                                 f"{row['instance_number']}.dcm")
        if not os.path.exists(dcm_path):
            issues.append((row["study_id"], row["series_id"], "missing_instance_for_coord"))
            continue

        pixels = load_dicom_pixels(dcm_path)
        orig_h, orig_w = pixels.shape[:2]
        scale_x = TARGET_SIZE / orig_w
        scale_y = TARGET_SIZE / orig_h
        rescaled_x = row["x"] * scale_x
        rescaled_y = row["y"] * scale_y

        if not (0 <= rescaled_x < TARGET_SIZE and 0 <= rescaled_y < TARGET_SIZE):
            issues.append((row["study_id"], row["series_id"], "coord_out_of_bounds"))

    print(f"Checked {len(fold_coords)} coordinates | Issues: {len(issues)}")
    if issues:
        print("STOP - coordinate rescaling is likely wrong (common cause: swapped "
              "width/height, or forgetting to rescale at all):")
        for i in issues[:20]:
            print(f"  - {i}")
        raise SystemExit(1)
    print("Clean: 0 out-of-bounds coordinates after rescaling.")


# ---------------------------------------------------------------------
# Step 2: visual grid spot-check - 25 samples, spread across series types
# ---------------------------------------------------------------------
def visual_grid_spot_check(fold, out_path="spot_check.png"):
    print("=" * 60)
    print(f"STEP 2: visual grid spot-check - fold {fold} -> {out_path}")
    print("=" * 60)

    coords_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_label_coordinates.csv"))
    series_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_series_descriptions.csv"))
    train_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{fold}_train_ids.csv"))["study_id"])

    merged = coords_df.merge(series_df, on=["study_id", "series_id"])
    merged = merged[merged["study_id"].isin(train_ids)]

    series_types = merged["series_description"].unique()
    per_type = max(N_GRID_SAMPLES // len(series_types), 1)

    parts = []
    for st in series_types:
        subset = merged[merged["series_description"] == st]
        parts.append(subset.sample(min(per_type, len(subset))))
    samples = pd.concat(parts).sample(frac=1).reset_index(drop=True).head(N_GRID_SAMPLES)

    n_cols = 5
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    for ax, (_, row) in zip(np.array(axes).flat, samples.iterrows()):
        dcm_path = os.path.join(IMAGES_ROOT, str(row["study_id"]), str(row["series_id"]),
                                 f"{row['instance_number']}.dcm")
        try:
            pixels = load_dicom_pixels(dcm_path)
            resized = resize_image(pixels, TARGET_SIZE)
            scale_x = TARGET_SIZE / pixels.shape[1]
            scale_y = TARGET_SIZE / pixels.shape[0]
            ax.imshow(resized, cmap="gray")
            ax.scatter(row["x"] * scale_x, row["y"] * scale_y, c="red", s=20)
            ax.set_title(f"{row['series_description']}\n{row['condition']}", fontsize=7)
        except Exception as e:
            ax.set_title(f"ERROR: {e}", fontsize=7, color="red")
        ax.axis("off")

    for ax in np.array(axes).flat[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"Saved {out_path}. Look at it once: do the red dots sit on discs/vertebrae, "
          f"and do the images look like sensible spine scans? ~30 seconds, judgment call.")


# ---------------------------------------------------------------------
# Step 3: raw vs. processed (n=10) + explicit laterality check
# ---------------------------------------------------------------------
def get_patient_side_at_column(dcm, x_col, image_width):
    """
    Best-effort heuristic using DICOM ImageOrientationPatient to map an
    image column position to anatomical left/right, per the LPS
    convention (DICOM patient x-axis points toward patient LEFT).

    This is a heuristic aid, NOT a certainty - treat "unknown" or a
    surprising result as "go look at the image yourself," not as
    ground truth. Most reliable on Axial T2, where left/right structures
    appear within the same slice; sagittal series will usually return
    "unknown" since the row direction isn't aligned with the L-R axis.
    """
    if not hasattr(dcm, "ImageOrientationPatient"):
        return "unknown"

    orientation = [float(v) for v in dcm.ImageOrientationPatient]
    row_cosine = orientation[0:3]  # direction of increasing column index (image x)

    dominant_axis = int(np.argmax(np.abs(row_cosine)))
    sign = np.sign(row_cosine[dominant_axis])
    if dominant_axis != 0:
        return "unknown"

    midpoint = image_width / 2
    moving_toward_positive_x = (x_col > midpoint) == (sign > 0)
    return "left" if moving_toward_positive_x else "right"


def raw_vs_processed_and_laterality_check(fold, n=N_RAW_VS_PROCESSED, out_path="raw_vs_processed.png"):
    print("=" * 60)
    print(f"STEP 3: raw vs. processed (n={n}) + laterality check - fold {fold} -> {out_path}")
    print("=" * 60)

    coords_df = pd.read_csv(os.path.join(RSNA_ROOT, "train_label_coordinates.csv"))
    train_ids = set(pd.read_csv(os.path.join(SPLITS_DIR, f"fold{fold}_train_ids.csv"))["study_id"])
    fold_coords = coords_df[coords_df["study_id"].isin(train_ids)]

    samples = fold_coords.sample(n)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))

    for i, (_, row) in enumerate(samples.iterrows()):
        dcm_path = os.path.join(IMAGES_ROOT, str(row["study_id"]), str(row["series_id"]),
                                 f"{row['instance_number']}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        pixels = dcm.pixel_array.astype(np.float32)
        resized = resize_image(pixels, TARGET_SIZE)
        scale_x = TARGET_SIZE / pixels.shape[1]
        scale_y = TARGET_SIZE / pixels.shape[0]

        axes[i, 0].imshow(pixels, cmap="gray")
        axes[i, 0].scatter(row["x"], row["y"], c="red")
        axes[i, 0].set_title(f"raw: {row['condition']}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(resized, cmap="gray")
        axes[i, 1].scatter(row["x"] * scale_x, row["y"] * scale_y, c="red")
        axes[i, 1].set_title("processed", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"Saved {out_path}. Confirm the dot lands on the SAME anatomical spot in "
          f"both panels for every row - not just 'somewhere on the image.'")

    lateral_rows = fold_coords[fold_coords["condition"].str.contains("left|right", case=False, na=False)]
    if lateral_rows.empty:
        print("No left/right-labeled conditions found in this fold's coordinates - skip laterality check.")
        return

    check_row = lateral_rows.sample(1).iloc[0]
    dcm_path = os.path.join(IMAGES_ROOT, str(check_row["study_id"]), str(check_row["series_id"]),
                             f"{check_row['instance_number']}.dcm")
    dcm = pydicom.dcmread(dcm_path)
    detected_side = get_patient_side_at_column(dcm, check_row["x"], dcm.pixel_array.shape[1])
    labeled_side = "left" if "left" in check_row["condition"].lower() else "right"

    print(f"\nLaterality check: study {check_row['study_id']}, condition = "
          f"'{check_row['condition']}' (labeled side = {labeled_side})")
    print(f"  DICOM-orientation-derived side at this coordinate: {detected_side}")
    if detected_side == "unknown":
        print("  Could not determine automatically (likely a sagittal series, where row "
              "direction isn't aligned with the L-R axis) - confirm this one by eye instead.")
    elif detected_side != labeled_side:
        print("  MISMATCH - STOP. This strongly suggests a left/right flip or swap bug. "
              "Do not proceed to training until this is resolved.")
        raise SystemExit(1)
    else:
        print("  Match - labeled side agrees with the orientation-derived side for this "
              "sample. Spot-check 2-3 more by hand before fully trusting it - this is a "
              "heuristic, not a proof.")


if __name__ == "__main__":
    check_patient_leakage()
    scan_fold_images(FOLD)
    scan_coordinates(FOLD)
    visual_grid_spot_check(FOLD)
    raw_vs_processed_and_laterality_check(FOLD)
    print("\nSteps 1-3 complete. Per your PR review workflow: have a DIFFERENT teammate "
          "re-run this script against your fold's output at least once before merging "
          "into main - fresh eyes catch what the author's blind spots miss, especially "
          "for the laterality check.")
