"""
migrate_axial_crops.py

One-time migration: applies the axial laterality fix
(detect_best_per_level_axial_dual / merge_series_detections_axial_dual,
added to crop_extraction.py) to a manifest.csv that was already fully
built under the OLD shared-crop logic for Axial T2.

Mirrors migrate_foraminal_crops.py exactly -- same reasoning applies:

Why not just re-run crop_extraction.py's main() again: load_completed_
study_ids() skips any study_id already present in manifest.csv,
REGARDLESS of which conditions' rows exist for it. Deleting only the
subarticular-stenosis rows and re-running main() would NOT trigger
reprocessing -- the study_id would still be "completed" (Sagittal T1 /
Sagittal T2-STIR rows still present) and skipped entirely, silently
leaving the subarticular rows permanently missing.

Why not a full from-scratch re-run: Sagittal T1 (foraminal narrowing) and
Sagittal T2/STIR (canal stenosis) crops are UNCHANGED by this fix -- this
migration only touches Axial T2. A full re-run repeats real YOLO
inference over series that don't need it, expensive given this project's
already-documented GPU-quota / second-Kaggle-account constraints.

This script instead:
  1. Backs up the existing manifest.
  2. Removes ONLY left/right_subarticular_stenosis rows (identifying
     affected studies as a side effect).
  3. Deletes the now-invalid shared Axial T2 crop files on disk (old
     naming has no _left/_right suffix, so old and new filenames can
     never collide -- this deletion is cleanup, not a correctness
     requirement).
  4. Re-runs ONLY the Axial T2 dual-mode detection for those studies,
     appending correctly-split subarticular-stenosis rows back.

Resumable: if interrupted, re-running skips studies that already have
subarticular rows back in the manifest, the same pattern
crop_extraction.py's own load_completed_study_ids() uses.

PREREQUISITE: crop_extraction.py must already have AXIAL_DUAL_MODE_SERIES,
detect_best_per_level_axial_dual(), merge_series_detections_axial_dual(),
and the broadened SIDE_SPLIT_SERIES merged in (this patch). This script
imports them directly and will fail loudly at import time if they're
missing -- intentional, not a bug to work around.

Usage:
    python migrate_axial_crops.py
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append("/kaggle/working/PDSCD")
from classification.crop_extraction import (
    DATA_ROOT, YOLO_WEIGHTS, CROPS_DIR, LEVELS, SEVERITY_MAP,
    MANIFEST_CSV, MANIFEST_FIELDS,
    level_slug, series_slug, build_stacked_crop,
    merge_series_detections_axial_dual, validate_yolo_class_mapping,
    append_rows_to_manifest,
)

SUBARTICULAR_CONDITIONS = {"left_subarticular_stenosis", "right_subarticular_stenosis"}
SPLIT_SERIES_DESC = "Axial T2"


def backup_manifest():
    backup_path = MANIFEST_CSV.replace(".csv", "_pre_axial_laterality_fix_backup.csv")
    if not os.path.exists(backup_path):
        shutil.copy(MANIFEST_CSV, backup_path)
        print(f"[migrate_axial] backed up manifest to {backup_path}")
    else:
        print(f"[migrate_axial] backup already exists at {backup_path} -- not overwriting")


def strip_old_subarticular_rows(manifest_df):
    affected_mask = manifest_df["condition"].isin(SUBARTICULAR_CONDITIONS)
    affected_study_ids = sorted(manifest_df.loc[affected_mask, "study_id"].unique())
    kept_df = manifest_df.loc[~affected_mask].reset_index(drop=True)
    print(f"[migrate_axial] removing {int(affected_mask.sum())} old subarticular-stenosis rows "
          f"across {len(affected_study_ids)} studies")
    return kept_df, affected_study_ids


def delete_old_shared_crop_files(affected_study_ids):
    deleted = 0
    for study_id in affected_study_ids:
        study_dir = os.path.join(CROPS_DIR, str(study_id))
        for level in LEVELS:
            old_path = os.path.join(
                study_dir, f"{series_slug(SPLIT_SERIES_DESC)}_{level_slug(level)}.npy"
            )
            if os.path.exists(old_path):
                os.remove(old_path)
                deleted += 1
    print(f"[migrate_axial] deleted {deleted} old shared Axial T2 crop files")


def load_already_reprocessed_study_ids():
    """Resume support -- studies that already have NEW (side-split)
    subarticular rows back in the manifest, in case a previous run of
    this script was interrupted partway through."""
    if not os.path.exists(MANIFEST_CSV):
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    return set(df.loc[df["condition"].isin(SUBARTICULAR_CONDITIONS), "study_id"].unique())


def reprocess_subarticular_for_study(model, study_id, train_df, series_df):
    """Mirrors crop_extraction.process_study()'s Axial T2 branch only --
    narrowed to just this one series_desc instead of looping over all of
    SERIES_BY_CONDITION, since Sagittal T1 / Sagittal T2-STIR rows for
    this study are untouched and already correct in the kept manifest."""
    study_dir = os.path.join(CROPS_DIR, str(study_id))
    os.makedirs(study_dir, exist_ok=True)

    sides = merge_series_detections_axial_dual(
        model, study_id, series_df, DATA_ROOT, SPLIT_SERIES_DESC
    )
    rows = []
    for side, (merged_best, merged_ordered, contributor_count) in sides.items():
        condition = f"{side}_subarticular_stenosis"
        for level_idx, level in enumerate(LEVELS):
            det = merged_best.get(level_idx)
            if det is None:
                continue
            conf, dcm_path, xyxy, shape = det
            ordered_slices = merged_ordered[level_idx]
            stacked = build_stacked_crop(ordered_slices, dcm_path, xyxy, shape, SPLIT_SERIES_DESC)
            if stacked is None:
                continue
            crop_path = os.path.join(
                study_dir,
                f"{series_slug(SPLIT_SERIES_DESC)}_{level_slug(level)}_{side}.npy",
            )
            np.save(crop_path, stacked)

            col = f"{condition}_{level_slug(level)}"
            sev_raw = train_df.loc[study_id].get(col)
            if pd.isna(sev_raw):
                continue

            was_dup = contributor_count.get(level_idx, 1) > 1
            rows.append({
                "study_id": study_id,
                "crop_image_path": crop_path,
                "condition": condition,
                "level": level,
                "severity": SEVERITY_MAP[sev_raw],
                "detection_conf": conf,
                "was_duplicate_resolved": was_dup,
            })
    return rows


def main():
    backup_manifest()

    manifest_df = pd.read_csv(MANIFEST_CSV)
    kept_df, affected_study_ids = strip_old_subarticular_rows(manifest_df)
    kept_df.to_csv(MANIFEST_CSV, index=False)
    print(f"[migrate_axial] manifest.csv rewritten with {len(kept_df)} rows "
          f"(subarticular rows pending re-add)")

    delete_old_shared_crop_files(affected_study_ids)

    train_df = pd.read_csv(f"{DATA_ROOT}/train.csv").set_index("study_id")
    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    model = YOLO(YOLO_WEIGHTS)
    validate_yolo_class_mapping(model)

    already_done = load_already_reprocessed_study_ids()
    if already_done:
        print(f"[migrate_axial] resuming -- {len(already_done)} studies already reprocessed, skipping them")

    total_rows_written = 0
    for i, study_id in enumerate(affected_study_ids):
        if study_id in already_done:
            continue
        rows = reprocess_subarticular_for_study(model, study_id, train_df, series_df)
        if rows:
            append_rows_to_manifest(rows)
            total_rows_written += len(rows)
        if i % 50 == 0:
            print(f"[migrate_axial] [{i}/{len(affected_study_ids)}] reprocessed, "
                  f"{total_rows_written} rows written so far")

    print(f"[migrate_axial] done. {total_rows_written} new (side-split) subarticular-stenosis "
          f"rows written to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
