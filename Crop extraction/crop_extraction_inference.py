"""
classification/crop_extraction_inference.py

Inference-time counterpart to crop_extraction.py's per-study crop logic.

crop_extraction.py's process_study() cannot be reused directly for real
inference: it requires severity labels from train.csv and silently skips
any (condition, level) whose crop-worthy detection exists but has no label
in train.csv. At inference time there is no severity to check -- every
study needs every (condition, level) resolved to either "a crop exists" or
"no crop was ever producible," full stop, with no label-based filtering
in between.

FIX (previous pass): the series_id merge/dedup loop -- multiple series_ids
sharing one series_description, merged down to one highest-confidence
detection per level -- was previously duplicated here as an inline copy
of crop_extraction.py::process_study()'s equivalent loop. Now imports and
calls crop_extraction.merge_series_detections() -- the single shared
implementation -- instead of reimplementing it.

NAMING: the duplicate-resolution flag is "was_duplicate_resolved" here,
matching crop_extraction.py's manifest column of the same name.

FIX (THIS pass -- laterality): crop_extraction.py's module docstring has
the full diagnostic chain. Short version: left/right foraminal narrowing
were confirmed to always receive an identical crop (both conditions map to
"Sagittal T1" in SERIES_BY_CONDITION, and the merge was keyed by
(series_desc, level) only). train_label_coordinates.csv confirms left and
right genuinely differ by slice in ~99.6% of foraminal-narrowing cases,
and a geometric proxy (DICOM ImagePositionPatient projected onto the
imaging plane's normal vector) was verified at 100% sign consistency
across 125 real left/right pairs to reliably tell which slice is which
side. This file now calls crop_extraction.merge_series_detections_by_side()
for series in SIDE_SPLIT_SERIES (currently just "Sagittal T1") instead of
the single-crop merge_series_detections(), and picks the correct side per
condition via crop_extraction.condition_side(). Axial T2 (subarticular
stenosis) is UNCHANGED -- still one shared crop per level -- since the
side-split proxy is not yet validated for axial geometry.

Because this now calls the exact same shared functions crop_extraction.py
uses for training, the two stay in lockstep by construction: a study whose
foraminal-narrowing crops were correctly split during training will be
split the identical way at inference, using the identical selection logic
-- not just "equivalent code," the same function calls.
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

from classification.crop_extraction import (
    LEVELS, SERIES_BY_CONDITION, PER_SERIES_CONFIG, SIDE_SPLIT_SERIES,
    build_stacked_crop, merge_series_detections, merge_series_detections_by_side,
    condition_side,
)


def extract_study_crops(model, study_id, series_df, data_root):
    """Returns dict keyed by (condition, level) -> crop info.
    NO entry exists for a (condition, level) that had no usable detection --
    that absence is exactly what assemble_study_output.py turns into
    status="missing_detection".

    For series in SIDE_SPLIT_SERIES (Sagittal T1), left/right conditions
    now resolve to genuinely different crops via the geometric side-split
    proxy, matching training exactly (same shared functions, same
    selection logic). For other series (Sagittal T2/STIR, Axial T2),
    behavior is unchanged -- one shared crop per level.

    Each returned entry also carries "was_duplicate_resolved": bool -- True
    if more than one series_id existed for that series_description/side
    and contributed a candidate detection for that level, i.e. a real
    duplicate was resolved by keeping the higher-confidence one.
    assemble_study_output.py uses this to set status="duplicate_resolved"
    instead of "ok"."""
    # (series_desc, level, side_or_None) -> crop info dict
    crop_by_series_level_side = {}

    for series_desc in set(SERIES_BY_CONDITION.values()):
        n_slices = PER_SERIES_CONFIG[series_desc]["n_slices"]

        if series_desc in SIDE_SPLIT_SERIES:
            sides = merge_series_detections_by_side(
                model, study_id, series_df, data_root, series_desc
            )
            for side, (merged_best, merged_ordered, contributor_count) in sides.items():
                for level_idx, level in enumerate(LEVELS):
                    det = merged_best.get(level_idx)
                    if det is None:
                        continue
                    conf, dcm_path, xyxy, shape = det
                    ordered_slices = merged_ordered[level_idx]
                    stacked = build_stacked_crop(ordered_slices, dcm_path, xyxy, shape, series_desc)
                    if stacked is None:
                        continue
                    was_duplicate_resolved = contributor_count.get(level_idx, 1) > 1
                    crop_by_series_level_side[(series_desc, level, side)] = {
                        "crop": stacked,
                        "series_description": series_desc,
                        "n_slices_used": n_slices,
                        "yolo_detection_confidence": conf,
                        "was_duplicate_resolved": was_duplicate_resolved,
                    }
        else:
            merged_best, merged_ordered, contributor_count = merge_series_detections(
                model, study_id, series_df, data_root, series_desc
            )
            for level_idx, level in enumerate(LEVELS):
                det = merged_best.get(level_idx)
                if det is None:
                    continue
                conf, dcm_path, xyxy, shape = det
                ordered_slices = merged_ordered[level_idx]
                stacked = build_stacked_crop(ordered_slices, dcm_path, xyxy, shape, series_desc)
                if stacked is None:
                    continue
                was_duplicate_resolved = contributor_count.get(level_idx, 1) > 1
                crop_by_series_level_side[(series_desc, level, None)] = {
                    "crop": stacked,
                    "series_description": series_desc,
                    "n_slices_used": n_slices,
                    "yolo_detection_confidence": conf,
                    "was_duplicate_resolved": was_duplicate_resolved,
                }

    result = {}
    for condition, series_desc in SERIES_BY_CONDITION.items():
        side = condition_side(condition) if series_desc in SIDE_SPLIT_SERIES else None
        for level in LEVELS:
            entry = crop_by_series_level_side.get((series_desc, level, side))
            if entry is not None:
                result[(condition, level)] = entry
    return result
