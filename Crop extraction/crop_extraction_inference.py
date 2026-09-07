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

This module reuses crop_extraction.py's actual detection/cropping helpers
(same YOLO model, same margins, same slice-stacking) rather than
reimplementing them, since duplicating that logic creates a second place
for it to drift out of sync -- exactly the class of bug this project has
already hit (BOX_SIZE_FRAC mismatch, percentile-vs-zscore normalization
mismatch, stale cached imports).

CHANGE (duplicate_resolved fix): when a study has more than one series_id
for the same series_description (e.g. two Sagittal T1 acquisitions), the
merge step below already picks the higher-confidence one per level -- but
previously that fact was discarded once the merge completed, with no
record left that a duplicate ever existed. pdscd_output_format.md promises
downstream consumers a "duplicate_resolved" status for exactly this
situation, so it can no longer be silently thrown away here. Each crop
entry now carries "had_duplicate": True whenever more than one series_id
contributed candidate detections for that (series_description, level),
regardless of which one ultimately won on confidence.
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

from classification.crop_extraction import (
    LEVELS, SERIES_BY_CONDITION, PER_SERIES_CONFIG,
    get_ordered_slices, detect_best_per_level, build_stacked_crop,
)


def extract_study_crops(model, study_id, series_df, data_root):
    """Returns dict keyed by (condition, level) -> crop info.
    NO entry exists for a (condition, level) that had no usable detection --
    that absence is exactly what assemble_study_output.py turns into
    status="missing_detection". Conditions sharing a series (e.g. left/right
    foraminal narrowing, both Sagittal T1) reuse the same underlying crop,
    same dedup-by-(series, level) approach as crop_extraction.py, so
    behavior matches training exactly.

    Each returned entry also carries "had_duplicate": bool -- True if more
    than one series_id existed for that series_description and contributed
    a candidate detection for that level, i.e. a real duplicate was
    resolved by keeping the higher-confidence one. assemble_study_output.py
    uses this to set status="duplicate_resolved" instead of "ok"."""
    cache = {}
    for series_desc in set(SERIES_BY_CONDITION.values()):
        series_ids = series_df[
            (series_df.study_id == study_id) &
            (series_df.series_description == series_desc)
        ].series_id.tolist()

        merged_best, merged_ordered = {}, {}
        # Tracks how many distinct series_ids actually contributed a
        # candidate detection for each level -- NOT just len(series_ids),
        # since a series_id with no detected boxes at all for a level
        # shouldn't count as a "duplicate" for that level.
        contributor_count = {}

        for sid in series_ids:
            sdir = f"{data_root}/train_images/{study_id}/{sid}"
            ordered = get_ordered_slices(sdir)
            if not ordered:
                continue
            best = detect_best_per_level(model, ordered)
            for lvl, val in best.items():
                contributor_count[lvl] = contributor_count.get(lvl, 0) + 1
                if lvl not in merged_best or val[0] > merged_best[lvl][0]:
                    merged_best[lvl] = val
                    merged_ordered[lvl] = ordered

        cache[series_desc] = (merged_best, merged_ordered, contributor_count)

    crop_by_series_level = {}
    for series_desc, (merged_best, merged_ordered, contributor_count) in cache.items():
        n_slices = PER_SERIES_CONFIG[series_desc]["n_slices"]
        for level_idx, level in enumerate(LEVELS):
            det = merged_best.get(level_idx)
            if det is None:
                continue
            conf, dcm_path, xyxy, shape = det
            ordered_slices = merged_ordered[level_idx]
            stacked = build_stacked_crop(ordered_slices, dcm_path, xyxy, shape, series_desc)
            if stacked is None:
                continue
            had_duplicate = contributor_count.get(level_idx, 1) > 1
            crop_by_series_level[(series_desc, level)] = {
                "crop": stacked,
                "series_description": series_desc,
                "n_slices_used": n_slices,
                "yolo_detection_confidence": conf,
                "had_duplicate": had_duplicate,
            }

    result = {}
    for condition, series_desc in SERIES_BY_CONDITION.items():
        for level in LEVELS:
            entry = crop_by_series_level.get((series_desc, level))
            if entry is not None:
                result[(condition, level)] = entry
    return result
