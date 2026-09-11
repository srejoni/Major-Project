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

FIX (this pass): the series_id merge/dedup loop -- multiple series_ids
sharing one series_description, merged down to one highest-confidence
detection per level -- was previously duplicated here as an inline copy
of crop_extraction.py::process_study()'s equivalent loop. Both were
equivalent at the time but had no mechanism preventing drift on the next
edit to either file. Now imports and calls
crop_extraction.merge_series_detections() -- the single shared
implementation -- instead of reimplementing it. Only the detection/
cropping primitives (get_ordered_slices, detect_best_per_level,
build_stacked_crop) and the merge loop itself are shared; everything
below that (padding to (condition, level), building the returned crop
dict) is inference-specific and stays here.

NAMING: the duplicate-resolution flag is "was_duplicate_resolved" here,
matching crop_extraction.py's manifest column of the same name (also
added this pass -- see that file's docstring). One name, one concept, in
both the training manifest and the inference crop dict.
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

from classification.crop_extraction import (
    LEVELS, SERIES_BY_CONDITION, PER_SERIES_CONFIG,
    build_stacked_crop, merge_series_detections,
)


def extract_study_crops(model, study_id, series_df, data_root):
    """Returns dict keyed by (condition, level) -> crop info.
    NO entry exists for a (condition, level) that had no usable detection --
    that absence is exactly what assemble_study_output.py turns into
    status="missing_detection". Conditions sharing a series (e.g. left/right
    foraminal narrowing, both Sagittal T1) reuse the same underlying crop,
    same dedup-by-(series, level) approach as crop_extraction.py -- now via
    the same shared merge_series_detections() call, so behavior matches
    training exactly by construction, not by coincidence.

    Each returned entry also carries "was_duplicate_resolved": bool -- True
    if more than one series_id existed for that series_description and
    contributed a candidate detection for that level, i.e. a real duplicate
    was resolved by keeping the higher-confidence one.
    assemble_study_output.py uses this to set status="duplicate_resolved"
    instead of "ok"."""
    crop_by_series_level = {}
    for series_desc in set(SERIES_BY_CONDITION.values()):
        merged_best, merged_ordered, contributor_count = merge_series_detections(
            model, study_id, series_df, data_root, series_desc
        )
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
            was_duplicate_resolved = contributor_count.get(level_idx, 1) > 1
            crop_by_series_level[(series_desc, level)] = {
                "crop": stacked,
                "series_description": series_desc,
                "n_slices_used": n_slices,
                "yolo_detection_confidence": conf,
                "was_duplicate_resolved": was_duplicate_resolved,
            }

    result = {}
    for condition, series_desc in SERIES_BY_CONDITION.items():
        for level in LEVELS:
            entry = crop_by_series_level.get((series_desc, level))
            if entry is not None:
                result[(condition, level)] = entry
    return result
