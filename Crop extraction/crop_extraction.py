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

Fixes applied after code review (previous pass):
- validate_severity_values() -- fail fast, once, at start, on any
  unexpected severity string in train.csv, instead of a raw KeyError
  hundreds of studies into a long run.
- validate_yolo_class_mapping() -- confirms best.pt's own class
  index->label map actually matches LEVELS' order before any crop is
  produced. Single highest-risk unverified item flagged in review:
  detect_best_per_level() uses int(box.cls.item()) directly as an index
  into LEVELS with nothing checking that assumption. A silent mismatch
  here would mislabel every crop by level, permanently, with no error
  anywhere -- this check turns that into a loud failure at startup.

Fixes applied (previous pass -- shared merge logic + duplicate tracking):
- merge_series_detections() -- the per-series_id merge/dedup loop
  (multiple series_ids sharing one series_description -> single
  highest-confidence detection per level) was previously written twice:
  once inline in this file's process_study(), once inline in
  crop_extraction_inference.py's extract_study_crops(). Both were
  equivalent today but had no mechanism stopping them from silently
  diverging on the next edit to either one -- the exact class of bug this
  project already hit twice (BOX_SIZE_FRAC, percentile-vs-zscore
  normalization). Factored into this single function; both files now call
  it, one implementation, one place to change it.
- was_duplicate_resolved now tracked and written to the manifest.
  Previously this file did not record whether more than one series_id
  contributed a candidate detection for a given (series, level) -- that
  information was silently discarded once the higher-confidence detection
  was picked. crop_extraction_inference.py's docstring assumed this
  manifest column already existed under this name; it did not, until now.
  This is a real correction, not a rename -- the training manifest gains
  a column it never had.

Fixes applied (previous pass -- YOLO class-order delimiter mismatch):
- _canonicalize_level_string() -- best.pt's class names use a hyphen
  ('L1-L2'), LEVELS uses a slash ('L1/L2'). Confirmed via a real
  validate_yolo_class_mapping() run: model.names was positionally
  identical to LEVELS, only the delimiter differed (YOLO/data.yaml class
  names typically can't contain '/'). Normalizing both sides before
  comparing lets the check verify what actually matters -- index order --
  without a cosmetic formatting difference tripping a false failure.
- load_completed_study_ids() now hard-fails if an existing manifest's
  header doesn't match MANIFEST_FIELDS exactly, instead of silently
  trusting stale state (e.g. a manifest from before was_duplicate_resolved
  existed).

Fixes applied THIS pass -- laterality / shared-crop fix for foraminal
narrowing (Sagittal T1):

Root cause (confirmed by direct code reading): the "keyed by (series_desc,
level)" design above means left_neural_foraminal_narrowing and
right_neural_foraminal_narrowing -- both mapped to "Sagittal T1" in
SERIES_BY_CONDITION -- were GUARANTEED to receive the identical
crop_image_path in every manifest row, since process_study() looked up
one shared cache entry for both. Confirmed structurally, then empirically.

Diagnostic chain:
1. Checked against train_label_coordinates.csv (a file this pipeline never
   previously read): for foraminal narrowing, left/right annotations
   share the same series_id 99.6% of the time but the same
   instance_number only 0.4% of the time -- left and right genuinely live
   on different slices in nearly every study. For subarticular stenosis,
   the split is 91.3% same series_id / 56.8% same instance_number -- a
   real ceiling exists for roughly half of cases (axial slices show both
   recesses in one frame more often), so that pair is deliberately NOT
   touched by this fix -- see SIDE_SPLIT_SERIES below.
2. Confirmed, against 125 known left/right instance_number pairs across 25
   studies, that projecting each slice's ImagePositionPatient onto the
   imaging plane's normal vector (row_dir x col_dir from
   ImageOrientationPatient -- derived geometrically, not assumed from a
   raw DICOM axis label) gives a scalar "position along the stack" that
   separates left from right with 100% sign consistency: left's projected
   position was lower than right's in all 125/125 cases.

This pass adds detect_best_per_level_by_side() and
merge_series_detections_by_side(), which split per-level detections by
that confirmed proxy instead of collapsing to one shared best-confidence
detection. Applied ONLY to series in SIDE_SPLIT_SERIES -- currently just
"Sagittal T1". Axial T2 is deliberately excluded: its slices stack along
the superior-inferior (vertebral level) axis, not left-right, so this
same normal-vector proxy has not been shown to transfer and needs its own
geometric check before being added here.

Crop filenames for split series now carry a "_left"/"_right" suffix to
avoid collision; filenames for non-split series (Sagittal T2/STIR, Axial
T2) are unchanged. The manifest schema itself is unchanged -- only the
crop_image_path a given row resolves to.

MIGRATION NOTE: this fix does not retroactively repair an existing
manifest.csv built under the old shared-crop logic -- load_completed_
study_ids() will skip any study_id already present, regardless of which
conditions' rows exist for it. See migrate_foraminal_crops.py for the
one-time migration path used instead of a full rerun.
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

# Series descriptions where the shared-crop bug is confirmed FIXABLE by a
# geometric left/right split, per the diagnostic chain in the module
# docstring above. Axial T2 is intentionally absent -- confirmed NOT yet
# validated for this proxy. Do not add it without an axial-specific
# geometric check first (axial slices stack along vertebral level, not
# left-right, so this same proxy may not transfer).
SIDE_SPLIT_SERIES = {"Sagittal T1"}

PER_SERIES_CONFIG = {
    "Sagittal T1":      {"margin": 0.5, "n_slices": 1},
    "Sagittal T2/STIR": {"margin": 0.5, "n_slices": 1},
    "Axial T2":         {"margin": 0.7, "n_slices": 3},
}

SEVERITY_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
YOLO_CONF_THRESHOLD = 0.25
CROP_SIZE = 224
MANIFEST_FIELDS = [
    "study_id", "crop_image_path", "condition", "level", "severity",
    "detection_conf", "was_duplicate_resolved",
]


def level_slug(level):
    return level.lower().replace("/", "_")


def series_slug(series_desc):
    return series_desc.lower().replace(" ", "_").replace("/", "_")


def condition_side(condition):
    """Returns 'left' / 'right' for a lateral condition string, None for a
    non-lateral one (e.g. spinal_canal_stenosis). Used to pick which half
    of a side-split crop pair a given condition's manifest row should
    point at."""
    if condition.startswith("left_"):
        return "left"
    if condition.startswith("right_"):
        return "right"
    return None


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


def _slice_side_position(dcm_path, normal_cache):
    """Returns ImagePositionPatient projected onto the imaging plane's
    normal vector (row_dir x col_dir from ImageOrientationPatient).
    normal_cache is a single-element list used as a mutable cell so the
    normal is computed once per series (it should be constant across
    slices of one series) and reused."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    if normal_cache[0] is None:
        row_dir, col_dir = iop[:3], iop[3:]
        normal_cache[0] = np.cross(row_dir, col_dir)
    return float(np.dot(ipp, normal_cache[0]))


def detect_best_per_level_by_side(model, ordered_slices):
    """Per-side counterpart to detect_best_per_level(), for series in
    SIDE_SPLIT_SERIES. Runs detection on every slice exactly as
    detect_best_per_level() does, but instead of collapsing immediately to
    one best-confidence detection per level, also records each slice's
    geometric side-position and splits candidates by the median position
    among them before picking a best-confidence detection within each half.

    Confirmed direction (125/125 = 100% consistent over a real sample):
    the LOWER-position half corresponds to left_X's annotated instance,
    the HIGHER-position half to right_X's.

    If a level only has a detection on a single slice, both "left" and
    "right" fall back to that same detection -- matches the confirmed
    ~0.4% of foraminal-narrowing pairs that are genuinely single-image by
    GT, so no separation is possible for those regardless of method.

    Returns {level_idx: {"left": (conf, dcm_path, xyxy, shape),
                          "right": (conf, dcm_path, xyxy, shape)}}
    """
    per_level_candidates = {}
    normal_cache = [None]
    for dcm_path in ordered_slices:
        try:
            position = _slice_side_position(dcm_path, normal_cache)
        except Exception:
            continue

        pixels = load_dicom_pixels(dcm_path)
        img8 = percentile_clip_to_uint8(pixels)
        rgb = np.stack([img8] * 3, axis=-1)
        results = model.predict(rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        for box in results.boxes:
            lvl, conf = int(box.cls.item()), float(box.conf.item())
            per_level_candidates.setdefault(lvl, []).append(
                (conf, dcm_path, box.xyxy[0].cpu().numpy(), img8.shape, position)
            )

    result = {}
    for lvl, candidates in per_level_candidates.items():
        if len(candidates) == 1:
            conf, dcm_path, xyxy, shape, _pos = candidates[0]
            det = (conf, dcm_path, xyxy, shape)
            result[lvl] = {"left": det, "right": det}
            continue

        positions = [c[4] for c in candidates]
        median_pos = float(np.median(positions))
        left_pool = [c for c in candidates if c[4] <= median_pos]
        right_pool = [c for c in candidates if c[4] > median_pos]
        if not left_pool:
            left_pool = candidates
        if not right_pool:
            right_pool = candidates

        best_left = max(left_pool, key=lambda c: c[0])
        best_right = max(right_pool, key=lambda c: c[0])
        result[lvl] = {
            "left": (best_left[0], best_left[1], best_left[2], best_left[3]),
            "right": (best_right[0], best_right[1], best_right[2], best_right[3]),
        }
    return result


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


def merge_series_detections(model, study_id, series_df, data_root, series_desc):
    """Runs YOLO detection across every series_id sharing this
    series_description for one study, merging down to the single
    highest-confidence detection per level.

    SHARED between crop_extraction.py (training, this file) and
    crop_extraction_inference.py (inference). Used for series NOT in
    SIDE_SPLIT_SERIES -- i.e. where either there's no laterality pair
    (spinal_canal_stenosis) or the side-split proxy hasn't been validated
    yet (Axial T2 / subarticular stenosis).

    Returns (merged_best, merged_ordered, contributor_count):
    - merged_best: {level_idx: (conf, dcm_path, xyxy, shape)} -- the
      winning detection per level after merging across all series_ids.
    - merged_ordered: {level_idx: ordered_slice_list} -- the ordered slice
      list belonging to the series_id that won for that level (needed for
      axial neighbor-slice stacking in build_stacked_crop()).
    - contributor_count: {level_idx: int} -- how many distinct series_ids
      actually produced a candidate detection for that level. A value > 1
      means more than one acquisition existed and the higher-confidence
      one was kept -- i.e. a duplicate was resolved, not just "no
      duplicate happened to exist." A series_id with zero detected boxes
      for a level does not count as a contributor for that level.
    """
    series_ids = series_df[
        (series_df.study_id == study_id) &
        (series_df.series_description == series_desc)
    ].series_id.tolist()

    merged_best, merged_ordered, contributor_count = {}, {}, {}
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

    return merged_best, merged_ordered, contributor_count


def merge_series_detections_by_side(model, study_id, series_df, data_root, series_desc):
    """Per-side counterpart to merge_series_detections(), for series in
    SIDE_SPLIT_SERIES. Merges across series_ids independently per side, so
    a genuine cross-series duplicate on the left doesn't get merged
    against the right's candidates, or vice versa.

    Returns {"left": (merged_best, merged_ordered, contributor_count),
             "right": (merged_best, merged_ordered, contributor_count)} --
    same triple-tuple shape as merge_series_detections(), once per side, so
    callers can treat each side identically to the original single-crop
    merge result.
    """
    series_ids = series_df[
        (series_df.study_id == study_id) &
        (series_df.series_description == series_desc)
    ].series_id.tolist()

    sides = {"left": ({}, {}, {}), "right": ({}, {}, {})}

    for sid in series_ids:
        sdir = f"{data_root}/train_images/{study_id}/{sid}"
        ordered = get_ordered_slices(sdir)
        if not ordered:
            continue
        by_side = detect_best_per_level_by_side(model, ordered)
        for lvl, side_dets in by_side.items():
            for side in ("left", "right"):
                merged_best, merged_ordered, contributor_count = sides[side]
                val = side_dets[side]
                contributor_count[lvl] = contributor_count.get(lvl, 0) + 1
                if lvl not in merged_best or val[0] > merged_best[lvl][0]:
                    merged_best[lvl] = val
                    merged_ordered[lvl] = ordered

    return sides


def load_completed_study_ids():
    """Resume support -- study_ids already fully written to an existing
    manifest.csv.

    SCHEMA GUARD: before trusting an existing manifest as resumable state,
    confirm its header actually matches MANIFEST_FIELDS. A manifest left
    over from before was_duplicate_resolved was added would silently get
    new 7-column rows appended under its old 6-column header -- pandas may
    then either throw a confusing tokenizing error, or in some
    configurations misalign columns rather than fail loudly. Failing here,
    once, with a clear message, is cheaper than debugging a quietly
    corrupted manifest later or relying on remembering to delete it by
    hand before every rerun.
    """
    if not os.path.exists(MANIFEST_CSV):
        return set()

    with open(MANIFEST_CSV, "r", newline="") as f:
        header = next(csv.reader(f))
    if header != MANIFEST_FIELDS:
        raise RuntimeError(
            f"Existing manifest at {MANIFEST_CSV} has columns {header}, "
            f"but this version of crop_extraction.py expects "
            f"{MANIFEST_FIELDS}. This almost always means the manifest was "
            f"written by an older version of this script (e.g. before "
            f"was_duplicate_resolved was added). Appending new rows under "
            f"this old header would corrupt the file. Delete "
            f"{MANIFEST_CSV} (and re-extract from scratch) before "
            f"re-running -- do not proceed with a mismatched manifest."
        )

    existing = pd.read_csv(MANIFEST_CSV)
    return set(existing["study_id"].unique())


def append_rows_to_manifest(rows):
    file_exists = os.path.exists(MANIFEST_CSV)
    with open(MANIFEST_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def validate_severity_values(train_df):
    """Fail fast, once, at start -- not mid-run via a raw KeyError on
    whichever study happens to hit an unexpected label string first.

    Iterates every column of train_df (all of which are per-condition/level
    severity columns in this dataset) and confirms every non-null value is
    one of SEVERITY_MAP's known strings before a single crop is produced.
    """
    valid = set(SEVERITY_MAP.keys())
    for col in train_df.columns:
        vals = set(train_df[col].dropna().unique())
        bad = vals - valid
        if bad:
            raise ValueError(
                f"Column '{col}' in train.csv contains unexpected severity "
                f"value(s) {bad} -- expected only {valid}. Fix before running "
                f"the full extraction; this would otherwise crash mid-run at "
                f"an arbitrary study."
            )
    print("[crop_extraction] train.csv severity values verified against SEVERITY_MAP.")


def _canonicalize_level_string(s):
    """best.pt's own class names use a hyphen ('L1-L2'), while this
    project's LEVELS constant uses a slash ('L1/L2') for the same level.
    Confirmed via a real validate_yolo_class_mapping() failure on the
    actual model: model.names was positionally identical to LEVELS, only
    the delimiter differed (YOLO/data.yaml class names typically can't
    contain '/'). Normalizing both to the same delimiter before comparing
    lets the check verify what actually matters -- index order -- without
    being defeated by a cosmetic formatting difference."""
    return s.replace("-", "/") if s else s


def validate_yolo_class_mapping(model):
    """Confirm best.pt's own class index->label map agrees with LEVELS'
    order (after normalizing the '-' vs '/' delimiter difference -- see
    _canonicalize_level_string). A real mismatch here would mislabel every
    crop by level, permanently, with no error anywhere. A delimiter
    difference alone is not that bug -- only a genuine positional/order
    mismatch should raise.
    """
    names = model.names  # dict: {0: 'L1-L2', 1: 'L2-L3', ...} (or '/'-delimited)
    for idx, level in enumerate(LEVELS):
        raw_name = names.get(idx)
        if _canonicalize_level_string(raw_name) != level:
            raise ValueError(
                f"YOLO class index {idx} maps to '{raw_name}' in "
                f"best.pt, but LEVELS[{idx}] = '{level}'. These must match "
                f"(after delimiter normalization) or every downstream crop "
                f"is mislabeled by level. model.names = {names}"
            )
    print("[crop_extraction] YOLO class order verified against LEVELS "
          "(delimiter-normalized match).")


def process_study(model, study_id, train_df, series_df, study_dir):
    """Returns this study's manifest rows.

    For series in SIDE_SPLIT_SERIES (currently Sagittal T1), left/right
    conditions now get genuinely different crops, keyed by
    (series_desc, level, side). For all other series (Sagittal T2/STIR,
    Axial T2), behavior is UNCHANGED -- still deduped by (series_desc,
    level) only, still one shared crop per level, since the side-split
    proxy is not yet validated for those.
    """
    cache = {}
    for series_desc in set(SERIES_BY_CONDITION.values()):
        if series_desc in SIDE_SPLIT_SERIES:
            cache[series_desc] = merge_series_detections_by_side(
                model, study_id, series_df, DATA_ROOT, series_desc
            )
        else:
            cache[series_desc] = merge_series_detections(
                model, study_id, series_df, DATA_ROOT, series_desc
            )

    # (series_desc, level_idx, side_or_None) -> (crop_path, conf, was_duplicate_resolved)
    crop_info = {}
    for series_desc, merged in cache.items():
        if series_desc in SIDE_SPLIT_SERIES:
            for side, (merged_best, merged_ordered, contributor_count) in merged.items():
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
                        study_dir,
                        f"{series_slug(series_desc)}_{level_slug(level)}_{side}.npy",
                    )
                    np.save(crop_path, stacked)
                    was_dup = contributor_count.get(level_idx, 1) > 1
                    crop_info[(series_desc, level_idx, side)] = (crop_path, conf, was_dup)
        else:
            merged_best, merged_ordered, contributor_count = merged
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
                was_dup = contributor_count.get(level_idx, 1) > 1
                crop_info[(series_desc, level_idx, None)] = (crop_path, conf, was_dup)

    rows = []
    for condition, series_desc in SERIES_BY_CONDITION.items():
        side = condition_side(condition) if series_desc in SIDE_SPLIT_SERIES else None
        for level_idx, level in enumerate(LEVELS):
            entry = crop_info.get((series_desc, level_idx, side))
            if entry is None:
                continue
            crop_path, conf, was_duplicate_resolved = entry

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
                "was_duplicate_resolved": was_duplicate_resolved,
            })
    return rows


from classification.splits import load_classification_split


def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    train_df = pd.read_csv(f"{DATA_ROOT}/train.csv").set_index("study_id")
    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    model = YOLO(YOLO_WEIGHTS)

    validate_severity_values(train_df)
    validate_yolo_class_mapping(model)

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
