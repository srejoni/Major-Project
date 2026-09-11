"""
check_gt_coordinate_separation.py

Step 0 of the laterality diagnostic.

Context confirmed by reading classification/crop_extraction.py directly:
process_study() caches crops in crop_path_by_series_level, keyed ONLY by
(series_desc, level_idx). Because SERIES_BY_CONDITION maps both
left_neural_foraminal_narrowing and right_neural_foraminal_narrowing to
"Sagittal T1" (and both subarticular conditions to "Axial T2"), the
row-building loop at the end of process_study() looks up that same cache
entry for both conditions -- so every left/right pair is GUARANTEED to
get the identical crop_image_path in manifest.csv. That part is no longer
a hypothesis; it's a structural fact of the code as written.

What's still open: detect_best_per_level() and the series_id-merge loop
above it in process_study() pick a single highest-confidence detection
per level, with no side-awareness anywhere, and crop_extraction.py never
reads train_label_coordinates.csv at all -- only train.csv (severity) and
train_series_descriptions.csv (series list). So the real question isn't
"did the code discard side info" -- it never had access to any in the
first place. This script checks whether train_label_coordinates.csv (the
file the pipeline currently ignores) actually contains real per-side
distinguishing signal that a fix could plumb in -- at BOTH the series_id
level (does process_study()'s series_id-merge loop risk collapsing two
genuinely different series into one, the same way it legitimately
collapses true re-scan duplicates elsewhere in this project) and the
instance_number level within one series_id (matching the pattern already
established for axial slice-selection: level/side info lives in which
InstanceNumber is used, not just which series_id).

If train_label_coordinates.csv shows real separation -> the fix is a data-
plumbing problem confined to crop_extraction.py / crop_extraction_inference.py.
If it doesn't -> part of the fix has to move into the model itself (side
conditioning), since no crop-selection change can manufacture information
that was never annotated separately to begin with.

Usage:
    python check_gt_coordinate_separation.py
    python check_gt_coordinate_separation.py --coords /path/to/train_label_coordinates.csv
"""
import argparse
import sys

import pandas as pd

# Matches DATA_ROOT in classification/crop_extraction.py.
DEFAULT_DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"

PAIRS = [
    ("left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"),
    ("left_subarticular_stenosis", "right_subarticular_stenosis"),
]


def _known_conditions():
    """Cross-checks this script's hardcoded PAIRS against the real
    SERIES_BY_CONDITION dict from crop_extraction.py when importable, so a
    future rename there can't silently make this diagnostic check the
    wrong strings and report a false negative. Falls back to the hardcoded
    PAIRS list if the classification package isn't on sys.path in this
    environment (e.g. running the diagnostic outside the Kaggle session)."""
    try:
        sys.path.append("/kaggle/working/PDSCD")
        from classification.crop_extraction import SERIES_BY_CONDITION
        return set(SERIES_BY_CONDITION.keys())
    except ImportError:
        return {c for pair in PAIRS for c in pair}


def load_coords(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"study_id", "series_id", "instance_number", "condition", "level"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"train_label_coordinates.csv missing expected columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )
    # Competition CSV uses "Left Neural Foraminal Narrowing" (title case,
    # spaces); the pipeline elsewhere uses snake_case. Normalize here.
    df["condition_norm"] = df["condition"].str.strip().str.lower().str.replace(" ", "_")

    known = _known_conditions()
    found = set(df["condition_norm"].unique())
    unmatched = known - found
    if unmatched:
        print(
            f"WARNING: {unmatched} from crop_extraction.py's SERIES_BY_CONDITION "
            f"were not found in train_label_coordinates.csv after normalization -- "
            f"check the raw 'condition' values (df['condition'].unique()) before "
            f"trusting the report below, since a spelling mismatch would silently "
            f"produce 0 matched rows for that condition rather than an error."
        )
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_cond, right_cond in PAIRS:
        left = df[df["condition_norm"] == left_cond][
            ["study_id", "level", "series_id", "instance_number"]
        ]
        right = df[df["condition_norm"] == right_cond][
            ["study_id", "level", "series_id", "instance_number"]
        ]
        m = left.merge(right, on=["study_id", "level"], suffixes=("_l", "_r"))
        if m.empty:
            rows.append({
                "pair": f"{left_cond} / {right_cond}",
                "n_matched": 0,
                "note": "no matches -- check condition string normalization above",
            })
            continue
        same_series = m["series_id_l"] == m["series_id_r"]
        same_instance = same_series & (m["instance_number_l"] == m["instance_number_r"])
        rows.append({
            "pair": f"{left_cond} / {right_cond}",
            "n_matched": len(m),
            "pct_same_series_id": round(100 * same_series.mean(), 1),
            "pct_same_series_and_instance": round(100 * same_instance.mean(), 1),
            "pct_genuinely_different_image": round(100 * (~same_instance).mean(), 1),
        })
    return pd.DataFrame(rows)


def _parse_args():
    """Running this file's contents pasted into a Kaggle/Jupyter cell means
    sys.argv is the kernel launcher's own arguments (e.g. 'colab_kernel_
    launcher.py -f /root/.../kernel-....json'), not anything typed by hand --
    the same issue already hit and fixed once in dry_run_assemble.py
    (pdscd_progress_update_v3.md §4.2). argparse chokes on '-f' and exits.
    Try argparse first, for real `python check_gt_coordinate_separation.py
    --coords ...` invocation; fall back to the default path (COORDS_PATH,
    editable below) when argparse can't parse what it's given, instead of
    letting SystemExit kill the cell."""
    default_path = f"{DEFAULT_DATA_ROOT}/train_label_coordinates.csv"

    # Edit this if you want a non-default path while running from a notebook
    # cell -- argparse's --coords flag has no effect in that case, since
    # there's no real command line to pass it on.
    COORDS_PATH = default_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default=default_path,
                     help="Path to train_label_coordinates.csv")
    try:
        args, _unknown = ap.parse_known_args()
        return args.coords
    except SystemExit:
        print(f"[check_gt_coordinate_separation] argparse couldn't read sys.argv "
              f"(expected in a notebook cell) -- using COORDS_PATH={COORDS_PATH}")
        return COORDS_PATH


if __name__ == "__main__":
    coords_path = _parse_args()
    df = load_coords(coords_path)
    out = report(df)
    print(out.to_string(index=False))
    print()
    print("Interpretation:")
    print("  - High pct_genuinely_different_image => raw data DOES support separate")
    print("    left/right crops. The fix belongs in crop_extraction.py /")
    print("    crop_extraction_inference.py's selection logic -- see")
    print("    per_side_crop_fix_REFERENCE.py. No model change needed for these cases.")
    print("  - High pct_same_series_id but low pct_same_series_and_instance =>")
    print("    the two sides live in the SAME series_id but different slices.")
    print("    A fix here must key crops by instance_number, not just series_id --")
    print("    a series_id-level fix alone would still collapse these.")
    print("  - High pct_same_series_and_instance => real ceiling: many studies")
    print("    genuinely share one image across both sides. A code-only")
    print("    crop-extraction fix cannot separate those -- see the model-side")
    print("    conditioning option in per_side_crop_fix_REFERENCE.py.")
