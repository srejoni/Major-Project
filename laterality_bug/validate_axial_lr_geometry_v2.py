"""
validate_axial_lr_geometry.py

Pre-fix validation: does a WITHIN-IMAGE x-position proxy reliably separate
left_subarticular_stenosis from right_subarticular_stenosis annotations?

Must be run and pass BEFORE any change to detect_best_per_level_by_side()
/ merge_series_detections_by_side() for Axial T2 -- mirrors the discipline
already used for Sagittal T1 (125/125 sign-consistency check, done before
that fix's code was written -- see crop_extraction.py module docstring).

CONFIRMED SCHEMA (notebook48c0744e7b, cells [25]-[27], Sept 14 2026):
  columns: study_id, series_id, instance_number, condition, level, x, y
  x, y: float64, real pixel coords in the ORIGINAL (un-resized) DICOM image
        -- not normalized (mean x=238.2, max=686.2; mean y=233.1, max=801.9).
  condition: Title Case + spaces -- canonicalized via
        classification.condition_names (NOT a local dict; see that module's
        docstring for why this isn't duplicated per-script).
  level: 'L1/L2'...'L5/S1', slash-delimited -- matches LEVELS exactly,
        unlike best.pt's hyphen-delimited model.names. No normalization
        needed here.
  DICOM path (study_id/series_id/instance_number.dcm) confirmed to exist
        on disk for at least one sampled row.

WHY THIS PROXY, NOT THE SAGITTAL ONE:
Sagittal T1's fix splits by POSITION ACROSS SLICES (imaging-plane normal),
because sagittal slices are physically offset left-to-right and left/right
findings live on different slices ~99.6% of the time. Axial slices stack
along superior-inferior instead, and left/right subarticular recesses are
usually visible WITHIN one image side by side -- so this script tests a
per-image, orientation-corrected pixel-x proxy instead. Because it's a
per-image property (not a cross-slice comparison), it should apply
identically whether the paired left/right annotations share an
instance_number or not -- that's specifically checked below, since
same-instance is only ~56.8% of cases and it's an open question whether
the proxy transfers to the rest.

Usage:
    python validate_axial_lr_geometry.py
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

import numpy as np
import pandas as pd
import pydicom

from classification.condition_names import (
    canonicalize_condition_column,
    validate_label_coordinates_conditions,
    assert_nonempty_merge,
)

DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"

SUBARTICULAR = ("left_subarticular_stenosis", "right_subarticular_stenosis")

# DICOM patient coordinate system (LPS convention): +x points toward the
# patient's LEFT. This is the standard this proxy relies on.
PATIENT_LEFT_AXIS = np.array([1.0, 0.0, 0.0])


def leftness(study_id, series_id, instance_number, x):
    """Sign-corrected pixel-x: positive values move toward the patient's
    left, regardless of this image's own row-direction in DICOM space.
    Uses only this row's OWN instance -- no cross-slice logic, unlike the
    Sagittal T1 proxy."""
    dcm_path = f"{DATA_ROOT}/train_images/{study_id}/{series_id}/{instance_number}.dcm"
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir = iop[:3]  # direction of increasing pixel-x (column index)
    sign = np.dot(row_dir, PATIENT_LEFT_AXIS)
    if abs(sign) < 1e-6:
        return None  # row_dir ~perpendicular to L-R axis -- unexpected for
                      # axial; don't fabricate a sign if it happens
    return float(np.sign(sign) * x)


def main():
    coords = pd.read_csv(f"{DATA_ROOT}/train_label_coordinates.csv")
    validate_label_coordinates_conditions(coords)
    coords = canonicalize_condition_column(coords)

    coords = coords[coords["condition"].isin(SUBARTICULAR)].copy()

    left = coords[coords["condition"] == "left_subarticular_stenosis"]
    right = coords[coords["condition"] == "right_subarticular_stenosis"]

    merged = pd.merge(left, right, on=["study_id", "level"], suffixes=("_left", "_right"))
    assert_nonempty_merge(
        merged, "left_subarticular_stenosis", "right_subarticular_stenosis",
        on=["study_id", "level"],
    )
    print(f"{len(merged)} study/level pairs with both a left and right "
          f"subarticular annotation\n")

    same_instance = merged["instance_number_left"] == merged["instance_number_right"]
    print(f"same instance_number:      {same_instance.sum():5d} ({100*same_instance.mean():.1f}%)")
    print(f"different instance_number: {(~same_instance).sum():5d} ({100*(~same_instance).mean():.1f}%)")

    n_consistent_same, n_checked_same = 0, 0
    n_consistent_diff, n_checked_diff = 0, 0
    n_skipped = 0
    mismatches = []

    for r in merged.itertuples(index=False):
        try:
            l_val = leftness(r.study_id, r.series_id_left, r.instance_number_left, r.x_left)
            r_val = leftness(r.study_id, r.series_id_right, r.instance_number_right, r.x_right)
        except Exception:
            n_skipped += 1
            continue
        if l_val is None or r_val is None:
            n_skipped += 1
            continue

        is_same = r.instance_number_left == r.instance_number_right
        consistent = l_val > r_val
        if is_same:
            n_checked_same += 1
            n_consistent_same += int(consistent)
        else:
            n_checked_diff += 1
            n_consistent_diff += int(consistent)
        if not consistent:
            mismatches.append((r.study_id, r.level, l_val, r_val, is_same))

    n_checked = n_checked_same + n_checked_diff
    n_consistent = n_consistent_same + n_consistent_diff
    print(f"\nSkipped (unreadable DICOM / degenerate orientation): {n_skipped}")
    print(f"\nOverall consistency (left-leftness > right-leftness): "
          f"{n_consistent}/{n_checked} ({100*n_consistent/max(n_checked,1):.2f}%)")
    print(f"  same-instance subset:      {n_consistent_same}/{n_checked_same} "
          f"({100*n_consistent_same/max(n_checked_same,1):.2f}%)")
    print(f"  different-instance subset: {n_consistent_diff}/{n_checked_diff} "
          f"({100*n_consistent_diff/max(n_checked_diff,1):.2f}%)")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches (first 10):")
        for study_id, level, lv, rv, same_inst in mismatches[:10]:
            print(f"  study {study_id} {level}: left={lv:.1f} right={rv:.1f} same_instance={same_inst}")


if __name__ == "__main__":
    main()
