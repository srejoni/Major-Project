"""
validate_axial_series_calibrated_midline.py

Motivated by validate_axial_single_box_midline.py's result: 97.39%
overall, below the relative (paired) proxy's 98.74% raw / ~99.87% after
excluding the 22 known GT-swap studies -- AND with a different failure
signature. That test's misclassifications had small margins (median
3.58px, max 51.30px), clustered near zero -- unlike the relative test's
large, all-or-nothing swap margins (median 30.58px, max 167.25px).

Small margins near zero suggest raw image-pixel midline (Columns/2) is a
BIASED estimate of the true anatomical left-right center: patients aren't
always perfectly centered in the imaging FOV, and that offset plausibly
varies by series/acquisition, not by label correctness.

THIS SCRIPT tests a stronger absolute reference: a PER-SERIES calibrated
midline, estimated from whichever OTHER levels in the same series have
both a left and right annotation (mean of their pair-center leftness
values: (left+right)/2), evaluated LEAVE-ONE-LEVEL-OUT so the held-out
level's own answer never leaks into its own calibration. This simulates
the real single-box production case: calibrate from other levels'
detections in this study's axial series, then classify the one level that
only has a single detected box.

NOTE ON PRODUCTION USE: at real extraction time there is no GT to
calibrate from -- the production version would calibrate from OTHER
levels' YOLO detections where both sides were found in one image, not
from train_label_coordinates.csv. This script only tests whether the
calibration IDEA reduces error, using GT pairs as a stand-in for "other
detected pairs in this series."

Directly comparable to validate_axial_single_box_midline.py's 97.39% /
18,718 correct out of 19,220 raw-midline result -- same annotations,
same leftness() sign convention, different reference point.
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

import numpy as np
import pandas as pd
import pydicom

from classification.condition_names import canonicalize_condition_column

DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
PATIENT_LEFT_AXIS = np.array([1.0, 0.0, 0.0])


def leftness(study_id, series_id, instance_number, x):
    """Sign-corrected pixel-x only -- no midline here, calibration supplies
    the reference point separately in this script."""
    dcm_path = f"{DATA_ROOT}/train_images/{study_id}/{series_id}/{instance_number}.dcm"
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir = iop[:3]
    raw_sign = np.dot(row_dir, PATIENT_LEFT_AXIS)
    if abs(raw_sign) < 1e-6:
        return None
    return float(np.sign(raw_sign) * x)


def main():
    coords = canonicalize_condition_column(pd.read_csv(f"{DATA_ROOT}/train_label_coordinates.csv"))
    sub = coords[coords["condition"].isin(
        ["left_subarticular_stenosis", "right_subarticular_stenosis"]
    )].copy()

    left = sub[sub.condition == "left_subarticular_stenosis"]
    right = sub[sub.condition == "right_subarticular_stenosis"]
    paired = pd.merge(left, right, on=["study_id", "level"], suffixes=("_left", "_right"))
    print(f"{len(paired)} paired (study_id, level) rows with both sides labeled\n")

    rows = []
    for r in paired.itertuples(index=False):
        lv = leftness(r.study_id, r.series_id_left, r.instance_number_left, r.x_left)
        rv = leftness(r.study_id, r.series_id_right, r.instance_number_right, r.x_right)
        if lv is None or rv is None:
            continue
        # Same series expected for both sides of one axial acquisition;
        # use the left side's series_id as the calibration group key.
        rows.append({
            "study_id": r.study_id, "series_id": r.series_id_left, "level": r.level,
            "left_leftness": lv, "right_leftness": rv,
            "pair_center": (lv + rv) / 2.0,
        })
    df = pd.DataFrame(rows)
    print(f"{len(df)} pairs usable after DICOM reads\n")

    n_correct, n_checked, n_no_calibration = 0, 0, 0
    margins = []

    for series_id, group in df.groupby("series_id"):
        for idx in group.index:
            others = group.drop(idx)
            if len(others) == 0:
                n_no_calibration += 2  # both sides of this level are unscorable
                continue
            calibrated_center = others["pair_center"].mean()
            row = group.loc[idx]

            for is_left_cond, val in [(True, row["left_leftness"]), (False, row["right_leftness"])]:
                margin = val - calibrated_center
                correct = (margin > 0) == is_left_cond
                n_checked += 1
                n_correct += int(correct)
                margins.append(margin if is_left_cond else -margin)  # normalize sign: positive = correct direction

    print(f"Calibrated-midline accuracy: {n_correct}/{n_checked} "
          f"({100*n_correct/max(n_checked,1):.2f}%)")
    print(f"Levels with no other calibratable level in their series "
          f"(no comparison possible, excluded): {n_no_calibration}")
    print(f"\nCompare against validate_axial_single_box_midline.py's raw "
          f"Columns/2 result: 18,718/19,220 = 97.39%")

    m = pd.Series(margins).abs()
    print(f"\n|margin| distribution (calibrated, sign-normalized so positive = correct direction):")
    print(m.describe())


if __name__ == "__main__":
    main()
