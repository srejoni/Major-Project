"""
validate_axial_single_box_midline.py

Second, harder validation gate. validate_axial_lr_geometry_v2.py only
tested a RELATIVE comparison: given both a left and right annotation for
one (study_id, level), is left's leftness > right's? That's cleared
(98.74% overall, ~0.13% once the 22 known whole-study GT-swap cases are
excluded -- see classify_axial_mismatches.py output, Sept 16 2026).

But the common real case at extraction time is a SINGLE detected box for
a level in a given image, with no second box to compare against. That
needs an ABSOLUTE rule, which has not yet been tested. This script tests
exactly that: for every INDIVIDUAL left/right subarticular annotation (not
just paired ones -- no merge, so N is larger and doesn't depend on both
sides being labeled), is its sign-corrected pixel-x on the correct side of
its OWN image's sign-corrected midline (Columns/2)?

If this clears the same bar, both cases the real fix needs (two boxes ->
relative compare; one box -> absolute midline compare) are validated and
detect_best_per_level_by_side()'s axial version can be written next.

Run after validate_axial_lr_geometry_v2.py -- reuses condition_names, no
new dependency.
"""
import sys
sys.path.append("/kaggle/working/PDSCD")

import numpy as np
import pandas as pd
import pydicom

from classification.condition_names import canonicalize_condition_column

DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"

# DICOM patient coordinate system (LPS convention): +x points toward the
# patient's LEFT -- same convention as validate_axial_lr_geometry_v2.py.
PATIENT_LEFT_AXIS = np.array([1.0, 0.0, 0.0])


def leftness_and_midline(study_id, series_id, instance_number, x):
    """Returns (sign-corrected box x, sign-corrected image midline) using
    ONLY this image's own metadata -- no partner annotation involved."""
    dcm_path = f"{DATA_ROOT}/train_images/{study_id}/{series_id}/{instance_number}.dcm"
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir = iop[:3]
    raw_sign = np.dot(row_dir, PATIENT_LEFT_AXIS)
    if abs(raw_sign) < 1e-6:
        return None, None
    sign = np.sign(raw_sign)
    return sign * x, sign * (ds.Columns / 2.0)


def main():
    coords = canonicalize_condition_column(
        pd.read_csv(f"{DATA_ROOT}/train_label_coordinates.csv")
    )
    sub = coords[coords["condition"].isin(
        ["left_subarticular_stenosis", "right_subarticular_stenosis"]
    )].copy()

    n_left = (sub.condition == "left_subarticular_stenosis").sum()
    n_right = (sub.condition == "right_subarticular_stenosis").sum()
    print(f"{len(sub)} individual subarticular annotations "
          f"({n_left} left / {n_right} right)\n")

    n_correct = {"left_subarticular_stenosis": 0, "right_subarticular_stenosis": 0}
    n_checked = {"left_subarticular_stenosis": 0, "right_subarticular_stenosis": 0}
    n_skipped = 0
    records = []

    for r in sub.itertuples(index=False):
        lv, mid = leftness_and_midline(r.study_id, r.series_id, r.instance_number, r.x)
        if lv is None:
            n_skipped += 1
            continue
        margin = lv - mid  # positive => classified as "left" of this image's midline
        is_left_cond = r.condition == "left_subarticular_stenosis"
        correct = (margin > 0) == is_left_cond
        n_checked[r.condition] += 1
        n_correct[r.condition] += int(correct)
        records.append({"study_id": r.study_id, "level": r.level,
                         "condition": r.condition, "margin": margin, "correct": correct})

    for cond in n_checked:
        c, n = n_correct[cond], n_checked[cond]
        print(f"{cond}: {c}/{n} correctly classified against absolute "
              f"per-image midline ({100*c/max(n,1):.2f}%)")

    total_c, total_n = sum(n_correct.values()), sum(n_checked.values())
    print(f"\nOverall: {total_c}/{total_n} ({100*total_c/max(total_n,1):.2f}%)   "
          f"Skipped (unreadable/degenerate orientation): {n_skipped}")

    df = pd.DataFrame(records)
    wrong = df[~df["correct"]]
    print(f"\n{len(wrong)} misclassified.")
    if len(wrong):
        per_study = wrong.groupby("study_id").size().sort_values(ascending=False)
        print(f"Studies with >=4 misclassified levels (likely same 22 "
              f"GT-swap studies as before): {(per_study >= 4).sum()}")
        print(f"|margin| among misclassified:")
        print(wrong["margin"].abs().describe())


if __name__ == "__main__":
    main()
