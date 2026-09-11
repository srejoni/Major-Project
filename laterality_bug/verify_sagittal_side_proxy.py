"""
verify_sagittal_side_proxy.py

Step 0 for the INFERENCE-time laterality fix -- foraminal narrowing /
Sagittal T1 ONLY. (Subarticular / Axial T2 needs a separate check: axial
slices stack along the superior-inferior axis, i.e. vertebral level, not
left/right, so this same geometric proxy does not obviously transfer.)

At training time, crop_extraction.py can look up the correct
instance_number for left_X vs right_X directly from
train_label_coordinates.csv. At inference time there is no ground truth --
extract_study_crops() needs some other signal to tell, among the candidate
slices YOLO detects a box on for a given level, which one is "the left
slice" and which is "the right one."

Proposed proxy: DICOM geometry. A sagittal series' slices are stacked along
the imaging plane's normal vector (row_dir x col_dir, from
ImageOrientationPatient). Projecting each slice's ImagePositionPatient onto
that normal gives a physical scalar position along the stack -- the
quantity that should separate a left-leaning parasagittal slice from a
right-leaning one, without assuming which raw DICOM axis or sign
convention this dataset happens to use.

This script does NOT assume the proxy works -- it checks it against known-
correct GT (train_label_coordinates.csv's own left/right instance_number
pairs, restricted to cases where they actually differ -- Step 0 already
found this is ~99.2% of foraminal narrowing pairs) before any inference
code is written around it.

Usage:
    python verify_sagittal_side_proxy.py --n-studies 25
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pydicom

DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
PAIR = ("left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing")


def slice_stack_positions(series_dir):
    """Returns {instance_number: scalar_position}, where scalar_position is
    ImagePositionPatient projected onto the imaging plane's normal vector --
    the physical coordinate along the slice-stacking direction, derived
    geometrically rather than assumed from a raw DICOM axis label."""
    dcm_paths = glob.glob(os.path.join(series_dir, "*.dcm"))
    positions = {}
    normal = None
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            row_dir, col_dir = iop[:3], iop[3:]
            if normal is None:
                normal = np.cross(row_dir, col_dir)
            positions[int(ds.InstanceNumber)] = float(np.dot(ipp, normal))
        except Exception:
            continue
    return positions


def load_left_right_instances(coords_path):
    df = pd.read_csv(coords_path)
    df["condition_norm"] = df["condition"].str.strip().str.lower().str.replace(" ", "_")
    left = df[df["condition_norm"] == PAIR[0]][
        ["study_id", "series_id", "level", "instance_number"]
    ]
    right = df[df["condition_norm"] == PAIR[1]][
        ["study_id", "series_id", "level", "instance_number"]
    ]
    m = left.merge(right, on=["study_id", "level"], suffixes=("_l", "_r"))
    # Only cases where left/right genuinely differ -- same-instance rows
    # (rare for this pair per Step 0) can't test a position-based proxy.
    m = m[m["instance_number_l"] != m["instance_number_r"]]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default=f"{DATA_ROOT}/train_label_coordinates.csv")
    ap.add_argument("--n-studies", type=int, default=25)
    args, _unknown = ap.parse_known_args()  # notebook-safe, see check_gt_coordinate_separation.py

    pairs = load_left_right_instances(args.coords)
    sample_studies = pairs["study_id"].drop_duplicates().sample(
        min(args.n_studies, pairs["study_id"].nunique()), random_state=0
    )
    pairs = pairs[pairs["study_id"].isin(sample_studies)]

    signs = []
    checked, skipped = 0, 0
    for _, row in pairs.iterrows():
        dir_l = f"{DATA_ROOT}/train_images/{row.study_id}/{row.series_id_l}"
        positions_l = slice_stack_positions(dir_l)
        if row.series_id_l == row.series_id_r:
            positions_r = positions_l
        else:
            dir_r = f"{DATA_ROOT}/train_images/{row.study_id}/{row.series_id_r}"
            positions_r = slice_stack_positions(dir_r)

        pos_l = positions_l.get(row.instance_number_l)
        pos_r = positions_r.get(row.instance_number_r)
        if pos_l is None or pos_r is None:
            skipped += 1
            continue
        signs.append(np.sign(pos_l - pos_r))
        checked += 1

    signs = np.array(signs)
    print(f"Checked {checked} left/right pairs across {sample_studies.nunique()} studies "
          f"({skipped} skipped -- instance not found in directory listing)")
    print(f"  sign(left_position - right_position) == +1: {int((signs == 1).sum())}")
    print(f"  sign(left_position - right_position) == -1: {int((signs == -1).sum())}")
    print(f"  sign == 0 (identical position, shouldn't happen): {int((signs == 0).sum())}")

    if checked == 0:
        print("\nNo pairs could be checked -- verify DATA_ROOT / train_images paths.")
        return

    consistent_pct = max((signs == 1).mean(), (signs == -1).mean()) * 100
    winning_sign = "+1" if (signs == 1).sum() >= (signs == -1).sum() else "-1"
    print(f"\n  Sign consistency: {consistent_pct:.1f}% (majority sign: {winning_sign})")
    print("\nInterpretation:")
    print("  - >~95% consistency => the proxy is usable. The winning sign becomes")
    print("    the rule in the real fix: e.g. if +1 wins, 'more positive projected")
    print("    position = left' (confirm the direction manually against 2-3 known")
    print("    studies before hardcoding it, since 'left' vs 'right' here is a label")
    print("    convention, not something this script can name on its own).")
    print("  - Meaningfully below that => geometry alone isn't reliable enough;")
    print("    don't build the inference fix on this signal without a secondary")
    print("    check (e.g. combining with detection confidence or box position).")


if __name__ == "__main__":
    main()
