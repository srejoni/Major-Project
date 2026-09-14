#!/usr/bin/env python3
"""
check_crop_sharing.py

Verifies that laterality-paired conditions do NOT silently share the same
physical crop image for the same (study_id, level) -- the laterality bug
described in PDSCD_master_context_v2.md, section 14, item 5.

For each laterality-paired condition family, computes:

    pct_identical_path = (# of (study_id, level) rows where crop_image_path
                           is identical between the left and right condition)
                          / (# of (study_id, level) pairs present on BOTH sides)

Usage:
    python check_crop_sharing.py --manifest /kaggle/working/classification_data/manifest.csv

Exit codes:
    0 = foraminal-narrowing pct_identical_path is at/below the fail threshold
    1 = foraminal-narrowing pct_identical_path is still above threshold
        (migration didn't take effect, or you're checking a stale manifest)
    2 = manifest not found, missing required columns, or no foraminal
        left/right pairs found at all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# --- Assumptions made explicit (per project convention: state, don't guess) ---
# 1. manifest.csv has at least these columns: study_id, crop_image_path,
#    condition, level (confirmed 7-column schema per progress_update_v3 Sec.6).
#    Extra columns (severity, detection_conf, was_duplicate_resolved) are
#    ignored here -- this script only checks path-sharing, not label integrity.
# 2. "condition" values match the schema enum exactly, e.g.
#    left_neural_foraminal_narrowing / right_neural_foraminal_narrowing.
# 3. A "pair" is (study_id, level) present in the manifest for BOTH sides of
#    one condition family. If one side is missing entirely (e.g. no YOLO
#    detection on that side, so its row was dropped pre-inference), that
#    (study_id, level) is excluded from the denominator rather than counted
#    as "not identical" -- otherwise a legitimate missing-detection would
#    quietly deflate pct_identical_path and mask the real bug rate. Excluded
#    counts are printed, not silently dropped.
# 4. If a condition family isn't present in the manifest at all (e.g.
#    subarticular hasn't been migrated/checked in this pass), it's reported
#    with n_pairs=0 / pct=n/a rather than crashing.

LATERALITY_PAIRS = {
    "neural_foraminal_narrowing": (
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
    ),
    "subarticular_stenosis": (
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ),
}

# The migration procedure says foraminal narrowing should land at "roughly
# 0.4%" after the fix, not an exact figure -- so this is a generous fail
# threshold to catch "the fix clearly didn't take," not a strict assertion.
FORAMINAL_FAIL_THRESHOLD_PCT = 5.0


def check_pair(df: pd.DataFrame, left_cond: str, right_cond: str) -> dict:
    left = df.loc[df["condition"] == left_cond, ["study_id", "level", "crop_image_path"]]
    right = df.loc[df["condition"] == right_cond, ["study_id", "level", "crop_image_path"]]

    left = left.rename(columns={"crop_image_path": "left_path"})
    right = right.rename(columns={"crop_image_path": "right_path"})

    merged = pd.merge(left, right, on=["study_id", "level"], how="outer", indicator=True)
    both_sides = merged[merged["_merge"] == "both"]
    one_sided = merged[merged["_merge"] != "both"]

    n_pairs = len(both_sides)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "n_identical": 0,
            "pct_identical_path": float("nan"),
            "n_one_sided_excluded": len(one_sided),
        }

    n_identical = int((both_sides["left_path"] == both_sides["right_path"]).sum())
    return {
        "n_pairs": n_pairs,
        "n_identical": n_identical,
        "pct_identical_path": 100.0 * n_identical / n_pairs,
        "n_one_sided_excluded": len(one_sided),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the post-migration manifest.csv")
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=FORAMINAL_FAIL_THRESHOLD_PCT,
        help="pct_identical_path above this value for foraminal narrowing fails the check (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.manifest)

    required_cols = {"study_id", "crop_image_path", "condition", "level"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: manifest is missing expected column(s): {sorted(missing_cols)}\n"
            f"       found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Loaded manifest: {args.manifest}  ({len(df)} rows)\n")

    results = {}
    for family, (left_cond, right_cond) in LATERALITY_PAIRS.items():
        results[family] = check_pair(df, left_cond, right_cond)

    header = f"{'family':<26}{'n_pairs':>9}{'n_identical':>13}{'pct_identical_path':>20}{'one_sided_excl':>16}"
    print(header)
    print("-" * len(header))
    for family, r in results.items():
        pct_str = "n/a" if pd.isna(r["pct_identical_path"]) else f"{r['pct_identical_path']:.2f}%"
        print(f"{family:<26}{r['n_pairs']:>9}{r['n_identical']:>13}{pct_str:>20}{r['n_one_sided_excluded']:>16}")
    print()

    foraminal = results["neural_foraminal_narrowing"]

    if pd.isna(foraminal["pct_identical_path"]):
        print(
            "ERROR: no left/right neural_foraminal_narrowing pairs found in this manifest.\n"
            "       Did the migration actually write foraminal rows back in? Is this the\n"
            "       post-migration manifest, or a stale/backup copy?",
            file=sys.stderr,
        )
        sys.exit(2)

    if foraminal["pct_identical_path"] > args.fail_threshold:
        print(
            f"FAIL: foraminal-narrowing pct_identical_path = {foraminal['pct_identical_path']:.2f}% "
            f"(> {args.fail_threshold}% threshold).\n"
            f"      Migration does not appear to have taken effect on this file -- check that\n"
            f"      --manifest points at the freshly migrated manifest.csv, not the pre-migration\n"
            f"      backup or a cached copy loaded earlier in the session.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"PASS: foraminal-narrowing pct_identical_path = {foraminal['pct_identical_path']:.2f}% "
        f"(<= {args.fail_threshold}% threshold)."
    )

    subart = results["subarticular_stenosis"]
    if not pd.isna(subart["pct_identical_path"]) and subart["pct_identical_path"] > args.fail_threshold:
        print(
            f"\nNOTE: subarticular_stenosis pct_identical_path is {subart['pct_identical_path']:.2f}% "
            f"-- this pair was NOT in scope for this migration (only foraminal narrowing was),\n"
            f"      so a high value here is expected right now, not a new failure. It's the same\n"
            f"      underlying design issue, tracked separately as master_context_v2 Sec.14 item 5."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
