"""
classification/condition_names.py

Standalone module -- deliberately NOT a helper inside crop_extraction.py
(would force an unnecessary ultralytics/YOLO import onto anything that
just wants coordinate-level diagnostics, e.g. the laterality_bug/
validation scripts), and deliberately NOT copy-pasted per-script (that
exact duplication pattern -- the same merge/dedup loop written twice --
was already found and fixed once in crop_extraction.py /
crop_extraction_inference.py; no reason to reintroduce it here).

WHY THIS EXISTS: train_label_coordinates.csv's `condition` column uses
Title Case with spaces ('Left Neural Foraminal Narrowing'), while every
other part of this project (SERIES_BY_CONDITION, the manifest's condition
column, pdscd_output_schema.json) uses snake_case
('left_neural_foraminal_narrowing'). Confirmed by direct inspection, not
assumed.

WHY IT MATTERS MORE THAN A NAMING NITPICK: pd.merge() across this mismatch
does NOT raise -- it silently returns 0 rows, and every downstream
rate/percentage then computes as 0/0 or n/a, indistinguishable from
"checked, found clean." Same failure category as this project's earlier
had_duplicate/was_duplicate_resolved key mismatch and the crop-root
shadowing bug: a wrong-but-successful lookup, not an error.

canonicalize_condition() is a deterministic transform validated against
the enum, not a hardcoded lookup table -- a new or re-cased value fails
loud (ValueError) instead of silently falling through a missing key.
"""

CONDITION_ENUM = {
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
}


def canonicalize_condition(raw):
    """Deterministic transform: strip -> lowercase -> collapse internal
    whitespace -> underscore-join. Validated against CONDITION_ENUM after
    the transform -- raises ValueError on anything that doesn't land on a
    known value, rather than silently returning a novel string that would
    then fail a downstream merge with zero rows instead of an error.
    """
    if raw is None:
        raise ValueError("canonicalize_condition() received None")
    cleaned = "_".join(str(raw).strip().lower().split())
    if cleaned not in CONDITION_ENUM:
        raise ValueError(
            f"canonicalize_condition(): '{raw}' normalized to '{cleaned}', "
            f"which is not one of the {len(CONDITION_ENUM)} known conditions "
            f"{sorted(CONDITION_ENUM)}. This almost always means either a "
            f"new/re-cased value appeared in the source data, or the enum "
            f"here is stale -- investigate before proceeding; do not add a "
            f"silent fallback."
        )
    return cleaned


def canonicalize_condition_column(df, column="condition"):
    """Applies canonicalize_condition() to df[column], returning a copy.
    Prints what was mapped (raw -> canonical) for visibility, matching
    this project's standing convention of never silently transforming
    data without a printed trace."""
    df = df.copy()
    mapping = {raw: canonicalize_condition(raw) for raw in df[column].unique()}
    print(f"[condition_names] normalized {len(mapping)} condition value(s): {mapping}")
    df[column] = df[column].map(mapping)
    return df


def assert_nonempty_merge(merged, context=""):
    """Raises if a join returned 0 rows -- the direct fix for the silent
    pd.merge()-returns-0-rows-instead-of-raising risk this module exists
    to prevent. Call this immediately after any merge keyed on a
    `condition` column (or anything else prone to the same silent-empty
    failure) rather than trusting downstream rate/percentage math to
    surface the problem -- 0/0 and n/a look identical to "checked, found
    clean."
    """
    if len(merged) == 0:
        raise ValueError(
            f"Merge returned 0 rows{f' ({context})' if context else ''}. "
            f"This is almost always a silent schema/casing mismatch on the "
            f"join key (e.g. Title Case vs snake_case condition strings), "
            f"not genuinely empty data -- verify the join keys match "
            f"before assuming the result is correct."
        )


def validate_label_coordinates_conditions(df, column="condition"):
    """Warns (does not raise) if any of the 5 expected conditions are
    absent from df[column] after canonicalization. A missing condition
    could be legitimate (a filtered subset) or could indicate an upstream
    canonicalization problem -- surfaced for a human to judge, not
    silently accepted or silently failed."""
    present = set(df[column].unique())
    missing = CONDITION_ENUM - present
    if missing:
        print(
            f"[condition_names] WARNING: expected condition(s) not present "
            f"after canonicalization: {sorted(missing)}. Confirm this is "
            f"expected (e.g. a filtered subset) before trusting downstream "
            f"stats."
        )
    else:
        print("[condition_names] all 5 expected conditions present after canonicalization.")
    return missing
