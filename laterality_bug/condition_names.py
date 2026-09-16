"""
classification/condition_names.py

Single source of truth for translating EXTERNAL condition-name strings onto
this project's snake_case condition enum, and for refusing to continue when
a join on those names silently matches nothing.

WHY THIS EXISTS
---------------
train_label_coordinates.csv names conditions in Title Case with spaces:

    ['Left Neural Foraminal Narrowing', 'Left Subarticular Stenosis',
     'Right Neural Foraminal Narrowing', 'Right Subarticular Stenosis',
     'Spinal Canal Stenosis']

...confirmed by direct inspection (notebook48c0744e7b cell [26], Sept 14
2026 -- all 5 values, not a sample). The project's own enum -- used by
crop_extraction.py's SERIES_BY_CONDITION, the manifest's `condition`
column, and pdscd_output_schema.json -- is snake_case:

    left_neural_foraminal_narrowing, left_subarticular_stenosis,
    right_neural_foraminal_narrowing, right_subarticular_stenosis,
    spinal_canal_stenosis

A pd.merge across that mismatch does NOT raise. It returns 0 rows and every
downstream percentage computes as 0/0 or n/a -- which reads like "no
problem found" rather than "nothing was compared." That failure mode is
the reason this module fails loud instead of returning a best-effort map.

SCOPE -- train.csv does NOT need this. Its severity columns are already
snake_case (`{condition}_{level_slug}`), proven by the fact that
crop_extraction.py's existing `train_df.loc[study_id].get(col)` lookup
produced 41,375 correctly-labeled manifest rows. Only
train_label_coordinates.csv carries the Title Case convention.

WHY A SEPARATE MODULE, not a helper inside crop_extraction.py:
  1. Importing crop_extraction pulls in ultralytics/YOLO and pydicom.
     Coordinate-only diagnostics (e.g. validate_axial_lr_geometry.py)
     shouldn't need a YOLO import to read a CSV.
  2. If each script keeps its own CONDITION_NAME_MAP dict, they drift --
     exactly the failure mode already fixed once by extracting
     merge_series_detections() out of two inline copies.

This module deliberately mirrors the shape of crop_extraction.py's
existing _canonicalize_level_string() / validate_yolo_class_mapping()
pair: a normalizer plus a loud validator.
"""

# The project enum. Must stay identical to crop_extraction.py's
# SERIES_BY_CONDITION keys and pdscd_output_schema.json's `condition` enum.
CONDITIONS = (
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
)

LATERAL_CONDITIONS = (
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
)


def canonicalize_condition(raw: str) -> str:
    """Normalize one external condition string to the project enum.

    Deterministic transform (strip -> lowercase -> collapse internal
    whitespace -> underscore-join) rather than a hardcoded lookup dict,
    so a new or re-cased value can't silently fall through a missing key.
    The result is then validated against CONDITIONS, so a value that
    normalizes to something NOT in the enum raises here instead of
    becoming a silently-unjoinable row later.

    Verified against all 5 real values in train_label_coordinates.csv --
    the transform reproduces the project enum exactly for every one.
    """
    canon = "_".join(str(raw).strip().lower().split())
    if canon not in CONDITIONS:
        raise ValueError(
            f"Condition string {raw!r} normalizes to {canon!r}, which is not "
            f"one of the project's {len(CONDITIONS)} conditions: {list(CONDITIONS)}. "
            f"Either the source file's naming convention changed, or this is a "
            f"condition the pipeline doesn't model -- do not map it by hand "
            f"without checking which."
        )
    return canon


def canonicalize_condition_column(df, col: str = "condition", inplace: bool = False):
    """Return `df` with `col` mapped onto the project enum.

    Reports the distinct mapping applied, so a run's log shows what was
    translated rather than leaving it implicit.
    """
    if col not in df.columns:
        raise KeyError(
            f"Column {col!r} not in dataframe; found: {list(df.columns)}"
        )
    out = df if inplace else df.copy()
    distinct = sorted(out[col].dropna().unique())
    mapping = {raw: canonicalize_condition(raw) for raw in distinct}
    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        print(f"[condition_names] normalized {len(changed)} condition value(s): {changed}")
    out[col] = out[col].map(mapping)
    return out


def assert_nonempty_merge(merged, left_desc: str, right_desc: str, on) -> None:
    """Fail loudly on a join that matched nothing.

    A 0-row merge is the specific silent failure this module exists to
    prevent: pandas treats it as a valid result, and every downstream
    rate/percentage then reads as 0/0 or n/a -- indistinguishable from
    "checked, found clean."
    """
    if len(merged) == 0:
        raise ValueError(
            f"Merge of {left_desc} x {right_desc} on {on} produced 0 rows.\n"
            f"This is almost never a real 'no overlap' result -- check that "
            f"BOTH sides were passed through canonicalize_condition_column() "
            f"before joining, and that the `level` strings on both sides use "
            f"the same delimiter (this project's enum uses 'L1/L2', while "
            f"best.pt's model.names uses 'L1-L2' -- see "
            f"crop_extraction._canonicalize_level_string)."
        )


def validate_label_coordinates_conditions(df, col: str = "condition") -> None:
    """Confirm a loaded train_label_coordinates.csv covers every condition
    the pipeline expects, after canonicalization.

    Missing conditions aren't necessarily fatal for every caller, so this
    warns rather than raises -- but it warns explicitly, so a diagnostic
    can't quietly report on 3 conditions while implying it covered 5.
    """
    present = {canonicalize_condition(v) for v in df[col].dropna().unique()}
    missing = set(CONDITIONS) - present
    if missing:
        print(
            f"[condition_names] WARNING: these conditions are absent from the "
            f"coordinates file after canonicalization: {sorted(missing)}. "
            f"Any per-condition result below covers only {sorted(present)}."
        )
    else:
        print(f"[condition_names] all {len(CONDITIONS)} conditions present and canonicalized.")
