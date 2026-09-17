"""
classify_axial_calibrated_mismatches.py

Same shape as classify_axial_mismatches.py (problem log §4c: group by
study, separate "systematic" (>=4/5 levels wrong -> likely whole-study
GT swap) from "sporadic" (true near-midline ambiguity) mismatches) -
pointed at the CALIBRATED single-box midline proxy's misses instead of
the relative (paired) proxy's misses.

ASSUMES the calibrated validator wrote a per-annotation results table
with columns: study_id, level, predicted_side, gt_side, margin.
Confirm this against the real script (Step 1 above) before trusting
anything below - these names are inferred, not confirmed.
"""

import pandas as pd

# --- adjust path/variable to however your calibrated validator persisted results ---
results = pd.read_csv("PATH_TO_CALIBRATED_VALIDATOR_OUTPUT.csv")

REQUIRED_COLS = {"study_id", "level", "predicted_side", "gt_side", "margin"}
missing = REQUIRED_COLS - set(results.columns)
if missing:
    raise ValueError(
        f"calibrated results missing expected columns: {missing}. "
        f"Actual columns: {list(results.columns)}. "
        "Re-check validate_axial_series_calibrated_r...py's output schema."
    )

results["match"] = results["predicted_side"] == results["gt_side"]
mismatches = results[~results["match"]].copy()
mismatches["abs_margin"] = mismatches["margin"].abs()

print(f"Total single-box annotations checked: {len(results)}")
print(f"Total mismatches: {len(mismatches)} ({len(mismatches)/len(results):.2%})")

# --- group by study, same >=4/5 threshold the relative-proxy classification used ---
per_study = results.groupby("study_id").agg(
    levels_checked=("level", "count"),
    levels_wrong=("match", lambda s: (~s).sum()),
)
per_study["frac_wrong"] = per_study["levels_wrong"] / per_study["levels_checked"]

SYSTEMATIC_THRESHOLD = 0.8  # >=4/5 levels wrong
systematic_studies = per_study[per_study["frac_wrong"] >= SYSTEMATIC_THRESHOLD].index.tolist()
print(f"\nStudies with >=80% of checked levels wrong: {len(systematic_studies)}")

# --- cross-reference against the known 22 GT-swap studies from the relative-proxy test ---
KNOWN_GT_SWAP_STUDIES = {
    52695609, 105895264, 183230492, 211314658, 416503281, 1106510276,
    859570985, 1018005303, 886995462, 1373010257, 1906657742, 1670838975,
    1459964234, 4072191052, 4017932238, 3857195576, 2447825792, 2797118205,
    3837345060, 3617361428, 3429409220, 3234424112,
}

systematic_set = set(systematic_studies)
overlap = systematic_set & KNOWN_GT_SWAP_STUDIES
new_unexplained = sorted(systematic_set - KNOWN_GT_SWAP_STUDIES)

print(f"Overlap with known 22: {len(overlap)} -> {sorted(overlap)}")
print(f"NEW unexplained systematic studies: {len(new_unexplained)}")
print(new_unexplained)

# sanity check: your smoke-test study shouldn't be in this list
print("4003253 in new_unexplained:", 4003253 in new_unexplained)

# --- margin distribution among all mismatches, same reporting shape as §4c ---
print("\n|margin| distribution among mismatches:")
print(mismatches["abs_margin"].describe())
print(f"|margin| < 5px (near-midline ambiguity): {(mismatches['abs_margin'] < 5).sum()}")
print(f"|margin| >= 20px (systematic/GT-scale gap): {(mismatches['abs_margin'] >= 20).sum()}")

# --- per-mismatch detail for the new unexplained studies, for spot-checking ---
detail = mismatches[mismatches["study_id"].isin(new_unexplained)].sort_values(["study_id", "level"])
print("\nPer-mismatch detail for new unexplained studies:")
print(detail[["study_id", "level", "predicted_side", "gt_side", "margin"]].to_string(index=False))

out_path = "/kaggle/working/classification_data/axial_calibrated_new_unexplained_studies.csv"
pd.DataFrame({"study_id": new_unexplained}).to_csv(out_path, index=False)
print(f"\nSaved {len(new_unexplained)} study_ids to {out_path}")
