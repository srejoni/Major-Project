import sys
sys.path.append("/kaggle/working/PDSCD")
sys.path.append("/kaggle/working/PDSCD/laterality_bug")

import pandas as pd
import validate_axial_single_box_midline as vs
from classification.condition_names import canonicalize_condition_column

KNOWN_GT_SWAP_STUDIES = {
    52695609, 105895264, 183230492, 211314658, 416503281, 1106510276,
    859570985, 1018005303, 886995462, 1373010257, 1906657742, 1670838975,
    1459964234, 4072191052, 4017932238, 3857195576, 2447825792, 2797118205,
    3837345060, 3617361428, 3429409220, 3234424112,
}

coords = canonicalize_condition_column(pd.read_csv(f"{vs.DATA_ROOT}/train_label_coordinates.csv"))
sub = coords[coords["condition"].isin(
    ["left_subarticular_stenosis", "right_subarticular_stenosis"]
)].copy()

records = []
for r in sub.itertuples(index=False):
    lv, mid = vs.leftness_and_midline(r.study_id, r.series_id, r.instance_number, r.x)
    if lv is None:
        continue
    margin = lv - mid
    is_left_cond = r.condition == "left_subarticular_stenosis"
    correct = (margin > 0) == is_left_cond
    records.append({"study_id": r.study_id, "correct": correct})

df = pd.DataFrame(records)
wrong = df[~df["correct"]]
per_study = wrong.groupby("study_id").size()
systematic_28 = set(per_study[per_study >= 4].index)

print(f"Systematic (>=4 misclassified) studies this test: {len(systematic_28)}")
print(f"Overlap with the 22 known GT-swap studies: {len(systematic_28 & KNOWN_GT_SWAP_STUDIES)}")
print(f"NEW studies not in the known 22: {sorted(systematic_28 - KNOWN_GT_SWAP_STUDIES)}")
print(f"Known 22 that did NOT reappear here: {sorted(KNOWN_GT_SWAP_STUDIES - systematic_28)}")
