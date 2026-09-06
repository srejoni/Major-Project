"""
classification/splits.py

Fix for Issue 1 (no held-out validation set): does NOT merge
fold1_train_ids.csv and fold1_val_ids.csv back into one "unified" pool.
fold1_train (~75%) trains the model; fold1_val (~25%) is an explicit
monitoring/checkpoint-selection holdout -- never touched for aggressive
tuning, and never the locked test set.

Re-checks disjointness and locked-test-set overlap here even though the
original splitting script and validate_preprocessing.py already verified
this, consistent with this project's "verify twice" pattern rather than
trusting a guarantee to still hold after being handled by new code.
"""

import pandas as pd

SPLITS_DIR = "/kaggle/working/PDSCD/configs/splits"


def load_classification_split():
    train_ids = pd.read_csv(f"{SPLITS_DIR}/fold1_train_ids.csv")["study_id"].tolist()
    val_ids = pd.read_csv(f"{SPLITS_DIR}/fold1_val_ids.csv")["study_id"].tolist()
    locked_ids = pd.read_csv(f"{SPLITS_DIR}/locked_test_ids.csv")["study_id"].tolist()

    train_set, val_set, locked_set = set(train_ids), set(val_ids), set(locked_ids)

    assert not (train_set & val_set), f"train/val overlap: {train_set & val_set}"
    assert not (train_set & locked_set), f"train/locked overlap: {train_set & locked_set}"
    assert not (val_set & locked_set), f"val/locked overlap: {val_set & locked_set}"

    print(f"[splits] train={len(train_ids)}  val={len(val_ids)}  "
          f"locked_test(untouched)={len(locked_ids)}")
    return train_ids, val_ids


if __name__ == "__main__":
    load_classification_split()
