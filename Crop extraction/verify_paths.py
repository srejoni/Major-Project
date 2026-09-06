"""
classification/verify_paths.py

Confirms actual file/folder paths on Kaggle before trusting any hardcoded
config value that assumes them -- per this project's repeated "verify with
!ls before assuming" lesson. Checks the 4 items margin_tuning.py depends on:
  - train_series_descriptions.csv
  - train_label_coordinates.csv
  - best.pt (YOLO weights)
  - train_images/ directory

Falls back to a filesystem search (find_by_name) for anything not resolved
by the candidate-path list -- equivalent to `!find /kaggle -iname "<name>"`.
"""

import os
import subprocess

REPO_ROOT = "/kaggle/working/PDSCD"

CANDIDATES = {
    "train_series_descriptions.csv": [
        "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv",
        "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv",
    ],
    "train_label_coordinates.csv": [
        "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv",
        "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv",
    ],
    "best.pt": [
        f"{REPO_ROOT}/localization/weights/best.pt",       # confirmed real location, per progress doc §4
        f"{REPO_ROOT}/weights/localization/best.pt",        # originally-suggested (wrong) location
        f"{REPO_ROOT}/localization/runs/detect/train/weights/best.pt",  # default ultralytics output path
    ],
    "train_images": [
        "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images",
        "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images",
    ],
}


def list_top_level(path):
    print(f"\n--- ls {path} ---")
    if os.path.exists(path):
        for entry in sorted(os.listdir(path)):
            print(" ", entry)
    else:
        print("  (path does not exist)")


def find_by_name(filename, search_root="/kaggle"):
    result = subprocess.run(
        ["find", search_root, "-iname", filename],
        capture_output=True, text=True, timeout=60
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def verify():
    list_top_level("/kaggle/input")
    list_top_level("/kaggle/working")

    resolved = {}
    print("\n--- Candidate-path check ---")
    for name, paths in CANDIDATES.items():
        found_path = next((p for p in paths if os.path.exists(p)), None)
        if found_path:
            print(f"[FOUND]     {name:<35} -> {found_path}")
            resolved[name] = found_path
        else:
            print(f"[NOT FOUND] {name:<35} -> none of {len(paths)} candidates matched")
            print(f"            searching filesystem for '{name}'...")
            hits = find_by_name(name)
            if hits:
                print(f"            found via search: {hits[0]}")
                if len(hits) > 1:
                    print(f"            ({len(hits)-1} other match(es) also found)")
                resolved[name] = hits[0]
            else:
                print("            not found anywhere under /kaggle")
                resolved[name] = None

    print("\n--- Resolved paths summary ---")
    for name, path in resolved.items():
        status = "OK" if path else "MISSING"
        print(f"  [{status}] {name}: {path}")

    return resolved


if __name__ == "__main__":
    verify()
