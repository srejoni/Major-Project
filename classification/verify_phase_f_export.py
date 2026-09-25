"""
scripts/verify_phase_f_export.py

Phase F verification: runs the real (non---skip-schema-check) 3-model
ensemble path in classification/assemble_study_output.py on a handful of
real studies and inspects the resulting exports for the two things Phase F
exists to check -- model_agreement and duplicate_resolved -- plus a set of
sanity assertions that would catch the two known regressions this pass is
guarding against (T3-5 import-time drift, T4-7 filename/note leakage).

Usage (from repo root, inside the Kaggle working dir where
classification/ is importable):

    python scripts/verify_phase_f_export.py
    python scripts/verify_phase_f_export.py --study-ids 12345 67890 54321
    python scripts/verify_phase_f_export.py --n-studies 2

Does NOT skip schema validation -- if assemble_study_output.py fails, this
fails too, loudly. This script does not modify assemble_study_output.py;
it only calls it as a subprocess and inspects the JSON it writes.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
ASSEMBLE_SCRIPT = REPO_ROOT / "classification" / "assemble_study_output.py"

REAL_STATUSES = ("ok", "duplicate_resolved")


def pick_study_ids(n):
    """Auto-select n real study_ids from train_series_descriptions.csv so
    this can be run with zero arguments on a fresh Kaggle session."""
    import pandas as pd
    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    ids = series_df["study_id"].drop_duplicates().tolist()
    if len(ids) < n:
        raise RuntimeError(f"Only {len(ids)} study_ids available, need {n}.")
    return ids[:n]


def run_assemble(study_id):
    """Real run -- no --skip-schema-check. Raises on any non-zero exit,
    since a Phase F failure here is exactly what this script exists to
    catch."""
    out_path = f"/kaggle/working/pdscd_output_{study_id}_phaseF.json"
    cmd = [
        sys.executable, str(ASSEMBLE_SCRIPT),
        "--study-id", str(study_id),
        "--out", out_path,
    ]
    print(f"[verify] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"assemble_study_output.py failed for study {study_id} "
            f"(exit {result.returncode}) -- see stderr above."
        )
    return Path(out_path)


def check_export(path):
    """The Phase F checks from the completion plan: 25 predictions,
    model_agreement distribution, duplicate_resolved count, a clean
    ensemble_method.note, and all 3 models present in model_versions."""
    with open(path) as f:
        data = json.load(f)

    problems = []

    if "_UNVALIDATED_DEV_ONLY" in path.name:
        problems.append("filename carries _UNVALIDATED_DEV_ONLY on what should be a real export")

    predictions = data.get("predictions", [])
    n_preds = len(predictions)
    if n_preds != 25:
        problems.append(f"expected 25 predictions, got {n_preds}")

    model_versions = data.get("model_versions", {})
    expected_models = {"efficientnet_b3", "convnext_tiny", "resnet34"}
    if set(model_versions.keys()) != expected_models:
        problems.append(f"model_versions keys = {sorted(model_versions.keys())}, expected {sorted(expected_models)}")

    note = data.get("ensemble_method", {}).get("note", "")
    if "skip-schema-check" in note.lower() or "dev/partial" in note.lower():
        problems.append(f"ensemble_method.note still reads as a dev/partial run: {note!r}")

    agreement = Counter(
        p.get("model_agreement") for p in predictions if p.get("status") in REAL_STATUSES
    )
    dup_resolved = sum(1 for p in predictions if p.get("status") == "duplicate_resolved")
    missing = sum(1 for p in predictions if p.get("status") == "missing_detection")

    # model_agreement must never be null on a status that isn't missing_detection
    null_agreement_on_real = [
        (p["condition"], p["level"]) for p in predictions
        if p.get("status") in REAL_STATUSES and p.get("model_agreement") is None
    ]
    if null_agreement_on_real:
        problems.append(
            f"model_agreement is null on {len(null_agreement_on_real)} non-missing "
            f"predictions: {null_agreement_on_real}"
        )

    return {
        "path": str(path),
        "study_id": data.get("study_id"),
        "schema_version": data.get("schema_version"),
        "pipeline_version": data.get("pipeline_version"),
        "n_predictions": n_preds,
        "missing_detection_count": missing,
        "duplicate_resolved_count": dup_resolved,
        "model_agreement_distribution": dict(agreement),
        "ensemble_note": note,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-ids", type=int, nargs="+", default=None,
                         help="Real study_ids to run. Default: auto-pick from train_series_descriptions.csv.")
    parser.add_argument("--n-studies", type=int, default=3,
                         help="How many studies to auto-pick if --study-ids is not given.")
    args = parser.parse_args()

    study_ids = args.study_ids or pick_study_ids(args.n_studies)
    print(f"[verify] Phase F check on study_ids: {study_ids}")

    results = []
    any_problems = False
    for sid in study_ids:
        out_path = run_assemble(sid)
        report = check_export(out_path)
        results.append(report)
        if report["problems"]:
            any_problems = True

    print("\n" + "=" * 70)
    print("PHASE F SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\nstudy_id={r['study_id']}  ({r['path']})")
        print(f"  schema_version={r['schema_version']}  pipeline_version={r['pipeline_version']}")
        print(f"  predictions count (expect 25): {r['n_predictions']}")
        print(f"  missing_detection_count={r['missing_detection_count']}")
        print(f"  duplicate_resolved_count={r['duplicate_resolved_count']}")
        print(f"  model_agreement distribution: {r['model_agreement_distribution']}")
        print(f"  ensemble_method.note: {r['ensemble_note']!r}")
        if r["problems"]:
            print(f"  PROBLEMS: {r['problems']}")
        else:
            print(f"  clean")

    if any_problems:
        print("\n[verify] Phase F FAILED -- see PROBLEMS above.")
        sys.exit(1)
    else:
        print("\n[verify] Phase F PASSED on all studies -- 3-model real export confirmed. "
              "Ready for Phase G (push everything, hand teammate real exports).")


if __name__ == "__main__":
    main()
