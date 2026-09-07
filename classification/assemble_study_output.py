"""
classification/assemble_study_output.py

THE MISSING PIECE: no prior step guaranteed a fixed 25-entry (5 conditions
x 5 levels) output per study. crop_extraction.py and build_dataloaders.py
only ever drop rows (missing severity, duplicates); nothing padded a
study's array back out to 25. This is that step.

Given a study's crop dict (crop_extraction_inference.extract_study_crops)
and loaded model(s) (ensemble_predict.load_models), this produces exactly
what pdscd_output_schema.json v1.0.0 requires: always 25 predictions, one
per (condition, level), with status="missing_detection" and all-null
fields for any combo that never produced a usable crop.

CHANGE (duplicate_resolved fix): crop_extraction_inference.py now threads
a "had_duplicate" flag through each crop entry -- True whenever more than
one series_id contributed a candidate detection for that (series, level)
and the higher-confidence one was kept. This was previously discarded
before reaching this file, meaning status="duplicate_resolved" (documented
and promised in pdscd_output_format.md §5) could never actually appear in
real output. It is now set correctly below instead of defaulting every
successful prediction to status="ok".

DOCUMENTED CONVENTIONS (previously undisclosed, code-only decisions --
now also written up in pdscd_output_format.md §5a; update that doc if
either value below ever changes):
- competition_weight for status="missing_detection": there is no severity
  to derive a weight from. An unassessed level is treated as
  maximum-attention, not minimum-attention -- defaulting to 1 (Normal/Mild's
  weight) would rank an unassessed level as equally low-priority as a
  confirmed benign finding, which is backwards for a clinical-adjacent use
  case. Default is 4 (Severe's weight) until the format-doc owner signs off
  on a different convention. This replaces an earlier, undocumented default
  of 1, which was a guess made before this was resolved.
- low_confidence_count threshold: ensemble top-class probability below this
  counts as "low confidence" in study_summary.

FIX (normalization): now imports classification.normalization.normalize_crop
-- the single function also used by crop_dataset.py at training time --
instead of reimplementing normalization inline. This project has already
hit two separate train/inference normalization-mismatch bugs; this closes
the door on a third one specifically at this seam.

FIX (_model_versions_block): now derives its output from the `models` dict
actually passed into assemble_study() / assemble_study(), not from the
static MODEL_REGISTRY. Previously these always matched only because
load_models() has no partial-failure path -- one refactor away from
silently claiming a model ran when it didn't. Now honest by construction.

FIX (partial-ensemble guard): main() now hard-fails before assembling
output if fewer than 3 models loaded, instead of only failing later at
schema validation with a less obvious error. --skip-schema-check exists
for local dev only and must never be used for anything handed to the LLM
integration teammate.

REMAINING BLOCKER (inherited from ensemble_predict.py): with only
efficientnet_b3 trained, model_versions/model_votes will only contain one
key, which pdscd_output_schema.json's additionalProperties:false rejects.
Real output from this script will correctly FAIL (now via the explicit
guard in main(), in addition to schema validation) until convnext_tiny and
resnet34 are trained and registered. Use --skip-schema-check to run anyway
for development before that happens.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

sys.path.append("/kaggle/working/PDSCD")

from classification.crop_extraction import SERIES_BY_CONDITION, LEVELS
from classification.ensemble_predict import predict_crop, MODEL_REGISTRY
from classification.normalization import normalize_crop

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "pdscd-ensemble-v1"
CONDITIONS = list(SERIES_BY_CONDITION.keys())

SEVERITY_LABEL_BY_CODE = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}
COMPETITION_WEIGHT_BY_CODE = {0: 1, 1: 2, 2: 4}

# --- documented conventions (previously undisclosed code-only decisions) ---
# competition_weight for status="missing_detection": there is no severity to
# derive a weight from. Per pdscd_output_format.md §5a, missing_detection is
# treated as maximum-attention, not minimum-attention -- defaulting to 1
# (Normal/Mild's weight) would rank an unassessed level as equally
# low-priority as a confirmed benign finding, which is backwards for a
# clinical-adjacent use case. Default is 4 (Severe's weight) until the
# format-doc owner signs off on a different convention.
MISSING_DETECTION_WEIGHT_DEFAULT = 4

# low_confidence_count threshold: ensemble top-class probability below this
# counts as "low confidence" in study_summary. Documented in
# pdscd_output_format.md §5a -- previously only defined here.
LOW_CONFIDENCE_THRESHOLD = 0.5


def _normalize_crop(crop_array, mean, std):
    """Thin wrapper kept for call-site stability; delegates to the single
    shared implementation in classification/normalization.py, which is also
    what ClassificationCropDataset uses at training time."""
    return normalize_crop(crop_array, mean, std)


def _model_versions_block(models):
    """Derive from the models actually loaded, not the static registry --
    keeps this honest if load_models() is ever changed to skip a failed
    checkpoint instead of raising."""
    return {name: MODEL_REGISTRY[name]["checkpoint"].split("/")[-1] for name in models}


def assemble_study(study_id, crops_by_condition_level, models, mean, std, device):
    predictions = []
    severity_counts = {"Normal/Mild": 0, "Moderate": 0, "Severe": 0}
    missing_count = 0
    low_conf_count = 0
    highest_code = None

    for condition in CONDITIONS:
        for level in LEVELS:
            crop_info = crops_by_condition_level.get((condition, level))

            if crop_info is None:
                predictions.append({
                    "condition": condition,
                    "level": level,
                    "status": "missing_detection",
                    "severity_code": None,
                    "severity_label": None,
                    "confidence": None,
                    "probabilities": None,
                    "competition_weight": MISSING_DETECTION_WEIGHT_DEFAULT,
                    "model_votes": None,
                    "model_agreement": None,
                    "source": {
                        "series_description": SERIES_BY_CONDITION[condition],
                        "n_slices_used": None,
                        "yolo_detection_confidence": None,
                    },
                })
                missing_count += 1
                continue

            tensor = _normalize_crop(crop_info["crop"], mean, std)
            result = predict_crop(models, tensor, device)

            # A real prediction exists, but if more than one series_id
            # contributed a candidate detection for this (series, level)
            # and the higher-confidence one was kept, that's the
            # duplicate-resolution case pdscd_output_format.md §5 promises
            # a distinct status for -- not a plain "ok".
            status = "duplicate_resolved" if crop_info.get("had_duplicate") else "ok"

            predictions.append({
                "condition": condition,
                "level": level,
                "status": status,
                "severity_code": result["severity_code"],
                "severity_label": result["severity_label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "competition_weight": COMPETITION_WEIGHT_BY_CODE[result["severity_code"]],
                "model_votes": result["model_votes"],
                "model_agreement": result["model_agreement"],
                "source": {
                    "series_description": crop_info["series_description"],
                    "n_slices_used": crop_info["n_slices_used"],
                    "yolo_detection_confidence": round(crop_info["yolo_detection_confidence"], 6),
                },
            })

            severity_counts[result["severity_label"]] += 1
            if result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                low_conf_count += 1
            if highest_code is None or result["severity_code"] > highest_code:
                highest_code = result["severity_code"]

    study_summary = {
        "highest_severity_code": highest_code,
        "highest_severity_label": SEVERITY_LABEL_BY_CODE.get(highest_code),
        "severity_counts": severity_counts,
        "missing_prediction_count": missing_count,
        "low_confidence_count": low_conf_count,
    }

    n_models = len(models)
    equal_weight = round(1.0 / n_models, 4) if n_models else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": int(study_id),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version": PIPELINE_VERSION,
        "model_versions": _model_versions_block(models),
        "ensemble_method": {
            "type": "soft_probability_averaging",
            "weights": {name: equal_weight for name in models},
            "note": f"Equal weighting across {n_models} available model(s). "
                    f"Schema requires exactly 3 -- see module docstring blocker.",
        },
        "study_summary": study_summary,
        "predictions": predictions,
    }


def validate_against_schema(output, schema_path):
    import jsonschema
    with open(schema_path) as f:
        schema = json.load(f)
    jsonschema.validate(instance=output, schema=schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", type=int, required=True)
    parser.add_argument("--schema-path", default="/kaggle/working/PDSCD/pdscd_output_schema.json")
    parser.add_argument("--skip-schema-check", action="store_true",
                         help="Use before all 3 models are trained -- real output "
                              "will otherwise correctly fail validation. Local dev "
                              "only -- never use for anything handed to the LLM "
                              "integration teammate.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    import pandas as pd
    from ultralytics import YOLO
    from classification.crop_extraction_inference import extract_study_crops
    from classification.ensemble_predict import load_models
    from classification.build_dataloaders import _load_fold1_stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _load_fold1_stats()
    models = load_models(device)

    if not args.skip_schema_check and len(models) != 3:
        raise RuntimeError(
            f"Only {len(models)} model(s) loaded ({list(models.keys())}). Real "
            f"output requires all 3 (schema additionalProperties: false on "
            f"model_votes/model_versions, and model_agreement is non-nullable "
            f"when status='ok'). This will correctly fail schema validation -- "
            f"do not ship this output. Use --skip-schema-check only for local "
            f"dev, never for anything handed to the LLM integration teammate."
        )

    DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
    YOLO_WEIGHTS = "/kaggle/working/PDSCD/localization/weights/best.pt"
    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    yolo_model = YOLO(YOLO_WEIGHTS)

    crops = extract_study_crops(yolo_model, args.study_id, series_df, DATA_ROOT)
    output = assemble_study(args.study_id, crops, models, mean, std, device)

    if not args.skip_schema_check:
        validate_against_schema(output, args.schema_path)
        print("[assemble] schema validation passed")

    out_path = args.out or f"/kaggle/working/pdscd_output_{args.study_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[assemble] wrote {out_path}")


if __name__ == "__main__":
    main()
