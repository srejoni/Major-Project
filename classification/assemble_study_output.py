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

OPEN SPEC QUESTION -- not resolved silently: competition_weight is a
required field on every prediction entry, including missing_detection
ones, but pdscd_output_format.md never defines what it should be when
there's no severity_code to derive it from (the normal derivation is
severity_code -> weight). This script currently defaults missing entries
to 1 (documented below at the point of use) -- that is a guess, not a
confirmed convention, and should be raised with whoever owns the format
doc before this ships.

REMAINING BLOCKER (inherited from ensemble_predict.py): with only
efficientnet_b3 trained, model_versions/model_votes will only contain one
key, which pdscd_output_schema.json's additionalProperties:false rejects.
Real output from this script will correctly FAIL schema validation until
convnext_tiny and resnet34 are trained and registered. Use
--skip-schema-check to run anyway for development before that happens.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

sys.path.append("/kaggle/working/PDSCD")

from classification.crop_extraction import SERIES_BY_CONDITION, LEVELS
from classification.ensemble_predict import predict_crop, MODEL_REGISTRY

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "pdscd-ensemble-v1"
CONDITIONS = list(SERIES_BY_CONDITION.keys())

SEVERITY_LABEL_BY_CODE = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}
COMPETITION_WEIGHT_BY_CODE = {0: 1, 1: 2, 2: 4}
MISSING_DETECTION_WEIGHT_DEFAULT = 1  # see "OPEN SPEC QUESTION" above


def _normalize_crop(crop_array, mean, std):
    """Numerically identical to ClassificationCropDataset.__getitem__'s
    normalization -- duplicated as a plain function here only because that
    Dataset expects a list of file paths, not an in-memory array. If
    crop_dataset.py's normalization ever changes, this must change with it.
    Worth factoring onto one shared function later; flagging the
    duplication rather than hiding it."""
    import numpy as np
    import torch
    img = crop_array.astype(np.float32)
    img = (img - mean) / (std + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1)


def _model_versions_block():
    return {name: MODEL_REGISTRY[name]["checkpoint"].split("/")[-1] for name in MODEL_REGISTRY}


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

            predictions.append({
                "condition": condition,
                "level": level,
                "status": "ok",
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
            if result["confidence"] < 0.5:
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
        "model_versions": _model_versions_block(),
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
                              "will otherwise correctly fail validation.")
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
