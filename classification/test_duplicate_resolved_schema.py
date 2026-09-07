"""
classification/test_duplicate_resolved_schema.py

One-off confirmation script -- NOT part of the pipeline, NOT meant to be
imported elsewhere. Run this once after pulling the patched
pdscd_output_schema.json and assemble_study_output.py, before relying on
either in production, to confirm the new duplicate_resolved branch in the
schema's allOf actually rejects null prediction fields and accepts real
ones -- exactly the same shape of check the existing "ok" branch already
passes, just on the new "duplicate_resolved" const.

This builds two full, schema-shaped study outputs (25 predictions each,
minItems/maxItems=25 per pdscd_output_schema.json) that differ in exactly
one entry:
  - CASE A: that entry has status="duplicate_resolved" with every
    prediction field null (severity_code, severity_label, confidence,
    probabilities, model_votes, model_agreement). Before this patch this
    passed validation silently. Expected result now: FAILS.
  - CASE B: the same entry, but with real, non-null values -- the shape
    pdscd_output_format.md §5 promises for duplicate_resolved ("treat like
    ok"). Expected result: PASSES.

A third case (CASE C) re-confirms the existing "ok" and "missing_detection"
branches were not broken by this edit, since they're the two branches this
duplicate_resolved case sits next to in the same allOf list.

Exit code is 0 only if all three cases behave as expected; non-zero
otherwise, so this is safe to wire into a CI/pre-push check later if
wanted -- not done here since no CI config was provided.
"""
import sys
import copy

sys.path.append("/kaggle/working/PDSCD")

from classification.assemble_study_output import (
    validate_against_schema, CONDITIONS, LEVELS,
)

SCHEMA_PATH = "/kaggle/working/PDSCD/pdscd_output_schema.json"


def _filler_prediction(condition, level):
    """A plain, always-valid missing_detection entry -- used to pad the
    predictions array out to the required 25 without the filler itself
    being the thing under test."""
    return {
        "condition": condition,
        "level": level,
        "status": "missing_detection",
        "severity_code": None,
        "severity_label": None,
        "confidence": None,
        "probabilities": None,
        "competition_weight": 4,
        "model_votes": None,
        "model_agreement": None,
        "source": {
            "series_description": None,
            "n_slices_used": None,
            "yolo_detection_confidence": None,
        },
    }


def _duplicate_resolved_prediction(with_real_values):
    """The single entry under test. condition/level are fixed to a real
    axial/subarticular combo -- the documented hotspot for this dataset's
    real duplicate-detection case -- purely for readability; the schema
    itself doesn't care which condition/level this is."""
    base = {
        "condition": "left_subarticular_stenosis",
        "level": "L4/L5",
        "status": "duplicate_resolved",
        "competition_weight": 2,
        "source": {
            "series_description": "Axial T2",
            "n_slices_used": 3,
            "yolo_detection_confidence": 0.91,
        },
    }
    if not with_real_values:
        base.update({
            "severity_code": None,
            "severity_label": None,
            "confidence": None,
            "probabilities": None,
            "model_votes": None,
            "model_agreement": None,
        })
    else:
        base.update({
            "severity_code": 1,
            "severity_label": "Moderate",
            "confidence": 0.62,
            "probabilities": {"normal_mild": 0.20, "moderate": 0.62, "severe": 0.18},
            "model_votes": {
                "efficientnet_b3": {
                    "severity_code": 1,
                    "probabilities": {"normal_mild": 0.20, "moderate": 0.60, "severe": 0.20},
                },
                "convnext_tiny": {
                    "severity_code": 1,
                    "probabilities": {"normal_mild": 0.15, "moderate": 0.65, "severe": 0.20},
                },
                "resnet34": {
                    "severity_code": 1,
                    "probabilities": {"normal_mild": 0.25, "moderate": 0.60, "severe": 0.15},
                },
            },
            "model_agreement": "unanimous_3_of_3",
        })
    return base


def _base_output(target_prediction):
    """Wraps one entry under test inside a full, otherwise-valid 25-entry
    study output -- the level validate_against_schema() actually operates
    on, not the bare prediction object."""
    predictions = [target_prediction]
    all_pairs = [(c, l) for c in CONDITIONS for l in LEVELS]
    for condition, level in all_pairs[:24]:
        predictions.append(_filler_prediction(condition, level))

    return {
        "schema_version": "1.0.0",
        "study_id": 999999999,
        "generated_at": "2026-09-08T00:00:00Z",
        "pipeline_version": "pdscd-ensemble-v1",
        "model_versions": {
            "efficientnet_b3": "effnetb3_fold_none_v1.pt",
            "convnext_tiny": "convnext_tiny_v1.pt",
            "resnet34": "resnet34_v1.pt",
        },
        "ensemble_method": {
            "type": "soft_probability_averaging",
            "weights": {"efficientnet_b3": 0.3333, "convnext_tiny": 0.3333, "resnet34": 0.3334},
        },
        "study_summary": {
            "highest_severity_code": 1,
            "highest_severity_label": "Moderate",
            "severity_counts": {"Normal/Mild": 0, "Moderate": 1, "Severe": 0},
            "missing_prediction_count": 24,
            "low_confidence_count": 0,
        },
        "predictions": predictions,
    }


def _run_case(name, output, expect_pass):
    try:
        validate_against_schema(output, SCHEMA_PATH)
        passed = True
        error_msg = None
    except Exception as e:
        passed = False
        error_msg = str(e).splitlines()[0]  # jsonschema errors are long; first line is enough

    ok = (passed == expect_pass)
    status = "PASS" if ok else "FAIL <-- UNEXPECTED"
    outcome = "validated OK" if passed else f"rejected ({error_msg})"
    print(f"[{status}] {name}: expected {'valid' if expect_pass else 'invalid'}, got {outcome}")
    return ok


def main():
    results = []

    # CASE A: duplicate_resolved with nulls -- must now be REJECTED
    out_a = _base_output(_duplicate_resolved_prediction(with_real_values=False))
    results.append(_run_case(
        "CASE A (duplicate_resolved, null fields)", out_a, expect_pass=False
    ))

    # CASE B: duplicate_resolved with real values -- must PASS
    out_b = _base_output(_duplicate_resolved_prediction(with_real_values=True))
    results.append(_run_case(
        "CASE B (duplicate_resolved, real values)", out_b, expect_pass=True
    ))

    # CASE C: confirm "ok" branch (adjacent in the same allOf) still passes
    out_c = copy.deepcopy(out_b)
    out_c["predictions"][0]["status"] = "ok"
    results.append(_run_case(
        "CASE C (status=ok, real values, regression check)", out_c, expect_pass=True
    ))

    # CASE D: confirm "missing_detection" branch still passes with nulls
    out_d = _base_output(_duplicate_resolved_prediction(with_real_values=False))
    out_d["predictions"][0]["status"] = "missing_detection"
    out_d["predictions"][0]["source"]["series_description"] = None
    out_d["predictions"][0]["source"]["n_slices_used"] = None
    out_d["predictions"][0]["source"]["yolo_detection_confidence"] = None
    results.append(_run_case(
        "CASE D (status=missing_detection, nulls, regression check)", out_d, expect_pass=True
    ))

    print()
    if all(results):
        print("All cases behaved as expected. Schema patch confirmed safe to push.")
        sys.exit(0)
    else:
        print("At least one case did NOT behave as expected. Do NOT push yet -- "
              "re-check pdscd_output_schema.json's duplicate_resolved allOf branch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
