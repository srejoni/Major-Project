#!/usr/bin/env python3
"""
classification/export_studies_for_handoff.py

Phase G driver. Orchestrates classification/assemble_study_output.py's own
assemble_study() across a batch of studies, producing one schema-valid
EnsembleStudyOutput JSON file per study for handoff to the LLM-integration
teammate.

This does NOT reimplement any prediction-assembly logic -- missing-detection
handling, competition_weight defaults, duplicate_resolved detection, model
version provenance, etc. all come from the real assemble_study() /
_model_versions_block() already verified in Phase F. This script's only job
is: (1) load YOLO + all 3 classification models + normalization stats once,
not once per study, (2) call the real per-study pipeline for each requested
study_id, (3) validate the result two ways before it's allowed to touch
disk, (4) write study_<id>.json (one file, never split by condition/model),
(5) refuse to ship a partial handoff if any study fails.

Two validation passes, not one:
  - validate_against_schema() -- assemble_study_output.py's own existing
    check, against pdscd_output_schema.json via jsonschema. Reused as-is
    (same function main() already calls) so this stays in lockstep with
    the real pipeline rather than a second, drifting implementation of it.
  - EnsembleStudyOutput.model_validate() -- the teammate's explicit ask
    (handoff spec point 7), against her schema.py directly. The two checks
    are known to be able to drift from each other (see
    llm_integration_context.md section 2) -- passing one is not proof of
    passing the other, so both run, and either one failing aborts the batch.

Place this file in classification/ (flat layout, no scripts/ dir -- see the
Phase F session log's environment note). pdscd_contract_schema.py (a
verbatim copy of the teammate's schema.py) must sit alongside it in the
same directory.

Usage:
    # 1. Single-study TEST export -- do this first, send just this file.
    python classification/export_studies_for_handoff.py \\
        --study-ids 7143189 --out-dir handoff/test

    # 2. Full batch, once the test file is approved.
    python classification/export_studies_for_handoff.py \\
        --study-ids-file study_ids_phase_g.txt \\
        --out-dir handoff/phase_g --zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Needed before the classification.* imports below can resolve -- mirrors
# the same defensive append assemble_study_output.py does at its own top,
# since callers (this script included) can't be relied on to have already
# put the repo root on sys.path.
sys.path.append("/kaggle/working/PDSCD")

from classification.assemble_study_output import (
    assemble_study,
    validate_against_schema,
    SCHEMA_PATH,
)

# Locked contract -- verbatim copy of the teammate's schema.py, saved under
# a name that can't collide with anything else in classification/. Two
# files in this repo were already both named MODEL_REGISTRY with different
# shapes (model_registry.py vs. ensemble_predict.py's checkpoint map) and
# that exact collision is what let the Phase F resnet34 bug go undetected --
# don't recreate that pattern by also calling this one schema.py. Sync it
# by hand whenever the teammate's copy changes; there's no shared import
# path between the two repos/envs.
from pdscd_contract_schema import EnsembleStudyOutput

# Same values main() uses -- not currently exposed as module-level
# constants in assemble_study_output.py, so they're duplicated here rather
# than imported. Worth pulling up to module level there at some point so
# this duplication goes away; not done here to keep this script's diff to
# itself.
DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
YOLO_WEIGHTS = "/kaggle/working/PDSCD/localization/weights/best.pt"


def export_one_study(study_id, yolo_model, series_df, models, mean, std, device, out_dir: Path):
    from classification.crop_extraction_inference import extract_study_crops

    crops = extract_study_crops(yolo_model, study_id, series_df, DATA_ROOT)

    # Never skip_schema_check here -- handoff spec point 2 is explicit that
    # the final export must not use --skip-schema-check.
    output = assemble_study(study_id, crops, models, mean, std, device,
                             skip_schema_check=False)

    validate_against_schema(output, SCHEMA_PATH)
    EnsembleStudyOutput.model_validate(output)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"study_{study_id}.json"
    # Write the dict assemble_study() actually returned, unmodified -- it
    # already uses the correct literal keys (e.g. severity_counts'
    # "Normal/Mild"/"Moderate"/"Severe") and the exact generated_at string
    # format main() ships. Round-tripping it through a pydantic dump instead
    # risks reformatting something (e.g. the "Z"-suffixed timestamp) that
    # both validators above already accepted as correct.
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--study-ids", help="comma-separated study IDs, e.g. 111,222,333")
    group.add_argument("--study-ids-file", help="path to a text file, one study ID per line")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", action="store_true", help="also bundle out-dir into a .zip")
    args = parser.parse_args()

    if args.study_ids:
        study_ids = [int(s.strip()) for s in args.study_ids.split(",") if s.strip()]
    else:
        study_ids = [
            int(line.strip())
            for line in Path(args.study_ids_file).read_text().splitlines()
            if line.strip()
        ]

    # Heavy imports + one-time setup, deferred to here rather than module
    # level -- mirrors assemble_study_output.py's own main(), which keeps
    # torch/pandas/ultralytics/crop-extraction out of the module's top-level
    # imports so the module stays importable without those deps elsewhere.
    import torch
    import pandas as pd
    from ultralytics import YOLO
    from classification.ensemble_predict import load_models
    from classification.build_dataloaders import _load_fold1_stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _load_fold1_stats()
    models = load_models(device)

    # Same guard main() runs, and the same reason: fail here, before the
    # expensive YOLO/crop-extraction work below, not partway through a batch.
    if len(models) != 3:
        print(
            f"[FAIL] only {len(models)} model(s) loaded ({list(models.keys())}). "
            f"Real output requires all 3. Aborting before any study is exported -- "
            f"do not pass skip_schema_check for a handoff export.",
            file=sys.stderr,
        )
        sys.exit(1)

    series_df = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")
    yolo_model = YOLO(YOLO_WEIGHTS)

    out_dir = Path(args.out_dir)
    written: list[Path] = []
    versions_seen: set[tuple] = set()

    for sid in study_ids:
        try:
            path, output = export_one_study(
                sid, yolo_model, series_df, models, mean, std, device, out_dir
            )
        except Exception as exc:
            # Fail loud, fail the whole batch -- a handoff directory missing
            # one study silently is worse than one that errors out. Matches
            # this codebase's existing philosophy (see _model_versions_block
            # raising loudly rather than defaulting on missing provenance).
            print(f"[FAIL] study {sid}: {exc}", file=sys.stderr)
            sys.exit(1)

        versions_seen.add(
            (output["pipeline_version"], tuple(sorted(output["model_versions"].items())))
        )
        written.append(path)
        print(f"[OK] {path}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_ids": study_ids,
        "n_studies": len(written),
        # Satisfies handoff spec point 8 as a single batch-level summary, on
        # top of the per-study pipeline_version/model_versions fields the
        # schema already requires in every individual file.
        "distinct_pipeline_model_version_combos": [
            {"pipeline_version": pv, "model_versions": dict(mv)} for pv, mv in versions_seen
        ],
    }
    (out_dir / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(written)} study file(s) + handoff_manifest.json to {out_dir}/")

    if len(versions_seen) > 1:
        print(
            "WARNING: these studies were exported under more than one "
            "pipeline_version/model_versions combo -- call this out explicitly "
            "in the handoff message (spec point 8 asks which run produced "
            "which result).",
            file=sys.stderr,
        )

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in written + [out_dir / "handoff_manifest.json"]:
                zf.write(f, arcname=f.name)
        print(f"Zipped to {zip_path}")


if __name__ == "__main__":
    main()
