#!/usr/bin/env python3
"""
export_studies_for_handoff.py
==============================
Phase G driver script. Produces one schema-valid EnsembleStudyOutput JSON
file per study, self-validates every file against the locked contract
before it's allowed to land in the output directory, and writes a small
manifest alongside them.

Place this under classification/ (the repo has no scripts/ dir -- see the
Phase F session log's environment note; Phase F's own tooling script hit
this the hard way).

--------------------------------------------------------------------------
ASSUMPTION TO VERIFY (the one thing in this file I could not confirm
against real source, since assemble_study_output.py itself hasn't been
shared in this conversation -- only its documented behavior from the
Phase F session log and completion plan):

    from assemble_study_output import assemble_study
    assemble_study(study_id: int, skip_schema_check: bool = False) -> dict

i.e. a function that runs the real 3-model ensemble for one study and
returns a JSON-serializable dict shaped like EnsembleStudyOutput. This is
consistent with what verify_phase_f_export.py already exercises
("asserts 25 predictions / all 3 models / no dev marker / clean note").
If the real name or signature differs, only `_run_assemble()` below needs
to change -- paste the actual file in and I'll wire it exactly.
--------------------------------------------------------------------------

Usage:
    # 1. Single-study TEST export -- do this first, send just this file.
    python export_studies_for_handoff.py --study-ids 12345 --out-dir handoff/test

    # 2. Full batch, once the test file is approved.
    python export_studies_for_handoff.py \\
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

# Locked contract -- verbatim copy of the teammate's schema.py, saved under
# a distinct name on purpose. Two files in this repo were already both
# named MODEL_REGISTRY with different shapes (model_registry.py vs.
# ensemble_predict.py) and that exact collision is what let the Phase F
# resnet34 bug go undetected -- don't recreate that pattern by calling this
# file schema.py too, since classification/ is flat with no packages to
# namespace it. Sync this file by hand whenever the teammate's schema.py
# changes; there is no shared import path between the two repos/envs.
from pdscd_contract_schema import EnsembleStudyOutput

# The real export function -- see the ASSUMPTION note above.
from assemble_study_output import assemble_study


def _run_assemble(study_id: int) -> dict:
    """Runs the real 3-model ensemble export for one study. Never pass
    skip_schema_check=True here -- point 2 of the handoff spec is explicit
    that the final export must not use --skip-schema-check."""
    return assemble_study(study_id, skip_schema_check=False)


def export_one_study(study_id: int, out_dir: Path) -> Path:
    raw = _run_assemble(study_id)

    # Validate what assemble_study() returned before writing anything.
    validated = EnsembleStudyOutput.model_validate(raw)

    # by_alias=True matters here: StudySummary.severity_counts is aliased
    # to the literal "Normal/Mild"/"Moderate"/"Severe" keys the JSON
    # schema expects (additionalProperties:false). SeverityCounts also has
    # populate_by_name=True, which is why *validating* snake_case keys
    # would silently succeed -- that only hides the problem, it doesn't
    # fix it. Always dump by_alias so the file on disk matches the
    # contract, not just what schema.py is lenient enough to accept back in.
    json_text = validated.model_dump_json(indent=2, by_alias=True)

    # Round-trip: validate the exact bytes about to be written, not just
    # the in-memory object above -- catches any dump-time issue (like the
    # one just described) rather than trusting the pre-dump validation.
    EnsembleStudyOutput.model_validate(json.loads(json_text))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"study_{study_id}.json"
    out_path.write_text(json_text)
    return out_path


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

    out_dir = Path(args.out_dir)
    written: list[Path] = []
    versions_seen: set[tuple] = set()

    for sid in study_ids:
        try:
            path = export_one_study(sid, out_dir)
        except Exception as exc:
            # Fail loud, fail the whole batch. A handoff directory that's
            # missing one study silently is worse than one that errors out.
            print(f"[FAIL] study {sid}: {exc}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(path.read_text())
        versions_seen.add(
            (data["pipeline_version"], tuple(sorted(data["model_versions"].items())))
        )
        written.append(path)
        print(f"[OK] {path}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_ids": study_ids,
        "n_studies": len(study_ids),
        # Satisfies point 8 (model/training-run versions) as a single
        # summary on top of the per-study model_versions/pipeline_version
        # fields the schema already requires in every file.
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
            "in the handoff message, since point 8 asks which run produced "
            "which result.",
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
