#!/usr/bin/env python3
"""I2 - validate_all_297.py: read-only validation of the 297 study handoffs (gate G1).

For every study_*.json in --dir297 (handoff_manifest.json is not a study) it runs:
  1. JSON Schema check, the same call validate_against_schema() makes in
     assemble_study_output.py (open schema, jsonschema.validate, no format checker)
  2. EnsembleStudyOutput.model_validate() from pdscd_contract_schema.py
  3. exactly 25 unique (condition, level) pairs, 5 conditions x 5 levels
  4. study_id in the JSON equals the id in the filename
and sha256-hashes the raw bytes of every file.

Both reference files (contract module + schema JSON) are read from the writer commit, not
from the working tree. assemble_study_output.py is not imported: importing it loads torch and
runs a contract check at import time, and validate_against_schema() is three lines.

Nothing in --dir297 is modified, repaired or regenerated. Writes only under --out:
  validation_report.md, handoff_297.sha256
Exit codes: 0 = every file passes all checks (G1 pass) | 1 = a failure (report still written) | 2 = bad arguments.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

DEFAULT_COMMIT = "e5e7432"
CONTRACT_PATH = "classification/pdscd_contract_schema.py"
SCHEMA_JSON_PATH = "classification/pdscd_output_schema.json"
FILE_RE = re.compile(r"^study_(\d+)\.json$")
MAX_ERR_CHARS = 6000


def git(repo, *args, must=True):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if must and p.returncode != 0:
        sys.exit("git " + " ".join(args) + " failed: " + p.stderr.decode("utf-8", "replace").strip())
    return p


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def clip(s):
    return s if len(s) <= MAX_ERR_CHARS else s[:MAX_ERR_CHARS] + f"\n[truncated, original {len(s)} chars]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="PDSCD git clone with the writer commit")
    ap.add_argument("--dir297", required=True, help="folder with the 297 study_*.json + handoff_manifest.json")
    ap.add_argument("--out", default="H2_package", help="output folder for the two deliverables")
    ap.add_argument("--commit", default=DEFAULT_COMMIT, help="writer commit")
    a = ap.parse_args()

    repo, d297, out = Path(a.repo), Path(a.dir297), Path(a.out)
    if not d297.is_dir():
        print(f"--dir297 is not a folder: {d297}")
        return 2
    out_r, d_r = out.resolve(), d297.resolve()
    if out_r == d_r or d_r in out_r.parents:
        print("--out must not be inside the handoff folder")
        return 2

    import jsonschema  # noqa: E402  (stdlib-only imports above so --help works anywhere)

    full = git(repo, "rev-parse", "--verify", a.commit + "^{commit}").stdout.decode().strip()
    ref = {}
    for path in (CONTRACT_PATH, SCHEMA_JSON_PATH):
        data = git(repo, "show", f"{full}:{path}").stdout
        blob_c = git(repo, "rev-parse", f"{full}:{path}").stdout.decode().strip()
        h = git(repo, "rev-parse", f"HEAD:{path}", must=False)
        if h.returncode != 0:
            vs = "absent at HEAD"
        else:
            vs = "same as HEAD" if h.stdout.decode().strip() == blob_c else "differs from HEAD"
        ref[path] = {"data": data, "sha256": sha256_bytes(data), "vs_head": vs}

    files = sorted(p for p in d297.glob("study_*.json") if p.is_file())
    others = sorted(p.name for p in d297.iterdir()
                    if p.is_file() and p.name != "handoff_manifest.json" and not p.name.startswith("study_"))
    manifest = d297 / "handoff_manifest.json"

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "pdscd_contract_schema.py").write_bytes(ref[CONTRACT_PATH]["data"])
        (tmp / "pdscd_output_schema.json").write_bytes(ref[SCHEMA_JSON_PATH]["data"])
        spec = importlib.util.spec_from_file_location("pdscd_contract_schema", tmp / "pdscd_contract_schema.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pdscd_contract_schema"] = mod  # registered first so pydantic can resolve annotations
        spec.loader.exec_module(mod)
        with open(tmp / "pdscd_output_schema.json", encoding="utf-8") as f:
            schema = json.load(f)

        for p in files:
            raw = p.read_bytes()
            r = {"file": p.name, "sha256": sha256_bytes(raw), "js": None, "pd": None, "grid": None, "sid": None, "errors": []}
            rows.append(r)
            try:
                o = json.loads(raw.decode("utf-8"))
            except Exception as e:
                r["errors"].append(("read/parse", f"{type(e).__name__}: {e}"))
                r["js"] = r["pd"] = r["grid"] = r["sid"] = False
                continue

            try:
                jsonschema.validate(instance=o, schema=schema)
                r["js"] = True
            except jsonschema.ValidationError as e:
                r["js"] = False
                where = "/".join(str(x) for x in e.absolute_path) or "(root)"
                r["errors"].append(("jsonschema", f"{e.message}\nat: {where}"))
            except Exception as e:
                r["js"] = False
                r["errors"].append(("jsonschema", f"{type(e).__name__}: {e}"))

            try:
                mod.EnsembleStudyOutput.model_validate(o)
                r["pd"] = True
            except Exception as e:
                r["pd"] = False
                r["errors"].append(("EnsembleStudyOutput", clip(str(e))))

            preds = o.get("predictions") if isinstance(o, dict) else None
            if isinstance(preds, list):
                pairs = [(q.get("condition"), q.get("level")) for q in preds if isinstance(q, dict)]
                nc, nl = len({c for c, _ in pairs}), len({l for _, l in pairs})
                r["grid"] = len(preds) == 25 and len(set(pairs)) == 25 and nc == 5 and nl == 5
                if not r["grid"]:
                    r["errors"].append(("grid", f"{len(preds)} predictions, {len(set(pairs))} unique (condition, level) "
                                                f"pairs, {nc} conditions, {nl} levels (expected 25, 25, 5, 5)"))
            else:
                r["grid"] = False
                r["errors"].append(("grid", "no predictions list"))

            m = FILE_RE.match(p.name)
            sid = o.get("study_id") if isinstance(o, dict) else None
            r["sid"] = bool(m) and sid == int(m.group(1))
            if not r["sid"]:
                r["errors"].append(("study_id", f"JSON study_id={sid!r} vs filename {p.name}"))

    # nothing modified: re-hash every file after the run
    changed = [r["file"] for r in rows if sha256_bytes((d297 / r["file"]).read_bytes()) != r["sha256"]]
    after = sorted(p.name for p in d297.glob("study_*.json") if p.is_file())

    n = len(rows)
    cnt = lambda k: sum(1 for r in rows if r[k])
    n_all = sum(1 for r in rows if r["js"] and r["pd"] and r["grid"] and r["sid"])
    unchanged = not changed and after == [r["file"] for r in rows]
    g1 = n == 297 and n_all == n and unchanged

    # ---- outputs
    out.mkdir(parents=True, exist_ok=True)
    sums = "".join(f"{r['sha256']}  {r['file']}\n" for r in rows)
    if manifest.is_file():
        sums += f"{sha256_bytes(manifest.read_bytes())}  handoff_manifest.json\n"
    (out / "handoff_297.sha256").write_text(sums, encoding="utf-8", newline="\n")

    L = []
    L.append("# Validation report: 297-study handoff (I2)\n")
    L.append(f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}  ")
    L.append("Read-only: no handoff file was modified, repaired or regenerated.\n")
    L.append("## Setup")
    L.append(f"- Handoff folder: `{d297}`")
    L.append(f"- Writer commit used for both validators: `{full}`")
    for path in (CONTRACT_PATH, SCHEMA_JSON_PATH):
        L.append(f"- `{path}` at that commit: sha256 `{ref[path]['sha256']}` ({ref[path]['vs_head']})")
    L.append(f"- pydantic {version('pydantic')}, jsonschema {version('jsonschema')}")
    L.append("- Method: the JSON Schema check repeats the call in `validate_against_schema()` "
             "(`assemble_study_output.py`): load the schema, `jsonschema.validate`, no format checker. "
             "The module itself is not imported because importing it loads torch and runs a contract check.")
    L.append("- `handoff_manifest.json` is not a study and is not validated; its sha256 is the last line of `handoff_297.sha256`.\n")
    L.append("## Result")
    L.append("| Check | Pass | Fail |\n|---|---|---|")
    for label, k in (("JSON Schema", "js"), ("`EnsembleStudyOutput.model_validate`", "pd"),
                     ("25 unique (condition, level), 5 x 5", "grid"), ("`study_id` equals filename id", "sid")):
        L.append(f"| {label} | {cnt(k)} | {n - cnt(k)} |")
    L.append(f"| **All four checks** | **{n_all}** | **{n - n_all}** |\n")
    L.append(f"- Study files found: {n} (expected 297)")
    L.append(f"- Files unchanged after the run (re-hashed): {'yes' if unchanged else 'NO: ' + ', '.join(changed)}")
    L.append(f"- Other files in the folder (not studies, not the manifest): {', '.join(others) if others else 'none'}\n")
    L.append("## Failures (verbatim)")
    failed = [r for r in rows if r["errors"]]
    if not failed:
        L.append("None.\n")
    for r in failed:
        L.append(f"### {r['file']}")
        for kind, msg in r["errors"]:
            L.append(f"**{kind}**\n```\n{msg}\n```")
        L.append("")
    L.append("## G1")
    L.append(f"Validators: **{'PASS' if g1 else 'FAIL'}** ({n_all}/{n} files pass all four checks; expected 297/297).")
    (out / "validation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"files {n} | jsonschema {cnt('js')} | EnsembleStudyOutput {cnt('pd')} | grid {cnt('grid')} | "
          f"study_id {cnt('sid')} | all four {n_all} | unchanged {unchanged}")
    print(f"G1 validators: {'PASS' if g1 else 'FAIL'}")
    print(f"Wrote {out}/validation_report.md and {out}/handoff_297.sha256")
    if not g1:
        print("Do not repair anything. Message the LLM owner now, listing the failing files (see the report).")
    return 0 if g1 else 1


if __name__ == "__main__":
    sys.exit(main())
