#!/usr/bin/env python3
"""I3 - scan the 297 handoff JSONs (read-only) and write FIELDS.md.

Usage:
  python build_fields_md.py --dir297 DIR --out DIR

Reads every study_*.json in --dir297 (and handoff_manifest.json, only to quote
n_studies and the GT-swap list). Writes FIELDS.md into --out. Standard library
only. Never writes into --dir297; hashes every file before the scan and again
after it, and reports whether anything changed.

Exit codes: 0 ok | 1 some file was not a UTF-8 JSON object (report still
written) | 2 bad arguments.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MISSING = "missing_detection"
MAX_DISTINCT = 40      # stop tracking distinct values of one field after this many
SHOW_DISTINCT = 12     # list values in a table cell only up to this many


def tname(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return type(v).__name__


def flatten(value, prefix=""):
    """Yield (path, leaf) for every leaf. Lists add a '<path>[len]' leaf (their
    length) and use '<path>[]' for their elements."""
    if isinstance(value, dict):
        if not value:
            yield prefix, value
        for k, v in value.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        yield f"{prefix}[len]", len(value)
        for v in value:
            yield from flatten(v, f"{prefix}[]")
    else:
        yield prefix, value


class PathStat:
    def __init__(self):
        self.n = 0
        self.types = Counter()
        self.values = Counter()
        self.overflow = False
        self.lo = None
        self.hi = None

    def add(self, v):
        self.n += 1
        self.types[tname(v)] += 1
        if v is None or isinstance(v, (dict, list)):
            return
        key = json.dumps(v, ensure_ascii=False)
        if key in self.values or len(self.values) < MAX_DISTINCT:
            self.values[key] += 1
        else:
            self.overflow = True
        num = isinstance(v, (int, float)) and not isinstance(v, bool)
        if num or isinstance(v, str):
            try:
                if self.lo is None or v < self.lo:
                    self.lo = v
                if self.hi is None or v > self.hi:
                    self.hi = v
            except TypeError:
                pass  # str vs number mixed in one field


def types_str(st):
    if len(st.types) == 1:
        return next(iter(st.types))
    return ", ".join(f"{t} x{n}" for t, n in st.types.most_common())


def values_str(st):
    if st.overflow:
        return f"{MAX_DISTINCT}+ distinct; range `{st.lo}` .. `{st.hi}`"
    if not st.values:
        return ""
    if len(st.values) <= SHOW_DISTINCT:
        items = sorted(st.values.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"`{k}` x{n}" for k, n in items)
    return f"{len(st.values)} distinct; range `{st.lo}` .. `{st.hi}`"


def md_cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return out


def get_path(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def jkey(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def unjk(s):
    return "null" if s == "null" else s


def file_sort_key(p):
    s = p.stem[len("study_"):]
    return (0, int(s), p.name) if s.isdigit() else (1, 0, p.name)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="I3: write FIELDS.md from the 297 handoff files (read-only).")
    ap.add_argument("--dir297", required=True, help="folder with study_*.json and handoff_manifest.json")
    ap.add_argument("--out", required=True, help="folder to write FIELDS.md into (not inside --dir297)")
    args = ap.parse_args()

    d = Path(args.dir297)
    out = Path(args.out)
    if not d.is_dir():
        print(f"ERROR: --dir297 is not a folder: {d}")
        return 2
    try:
        out.resolve().relative_to(d.resolve())
    except ValueError:
        pass
    else:
        print("ERROR: --out must not be inside --dir297 (read-only rule)")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(d.glob("study_*.json"), key=file_sort_key)
    manifest_path = d / "handoff_manifest.json"
    others = sorted(p.name for p in d.iterdir()
                    if p.is_file() and not p.name.startswith("study_") and p.name != "handoff_manifest.json")

    # ---- read (bytes -> hash -> parse) -------------------------------------
    hashes, failures, studies = {}, [], []
    for p in files:
        raw = p.read_bytes()
        hashes[p.name] = sha(raw)
        try:
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("top level is not a JSON object")
        except Exception as e:  # reported verbatim, never repaired
            failures.append((p.name, f"{type(e).__name__}: {e}"))
            continue
        studies.append((p.name, obj))

    manifest = None
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        hashes[manifest_path.name] = sha(raw)
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except Exception:
            manifest = None

    # ---- analyse -----------------------------------------------------------
    study_stats, pred_stats = {}, {}
    present = {"md": Counter(), "other": Counter()}
    nulls = {"md": Counter(), "other": Counter()}
    grp_rows = Counter()
    enum_keys = ["status", "model_agreement", "severity_label", "severity_code", "condition", "level",
                 "competition_weight", "source.series_description", "source.n_slices_used"]
    enums = {k: Counter() for k in enum_keys}
    by_status = {"model_agreement": Counter(), "severity_label": Counter()}
    code_label = Counter()
    status_studies = Counter()
    per_study_md, per_study_split = Counter(), Counter()
    ss_scalars, cand_by_study = {}, {}
    n_pred, non_object, bad_pred_arrays = 0, 0, 0
    samples = {}
    first_study_block = None

    for name, obj in studies:
        rest = {k: v for k, v in obj.items() if k != "predictions"}
        if first_study_block is None:
            first_study_block = (name, rest)
        ss = {}
        for path, v in flatten(rest):
            study_stats.setdefault(path, PathStat()).add(v)
            if path.startswith("study_summary") and isinstance(v, int) and not isinstance(v, bool):
                ss[path] = v
        ss_scalars[name] = ss
        preds = obj.get("predictions")
        if isinstance(preds, list):
            study_stats.setdefault("predictions[len]", PathStat()).add(len(preds))
        else:
            bad_pred_arrays += 1
            study_stats.setdefault("predictions", PathStat()).add(preds)
            preds = []

        st_c, lab_c, agr_c = Counter(), Counter(), Counter()
        lt05 = 0
        for idx, pr in enumerate(preds):
            if not isinstance(pr, dict):
                non_object += 1
                continue
            n_pred += 1
            status = pr.get("status")
            g = "md" if status == MISSING else "other"
            grp_rows[g] += 1
            seen, nullseen = set(), set()
            for path, v in flatten(pr):
                pred_stats.setdefault(path, PathStat()).add(v)
                seen.add(path)
                if v is None:
                    nullseen.add(path)
            for path in seen:
                present[g][path] += 1
            for path in nullseen:
                nulls[g][path] += 1
            for k in enum_keys:
                enums[k][jkey(get_path(pr, k))] += 1
            for k in by_status:
                by_status[k][(jkey(status), jkey(pr.get(k)))] += 1
            code_label[(jkey(pr.get("severity_code")), jkey(pr.get("severity_label")))] += 1
            st_c[jkey(status)] += 1
            lab_c[jkey(pr.get("severity_label"))] += 1
            agr_c[jkey(pr.get("model_agreement"))] += 1
            conf = pr.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf < 0.5:
                lt05 += 1
            agreement = pr.get("model_agreement")
            for label, hit in (("ok", status == "ok"), ("missing_detection", status == MISSING),
                               ("split_no_majority", agreement == "split_no_majority"),
                               ("majority_2_of_3", agreement == "majority_2_of_3")):
                if hit and label not in samples:
                    samples[label] = (name, idx, pr)

        for s in st_c:
            status_studies[s] += 1
        per_study_md[st_c.get(jkey(MISSING), 0)] += 1
        per_study_split[agr_c.get(jkey("split_no_majority"), 0)] += 1
        cand = {"n_predictions": sum(st_c.values()),
                "n_conf_lt_0.5 (non-null)": lt05,
                "n_conf_lt_0.5 + n_missing_detection": lt05 + st_c.get(jkey(MISSING), 0)}
        for k, n in st_c.items():
            cand[f"n_status={unjk(k)}"] = n
        for k, n in lab_c.items():
            cand[f"n_severity_label={unjk(k)}"] = n
        for k, n in agr_c.items():
            cand[f"n_model_agreement={unjk(k)}"] = n
        cand_by_study[name] = cand

    # ---- prove nothing changed ---------------------------------------------
    unchanged = True
    for fname, h in hashes.items():
        if sha((d / fname).read_bytes()) != h:
            unchanged = False

    # ---- write report ------------------------------------------------------
    L = []
    L.append("# FIELDS.md: field reference for the 297-study handoff (I3)")
    L.append("")
    L.append(f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("Read-only scan: no handoff file was modified, repaired or regenerated. Every value below was counted from the files.")
    L.append("")
    L.append("## 1 Scan")
    L.append(f"- Folder: `{d}`")
    L.append(f"- Study files found: {len(files)}; parsed as UTF-8 JSON objects: {len(studies)}; failed: {len(failures)}")
    if manifest is not None and isinstance(manifest, dict):
        L.append(f"- `handoff_manifest.json` (not a study): `n_studies` = {manifest.get('n_studies')}, "
                 f"`split` = `{manifest.get('split')}`, `known_gt_swap_ids_in_set` = {manifest.get('known_gt_swap_ids_in_set')}")
    else:
        L.append("- `handoff_manifest.json`: not found or not readable")
    L.append(f"- Other files in the folder (not studies, not the manifest): {', '.join(others) if others else 'none'}")
    L.append(f"- Predictions scanned: {n_pred}; predictions that are not JSON objects: {non_object}; studies whose `predictions` is not an array: {bad_pred_arrays}")
    L.append(f"- Files unchanged after the scan (re-hashed): {'yes' if unchanged else 'NO - investigate'}")
    if failures:
        L.append("")
        L.append("Failures (verbatim, not repaired):")
        for fname, msg in failures:
            L.append(f"- `{fname}`: {msg}")
    L.append("")

    n_st = len(studies)
    L.append("## 2 Study-level fields (everything except `predictions`)")
    L.append(f"`present` counts studies out of {n_st}. Paths use `.` for nesting and `[len]` for an array's length.")
    L.append("")
    rows = [(p, f"{s.n}/{n_st}", types_str(s), values_str(s)) for p, s in study_stats.items()]
    L += table(["path", "present", "type", "values seen"], rows)
    L.append("")

    L.append(f"## 3 Prediction fields (all {n_pred} predictions)")
    L.append("`present` counts predictions. A `null` in the type column is a real JSON null.")
    L.append("")
    rows = [(p, f"{s.n}/{n_pred}", types_str(s), values_str(s)) for p, s in pred_stats.items()]
    L += table(["path", "present", "type", "values seen"], rows)
    L.append("")

    L.append("## 4 Values seen")
    for k in enum_keys:
        c = enums[k]
        L.append(f"- `{k}`: " + ", ".join(f"`{unjk(v)}` x{n}" for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))))
    L.append("")
    L.append("`severity_code` with `severity_label` (pairs as they occur):")
    L.append("")
    L += table(["severity_code", "severity_label", "predictions"],
               [(unjk(a), unjk(b), n) for (a, b), n in sorted(code_label.items(), key=lambda kv: (-kv[1], kv[0]))])
    L.append("")
    for k in ("model_agreement", "severity_label"):
        L.append(f"`{k}` by `status`:")
        L.append("")
        L += table(["status", k, "predictions"],
                   [(unjk(a), unjk(b), n) for (a, b), n in sorted(by_status[k].items(), key=lambda kv: (kv[0], -kv[1]))])
        L.append("")
    L.append("**Allow-list for the CV page (I6 rule: show only values seen here).**")
    for k in ("status", "model_agreement", "severity_label"):
        vals = [unjk(v) for v in sorted(enums[k]) if v != "null"]
        L.append(f"- `{k}`: " + (", ".join(f"`{json.loads(v) if v.startswith(chr(34)) else v}`" for v in vals) or "(none)"))
    L.append("")

    L.append("## 5 What is null or absent when `status` = `missing_detection`")
    L.append(f"Rows in group: `missing_detection` = {grp_rows['md']}; all other statuses = {grp_rows['other']}. "
             "`pattern` describes the `missing_detection` rows: **absent** (path never appears), **null** (always JSON null), "
             "**non-null** (never null), **mixed**. An array path counts a row once.")
    L.append("")
    rows = []
    for p in pred_stats:
        def cell(g):
            return f"{present[g][p]} present, {nulls[g][p]} null"
        pm, nm = present["md"][p], nulls["md"][p]
        if grp_rows["md"] == 0:
            pat = "no such rows"
        elif pm == 0:
            pat = "absent"
        elif nm == pm and pm == grp_rows["md"]:
            pat = "null"
        elif nm == 0 and pm == grp_rows["md"]:
            pat = "non-null"
        else:
            pat = "mixed"
        rows.append((p, pat, cell("md"), cell("other")))
    L += table(["path", "pattern", "missing_detection rows", "other rows"], rows)
    L.append("")

    L.append("## 6 Status counts")
    L.append("")
    L += table(["status", "predictions", "studies containing it"],
               [(unjk(v), n, status_studies[v]) for v, n in sorted(enums["status"].items(), key=lambda kv: (-kv[1], kv[0]))])
    L.append("")
    L.append("Studies by number of `missing_detection` predictions: " +
             ", ".join(f"{k}: {n}" for k, n in sorted(per_study_md.items())))
    L.append("")
    L.append("Studies by number of `split_no_majority` predictions: " +
             ", ".join(f"{k}: {n}" for k, n in sorted(per_study_split.items())))
    L.append("")

    L.append("## 7 `study_summary` cross-check (informational)")
    L.append("For each integer field under `study_summary`, which counts recomputed from the predictions equal it in every study that has it. "
             "A match only means something if the field's range is not constant.")
    L.append("")
    ss_paths = []
    for ss in ss_scalars.values():
        for p in ss:
            if p not in ss_paths:
                ss_paths.append(p)
    cand_names = []
    for c in cand_by_study.values():
        for k in c:
            if k not in cand_names:
                cand_names.append(k)
    rows = []
    for p in ss_paths:
        have = [nm for nm in ss_scalars if p in ss_scalars[nm]]
        full, best = [], (None, -1)
        for c in cand_names:
            m = sum(1 for nm in have if ss_scalars[nm][p] == cand_by_study[nm].get(c, 0))
            if m == len(have):
                full.append(c)
            if m > best[1]:
                best = (c, m)
        st = study_stats.get(p)
        rng = f"{st.lo} .. {st.hi}" if st is not None else ""
        verdict = ", ".join(full) if full else f"none (closest: {best[0]}, {best[1]}/{len(have)})"
        rows.append((p, f"{len(have)}/{n_st}", rng, verdict))
    L += table(["study_summary field", "present", "range", "equals (all studies)"], rows) if rows else ["(no integer fields under study_summary)"]
    L.append("")

    L.append("## 8 Verbatim samples (first match in ascending study id)")
    if first_study_block:
        L.append("")
        L.append(f"Study-level block of `{first_study_block[0]}` (everything except `predictions`):")
        L.append("")
        L.append("```json")
        L.append(json.dumps(first_study_block[1], indent=2, ensure_ascii=False))
        L.append("```")
    for label in ("ok", "majority_2_of_3", "split_no_majority", "missing_detection"):
        L.append("")
        if label in samples:
            nm, idx, pr = samples[label]
            what = "status ok" if label == "ok" else ("status missing_detection" if label == MISSING else f"model_agreement {label}")
            L.append(f"Prediction with {what}: `{nm}`, `predictions[{idx}]`:")
            L.append("")
            L.append("```json")
            L.append(json.dumps(pr, indent=2, ensure_ascii=False))
            L.append("```")
        else:
            L.append(f"No prediction with {label} was found.")
    L.append("")

    report = out / "FIELDS.md"
    report.write_text("\n".join(L), encoding="utf-8", newline="\n")
    status_summary = ", ".join(f"{json.loads(v)} {n}" for v, n in sorted(enums["status"].items()))
    print(f"files {len(files)} | parsed {len(studies)} | predictions {n_pred} | status: {status_summary} | unchanged {unchanged}")
    print(f"Wrote {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
