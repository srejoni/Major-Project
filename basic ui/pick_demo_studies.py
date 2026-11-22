#!/usr/bin/env python3
"""I5 - choose the demo studies from the 297 handoff files (read-only).

Usage:
  python pick_demo_studies.py --dir297 DIR --out DIR [--exclude ID ...]

Reads every study_*.json in --dir297 (and handoff_manifest.json for its
GT-swap list). Writes demo_studies.json into --out. Standard library only.
Never writes into --dir297; hashes every file before and after the scan.

Selection uses JSON fields only. "Flagged" below is an estimate for choosing
studies (missing_detection, split_no_majority, or confidence < 0.70). The CLI
decides the real flags; nothing here replaces it.

Exit codes: 0 ok | 1 some file could not be read (nothing written) | 2 bad arguments.
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

GATE = 0.70
DESCRIPTIVE = 0.5
SEED = 20261122
DEFAULT_EXCLUDE = [2797118205]
MISSING = "missing_detection"
SPLIT = "split_no_majority"
PLAN_ORDER = ["clean", "typical", "low_confidence_heavy", "missing_detection",
              "split_no_majority", "several_severe", "random_1", "random_2"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def metrics(obj):
    preds = obj["predictions"]
    m = {"n_findings": len(preds), "n_flagged": 0, "n_missing_detection": 0, "n_split_no_majority": 0,
         "n_gate_lt_0.70": 0, "n_conf_lt_0.5": 0, "n_severe": 0, "n_moderate": 0,
         "n_duplicate_resolved": 0, "n_split_and_gate": 0}
    confs, miss_rows, split_rows, codes = [], [], [], []
    for p in preds:
        conf = p.get("confidence")
        miss = p.get("status") == MISSING
        split = p.get("model_agreement") == SPLIT
        low = is_num(conf) and conf < GATE
        if is_num(conf):
            confs.append(conf)
            if conf < DESCRIPTIVE:
                m["n_conf_lt_0.5"] += 1
        if low:
            m["n_gate_lt_0.70"] += 1
        if miss:
            m["n_missing_detection"] += 1
            miss_rows.append((p.get("condition"), p.get("level")))
        if split:
            m["n_split_no_majority"] += 1
            split_rows.append((p.get("condition"), p.get("level"), conf))
            if low:
                m["n_split_and_gate"] += 1
        if miss or split or low:
            m["n_flagged"] += 1
        if p.get("severity_label") == "Severe":
            m["n_severe"] += 1
        if p.get("severity_label") == "Moderate":
            m["n_moderate"] += 1
        if p.get("status") == "duplicate_resolved":
            m["n_duplicate_resolved"] += 1
        if is_num(p.get("severity_code")):
            codes.append(p["severity_code"])
    m["n_unflagged"] = m["n_findings"] - m["n_flagged"]
    m["mean_confidence"] = round(sum(confs) / len(confs), 4) if confs else None
    summ = obj.get("study_summary")
    hs = summ.get("highest_severity_code") if isinstance(summ, dict) else None
    m["_highest_ok"] = (hs == max(codes)) if codes else (hs is None)
    return m, miss_rows, split_rows


def choose(pool, filters, key):
    """First non-empty filter wins (strictest first). Returns (study, relaxed)."""
    for i, f in enumerate(filters):
        cand = [s for s in pool if f(s["m"])]
        if cand:
            return min(cand, key=key), i > 0
    return None, False


def main():
    ap = argparse.ArgumentParser(description="I5: write demo_studies.json from the 297 handoff files (read-only).")
    ap.add_argument("--dir297", required=True)
    ap.add_argument("--out", required=True, help="folder for demo_studies.json (not inside --dir297)")
    ap.add_argument("--exclude", type=int, nargs="*", default=DEFAULT_EXCLUDE,
                    help="study ids to keep out of the demo set (default: the GT-swap id 2797118205)")
    args = ap.parse_args()

    d, out = Path(args.dir297), Path(args.out)
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

    files = sorted(d.glob("study_*.json"), key=lambda p: (len(p.stem), p.stem))
    mp = d / "handoff_manifest.json"
    hashes, failures, studies = {}, [], []
    for p in files:
        raw = p.read_bytes()
        hashes[p.name] = sha(raw)
        try:
            obj = json.loads(raw.decode("utf-8"))
            sid = obj["study_id"]
            if isinstance(sid, bool) or not isinstance(sid, int):
                raise ValueError("study_id is not an integer")
            m, miss_rows, split_rows = metrics(obj)
        except Exception as e:
            failures.append((p.name, f"{type(e).__name__}: {e}"))
            continue
        studies.append({"id": sid, "file": p.name, "m": m, "miss_rows": miss_rows, "split_rows": split_rows})

    manifest_swap = []
    if mp.is_file():
        raw = mp.read_bytes()
        hashes[mp.name] = sha(raw)
        try:
            manifest_swap = [int(x) for x in json.loads(raw.decode("utf-8")).get("known_gt_swap_ids_in_set", [])]
        except Exception:
            manifest_swap = []

    unchanged = all(sha((d / n).read_bytes()) == h for n, h in hashes.items())
    if failures:
        print("ERROR: some files could not be read; nothing written")
        for n, msg in failures:
            print(f"  {n}: {msg}")
        return 1

    excluded = sorted(set(args.exclude) | set(manifest_swap))
    studies.sort(key=lambda s: s["id"])
    pool = [s for s in studies if s["id"] not in excluded]
    if len({s["id"] for s in studies}) != len(studies):
        print("ERROR: duplicate study_id values across files; nothing written")
        return 1

    confs_pool = [s["m"]["mean_confidence"] for s in pool if s["m"]["mean_confidence"] is not None]
    if not pool or not confs_pool:
        print(f"ERROR: no usable study_*.json files in {d} (found {len(files)}, after exclusions {len(pool)}); nothing written")
        return 1
    med_flag = statistics.median(s["m"]["n_flagged"] for s in pool)
    med_conf = statistics.median(confs_pool)
    cohort_totals = {
        "findings": sum(s["m"]["n_findings"] for s in studies),
        "flagged_union": sum(s["m"]["n_flagged"] for s in studies),
        "missing_detection": sum(s["m"]["n_missing_detection"] for s in studies),
        "split_no_majority": sum(s["m"]["n_split_no_majority"] for s in studies),
        "split_and_conf_lt_0.70": sum(s["m"]["n_split_and_gate"] for s in studies),
        "conf_lt_0.70_not_split": sum(s["m"]["n_gate_lt_0.70"] - s["m"]["n_split_and_gate"] for s in studies),
        "conf_lt_0.5": sum(s["m"]["n_conf_lt_0.5"] for s in studies),
    }
    hs_bad = [s["id"] for s in studies if not s["m"]["_highest_ok"]]

    plain = lambda m: m["n_missing_detection"] == 0 and m["n_split_no_majority"] == 0
    anyone = lambda m: True
    picks, used = {}, set()

    def take(tag, filters, key, reason_fn):
        cand_pool = [s for s in pool if s["id"] not in used]
        s, relaxed = choose(cand_pool, filters, key)
        if s is None:
            return
        used.add(s["id"])
        picks[tag] = (s, reason_fn(s) + (" (relaxed: no study met the strict rule)" if relaxed else ""))

    mc = lambda s: s["m"]["mean_confidence"] if s["m"]["mean_confidence"] is not None else 0.0
    nf = lambda s: s["m"]["n_findings"]
    take("missing_detection", [lambda m: m["n_missing_detection"] >= 1 and m["n_split_no_majority"] == 0,
                               lambda m: m["n_missing_detection"] >= 1],
         lambda s: (s["m"]["n_missing_detection"], s["m"]["n_gate_lt_0.70"], s["id"]),
         lambda s: (f"{s['m']['n_missing_detection']} of {nf(s)} findings has no YOLO box (missing_detection: "
                    + ", ".join(f"{c} {l}" for c, l in s["miss_rows"]) + "); fewest other flags among such studies; "
                    f"{sum(1 for x in studies if x['m']['n_missing_detection'] > 0)} of {len(studies)} studies have one"))
    take("split_no_majority", [lambda m: m["n_split_no_majority"] >= 1 and m["n_missing_detection"] == 0,
                               lambda m: m["n_split_no_majority"] >= 1],
         lambda s: (s["m"]["n_split_no_majority"], s["m"]["n_gate_lt_0.70"], s["id"]),
         lambda s: (f"{s['m']['n_split_no_majority']} of {nf(s)} findings is split_no_majority ("
                    + ", ".join(f"{c} {l}, top-class probability {cf:.3f}" for c, l, cf in s["split_rows"])
                    + f"); {sum(1 for x in studies if x['m']['n_split_no_majority'] > 0)} of {len(studies)} studies have one"))
    take("low_confidence_heavy", [plain, anyone],
         lambda s: (-s["m"]["n_gate_lt_0.70"], mc(s), s["id"]),
         lambda s: (f"{s['m']['n_gate_lt_0.70']} of {nf(s)} findings below the 0.70 gate (cohort median flagged {med_flag:g}); "
                    f"mean confidence {mc(s):.3f}; {s['m']['n_unflagged']} unflagged"))
    take("several_severe", [plain, anyone],
         lambda s: (-s["m"]["n_severe"], s["id"]),
         lambda s: (f"{s['m']['n_severe']} of {nf(s)} findings are Severe (most among the remaining studies); "
                    f"{s['m']['n_flagged']} flagged, mean confidence {mc(s):.3f}"))
    take("clean", [plain, anyone],
         lambda s: (s["m"]["n_flagged"], -mc(s), s["id"]),
         lambda s: (f"{s['m']['n_flagged']} of {nf(s)} findings flagged (fewest among plain studies; cohort median {med_flag:g}); "
                    f"mean confidence {mc(s):.3f}"))
    take("typical", [plain, anyone],
         lambda s: (abs(s["m"]["n_flagged"] - med_flag), abs(mc(s) - med_conf), s["id"]),
         lambda s: (f"{s['m']['n_flagged']} of {nf(s)} findings flagged vs cohort median {med_flag:g}; "
                    f"mean confidence {mc(s):.3f} vs cohort median {med_conf:.3f}"))
    rest = sorted(s["id"] for s in pool if s["id"] not in used)
    by_id = {s["id"]: s for s in pool}
    for i, sid in enumerate(random.Random(SEED).sample(rest, min(2, len(rest))), start=1):
        s = by_id[sid]
        used.add(sid)
        picks[f"random_{i}"] = (s, f"random draw {i} (seed {SEED}); {s['m']['n_flagged']} of {nf(s)} findings flagged, "
                                   f"mean confidence {mc(s):.3f}")

    ordered = [t for t in PLAN_ORDER if t in picks]
    studies_out = []
    for t in ordered:
        s, reason = picks[t]
        stats = {k: v for k, v in s["m"].items() if not k.startswith("_") and k != "n_split_and_gate"}
        studies_out.append({"study_id": s["id"], "tag": t, "reason": reason, "stats": stats})
    doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "source_folder": str(d),
        "n_studies_scanned": len(studies),
        "excluded_ids": excluded,
        "seed": SEED,
        "note": ("Chosen from JSON fields only. n_flagged is an estimate (missing_detection, split_no_majority, "
                 "or confidence < 0.70); the CLI decides the real flags. Compare with the dry run (L8) before relying on it."),
        "ids": [x["study_id"] for x in studies_out],
        "studies": studies_out,
    }
    (out / "demo_studies.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"files {len(files)} | studies {len(studies)} | pool {len(pool)} (excluded {excluded}) | picked {len(studies_out)} | unchanged {unchanged}")
    print("cohort (JSON fields): " + ", ".join(f"{k} {v}" for k, v in cohort_totals.items()))
    print(f"study_summary.highest_severity_code == max severity_code: {'all studies' if not hs_bad else 'MISMATCH in ' + str(hs_bad[:10])}")
    for x in studies_out:
        st = x["stats"]
        print(f"{x['tag']:<22} {x['study_id']:<12} flagged {st['n_flagged']:>2}/25  <0.70 {st['n_gate_lt_0.70']:>2}  missing {st['n_missing_detection']}  "
              f"split {st['n_split_no_majority']}  severe {st['n_severe']:>2}  mean conf {st['mean_confidence']}")
    if len(studies_out) < 8:
        print(f"WARNING: only {len(studies_out)} studies picked")
    print(f"Wrote {out / 'demo_studies.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
