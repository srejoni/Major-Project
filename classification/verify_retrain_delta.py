"""
classification/verify_retrain_delta.py

Phase C of the PDSCD completion plan: did the retrain on the laterality-fixed
crops improve or hold (a) pooled Moderate recall and (b) the lateralized-
accuracy-gap, relative to the pre-retrain EfficientNet-B3 checkpoint?

Why this exists instead of just re-running verify_training_quality.py:
  * That script's axial-vs-sagittal check is a macro-F1 gap. The
    lateralized-accuracy-gap defined in PDSCD_imaging_context.md s5 is
      accuracy(spinal_canal_stenosis) - mean accuracy(4 lateralized conditions)
    It prints per-condition accuracy but never aggregates that, nor pooled
    Moderate recall.
  * The documented baseline survives only as ranges (Moderate recall
    0.29-0.49 across conditions). --baseline-checkpoint re-scores the
    pre-retrain checkpoint on the SAME val rows in the SAME pass, so the
    comparison is exact and controlled.

Reuses load_model / confusion_matrix_3x3 / per_class_recall / macro_f1 /
print_report from verify_training_quality.py unchanged, so metric definitions
match the run that produced the baseline.

Trains nothing. Uses val_loader only -- never the locked test set.
Does NOT investigate the deferred 32/10 left/right row-loss skew.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from classification.build_dataloaders import build_classification_dataloaders  # noqa: E402
from classification.verify_training_quality import (  # noqa: E402
    load_model, print_report, confusion_matrix_3x3, per_class_recall, macro_f1,
)

CANAL = "spinal_canal_stenosis"
FORAMINAL = ("left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing")
SUBARTICULAR = ("left_subarticular_stenosis", "right_subarticular_stenosis")
LATERALIZED = FORAMINAL + SUBARTICULAR
ALL_CONDITIONS = (CANAL,) + LATERALIZED
CLASS_WEIGHTS = np.array([1.0, 2.0, 4.0])  # project convention (train.py CE weights)
MOD, SEV = 1, 2

# Proposed, not from the plan: +/- band around "no change" for Moderate recall
# and the gap. Adjust with --noise-band.
NOISE_BAND = 0.03

# Pre-retrain figures as documented in PDSCD_imaging_context.md s5.
# Ranges, not per-condition values -- context only, not used for the verdict
# when --baseline-checkpoint is supplied.
DOC = {
    "val_acc_logged": 0.8046,
    "moderate_recall_range": (0.29, 0.49),
    "severe_recall_range": (0.48, 0.62),
    "canal_acc_range": (0.86, 0.89),
    "lateralized_acc_range": (0.74, 0.80),
    "gap_range": (0.86 - 0.80, 0.89 - 0.74),
    "axial_macro_f1": 0.6564,
    "sagittal_macro_f1": 0.6008,
    "val_rows_pre_migration": 10353,
    "rows_lost_in_migration_total": 42,
}
BASELINE_REPRO_TOL = 0.02


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo_root):
    try:
        r = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def collect_outputs(models, loader, device):
    """One pass over the loader; every model scores the identical batch.
    Returns (conditions[N], labels[N], {name: log_probs[N,3]})."""
    conds, labels = [], []
    logp = {name: [] for name in models}
    for tensors, lab, _study_ids, cond, _levels in loader:
        x = tensors.to(device)
        for name, m in models.items():
            out = torch.log_softmax(m(x).double(), dim=1)
            logp[name].append(out.cpu().numpy())
        labels.append(lab.cpu().numpy())
        conds.extend(list(cond))
    return (np.asarray(conds), np.concatenate(labels).astype(int),
            {k: np.concatenate(v) for k, v in logp.items()})


def summarize(conds, labels, logp):
    preds = logp.argmax(axis=1)
    nll = -logp[np.arange(len(labels)), labels]
    w = CLASS_WEIGHTS[labels]

    per_cond, cm_total = {}, np.zeros((3, 3), dtype=int)
    for c in sorted(set(conds.tolist())):
        m = conds == c
        cm = confusion_matrix_3x3(labels[m], preds[m])
        cm_total += cm
        per_cond[c] = {
            "n": int(cm.sum()),
            "accuracy": float(np.trace(cm) / cm.sum()),
            "macro_f1": float(macro_f1(cm)),
            "recall": [None if r is None else float(r) for r in per_class_recall(cm)],
            "support": [int(s) for s in cm.sum(axis=1)],
            "weighted_nll": float((w[m] * nll[m]).sum() / w[m].sum()),
        }

    missing = [c for c in ALL_CONDITIONS if c not in per_cond]
    if missing:
        sys.exit(f"[delta] FATAL: conditions absent from val predictions: {missing}. "
                 f"Refusing to compute a gap from a partial set (0/0 reads as 'clean').")
    sup = cm_total.sum(axis=1)
    if sup[MOD] == 0 or sup[SEV] == 0:
        sys.exit(f"[delta] FATAL: zero Moderate/Severe support in val (support={sup.tolist()}).")

    pooled = {
        "n": int(cm_total.sum()),
        "accuracy": float(np.trace(cm_total) / cm_total.sum()),
        "recall": [float(cm_total[i, i] / sup[i]) if sup[i] else None for i in range(3)],
        "support": [int(s) for s in sup],
        "nll": float(nll.mean()),
        "weighted_nll": float((w * nll).sum() / w.sum()),
    }
    return {"per_condition": per_cond, "pooled": pooled}


def derive(s):
    acc = {c: s["per_condition"][c]["accuracy"] for c in ALL_CONDITIONS}
    f1 = {c: s["per_condition"][c]["macro_f1"] for c in ALL_CONDITIONS}
    mean = lambda cs, d: float(np.mean([d[c] for c in cs]))  # noqa: E731
    lat, fora, sub = mean(LATERALIZED, acc), mean(FORAMINAL, acc), mean(SUBARTICULAR, acc)
    return {
        "canal_acc": acc[CANAL],
        "lateralized_acc": lat,
        "foraminal_acc": fora,
        "subarticular_acc": sub,
        "gap": acc[CANAL] - lat,                 # the Phase C gap
        "gap_foraminal": acc[CANAL] - fora,      # per status table: already post-fix before the retrain
        "gap_subarticular": acc[CANAL] - sub,    # per status table: the crops the retrain actually changed
        "moderate_recall": s["pooled"]["recall"][MOD],
        "severe_recall": s["pooled"]["recall"][SEV],
        "pooled_acc": s["pooled"]["accuracy"],
        "nll": s["pooled"]["nll"],
        "weighted_nll": s["pooled"]["weighted_nll"],
        "axial_macro_f1": mean(SUBARTICULAR, f1),
        "sagittal_macro_f1": mean((CANAL,) + FORAMINAL, f1),
    }


def judge(new_d, old_d, band):
    d_mod = new_d["moderate_recall"] - old_d["moderate_recall"]
    d_gap = new_d["gap"] - old_d["gap"]
    mod = "IMPROVED" if d_mod >= band else "WORSE" if d_mod <= -band else "HELD"
    gap = "IMPROVED" if d_gap <= -band else "WORSE" if d_gap >= band else "HELD"
    if "WORSE" in (mod, gap):
        overall = "NOT IMPROVED"
    elif "IMPROVED" in (mod, gap):
        overall = "IMPROVED"
    else:
        overall = "HELD"
    warnings = []
    if new_d["severe_recall"] - old_d["severe_recall"] <= -0.05:
        warnings.append("pooled Severe recall fell >= 0.05 (not part of the verdict, but check it)")
    return {"overall": overall, "moderate_recall": mod, "gap": gap,
            "delta_moderate_recall": d_mod, "delta_gap": d_gap,
            "mixed": ("WORSE" in (mod, gap)) and ("IMPROVED" in (mod, gap)),
            "warnings": warnings}


# ---------------------------------------------------------------- printing
def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _line(label, doc, new, old=None, nd=4):
    o = _f(old, nd) if old is not None else "-"
    d = f"{new - old:+.{nd}f}" if old is not None else "-"
    print(f"  {label:34s} {doc:>16s} {o:>9s} {_f(new, nd):>9s} {d:>9s}")


def print_comparison(new_d, old_d):
    g = lambda k: None if old_d is None else old_d[k]  # noqa: E731
    r = lambda t: f"{t[0]:.2f}-{t[1]:.2f}"  # noqa: E731
    print("\n" + "=" * 78)
    print("PHASE C -- NEW vs PRE-RETRAIN (identical val rows, same pass)")
    print("=" * 78)
    print(f"  {'metric':34s} {'documented':>16s} {'old':>9s} {'new':>9s} {'delta':>9s}")
    _line("pooled val accuracy", f"{DOC['val_acc_logged']:.4f} logged", new_d["pooled_acc"], g("pooled_acc"))
    _line("pooled Moderate recall", r(DOC["moderate_recall_range"]) + "/cond", new_d["moderate_recall"], g("moderate_recall"))
    _line("pooled Severe recall", r(DOC["severe_recall_range"]) + "/cond", new_d["severe_recall"], g("severe_recall"))
    _line("canal accuracy", r(DOC["canal_acc_range"]), new_d["canal_acc"], g("canal_acc"))
    _line("lateralized mean accuracy (4)", r(DOC["lateralized_acc_range"]), new_d["lateralized_acc"], g("lateralized_acc"))
    _line("LATERALIZED-ACCURACY-GAP", r(DOC["gap_range"]), new_d["gap"], g("gap"))
    _line("  gap vs foraminal only (2)", "-", new_d["gap_foraminal"], g("gap_foraminal"))
    _line("  gap vs subarticular only (2)", "-", new_d["gap_subarticular"], g("gap_subarticular"))
    _line("axial macro-F1 (2 subarticular)", f"{DOC['axial_macro_f1']:.4f}", new_d["axial_macro_f1"], g("axial_macro_f1"))
    _line("sagittal macro-F1 (3)", f"{DOC['sagittal_macro_f1']:.4f}", new_d["sagittal_macro_f1"], g("sagittal_macro_f1"))
    _line("weighted NLL (1/2/4, no smoothing)", "-", new_d["weighted_nll"], g("weighted_nll"))
    _line("plain NLL", "-", new_d["nll"], g("nll"))


def print_per_condition(new_s, old_s):
    print("\n  per-condition (old -> new)")
    print(f"  {'condition':34s} {'n':>6s} {'acc old':>8s} {'acc new':>8s} "
          f"{'Mod old':>8s} {'Mod new':>8s} {'Mod n':>6s}")
    for c in ALL_CONDITIONS:
        n = new_s["per_condition"][c]
        o = old_s["per_condition"][c] if old_s else None
        print(f"  {c:34s} {n['n']:6d} "
              f"{_f(o['accuracy']) if o else '-':>8s} {_f(n['accuracy']):>8s} "
              f"{_f(o['recall'][MOD]) if o else '-':>8s} {_f(n['recall'][MOD]):>8s} "
              f"{n['support'][MOD]:6d}")


def print_verdict(verdict, band):
    print("\n" + "=" * 78)
    print(f"VERDICT (noise band +/-{band:.2f} -- proposed threshold, not from the plan)")
    print("=" * 78)
    if verdict["overall"] in ("NO_EXACT_BASELINE", "INVALID_BASELINE"):
        return
    print(f"  Moderate recall  delta {verdict['delta_moderate_recall']:+.4f}  -> {verdict['moderate_recall']}")
    print(f"  Lateralized gap  delta {verdict['delta_gap']:+.4f}  -> {verdict['gap']}   (negative = better)")
    print(f"  OVERALL: {verdict['overall']}" + ("  (mixed: one metric improved, one got worse)" if verdict["mixed"] else ""))
    for w in verdict["warnings"]:
        print(f"  WARN: {w}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Same manifest train.py used.")
    ap.add_argument("--new-checkpoint", required=True,
                    help="Explicit path. No default on purpose: a default can silently load a stale file.")
    ap.add_argument("--expect-new-sha-prefix", required=True,
                    help="Hard-stop unless the new checkpoint's sha256 starts with this (d5f1cbb3 for the Sept 18 retrain).")
    ap.add_argument("--baseline-checkpoint", default=None,
                    help="Pre-retrain checkpoint, scored on the same rows in the same pass.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--noise-band", type=float, default=NOISE_BAND)
    ap.add_argument("--out-json", default="/kaggle/working/phase_c/phase_c_results.json")
    a = ap.parse_args()

    # ---- gates that must pass before any GPU time is spent
    new_sha = sha256_of(a.new_checkpoint)
    print(f"[delta] new checkpoint  sha256={new_sha}")
    if not new_sha.startswith(a.expect_new_sha_prefix.lower()):
        sys.exit(f"[delta] FATAL: new checkpoint sha256 does not start with "
                 f"{a.expect_new_sha_prefix!r}. Wrong/stale file -- not the retrained model.")
    old_sha = None
    if a.baseline_checkpoint:
        old_sha = sha256_of(a.baseline_checkpoint)
        print(f"[delta] baseline ckpt   sha256={old_sha}")
        if old_sha == new_sha:
            sys.exit("[delta] FATAL: baseline and new checkpoints are byte-identical. "
                     "The git-history lookup returned the retrained file.")
    else:
        print("[delta] no --baseline-checkpoint: documented ranges only, no automated verdict.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[delta] device={device}  repo HEAD={git_head(REPO_ROOT)}")
    models = {"new": load_model(a.new_checkpoint, device)}
    if a.baseline_checkpoint:
        models["old"] = load_model(a.baseline_checkpoint, device)

    loaders = build_classification_dataloaders(a.manifest, batch_size=a.batch_size)
    print(f"[delta] build_classification_dataloaders returned {len(loaders)} items; using index 1 as val")
    conds, labels, logp = collect_outputs(models, loaders[1], device)

    n = len(labels)
    lo = DOC["val_rows_pre_migration"] - DOC["rows_lost_in_migration_total"]
    hi = DOC["val_rows_pre_migration"]
    rows_ok = lo <= n <= hi
    print(f"[delta] val rows scored: {n}  (documented pre-migration {hi}; expect {lo}-{hi})"
          + ("" if rows_ok else "  <-- WARN: outside expected range, check the loader/manifest before trusting this"))

    new_s = summarize(conds, labels, logp["new"])
    new_d = derive(new_s)
    old_s = summarize(conds, labels, logp["old"]) if "old" in logp else None
    old_d = derive(old_s) if old_s else None

    print("\n>>> Full per-condition report for the NEW checkpoint (verify_training_quality.print_report, unmodified)")
    preds_new = logp["new"].argmax(axis=1)
    print_report({c: (labels[conds == c].tolist(), preds_new[conds == c].tolist())
                  for c in sorted(set(conds.tolist()))})

    print_comparison(new_d, old_d)
    print_per_condition(new_s, old_s)

    repro_ok = True
    if old_d is not None:
        lo_c, hi_c = DOC["canal_acc_range"]
        repro_ok = (lo_c - BASELINE_REPRO_TOL) <= old_d["canal_acc"] <= (hi_c + BASELINE_REPRO_TOL)
        print(f"\n  baseline reproduction check: old-checkpoint canal accuracy {old_d['canal_acc']:.4f} "
              f"vs documented {lo_c:.2f}-{hi_c:.2f} (+/-{BASELINE_REPRO_TOL:.2f}) -> "
              + ("PASS" if repro_ok else "FAIL"))
        print("  (canal crops were untouched by the migration, so the old model should reproduce its documented canal accuracy.)")

    if old_d is None:
        lo_m = DOC["moderate_recall_range"][0]
        hi_g = DOC["gap_range"][1]
        floor_ok = all(new_s["per_condition"][c]["recall"][MOD] >= lo_m for c in ALL_CONDITIONS)
        gap_ok = new_d["gap"] <= hi_g
        verdict = {"overall": "NO_EXACT_BASELINE",
                   "weak_check_moderate_recall_all_conditions_ge_doc_floor": floor_ok,
                   "weak_check_gap_le_doc_worst_case": gap_ok}
        print_verdict(verdict, a.noise_band)
        print("  No exact baseline supplied. Weak evidence only (documented ranges are too coarse for IMPROVED/HELD):")
        print(f"    every condition's Moderate recall >= {lo_m:.2f} (documented floor): {'PASS' if floor_ok else 'FAIL'}")
        print(f"    gap <= {hi_g:.2f} (documented worst case):                          {'PASS' if gap_ok else 'FAIL'}")
    elif not repro_ok:
        verdict = {"overall": "INVALID_BASELINE"}
        print_verdict(verdict, a.noise_band)
        print("  INVALID: the old checkpoint did not reproduce its documented canal accuracy. The git-history blob is")
        print("  probably not the pre-retrain model, or the val rows/loader differ. Do not use the deltas above.")
    else:
        verdict = judge(new_d, old_d, a.noise_band)
        print_verdict(verdict, a.noise_band)

    out = {
        "generated_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "repo_head": git_head(REPO_ROOT),
        "device": str(device),
        "manifest": a.manifest,
        "n_val_rows": n,
        "n_val_rows_in_expected_range": rows_ok,
        "new_checkpoint": {"path": a.new_checkpoint, "sha256": new_sha},
        "baseline_checkpoint": ({"path": a.baseline_checkpoint, "sha256": old_sha}
                                if a.baseline_checkpoint else None),
        "noise_band": a.noise_band,
        "documented_baseline": DOC,
        "new": {"summary": new_s, "derived": new_d},
        "old": ({"summary": old_s, "derived": old_d} if old_s else None),
        "verdict": verdict,
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[delta] wrote {a.out_json}")


if __name__ == "__main__":
    main()
